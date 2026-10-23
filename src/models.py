"""Two ADAPTED transformer architectures for dense LULC prediction on the GeoRAG grid.

Honesty note (see docs/MODEL_INVESTIGATION.md):
- mts-lite-adapted  ~ direction of the "multi-temporal Swin Transformer (MTS)" of
  Voelsen et al. (2024). No official code exists; this is a Swin-style
  shifted-window dense encoder with an optional second temporal slot.
- dla-lite-adapted  ~ direction of "Dual level attention based lightweight vision
  transformer" (Bansal & Tripathi 2024). No official code exists; this implements a
  dual-level scheme: token-level (spatial window) attention + channel-level
  (cross-covariance) attention.

NEITHER is a reproduction of a published implementation. Both share the same
patchify (4x4), width, depth and per-pixel head so capacity stays comparable.
"""
import math

import torch
import torch.nn as nn

# Host workaround: this torch 2.14.1+cpu build's oneDNN path crashes with SIGFPE
# on this VM's CPU (verified: Conv2d SIGFPE with mkldnn on, works with it off).
torch.backends.mkldnn.enabled = False

PATCH = 4          # pixels per token edge
WINDOW = 7         # attention window (in tokens)
WIDTH = 64
DEPTH = 6
HEADS = 4
MLP_RATIO = 4.0
DROP_PATH = 0.05


# ----------------------------------------------------------------------------- utils
def roll(x, shift):
    """x: (B,H,W,C) -> cyclically shifted tensor (Swin shift)."""
    if shift == 0:
        return x
    return torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))


def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def window_reverse(windows, ws, H, W):
    B = windows.shape[0] // (H // ws) // (W // ws)
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


def drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device).bernoulli_(1.0 - p)
    return x * mask / (1.0 - p)


def swin_shift_mask(B, H, W, ws, shift, device):
    """Attention mask for shifted windows.

    Two tokens may attend iff they came from the same window of the UNSHIFTED
    image. Ids are tiled on the original grid, rolled exactly like the data, then
    partitioned the same way.
    """
    sh = ws // 2 if shift else 0
    ids = torch.zeros((1, H, W, 1), device=device)
    cnt = 0
    for i in range(0, H, ws):
        for j in range(0, W, ws):
            ids[:, i:i + ws, j:j + ws, :] = cnt
            cnt += 1
    if sh:
        ids = torch.roll(ids, shifts=(-sh, -sh), dims=(1, 2))
    win = window_partition(ids, ws).squeeze(-1)                    # (nW, N)
    attn = win.unsqueeze(2) - win.unsqueeze(1)                     # (nW, N, N)
    attn = attn.masked_fill(attn != 0, float("-inf"))
    nW = win.shape[0]
    return attn.reshape(1, nW, 1, win.shape[1], win.shape[1]) \
               .expand(B, -1, -1, -1, -1).reshape(B * nW, 1,
                                                   win.shape[1], win.shape[1])


# ------------------------------------------------------------------- window attention
class WindowAttention(nn.Module):
    """Token-level attention inside a ws x ws window grid, with relative bias."""

    def __init__(self, dim, heads, ws, shift=False):
        super().__init__()
        self.dim, self.heads, self.ws, self.shift = dim, heads, ws, shift
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros(heads, (2 * ws - 1) ** 2))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, x, attn_mask=None):
        """x: (B*nW, N, C); attn_mask: (B*nW, 1, N, N) additive (0 / -inf) or None."""
        BnW, N, C = x.shape
        qkv = self.qkv(x).reshape(BnW, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)                 # (BnW,h,N,d)
        attn = (q * self.scale) @ k.transpose(-2, -1)                  # (BnW,h,N,N)

        n = int(math.sqrt(N))
        assert n * n == N, f"window not square: {N}"
        coords = torch.stack(torch.meshgrid(torch.arange(n, device=x.device),
                                            torch.arange(n, device=x.device),
                                            indexing="ij")).reshape(2, -1)  # (2,N)
        rel = coords.unsqueeze(2) - coords.unsqueeze(1)                 # (2,N,N)
        rel[0] += n - 1
        rel[1] += n - 1
        idx = (rel[0] * (2 * n - 1) + rel[1]).reshape(-1)               # (N*N,)
        bias = self.rel_bias[:, idx].reshape(self.heads, N, N).unsqueeze(0)
        attn = attn + bias                                             # (1,h,N,N)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(BnW, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------- blocks
class SwinBlock(nn.Module):
    """Window attention with alternating cyclic shift (the Swin mechanism)."""

    def __init__(self, dim, heads, ws, shift, drop_path=DROP_PATH):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, ws, shift)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * MLP_RATIO)),
                                 nn.GELU(),
                                 nn.Linear(int(dim * MLP_RATIO), dim))
        self.dp = drop_path

    def forward(self, x):
        B, H, W, C = x.shape
        sh = self.ws // 2 if self.shift else 0
        x_s = roll(x, sh)
        wins = window_partition(self.norm1(x_s), self.ws)              # (B*nW,N,C)
        mask = swin_shift_mask(B, H, W, self.ws, self.shift,
                               x.device) if self.shift else None
        wins = self.attn(wins, mask)
        out = window_reverse(wins, self.ws, H, W)
        out = roll(out, -sh) if self.shift else out
        x = x + drop_path(out, self.dp, self.training)
        x = x + self.mlp(self.norm2(x))
        return x


class ChannelAttention(nn.Module):
    """Channel-level attention (cross-covariance over feature channels)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        B, H, W, C = x.shape
        xn = self.norm(x).reshape(B, H * W, C)
        q, k, v = self.q(xn), self.k(xn), self.v(xn)                   # (B,N,C)
        attn = (k.transpose(1, 2) @ q) * self.scale                     # (B,C,C)
        attn = attn.softmax(dim=-1)
        out = self.proj(v @ attn).reshape(B, H, W, C)
        return out


class DualAttentionBlock(nn.Module):
    """Dual-level attention: (1) token-level spatial window attention,
    (2) channel-level cross-covariance attention, then MLP."""

    def __init__(self, dim, heads, ws, drop_path=DROP_PATH):
        super().__init__()
        self.ws = ws
        self.norm1 = nn.LayerNorm(dim)
        self.spatial = WindowAttention(dim, heads, ws, shift=False)
        self.norm2 = nn.LayerNorm(dim)
        self.channel = ChannelAttention(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * MLP_RATIO)),
                                 nn.GELU(),
                                 nn.Linear(int(dim * MLP_RATIO), dim))
        self.dp = drop_path

    def forward(self, x):
        B, H, W, C = x.shape
        wins = window_partition(self.norm1(x), self.ws)
        out = window_reverse(self.spatial(wins), self.ws, H, W)
        x = x + drop_path(out, self.dp, self.training)
        x = x + drop_path(self.channel(x), self.dp, self.training)
        x = x + self.mlp(self.norm3(x))
        return x


# ------------------------------------------------------------------------- full nets
class DenseTransformer(nn.Module):
    """Patchify -> N attention blocks -> per-pixel 4x4 head (dense logits)."""

    def __init__(self, in_ch=9, n_classes=4, block="swin"):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, WIDTH, kernel_size=PATCH, stride=PATCH)
        blocks = []
        for i in range(DEPTH):
            if block == "swin":
                blocks.append(SwinBlock(WIDTH, HEADS, WINDOW, shift=(i % 2 == 1)))
            else:
                blocks.append(DualAttentionBlock(WIDTH, HEADS, WINDOW))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, n_classes * PATCH * PATCH)
        self.n_classes = n_classes

    def forward(self, x):
        """x: (B,C,H,W), H/W divisible by PATCH and WINDOW -> logits (B,K,H,W)."""
        B, C, H, W = x.shape
        t = self.patch(x)                              # (B,WIDTH,H/4,W/4)
        t = t.permute(0, 2, 3, 1)                      # (B,th,tw,WIDTH)
        for blk in self.blocks:
            t = blk(t)
        t = self.head(self.norm(t))                    # (B,th,tw,K*4*4)
        th, tw = t.shape[1], t.shape[2]
        t = t.view(B, th, tw, self.n_classes, PATCH, PATCH)
        t = t.permute(0, 3, 1, 4, 2, 5).reshape(B, self.n_classes,
                                                 th * PATCH, tw * PATCH)
        return t


def build_model(kind, in_ch=9, n_classes=4):
    if kind == "mts":
        return DenseTransformer(in_ch=in_ch, n_classes=n_classes, block="swin")
    if kind == "dla":
        return DenseTransformer(in_ch=in_ch, n_classes=n_classes, block="dual")
    raise ValueError(kind)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
