#!/usr/bin/env python3
"""Train one adapted transformer (mts / dla) on one experiment.

Usage:
  python src/03_train_transformer.py --model dla --experiment E0_2015

Split logic is imported from src/02_run_baselines.py so transformers see EXACTLY
the same train/val/test pixels as the RF/XGBoost baselines.
"""
import argparse
import importlib.util
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import models as M  # noqa: E402

_spec = importlib.util.spec_from_file_location("georag_baselines",
                                               os.path.join(HERE, "02_run_baselines.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

PAD = 96          # reflect pad around the grid for random crop sampling
CROP = 140        # 140 px = 35 tokens = 5x5 windows of 7 (divisible by 4 and 7)
FULL_H, FULL_W = 672, 700   # 669->672, 693->700 so tokens 168x175 are 7-divisible
OUT_DIR = os.path.join(C.ROOT, "experiments", "transformers")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")


def split_for(exp, spatial=False):
    if spatial:
        if exp != "E0_2015":
            raise SystemExit("--spatial is implemented for E0_2015 only")
        _, labels, valid = C.load_year(2015)
        tr, va, te, _meta = C.spatial_split(labels, valid)
        return tr, va, te, (2015,), 2015
    s = _base.get_split(exp)
    if len(s) == 6:
        tr, va, _te, train_years, test_year, te_idx = s
    else:
        tr, va, te, train_years, test_year = s
        te_idx = te
    return tr, va, te_idx, train_years, test_year


def flat_mask(idx, shape=(C.GRID_H, C.GRID_W)):
    m = np.zeros(shape[0] * shape[1], dtype=bool)
    m[np.asarray(idx)] = True
    return m.reshape(shape)


def ce_targets(labels):
    """Native labels {0(background),1..4} -> CE class indices {0..3}, bg=0 (masked)."""
    return np.where(labels >= 1, labels - 1, 0).astype(np.int64)


class ExperimentData:
    """Standardized arrays + padded crop tensors + masks for one experiment."""

    def __init__(self, exp, kind, train_years, test_year, tr, va, te_idx):
        self.exp, self.kind = exp, kind
        self.in_ch = 19 if kind == "mts" else 9
        self.train_years = list(train_years)
        self.test_year = test_year

        # normalization stats from TRAIN pixels only (no test-year information)
        parts = []
        for y in self.train_years:
            feats, _, _ = C.load_year(y)
            parts.append(C.flat(feats)[self.year_idx(y, tr, "train")])
        all_tr = np.concatenate(parts)
        self.mean = all_tr.mean(axis=0).astype(np.float32)
        self.std = all_tr.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0

        self.feats, self.labels, self.valid = {}, {}, {}
        self.x_pad, self.y_pad = {}, {}
        self.train_pad, self.val_pad = {}, {}
        for y in set(self.train_years + [test_year]):
            feats, labels, valid = C.load_year(y)
            x = np.moveaxis(feats, 0, -1)                     # (H,W,9)
            x = (x - self.mean) / self.std
            x[~np.isfinite(x)] = 0.0                          # invalid -> train mean
            x = np.moveaxis(x, -1, 0).astype(np.float32)      # (9,H,W)
            self.feats[y], self.labels[y], self.valid[y] = x, labels, valid
            self.x_pad[y] = np.pad(x, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect")
            self.y_pad[y] = np.pad(ce_targets(labels), PAD, mode="constant",
                                   constant_values=0)
            trm = flat_mask(self.year_idx(y, tr, "train")) if y in self.train_years \
                else np.zeros_like(valid)
            vam = flat_mask(self.year_idx(y, va, "val")) if y in self.train_years \
                else np.zeros_like(valid)
            self.train_pad[y] = np.pad(trm, PAD, mode="constant",
                                       constant_values=False)
            self.val_pad[y] = np.pad(vam, PAD, mode="constant", constant_values=False)

        # multi-temporal history slot (mts only): [hist, cur, flag]
        self.hist_full, self.hist_pad = {}, {}
        if kind == "mts":
            for y in set(self.train_years + [test_year]):
                if exp.startswith("E3") and y == 2020:
                    h = self.feats[2015]
                elif exp.startswith("E3") and y == 2025:
                    h = self.feats[2020]
                else:
                    h = np.zeros_like(self.feats[y])
                self.hist_full[y] = h
                self.hist_pad[y] = np.pad(h, ((0, 0), (PAD, PAD), (PAD, PAD)),
                                          mode="reflect")

        ys = np.concatenate([self.labels[y].ravel()[self.year_idx(y, tr, "train")]
                             for y in self.train_years])
        self.weights = _base.weights_from_y(ys)
        self.w_t = torch.tensor([self.weights[c] for c in C.CLASSES],
                                dtype=torch.float32)

    def year_idx(self, y, idx, which):
        if self.exp.startswith("E3"):
            return _base.pooled()[y][0 if which == "train" else 1]
        return idx

    # ---- inputs -----------------------------------------------------------
    def crop_input(self, y, r, c):
        cur = self.x_pad[y][:, r:r + CROP, c:c + CROP]
        if self.kind != "mts":
            return cur
        hist = self.hist_pad[y][:, r:r + CROP, c:c + CROP]
        flag = (1.0 if np.any(self.hist_full[y]) else 0.0)
        ch = np.full((1, CROP, CROP), flag, dtype=np.float32)
        return np.concatenate([hist, cur, ch], axis=0)

    def full_input(self, y):
        """(C, 672, 700) reflect-padded full-grid input."""
        cur = self.feats[y]
        if self.kind != "mts":
            x = cur
        else:
            hist = self.hist_full[y]
            flag = np.full((1, C.GRID_H, C.GRID_W),
                           1.0 if np.any(hist) else 0.0, dtype=np.float32)
            x = np.concatenate([hist, cur, flag], axis=0)
        return np.pad(x, ((0, 0), (0, FULL_H - C.GRID_H), (0, FULL_W - C.GRID_W)),
                      mode="reflect")


def full_grid_forward(model, data, y):
    """One dense pass over the whole grid -> (logits[4,669,693], seconds)."""
    xt = torch.from_numpy(data.full_input(y)[None])
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        logits = model(xt)
        secs = time.time() - t0
    return logits[0, :, :C.GRID_H, :C.GRID_W], secs


def evaluate_val(model, data):
    """Unweighted CE over each training year's val pixels (checkpoint selection)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for y in data.train_years:
            logits, _ = full_grid_forward(model, data, y)
            vm = torch.from_numpy(data.val_pad[y][PAD:PAD + C.GRID_H,
                                                  PAD:PAD + C.GRID_W])
            if int(vm.sum()) == 0:
                continue
            lab = torch.from_numpy(ce_targets(data.labels[y]))
            loss = F.cross_entropy(logits.unsqueeze(0), lab.unsqueeze(0),
                                   reduction="none")[0]
            total += float(loss[vm].sum())
            n += int(vm.sum())
    return total / max(n, 1)


def train(exp, kind, seed=42, max_epochs=25, patience=5, steps=80, batch=4,
          lr=3e-4, spatial=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    tr, va, te_idx, train_years, test_year = split_for(exp, spatial=spatial)
    data = ExperimentData(exp, kind, train_years, test_year, tr, va, te_idx)
    model = M.build_model(kind, in_ch=data.in_ch, n_classes=4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs,
                                                       eta_min=3e-5)
    n_params = M.count_params(model)
    H2, W2 = C.GRID_H + 2 * PAD, C.GRID_W + 2 * PAD
    print(f"[{exp}/{kind}] params={n_params} train_years={train_years} "
          f"test={test_year} in_ch={data.in_ch}", flush=True)

    best_val, best_epoch, bad = float("inf"), 0, 0
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{exp}_{kind}_seed{seed}.pt")
    t_start = time.time()
    epoch, last_loss = 0, float("nan")
    for epoch in range(1, max_epochs + 1):
        model.train()
        for _ in range(steps):
            xs, ys, ms = [], [], []
            for _b in range(batch):
                y = data.train_years[np.random.randint(len(data.train_years))]
                r = np.random.randint(0, H2 - CROP)
                c = np.random.randint(0, W2 - CROP)
                xs.append(data.crop_input(y, r, c))
                ys.append(data.y_pad[y][r:r + CROP, c:c + CROP])
                ms.append(data.train_pad[y][r:r + CROP, c:c + CROP])
            xt = torch.from_numpy(np.stack(xs))
            yt = torch.from_numpy(np.stack(ys))
            mt = torch.from_numpy(np.stack(ms))
            if int(mt.sum()) == 0:
                continue
            logits = model(xt)
            loss = F.cross_entropy(logits, yt, weight=data.w_t, reduction="none")
            last_loss = float((loss * mt).sum().detach() / mt.sum())
            opt.zero_grad()
            (loss * mt).sum().backward()
            opt.step()
        sched.step()

        val_loss = evaluate_val(model, data)
        improved = val_loss < best_val - 1e-4
        if improved:
            best_val, best_epoch, bad = val_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "mean": data.mean,
                        "std": data.std, "in_ch": data.in_ch, "kind": kind,
                        "exp": exp, "seed": seed, "best_epoch": best_epoch,
                        "val_loss": best_val, "params": n_params},
                       ckpt_path)
        else:
            bad += 1
        print(f"  epoch {epoch:2d} train_loss={last_loss:.4f} "
              f"val_loss={val_loss:.4f}{'  BEST' if improved else ''}", flush=True)
        if bad >= patience:
            print(f"  early stop at epoch {epoch} (best {best_epoch})", flush=True)
            break
    train_time = time.time() - t_start

    # final: best checkpoint -> full-grid inference -> test metrics
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    logits, inf_secs = full_grid_forward(model, data, test_year)
    pred = logits.argmax(dim=0).numpy() + 1                  # 0-based -> native 1-4
    te_mask = flat_mask(te_idx) if test_year in data.train_years \
        else data.valid[test_year]
    res = C.evaluate(data.labels[test_year][te_mask], pred[te_mask])
    res["train_time_s"] = round(train_time, 2)
    res["inference_time_s"] = round(inf_secs, 3)
    res["model_size_bytes"] = int(os.path.getsize(ckpt_path))
    res["checkpoint"] = os.path.relpath(ckpt_path, C.ROOT)
    res["params"] = n_params
    res["best_epoch"] = best_epoch
    res["epochs_run"] = epoch
    res["best_val_loss"] = round(best_val, 5)

    out = {
        "experiment": exp, "train_years": list(data.train_years),
        "test_year": test_year, "seed": seed,
        "model": {"name": f"{kind}-lite-adapted", "kind": kind,
                  "architecture_source": (
                      "custom adaptation in the Swin-style shifted-window direction "
                      "of Voelsen et al. 2024 MTS; NO official code was available")
                  if kind == "mts" else
                  ("custom adaptation in the token+channel dual-attention direction "
                   "of Bansal & Tripathi 2024; NO official code was available"),
                  "params": n_params, "input_channels": data.in_ch},
        "input_type": ("patch_tokens_19ch_hist_cur_flag" if kind == "mts"
                       else "patch_tokens_9ch"),
        "validation_type": (
            "spatially_independent_block_holdout" if spatial else
            "stratified_random_pixel_split" if test_year in data.train_years
            else "temporal_holdout_test_year"),
        "config": {"optimizer": "AdamW", "lr": lr, "weight_decay": 0.01,
                   "batch": batch, "steps_per_epoch": steps, "crop_px": CROP,
                   "max_epochs": max_epochs, "patience": patience,
                   "class_weights": data.weights,
                   "normalization": "per-channel z-score fit on train pixels only",
                   "loss": "cross-entropy with sqrt-inverse-frequency class weights"},
        "n_train_pixels": int(sum(flat_mask(data.year_idx(y, tr, "train")).sum()
                                  for y in data.train_years)),
        "n_val_pixels": int(sum(flat_mask(data.year_idx(y, va, "val")).sum()
                                for y in data.train_years)),
        "n_test_pixels": int(te_mask.sum()),
        "hardware": "CPU-only, 2 cores, no GPU (torch.set_num_threads(2))",
        "results": res,
    }
    json_path = os.path.join(OUT_DIR,
                             f"{exp}_{kind}_seed{seed}"
                             f"{'_spatial' if spatial else ''}.json")
    C.save_json(out, json_path)
    print(f"  -> test acc={res['accuracy']:.4f} f1={res['f1_macro']:.4f} "
          f"train={res['train_time_s']}s infer={res['inference_time_s']}s "
          f"best_epoch={best_epoch}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mts", "dla"], required=True)
    ap.add_argument("--experiment", required=True,
                    choices=["E0_2015", "E0_2020", "E0_2025",
                             "E1_2015_to_2020", "E2_2015_to_2025",
                             "E3_2015_2020_to_2025"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--spatial", action="store_true",
                    help="use the spatially independent block split (E0_2015 only)")
    a = ap.parse_args()
    train(a.experiment, a.model, seed=a.seed, max_epochs=a.max_epochs,
          patience=a.patience, steps=a.steps, batch=a.batch, lr=a.lr,
          spatial=a.spatial)


if __name__ == "__main__":
    main()
