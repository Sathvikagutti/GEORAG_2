"""Shared helpers: data loading, splits, class weights, metrics.

Single source of truth for evaluation so every model is scored identically.
"""
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVED = os.path.join(ROOT, "derived")
GRID_W, GRID_H = 693, 669
YEARS = (2015, 2020, 2025)
SEED = 42
FEATURE_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDBI", "NDWI"]
CLASSES = (1, 2, 3, 4)  # native ids: 1 Water, 2 Urban, 3 Soil, 4 Vegetation


def load_year(year):
    """Return (feats[9,H,W] float32, labels[H,W] uint8, valid[H,W] bool)."""
    feats = np.load(os.path.join(DERIVED, f"{year}_features.npy"))
    labels = np.load(os.path.join(DERIVED, f"{year}_labels.npy"))
    valid = np.all(np.isfinite(feats), axis=0) & np.isin(labels, CLASSES)
    return feats, labels, valid


def flat(feats):
    """[9,H,W] -> [H*W,9] pixel-first table."""
    return np.moveaxis(feats, 0, -1).reshape(-1, feats.shape[0])


def stratified_split(valid_mask, labels, train=0.60, val=0.10, seed=SEED):
    """60/10/30 stratified split over valid pixels of one year.

    Returns index arrays into the flattened H*W grid.
    """
    from sklearn.model_selection import train_test_split
    idx = np.flatnonzero(valid_mask.ravel())
    y = labels.ravel()[idx]
    tr, rest = train_test_split(idx, test_size=1 - train, stratify=y, random_state=seed)
    te_share = 0.30 / (0.30 + 0.10)
    va, te = train_test_split(rest, test_size=te_share,
                              stratify=labels.ravel()[rest], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def pooled_split(years, train=0.60, seed=SEED):
    """Stratified 60/10/30 split over the pooled valid pixels of the given years."""
    from sklearn.model_selection import train_test_split
    idxs, ys = [], []
    for y in years:
        _, labels, valid = load_year(y)
        idx = np.flatnonzero(valid.ravel())
        idxs.append(idx)
        ys.append(labels.ravel()[idx])
    cat_idx = np.concatenate(idxs)
    cat_y = np.concatenate(ys)
    pos = np.arange(len(cat_idx))
    pos_tr, pos_rest = train_test_split(pos, test_size=1 - train, stratify=cat_y,
                                        random_state=seed)
    te_share = 0.30 / (0.30 + 0.10)
    pos_va, pos_te = train_test_split(pos_rest, test_size=te_share,
                                      stratify=cat_y[pos_rest], random_state=seed)
    tr, va, te = cat_idx[pos_tr], cat_idx[pos_va], cat_idx[pos_te]
    # split membership back per year
    out = {}
    for y, idx in zip(years, idxs):
        out[y] = (np.sort(tr[np.isin(tr, idx)]),
                  np.sort(va[np.isin(va, idx)]),
                  np.sort(te[np.isin(te, idx)]))
    return out


def class_weights(labels, idx):
    """Inverse-frequency sqrt weights normalized to mean 1, from a train index set."""
    y = labels.ravel()[idx]
    counts = np.array([(y == c).sum() for c in CLASSES], dtype=np.float64)
    freq = counts / counts.sum()
    w = np.sqrt(1.0 / np.maximum(freq, 1e-12))
    w = w / w.mean()
    return {c: float(w[i]) for i, c in enumerate(CLASSES)}


def sklearn_sample_weights(weights, y):
    return np.array([weights[int(v)] for v in y], dtype=np.float64)


def evaluate(y_true, y_pred, prefix=""):
    """The ONE metric function used for every model in this project."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(CLASSES)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                  labels=labels, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           labels=labels, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }
    p = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    for i, c in enumerate(labels):
        out["per_class"][str(c)] = {
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
        }
    return {prefix + k: v for k, v in out.items()} if prefix else out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def spatial_split(labels, valid, block=120):
    """Spatially independent checkerboard block split (SEPARATE experiment).

    Blocks of `block` px on each axis; role = (block_row + block_col) % 3:
    1 = train, 2 = val, 0 = held-out spatial test. Deterministic (no seed).
    This is NOT the random pixel split used in E0.
    """
    H, W = valid.shape
    rows = np.arange(H) // block
    cols = np.arange(W) // block
    idx = np.flatnonzero(valid.ravel())
    r = (idx // W) // block
    c = (idx % W) // block
    rm = (r + c) % 3
    return (np.sort(idx[rm == 1]), np.sort(idx[rm == 2]), np.sort(idx[rm == 0]),
            {"block_px": block, "role_rule": "(block_row+block_col)%3: "
                                              "1=train,2=val,0=test"})


def experiment_years(exp):
    """Map experiment id -> (train_years, test_year)."""
    return {
        "E0_2015": ((2015,), 2015), "E0_2020": ((2020,), 2020), "E0_2025": ((2025,), 2025),
        "E1_2015_to_2020": ((2015,), 2020), "E2_2015_to_2025": ((2015,), 2025),
        "E3_2015_2020_to_2025": ((2015, 2020), 2025),
    }[exp]
