#!/usr/bin/env python3
"""Train the tabular baselines (Random Forest, XGBoost) for all experiments.

These are re-trained here from the Drive data — the Drive contains no previous
Member 2 code or results. Outputs go to experiments/baselines/ (nothing outside
that directory is touched, and no file is ever overwritten from a previous run
without the model name in the filename).
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT_DIR = os.path.join(C.ROOT, "experiments", "baselines")
EXPERIMENTS = ["E0_2015", "E0_2020", "E0_2025",
               "E1_2015_to_2020", "E2_2015_to_2025", "E3_2015_2020_to_2025"]

_cache = {}
_pooled = None


def load_single(year):
    if year not in _cache:
        _cache[year] = C.load_year(year)
    return _cache[year]


def pooled():
    global _pooled
    if _pooled is None:
        _pooled = C.pooled_split((2015, 2020))
    return _pooled


def get_split(exp):
    """Return (train_idx, val_idx, test_idx, train_years, test_year).

    train/val idx are FLAT grid positions of the training years' pixels.
    For temporal experiments test_idx = ALL valid pixels of the test year.
    For same-year experiments test_idx = the held-out 30 % stratified split.
    """
    train_years, test_year = C.experiment_years(exp)
    same_year = test_year in train_years and len(train_years) == 1
    if same_year:
        _, labels, valid = load_single(train_years[0])
        tr, va, te = C.stratified_split(valid, labels)
        return tr, va, te, train_years, test_year
    if exp.startswith("E3"):
        sp = pooled()
        tr = np.sort(np.concatenate([sp[2015][0], sp[2020][0]]))
        va = np.sort(np.concatenate([sp[2015][1], sp[2020][1]]))
        te = np.sort(np.concatenate([sp[2015][2], sp[2020][2]]))
    else:  # E1 / E2: split within 2015 only
        _, labels, valid = load_single(2015)
        tr, va, te = C.stratified_split(valid, labels)
    _, labels_t, valid_t = load_single(test_year)
    te_idx = np.flatnonzero(valid_t.ravel())
    return tr, va, te, train_years, test_year, te_idx


def train_index_for_year(exp, train_years, tr):
    """Per-year train index (pooled splits store indices per year)."""
    if exp.startswith("E3"):
        return {y: pooled()[y][0] for y in train_years}
    return {train_years[0]: tr}


def val_index_for_year(exp, train_years, va):
    if exp.startswith("E3"):
        return {y: pooled()[y][1] for y in train_years}
    return {train_years[0]: va}


def weights_from_y(y):
    counts = np.array([(y == c).sum() for c in C.CLASSES], dtype=np.float64)
    freq = counts / counts.sum()
    w = np.sqrt(1.0 / np.maximum(freq, 1e-12))
    w = w / w.mean()
    return {c: float(w[i]) for i, c in enumerate(C.CLASSES)}


def build_xy(exp, which, tr, va):
    """Stack X/y for train or val across training years."""
    train_years, _ = C.experiment_years(exp)
    idx_map = train_index_for_year(exp, train_years, tr) if which == "train" \
        else val_index_for_year(exp, train_years, va)
    Xs, ys = [], []
    for y in train_years:
        feats, labels, _ = load_single(y)
        idx = idx_map[y]
        Xs.append(C.flat(feats)[idx])
        ys.append(labels.ravel()[idx])
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int32)


def run():
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    os.makedirs(OUT_DIR, exist_ok=True)
    for exp in EXPERIMENTS:
        split = get_split(exp)
        if len(split) == 6:
            tr, va, _te, train_years, test_year, te_idx = split
        else:
            tr, va, te, train_years, test_year = split
            te_idx = te
        print(f"\n=== {exp}: train={list(train_years)} test={test_year} ===", flush=True)

        X_tr, y_tr = build_xy(exp, "train", tr, va)
        X_va, y_va = build_xy(exp, "val", tr, va)
        feats_t, labels_t, valid_t = load_single(test_year)
        X_test = C.flat(feats_t)[te_idx]
        y_test = labels_t.ravel()[te_idx].astype(np.int32)

        weights = weights_from_y(y_tr)
        sw_tr = C.sklearn_sample_weights(weights, y_tr)

        results = {"experiment": exp, "train_years": list(train_years),
                   "test_year": test_year, "seed": C.SEED,
                   "input_type": "tabular_9_features",
                   "validation_type": "stratified_random_pixel_split" if test_year in train_years
                   else "temporal_holdout_test_year",
                   "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
                   "n_test": int(len(y_test)), "class_weights": weights,
                   "models": {}}

        # Random Forest --------------------------------------------------------------
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=300, random_state=C.SEED, n_jobs=2)
        rf.fit(X_tr, y_tr, sample_weight=sw_tr)
        rf_train_s = time.time() - t0
        t0 = time.time()
        pred_rf = rf.predict(X_test)
        rf_inf_s = time.time() - t0
        res = C.evaluate(y_test, pred_rf)
        res["train_time_s"] = round(rf_train_s, 2)
        res["inference_time_s"] = round(rf_inf_s, 3)
        res["model_size_bytes"] = _measure_and_discard(rf, f"rf_{exp}")
        res["checkpoint"] = None
        res["checkpoint_note"] = ("RF pickle was measured then discarded (too large to "
                                  "bundle); retrain exactly with src/02_run_baselines.py, "
                                  "seed 42")
        results["models"]["random_forest"] = res
        print(f"  RF  acc={res['accuracy']:.4f} f1={res['f1_macro']:.4f} "
              f"({rf_train_s:.1f}s, inf {rf_inf_s:.2f}s)", flush=True)

        # XGBoost --------------------------------------------------------------------
        t0 = time.time()
        xgb = XGBClassifier(
            n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob",
            random_state=C.SEED, n_jobs=2, tree_method="hist",
            early_stopping_rounds=40, eval_metric="mlogloss",
        )
        xgb.fit(X_tr, y_tr - 1, sample_weight=sw_tr,
                eval_set=[(X_va, y_va - 1)], verbose=False)
        xgb_train_s = time.time() - t0
        t0 = time.time()
        pred_xgb = xgb.predict(X_test).astype(np.int32) + 1
        xgb_inf_s = time.time() - t0
        res = C.evaluate(y_test, pred_xgb)
        res["train_time_s"] = round(xgb_train_s, 2)
        res["inference_time_s"] = round(xgb_inf_s, 3)
        res["best_iteration"] = int(getattr(xgb, "best_iteration", 0)) + 1
        xgb_path = _save(xgb, f"xgb_{exp}")
        res["model_size_bytes"] = int(os.path.getsize(xgb_path))
        res["checkpoint"] = os.path.relpath(xgb_path, C.ROOT)
        results["models"]["xgboost"] = res
        print(f"  XGB acc={res['accuracy']:.4f} f1={res['f1_macro']:.4f} "
              f"({xgb_train_s:.1f}s, inf {xgb_inf_s:.2f}s, "
              f"iters={res['best_iteration']})", flush=True)

        C.save_json(results, os.path.join(OUT_DIR, f"{exp}.json"))


def _save(model, name):
    path = os.path.join(OUT_DIR, name + ".pkl")
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    return path


def _measure_and_discard(model, name):
    """Serialize once to measure the real model size, then delete the file."""
    path = os.path.join(OUT_DIR, name + "._sizecheck.pkl")
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    size = os.path.getsize(path)
    os.remove(path)
    return int(size)


if __name__ == "__main__":
    run()
