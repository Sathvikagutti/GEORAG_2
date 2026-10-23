#!/usr/bin/env python3
"""SEPARATE experiment: spatially independent validation (blocked hold-out, 2015).

Splits the 2015 grid into 120x120 blocks with role (block_row+block_col)%3
(1=train, 2=val, 0=test). Train pixels and test pixels are therefore in different
spatial parts of the district. This is NOT the E0 random pixel split and is never
merged into results/model_comparison.csv.

Trains RandomForest + XGBoost here; picks up transformer results from
experiments/transformers/E0_2015_*_seed42_spatial.json (run src/03_train_transformer.py
--spatial first for each transformer).

Output: results/spatial_validation.json + results/spatial_validation.csv
"""
import glob
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

_spec = importlib.util.spec_from_file_location("georag_baselines",
                                               os.path.join(HERE, "02_run_baselines.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

OUT = os.path.join(C.ROOT, "results")


def main():
    feats, labels, valid = C.load_year(2015)
    tr, va, te, meta = C.spatial_split(labels, valid)
    print(f"spatial split 2015: train={len(tr)} val={len(va)} test={len(te)} "
          f"({meta})", flush=True)
    X, y_all = C.flat(feats), labels.ravel()
    X_tr, y_tr = X[tr], y_all[tr].astype(np.int32)
    X_va, y_va = X[va], y_all[va].astype(np.int32)
    X_te, y_te = X[te], y_all[te].astype(np.int32)
    weights = _base.weights_from_y(y_tr)
    sw = C.sklearn_sample_weights(weights, y_tr)

    results = {"experiment": "E0_2015_spatial_block_split",
               "split": meta, "n_train": len(tr), "n_val": len(va),
               "n_test": len(te), "seed": C.SEED, "models": {}}

    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, random_state=C.SEED, n_jobs=2)
    rf.fit(X_tr, y_tr, sample_weight=sw)
    rf_t = time.time() - t0
    t0 = time.time()
    res = C.evaluate(y_te, rf.predict(X_te))
    res["train_time_s"] = round(rf_t, 2)
    res["inference_time_s"] = round(time.time() - t0, 3)
    res["checkpoint"] = None
    res["checkpoint_note"] = "not persisted (see LIMITATIONS.md §6)"
    results["models"]["random_forest"] = res
    print(f"  RF  acc={res['accuracy']:.4f} f1={res['f1_macro']:.4f} "
          f"({rf_t:.1f}s)", flush=True)

    t0 = time.time()
    xgb = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="multi:softprob", random_state=C.SEED,
                        n_jobs=2, tree_method="hist",
                        early_stopping_rounds=40, eval_metric="mlogloss")
    xgb.fit(X_tr, y_tr - 1, sample_weight=sw, eval_set=[(X_va, y_va - 1)],
            verbose=False)
    xgb_t = time.time() - t0
    res = C.evaluate(y_te, xgb.predict(X_te).astype(np.int32) + 1)
    res["train_time_s"] = round(xgb_t, 2)
    res["inference_time_s"] = None
    import pickle
    pkl = os.path.join(C.ROOT, "experiments", "baselines",
                       "xgb_E0_2015_spatial.pkl")
    with open(pkl, "wb") as fh:
        pickle.dump(xgb, fh)
    res["model_size_bytes"] = os.path.getsize(pkl)
    res["checkpoint"] = os.path.relpath(pkl, C.ROOT)
    results["models"]["xgboost"] = res
    print(f"  XGB acc={res['accuracy']:.4f} f1={res['f1_macro']:.4f} "
          f"({xgb_t:.1f}s)", flush=True)

    # transformers (already trained with --spatial)
    for p in sorted(glob.glob(os.path.join(C.ROOT, "experiments", "transformers",
                                           f"E0_2015_*_seed{C.SEED}_spatial.json"))):
        d = C.load_json(p)
        results["models"][f"{d['model']['kind']}_adapted"] = d["results"]
        r = d["results"]
        print(f"  {d['model']['kind']}  acc={r['accuracy']:.4f} "
              f"f1={r['f1_macro']:.4f} ({r['train_time_s']}s)", flush=True)

    C.save_json(results, os.path.join(OUT, "spatial_validation.json"))
    # csv
    rows = ["model,accuracy,precision_macro,recall_macro,f1_macro,kappa,"
            "training_time_s,inference_time_s,n_test"]
    for k, m in results["models"].items():
        rows.append(f"{k},{m['accuracy']:.4f},{m['precision_macro']:.4f},"
                    f"{m['recall_macro']:.4f},{m['f1_macro']:.4f},"
                    f"{m['kappa']:.4f},{m.get('train_time_s')},"
                    f"{m.get('inference_time_s')},{m.get('n_samples')}")
    with open(os.path.join(OUT, "spatial_validation.csv"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print("wrote results/spatial_validation.json/.csv")


if __name__ == "__main__":
    main()
