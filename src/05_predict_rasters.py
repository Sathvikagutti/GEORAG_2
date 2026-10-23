#!/usr/bin/env python3
"""Generate prediction GeoTIFFs for selected model families, one per year.

- Models: the evaluated E0_<year> configuration of each requested family
  (random_forest, xgboost, dla, mts) - the same models whose metrics were scored.
- Coverage: every pixel with finite 9 features (labels are NOT required to predict).
- Output class numbering (brief's required scheme):
    1 = Built-up (native 2 Urban)   2 = Vegetation (native 4)
    3 = Water     (native 1 Water)  4 = Open Land  (native 3 Soil)
    0 = NoData (unmapped pixels: missing imagery / outside scene)
- Original Drive files are never touched; outputs go to outputs/predictions/.

Usage: python src/05_predict_rasters.py --models dla mts xgboost random_forest
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT_DIR = os.path.join(C.ROOT, "outputs", "predictions")
YEARS = (2015, 2020, 2025)

# native (training) id -> output id required by the brief
NATIVE_TO_OUT = {1: 3, 2: 1, 3: 4, 4: 2}   # Water->3, Urban->1, Soil->4, Veg->2
OUT_NAMES = {0: "NoData", 1: "Built-up", 2: "Vegetation", 3: "Water", 4: "Open Land"}
NATIVE_NAMES = {0: "background", 1: "Water", 2: "Urban", 3: "Soil", 4: "Vegetation"}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "georag_baselines", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "02_run_baselines.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

MODEL_FILENAME = {"random_forest": "random_forest", "xgboost": "xgboost",
                  "dla": "dla-lite-adapted", "mts": "mts-lite-adapted"}


def native_to_out(arr):
    out = np.zeros_like(arr, dtype=np.uint8)
    for nat, o in NATIVE_TO_OUT.items():
        out[arr == nat] = o
    return out


def write_tif(path, data_out, model_name, year, extra_tags=None):
    """data_out: uint8 array in OUTPUT numbering, 0 = NoData."""
    profile = {
        "driver": "GTiff", "height": C.GRID_H, "width": C.GRID_W, "count": 1,
        "dtype": "uint8", "crs": "EPSG:4326",
        "transform": rasterio.Affine(0.00026949458523585647, 0.0,
                                     80.55489596743509, 0.0,
                                     -0.00026949458523585647,
                                     16.596285042579748),
        "nodata": 0, "compress": "deflate", "tiled": False,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data_out, 1)
        dst.set_band_description(1, "land_use (output scheme)")
        tags = {
            "class_scheme_output": " ".join(f"{k}={v}" for k, v in OUT_NAMES.items()),
            "class_scheme_native_member1": " ".join(
                f"{k}={v}" for k, v in NATIVE_NAMES.items()),
            "mapping_native_to_output": "1Water->3, 2Urban->1Built-up, "
                                        "3Soil->4OpenLand, 4Vegetation->2",
            "model": model_name, "year": str(year),
            "grid": f"{C.GRID_W}x{C.GRID_H} 30m EPSG:4326 shared analysis grid",
            "generated_by": "GeoRAG Member 2 model-upgrade pipeline (src/05_predict_rasters.py)",
        }
        if extra_tags:
            tags.update(extra_tags)
        dst.update_tags(**tags)


def predict_year(family, year):
    """Return native-id prediction array (uint8, 0 where unmapped) + timing."""
    feats, labels, valid = C.load_year(year)
    finite = np.all(np.isfinite(feats), axis=0)

    if family in ("random_forest", "xgboost"):
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        X = C.flat(feats)
        t0 = time.time()
        if family == "xgboost":
            pkl = os.path.join(C.ROOT, "experiments", "baselines", f"xgb_E0_{year}.pkl")
            with open(pkl, "rb") as fh:
                model = pickle.load(fh)
            pred0 = model.predict(X)
            pred = pred0.astype(np.int32) + 1
        else:
            # deterministic retrain of the discarded RF checkpoint (seed 42)
            _, lab_tr, _ = C.load_year(year)
            tr, _va, _te = _base.get_split(f"E0_{year}")[:3]
            y_tr = lab_tr.ravel()[tr]
            weights = _base.weights_from_y(y_tr)
            sw = C.sklearn_sample_weights(weights, y_tr)
            model = RandomForestClassifier(n_estimators=300, random_state=C.SEED,
                                           n_jobs=2)
            model.fit(X[tr], y_tr, sample_weight=sw)
            pred = model.predict(X)
        infer_s = time.time() - t0
    else:  # transformer checkpoints
        import torch
        import models as M
        torch.set_num_threads(2)
        ckpt_path = os.path.join(C.ROOT, "experiments", "transformers",
                                 "checkpoints", f"E0_{year}_{family}_seed42.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = M.build_model(family, in_ch=ckpt["in_ch"], n_classes=4)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        mean, std = ckpt["mean"], ckpt["std"]
        x = np.moveaxis(feats, 0, -1)
        x = (x - mean) / std
        x[~np.isfinite(x)] = 0.0
        x = np.moveaxis(x, -1, 0).astype(np.float32)          # (9,H,W)
        if ckpt["in_ch"] == 19:
            # mts history slot: E0 mapping models are single-date (as they were
            # trained) -> history = zeros (standardized mean), flag = 0
            zeros = np.zeros_like(x)
            x = np.concatenate([zeros, x, np.zeros((1, C.GRID_H, C.GRID_W),
                                                   dtype=np.float32)], axis=0)
        x = np.pad(x, ((0, 0), (0, 672 - C.GRID_H), (0, 700 - C.GRID_W)),
                   mode="reflect")
        t0 = time.time()
        with torch.no_grad():
            logits = model(torch.from_numpy(x[None]))
        infer_s = time.time() - t0
        pred = logits[0, :, :C.GRID_H, :C.GRID_W].argmax(0).numpy() + 1

    out = np.zeros((C.GRID_H, C.GRID_W), dtype=np.uint8)
    pred = np.asarray(pred).reshape(C.GRID_H, C.GRID_W)
    out[finite] = pred[finite]                 # map everything with imagery
    return native_to_out(out), infer_s, finite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    choices=["random_forest", "xgboost", "dla", "mts"])
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}
    for fam in a.models:
        summary[fam] = {}
        for year in YEARS:
            t0 = time.time()
            data_out, infer_s, finite = predict_year(fam, year)
            fname = f"pred_{year}_{MODEL_FILENAME[fam]}.tif"
            path = os.path.join(OUT_DIR, fname)
            write_tif(path, data_out, MODEL_FILENAME[fam], year,
                      {"inference_time_s": f"{infer_s:.3f}"})
            vals, cnts = np.unique(data_out, return_counts=True)
            summary[fam][year] = {
                "file": os.path.relpath(path, C.ROOT),
                "mapped_pixels": int((data_out > 0).sum()),
                "nodata_pixels": int((data_out == 0).sum()),
                "class_counts_out": {int(v): int(c) for v, c in zip(vals, cnts)},
                "inference_time_s": round(infer_s, 3),
                "wall_s": round(time.time() - t0, 1),
            }
            print(f"{fname}: mapped={summary[fam][year]['mapped_pixels']} "
                  f"nodata={summary[fam][year]['nodata_pixels']} "
                  f"counts={summary[fam][year]['class_counts_out']}", flush=True)
    C.save_json(summary, os.path.join(C.ROOT, "outputs", "predictions",
                                      "prediction_manifest.json"))
    print("manifest -> outputs/predictions/prediction_manifest.json")


if __name__ == "__main__":
    main()
