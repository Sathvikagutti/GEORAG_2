#!/usr/bin/env python3
"""Build the shared 9-feature stacks, label arrays and valid masks for 2015/2020/2025.

Reads ONLY the preserved copies under drive_data/ and extracted/ (never writes there).
Outputs: derived/{year}_features.npy, derived/{year}_labels.npy, derived/manifest.json

Grid: 693 x 669, 30 m, EPSG:4326, origin (80.55489596743509, 16.596285042579748)
"""
import json
import os

import numpy as np
import rasterio

W, H = 693, 669
TRANSFORM = (0.00026949458523585647, 0.0, 80.55489596743509,
             0.0, -0.00026949458523585647, 16.596285042579748)
CRS = "EPSG:4326"
YEARS = (2015, 2020, 2025)
FEATURES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDBI", "NDWI"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVED = os.path.join(ROOT, "derived")


def indices(blue, green, red, nir, swir1):
    """Member 1's index formulas, verified against the Drive Index/ML-Features rasters."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndbi = (swir1 - nir) / (swir1 + nir)
        ndwi = (green - nir) / (green + nir)
    return ndvi, ndbi, ndwi


def load_year(year):
    if year in (2015, 2020):
        path = os.path.join(ROOT, "drive_data", "satellite", f"Vijayawada_Landsat_{year}.tif")
        with rasterio.open(path) as src:
            assert (src.width, src.height) == (W, H), (path, src.width, src.height)
            raw = src.read().astype(np.float64)
    else:  # 2025: window read from the larger scene (col 19, row 200), boundless
        path = os.path.join(ROOT, "drive_data", "satellite", "2025")
        with rasterio.open(path) as src:
            from rasterio.windows import Window
            win = Window(18.99999999995648, 200.00000000001162, 692.9999999999862, 668.9999999999975)
            raw = src.read(window=win, boundless=True, fill_value=np.nan).astype(np.float64)
            assert raw.shape[1:] == (H, W), raw.shape
    blue, green, red, nir, swir1, swir2 = raw[:6]
    ndvi, ndbi, ndwi = indices(blue, green, red, nir, swir1)
    feats = np.stack([blue, green, red, nir, swir1, swir2, ndvi, ndbi, ndwi]).astype(np.float32)

    lab_path = os.path.join(ROOT, "extracted", "Vijayawada_Training_Data",
                            f"Labels_{year}_Aligned.tif")
    with rasterio.open(lab_path) as src:
        assert (src.width, src.height) == (W, H), (lab_path, src.width, src.height)
        assert tuple(src.transform)[:6] == TRANSFORM, (lab_path, tuple(src.transform))
        labels = src.read(1)
    return feats, labels


def main():
    os.makedirs(DERIVED, exist_ok=True)
    manifest = {"grid": {"width": W, "height": H, "crs": CRS, "transform": TRANSFORM,
                         "pixel_size_deg": TRANSFORM[0]},
                "features": FEATURES, "label_scheme_native": {
                    "0": "background/unlabeled", "1": "Water", "2": "Urban",
                    "3": "Soil", "4": "Vegetation"},
                "output_scheme_required": {
                    "1": "Built-up (native 2 Urban)", "2": "Vegetation (native 4)",
                    "3": "Water (native 1)", "4": "Open Land (native 3)",
                    "0": "NoData (native 0)"},
                "years": {}}
    for year in YEARS:
        feats, labels = load_year(year)
        finite = np.all(np.isfinite(feats), axis=0)
        valid = finite & np.isin(labels, [1, 2, 3, 4])
        np.save(os.path.join(DERIVED, f"{year}_features.npy"), feats)
        np.save(os.path.join(DERIVED, f"{year}_labels.npy"), labels)
        counts = {str(int(v)): int(c) for v, c in
                  zip(*np.unique(labels[valid], return_counts=True))}
        manifest["years"][str(year)] = {
            "valid_pixels": int(valid.sum()),
            "invalid_pixels": int((~valid).sum()),
            "invalid_reason_nonfinite_features": int((~finite).sum()),
            "class_counts_valid": counts,
        }
        print(f"{year}: valid={valid.sum()} invalid={(~valid).sum()} classes={counts}")
    with open(os.path.join(DERIVED, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest written ->", os.path.join(DERIVED, "manifest.json"))


if __name__ == "__main__":
    main()
