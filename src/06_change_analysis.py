#!/usr/bin/env python3
"""Dependent outputs for one model family's prediction rasters:

- change-detection rasters (2015->2020, 2020->2025)
- 4x4 transition matrices (counts and geodesic area)
- class-wise area statistics per year
- built-up expansion tables (transitions INTO output class 1 = Built-up only)
- urban-growth summary json

Areas: CRS is geographic (EPSG:4326), so pixel areas are computed geodesically
(WGS84) per grid row and multiplied by per-row pixel counts.

Usage: python src/06_change_analysis.py --model mts-lite-adapted
       (name must match a file produced by 05_predict_rasters.py)
"""
import argparse
import json
import os
import sys

import numpy as np
import rasterio
from pyproj import Geod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PRED_DIR = os.path.join(C.ROOT, "outputs", "predictions")
OUT_CLASSES = (1, 2, 3, 4)
OUT_NAMES = {1: "Built-up", 2: "Vegetation", 3: "Water", 4: "Open Land"}
NATIVE_NAMES = {1: "Water", 2: "Urban", 3: "Soil", 4: "Vegetation"}
PAIRS = ((2015, 2020), (2020, 2025))
PS = 0.00026949458523585647
LON0 = 80.55489596743509
LAT_TOP = 16.596285042579748
LON1 = LON0 + PS  # ONE pixel width (was wrongly the full grid width -> areas x693)

GEOD = Geod(ellps="WGS84")


def row_areas_m2():
    """Geodesic area (m^2) of one grid pixel for each row of the analysis grid."""
    areas = np.empty(C.GRID_H)
    for j in range(C.GRID_H):
        lat_t = LAT_TOP - j * PS
        lat_b = LAT_TOP - (j + 1) * PS
        area, _ = GEOD.polygon_area_perimeter([LON0, LON1, LON1, LON0],
                                              [lat_b, lat_b, lat_t, lat_t])
        areas[j] = abs(area)
    return areas


def load_map(model_name, year):
    path = os.path.join(PRED_DIR, f"pred_{year}_{model_name}.tif")
    with rasterio.open(path) as src:
        assert (src.width, src.height) == (C.GRID_W, C.GRID_H)
        return src.read(1)


def area_stats(model_name, row_area):
    base = os.path.join(C.ROOT, "outputs", "areas")
    os.makedirs(base, exist_ok=True)
    out = {}
    for year in (2015, 2020, 2025):
        m = load_map(model_name, year)
        total_px = int((m > 0).sum())
        total_area = float((row_area[:, None] * (m > 0)).sum())
        rows = ["class_id_out,class_name_out,class_name_native,pixel_count,"
                "area_m2,area_ha,area_km2,pct_of_mapped"]
        per_class = {}
        for c in OUT_CLASSES:
            mask = (m == c)
            px = int(mask.sum())
            area_m2 = float((row_area[:, None] * mask).sum())
            pct = 100.0 * px / total_px if total_px else 0.0
            inv = {v: k for k, v in {1: 3, 2: 1, 3: 4, 4: 2}.items()}
            native = inv[c]
            rows.append(f"{c},{OUT_NAMES[c]},{NATIVE_NAMES[native]},{px},"
                        f"{area_m2:.2f},{area_m2 / 1e4:.4f},"
                        f"{area_m2 / 1e6:.6f},{pct:.3f}")
            per_class[str(c)] = {"pixel_count": px, "area_m2": area_m2,
                                 "area_ha": area_m2 / 1e4}
        path = os.path.join(base, f"{model_name}_area_{year}.csv")
        with open(path, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        out[str(year)] = {"total_mapped_pixels": total_px,
                          "total_mapped_area_m2": total_area,
                          "per_class": per_class,
                          "csv": os.path.relpath(path, C.ROOT)}
    return out


def change_pair(model_name, y1, y2, row_area):
    a = load_map(model_name, y1)
    b = load_map(model_name, y2)
    common = (a > 0) & (b > 0)
    # transition id 1..16: (from-1)*4 + (to-1) + 1 over OUT classes 1..4
    trans = np.zeros((C.GRID_H, C.GRID_W), dtype=np.uint8)
    trans[common] = (a[common] - 1).astype(np.uint16) * 4 + (b[common] - 1) + 1
    trans = trans.astype(np.uint8)

    outdir = os.path.join(C.ROOT, "outputs", "change_detection", model_name)
    os.makedirs(outdir, exist_ok=True)
    # change raster
    profile = {"driver": "GTiff", "height": C.GRID_H, "width": C.GRID_W,
               "count": 1, "dtype": "uint8", "crs": "EPSG:4326",
               "transform": rasterio.Affine(PS, 0.0, LON0, 0.0, -PS, LAT_TOP),
               "nodata": 0, "compress": "deflate"}
    legend = {f"{i:02d}": f"{OUT_NAMES[(i - 1) // 4 + 1]}->{OUT_NAMES[(i - 1) % 4 + 1]}"
              for i in range(1, 17)}
    cpath = os.path.join(outdir, f"change_{y1}_{y2}.tif")
    with rasterio.open(cpath, "w", **profile) as dst:
        dst.write(trans, 1)
        dst.set_band_description(1, f"transition {y1}->{y2}")
        dst.update_tags(legend="; ".join(f"{k}={v}" for k, v in legend.items()),
                        common_pixels=int(common.sum()),
                        model=model_name)

    # per-row transition counts -> matrices
    counts = np.zeros((C.GRID_H, 16), dtype=np.int64)
    for j in range(C.GRID_H):
        counts[j] = np.bincount(trans[j][trans[j] > 0], minlength=17)[1:17]
    area_mat = (counts * row_area[:, None]).sum(axis=0)      # m^2 per transition
    mat_counts = counts.sum(axis=0).reshape(4, 4)
    mat_area_ha = (area_mat / 1e4).reshape(4, 4)

    cpath_csv = os.path.join(outdir, f"transition_matrix_{y1}_{y2}_counts.csv")
    apath_csv = os.path.join(outdir, f"transition_matrix_{y1}_{y2}_area_ha.csv")
    with open(cpath_csv, "w") as fh:
        fh.write("from\\to," + ",".join(OUT_NAMES[c] for c in OUT_CLASSES) + "\n")
        for i, fc in enumerate(OUT_CLASSES):
            fh.write(f"{OUT_NAMES[fc]}," +
                     ",".join(str(int(mat_counts[i, k])) for k in range(4)) + "\n")
    with open(apath_csv, "w") as fh:
        fh.write("from\\to," + ",".join(OUT_NAMES[c] for c in OUT_CLASSES) + "\n")
        for i, fc in enumerate(OUT_CLASSES):
            fh.write(f"{OUT_NAMES[fc]}," +
                     ",".join(f"{mat_area_ha[i, k]:.4f}" for k in range(4)) + "\n")

    # built-up expansion: transitions INTO output class 1 (Built-up) from others
    bu_idx = 0  # class 1 is first column/row block
    bpath = os.path.join(outdir, f"builtup_expansion_{y1}_{y2}.csv")
    rows = ["source_class_out,source_class_native,pixels,area_ha,note"]
    gain_px, gain_ha = 0, 0.0
    for src_i, sc in enumerate(OUT_CLASSES):
        if sc == 1:
            continue
        px = int(mat_counts[src_i, bu_idx])
        ha = float(mat_area_ha[src_i, bu_idx])
        # src_i is an OUTPUT class id: 1 Built-up, 2 Vegetation, 3 Water, 4 Open Land
        native_src = {2: "Vegetation", 3: "Water", 4: "Soil"}[sc]
        rows.append(f"{sc},{native_src},{px},{ha:.4f},into Built-up (class 1)")
        gain_px += px
        gain_ha += ha
    stable_px = int(mat_counts[bu_idx, bu_idx])
    stable_ha = float(mat_area_ha[bu_idx, bu_idx])
    loss_px = int(mat_counts[bu_idx, :].sum() - stable_px)
    loss_ha = float(mat_area_ha[bu_idx, :].sum() - stable_ha)
    rows.append(f"1,stable_Built-up,{stable_px},{stable_ha:.4f},no change")
    with open(bpath, "w") as fh:
        fh.write("\n".join(rows) + "\n")

    summary = {
        "model": model_name, "pair": f"{y1}->{y2}",
        "common_pixels": int(common.sum()),
        "changed_pixels": int(((trans > 0) & ~np.isin(trans, [1, 6, 11, 16])).sum()),
        "builtup": {
            "gain_pixels": gain_px, "gain_area_ha": gain_ha,
            "loss_pixels": loss_px, "loss_area_ha": loss_ha,
            "net_change_ha": gain_ha - loss_ha,
            "stable_pixels": stable_px, "stable_area_ha": stable_ha,
        },
        "transition_counts_matrix_rows_from_cols_to": mat_counts.tolist(),
        "transition_area_ha_matrix": np.round(mat_area_ha, 4).tolist(),
        "files": {"change_raster": os.path.relpath(cpath, C.ROOT),
                  "counts_csv": os.path.relpath(cpath_csv, C.ROOT),
                  "area_csv": os.path.relpath(apath_csv, C.ROOT),
                  "builtup_csv": os.path.relpath(bpath, C.ROOT)},
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="prediction file tag, e.g. mts-lite-adapted, xgboost")
    a = ap.parse_args()
    row_area = row_areas_m2()
    print(f"pixel geodesic area: {row_area[0]:.2f} m2 (row 0) .. "
          f"{row_area[-1]:.2f} m2 (row {C.GRID_H - 1})")

    areas = area_stats(a.model, row_area)
    pairs = [change_pair(a.model, y1, y2, row_area) for y1, y2 in PAIRS]

    # urban growth roll-up across both intervals
    growth = {"model": a.model,
              "note": "urban growth = transitions INTO output class 1 (Built-up) "
                      "only; other transitions are land-cover change, not growth",
              "area_method": "geodesic WGS84 pixel area per grid row x counts",
              "intervals": {}}
    for p in pairs:
        counts = np.array(p["transition_counts_matrix_rows_from_cols_to"])
        area_ha = np.array(p["transition_area_ha_matrix"])
        breakdown = {}
        for src_i, sc in enumerate(OUT_CLASSES):
            if sc == 1:
                continue
            breakdown[OUT_NAMES[sc]] = {
                "pixels_into_builtup": int(counts[src_i, 0]),
                "area_ha_into_builtup": round(float(area_ha[src_i, 0]), 4)}
        g = dict(p["builtup"])
        g["source_breakdown"] = breakdown
        growth["intervals"][p["pair"]] = g
    out = {"areas": areas, "change_pairs": pairs, "urban_growth": growth}
    C.save_json(out, os.path.join(C.ROOT, "outputs", "change_detection",
                                  f"{model_safe(a.model)}_analysis.json"))
    print(json.dumps({p["pair"]: p["builtup"] for p in pairs}, indent=2))


def model_safe(name):
    return name.replace("/", "_")


if __name__ == "__main__":
    main()
