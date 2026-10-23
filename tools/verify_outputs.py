#!/usr/bin/env python3
"""Verify every generated raster: CRS, size, transform, values, NoData, coverage.

Checks required by the project brief section 10:
  - CRS == EPSG:4326, dimensions 693x669, geotransform identical to Member 1's
    aligned label grid (spatial alignment preserved)
  - prediction values only in {0,1,2,3,4} (brief's output scheme)
  - 2025 predictions: grid rows 543-668 must be NoData (0)
  - change rasters: values in {0..16}
Prints PASS/FAIL per file; exit code 1 on any failure.
"""
import glob
import os
import sys

import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "extracted", "Vijayawada_Training_Data",
                   "Labels_2015_Aligned.tif")
EXPECTED = (693, 669, "EPSG:4326")
TRANSFORM_TOL = 1e-12


def main():
    failures = []
    with rasterio.open(REF) as ref:
        ref_transform = tuple(ref.transform)[:6]

    files = sorted(glob.glob(os.path.join(ROOT, "outputs", "predictions", "*.tif")))
    files += sorted(glob.glob(os.path.join(ROOT, "outputs", "change_detection",
                                           "*", "*.tif")))
    for p in files:
        rel = os.path.relpath(p, ROOT)
        problems = []
        with rasterio.open(p) as s:
            if (s.width, s.height) != (EXPECTED[0], EXPECTED[1]):
                problems.append(f"size {s.width}x{s.height}")
            if str(s.crs) != EXPECTED[2]:
                problems.append(f"crs {s.crs}")
            if max(abs(a - b) for a, b in zip(tuple(s.transform)[:6],
                                              ref_transform)) > TRANSFORM_TOL:
                problems.append("transform differs from label grid")
            arr = s.read(1)
        is_pred = "predictions" in rel
        if is_pred:
            vals = set(np.unique(arr).tolist())
            if not vals <= {0, 1, 2, 3, 4}:
                problems.append(f"values {sorted(vals)}")
            if "pred_2025_" in rel:
                south = arr[543:, :]
                if np.any(south != 0):
                    problems.append(f"2025 south strip not NoData "
                                    f"({int((south != 0).sum())} px)")
            if (arr > 0).sum() == 0:
                problems.append("fully empty")
        else:
            vals = set(np.unique(arr).tolist())
            if not vals <= set(range(17)):
                problems.append(f"values {sorted(vals)}")
        status = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        print(f"  [{status}] {rel}")
        if problems:
            failures.append(rel)
    print(f"\n{len(files) - len(failures)}/{len(files)} rasters PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
