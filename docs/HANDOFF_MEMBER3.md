# GeoRAG — Technical Handoff for Member 3 (GIS / RAG Decision-Support Stage)

Date: 2026-10-22. From: Member 2 (model upgrade & validation).
Purpose: everything Member 3 needs to consume the LULC maps, change analysis and
area statistics in a GIS or RAG pipeline — without reading Member 2's code.

---

## 1. What you get (the products)

All under `outputs/`, all generated from the **recommended model**
(`mts-lite-adapted`; rationale in `results/recommendation.json` and §3 here),
with `dla-lite-adapted`, `xgboost` and `random_forest` variants shipped alongside
for comparison — every metric quoted in this document is held-out measured
performance, not training fit.

| Product | Path | Format |
|---|---|---|
| LULC maps 2015 / 2020 / 2025 | `outputs/predictions/pred_<year>_<model>.tif` | GeoTIFF, 1 band, Byte, EPSG:4326, 693×669 (2025: 743×743 scene grid) |
| Prediction inventory | `outputs/predictions/prediction_manifest.json` | JSON: per-file class mapping, mapped-pixel counts, inference times |
| Transition matrices + built-up change | `outputs/change_detection/<model>/` + `<model>_analysis.json` | JSON + CSV |
| Class area statistics per year | `outputs/areas/<model>_area_<year>.csv` | CSV: class, pixels, geodesic m², km², share |
| Figures (maps, confusion matrices, comparisons) | `outputs/figures/*.png` | PNG |

## 2. Class schemes — critical

Member 1's **label** ids (used for training and every metric in this package):

| Native id | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Meaning | Background | Water | Urban | Soil | Vegetation |

The **prediction rasters** are written in the project brief's required *output*
scheme, as a fixed bijection of the native ids:

| Output id | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Meaning | NoData | Built-up | Vegetation | Water | Open Land |
| Comes from native | 0 | 2 Urban | 4 Vegetation | 1 Water | 3 Soil |

The mapping is stored in each raster's GeoTIFF metadata tags and in
`prediction_manifest.json`. Any GIS styling or RAG query should use the output
scheme table; never mix the two. (This numbering was documented but not
human-confirmed — see `docs/LIMITATIONS.md` §2; changing it later is a fixed
remap, not a retrain.)

## 3. Model performance you can quote

Same-year (stratified random pixel split, seed 42) — test accuracy / macro-F1:

| Year | RF | XGB | dla-lite-adapted | **mts-lite-adapted** |
|---|---|---|---|---|
| 2015 | .8469 / .8076 | .8390 / .7951 | .8702 / .8137 | **.9024 / .8512** |
| 2020 | .8660 / .8558 | .8617 / .8510 | .8854 / .8711 | **.9273 / .9114** |
| 2025 | .8765 / .8736 | .8739 / .8697 | .9016 / .8877 | **.9306 / .9168** |

Temporal generalisation (train on earlier year(s), test on held-out year):

| Experiment | RF | XGB | dla-lite-adapted | **mts-lite-adapted** |
|---|---|---|---|---|
| 2015 → 2020 | .8084 / .7717 | .8154 / .7879 | .8600 / .8401 | **.8891 / .8568** |
| 2015 → 2025 | .7483 / .7359 | .7588 / .7412 | .7952 / .7861 | **.8382 / .8173** |
| 2015+2020 → 2025 | .8095 / .7959 | .8008 / .7840 | .8657 / .8505 | **.8987 / .8782** |

`mts-lite-adapted` wins every experiment, by 2–9 accuracy points over the
retrained baselines. Its E3 result (train on 2015+2020, test on 2025,
acc .8987) is the strongest evidence that its temporal history input improves
generalisation to unseen dates — the E2 vs E3 gap for all models (e.g. mts
.8382 → .8987) shows how much a second training date helps.

A separate **spatially independent** experiment (blocked hold-out, 2015; train
and test pixels in different 120×120-pixel blocks) is in
`results/spatial_validation.*`: RF .8417/.7800, XGB .8401/.7793, dla .8775/.7911,
mts .8921/.8102 (accuracy/macro-F1). Numbers there are systematically lower than
random-split numbers — that is expected and is the more honest estimate for
wall-to-wall mapping. Quote it when asked "how good is the map really", quote
the main table for model-to-model comparison.

## 4. Loading in Python / GIS

```python
import rasterio, numpy as np
with rasterio.open("outputs/predictions/pred_2025_mts-lite-adapted.tif") as src:
    a = src.read(1)                      # 0 = NoData, 1..4 = output classes
    mapping = src.tags().get("class_mapping", "see prediction_manifest.json")
    mask = a > 0                         # mapped district area only
```
QGIS: drag the `.tif` in; values 1–4 with the palette from §2.
CRS is EPSG:4326 (Member 1's grid); pixel size ~0.000278° (~30 m);
geodesic pixel area varies by row (857.67 m² at the district's latitude —
use `outputs/areas/*.csv` for areas, don't multiply by a constant).

## 5. Change / urban-growth products

- `outputs/change_detection/<model>/transition_<y1>_<y2>.csv` — full 4×4
  transition matrix (rows = y1 class, cols = y2 class, pixels and km²).
- `<model>_analysis.json` — roll-up: transitions **into output class 1
  (Built-up)** = urban growth, per interval, in km²; plus per-class gains/losses.
- Only transitions into Built-up are "growth"; the rest is land-cover change.
- Caveats: 2015/2020 district-boundary masking differs from the rectangular
  2025 footprint, and the southern 126 rows of the 2015 grid have no 2025
  imagery (87,318 px). Change products restrict to the common valid area;
  see `docs/LIMITATIONS.md` §4–5.

## 6. What the RAG layer can safely assert

Safe: class areas and shares per year (from `outputs/areas/`), transition counts
and urban-growth km² per interval (from `outputs/change_detection/`), and model
accuracy figures from §3 of this document.
Not safe: pixel-level claims about individual locations (models err most on
Soil↔Urban confusion), causal explanations of change, or any statement about
years other than 2015/2020/2025. Attribute products to "GeoRAG LULC model
(mts-lite-adapted, held-out accuracy 0.93 for 2025)" rather than to Member 1's
labels.

## 7. Retraining / updating with new data

New year (new label raster + scene): add the scene to the input folder, extend
`src/01_build_features.py`'s year table, run `02`+`03` (or ask Member 2's
scripts via README §5), then `04`→`07`. New classes require changing the head,
the class-weight logic and both scheme tables — do not do this silently.
All randomness is seeded (42); CPU-only training of one run is ~6–8 minutes.
