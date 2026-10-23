# GeoRAG — Member 2 Experiment Design (written BEFORE any training)

Date: 2026-10-22
Status: this document fixes the experimental protocol. Any deviation recorded later
must be listed in section 12 (Deviations) with the reason.

---

## 1. Purpose

Investigate whether the two architectures suggested by the team —
(A) Multi-Temporal Swin Transformer (MTS) and
(B) Dual-Level Attention Lightweight Vision Transformer —
are suitable for the GeoRAG Member 2 workflow, and if so, whether they improve on the
tabular Random Forest / XGBoost approach, under one shared, reproducible protocol.

## 2. Verified data sources (Google Drive folder, inspected 2026-10-22)

| Item | Drive path | Grid |
|---|---|---|
| Landsat 2015 (7 bands) | `GeoRAG_Satellite/Vijayawada_Landsat_2015.tif` | 693x669, 30 m, EPSG:4326 |
| Landsat 2020 (7 bands) | `GeoRAG_Satellite/Vijayawada_Landsat_2020.tif` | 693x669, 30 m, EPSG:4326 |
| Landsat 2025 scene (7 bands) | `GeoRAG_Satellite/2025` | 743x743, 30 m, EPSG:4326 (different origin) |
| 2025 ML features (10 bands) | `GeoRAG_Satellite/Vijayawada_2025_ML_Features.tif` | 743x743 (same grid as `2025`) |
| Indices 2015/2020/2025 | `GeoRAG_Indices/Vijayawada_Indices_YYYY.tif` | 693x669, EPSG:4326 |
| Labels 2015/2020/2025 (10 m) | `GeoRAG_LandUse_Labels/Vijayawada_LandUse_Labels_YYYY.tif` | 2077x2004, EPSG:4326 |
| Aligned labels 30 m | `Vijayawada_Training_Data.zip/Labels_YYYY_Aligned.tif` | 693x669 (SAME grid as Landsat/Indices) |
| Training samples CSV | `Vijayawada_Training_Data.zip/Training_Samples.csv` | 956,134 rows, columns: year,row,col,lon,lat,NDVI,NDBI,NDWI,class_id,class_name |

**Shared analysis grid:** 693 x 669 pixels, 30 m, EPSG:4326,
origin (80.55489596743509, 16.596285042579748). The aligned label rasters already sit
on this grid, so every model uses exactly the same pixels, labels, and evaluation set.

**2025 coverage limitation (verified):** the 2025 scene starts at row 200 / col 19 of
that grid, so grid rows 543-668 (126 rows = 87,318 pixels = 18.8 % of the grid) have no
2025 imagery. Those pixels are excluded from 2025 training/evaluation and will be
written as NoData in 2025 outputs.

## 3. Working-copy and preservation policy

- The Google Drive folder is the authoritative source and is NOT modified.
- Everything downloaded here is a copy: `drive_downloads/` (original ZIPs),
  `drive_data/` (original GeoTIFFs), `extracted/` (ZIP contents). These are read-only
  inputs; no script writes into them.
- New code goes to `src/`, new results to `experiments/` and `outputs/`.
- No existing output can be overwritten because the Drive contains no Member 2 outputs
  (verified: no Python code, no models, no prediction rasters, no metrics were present).

## 4. Features (9 per pixel, identical for every model)

Reflectance bands (band order as stored, verified descriptions): Blue, Green, Red, NIR,
SWIR1, SWIR2 — **thermal band excluded** (it is not in the 9-feature specification).
Spectral indices: NDVI = (NIR-Red)/(NIR+Red), NDBI = (SWIR1-NIR)/(SWIR1+NIR),
NDWI = (Green-NIR)/(Green+NIR).

Verification performed: these formulas reproduce Member 1's `Vijayawada_Indices_*.tif`
and `Vijayawada_2025_ML_Features.tif` bands with max abs difference 3e-8 (float32
rounding) and identical NaN masks. Member 1's preprocessing is therefore reproduced,
not changed.

Per-year construction:
- 2015: bands from `Vijayawada_Landsat_2015.tif` + computed indices.
- 2020: bands from `Vijayawada_Landsat_2020.tif` + computed indices.
- 2025: window (col 19, row 200, 693x669) from `GeoRAG_Satellite/2025` + computed
  indices (cross-checked against `Vijayawada_2025_ML_Features.tif`, exact match).
  **Why not `Vijayawada_Indices_2025.tif`?** Two inconsistent 2025 products exist in the
  Drive. The older Sep-20 Index raster has 199,388 NaN pixels (43 % of the grid,
  including 71,717 Water-labelled pixels) and no matching 6-band scene in the Drive;
  where it and the newer Oct-7 scene are both finite they disagree strongly
  (mean |dNDVI| = 0.099, max 0.93). The newer scene reproduces the Oct-7
  `Vijayawada_2025_ML_Features.tif` exactly and uses the same 6-bands-plus-3-indices
  construction as 2015/2020, so it is used for all 2025 features. Consequence: our 2025
  evaluation pixel set (314,773) differs from Member 1's CSV 2025 rows (263,513, which
  came from the older index raster); 2015/2020 valid pixel sets match the CSV exactly.

`Year` and `Label` are never predictors. `Year` is kept as metadata only.

## 5. Labels and class numbering (important, read before using outputs)

Member 1's actual label rasters and CSV use this scheme (verified three ways: ZIP README,
CSV `class_name` column, raster class counts):

| Member 1 native ID | Meaning | Brief's required output ID |
|---|---|---|
| 1 | Water | 3 |
| 2 | Urban | 1 (Built-up) |
| 3 | Soil | 4 (Open Land) |
| 4 | Vegetation | 2 |
| 0 | Background / unlabeled | 0 (NoData) |

- Training and evaluation always use Member 1's NATIVE ids against Member 1's own
  labels. Originals are never modified or renumbered.
- NEW prediction rasters are written in the brief's required numbering
  (1 = Built-up, 2 = Vegetation, 3 = Water, 4 = Open Land) via the fixed bijection
  above, because the brief defines that as the valid class set for predictions.
  Every raster carries `class_scheme` metadata tags and the mapping is repeated in
  README/INDEX/handoff docs. Metrics are unaffected by a consistent relabeling.
- This numbering decision was not confirmed by the user (questions were skipped); it is
  reversible with the documented one-line remap and is listed in Deviations (section 12).

## 6. Valid-pixel masks

A pixel is valid for a year when: all 9 features are finite AND the aligned label for
that year is in {1,2,3,4}. Counts are written to `derived/manifest.json` by
`src/01_build_features.py`. No pixel is used in more than one split role within an
experiment.

## 7. Experiments

Seeds: 42 everywhere (python, numpy, torch, split shuffling).

- **E0 same-year (per year 2015/2020/2025):** stratified random pixel split of that
  year's valid pixels: 60 % train / 10 % val / 30 % test, stratified by class,
  seed 42. This is a *random pixel split* — explicitly NOT spatial validation.
  Val is used for early stopping / checkpoint selection only.
- **E1:** train 2015 (60/10 split of 2015 valid pixels), test on ALL valid 2020 pixels.
- **E2:** train 2015 (same split), test on ALL valid 2025 pixels.
- **E3:** train pooled 2015 + 2020 valid pixels (60/10 split applied within the pool),
  test on ALL valid 2025 pixels.

Rules: test-year labels never enter training, tuning, early stopping, or model
selection. Scalers/normalization statistics are fit on the training split only and
stored with the checkpoint. Identical metrics code (one function) is used for all
models. Any input-representation difference between models is documented in section 12.

Optional experiment (only if time allows, reported separately, never merged):
spatially independent validation = checkerboard block split (one 120x120 block per
3x3 tiling held out) on E0-2015 for RF/XGBoost and the best transformer.

## 8. Models

Baselines (re-trained here because no previous results exist in the Drive — marked
`retrained-baseline`, never presented as "previous Member 2 results"):

- **RF**: RandomForestClassifier, n_estimators=300, class_weight balanced via sample
  weights, n_jobs=2, seed 42.
- **XGB**: XGBoost 3.4.1, 400 trees, lr 0.1, max_depth 6, subsample 0.8,
  colsample 0.8, seed 42, sample weights as RF.

New models (both are **adapted reimplementations — the published code could not be
found**, so neither may be called "the published model"; see `docs/MODEL_INVESTIGATION.md`):

- **mts-lite-adapted** (Model A direction): Swin-style shifted-window dense encoder on
  the 693x669 grid; 4x4 patch tokens, window attention 7x7 with alternating shift,
  width 64, 6 blocks, per-pixel head (4x4 outputs per token). Supports an optional
  second temporal slot (history, current) with a learned mask token — used only in E3,
  where a leakage-free sequence exists (train: [masked,2015] and [2015,2020];
  test: [2020,2025]). In E0/E1/E2 the history slot is masked for train and test.
- **dla-lite-adapted** (Model B direction): dual-level attention — token-level spatial
  attention in 7x7 windows + channel-level (cross-covariance) attention, width 64,
  6 blocks, same patch/head layout, single-date input in all experiments.

Shared transformer training config: AdamW, lr 3e-4, weight decay 0.01, cosine-to-constant
schedule, cross-entropy with inverse-frequency sqrt class weights, batch = four 128x128
random crops (flips allowed), max 25 epochs, early stopping patience 5 on validation
loss (validation = full-grid pass over that experiment's train-year val pixels),
best-checkpoint selection = lowest val loss. Standardization (mean/std per channel) fit
on training pixels only and saved next to the checkpoint.

## 9. Metrics and recorded costs (one shared `evaluate()` function)

Accuracy, Macro-P/R/F1, per-class P/R/F1, 4x4 confusion matrix, Cohen's Kappa,
train time (s), inference time for the full grid (s), model size (bytes).
Every metric is computed on actual predictions only; nothing is estimated.

## 10. Predictions and dependent outputs

Only after a model is trained AND evaluated:

- Write per-year class rasters for the models selected for mapping (planned: best
  transformer + best baseline), one file per model-year, filenames carry the model tag,
  e.g. `pred_2015_mts-lite-adapted.tif`. Existing rasters are never overwritten; files
  live under `outputs/predictions/`.
- Checks on every raster: width 693, height 669, CRS EPSG:4326, transform identical to
  the label grid, values only {0,1,2,3,4} in the brief's numbering, 2025 rows 543-668
  = NoData.
- Dependent outputs regenerated ONLY for new prediction rasters: change-detection rasters
  (2015->2020, 2020->2025), transition matrices, per-class area statistics, built-up
  expansion tables (transitions INTO Built-up only), urban-growth summary, model
  comparison table, figures. Nothing else in the project is touched.

## 11. Area calculations

CRS is geographic, so pixel areas are computed geodesically with pyproj (WGS84) per
grid row and multiplied by class pixel counts (defensible method for geographic
rasters; documented in the area tables). Units: m², ha, km².

## 12. Deviations (append during the work)

- (design) Class numbering of new outputs follows the brief's 1=Built-up scheme while
  training/evaluation use Member 1's native IDs (section 5). Not user-confirmed;
  documented and reversible.
- (data) 2025 features are built from the newer Oct-7 `GeoRAG_Satellite/2025` scene,
  not from `Vijayawada_Indices_2025.tif` that Member 1's 2025 CSV rows were extracted
  from (rationale and numbers in section 4). Member 1's files are unchanged; only our
  derived feature stack and hence our 2025 evaluation set differ.
- (scope) Prediction rasters + dependent outputs were generated for **all four**
  model families (RF, XGB, dla, mts) instead of only the single selected one: each
  raster is <1 MB, so teammates can compare maps directly; the *recommendation* still
  names one model for operational use.
- (validation) The optional spatially independent block experiment (design §7) was
  run for E0_2015 with RF, XGB and both transformers and is reported only in
  `results/spatial_validation.*`, never merged into the main table.
- (pending) none yet.
- (implementation) XGBoost is fit on labels remapped to 0-3 (its sklearn API requires
  0-based classes) and predictions are mapped back to native 1-4; metrics unaffected.
- (implementation) Random Forest checkpoints are measured for size and then discarded:
  a single 300-tree forest serializes to ~1.35 GB, and six of them would not fit the
  workspace or a shareable ZIP. The exact model is reproducible bit-for-bit by
  `src/02_run_baselines.py` with seed 42; recorded sizes are the true serialized sizes.
