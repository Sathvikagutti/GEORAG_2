# GeoRAG Member 2 — Known Limitations and Unresolved Issues

Recorded 2026-10-22 and kept honest: nothing here is "explained away", and none of it
is presented as invalidating the whole project.

## 1. Validation design

- **Random pixel splits are not spatial validation.** E0 uses a stratified random pixel
  split (60/10/30, seed 42); temporal experiments hold out entire years. Neither
  guarantees spatial independence between train and test pixels within a year.
  The spatially independent (blocked) validation experiment **was run** and is in
  `results/spatial_validation.*` (never merged into the main table): 120×120-px
  blocks, role `(block_row+block_col)%3` (1=train, 2=val, 0=test), 2015, seed 42.
  Results: RF .8417/.7800, XGB .8401/.7793, dla .8775/.7911,
  mts .8921/.8102 (accuracy/macro-F1). Order of models is unchanged, but
  transformer F1 drops ~4–6 points vs the random split (dla .8137→.7911,
  mts .8512→.8102) — that gap is the spatial-correlation optimism of the main
  table, and blocked numbers are the honest ones for wall-to-wall mapping.
  The blocked split trains on only ~26% of the grid pixels, so it is also a
  smaller-training-set comparison; it is one split at one block size, not a
  multi-fold estimate.
- Adjacent pixels are strongly correlated (30 m grid, one district), so same-year
  accuracies are optimistic estimates of real-world generalisation. The temporal
  experiments (train one year → test another) are the more meaningful ones.

## 2. Labels

- **Unverified label provenance.** Member 1's own README states the labels "have not
  been independently verified as ground truth". Everything measured here is agreement
  with those labels, not with field truth.
- **Sampling design unknown.** How the training pixels/labels were originally sampled
  or interpreted is not documented anywhere in the Drive.
- **Class-numbering conflict.** The brief requires outputs numbered 1=Built-up,
  2=Vegetation, 3=Water, 4=Open Land, while Member 1's labels are 1=Water, 2=Urban,
  3=Soil, 4=Vegetation. Training/evaluation use Member 1's native ids unchanged; new
  prediction rasters are written in the brief's numbering (fixed bijection, recorded in
  README/INDEX/metadata tags). This choice was **not confirmed by a human** — the
  clarifying questions were skipped. Reversal is a one-line remap.

## 3. Data availability across years

- **Different observation counts per year:** valid labeled pixels are 2015: 348,549;
  2020: 344,072; 2025: 314,773. Class balance also shifts (Vegetation is the smallest
  class every year: 9,666 → 13,527 → 11,983).
- **2025 imagery window limitation:** the 2025 scene covers only grid rows 0–542;
  rows 543–668 (87,318 px, 18.8 % of the grid) have no 2025 imagery. Those pixels are
  excluded from 2025 evaluation and are NoData in 2025 outputs and in the
  2020→2025 change detection.
- **Two inconsistent 2025 feature products exist in the Drive** (verified):
  `GeoRAG_Indices/Vijayawada_Indices_2025.tif` (Sep 20; 199,388 NaN pixels incl.
  71,717 Water-labeled pixels; source 6-band scene not in the Drive) vs
  `GeoRAG_Satellite/2025` + `Vijayawada_2025_ML_Features.tif` (Oct 7; exact match to
  each other; mean |ΔNDVI| = 0.099 against the older product where both are finite).
  We used the newer scene for 2025 features (design rationale §4). Consequence: our
  2025 evaluation set (314,773 px) differs from Member 1's CSV 2025 rows (263,513).
  2015/2020 valid-pixel sets match the CSV **exactly**.

## 4. Missing previous Member 2 artifacts

- The Google Drive folder contains **no** Member 2 code, trained models, prediction
  rasters, transition matrices, area statistics, figures, reports, handoff documents,
  and **none of the `training_YYYY.csv` files** named in the brief. Only Member 1
  datasets + two ZIPs are present (verified by full recursive listing).
- Therefore: no comparison against *previous* results is possible; RF/XGBoost were
  **re-trained here under the shared protocol** and are labelled
  `RandomForest/XGBoost (retrained baseline)` everywhere.

## 5. Architectures

- Neither proposed model has a public official implementation (MTS: no repo found for
  Voelsen et al. 2024; DLA lightweight ViT: Bansal & Tripathi 2024 is paywalled, no
  code found). The implemented models are **documented adaptations**, never claimed to
  be the published models (`architecture_source` field in every result JSON).
- True multi-temporal sequences are only leakage-free in experiment E3 (see
  MODEL_INVESTIGATION §1.2); in E1/E2 the history slot is masked. MTS's defining
  advantage therefore cannot be fully exercised with only three dates.
- Same-year comparisons between tabular (RF/XGB) and patch (transformer) models differ
  in input representation (per-pixel features vs 140×140 context windows). Splits,
  labels, metrics and seeds are identical; only the input representation differs.

## 6. Compute environment

- CPU-only host: 2 cores, ~3.9 GB RAM, **no GPU**. All training times in the results
  are CPU times and would change on other hardware.
- PyTorch 2.14.1+cpu's oneDNN path crashes with SIGFPE on this host's CPU
  (verified: Conv2d crashes with mkldnn on, works with it off) →
  `torch.backends.mkldnn.enabled = False` is set in `src/models.py`.
- Random Forest checkpoints (1.35 GB serialized each) were measured then discarded to
  keep the package shareable; they are reproducible exactly with seed 42.
- XGBoost early stopping never triggered (all runs used the full 400 trees).

## 7. Predictions vs evaluation

- Prediction mappers cover **all pixels with finite imagery**, including pixels whose
  label is 0/background — so mapped area (e.g. 2015: 361,732 px) is larger than the
  labeled evaluation area (348,549 px). Evaluation numbers only refer to labeled pixels.
- **Mapped footprints differ between years** (scene properties, not our choice):
  the 2015/2020 scenes are NoData outside the municipal boundary (circular footprint,
  101,885 / 118,729 px) while the 2025 scene covers a rectangle (its only gap is the
  southern 126 rows). Change detection always uses the intersection of both maps' valid
  footprints.
- Each year's map comes from that year's E0 (same-year) model, so change detection
  compares three independently trained same-year models. This is standard practice but
  means some map-to-map difference may come from model differences as well as real
  land-cover change.
