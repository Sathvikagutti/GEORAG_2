# GeoRAG — Member 2: Model Upgrade, Scientific Validation & Project Handoff

Final-year project: **GeoRAG — AI-Powered Geospatial Decision Support System for
Sustainable Land-Use and Urban Planning.**
This package covers Member 2's work: LULC classification, temporal validation,
satellite raster prediction, change/urban-growth analysis, evaluation, and the
technical handoff.

Work date: 2026-10-22. Source of truth: the team's Google Drive folder (`GeoRAG`),
inspected recursively (`docs/DRIVE_INVENTORY.md`). **The Drive was never modified**;
everything here works from preserved local copies under `drive_data/`,
`drive_downloads/` and `extracted/`.

---

## 1. What changed and why (short version)

1. The Drive contains **no Member 2 code, models or results at all** (no RF/XGBoost
   outputs, none of the `training_YYYY.csv` files, no `georag/` folder) — verified by
   a full recursive listing. So nothing from the previous workflow could be compared
   against directly, and RF/XGBoost were **re-trained here under one shared protocol**
   (clearly labelled *retrained baseline*).
2. Both teammate-proposed architectures were investigated with sources
   (`docs/MODEL_INVESTIGATION.md`):
   - **MTS** matches Voelsen et al. 2024 (PFG) — but no official code/weights exist.
   - **Dual-level attention lightweight ViT** matches Bansal & Tripathi 2024
     (*Computers & Geosciences*) — paywalled, no code found.
   Neither can be reproduced exactly, so two **clearly labelled adaptations** were
   implemented (`mts-lite-adapted`, `dla-lite-adapted`), never claimed to be the
   published models.
3. The data **does** support image-patch models (Case A): full 693×669 30 m grids for
   2015/2020/2025 with 6 bands + 3 verified indices and spatially complete labels.
   Features were rebuilt to match Member 1's preprocessing exactly (max |Δ| = 3e-8).
4. Six experiments were defined *before* training (`docs/EXPERIMENT_DESIGN.md`):
   E0 same-year (3×) and E1/E2/E3 temporal hold-outs, one seed (42), one metric
   function, identical splits for every model.

## 2. Headline results

All numbers below are measured; the full table (with train/inference times, sizes,
checkpoints, per-class metrics) is in `results/model_comparison.csv` /
`results/model_comparison.md`; aggregation in `results/recommendation.json`.

Test accuracy / macro-F1 (seed 42; **best in bold**; identical splits for all models):

| Experiment | RandomForest | XGBoost | dla-lite-adapted | **mts-lite-adapted** |
|---|---|---|---|---|
| E0 2015→2015 (same-year) | .8469 / .8076 | .8390 / .7951 | .8702 / .8137 | **.9024 / .8512** |
| E0 2020→2020 (same-year) | .8660 / .8558 | .8617 / .8510 | .8854 / .8711 | **.9273 / .9114** |
| E0 2025→2025 (same-year) | .8765 / .8736 | .8739 / .8697 | .9016 / .8877 | **.9306 / .9168** |
| E1 2015→2020 (temporal) | .8084 / .7717 | .8154 / .7879 | .8600 / .8401 | **.8891 / .8568** |
| E2 2015→2025 (temporal) | .7483 / .7359 | .7588 / .7412 | .7952 / .7861 | **.8382 / .8173** |
| E3 2015+2020→2025 (temporal) | .8095 / .7959 | .8008 / .7840 | .8657 / .8505 | **.8987 / .8782** |

Aggregates (unweighted mean over the 3 runs per group, `results/recommendation.json`):

| Model | mean same-year F1 | mean temporal F1 | mean temporal accuracy |
|---|---|---|---|
| RandomForest | .8457 | .7678 | .7888 |
| XGBoost | .8386 | .7710 | .7916 |
| dla-lite-adapted | .8575 | .8256 | .8403 |
| **mts-lite-adapted** | **.8932** | **.8508** | **.8753** |

**mts-lite-adapted wins every one of the 12 runs** (4 model families × 6
experiments), by 2–9 accuracy points over the retrained baselines. The E2→E3 jump
for every model (.8382→.8987 for mts; RF .7483→.8095) quantifies how much a second
date helps temporal generalisation. Checkpoint paths, times, sizes and per-class
tables are in `results/model_comparison.*`.

The **separate** spatially independent validation experiment (blocked hold-out on
2015, never mixed into the table above) is in `results/spatial_validation.*`:
RF .8417/.7800 · XGB .8401/.7793 · dla .8775/.7911 · **mts .8921/.8102**
(acc/macro-F1). Model ranking is unchanged; the 4–6 point F1 drop vs the random
split is the spatial-correlation optimism of the main table — quote blocked numbers
for real mapping quality, the main table for model-to-model comparison.

## 3. Class schemes — READ BEFORE USING ANY OUTPUT

Member 1's labels (training/evaluation use these, unchanged):

| Native id | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Member 1 meaning | Background | Water | Urban | Soil | Vegetation |

New prediction rasters are written in the **brief's required output scheme**:

| Output id | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Meaning | NoData | Built-up | Vegetation | Water | Open Land |
| Native source | 0 | 2 Urban | 4 Vegetation | 1 Water | 3 Soil |

Mapping is a fixed bijection, stored in every raster's metadata tags, in
`outputs/predictions/prediction_manifest.json`, and applied in
`src/05_predict_rasters.py`. Metrics are computed in native ids (consistent either
way). This numbering choice was documented but not human-confirmed
(see `docs/LIMITATIONS.md` §2).

## 4. Contents map

`INDEX.md` is the top-level index; `MANIFEST.csv` lists every packaged file with
size and SHA-256. In brief: `src/` pipeline code · `docs/` design, inventory, model
investigation, limitations, handoff · `experiments/` raw run outputs (baselines +
transformers, JSON + checkpoints) · `results/` comparison tables · `outputs/`
prediction rasters, change detection, areas, figures · `drive_data/` + `drive_downloads/`
unchanged Member 1 inputs.

## 5. Reproduce from scratch

```bash
# 0. environment (Python 3.12)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. restore Member 1 extraction (excluded from the ZIP because it is a ZIP already)
mkdir -p extracted
unzip -o -q drive_downloads/Vijayawada_Training_Data.zip -d extracted/
unzip -o -q drive_downloads/pixel_inspection_results.zip -d extracted/

# 2. build the shared 9-feature stacks + labels + masks (writes derived/)
.venv/bin/python src/01_build_features.py

# 3. baselines: RF + XGBoost for all six experiments (writes experiments/baselines/)
.venv/bin/python src/02_run_baselines.py

# 4. transformers: 6 experiments x 2 models (writes experiments/transformers/)
./run_all_transformers.sh            # or call src/03_train_transformer.py per run

# 5. optional separate spatial validation
.venv/bin/python src/03_train_transformer.py --model mts --experiment E0_2015 --spatial
.venv/bin/python src/03_train_transformer.py --model dla --experiment E0_2015 --spatial
.venv/bin/python src/08_spatial_validation.py

# 6. collect results + tables
.venv/bin/python src/04_collect_results.py

# 7. predictions -> change detection -> figures
.venv/bin/python src/05_predict_rasters.py --models mts dla xgboost random_forest
.venv/bin/python src/06_change_analysis.py --model mts-lite-adapted
.venv/bin/python src/06_change_analysis.py --model dla-lite-adapted
.venv/bin/python src/06_change_analysis.py --model xgboost
.venv/bin/python src/06_change_analysis.py --model random_forest
.venv/bin/python src/07_figures.py

# 8. verify every raster, then rebuild this package
.venv/bin/python tools/verify_outputs.py
.venv/bin/python tools/build_package.py
```

Everything is seeded (42). Training was done CPU-only: 2 cores, no GPU
(`results/environment.json`). Note `src/models.py` disables PyTorch's oneDNN because
this host's build SIGFPEs without it.

## 6. Protocol guarantees (details in `docs/EXPERIMENT_DESIGN.md`)

- Test-year labels never touched training/tuning/model selection; scalers fit on
  training pixels only; one shared `evaluate()` for all models.
- Random pixel splits are labelled `stratified_random_pixel_split` — **not** spatial
  validation; the separate blocked experiment is labelled
  `spatially_independent_block_holdout`.
- Temporal input sequences (MTS history slot) are only used in E3, the one
  leakage-free construction available with three dates.
- Predictions are only written after evaluation; original rasters are never
  overwritten; new outputs live under `outputs/` with model names in filenames.

## 7. Limitations (full list: `docs/LIMITATIONS.md`)

Label provenance unverified (Member 1's own note) · sampling design undocumented ·
2025 southern strip has no imagery (87,318 px) and two inconsistent 2025 index
products exist (we used the newer scene; documented) · observation counts differ per
year · no GPU · adaptations are not the published architectures · previous Member 2
results absent from the Drive.
