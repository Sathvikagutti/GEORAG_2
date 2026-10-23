#!/bin/bash
# Final compute driver: runs everything remaining after the 12-run matrix.
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== [0] waiting for E3 mts metrics ==="
for i in $(seq 1 120); do
  [ -f experiments/transformers/E3_2015_2020_to_2025_mts_seed42.json ] && break
  sleep 15
done
[ -f experiments/transformers/E3_2015_2020_to_2025_mts_seed42.json ] \
  && echo "E3 mts metrics present" || { echo "E3 mts metrics MISSING - aborting"; exit 1; }

echo "=== [1] spatial transformer runs (dla, mts on E0_2015) $(date +%H:%M:%S) ==="
$PY src/03_train_transformer.py --model dla --experiment E0_2015 --spatial \
  >> logs/spatial.log 2>&1 || echo "dla spatial FAILED rc=$?"
$PY src/03_train_transformer.py --model mts --experiment E0_2015 --spatial \
  >> logs/spatial.log 2>&1 || echo "mts spatial FAILED rc=$?"

echo "=== [2] spatial validation (RF+XGB+collect) $(date +%H:%M:%S) ==="
$PY src/08_spatial_validation.py >> logs/spatial.log 2>&1 || echo "08 FAILED rc=$?"

echo "=== [3] full results collection $(date +%H:%M:%S) ==="
$PY src/04_collect_results.py || echo "04 FAILED rc=$?"

echo "=== [4] prediction rasters (all 4 families) $(date +%H:%M:%S) ==="
$PY src/05_predict_rasters.py --models random_forest xgboost dla mts \
  >> logs/predict.log 2>&1 || echo "05 FAILED rc=$?"

echo "=== [5] change analysis per family $(date +%H:%M:%S) ==="
for tag in random_forest xgboost dla-lite-adapted mts-lite-adapted; do
  $PY src/06_change_analysis.py --model "$tag" >> logs/change.log 2>&1 \
    || echo "06 $tag FAILED rc=$?"
done

echo "=== [6] figures $(date +%H:%M:%S) ==="
$PY src/07_figures.py >> logs/figures.log 2>&1 || echo "07 FAILED rc=$?"

echo "ALL_FINAL_STEPS_FINISHED $(date +%H:%M:%S)"
