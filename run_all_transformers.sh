#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python
for exp in E0_2015 E0_2020 E0_2025 E1_2015_to_2020 E2_2015_to_2025 E3_2015_2020_to_2025; do
  for model in dla mts; do
    echo "########## START $exp $model $(date +%H:%M:%S) ##########"
    $PY src/03_train_transformer.py --model $model --experiment $exp
    echo "########## DONE  $exp $model rc=$? $(date +%H:%M:%S) ##########"
  done
done
echo "ALL_TRANSFORMERS_FINISHED"
