#!/usr/bin/env bash
set -euo pipefail

# Datasets to process; adjust as needed.
DATASETS=(moon_scatter swiss_roll central_banana)

SCRIPT="entropy_calculation.py"

for ds in "${DATASETS[@]}"; do
  echo "=== Running entropy calculation for ${ds} (reg=0.0) ==="
  python "${SCRIPT}" --dataset "${ds}" --reg_strength 0.0 --save_dir outputs/${ds}_reg_0.0 --train_steps 20000 --gpu_id 7 "$@"
  echo "=== Running entropy calculation for ${ds} (reg=0.3) ==="
  python "${SCRIPT}" --dataset "${ds}" --reg_strength 0.3 --save_dir outputs/${ds}_reg_0.3 --train_steps 20000 --gpu_id 7 "$@"
done
