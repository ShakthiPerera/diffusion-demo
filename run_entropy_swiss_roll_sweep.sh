#!/usr/bin/env bash
set -euo pipefail

# Noise levels to sweep for swiss_roll
# NOISE_LEVELS=(0.01 0.1 0.2 0.5 0.7)
NOISE_LEVELS=(0.5)
SCRIPT="entropy_calculation.py"

for noise in "${NOISE_LEVELS[@]}"; do
  for reg in 0.0 0.3; do
    echo "=== swiss_roll | noise=${noise} | reg=${reg} ==="
    python "${SCRIPT}" \
      --dataset swiss_roll \
      --save_dir outputs/swiss_roll_${noise}_reg_${reg} \
      --noise_level "${noise}" \
      --reg_strength "${reg}" \
      --train_steps 20000 \
      --gpu_id 6 \
      "$@"
  done
done
