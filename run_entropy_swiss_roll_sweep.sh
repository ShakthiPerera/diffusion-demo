#!/usr/bin/env bash
set -euo pipefail

# Noise levels to sweep for swiss_roll
NOISE_LEVELS=(0.05 0.1 0.2 0.3)
SCRIPT="entropy_calculation.py"

for noise in "${NOISE_LEVELS[@]}"; do
  for reg in 0.0 0.3; do
    echo "=== swiss_roll | noise=${noise} | reg=${reg} ==="
    python "${SCRIPT}" \
      --dataset swiss_roll \
      --noise_level "${noise}" \
      --reg_strength "${reg}" \
      --train_steps 1000 \
      "$@"
  done
done
