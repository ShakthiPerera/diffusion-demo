#!/bin/bash

# Compare SNR-weighted DDPM (no regularisation) vs ISO regularisation (reg=0.3)
# with SNR weighting, across all supported datasets. One run per dataset.
#
# Outputs are separated using --run_suffix so folders don't collide:
#   outputs/<dataset>/iso_snr_ddpm/reg_0.0
#   outputs/<dataset>/iso_snr_iso_reg0.3/reg_0.3

set -euo pipefail

# Set a deterministic cuBLAS workspace configuration if not provided
# export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}

# All supported datasets
DATASETS=(
  swiss_roll
  central_banana
  moon_circles
  moon_scatter
)

# Single seed for a single run per dataset
SEED=${SEED:-0}

echo "Comparing SNR-DDPM vs ISO (reg=0.3 with constant) across datasets"

for DATASET in "${DATASETS[@]}"; do
  echo "\nDataset: ${DATASET}"

  # 2) ISO with reg=0.3 and SNR weighting
  echo "  Running ISO (reg=0.01, weighting=constant)"
  python train.py \
    --dataset "${DATASET}" \
    --weighting constant \
    --reg_type iso \
    --reg_strength 0.01 \
    --run_suffix constant_iso_reg0.01 \
    --random_state "${SEED}" \
    --gpu_id 4 \
    "$@"
done

echo "All comparisons completed."

