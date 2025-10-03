#!/bin/bash

# Compare SNR-weighted DDPM (no regularisation) vs ISO regularisation (reg=0.3)
# with SNR weighting, across all supported datasets. One run per dataset.
#
# Outputs are separated using --run_suffix so folders don't collide:
#   outputs/<dataset>/iso_snr_ddpm/reg_0.0
#   outputs/<dataset>/iso_snr_iso_reg0.3/reg_0.3

set -euo pipefail

# Set a deterministic cuBLAS workspace configuration if not provided
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}

# All supported datasets
DATASETS=(
  swiss_roll
  central_banana
  moon_circles
  moon_scatter
)

# Single seed for a single run per dataset
SEED=${SEED:-42}

echo "Comparing SNR-DDPM vs ISO (reg=0.3 with SNR) across datasets"

for DATASET in "${DATASETS[@]}"; do
  echo "\nDataset: ${DATASET}"

  # 1) SNR-DDPM baseline: no regularisation (reg_strength=0.0), SNR weighting
  echo "  Running SNR-DDPM baseline (reg=0.0, weighting=snr)"
  python train.py \
    --dataset "${DATASET}" \
    --weighting snr \
    --reg_type iso \
    --reg_strength 0.0 \
    --run_suffix snr_ddpm \
    --random_state "${SEED}" \
    "$@"

  # 2) ISO with reg=0.3 and SNR weighting
  echo "  Running ISO (reg=0.3, weighting=snr)"
  python train.py \
    --dataset "${DATASET}" \
    --weighting snr \
    --reg_type iso \
    --reg_strength 0.3 \
    --run_suffix snr_iso_reg0.3 \
    --random_state "${SEED}" \
    "$@"
done

echo "All comparisons completed."

