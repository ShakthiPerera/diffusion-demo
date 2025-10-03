#!/bin/bash

# Run normal DDPM (no SNR weighting, no regularisation) AND ISO-regularised
# (reg=0.3) models with different variance (beta) schedules.
# One run per (schedule, model) for all datasets.
#
# Outputs are separated using --run_suffix per schedule/model so folders are unique:
#   Baseline DDPM: outputs/<dataset>/iso_ddpm_linear/reg_0.0
#   ISO (0.3):     outputs/<dataset>/iso_iso_linear/reg_0.3

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

# Schedules supported by the codebase
SCHEDULES=(cosine linear quadratic sigmoid)

# Single seed for a single run per (dataset,schedule)
SEED=${SEED:-42}

echo "Running DDPM schedule sweep across datasets"

for DATASET in "${DATASETS[@]}"; do
  echo "\nDataset: ${DATASET}"
  for S in "${SCHEDULES[@]}"; do
    # Baseline DDPM (no regularisation), constant weighting
    echo "  Schedule: ${S} | Model: DDPM (reg=0.0)"
    python train.py \
      --dataset "${DATASET}" \
      --schedule "${S}" \
      --weighting constant \
      --reg_type iso \
      --reg_strength 0.0 \
      --run_suffix "ddpm_${S}" \
      --random_state "${SEED}" \
      "$@"

    # ISO-regularised (reg=0.3), constant weighting
    echo "  Schedule: ${S} | Model: ISO (reg=0.3)"
    python train.py \
      --dataset "${DATASET}" \
      --schedule "${S}" \
      --weighting constant \
      --reg_type iso \
      --reg_strength 0.3 \
      --run_suffix "iso_${S}" \
      --random_state "${SEED}" \
      "$@"
  done
done

echo "DDPM schedule sweep completed."
