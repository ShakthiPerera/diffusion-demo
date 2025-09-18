#!/bin/bash

# Run multiple experiments on the moon_circles dataset with three random seeds.
#
# This script sweeps over a range of regularisation strengths for every
# supported regularisation type (including iso) and repeats the entire
# sweep three times with different random seeds.  A run suffix is used
# to separate outputs from each repeat, so results will be written to
# outputs/moon_circles/<reg_type>_run_<n>/reg_<value>.

set -e

# Set a deterministic cuBLAS workspace configuration if not provided
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}

# Dataset name for this script
DATASET="moon_circles"

# Comma-separated list of regularisation strengths to evaluate
REG_VALUES="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

# List of regularisation types to test (iso and all other metrics)
REG_TYPES=(iso mean_l2 var_l2 skew kurt var_mi kl mmd_linear mmd_rbf)

# Base seed for reproducibility; each run increments this value
BASE_SEED=42

echo "Running experiments for dataset ${DATASET}"

# Repeat the sweep three times with different seeds
for run in 1 2 3; do
  seed=$((BASE_SEED + run - 1))
  run_suffix="run_${run}"
  echo "\nStarting repeat ${run} with random_state=${seed} and run_suffix=${run_suffix}"
  for reg_type in "${REG_TYPES[@]}"; do
    echo "  Running reg_type=${reg_type}"
    # Enable detailed metrics only for the first run
    if [ "$run" -eq 1 ]; then
      python train.py \
        --dataset "${DATASET}" \
        --reg_type "${reg_type}" \
        --reg_values "${REG_VALUES}" \
        --run_suffix "${run_suffix}" \
        --random_state "${seed}" \
        --enable_metrics \
        "$@"
    else
      python train.py \
        --dataset "${DATASET}" \
        --reg_type "${reg_type}" \
        --reg_values "${REG_VALUES}" \
        --run_suffix "${run_suffix}" \
        --random_state "${seed}" \
        "$@"
    fi
  done
done

echo "All experiments for ${DATASET} completed."