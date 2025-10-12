#!/bin/bash

# Run baseline and ISO-regularised variants of DDPM, SNR-weighted DDPM,
# and Improved DDPM across all supported 2D datasets.

set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}

DATASETS=(
  eight_gaussians
  moons
  swiss_roll
  banana
  central_banana
  moon_circles
  banana_circles
  moon_scatter
)

REG_STRENGTH_ISO=${REG_STRENGTH_ISO:-0.3}
SEED=${SEED:-41}

EXTRA_ARGS=("$@")

run_method() {
  local method=$1
  local dataset=$2
  local reg_strength=$3
  local suffix=$4
  echo "Method=${method} | Dataset=${dataset} | reg_strength=${reg_strength}"
  python main.py \
    --method "${method}" \
    --dataset "${dataset}" \
    --reg_strength "${reg_strength}" \
    --run_suffix "${suffix}" \
    --random_state "${SEED}" \
    "${EXTRA_ARGS[@]}"
}

for dataset in "${DATASETS[@]}"; do
  printf "\n=== Dataset: %s ===\n" "${dataset}"
  run_method ddpm "${dataset}" 0.0 "ddpm_base"
  run_method ddpm "${dataset}" "${REG_STRENGTH_ISO}" "ddpm_iso"
  run_method snr "${dataset}" 0.0 "snr_base"
  run_method snr "${dataset}" "${REG_STRENGTH_ISO}" "snr_iso"
  run_method improved "${dataset}" 0.0 "improved_base"
  run_method improved "${dataset}" "${REG_STRENGTH_ISO}" "improved_iso"
  printf "Completed runs for %s\n\n" "${dataset}"
done

echo "All method comparisons completed."
