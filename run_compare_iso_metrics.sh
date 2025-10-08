#!/bin/bash

# Compare constant-weighted DDPM runs across the five isotropy regularisers
# (iso, iso_frob, iso_split, iso_logeig, iso_bures) at a fixed reg_strength.
# Each invocation trains one model per regulariser by calling train.py with
# identical hyperparameters except for --reg_type.

set -euo pipefail

REG_STRENGTH=${REG_STRENGTH:-0.3}
SEED=${SEED:-0}
RUN_SUFFIX=${RUN_SUFFIX:-iso_compare}

# Datasets to sweep; override via DATASETS env (space-separated list)
DEFAULT_DATASETS=("swiss_roll" "moon_scatter" "moon_circles" "central_banana")
if [[ -n "${DATASETS:-}" ]]; then
  read -ra SELECTED_DATASETS <<< "${DATASETS}"
else
  SELECTED_DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# Optional extra arguments passed straight to train.py (e.g. --train_steps 20000)
EXTRA_ARGS=("$@")

ISO_REG_TYPES=(
  iso
  iso_frob
  iso_split
  iso_logeig
  iso_bures
)

for DATASET in "${SELECTED_DATASETS[@]}"; do
  echo ""
  echo "Dataset: ${DATASET} | reg_strength=${REG_STRENGTH}"
  for REG_TYPE in "${ISO_REG_TYPES[@]}"; do
    echo "  -> Training with reg_type=${REG_TYPE}"
    python train.py \
      --dataset "${DATASET}" \
      --weighting constant \
      --reg_strength "${REG_STRENGTH}" \
      --reg_type "${REG_TYPE}" \
      --run_suffix "${RUN_SUFFIX}" \
      --random_state "${SEED}" \
      "${EXTRA_ARGS[@]}"
  done
done

echo ""
echo "All isotropy runs completed."
