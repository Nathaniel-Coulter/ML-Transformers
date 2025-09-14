#!/usr/bin/env bash
set -euo pipefail

NPZS=("outputs/tensors_L252_H1.npz" "outputs/tensors_L252_H5.npz" "outputs/tensors_L252_H20.npz")
ENCS=("pointwise" "patch" "ivar" "cross" "autoformer" "fedformer" "informer" "timesnet" "timexer")
SEEDS=(0 1 2)

mkdir -p outputs/metrics outputs/preds

for NPZ in "${NPZS[@]}"; do
  base=$(basename "${NPZ%.npz}")   # e.g., tensors_L252_H1
  for ENC in "${ENCS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      extras=()
      if [[ "$ENC" == "ivar" ]]; then
        extras+=(--ivar-time2chan 128)
      fi
      if [[ "$ENC" == "patch" || "$ENC" == "cross" || "$ENC" == "timexer" ]]; then
        extras+=(--patch-len 16 --stride 8)
      fi

      TAG="${ENC}_${base}_s${SEED}"
      echo "=== NPZ=$NPZ ENC=$ENC SEED=$SEED TAG=$TAG ==="
      python -m src.train_baselines \
        --npz "$NPZ" \
        --encoder "$ENC" \
        --epochs 30 --bs 128 --lr 3e-4 --weight-decay 1e-4 \
        --seed "$SEED" \
        --eval-test --dump-preds \
        --tag "$TAG" "${extras[@]}"
    done
  done
done
