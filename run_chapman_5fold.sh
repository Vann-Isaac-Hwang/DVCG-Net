#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DUAL_ECG_DATA_ROOT:-$PROJECT_DIR/data}"
DATA_DIR="${CHAPMAN_DATA_DIR:-$DATA_ROOT/Chapman_Data/processed_sr100_freq10}"
FOLDS_PATH="${CHAPMAN_FOLDS_PATH:-$DATA_ROOT/Chapman_Data/chapman_folds.csv}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-5}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL="${MODEL:-full}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  ./run_chapman_5fold.sh

Environment overrides:
  MODEL=full START_FOLD=0 END_FOLD=1 EPOCHS=2 PATIENCE=1
  PYTHON_BIN=python NUM_WORKERS=0
  CHAPMAN_DATA_DIR="/path/to/processed_sr100_freq10"
  CHAPMAN_FOLDS_PATH="/path/to/chapman_folds.csv"
EOF
    exit 0
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Chapman data directory not found: $DATA_DIR" >&2
    exit 1
fi
if [[ ! -f "$FOLDS_PATH" ]]; then
    echo "Chapman fold metadata not found: $FOLDS_PATH" >&2
    exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$PROJECT_DIR/logs_chapman_${timestamp}"
mkdir -p "$log_dir"

echo "Data:       $DATA_DIR"
echo "Fold file:  $FOLDS_PATH"
echo "Fold range: $START_FOLD -> $END_FOLD"
echo "Model:      $MODEL"

DUAL_ECG_DATA_ROOT="$DATA_ROOT" CHAPMAN_FOLDS_PATH="$FOLDS_PATH" "$PYTHON_BIN" -X utf8 main_train.py \
    --dataset CHAPMAN \
    --loss CE \
    --task single_label \
    --ablation "$MODEL" \
    --data_dir "$DATA_DIR" \
    --start_fold "$START_FOLD" \
    --end_fold "$END_FOLD" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_clip 0 \
    --num_workers "$NUM_WORKERS" \
    --deterministic true \
    --save_last_every 5 \
    --lightweight_best true \
    --disable_tqdm \
    >"$log_dir/train.log" 2>&1

DUAL_ECG_DATA_ROOT="$DATA_ROOT" CHAPMAN_FOLDS_PATH="$FOLDS_PATH" "$PYTHON_BIN" -X utf8 main_eval.py \
    --dataset CHAPMAN \
    --loss CE \
    --task single_label \
    --ablation "$MODEL" \
    --data_dir "$DATA_DIR" \
    --start_fold "$START_FOLD" \
    --end_fold "$END_FOLD" \
    >"$log_dir/eval.log" 2>&1

echo "Finished. Logs: $log_dir"
