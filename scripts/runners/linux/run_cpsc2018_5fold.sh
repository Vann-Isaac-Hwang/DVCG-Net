#!/usr/bin/env bash
set -euo pipefail

# Linux 5-fold CPSC2018 train/eval runner using the paper protocol.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${ECG_ENV_PYTHON:-}" && -x "$ECG_ENV_PYTHON" ]]; then
        PYTHON_BIN="$ECG_ENV_PYTHON"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python"
    elif [[ -x "$HOME/.conda/envs/ecg_env/bin/python" ]]; then
        PYTHON_BIN="$HOME/.conda/envs/ecg_env/bin/python"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi
if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
DATA_ROOT="${DUAL_ECG_DATA_ROOT:-$PROJECT_DIR/data}"
DATA_DIR="${CPSC_DATA_DIR:-$DATA_ROOT/CPSC_Data/processed_sr100_freq10}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-5}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
GRAD_CLIP="${GRAD_CLIP:-0}"
LIVE_PROGRESS="${LIVE_PROGRESS:-1}"

# Override with a space-separated list, for example:
# MODELS="resnet1d no_freq_split_aff" ./scripts/runners/linux/run_cpsc2018_5fold.sh
MODELS="${MODELS:-resnet1d transformer1d official_patchtst official_itransformer official_fedformer official_medformer official_medgnn official_fouriergnn tgllnet no_freq_split_aff}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  ./scripts/runners/linux/run_cpsc2018_5fold.sh

Environment overrides:
  MODELS="resnet1d no_freq_split_aff"
  START_FOLD=0 END_FOLD=1 EPOCHS=2 PATIENCE=1
  ECG_ENV_PYTHON=/path/to/ecg_env/bin/python NUM_WORKERS=0 BATCH_SIZE=16 GRAD_CLIP=1.0
  PYTHON_BIN=/path/to/custom/python  # explicit interpreter override
  RESUME_CHECKPOINT="/path/to/last_checkpoint.pth"
  CPSC_DATA_DIR="/path/to/processed_sr100_freq10"
  LIVE_PROGRESS=0  # disable live terminal progress; keep file logging only
  ./scripts/runners/linux/run_cpsc2018_5fold.sh --check-env
EOF
    exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c 'import torch, einops' >/dev/null 2>&1; then
    echo "Selected Python is missing required packages (torch, einops): $PYTHON_BIN" >&2
    echo "Activate ecg_env or set ECG_ENV_PYTHON=/path/to/ecg_env/bin/python." >&2
    exit 1
fi
if [[ "${1:-}" == "--check-env" ]]; then
    "$PYTHON_BIN" - <<'PY'
import einops
import sys
import torch
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"einops: {einops.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
    exit 0
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "CPSC processed data directory not found: $DATA_DIR" >&2
    exit 1
fi

required_files=(x_lf.pth x_hf.pth labels.pth)
for name in "${required_files[@]}"; do
    if [[ ! -f "$DATA_DIR/$name" ]]; then
        echo "Missing CPSC file: $DATA_DIR/$name" >&2
        exit 1
    fi
done

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$PROJECT_DIR/logs_cpsc2018_${timestamp}"
mkdir -p "$log_dir"

echo "Project:      $PROJECT_DIR"
echo "Python:       $PYTHON_BIN"
echo "Python path:  $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "CUDA:         $("$PYTHON_BIN" -c 'import torch; print(torch.cuda.is_available())')"
echo "Data:         $DATA_DIR"
echo "Fold range:   $START_FOLD -> $END_FOLD"
echo "Epochs:       $EPOCHS"
echo "Patience:     $PATIENCE"
echo "Grad clip:    $GRAD_CLIP"
echo "Models:       $MODELS"
echo "Protocol:     lr=1e-4, val_calibrated, disjoint train/val/test"

for model in $MODELS; do
    ablation="$model"
    extra_args=()
    if [[ "$model" == "no_freq_split_aff" ]]; then
        extra_args+=(--aff_top_ratio 0.2 --aff_component dominant)
    fi
    if [[ -n "$BATCH_SIZE" ]]; then
        extra_args+=(--batch_size "$BATCH_SIZE")
    fi
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        extra_args+=(--resume_checkpoint "$RESUME_CHECKPOINT")
    fi

    train_log="$log_dir/train_${model}.log"
    eval_log="$log_dir/eval_${model}.log"

    echo
    echo "========== TRAIN $model =========="
    train_args=(
        --dataset CPSC
        --loss BCE
        --task multilabel
        --ablation "$ablation"
        --data_dir "$DATA_DIR"
        --start_fold "$START_FOLD"
        --end_fold "$END_FOLD"
        --epochs "$EPOCHS"
        --patience "$PATIENCE"
        --lr 1e-4
        --weight_decay 1e-4
        --grad_clip "$GRAD_CLIP"
        --num_workers "$NUM_WORKERS"
        --deterministic true
        --save_last_every 5
        --lightweight_best true
        "${extra_args[@]}"
    )
    if [[ "$LIVE_PROGRESS" != "1" ]]; then
        train_args+=(--disable_tqdm)
    fi

    "$PYTHON_BIN" -u -X utf8 main_train.py \
        "${train_args[@]}" \
        2>&1 | tee "$train_log"

    checkpoint_dir="$PROJECT_DIR/checkpoints/CPSC_BCE_Weights"
    checkpoint_pattern="best_dual_dgnn_cpsc_bce_${ablation}_fold_*.pth"
    if [[ "$model" == "no_freq_split_aff" ]]; then
        checkpoint_pattern="best_dual_dgnn_cpsc_bce_aff0p2_dom_no_freq_split_aff_fold_*.pth"
    fi
    expected_count=$((END_FOLD - START_FOLD))
    checkpoint_count=0
    for ((fold_idx=START_FOLD; fold_idx<END_FOLD; fold_idx++)); do
        if [[ "$model" == "no_freq_split_aff" ]]; then
            fold_checkpoint="$checkpoint_dir/best_dual_dgnn_cpsc_bce_aff0p2_dom_no_freq_split_aff_fold_${fold_idx}.pth"
        else
            fold_checkpoint="$checkpoint_dir/best_dual_dgnn_cpsc_bce_${ablation}_fold_${fold_idx}.pth"
        fi
        if [[ -f "$fold_checkpoint" ]]; then
            checkpoint_count=$((checkpoint_count + 1))
        fi
    done
    if [[ "$checkpoint_count" -lt "$expected_count" ]]; then
        echo "Training completed but only $checkpoint_count/$expected_count checkpoints found for $model." >&2
        echo "See log: $train_log" >&2
        exit 1
    fi

    echo "========== EVAL $model =========="
    "$PYTHON_BIN" -u -X utf8 main_eval.py \
        --dataset CPSC \
        --loss BCE \
        --task multilabel \
        --ablation "$ablation" \
        --data_dir "$DATA_DIR" \
        --threshold_mode val_calibrated \
        --start_fold "$START_FOLD" \
        --end_fold "$END_FOLD" \
        "${extra_args[@]}" \
        2>&1 | tee "$eval_log"

    record_args=(
        --model "$model"
        --run-dir "$log_dir"
        --start-fold "$START_FOLD"
        --end-fold "$END_FOLD"
        --epochs "$EPOCHS"
        --patience "$PATIENCE"
        --grad-clip "$GRAD_CLIP"
    )
    if [[ -n "$BATCH_SIZE" ]]; then
        record_args+=(--batch-size "$BATCH_SIZE")
    fi
    "$PYTHON_BIN" -X utf8 scripts/runners/linux/record_cpsc_run.py "${record_args[@]}"

    echo "Finished $model"
    echo "  train log: $train_log"
    echo "  eval log:  $eval_log"
done

echo
echo "CPSC2018 5-fold run complete."
echo "Logs: $log_dir"
