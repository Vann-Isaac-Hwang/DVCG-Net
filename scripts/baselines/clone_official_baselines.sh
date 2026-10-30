#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${OFFICIAL_BASELINES_ROOT:-$PROJECT_DIR/external_baselines}"
mkdir -p "$ROOT"
cd "$ROOT"

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "[update] $dir"
    git -C "$dir" fetch --all --tags --prune
  else
    echo "[clone] $url -> $dir"
    git clone "$url" "$dir"
  fi
  echo "[commit] $dir $(git -C "$dir" rev-parse HEAD 2>/dev/null || true)"
}

clone_or_update https://github.com/DL4mHealth/Medformer Medformer
clone_or_update https://github.com/yuqinie98/PatchTST PatchTST
clone_or_update https://github.com/thuml/iTransformer iTransformer
clone_or_update https://github.com/MAZiqing/FEDformer FEDformer
clone_or_update https://github.com/aikunyi/FourierGNN FourierGNN
clone_or_update https://github.com/aikunyi/MedGNN MedGNN

echo
echo "Done. Now run:"
echo "  cd \"$PROJECT_DIR\""
echo "  python -m scripts.diagnostics.check_official_baseline_sources"
echo "  python -m scripts.diagnostics.smoke_test_models --models official_fouriergnn --device cuda --batch_size 2 --seq_len 1000"
