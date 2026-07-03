#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?model: llama | qwen | vicuna}"
OUTPUT="${2:-${ROOT}/evaluation/calibration/lts_${MODEL}_params_t_1.pt}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1

python "${ROOT}/evaluation/arc_lts_calibrate.py" \
  --model "${MODEL}" \
  --temperature 1.0 \
  --output "${OUTPUT}"
