#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?model: llama | qwen | vicuna}"
BENCHMARK="${2:?benchmark: mt_bench | humaneval | gsm8k | alpaca}"
TOTAL_TOKENS="${3:-60}"
DEPTH="${4:-7}"
OUTPUT_DIR="${5:-${ROOT}/results/arc/${MODEL}/${BENCHMARK}/tt${TOTAL_TOKENS}_d${DEPTH}}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1

python "${ROOT}/evaluation/arc_total_depth_eval.py" \
  --model "${MODEL}" \
  --benchmark "${BENCHMARK}" \
  --mode arc \
  --total-tokens "${TOTAL_TOKENS}" \
  --depth "${DEPTH}" \
  --temperature 1.0 \
  --full-data \
  --output-dir "${OUTPUT_DIR}"
