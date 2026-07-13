#!/usr/bin/env bash
# Open-domain HybridQA smoke: run the 35B reader over N frozen per-query inputs
# (table + top-30 cell-link passages + CoT prompt) and score EM/F1.
# No retrieval / no weights needed — the evidence is pre-assembled.
set -euo pipefail
cd "$(dirname "$0")/.."
for v in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do unset $v || true; done
API_BASE="${API_BASE:-http://127.0.0.1:9543/v1}"
MODEL="${MODEL:-qwen3.6-35b}"
N="${N:-30}"
python -u scripts/open/remote_timed_reader.py \
  analysis/reader_prompts/sparqx_cot_hybridqa.jsonl.gz \
  --api-base "$API_BASE" --api-key EMPTY --model "$MODEL" \
  --concurrency 16 --max-queries "$N" --tag smoke
# Full-set (N=0/all): EM 0.5069 / F1 0.5528 (paper Table VI: 0.508 / 0.549).
