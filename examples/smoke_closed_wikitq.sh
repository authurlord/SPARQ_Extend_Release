#!/usr/bin/env bash
# Closed-setting smoke test: WikiTQ, first 20 questions, Qwen3.6-35B reader.
# Verifies the full closed pipeline (router -> select row/col -> execute SQL ->
# verifier -> final QA) runs end-to-end and scores.
#
# Prereqs:
#   * A reader served at $API_BASE (see scripts/closed/start_vllm_35b.sh).
#   * Retrieval checkpoints: BGE-M3 + the H-STAR router/check rerankers
#     (paths below; download BGE-M3 from HF, see docs/MODEL_CARDS.md).
#   * WikiTQ cached for 🤗 datasets.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/src/schedule_pipeline"
for v in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do unset $v || true; done
export TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

API_BASE="${API_BASE:-http://127.0.0.1:9543/v1}"
MODEL="${MODEL:-qwen3.6-35b}"
EMB="${EMB:-BAAI/bge-m3}"
ROUTER="${ROUTER:?set ROUTER=/path/to/H-STAR/router/wikitq}"
CHECK="${CHECK:?set CHECK=/path/to/H-STAR/check/wikitq}"
N="${N:-20}"

# Force retrieval models to CPU if you have no spare GPU (small N only):
#   export CUDA_VISIBLE_DEVICES=""

python -u src/schedule_pipeline/run_full_pipeline_wikitq_api.py \
  --api_base "$API_BASE" --api_key EMPTY --model_name "$MODEL" \
  --embedding_model_path "$EMB" \
  --router_model_path "$ROUTER" \
  --check_model_path "$CHECK" \
  --dataset_name wikitq --split test \
  --tmp_save_path tmp/wikitq_35b_smoke \
  --first_n "$N" --n_parallel 8 --concurrency 8
