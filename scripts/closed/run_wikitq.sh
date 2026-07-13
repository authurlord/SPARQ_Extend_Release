#!/bin/bash
# WikiTQ reproduction script for SPARQ-Extend
# Usage: bash scripts/run_wikitq.sh [MODEL_TAG] [FIRST_N]
# Examples:
#   bash scripts/run_wikitq.sh qwen3.5-35b        # full test with 35B
#   bash scripts/run_wikitq.sh qwen3.5-4b          # full test with 4B
#   bash scripts/run_wikitq.sh qwen3.5-35b 100     # quick test first 100

set -euo pipefail

MODEL_TAG="${1:-qwen3.5-35b}"
FIRST_N="${2:-}"

# Resolve model-specific config
case "${MODEL_TAG}" in
  qwen3.5-35b|35b)
    LLM_NAME="qwen3.5-35b"
    CONCURRENCY=8
    TMP_DIR="results/qwen35_35b/wikitq"
    ;;
  qwen3.5-4b|4b)
    LLM_NAME="qwen3.5-4b"
    CONCURRENCY=32
    TMP_DIR="results/qwen35_4b/wikitq"
    ;;
  *)
    echo "Unsupported model: ${MODEL_TAG}. Use qwen3.5-35b or qwen3.5-4b"
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${PROJECT_DIR}/src"
PIPELINE_DIR="${SRC_DIR}/schedule_pipeline"

# Environment
export PYTHONPATH="${SRC_DIR}:${PIPELINE_DIR}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Disable proxy for localhost API calls
export no_proxy="localhost,127.0.0.1"
unset http_proxy https_proxy all_proxy

# Paths
EMBEDDING_MODEL="/data/workspace/yanmy/models/bge-m3"
ROUTER_MODEL="/data/workspace/yanmy/HybridRAG/H-STAR/router/bge-m3-finetuned/"
CHECK_MODEL="/data/workspace/yanmy/HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned/"
API_BASE="http://localhost:8000/v1"
API_KEY="api-key-qwen3"

mkdir -p "${PROJECT_DIR}/${TMP_DIR}"

echo "=============================================="
echo "SPARQ-Extend: WikiTQ Reproduction"
echo "=============================================="
echo "Model:           ${LLM_NAME}"
echo "Concurrency:     ${CONCURRENCY}"
echo "Embedding:       ${EMBEDDING_MODEL}"
echo "Router:          ${ROUTER_MODEL}"
echo "Check:           ${CHECK_MODEL}"
echo "Output:          ${TMP_DIR}"
[ -n "${FIRST_N}" ] && echo "First N:         ${FIRST_N}"
echo "=============================================="

# Health check vLLM
echo "Checking vLLM server..."
if ! curl -s "${API_BASE}/models" -H "Authorization: Bearer ${API_KEY}" | grep -q "${LLM_NAME}"; then
    echo "ERROR: vLLM server not serving ${LLM_NAME} at ${API_BASE}"
    echo "Start it with: cd /data/workspace/yanmy/Tab-Agent && bash script/start_vllm_qwen_tp2.sh ${MODEL_TAG} 8000"
    exit 1
fi
echo "vLLM server OK: ${LLM_NAME}"

# Build command
CMD="python ${PIPELINE_DIR}/run_full_pipeline_wikitq.py \
  --llm_path ${LLM_NAME} \
  --llm_name ${LLM_NAME} \
  --embedding_model_path ${EMBEDDING_MODEL} \
  --router_model_path ${ROUTER_MODEL} \
  --check_model_path ${CHECK_MODEL} \
  --api_base ${API_BASE} \
  --api_key ${API_KEY} \
  --dataset_name wikitq \
  --split test \
  --tmp_save_path ${PROJECT_DIR}/${TMP_DIR} \
  --tau 0.82 \
  --check_tau 0.8 \
  --n_parallel 32 \
  --select_sample_num 2 \
  --sql_sample_num 3 \
  --llm_concurrency ${CONCURRENCY} \
  --temperature 0.7 \
  --top_p 0.8"

[ -n "${FIRST_N}" ] && CMD="${CMD} --first_n ${FIRST_N}"

echo ""
echo "Running: ${CMD}"
echo ""

cd "${PIPELINE_DIR}"
eval ${CMD} 2>&1 | tee "${PROJECT_DIR}/${TMP_DIR}/test_run.log"

echo ""
echo "=============================================="
echo "WikiTQ reproduction complete. Results in: ${TMP_DIR}"
echo "=============================================="
