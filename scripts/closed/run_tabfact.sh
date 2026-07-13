#!/bin/bash
# TabFact reproduction script for SPARQ-Extend
set -euo pipefail
MODEL_TAG="${1:-qwen3.5-35b}"
FIRST_N="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${PROJECT_DIR}/src"
PIPELINE_DIR="${SRC_DIR}/schedule_pipeline"

export PYTHONPATH="${SRC_DIR}:${PIPELINE_DIR}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
unset http_proxy https_proxy all_proxy
export no_proxy="localhost,127.0.0.1"

TMP_DIR="results/${MODEL_TAG//./_}/tabfact"
mkdir -p "${PROJECT_DIR}/${TMP_DIR}"

echo "=== TabFact Reproduction: ${MODEL_TAG} ==="

CMD="/home/aizoo/miniconda3/envs/hstar/bin/python ${PIPELINE_DIR}/run_full_pipeline_tabfact.py \
  --llm_path ${MODEL_TAG} --llm_name ${MODEL_TAG} \
  --embedding_model_path /data/workspace/yanmy/models/bge-m3 \
  --router_model_path /data/workspace/yanmy/HybridRAG/H-STAR/router/bge-m3-router-tab_fact-hn \
  --check_model_path /data/workspace/yanmy/HybridRAG/H-STAR/check/output/bge-reranker-v2-m3-finetuned-tab_fact \
  --api_base http://localhost:8000/v1 --api_key api-key-qwen3 \
  --dataset_name tab_fact --split test_small \
  --tmp_save_path ${PROJECT_DIR}/${TMP_DIR} \
  --tau 0.75 --check_tau 0.8 --n_parallel 32 \
  --select_sample_num 2 --sql_sample_num 3 --llm_concurrency 8 \
  --temperature 0.7 --top_p 0.8 \
  --skip_preprocess"

[ -n "${FIRST_N}" ] && CMD="${CMD} --first_n ${FIRST_N}"
cd "${PIPELINE_DIR}" && eval ${CMD} 2>&1 | tee "${PROJECT_DIR}/${TMP_DIR}/test_run.log"
