#!/usr/bin/env bash
# FetaQA conference reproduction on the 35B reader (9543).
# Run AFTER NIAT frees 9543 (serialize on 9543). Scores ROUGE-L fmeasure.
set -euo pipefail
cd "$(dirname "$0")"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="127.0.0.1,localhost,192.168.12.43,192.168.0.0/16"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3
export SPARQX_SQL_TIMEOUT=8

python -u run_full_pipeline_fetaqa_api.py \
  --api_base http://192.168.12.43:9543/v1 --model_name qwen3.6-35b --api_key EMPTY \
  --embedding_model_path /home/yanmy/models/bge-m3 \
  --router_model_path /home/yanmy/HybridRAG/H-STAR/router/wikitq/ \
  --check_model_path /home/yanmy/HybridRAG/H-STAR/check/wikitq/ \
  --dataset_name fetaqa --split test --tmp_save_path tmp/fetaqa_35b \
  --first_n -1 --n_parallel 64 --concurrency 64 --max_tokens 2048
