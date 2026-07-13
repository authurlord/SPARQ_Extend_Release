#!/usr/bin/env bash
cd /home/yanmy/SPARQ/schedule_pipeline
export PYTHONPATH="$(pwd)/..:$(pwd):$PYTHONPATH" TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2
for v in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do unset $v; done
t0=$(date +%s)
python -u run_full_pipeline_wikitq_api.py \
  --api_base http://127.0.0.1:8000/v1 --model_name qwen3.6-35b \
  --embedding_model_path /home/yanmy/models/bge-m3 \
  --router_model_path /home/yanmy/HybridRAG/H-STAR/router/wikitq/ \
  --check_model_path /home/yanmy/HybridRAG/H-STAR/check/wikitq/ \
  --dataset_name wikitq --split test --tmp_save_path tmp/wikitq_35b \
  --first_n -1 --n_parallel 48 --concurrency 48
echo "WIKITQ_35B_DONE rc=$? runtime=$(( $(date +%s)-t0 ))s"
