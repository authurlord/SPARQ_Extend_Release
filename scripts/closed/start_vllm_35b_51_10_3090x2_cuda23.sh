#!/bin/bash
# vLLM Qwen3.5-35B-A3B-FP8 on 51.10, 2× 3090 (cuda 2,3) — ALL optimizations.
# Lifted from Tab-Agent/script/start_vllm_qwen_tp2.sh (qwen3.5-35b case), pinned to cuda 2,3.
#
#   bash scripts/start_vllm_35b_51_10_3090x2_cuda23.sh [PORT]
#
# Overridable via env: GPU_MEM, MAX_MODEL_LEN, MAX_BATCHED, MAX_SEQS, PERF_MODE, MODEL.
set -euo pipefail

PORT=${1:-9543}

# --- 51.10 / 2× 3090 placement (GPU policy: cuda 2,3 only) ---
export CUDA_VISIBLE_DEVICES=2,3
# vLLM env that the Tab-Agent runs relied on:
export PATH="/home/aizoo/miniconda3/envs/vllm-qwen3.5/bin:$PATH"   # vllm-qwen3.5 env
export VLLM_SERVER_DEV_MODE=1          # enables /sleep + /wake_up (DEV_MODE=1)
export SAFETENSORS_FAST_GPU=1          # faster weight load

MODEL="${MODEL:-/data/workspace/yanmy/models/Qwen3.5-35B-A3B-FP8}"   # FP8 MoE checkpoint
SERVED="qwen3.5-35b"
GPU_MEM="${GPU_MEM:-0.90}"             # 3090=24G; 0.90 leaves headroom for KV
MAX_MODEL_LEN="${MAX_MODEL_LEN:-23000}"
MAX_BATCHED="${MAX_BATCHED:-23000}"
MAX_SEQS="${MAX_SEQS:-8}"             # 35B MoE: keep modest for latency
PERF_MODE="${PERF_MODE:-interactivity}"

echo ">>> Qwen3.5-35B-A3B-FP8  TP2 on cuda 2,3 (51.10 2× 3090)  port=${PORT}"
echo ">>> gpu_mem=${GPU_MEM} max_model_len=${MAX_MODEL_LEN} max_seqs=${MAX_SEQS} perf=${PERF_MODE}"

/home/aizoo/miniconda3/envs/vllm-qwen3.5/bin/vllm serve "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --api-key "api-key-qwen3" \
  --tensor-parallel-size 2 \
  --attention-backend FLASHINFER \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_BATCHED}" \
  --max-num-seqs "${MAX_SEQS}" \
  --kv-cache-dtype auto \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-sleep-mode \
  --async-scheduling \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --performance-mode "${PERF_MODE}" \
  --reasoning-parser qwen3 \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only
