#!/bin/bash
# Serve the Qwen3.6-35B-A3B-FP8 reader used by SPARQ+ (closed + open pipelines).
# This is the authoritative launch config used for the paper reproduction on
# a 2-GPU node (tested 2x A100-80G and 2x RTX 4090).
#
# Override via env:
#   MODEL     model dir or HF id   (default: Qwen/Qwen3.6-35B-A3B-FP8)
#   GPUS      CUDA device ids      (default: 0,1)
#   PORT      HTTP port            (default: 9543)
#   MAXLEN    max context length   (default: 32768)
#
# IMPORTANT (matches the paper runs, do NOT change for reproduction):
#   * enable_thinking=false is served as the default chat-template kwarg — the
#     35B is a thinking-capable model; the SPARQ pipeline expects direct
#     (non-thinking) answers, so this MUST be set or the `.content` field can
#     come back null on the SQL/operator path.
#   * NO --enforce-eager in the formal config (it is a low-memory smoke-only
#     shortcut and changes throughput/behaviour).
set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-9543}"
MAXLEN="${MAXLEN:-32768}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_SERVER_DEV_MODE=1
export VLLM_NO_USAGE_STATS=1

exec vllm serve "$MODEL" \
  --served-model-name qwen3.6-35b \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len "$MAXLEN" \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --host 0.0.0.0 --port "$PORT"
