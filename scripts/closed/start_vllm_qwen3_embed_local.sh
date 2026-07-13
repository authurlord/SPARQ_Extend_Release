#!/bin/bash
# vLLM serving Qwen3-Embedding-0.6B on 51.10 cuda 0.
# Endpoint: /v1/embeddings (OpenAI-compatible).
# Call pattern: client.embeddings.create(input=[1024 texts]) per request.

set -uo pipefail

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1

MODEL="${MODEL:-/home/yanmy/models/Qwen3-Embedding-0.6B}"
PORT="${PORT:-8090}"
API_KEY="${VLLM_EMBED_API_KEY:-embed-key-qwen3}"
GPU_MEM="${GPU_MEM:-0.40}"

LOG=/tmp/_vllm_qwen3_embed.log
PIDFILE=/tmp/_vllm_qwen3_embed.pid

echo "[$(date)] launching vLLM Qwen3-Embedding-0.6B on cuda:0 port=$PORT" > "$LOG"
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
if [ "${GPU_USED:-0}" -gt 1000 ]; then
  echo "[FATAL] GPU 0 has ${GPU_USED} MiB used" | tee -a "$LOG"
  exit 1
fi
if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}$"; then
  echo "[FATAL] port ${PORT} already in use" | tee -a "$LOG"
  exit 1
fi

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen3-embed \
  --host 0.0.0.0 \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --runner pooling \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len 4096 \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 256 \
  >> "$LOG" 2>&1 &

PID=$!
echo $PID > "$PIDFILE"
echo "[$(date)] launched pid=$PID  log=$LOG" >> "$LOG"
echo "$PID"

for i in $(seq 1 60); do
  sleep 5
  if curl -s --max-time 3 -H "Authorization: Bearer ${API_KEY}" \
       "http://127.0.0.1:${PORT}/v1/models" | grep -q 'qwen3-embed'; then
    echo "[$(date)] /v1/models OK after $((i * 5))s" >> "$LOG"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[FATAL $(date)] vllm pid $PID died during warmup" >> "$LOG"
    tail -50 "$LOG"
    exit 1
  fi
done
echo "[WARN] did not become healthy in 5 min" >> "$LOG"
exit 1
