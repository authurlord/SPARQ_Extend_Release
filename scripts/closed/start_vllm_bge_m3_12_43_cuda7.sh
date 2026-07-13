#!/bin/bash
# vllm 0.19.1 embedder for bge-m3 on 12.43 GPU 7 (port 8001).
# Mirrors PASCAL launch flags + KGQA --task embed pattern.
set -uo pipefail

source /data/home/wangys/anaconda3/etc/profile.d/conda.sh
conda activate vllm_pascal
export PATH=/data/home/wangyaoshu/.local/bin:/data/home/wangys/.local/bin:$PATH

export CUDA_VISIBLE_DEVICES=7
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1

MODEL="${MODEL:-/data/home/wangyaoshu/models/bge-m3}"
PORT="${PORT:-8001}"
API_KEY="${VLLM_EMBED_API_KEY:-embed-key-m3}"

LOG=/tmp/_vllm_bge_m3.log
PIDFILE=/tmp/_vllm_bge_m3.pid

echo "[$(date)] launching vllm 0.19.1 bge-m3 embedder on cuda:7 port=$PORT" > "$LOG"
VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>&1 || echo unknown)
echo "[$(date)] vllm version: $VLLM_VER" | tee -a "$LOG"

if [ ! -d "$MODEL" ]; then
  echo "[FATAL] model dir not found: $MODEL" | tee -a "$LOG"; exit 1
fi
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 | tr -d ' ')
if [ "${GPU_USED:-0}" -gt 1000 ]; then
  echo "[FATAL] GPU 7 has ${GPU_USED} MiB used; resolve before launching." | tee -a "$LOG"
  exit 1
fi
if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}$"; then
  echo "[FATAL] port ${PORT} already in use" | tee -a "$LOG"; exit 1
fi

# vllm 0.19.1 dropped `--task embed`; auto-detection picks up `embed` from
# the model's config.json (`pooling` field). Fallback flag is `--runner pooling`.
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name bge-m3 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --runner pooling \
  --gpu-memory-utilization "${GPU_MEM:-0.30}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  >> "$LOG" 2>&1 &

PID=$!
echo $PID > "$PIDFILE"
echo "[$(date)] launched pid=$PID  log=$LOG  pidfile=$PIDFILE" >> "$LOG"
echo "$PID"

for i in $(seq 1 30); do
  sleep 6
  if curl -s --max-time 3 -H "Authorization: Bearer ${API_KEY}" \
       "http://127.0.0.1:${PORT}/v1/models" | grep -q 'bge-m3'; then
    echo "[$(date)] /v1/models OK after $((i * 6))s" >> "$LOG"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[FATAL $(date)] embed vllm pid $PID died during warmup" >> "$LOG"
    tail -50 "$LOG" >> "$LOG.died"
    exit 1
  fi
done
echo "[WARN $(date)] embed vllm did not become healthy in 3 min; check $LOG" >> "$LOG"
exit 1
