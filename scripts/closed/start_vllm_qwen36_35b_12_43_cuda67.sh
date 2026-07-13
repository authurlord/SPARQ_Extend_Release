#!/bin/bash
# Launch own Qwen3.6-35B-A3B-FP8 on 12.43 cuda 6,7 port 9541.
# Owner: wangys. Use this instead of zhouyue's 9540.
set -uo pipefail

source /data/home/wangys/anaconda3/etc/profile.d/conda.sh
conda activate vllm_pascal
export PATH=/data/home/wangyaoshu/.local/bin:/data/home/wangys/.local/bin:$PATH

export CUDA_VISIBLE_DEVICES=6,7
export VLLM_SERVER_DEV_MODE=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1

MODEL="${MODEL:-/data/home/wangys/model/Qwen3.6-35B-A3B-FP8}"
PORT="${PORT:-9541}"
API_KEY="${API_KEY:-EMPTY}"

LOG=/tmp/_vllm_qwen36_35b_9541.log
PIDFILE=/tmp/_vllm_qwen36_35b_9541.pid

echo "[$(date)] launching vllm Qwen3.6-35B-A3B-FP8 TP=2 cuda:6,7 port=$PORT" > "$LOG"
VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>&1 || echo unknown)
echo "[$(date)] vllm version: $VLLM_VER" | tee -a "$LOG"

if [ ! -d "$MODEL" ]; then
  echo "[FATAL] model dir not found: $MODEL" | tee -a "$LOG"; exit 1
fi

nohup vllm serve "$MODEL" \
  --served-model-name Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port "$PORT" \
  >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "[$(date)] vllm pid=$(cat $PIDFILE)" | tee -a "$LOG"
