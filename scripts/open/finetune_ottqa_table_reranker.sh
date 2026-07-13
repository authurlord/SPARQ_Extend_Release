#!/bin/bash
# Fine-tune bge-reranker-v2-m3 as OTT-QA Stage 1 TABLE reranker.
# 51.10 cuda 0,1 DDP (vLLM Qwen3-Embed temporarily stopped).
# Per user 2026-05-20 GPU policy: 51.10 cuda 0,1 OK if free; 12.43 cuda 6,7 OK if free.

set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1

OUT=models/ottqa_table_reranker_v1
TRAIN_DATA=data/ottqa_table_reranker_training/train_v1.jsonl

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "[err] $TRAIN_DATA missing — run scripts/prepare_ottqa_table_reranker_data.py first" >&2
    exit 2
fi

mkdir -p "$OUT"

# Optimizations:
# - max_len 320 (vs 512): table_text ~250 tok + query ~20 tok = 270; 320 = small slack
# - no gradient_checkpointing (24GB headroom × 2 since vLLM stopped)
# - DDP 2-GPU
# - per_device_bs=8 × accum=2 × 2 GPU = effective 32 (matches old single-gpu)
torchrun --nproc_per_node 2 --master_port 29507 \
    -m FlagEmbedding.finetune.reranker.encoder_only.base \
    --model_name_or_path /home/yanmy/models/bge-reranker-v2-m3 \
    --train_data "$TRAIN_DATA" \
    --learning_rate 1e-5 \
    --bf16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_drop_last True \
    --train_group_size 8 \
    --max_len 320 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 50 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --output_dir "$OUT" \
    --overwrite_output_dir \
    --knowledge_distillation False \
    --report_to none \
    --ddp_find_unused_parameters False \
    2>&1 | tee "$OUT/train.log"
