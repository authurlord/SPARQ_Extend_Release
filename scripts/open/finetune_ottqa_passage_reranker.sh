#!/bin/bash
# Fine-tune bge-reranker-v2-m3 on OTT-QA passage reranker training data.
# Reuses HybridQA FlagEmbedding pipeline (same format), only data path differs.
# Per user 2026-05-20: cuda 0,1 on 51.10 (free), authorized.

set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1

OUT=models/ottqa_passage_reranker_v1
TRAIN_DATA=data/ottqa_passage_reranker_training/train.jsonl

mkdir -p "$OUT"

torchrun --nproc_per_node 2 --master_port 29505 \
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
    --max_len 256 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 50 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --gradient_checkpointing \
    --output_dir "$OUT" \
    --overwrite_output_dir \
    --knowledge_distillation False \
    --report_to none \
    --ddp_find_unused_parameters False \
    2>&1 | tee "$OUT/train.log"
