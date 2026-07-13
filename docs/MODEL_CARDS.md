# Model Cards

SPARQ+ is **offline**: every model runs locally (vLLM for LLMs, FlagEmbedding
for retrievers). No online LLM API is used. We ship no weights — download from
HuggingFace using the links below.

## 1. Readers (LLMs, served via vLLM)

| Role | Model | HuggingFace |
|---|---|---|
| **Main reader / paper anchor** | Qwen3.6-35B-A3B-FP8 | https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8 |
| Closed router + 4B reader | Qwen3-4B-Instruct-2507 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 |
| Compressed reader (ablation) | Qwen3-30B-A3B-Instruct-2507-FP8 | https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 |

**Serving (35B, 2 GPUs).** `bash scripts/closed/start_vllm_35b.sh`
(TP=2, `--gpu-memory-utilization 0.85`, `--max-model-len 32768`,
`--kv-cache-dtype fp8`, `--reasoning-parser qwen3`, `--max-num-seqs 32`).

**Mandatory:** the launcher serves `--default-chat-template-kwargs
'{"enable_thinking": false}'`. `enable_thinking=false` is required for the
Qwen3 reader on the SPARQ pipeline — the operator/SQL path reads
`message.content` directly, which is `null` when the model emits a thinking
trace instead of a direct answer.

**Served model id.** `qwen3.6-35b`. The closed runners take `--model_name
qwen3.6-35b`; the open E2E scripts take `--model qwen3.6-35b`.

## 2. Retrievers (embedding + rerank, via FlagEmbedding)

| Role | Model | HuggingFace |
|---|---|---|
| Open-domain node/query encoder (tables, cells, passages, queries) | Qwen3-Embedding-0.6B | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B |
| Closed-setting dense leg + embedding | BGE-M3 | https://huggingface.co/BAAI/bge-m3 |
| Table & passage reranker (fine-tuned in this work) | BGE-reranker-v2-m3 (base) | https://huggingface.co/BAAI/bge-reranker-v2-m3 |

The fine-tuned reranker checkpoints (OTT-QA / HybridQA table & passage
rerankers) and pre-computed embeddings are **not required** for a reviewer to
reproduce the main numbers — the retrieval outputs they produce are shipped as
intermediate products (`analysis/…/top1_*.jsonl`, per-method predictions). See
[`../data/README.md`](../data/README.md). To retrain them from scratch, use
`scripts/open/prepare_ottqa_*_reranker_data.py` +
`scripts/open/finetune_ottqa_*_reranker.sh` on the `bge-reranker-v2-m3` base.

Small trained **GNN** weights (bipartite query→table encoder, ~8–26 MB) are
included in git under `models/` — these are the only trained weights shipped
in-repo.

## 3. Hardware

| Model | Requirement |
|---|---|
| Qwen3.6-35B-A3B-FP8 | 2 GPUs, ≥ 40 GB total VRAM (2× A100-80G or 2× RTX 4090) |
| Qwen3-4B-Instruct-2507 | 1 GPU ≥ 16 GB |
| Qwen3-Embedding-0.6B / BGE | 1 GPU ≥ 12 GB, or CPU for small runs |
