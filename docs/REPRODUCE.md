# Reproduction Guide

This guide gives the exact, verified commands to reproduce the SPARQ+ headline
numbers, for both settings. Hardware differs across sites, so treat these as
**main-number** reproductions (the pipeline + intermediate products are
deterministic; the reader is an LLM, so tiny variance is expected).

All commands assume the reader is served (see below) and proxies are unset for
localhost calls (`unset http_proxy https_proxy all_proxy`).

## 0. Serve the reader

```bash
# 2 GPUs. Edit GPUS/PORT as needed. Serves qwen3.6-35b at :9543.
GPUS=0,1 PORT=9543 bash scripts/closed/start_vllm_35b.sh
# health: curl -s --noproxy '*' http://127.0.0.1:9543/v1/models
```

The launcher serves `--default-chat-template-kwargs '{"enable_thinking":
false}'` — **required**. The 35B is thinking-capable; the SPARQ operator/SQL
path reads `message.content` directly, which is `null` if the model emits a
thinking trace. Do **not** add `--enforce-eager` to the formal config (it is a
smoke-only low-memory shortcut).

## 1. Closed setting (WikiTQ / TabFact / FeTaQA / TableBench / NIAT)

Closed runners live in `src/schedule_pipeline/` and load retrieval models
(BGE-M3 + H-STAR router/check rerankers) locally. For a small smoke you can
force those to CPU with `CUDA_VISIBLE_DEVICES=""`.

**Smoke (WikiTQ, 20 q):**
```bash
ROUTER=/path/to/H-STAR/router/wikitq CHECK=/path/to/H-STAR/check/wikitq \
API_BASE=http://127.0.0.1:9543/v1 N=20 bash examples/smoke_closed_wikitq.sh
# verified: Accuracy 95.00% (19/20); full 13-step pipeline runs clean.
```

**Full-set headline numbers (Qwen3.6-35B reader):**

| Dataset | Runner (`src/schedule_pipeline/`) | Split | Metric | 35B |
|---|---|---|---|---:|
| WikiTQ | `run_full_pipeline_wikitq_api.py --dataset_name wikitq` | test (4344) | EM | **84.55** |
| TabFact | `run_full_pipeline_wikitq_api.py --dataset_name tab_fact` | test_small (2024) | acc | **93.82** |
| FeTaQA | `run_full_pipeline_fetaqa_api.py` | test (2003) | ROUGE-L | **0.5036** |
| TableBench | `run_pipeline_tablebench_pot_direct.py` | (886) | ROUGE-L | **0.4671** |
| NIAT | `run_pipeline_niat_pot_direct.py` | (2932) | EM | **77.86** |

Scoring reuses the repo evaluator verbatim
(`src/utils_closed/evaluator.py`, `prompt_generate.evaluate_predictions`).

## 2. Open setting (OTT-QA, strict-1690 subset)

Stage-1 top-1 tables are precomputed per retrieval leg
(`analysis/ottqa_strict1690/top1_{bm25,dense,gnn,rrf}.jsonl`), so the E2E
reader can be run without rerunning retrieval.

**Smoke (OTT-QA RRF, 30 q):**
```bash
API_BASE=http://127.0.0.1:9543/v1 N=30 METHOD=rrf bash examples/smoke_open_ottqa.sh
# verified: stage1@1 0.767, EM 0.50 / F1 0.518 (30-q, 3-leg RRF baseline, 0 errors).
```

**Weight-free reranker path (reproduce EM 65.92 without the 15 GB reranker):**
The table-reranker top-K output is shipped at
`analysis/ottqa_strict1690/reranked_cands_v1.jsonl`, so you can get the reranked
top-1 table without loading the reranker checkpoint:
```bash
python scripts/open/eval_e2e_1690.py \
  --subset analysis/ottqa_dev_strict1690.jsonl \
  --ottqa-tables data/ottqa_repo/data/traindev_tables.json \
  --pool analysis/open_pool_full/passages.jsonl \
  --reranked-cands-jsonl analysis/ottqa_strict1690/reranked_cands_v1.jsonl \
  --K-pas 30 --api-base http://127.0.0.1:9543/v1 --api-key EMPTY --model qwen3.6-35b \
  --tag reranker_v1_top1 --out-dir analysis/ottqa_strict1690
# -> EM 65.92 / F1 71.60
```

**Full 1690 E2E:**
```bash
python scripts/open/eval_ottqa_per_method_e2e.py --method rrf \
  --top1-jsonl analysis/ottqa_strict1690/top1_rrf.jsonl \
  --subset analysis/ottqa_dev_strict1690.jsonl \
  --ottqa-tables data/ottqa_repo/data/traindev_tables.json \
  --pool analysis/open_pool_full/passages.jsonl \
  --K-pas 20 --api-base http://127.0.0.1:9543/v1 --api-key EMPTY --model qwen3.6-35b \
  --concurrency 32
```

| Config | EM | F1 |
|---|---:|---:|
| 3-leg RRF top-1 (no reranker) | 59.94 | 65.07 |
| + table reranker top-1 (`models/ottqa_table_reranker_v1`, off-git) | **65.92** | **71.60** |
| Oracle ceiling (gold table + BM25 top-30) | 74.44 | 79.68 |

Stage-1 accuracy is the bottleneck: conditional `EM | stage1 correct ≈ 0.74`
≈ the oracle ceiling, i.e. Stages 2–4 are saturated.

Full-dev OTT-QA (2214) with the 5.96M `all_passages.json` pool: **EM 0.6762 /
F1 0.7315** (see `scripts/open/export_sparqx_evidence_for_helios.py` and
`docs/DATA_HANDOFF.md`).

## 3. HybridQA (open, anchored table)

```bash
python src/schedule_pipeline/run_full_pipeline_hybridqa.py \
  --hybridqa-raw-dir data/hybridqa_raw --split validation --first-n 100 \
  --api-base http://127.0.0.1:9543/v1 --model qwen3.6-35b
```

Headline: 35B + passage reranker top-20 = **EM 73.92** (beats full-context
teacher 69.40 at ~37% less context).

## Notes

* GPU policy for the reference environment: only specific GPUs are permitted
  per host; the launcher does not pin GPUs beyond `CUDA_VISIBLE_DEVICES` — set
  it to free devices and never contend with other users' processes.
* A comprehensive map of every script/artifact in the source workspace is in
  [`REPRODUCTION_MAP.md`](./REPRODUCTION_MAP.md).
