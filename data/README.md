# Data

SPARQ+ uses 7 public benchmarks. We **do not redistribute raw datasets** —
download each from its upstream source (repo / HuggingFace) below. For
convenience we ship the **intermediate products** needed to reproduce the
headline numbers without rerunning retrieval/training (see §Reviewer bundle).

## 1. Datasets (download from upstream)

| Setting | Benchmark | Repo | HuggingFace |
|---|---|---|---|
| Closed | WikiTQ | https://github.com/ppasupat/WikiTableQuestions | https://huggingface.co/datasets/wikitablequestions |
| Closed | TabFact | https://github.com/wenhuchen/Table-Fact-Checking | https://huggingface.co/datasets/tab_fact |
| Closed | FeTaQA | https://github.com/Yale-LILY/FeTaQA | https://huggingface.co/datasets/DongfuJiang/FeTaQA |
| Closed | TableBench | https://github.com/TableBench/TableBench | https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench |
| Open | HybridQA | https://github.com/wenhuchen/HybridQA | https://huggingface.co/datasets/wenhu/hybrid_qa |
| Open | OTT-QA | https://github.com/wenhuchen/OTT-QA | (tables + released_data in the repo) |

The closed-setting loaders read WikiTQ/TabFact/FeTaQA/TableBench via
🤗 `datasets`; set `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` once cached.

## 2. Intermediate products (this repo, in git)

Small artifacts shipped in-repo so you can reproduce the main numbers directly:

- `analysis/ottqa_dev_strict1690.jsonl` — the 1690 strict-recoverable OTT-QA
  dev subset (paper's open-setting evaluation set).
- `analysis/ottqa_strict1690/top1_{bm25,dense,gnn,rrf,structir}.jsonl` —
  Stage-1 top-1 retrieved table per query, per retrieval leg (feeds the E2E
  reader directly, so Stage-1 need not be rerun).
- `analysis/ottqa_strict1690/*.summary.json` — headline E2E / oracle summaries.
- `models/ottqa_query_table_gnn_v2_instruct/`, `models/hybridqa_table_gnn_qwen3_instruct_v3/`
  — trained bipartite query→table GNN weights (Stage-1 graph leg).

## 3. Large data blobs (Reviewer bundle — off-git)

Two big blobs are **not in git**. For reviewers they contain **only data +
intermediate products (no model weights)** and are provided as a download
bundle (Google Drive; a maintainer copy with weights also lives on the PI's
Quark netdisk). Full details + exact code consumers in
[`../docs/DATA_HANDOFF.md`](../docs/DATA_HANDOFF.md).

| Blob | Size | Belongs to | What |
|---|---:|---|---|
| `data_misc.tar.gz` | 227 MB | both | HybridQA raw (`WikiTables-WithLinks*.zip` + train/dev/test) + IBM TableIR table pool (`corpus_structure.jsonl`, 8891 tables) + reranker training sets. **Does NOT contain the passage pools below.** |
| `open_pool/passages.jsonl` | 73 MB | **HybridQA** | HybridQA full-dev linked-passage pool — **75,642 passages** (3,053 dev tables). This is the *complete* HybridQA passage pool; there is no larger one. |
| `open_pool_full/passages.jsonl` | 232 MB | **OTT-QA** | OTT-QA passage pool — **240,042 passages** (8,891 tables). Used by the strict-1690 E2E. |
| `ottqa_all_passages/all_passages.json` | 3.2 GB | **OTT-QA only** | OTT-QA official **5.96M** Wikipedia passage corpus, streamed with `ijson`. **Only OTT-QA uses this**; needed for the OTT-QA full-dev-2214 EM 0.676 run. HybridQA never uses it. |

**Passage-pool matrix (which pool each dataset uses):**

| Dataset | Complete open-domain pool | Result |
|---|---|---|
| HybridQA (n=3466) | `open_pool/passages.jsonl` — **75,642** | EM 0.508 / F1 0.549 |
| OTT-QA strict-1690 | `open_pool_full/passages.jsonl` — **240,042** | EM 65.92 (reranker top-1) |
| OTT-QA full-dev (n=2214) | `all_passages.json` — **5.96M** | EM 0.676 / F1 0.732 |

Restore: `tar -xzf data_misc.tar.gz` at the repo root. **Note:** the passage
pools are *not* inside `data_misc.tar.gz` — they are separate intermediate
products (in the reviewer data bundle). You do **not** need any pool to
reproduce the headline EM/F1: the frozen per-query reader inputs
(`analysis/reader_prompts/sparqx_cot_{hybridqa,ottqa}.jsonl.gz`, see
[`../docs/REPRODUCE.md`](../docs/REPRODUCE.md)) already bake in the correct
passages. Pools are only needed to re-run retrieval from scratch.

## 4. Final layout the code expects

```
SPARQ_Extend_Release/
├── data/
│   ├── hybridqa_raw/            # train/dev/test.json + WikiTables-WithLinks*.zip  (from data_misc.tar.gz)
│   ├── ottqa_raw_full/          # IBM TableIR: corpus_structure.jsonl (8891 tables) + parquet/qrels
│   ├── ottqa_repo/data/traindev_tables.json   # OTT-QA tables (from the OTT-QA repo)
│   └── ottqa_all_passages/all_passages.json   # 3.2 GB, 5.96M passages — OTT-QA full-dev only
└── analysis/
    ├── open_pool/passages.jsonl       # 75,642 — HybridQA passage pool
    └── open_pool_full/passages.jsonl  # 240,042 — OTT-QA strict-1690 passage pool
```
