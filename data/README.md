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

| Blob | Size | What |
|---|---:|---|
| `data_misc.tar.gz` | 227 MB | HybridQA raw (`WikiTables-WithLinks*.zip` + train/dev/test) + IBM TableIR table pool (`corpus_structure.jsonl`, 8891 tables) + reranker training sets |
| `ottqa_all_passages/all_passages.json` | 3.2 GB | OTT-QA official 5.96M Wikipedia passage pool (open-domain leg; streamed with `ijson`) |

Restore: `tar -xzf data_misc.tar.gz` at the repo root (paths are already
`data/…`). The strict-1690 OTT-QA E2E smoke needs only `data_misc.tar.gz`
(IBM TableIR tables + HybridQA passage pool); the 3.2 GB `all_passages.json`
is only needed for the full-dev 2214 open-domain run.

## 4. Final layout the code expects

```
SPARQ_Extend_Release/
└── data/
    ├── hybridqa_raw/            # train/dev/test.json + WikiTables-WithLinks*.zip  (from data_misc.tar.gz)
    ├── ottqa_raw_full/          # IBM TableIR: corpus_structure.jsonl (8891 tables) + parquet/qrels
    ├── ottqa_repo/data/traindev_tables.json   # OTT-QA tables (from the OTT-QA repo)
    └── ottqa_all_passages/all_passages.json   # 3.2 GB, 5.96M passages (full-dev only)
```
