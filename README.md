# SPARQ+: Structure-Conditioned Evidence Retrieval and Adaptive Reasoning for Offline Open-Domain Table-Text QA

Code release accompanying the paper **“SPARQ+: Structure-Conditioned
Evidence Retrieval and Adaptive Reasoning for Offline Open-Domain
Table–Text Question Answering.”**  The full-version PDF is included at the
repository root: [`SPARQ+_full_version.pdf`](./SPARQ%2B_full_version.pdf).

SPARQ+ is a cost-efficient framework for **offline** table–text QA that
handles both settings:

* **Closed setting** — the target table is given; the challenge is to
  interpret and reason over it. SPARQ+ uses a router + typed operators
  (Select Row/Col, Execute SQL/Py, RAG) + a verifier with rollback/fallback.
* **Open setting** — the relevant table *and* supporting passages must be
  discovered from a corpus first. SPARQ+ fuses lexical + dense + graph
  signals for table retrieval, converts corpus-wide passage search into a
  **budgeted cell-link neighborhood lookup**, then reasons with a
  passage-aware router and a mutual-information verifier.

All components are unified by an information-bottleneck objective that
minimizes evidence cost while preserving answer sufficiency. Everything runs
**offline** against locally served Qwen3 models (vLLM) — no online LLM API.

## Benchmarks

| Setting | Benchmark | Paper | Data source (repo · HF) |
|---|---|---|---|
| Closed | WikiTableQuestions (WikiTQ) | [Pasupat & Liang, ACL 2015](https://aclanthology.org/P15-1142/) | [GitHub](https://github.com/ppasupat/WikiTableQuestions) · [HF](https://huggingface.co/datasets/wikitablequestions) |
| Closed | TabFact | [Chen et al., ICLR 2020](https://openreview.net/forum?id=rkeJRhNYDH) | [GitHub](https://github.com/wenhuchen/Table-Fact-Checking) · [HF](https://huggingface.co/datasets/tab_fact) |
| Closed | FeTaQA | [Nan et al., TACL 2022](https://aclanthology.org/2022.tacl-1.3/) | [GitHub](https://github.com/Yale-LILY/FeTaQA) · [HF](https://huggingface.co/datasets/DongfuJiang/FeTaQA) |
| Closed | TableBench | [Wu et al., AAAI 2025](https://tablebench.github.io/) | [GitHub](https://github.com/TableBench/TableBench) · [HF](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench) |
| Closed | NIAT (numerical-inference table QA) | see paper §V | included via TableBench-style loader |
| Open | HybridQA | [Chen et al., Findings 2020](https://aclanthology.org/2020.findings-emnlp.91/) | [GitHub](https://github.com/wenhuchen/HybridQA) · [HF](https://huggingface.co/datasets/wenhu/hybrid_qa) |
| Open | OTT-QA | [Chen et al., ICLR 2021](https://openreview.net/forum?id=MmCRswl1UYl) | [GitHub](https://github.com/wenhuchen/OTT-QA) |

Models used are documented with HuggingFace links in
[`docs/MODEL_CARDS.md`](./docs/MODEL_CARDS.md). Datasets + intermediate
products are documented in [`data/README.md`](./data/README.md) and
[`docs/DATA_HANDOFF.md`](./docs/DATA_HANDOFF.md).

## Reproduction

All numbers are in the paper (`SPARQ+_full_version.pdf`). This repository
provides the code, intermediate products, and step-by-step commands to
reproduce them — see **[`docs/REPRODUCE.md`](./docs/REPRODUCE.md)** for the
exact per-benchmark commands (closed + open) and the weight-free / retrieval-free
paths. Start with the smoke tests under [`examples/`](./examples) (below).

## Repository layout

```
SPARQ_Extend_Release/
├── README.md                     ← you are here
├── SPARQ+_full_version.pdf       ← the paper (full version)
├── LICENSE                       ← MIT
├── requirements.txt
├── docs/
│   ├── MODEL_CARDS.md            ← all Qwen3 / BGE models + HF links
│   └── DATA_HANDOFF.md           ← where the large data blobs live + how code reads them
├── data/
│   └── README.md                 ← per-dataset download (repo + HF) + reviewer bundle
├── src/
│   ├── schedule_pipeline/        ← closed runners (wikitq/tabfact/fetaqa/tablebench/niat) + HybridQA + prompts
│   ├── utils/                    ← open-setting metrics/eval (hybridqa_metrics, sparq_eval, hybridqa_local)
│   ├── utils_closed/             ← vendored closed-setting utils (router, operators, SQL exec, evaluator)
│   └── prompts_closed/           ← closed-setting operator/reasoning prompts
├── scripts/
│   ├── closed/                   ← vLLM launchers + closed run scripts
│   └── open/                     ← OTT-QA stage-1 build, GNN train, rerankers, E2E readers
├── models/                       ← small trained GNN weights (bipartite query→table)
├── analysis/                     ← small intermediate products (strict-1690 subset, top-1 tables, summaries)
└── examples/                     ← smoke tests (closed + open)
```

## Quick start

```bash
# 1. Install deps + a CUDA build of vLLM (>= 0.19.1)
pip install -r requirements.txt
pip install "vllm>=0.19.1"

# 2. Serve a reader (Qwen3.6-35B-A3B-FP8) on 2 GPUs. See docs/MODEL_CARDS.md.
bash scripts/closed/start_vllm_35b.sh          # cuda 0,1 by default; edit CUDA_VISIBLE_DEVICES

# 3a. Closed smoke — WikiTQ, 20 questions
bash examples/smoke_closed_wikitq.sh

# 3b. Open smoke — OTT-QA E2E, 20 questions
bash examples/smoke_open_ottqa.sh
```

See [`examples/`](./examples) for the smoke scripts and
[`docs/DATA_HANDOFF.md`](./docs/DATA_HANDOFF.md) for the data download.

## Hardware

| Component | Recipe |
|---|---|
| Reader (35B-A3B-FP8) | 2 GPUs, ≥ 40 GB total VRAM (tested 2× A100-80G, 2× RTX 4090) |
| Retrieval (Qwen3-Embedding-0.6B / BGE) | 1 GPU ≥ 12 GB, or CPU for small runs |
| Pipeline driver | CPU + ~10 GB RAM |

The reader launcher `scripts/closed/start_vllm_35b.sh` serves
`--default-chat-template-kwargs '{"enable_thinking": false}'` (required — the
operator/SQL path reads `message.content` directly).

## License

MIT — see [`LICENSE`](./LICENSE). Please cite the upstream benchmarks and
Qwen3 models as appropriate (links above and in `docs/MODEL_CARDS.md`).
