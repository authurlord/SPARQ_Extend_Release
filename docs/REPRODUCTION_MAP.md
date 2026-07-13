# Script Map

Where each piece of the pipeline lives in this release. See
[`REPRODUCE.md`](./REPRODUCE.md) for the run commands.

## Readers / servers — `scripts/closed/`

| Script | Serves |
|---|---|
| `start_vllm_35b.sh` | **Recommended.** Qwen3.6-35B-A3B-FP8, TP=2, `enable_thinking=false` default (portable) |
| `start_vllm_qwen36_35b_12_43_cuda67.sh` | Reference 2×A100/4090 launch (authoritative config used for the paper) |
| `start_vllm_35b_51_10_3090x2_cuda23.sh` | Reference 2×RTX-3090 launch |
| `start_vllm_bge_m3_12_43_cuda7.sh`, `start_vllm_qwen3_embed_local.sh` | Embedders (BGE-M3, Qwen3-Embedding-0.6B) |

## Closed setting — `src/schedule_pipeline/` + `scripts/closed/`

| Dataset | Runner (`src/schedule_pipeline/`) | Wrapper (`scripts/closed/`) |
|---|---|---|
| WikiTQ / TabFact | `run_full_pipeline_wikitq_api.py` | `run_wikitq.sh`, `run_tabfact.sh` |
| FeTaQA | `run_full_pipeline_fetaqa_api.py` | `run_fetaqa.sh`, `run_fetaqa_35b.sh` |
| TableBench | `run_pipeline_tablebench_pot_direct.py` | `run_tablebench.sh` |
| NIAT | `run_pipeline_niat_pot_direct.py` | `run_niat.sh` |

Shared closed utils: `src/utils_closed/` (router, operators, SQL exec via
`multi_db_v2.py`, `evaluator.py`, `prompt_generate.py`). Prompts:
`src/prompts_closed/`.

## Open setting — `scripts/open/`

**OTT-QA (4-stage):**
- Stage-1 build: `build_ottqa_pool_metadata.py`, `embed_ottqa_all_via_vllm.py`,
  `embed_ottqa_cells_via_vllm.py`, `build_ottqa_bipartite_graph.py`,
  `train_query_table_gnn.py`, `eval_ottqa_gnn_table_recall.py`
- Rerankers: `prepare_ottqa_{table,passage}_reranker_data.py` +
  `finetune_ottqa_{table,passage}_reranker.sh`
- **E2E readers:** `eval_ottqa_per_method_e2e.py` (per retrieval leg),
  `eval_e2e_1690.py` (baseline + reranker), `agent_integration_ottqa_e2e.py`
  (ReAct-light), `eval_strict1690_oracle.py` (ceiling)
- Full-dev 5.96M pool export: `export_sparqx_evidence_for_helios.py`

**HybridQA:**
- `src/schedule_pipeline/run_full_pipeline_hybridqa.py` (main runner)
- `eval_hybridqa_open_e2e.py`, `eval_hybridqa_stage1_3leg_v2.py`

Shared open utils: `src/utils/` (`hybridqa_metrics.py`, `sparq_eval.py`,
`hybridqa_local.py` — reads the WikiTables zip in place).

## Shipped artifacts

- `models/ottqa_query_table_gnn_v2_instruct/`, `models/hybridqa_table_gnn_qwen3_instruct_v3/`
  — trained GNN weights (in git).
- `analysis/ottqa_dev_strict1690.jsonl`, `analysis/ottqa_strict1690/top1_*.jsonl`,
  `*.summary.json` — Stage-1 outputs + headline summaries (in git).
- Reranker checkpoints, embeddings, passage pools, the 5.96M passage corpus —
  off-git; see [`DATA_HANDOFF.md`](./DATA_HANDOFF.md).
