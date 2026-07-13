#!/usr/bin/env bash
# Open-setting smoke test: OTT-QA (strict-1690 subset), 30 questions, 3-leg RRF
# retrieval -> cell-link BM25 passages -> Qwen3.6-35B reader.
# Verifies the open pipeline runs end-to-end and scores EM/F1.
#
# Uses precomputed Stage-1 top-1 tables (analysis/ottqa_strict1690/top1_rrf.jsonl)
# so no retrieval GPU is needed for the smoke — only the reader endpoint.
#
# Prereqs:
#   * Reader served at $API_BASE (scripts/closed/start_vllm_35b.sh).
#   * data_misc.tar.gz extracted (IBM TableIR tables + HybridQA passage pool),
#     and OTT-QA traindev_tables.json from the OTT-QA repo. See data/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
for v in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do unset $v || true; done

API_BASE="${API_BASE:-http://127.0.0.1:9543/v1}"
MODEL="${MODEL:-qwen3.6-35b}"
N="${N:-30}"
METHOD="${METHOD:-rrf}"
POOL="${POOL:-analysis/open_pool_full/passages.jsonl}"        # from data_misc.tar.gz
TABLES="${TABLES:-data/ottqa_repo/data/traindev_tables.json}" # from OTT-QA repo

python -u scripts/open/eval_ottqa_per_method_e2e.py \
  --method "$METHOD" \
  --top1-jsonl "analysis/ottqa_strict1690/top1_${METHOD}.jsonl" \
  --subset analysis/ottqa_dev_strict1690.jsonl \
  --ottqa-tables "$TABLES" \
  --pool "$POOL" \
  --K-pas 20 \
  --api-base "$API_BASE" --api-key EMPTY --model "$MODEL" \
  --max-queries "$N" --concurrency 16 \
  --out-jsonl analysis/ottqa_strict1690/smoke_${METHOD}_${N}.jsonl \
  --out-summary analysis/ottqa_strict1690/smoke_${METHOD}_${N}.summary.json

echo "--- summary ---"
cat analysis/ottqa_strict1690/smoke_${METHOD}_${N}.summary.json
# Expected (30-sample smoke): stage1@1 ~0.77, EM ~0.50 (3-leg RRF baseline).
# Full 1690 baseline = EM 59.94 / F1 65.07; + table reranker top-1 = EM 65.92 / F1 71.60.
