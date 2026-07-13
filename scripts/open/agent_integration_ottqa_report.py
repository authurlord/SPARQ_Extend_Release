#!/usr/bin/env python3
"""Build the agent-integration OTT-QA E2E summary + markdown report."""
import json
from pathlib import Path

OUTDIR = Path("results/agent_integration/ottqa_e2e")
REF_EM, REF_F1, REF_R1, REF_CONDEM = 0.6592, 0.716, 0.8604, 0.7407
TOOLS = ["structir", "sparqx", "bm25", "dense"]
LABEL = {"structir": "StructIR", "sparqx": "SPARQ-X-own (RRF)",
         "bm25": "BM25", "dense": "dense (bge-m3)"}

summ = {}
for t in TOOLS:
    p = OUTDIR / f"{t}.summary.json"
    if p.exists():
        summ[t] = json.loads(p.read_text())

agg = {"reference_sparqx_reranker_top1": {
        "table_R@1": REF_R1, "EM": REF_EM, "F1": REF_F1,
        "EM_given_stage1_correct": REF_CONDEM,
        "source": "analysis/ottqa_strict1690/e2e_reranker_v1_top1.summary.json"},
       "tools": summ}
(OUTDIR / "summary.json").write_text(json.dumps(agg, indent=2))

lines = []
lines.append("# OTT-QA Agent-Integration E2E (strict-1690)\n")
lines.append("SPARQ-X-aligned: an agent calls `retrieve_tables(query)` (one tool call), "
             "takes top-1 table, then the FIXED SPARQ-X downstream answers "
             "(top-1 table -> cell-link passages BM25 top-20 in-table -> 35B CoT reader). "
             "Only the `retrieve_tables` backend is swapped; everything downstream is identical "
             "(verbatim from `scripts/eval_ottqa_per_method_e2e.py`).\n")
lines.append(f"**Reference** = SPARQ-X own table-reranker top-1 pipeline: "
             f"R@1 {REF_R1:.4f}, E2E EM **{REF_EM:.4f}** / F1 {REF_F1:.4f}, "
             f"EM|stage1✓ {REF_CONDEM:.4f} "
             f"(`analysis/ottqa_strict1690/e2e_reranker_v1_top1.summary.json`).\n")
lines.append("| tool | table R@1 | E2E EM | F1 | EM\\|stage1✓ | n | err | ΔEM vs ref |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for t in TOOLS:
    if t not in summ: continue
    s = summ[t]
    dEM = s["EM"] - REF_EM
    lines.append(f"| {LABEL[t]} | {s['table_R@1']:.4f} | **{s['EM']:.4f}** | "
                 f"{s['F1']:.4f} | {s['EM_given_stage1_correct']:.4f} | "
                 f"{s['n_scored']} | {s['n_errors']} | {dEM:+.4f} |")
lines.append(f"| _SPARQ-X reranker (ref)_ | {REF_R1:.4f} | {REF_EM:.4f} | {REF_F1:.4f} | "
             f"{REF_CONDEM:.4f} | 1690 | 0 | — |")
lines.append("")

# honest reading
if "structir" in summ:
    si = summ["structir"]
    lines.append("## Honest reading\n")
    gap = REF_EM - si["EM"]
    lines.append(f"- StructIR-as-tool E2E EM = **{si['EM']:.4f}** vs SPARQ-X reference {REF_EM:.4f} "
                 f"(gap {gap:+.4f}).")
    lines.append(f"- StructIR table R@1 on strict-1690 = {si['table_R@1']:.4f} "
                 f"(full-2214 R@1 = 83.06, sanity-verified). SPARQ-X reranker R@1 = {REF_R1:.4f}.")
    lines.append(f"- Conditional EM|stage1✓ = {si['EM_given_stage1_correct']:.4f} "
                 f"vs reference {REF_CONDEM:.4f}: this isolates the SHARED downstream. "
                 f"If close, the E2E gap is driven by table-retrieval R@1, not the reader.")
    verdict = ("StructIR-as-tool APPROACHES SPARQ-X" if si["EM"] >= 0.60
               else "StructIR-as-tool is FAR BELOW SPARQ-X (investigate)")
    lines.append(f"- **Verdict: {verdict}.**")

(OUTDIR / "OTTQA_AGENT_E2E.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
