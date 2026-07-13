#!/usr/bin/env python3
"""Agent-integration OTT-QA E2E (strict-1690), SPARQ-X-aligned.

A ReAct-light agent uses a table-retrieval TOOL `retrieve_tables(query)`
(ONE tool call by design), takes the top-1 table, then the FIXED SPARQ-X
downstream answers:
    top-1 table -> cell-link passages (BM25 top-K in-table) -> 35B CoT reader.

The retrieve_tables BACKEND is swapped across:
    {structir, rrf (=SPARQ-X-own 3-leg), bm25, dense}
Everything downstream (passage stage, reader prompt, decoding) is IDENTICAL,
reused VERBATIM from scripts/eval_ottqa_per_method_e2e.py.

FULL per-query trajectory is stored:
    {question_id, question, gold_answer, tool, retrieved_table_id,
     stage1_correct, cell_link_passages_used, reader_prompt, reader_answer,
     EM, F1}
-> results/agent_integration/ottqa_e2e/<tool>.jsonl

Top-1 table per tool comes from precomputed
    analysis/ottqa_strict1690/top1_<backend>.jsonl
(backend map: structir->structir, sparqx->rrf, bm25->bm25, dense->dense).
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import utils.hybridqa_metrics as _hyb
parse_answer = _hyb.parse_answer
from utils.sparq_eval import sparq_em_f1

# --- VERBATIM reader prompt + table render + bm25 from eval_ottqa_per_method_e2e.py ---
PROMPT_TEMPLATE = (REPO_ROOT / "src/schedule_pipeline/hybridqa_qa_cot.txt").read_text() + """

Now answer:
Question: {question}
Selected Table:
{table_text}
SQL Result:
(no SQL run; rely on Selected Table and Linked Passages directly)
Linked Passages:
{passages_text}
"""


def render_table(t: dict, h_names: list[str], max_rows: int = 30,
                  cell_chars: int = 200) -> str:
    title = (t.get("title") or "")[:300]
    section = (t.get("section_title") or "")[:200]
    caption = (f"# {title}{' — ' + section if section else ''}" if title else "").strip()
    lines = []
    if caption: lines.append(caption)
    lines.append("col : row_id | " + " | ".join(h_names))
    for r_idx, row in enumerate(t.get("data", [])):
        if r_idx >= max_rows: break
        cell_strs = [str(r_idx)]
        for cell in row:
            if isinstance(cell, list) and len(cell) >= 2:
                cell_strs.append(str(cell[0])[:cell_chars])
            else:
                cell_strs.append(str(cell)[:cell_chars])
        lines.append(f"row {r_idx} : " + " | ".join(cell_strs))
    return "\n".join(lines)


def bm25_rank(query: str, candidates: list[tuple[str, str]]) -> list[int]:
    if not candidates: return []
    from rank_bm25 import BM25Okapi
    tok = lambda s: re.findall(r"\w+", s.lower())
    docs = [tok(anchor + " " + text) for anchor, text in candidates]
    if not any(docs): return list(range(len(candidates)))
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(tok(query))
    return sorted(range(len(candidates)), key=lambda i: -scores[i])


async def call_llm(client, model, prompt, max_tokens=2048):
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return r.choices[0].message.content or ""
    except Exception as exc:
        return f"__ERROR__: {exc!r}"


# backend label -> top1 jsonl basename
BACKEND_TOP1 = {
    "structir": "top1_structir.jsonl",
    "sparqx":   "top1_rrf.jsonl",     # SPARQ-X-own 3-leg RRF retrieval
    "bm25":     "top1_bm25.jsonl",
    "dense":    "top1_dense.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, choices=list(BACKEND_TOP1.keys()))
    ap.add_argument("--subset", default="analysis/ottqa_dev_strict1690.jsonl", type=Path)
    ap.add_argument("--ottqa-tables",
                     default="data/ottqa_repo/data/traindev_tables.json", type=Path)
    ap.add_argument("--pool", default="analysis/open_pool_full/passages.jsonl", type=Path)
    ap.add_argument("--K-pas", type=int, default=20)
    ap.add_argument("--passage-chars", type=int, default=2500)
    ap.add_argument("--api-base", default="http://192.168.12.43:9543/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="qwen3.6-35b")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--outdir", default="results/agent_integration/ottqa_e2e", type=Path)
    args = ap.parse_args()

    for k in ("http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
        os.environ.pop(k, None)

    tool = args.tool
    top1_jsonl = Path("analysis/ottqa_strict1690") / BACKEND_TOP1[tool]
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.outdir / f"{tool}.jsonl"

    t0 = time.time()
    subset = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    if args.max_queries: subset = subset[: args.max_queries]

    top1_rows = [json.loads(l) for l in top1_jsonl.read_text().splitlines() if l.strip()]
    qid_to_top1 = {r["qid"]: r["top1_tid"] for r in top1_rows}
    print(f"[{tool}] subset={len(subset)} top1_jsonl={top1_jsonl.name} ({len(top1_rows)})", flush=True)

    # resume: skip qids already present (and valid) in out_jsonl
    done_qids = set()
    if out_jsonl.exists():
        for line in out_jsonl.read_text().splitlines():
            if not line.strip(): continue
            try:
                rr = json.loads(line)
            except Exception:
                continue
            # only treat as done if it has a real reader answer (not a transient error)
            if rr.get("retrieved_table_id") is not None and not (rr.get("error") and str(rr.get("error")).startswith("__ERROR__")):
                done_qids.add(rr["question_id"])
    if done_qids:
        before = len(subset)
        subset = [r for r in subset if r["question_id"] not in done_qids]
        print(f"[{tool}] RESUME: {len(done_qids)} already done, {len(subset)}/{before} remaining", flush=True)

    print(f"[{tool}] load ottqa traindev tables ...", flush=True)
    ottqa_tables = json.loads(args.ottqa_tables.read_text())
    pool = {}
    with args.pool.open() as f:
        for line in f:
            if line.strip():
                rr = json.loads(line)
                pid = (rr.get("pid") or "").lower()
                if pid: pool[pid] = (rr.get("summary") or "").strip()
    print(f"[{tool}] ottqa_tables={len(ottqa_tables)} pool={len(pool)}", flush=True)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.api_base.rstrip("/"),
                         api_key=args.api_key, timeout=180)
    sem = asyncio.Semaphore(args.concurrency)

    fout = out_jsonl.open("a")
    em_sum = f1_sum = 0.0; n_seen = n_err = 0; n_s1_correct = 0
    progress = 0; lock = asyncio.Lock()

    async def _one(r):
        nonlocal em_sum, f1_sum, n_seen, n_err, n_s1_correct, progress
        async with sem:
            qid = r["question_id"]; q = r["question"]; gold = r["answer_text"]
            gold_tid = r["table_id"]
            # === AGENT: single retrieve_tables tool call (backend swapped) ===
            top1_tid = qid_to_top1.get(qid)
            if top1_tid is None:
                async with lock:
                    n_err += 1; progress += 1
                    fout.write(json.dumps({
                        "question_id": qid, "question": q, "gold_answer": gold,
                        "tool": tool, "retrieved_table_id": None,
                        "stage1_correct": False, "cell_link_passages_used": [],
                        "reader_prompt": None, "reader_answer": None,
                        "EM": 0.0, "F1": 0.0, "kind": r["kind"],
                        "error": "no top1 for qid",
                    }, ensure_ascii=False) + "\n"); fout.flush()
                return
            s1_correct = (top1_tid == gold_tid)

            # === FIXED DOWNSTREAM (verbatim SPARQ-X stages) ===
            t_ottqa = ottqa_tables.get(top1_tid, {})
            h_names = [h[0] if isinstance(h, list) else str(h)
                        for h in t_ottqa.get("header", [])]
            table_text = render_table(t_ottqa, h_names)

            cand_records = []; seen = set()
            for row in t_ottqa.get("data", []):
                for cell in row:
                    if isinstance(cell, list) and len(cell) >= 2:
                        for u in (cell[1] or []):
                            pid = u.lower()
                            if pid in seen or pid not in pool: continue
                            seen.add(pid)
                            cand_records.append((pid, str(cell[0])[:60], pool[pid]))
            cand_anchors = [(c[1], c[2][: args.passage_chars]) for c in cand_records]
            order = bm25_rank(q, cand_anchors)[: args.K_pas]
            chosen = [cand_records[k] for k in order]
            passages_text = "\n".join(
                f"[Passage {kk+1}] pid={cr[0][:50]} cell={cr[1]}\n{cr[2][: args.passage_chars]}"
                for kk, cr in enumerate(chosen)
            ) or "(none)"
            prompt = PROMPT_TEMPLATE.format(
                table_text=table_text, passages_text=passages_text, question=q)
            raw = await call_llm(client, args.model, prompt)

        err = raw.startswith("__ERROR__")
        pred = parse_answer(raw) if not err else ""
        em, f1 = sparq_em_f1(pred, gold, question=q)
        passages_used = [{"pid": cr[0], "cell": cr[1]} for cr in chosen]
        async with lock:
            fout.write(json.dumps({
                "question_id": qid, "question": q, "gold_answer": gold,
                "tool": tool, "retrieved_table_id": top1_tid, "gold_table_id": gold_tid,
                "stage1_correct": bool(s1_correct),
                "n_cands": len(cand_records),
                "cell_link_passages_used": passages_used,
                "reader_prompt": prompt,
                "reader_answer": pred,
                "reader_raw": raw[:1500],
                "EM": float(em), "F1": round(f1, 4),
                "kind": r["kind"],
                "error": raw if err else None,
            }, ensure_ascii=False) + "\n"); fout.flush()
            if err: n_err += 1
            else:
                em_sum += em; f1_sum += f1; n_seen += 1
                if s1_correct: n_s1_correct += 1
            progress += 1
            if progress % 100 == 0:
                el = time.time() - t0
                print(f"[{tool}] {progress}/{len(subset)} EM={em_sum/max(1,n_seen):.4f} "
                      f"F1={f1_sum/max(1,n_seen):.4f} s1={n_s1_correct/max(1,n_seen):.4f} "
                      f"err={n_err} {el:.0f}s", flush=True)

    async def _run():
        await asyncio.gather(*[_one(r) for r in subset])
    asyncio.run(_run())
    fout.close()

    # summary: recompute everything from the file (dedup by qid, keep last valid)
    full_subset = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    if args.max_queries: full_subset = full_subset[: args.max_queries]
    n_total = len(full_subset)
    rec_by_qid = {}
    for line in out_jsonl.read_text().splitlines():
        if not line.strip(): continue
        rr = json.loads(line)
        if rr.get("retrieved_table_id") is None:
            continue  # skip no-top1 / transient placeholders
        rec_by_qid[rr["question_id"]] = rr
    recs = list(rec_by_qid.values())
    n_seen = len(recs); n_err = n_total - n_seen
    n_s1_correct = sum(1 for r in recs if r["stage1_correct"])
    em_sum = sum(r["EM"] for r in recs); f1_sum = sum(r["F1"] for r in recs)
    by_s1 = {"correct": [r for r in recs if r["stage1_correct"]],
             "wrong":   [r for r in recs if not r["stage1_correct"]]}
    by_kind = {}
    for r in recs: by_kind.setdefault(r["kind"], []).append(r)
    def agg(rows):
        if not rows: return {"n": 0, "EM": 0.0, "F1": 0.0}
        return {"n": len(rows),
                "EM": sum(r["EM"] for r in rows)/len(rows),
                "F1": sum(r["F1"] for r in rows)/len(rows)}
    summary = {
        "tool": tool, "model": args.model, "top1_source": top1_jsonl.name,
        "n_total": n_total, "n_scored": n_seen, "n_errors": n_err,
        "n_stage1_correct": n_s1_correct,
        "table_R@1": n_s1_correct / max(1, n_total),
        "EM": em_sum / max(1, n_seen), "F1": f1_sum / max(1, n_seen),
        "EM_given_stage1_correct": agg(by_s1["correct"])["EM"],
        "F1_given_stage1_correct": agg(by_s1["correct"])["F1"],
        "by_kind": {k: agg(v) for k, v in by_kind.items()},
        "by_stage1_correct": {"correct": agg(by_s1["correct"]),
                                "wrong": agg(by_s1["wrong"])},
        "wall_sec": round(time.time() - t0, 1),
    }
    (args.outdir / f"{tool}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== AGENT-E2E [{tool}] ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
