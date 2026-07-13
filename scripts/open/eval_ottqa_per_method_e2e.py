#!/usr/bin/env python3
"""OTT-QA strict-1690 per-method E2E QA driver.

For ONE retrieval method (bm25 / dense / gnn / rrf):
  1. Read top-1 table id per query from precomputed top1_<method>.jsonl
  2. Stage 2: cell-link union of that 1 table → BM25 → top-K_pas passages
  3. Stage 3: 35B reader (Qwen3.6-35B-A3B-FP8) + hybridqa_qa_cot.txt prompt
  4. Per-row EM/F1 + summary

Inputs assume `compute_ottqa_per_method_top1.py` has produced
analysis/ottqa_strict1690/top1_<method>.jsonl.

Reader call mirrors eval_strict1690_hier.py exactly (same prompt template,
same passage rendering, same parse_answer + sparq_em_f1). The ONLY
difference is K_tab=1 fixed to the supplied top1_tid (zero overlap with
the anchored 65.92 reranker pipeline, since that uses table reranker output).
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, sys, time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import utils.hybridqa_metrics as _hyb
parse_answer = _hyb.parse_answer
from utils.sparq_eval import sparq_em_f1


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["bm25","dense","gnn","rrf"])
    ap.add_argument("--top1-jsonl", default=None, type=Path,
                     help="defaults to analysis/ottqa_strict1690/top1_<method>.jsonl")
    ap.add_argument("--subset", default="analysis/ottqa_dev_strict1690.jsonl", type=Path)
    ap.add_argument("--ottqa-tables",
                     default="data/ottqa_repo/data/traindev_tables.json", type=Path)
    ap.add_argument("--pool", default="analysis/open_pool_full/passages.jsonl", type=Path)
    ap.add_argument("--K-pas", type=int, default=20)
    ap.add_argument("--passage-chars", type=int, default=2500)
    ap.add_argument("--api-base", default="http://192.168.12.43:9543/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--out-jsonl", default=None, type=Path)
    ap.add_argument("--out-summary", default=None, type=Path)
    args = ap.parse_args()

    for k in ("http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
        os.environ.pop(k, None)

    method = args.method
    if args.top1_jsonl is None:
        args.top1_jsonl = Path(f"analysis/ottqa_strict1690/top1_{method}.jsonl")
    if args.out_jsonl is None:
        args.out_jsonl = Path(f"analysis/ottqa_strict1690/per_method_{method}_35b.jsonl")
    if args.out_summary is None:
        args.out_summary = Path(f"analysis/ottqa_strict1690/per_method_{method}_35b.summary.json")
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    subset = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    if args.max_queries: subset = subset[: args.max_queries]
    qid_to_subset = {r["question_id"]: r for r in subset}

    top1_rows = [json.loads(l) for l in args.top1_jsonl.read_text().splitlines() if l.strip()]
    qid_to_top1 = {r["qid"]: r["top1_tid"] for r in top1_rows}
    print(f"[load] subset={len(subset)} top1_jsonl={len(top1_rows)}", flush=True)

    print(f"[load] ottqa traindev tables ...", flush=True)
    ottqa_tables = json.loads(args.ottqa_tables.read_text())
    pool = {}
    with args.pool.open() as f:
        for line in f:
            if line.strip():
                rr = json.loads(line)
                pid = (rr.get("pid") or "").lower()
                if pid: pool[pid] = (rr.get("summary") or "").strip()
    print(f"[load] ottqa_tables={len(ottqa_tables)} pool={len(pool)}", flush=True)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.api_base.rstrip("/"),
                         api_key=args.api_key, timeout=180)
    sem = asyncio.Semaphore(args.concurrency)

    fout = args.out_jsonl.open("w")
    em_sum = f1_sum = 0.0; n_seen = n_err = 0
    n_s1_correct = 0
    progress = 0; lock = asyncio.Lock()

    async def _one(r):
        nonlocal em_sum, f1_sum, n_seen, n_err, n_s1_correct, progress
        async with sem:
            qid = r["question_id"]
            q = r["question"]
            gold = r["answer_text"]
            gold_tid = r["table_id"]
            top1_tid = qid_to_top1.get(qid)
            if top1_tid is None:
                async with lock:
                    n_err += 1; progress += 1
                return
            s1_correct = (top1_tid == gold_tid)

            t_ottqa = ottqa_tables.get(top1_tid, {})
            h_names = [h[0] if isinstance(h, list) else str(h)
                        for h in t_ottqa.get("header", [])]
            table_text = render_table(t_ottqa, h_names)

            # cell-link union of this 1 table
            cand_records = []
            seen = set()
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
        async with lock:
            fout.write(json.dumps({
                "qid": qid, "question": q, "gold": gold,
                "kind": r["kind"], "gold_tid": gold_tid, "top1_tid": top1_tid,
                "s1_correct": bool(s1_correct),
                "n_cands": len(cand_records), "n_chosen": len(chosen),
                "predicted": pred, "raw": raw[:1500],
                "em": float(em), "f1": round(f1, 4),
                "method": method, "reader": args.model,
                "error": raw if err else None,
            }, ensure_ascii=False) + "\n"); fout.flush()
            if err: n_err += 1
            else:
                em_sum += em; f1_sum += f1; n_seen += 1
                if s1_correct: n_s1_correct += 1
            progress += 1
            if progress % 100 == 0:
                el = time.time() - t0
                print(f"[{method}] {progress}/{len(subset)} EM={em_sum/max(1,n_seen):.4f} "
                      f"F1={f1_sum/max(1,n_seen):.4f} s1_acc={n_s1_correct/max(1,n_seen):.4f} "
                      f"err={n_err} {el:.0f}s", flush=True)

    async def _run():
        await asyncio.gather(*[_one(r) for r in subset])
    asyncio.run(_run())
    fout.close()

    n = max(1, n_seen)
    by_s1 = {"s1_correct": [], "s1_wrong": []}
    by_kind = {}
    for line in args.out_jsonl.read_text().splitlines():
        if line.strip():
            rr = json.loads(line)
            (by_s1["s1_correct"] if rr["s1_correct"] else by_s1["s1_wrong"]).append(rr)
            by_kind.setdefault(rr["kind"], []).append(rr)
    def agg(rows):
        if not rows: return {"n": 0, "EM": 0.0, "F1": 0.0}
        return {"n": len(rows),
                "EM": sum(r["em"] for r in rows)/len(rows),
                "F1": sum(r["f1"] for r in rows)/len(rows)}
    summary = {
        "method": method, "model": args.model,
        "n_total": len(subset), "n_scored": n_seen, "n_errors": n_err,
        "n_stage1_correct": n_s1_correct,
        "stage1_top1_acc": n_s1_correct / max(1, len(subset)),
        "EM": em_sum / n, "F1": f1_sum / n,
        "by_kind": {k: agg(v) for k, v in by_kind.items()},
        "by_stage1_correct": {
            "correct": agg(by_s1["s1_correct"]),
            "wrong":   agg(by_s1["s1_wrong"]),
        },
        "wall_sec": round(time.time() - t0, 1),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2))
    print(f"\n=== OTT-QA strict-1690 per-method [{method}] ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
