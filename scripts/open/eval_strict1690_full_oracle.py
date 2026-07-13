#!/usr/bin/env python3
"""Upper-bound reader eval on strict-1690 OTT-QA dev subset.

Setup (deliberately favors the reader to isolate its ceiling):
  - Gold table (from ottqa_repo traindev_tables) — NO retrieval
  - Linked passages from gold table cells, resolved via HybridQA 240K pool
    (i.e. all passages the reader could have, given the gold table)
  - 35B reader on 12.43 cuda 6,7 (own vLLM, via SSH tunnel 127.0.0.1:9540)

Output:
  - Per-query JSONL with EM/F1 + kind (table_only | passage_recoverable)
  - Summary JSON with overall + per-kind metrics
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import utils.hybridqa_metrics as _hyb
parse_answer = _hyb.parse_answer
from utils.sparq_eval import sparq_em_f1


PROMPT_TEMPLATE = (Path(__file__).resolve().parent.parent /
                   "src/schedule_pipeline/hybridqa_qa_cot.txt").read_text() + (
"""

Now answer:
Question: {question}
Selected Table:
{table_text}
SQL Result:
(no SQL run; rely on Selected Table and Linked Passages directly)
Linked Passages:
{passages_text}
""")


def load_pool(path: Path) -> dict[str, str]:
    """url(lower) → longest summary."""
    pool: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pid = (r.get("pid") or "").lower()
            sm = (r.get("summary") or "").strip()
            if pid and sm and (pid not in pool or len(sm) > len(pool[pid])):
                pool[pid] = sm
    return pool


def render_gold_table(t: dict, pool: dict[str, str],
                      gold_passage_pids: list[str],
                      max_rows: int = 30) -> tuple[str, list[str]]:
    """FULL ORACLE: table = gold, passages = exactly the gold-answer-bearing URLs.

    No BM25, no extra noise. The model gets the table and the single (or few)
    passages that we KNOW contain the answer. This is the strict reader ceiling.
    """
    title = (t.get("title") or "")[:300]
    section = (t.get("section_title") or "")[:200]
    caption = (f"# {title}{' — ' + section if section else ''}" if title else "").strip()
    header = t.get("header", [])
    h_names = [h[0] if isinstance(h, list) else str(h) for h in header]
    lines = []
    if caption: lines.append(caption)
    lines.append("col : row_id | " + " | ".join(h_names))

    # row→col→cell anchors to locate gold pids in the table
    anchor_by_pid: dict[str, tuple[int, str, str]] = {}
    for r_idx, row in enumerate(t.get("data", [])):
        if r_idx >= max_rows: break
        cell_strs = [str(r_idx)]
        for c_idx, cell in enumerate(row):
            if isinstance(cell, list) and len(cell) >= 2:
                cell_text = str(cell[0])[:80]
                cell_strs.append(cell_text)
                col_name = h_names[c_idx] if c_idx < len(h_names) else f"col{c_idx}"
                for url in (cell[1] or []):
                    pid = url.lower()
                    if pid not in anchor_by_pid:
                        anchor_by_pid[pid] = (r_idx, col_name, cell_text[:40])
            else:
                cell_strs.append(str(cell)[:80])
        lines.append(f"row {r_idx} : " + " | ".join(cell_strs))

    passages_used = []
    for k, pid in enumerate(gold_passage_pids):
        txt = pool.get(pid.lower(), "")
        if not txt: continue
        anchor = anchor_by_pid.get(pid.lower())
        if anchor:
            r_idx, col_name, cell_text = anchor
            passages_used.append(
                f"[Passage {k+1}] row={r_idx} col={col_name} cell={cell_text}\n{txt[:1500]}"
            )
        else:
            passages_used.append(f"[Passage {k+1}] {pid}\n{txt[:1500]}")
    return "\n".join(lines), passages_used


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
    ap.add_argument("--subset", default="analysis/ottqa_dev_strict1690.jsonl", type=Path)
    ap.add_argument("--tables", default="data/ottqa_repo/data/traindev_tables.json", type=Path)
    ap.add_argument("--pool", default="analysis/open_pool_full/passages.jsonl", type=Path)
    ap.add_argument("--api-base", default="http://127.0.0.1:9540/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="qwen3.6-35b")
    ap.add_argument("--concurrency", default=48, type=int)
    ap.add_argument("--max-queries", default=0, type=int)
    ap.add_argument("--out-jsonl", default="analysis/ottqa_strict1690/oracle_reader_35b.jsonl", type=Path)
    ap.add_argument("--out", default="analysis/ottqa_strict1690/oracle_reader_35b.summary.json", type=Path)
    args = ap.parse_args()

    for k in ("http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None); os.environ.pop(k.upper(), None)

    t0 = time.time()
    queries = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    tables = json.loads(args.tables.read_text())
    pool = load_pool(args.pool)
    print(f"[load] {len(queries)} queries, {len(tables)} tables, {len(pool)} pool urls", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out_jsonl.open("w")

    # resume?
    seen_qids = set()

    # accumulators
    by_kind = defaultdict(lambda: {"em": 0.0, "f1": 0.0, "n": 0})
    em_sum = f1_sum = 0.0; n_seen = 0; n_err = 0
    n_with_passages = 0
    progress = 0
    started = time.time()
    lock = asyncio.Lock()

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.api_base.rstrip("/"), api_key=args.api_key, timeout=180)
    sem = asyncio.Semaphore(args.concurrency)

    async def _one(q):
        nonlocal em_sum, f1_sum, n_seen, n_err, n_with_passages, progress
        async with sem:
            qid = q["question_id"]
            tid = q["table_id"]
            t = tables.get(tid, {})
            if not t:
                table_text = f"(table {tid} not found)"
                passages = []
            else:
                table_text, passages = render_gold_table(
                    t, pool, q.get("gold_passage_pids") or [])
            passages_text = ("Linked passages:\n" + "\n".join(passages)) if passages else ""
            prompt = PROMPT_TEMPLATE.format(table_text=table_text,
                                            passages_text=passages_text,
                                            question=q["question"])
            raw = await call_llm(client, args.model, prompt)
        err = raw.startswith("__ERROR__")
        pred = parse_answer(raw) if not err else ""
        em, f1 = sparq_em_f1(pred, q["answer_text"], question=q["question"])
        async with lock:
            fout.write(json.dumps({
                "qid": qid, "question": q["question"], "gold": q["answer_text"],
                "kind": q["kind"], "n_passages": len(passages),
                "predicted": pred, "raw": raw[:400],
                "em": float(em), "f1": round(f1, 4),
                "error": raw if err else None,
            }, ensure_ascii=False) + "\n"); fout.flush()
            if err: n_err += 1
            else:
                em_sum += em; f1_sum += f1; n_seen += 1
                bk = by_kind[q["kind"]]
                bk["em"] += em; bk["f1"] += f1; bk["n"] += 1
                if passages: n_with_passages += 1
            progress += 1
            if progress % 50 == 0:
                print(f"[eval] {progress}/{len(queries)} "
                      f"EM={em_sum/max(1,n_seen):.4f} F1={f1_sum/max(1,n_seen):.4f} "
                      f"err={n_err} wp={n_with_passages} "
                      f"{time.time()-started:.0f}s", flush=True)

    async def _run():
        await asyncio.gather(*[_one(q) for q in queries])
    asyncio.run(_run())
    fout.close()

    n = max(1, n_seen)
    summary = {
        "n_total": len(queries), "n_scored": n_seen, "n_errors": n_err,
        "n_with_passages": n_with_passages,
        "EM": em_sum / n, "F1": f1_sum / n,
        "by_kind": {k: {"n": v["n"],
                         "EM": v["em"] / max(1, v["n"]),
                         "F1": v["f1"] / max(1, v["n"])}
                     for k, v in by_kind.items()},
        "wall_sec": round(time.time() - t0, 1),
    }
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== Oracle reader on strict-1690 ({args.model}) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
