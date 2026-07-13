#!/usr/bin/env python3
"""SPARQ-X on FULL OTT-QA dev (2214) with the REAL 5.96M-passage corpus (all_passages.json).

Pipeline (main method): reranked top-1 table (Stage-1) -> cell-link passages from the 5M corpus
(Stage-2, BM25 top-K within N(T_hat)) -> 35B reader (Stage-3). Lifts the strict-1690 /
524-unreachable constraint -> honest full-dev EM/F1.
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, sys, time, urllib.parse
from pathlib import Path
import ijson

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import utils.hybridqa_metrics as _hyb
parse_answer = _hyb.parse_answer
from utils.sparq_eval import sparq_em_f1
sys.path.insert(0, str(REPO / "scripts"))
from eval_ottqa_per_method_e2e import render_table, bm25_rank, PROMPT_TEMPLATE

TOK = lambda s: re.findall(r"\w+", s.lower())


def url_to_title(u: str) -> str:
    return urllib.parse.unquote(u.replace("/wiki/", "")).replace("_", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranked", default="analysis/ottqa_full2214/reranked_cands_v1.jsonl", type=Path)
    ap.add_argument("--subset", default="analysis/ottqa_dev_full2214.jsonl", type=Path)
    ap.add_argument("--passages", default="data/ottqa_5m/all_passages.json", type=Path)
    ap.add_argument("--ottqa-tables", default="data/ottqa_repo/data/traindev_tables.json", type=Path)
    ap.add_argument("--K-pas", type=int, default=20)
    ap.add_argument("--passage-chars", type=int, default=2500)
    ap.add_argument("--api-base", default="http://192.168.12.43:9543/v1")
    ap.add_argument("--model", default="Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--out", default="analysis/ottqa_full2214/e2e_5m_reranker_top1", type=str)
    args = ap.parse_args()
    for k in ("http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
        os.environ.pop(k, None)
    t0 = time.time()

    subset = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    if args.max_queries: subset = subset[: args.max_queries]
    qid2gold = {r["question_id"]: r for r in subset}
    top1 = {}
    for l in args.reranked.read_text().splitlines():
        if l.strip():
            d = json.loads(l); top1[d["qid"]] = d["reranked_table_ids"][0]
    tables = json.loads(args.ottqa_tables.read_text())
    print(f"[load] subset={len(subset)} reranked={len(top1)} tables={len(tables)}", flush=True)

    # collect cell-link titles needed across all top-1 tables
    need = set()
    for r in subset:
        t = tables.get(top1.get(r["question_id"]), {})
        for row in t.get("data", []):
            for cell in row:
                if isinstance(cell, list) and len(cell) >= 2:
                    for u in (cell[1] or []): need.add(url_to_title(u))
    print(f"[need] {len(need)} unique cell-link titles; streaming 5M corpus ...", flush=True)

    # stream-extract only needed titles from the 5.96M corpus
    pool = {}; seen = 0
    with args.passages.open() as f:
        for obj in ijson.items(f, "item"):
            seen += 1
            cid = obj.get("chunk_id", "")
            if cid in need:
                pool[cid] = (obj.get("text") or "").strip()
            if seen % 1_000_000 == 0:
                print(f"  streamed {seen//1_000_000}M, matched {len(pool)}/{len(need)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[pool] matched {len(pool)}/{len(need)} cell-link titles in {seen} passages ({time.time()-t0:.0f}s)", flush=True)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.api_base.rstrip("/"), api_key="EMPTY", timeout=180)
    sem = asyncio.Semaphore(args.concurrency); lock = asyncio.Lock()
    fout = open(args.out + ".jsonl", "w")
    em_sum = f1_sum = 0.0; n = nerr = ns1 = ncov = 0; prog = 0

    async def one(r):
        nonlocal em_sum, f1_sum, n, nerr, ns1, ncov, prog
        qid = r["question_id"]; q = r["question"]; gold = r["answer_text"]; gtid = r["table_id"]
        async with sem:
            tid = top1.get(qid); s1 = (tid == gtid)
            t = tables.get(tid, {})
            h = [hh[0] if isinstance(hh, list) else str(hh) for hh in t.get("header", [])]
            table_text = render_table(t, h)
            recs, seenp = [], set()
            for row in t.get("data", []):
                for cell in row:
                    if isinstance(cell, list) and len(cell) >= 2:
                        for u in (cell[1] or []):
                            ti = url_to_title(u)
                            if ti in seenp or ti not in pool: continue
                            seenp.add(ti); recs.append((str(cell[0])[:60], pool[ti]))
            order = bm25_rank(q, [(a, x[: args.passage_chars]) for a, x in recs])[: args.K_pas]
            chosen = [recs[i] for i in order]
            ptext = "\n".join(f"[Passage {k+1}] {c[1][: args.passage_chars]}" for k, c in enumerate(chosen)) or "(none)"
            prompt = PROMPT_TEMPLATE.format(table_text=table_text, passages_text=ptext, question=q)
            try:
                rr = await client.chat.completions.create(model=args.model,
                    messages=[{"role": "user", "content": prompt}], max_tokens=2048, temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                raw = rr.choices[0].message.content or ""; err = None
            except Exception as e:
                raw = ""; err = repr(e)[:160]
        pred = parse_answer(raw) if not err else ""
        em, f1 = (sparq_em_f1(pred, gold, question=q) if not err else (0.0, 0.0))
        async with lock:
            if err: nerr += 1
            else:
                em_sum += em; f1_sum += f1; n += 1; ns1 += int(s1); ncov += int(len(recs) > 0)
            fout.write(json.dumps({"qid": qid, "gold": gold, "gold_tid": gtid, "top1_tid": tid,
                "s1_correct": bool(s1), "n_cell_link": len(recs), "predicted": pred,
                "em": float(em), "f1": round(f1, 4), "in_strict1690": r.get("in_strict1690"),
                "error": err}, ensure_ascii=False) + "\n")
            prog += 1
            if prog % 200 == 0:
                print(f"  [{prog}/{len(subset)}] EM={em_sum/max(1,n):.4f} F1={f1_sum/max(1,n):.4f} "
                      f"s1={ns1/max(1,n):.3f} cov={ncov/max(1,n):.3f} err={nerr} {time.time()-t0:.0f}s", flush=True)

    async def run(): await asyncio.gather(*[one(r) for r in subset])
    asyncio.run(run()); fout.close()

    # full-dev + strict-1690-only breakdown
    rows = [json.loads(l) for l in Path(args.out + ".jsonl").read_text().splitlines() if l.strip()]
    ok = [r for r in rows if not r["error"]]
    def agg(rs): return {"n": len(rs), "EM": round(sum(r["em"] for r in rs)/max(1,len(rs)), 4),
                          "F1": round(sum(r["f1"] for r in rs)/max(1,len(rs)), 4)}
    summary = {"method": "SPARQ-X full-dev (reranker top-1 + 5M cell-link + 35B)",
               "full_dev_2214": agg(ok), "strict1690_subset": agg([r for r in ok if r["in_strict1690"]]),
               "extra_524": agg([r for r in ok if not r["in_strict1690"]]),
               "stage1_top1_acc": round(ns1/max(1,n), 4), "celllink_coverage": round(ncov/max(1,n), 4),
               "n_errors": nerr, "wall_sec": round(time.time()-t0, 1)}
    Path(args.out + ".summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SPARQ-X FULL OTT-QA dev ==="); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
