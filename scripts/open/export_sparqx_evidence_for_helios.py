#!/usr/bin/env python3
"""Export SPARQ-X Stage 1-2 retrieval evidence (per-query top-1 table + ranked cell-link
passages) for a fixed-evidence cross-reader comparison on a remote machine (HELIOS).

Reproduces EXACTLY the evidence the SPARQ-X full-dev reader saw in
analysis/ottqa_full2214/e2e_5m_reranker_top1 (EM 0.6762 / F1 0.7315): reranked top-1 table
-> render_table -> cell-link union from the 5.96M corpus -> BM25 top-K_pas. No LLM call here;
deterministic (the reader used temperature 0), so HELIOS gets byte-identical evidence.

Output: one JSON per query with {qid, question, gold, retrieved_table_id, gold_table_id,
s1_correct, in_strict1690, table_text, passages[], passage_ids[], n_cell_link}.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.parse
from pathlib import Path
import ijson

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from eval_ottqa_per_method_e2e import render_table, bm25_rank


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
    ap.add_argument("--out", default="experiments/exports/sparqx_retrieval_for_helios.jsonl", type=Path)
    args = ap.parse_args()
    t0 = time.time()

    subset = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    qid2row = {r["question_id"]: r for r in subset}
    top1 = {}
    for l in args.reranked.read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            ids = d.get("reranked_table_ids") or d.get("rrf_cand_table_ids")
            top1[d["qid"]] = ids[0] if ids else None
    tables = json.loads(args.ottqa_tables.read_text())
    print(f"[load] subset={len(subset)} reranked={len(top1)} tables={len(tables)}", flush=True)

    # cell-link titles needed across all top-1 tables
    need = set()
    for r in subset:
        t = tables.get(top1.get(r["question_id"]), {})
        for row in t.get("data", []):
            for cell in row:
                if isinstance(cell, list) and len(cell) >= 2:
                    for u in (cell[1] or []):
                        need.add(url_to_title(u))
    print(f"[need] {len(need)} unique cell-link titles; streaming 5M corpus ...", flush=True)

    pool = {}; seen = 0
    with args.passages.open() as f:
        for obj in ijson.items(f, "item"):
            seen += 1
            cid = obj.get("chunk_id", "")
            if cid in need:
                pool[cid] = (obj.get("text") or "").strip()
            if seen % 1_000_000 == 0:
                print(f"  streamed {seen//1_000_000}M, matched {len(pool)}/{len(need)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[pool] matched {len(pool)}/{len(need)} in {seen} passages ({time.time()-t0:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out.open("w")
    n = 0
    for r in subset:
        qid = r["question_id"]; q = r["question"]
        tid = top1.get(qid); gtid = r["table_id"]; s1 = (tid == gtid)
        t = tables.get(tid, {})
        h = [hh[0] if isinstance(hh, list) else str(hh) for hh in t.get("header", [])]
        table_text = render_table(t, h)
        recs = []; seenp = set()  # (title, cell, text)
        for row in t.get("data", []):
            for cell in row:
                if isinstance(cell, list) and len(cell) >= 2:
                    for u in (cell[1] or []):
                        ti = url_to_title(u)
                        if ti in seenp or ti not in pool:
                            continue
                        seenp.add(ti); recs.append((ti, str(cell[0])[:60], pool[ti]))
        order = bm25_rank(q, [(c[1], c[2][: args.passage_chars]) for c in recs])[: args.K_pas]
        chosen = [recs[i] for i in order]
        fout.write(json.dumps({
            "qid": qid, "question": q, "gold": r["answer_text"],
            "retrieved_table_id": tid, "gold_table_id": gtid,
            "s1_correct": bool(s1), "in_strict1690": r.get("in_strict1690"),
            "kind": r.get("kind"),
            "table_text": table_text,
            "passages": [c[2][: args.passage_chars] for c in chosen],
            "passage_ids": [c[0] for c in chosen],
            "n_cell_link": len(recs),
        }, ensure_ascii=False) + "\n")
        n += 1
        if n % 500 == 0:
            print(f"  wrote {n}/{len(subset)} ({time.time()-t0:.0f}s)", flush=True)
    fout.close()
    print(f"[done] exported {n} queries -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
