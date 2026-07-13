#!/usr/bin/env python3
"""Build FlagEmbedding-format training data for OTT-QA passage reranker.

For each OTT-QA train query with passage-type answer-node:
  - pos = pool summary of each gold passage URL (must be in 240K HybridQA pool)
  - neg = pool summaries of OTHER cell-linked URLs from the gold table
          (must be in pool, and NOT contain the gold answer-text as substring)
Query format: "[QUERY] question" (no cell context — keep simple, in-distribution test)

Output: data/ottqa_passage_reranker_training/train.jsonl
"""
from __future__ import annotations
import argparse, json, random, re, zipfile
from collections import Counter
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-linked-zip",
                    default="data/ottqa_repo/preprocessed_data/train_linked.json.zip",
                    type=Path)
    ap.add_argument("--ottqa-tables",
                    default="data/ottqa_repo/data/traindev_tables.json",
                    type=Path)
    ap.add_argument("--pool",
                    default="analysis/open_pool_full/passages.jsonl",
                    type=Path)
    ap.add_argument("--max-pos-per-query", type=int, default=3)
    ap.add_argument("--max-neg-per-query", type=int, default=15)
    ap.add_argument("--max-text-chars", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/ottqa_passage_reranker_training/train.jsonl",
                    type=Path)
    args = ap.parse_args()
    random.seed(args.seed)

    # Load pool
    pool = {}
    with args.pool.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pid = (r.get("pid") or "").lower()
                if pid: pool[pid] = (r.get("summary") or "").strip()
    print(f"[load] pool: {len(pool)}", flush=True)

    # Load train queries
    with zipfile.ZipFile(args.train_linked_zip) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        with zf.open(names[0]) as f:
            train = json.load(f)
    print(f"[load] train: {len(train)}", flush=True)

    # Load tables
    tables = json.loads(args.ottqa_tables.read_text())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_skipped_no_pos = n_skipped_no_neg = 0
    pos_counts, neg_counts = [], []
    with args.out.open("w") as fout:
        for r in train:
            q = (r.get("question") or "").strip()
            ans = norm(r.get("answer-text", ""))
            if not q or not ans: continue
            nodes = r.get("answer-node", []) or []
            tid = r.get("table_id", "")
            t = tables.get(tid, {})
            if not t: continue

            # Pos: gold passage URLs (passage-type with answer in pool summary)
            gold_pids = set()
            for n in nodes:
                if len(n) >= 4 and n[3] == "passage" and n[2]:
                    pid = n[2].lower()
                    if pid in pool and ans in norm(pool[pid]):
                        gold_pids.add(pid)
            pos_texts = []
            for pid in list(gold_pids)[: args.max_pos_per_query]:
                pos_texts.append(pool[pid][: args.max_text_chars])
            if not pos_texts:
                n_skipped_no_pos += 1; continue

            # Neg: other cell-linked URLs from gold table, not gold and not containing answer
            neg_pids = []
            for row in t.get("data", []):
                for cell in row:
                    if isinstance(cell, list) and len(cell) >= 2:
                        for u in (cell[1] or []):
                            pid = u.lower()
                            if pid in pool and pid not in gold_pids:
                                if ans not in norm(pool[pid]):
                                    neg_pids.append(pid)
            # Dedup, shuffle, sample
            neg_pids = list(dict.fromkeys(neg_pids))
            random.shuffle(neg_pids)
            neg_texts = [pool[pid][: args.max_text_chars]
                         for pid in neg_pids[: args.max_neg_per_query]]
            if not neg_texts:
                n_skipped_no_neg += 1; continue

            fout.write(json.dumps({
                "query": q,
                "pos": pos_texts,
                "neg": neg_texts,
            }, ensure_ascii=False) + "\n")
            n_written += 1
            pos_counts.append(len(pos_texts))
            neg_counts.append(len(neg_texts))

    print(f"\n[done] wrote {n_written} samples")
    print(f"  skipped (no pos): {n_skipped_no_pos}")
    print(f"  skipped (no neg): {n_skipped_no_neg}")
    print(f"  avg pos: {sum(pos_counts)/max(1,len(pos_counts)):.2f}")
    print(f"  avg neg: {sum(neg_counts)/max(1,len(neg_counts)):.2f}")
    print(f"  total pos: {sum(pos_counts)}, total neg: {sum(neg_counts)}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
