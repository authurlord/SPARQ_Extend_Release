#!/usr/bin/env python3
"""Build the strict-recoverable 1690 dev subset.

Output one record per kept query with:
  - question_id, question, answer-text, table_id, answer-node
  - kind: "table_only" | "passage_recoverable"
  - gold_passage_pids: list of HybridQA pool pids that contain the answer-text
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

DEV = Path("data/ottqa_repo/preprocessed_data/dev_linked.json")
POOL = Path("analysis/open_pool_full/passages.jsonl")
OUT = Path("analysis/ottqa_dev_strict1690.jsonl")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def main():
    dev = json.loads(DEV.read_text())
    pool: dict[str, list[str]] = defaultdict(list)
    with POOL.open() as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pid = (r.get("pid") or "").lower()
            sm = norm(r.get("summary", ""))
            if pid and sm:
                pool[pid].append(sm)
    print(f"[load] dev={len(dev)} pool={len(pool)} pids", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_keep = n_table = n_passage = 0
    with OUT.open("w") as fo:
        for r in dev:
            ans = norm(r.get("answer-text", ""))
            if not ans: continue
            nodes = r.get("answer-node", []) or []
            types = [n[3] for n in nodes if len(n) >= 4]
            has_table = "table" in types
            has_passage = "passage" in types

            kind = None
            gold_pids: list[str] = []
            if not has_passage:
                kind = "table_only"
            else:
                # find passage URLs whose pool summary contains the answer
                urls = [n[2].lower() for n in nodes
                        if len(n) >= 4 and n[3] == "passage" and n[2]]
                for u in urls:
                    if u in pool:
                        for s in pool[u]:
                            if ans in s:
                                gold_pids.append(u)
                                break
                if gold_pids:
                    kind = "passage_recoverable"
                elif has_table:
                    # passage failed but table exists → treat as table_only
                    kind = "table_only"
            if kind is None:
                continue
            n_keep += 1
            if kind == "table_only": n_table += 1
            else: n_passage += 1
            fo.write(json.dumps({
                "question_id": r.get("question_id"),
                "question": r["question"],
                "answer_text": r["answer-text"],
                "table_id": r["table_id"],
                "answer_node": nodes,
                "kind": kind,
                "gold_passage_pids": gold_pids,
            }, ensure_ascii=False) + "\n")

    print(f"\n[done] {n_keep} kept ({n_table} table_only, {n_passage} passage_recoverable) → {OUT}")


if __name__ == "__main__":
    main()
