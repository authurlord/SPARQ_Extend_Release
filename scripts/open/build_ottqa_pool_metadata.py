#!/usr/bin/env python3
"""Build OTT-QA open pool metadata mirroring HybridQA's analysis/open_pool layout.

Output to analysis/ottqa_open_pool/:
  table_ids.json          ← list of 8,891 IBM corpus _ids (order matches table_embeddings.npy)
  passage_ids.json        ← symlink/copy of HybridQA pool's (same 240K)
  cells.jsonl             ← per-cell metadata: {table_id, row, col, text, urls[]}
  table_embeddings.npy    ← symlink to analysis/ottqa_small/table_embeddings.npy
  passage_embeddings.npy  ← symlink to analysis/open_pool_full/passage_embeddings.npy

These mirror the structure consumed by `scripts/build_hetero_graph.py`.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibm-corpus", default="data/ottqa_raw_full/corpus_structure.jsonl", type=Path)
    ap.add_argument("--ottqa-tables", default="data/ottqa_repo/data/traindev_tables.json", type=Path)
    ap.add_argument("--hybridqa-passage-ids",
                     default="analysis/open_pool_full/passage_ids.json", type=Path)
    ap.add_argument("--out-dir", default="analysis/ottqa_open_pool", type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. table_ids
    ibm = [json.loads(l) for l in args.ibm_corpus.read_text().splitlines() if l.strip()]
    table_ids = [t["_id"] for t in ibm]
    (args.out_dir / "table_ids.json").write_text(json.dumps(table_ids))
    print(f"[ids] tables: {len(table_ids)}", flush=True)

    # 2. passage_ids (reuse HybridQA)
    pids = json.loads(args.hybridqa_passage_ids.read_text())
    (args.out_dir / "passage_ids.json").write_text(json.dumps(pids))
    print(f"[ids] passages: {len(pids)}", flush=True)

    # 3. cells.jsonl from ottqa_repo
    ottqa_tables = json.loads(args.ottqa_tables.read_text())
    n_cells = 0
    with (args.out_dir / "cells.jsonl").open("w") as f:
        for tid in table_ids:
            t = ottqa_tables.get(tid, {})
            if not t: continue
            header = t.get("header") or []
            h_names = [h[0] if isinstance(h, list) else str(h) for h in header]
            title = (t.get("title") or "")[:200]
            for r_idx, row in enumerate(t.get("data", [])):
                for c_idx, cell in enumerate(row):
                    if isinstance(cell, list) and len(cell) >= 2:
                        text = str(cell[0])[:200]
                        urls = [u.lower() for u in (cell[1] or [])]
                    else:
                        text = str(cell)[:200]
                        urls = []
                    col_name = h_names[c_idx] if c_idx < len(h_names) else f"col{c_idx}"
                    f.write(json.dumps({
                        "table_id": tid, "table_title": title,
                        "row": r_idx, "col": c_idx, "col_name": col_name,
                        "text": text, "urls": urls,
                    }, ensure_ascii=False) + "\n")
                    n_cells += 1
    print(f"[cells] wrote {n_cells} cells to {args.out_dir/'cells.jsonl'}", flush=True)

    # 4. Symlinks for embeddings (reuse existing)
    sym_pairs = [
        (Path("analysis/ottqa_small/table_embeddings.npy").resolve(),
          args.out_dir / "table_embeddings.npy"),
        (Path("analysis/open_pool_full/passage_embeddings.npy").resolve(),
          args.out_dir / "passage_embeddings.npy"),
        (Path("analysis/open_pool_full/passage_embeddings_gnn.npy").resolve(),
          args.out_dir / "passage_embeddings_gnn.npy"),
    ]
    for src, dst in sym_pairs:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        print(f"[symlink] {dst.name} → {src}", flush=True)

    print(f"\n[done] OTT-QA open pool metadata at {args.out_dir}/")


if __name__ == "__main__":
    main()
