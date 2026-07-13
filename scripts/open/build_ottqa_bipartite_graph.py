#!/usr/bin/env python3
"""Build a lightweight Table↔Passage bipartite HeteroData over OTT-QA.

Saves memory by collapsing rows/cells into a single table↔passage edge type:
  edge weight = number of cells in the table that link to the passage.

Node types:
  'table'   (8,891)    x: Qwen3-Embed table_embeddings
  'passage' (240,042)  x: Qwen3-Embed passage_embeddings

Edges (bidirectional):
  table -- has_passage --> passage
  passage -- in_table --> table

Output: analysis/ottqa_open_pool/bipartite_graph.pt
"""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="analysis/ottqa_open_pool", type=Path)
    ap.add_argument("--table-emb", default="table_embeddings_qwen3.npy")
    ap.add_argument("--passage-emb", default="passage_embeddings_qwen3.npy")
    ap.add_argument("--out", default="analysis/ottqa_open_pool/bipartite_graph.pt", type=Path)
    args = ap.parse_args()

    t0 = time.time()
    table_emb = np.load(args.pool_dir / args.table_emb)
    passage_emb = np.load(args.pool_dir / args.passage_emb)
    table_ids = json.loads((args.pool_dir / "table_ids.json").read_text())
    passage_ids = json.loads((args.pool_dir / "passage_ids.json").read_text())
    print(f"[load] tables={table_emb.shape} passages={passage_emb.shape}", flush=True)

    tid_to_idx = {t: i for i, t in enumerate(table_ids)}
    pid_to_idx = {p: i for i, p in enumerate(passage_ids)}

    # Build table→passage edges with cell-count weights
    edge_count = defaultdict(int)
    with (args.pool_dir / "cells.jsonl").open() as f:
        for line in f:
            if not line.strip(): continue
            c = json.loads(line)
            ti = tid_to_idx.get(c["table_id"])
            if ti is None: continue
            for u in c.get("urls", []) or []:
                if u in pid_to_idx:
                    edge_count[(ti, pid_to_idx[u])] += 1
    pairs = list(edge_count.items())
    print(f"[edges] unique table-passage pairs: {len(pairs)}", flush=True)

    src = np.array([p[0][0] for p in pairs], dtype=np.int64)
    dst = np.array([p[0][1] for p in pairs], dtype=np.int64)
    weights = np.array([p[1] for p in pairs], dtype=np.float32)

    data = HeteroData()
    data["table"].x = torch.from_numpy(table_emb).float()
    data["passage"].x = torch.from_numpy(passage_emb).float()
    data[("table", "has_passage", "passage")].edge_index = torch.tensor(
        np.stack([src, dst]), dtype=torch.long)
    data[("passage", "in_table", "table")].edge_index = torch.tensor(
        np.stack([dst, src]), dtype=torch.long)
    # Edge attrs (cell count, for weighted GAT)
    data[("table", "has_passage", "passage")].edge_attr = torch.from_numpy(weights).float().unsqueeze(-1)
    data[("passage", "in_table", "table")].edge_attr = torch.from_numpy(weights).float().unsqueeze(-1)

    print(f"[done] {data}  {time.time()-t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "data": data,
        "table_ids": table_ids,
        "passage_ids": passage_ids,
    }, args.out)
    print(f"[saved] {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
