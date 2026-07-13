#!/usr/bin/env python3
"""Embed OTT-QA cells via vLLM /v1/embeddings.

Pattern (per user 2026-05-20): SERIAL requests, each with input=[N texts].
NO async/concurrent — server batches internally per request.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import numpy as np
from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="analysis/ottqa_open_pool/cells.jsonl", type=Path)
    ap.add_argument("--out", default="analysis/ottqa_open_pool/cell_embeddings.npy", type=Path)
    ap.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--api-key", default="embed-key-qwen3")
    ap.add_argument("--model", default="qwen3-embed")
    ap.add_argument("--batch-size", type=int, default=1024,
                     help="texts per request (user spec)")
    args = ap.parse_args()

    for k in ("http_proxy","https_proxy","all_proxy"):
        os.environ.pop(k, None); os.environ.pop(k.upper(), None)

    cells = [json.loads(l) for l in args.cells.read_text().splitlines() if l.strip()]
    print(f"[load] {len(cells)} cells", flush=True)

    texts = [
        f"{c.get('table_title','')[:120]} | {c.get('text','')[:200]}"
        for c in cells
    ]

    client = OpenAI(base_url=args.api_base.rstrip("/"), api_key=args.api_key)

    t0 = time.time()
    embs: list[np.ndarray] = []
    n = len(texts); bs = args.batch_size
    for s in range(0, n, bs):
        batch = texts[s: s + bs]
        # Single request with N inputs → returns N embeddings
        resp = client.embeddings.create(model=args.model, input=batch)
        # resp.data is a list of EmbeddingObject; .embedding is the vector
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        embs.append(vecs)
        el = time.time() - t0
        rate = (s + len(batch)) / max(1, el)
        print(f"  [{s+len(batch)}/{n}] {rate:.0f} cells/s  {el:.0f}s "
              f"({len(batch)} per req, dim={vecs.shape[1]})", flush=True)
    emb = np.vstack(embs)
    print(f"[done] {emb.shape}, {time.time()-t0:.0f}s, → {args.out}")
    np.save(args.out, emb)


if __name__ == "__main__":
    main()
