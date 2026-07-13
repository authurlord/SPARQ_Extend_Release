#!/usr/bin/env python3
"""Embed OTT-QA tables / passages / queries with Qwen3-Embedding-0.6B (consistent encoder).

Output (overwriting symlinks in analysis/ottqa_open_pool/):
  table_embeddings_qwen3.npy        (8,891 × 1024)
  passage_embeddings_qwen3.npy      (240,042 × 1024)
  dev_query_embeddings_qwen3.npy    (2,214 × 1024)
  train_query_embeddings_qwen3.npy  (~41,469 × 1024 — for GNN training)
"""
from __future__ import annotations
import argparse, json, os, time, zipfile
from pathlib import Path

import numpy as np
from openai import OpenAI


def encode_texts(client: OpenAI, model: str, texts: list[str], batch: int = 1024) -> np.ndarray:
    embs = []
    t0 = time.time()
    n = len(texts)
    for s in range(0, n, batch):
        b = texts[s: s + batch]
        resp = client.embeddings.create(model=model, input=b)
        embs.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
        if (s // batch) % 10 == 0:
            el = time.time() - t0
            print(f"  [{s+len(b)}/{n}] {(s+len(b))/max(1,el):.0f}/s  {el:.0f}s", flush=True)
    return np.vstack(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--api-key", default="embed-key-qwen3")
    ap.add_argument("--model", default="qwen3-embed")
    ap.add_argument("--out-dir", default="analysis/ottqa_open_pool", type=Path)
    ap.add_argument("--ibm-corpus", default="data/ottqa_raw_full/corpus_structure.jsonl", type=Path)
    ap.add_argument("--pool", default="analysis/open_pool_full/passages.jsonl", type=Path)
    ap.add_argument("--queries-ibm", default="data/ottqa_raw_full/dev_queries.jsonl", type=Path)
    ap.add_argument("--train-linked-zip",
                     default="data/ottqa_repo/preprocessed_data/train_linked.json.zip", type=Path)
    args = ap.parse_args()

    for k in ("http_proxy","https_proxy","all_proxy"):
        os.environ.pop(k, None); os.environ.pop(k.upper(), None)

    client = OpenAI(base_url=args.api_base.rstrip("/"), api_key=args.api_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tables
    out_path = args.out_dir / "table_embeddings_qwen3.npy"
    if not out_path.exists():
        ibm = [json.loads(l) for l in args.ibm_corpus.read_text().splitlines() if l.strip()]
        texts = []
        for t in ibm:
            title = (t.get("meta_data") or t.get("title") or "")[:200]
            headers = " | ".join(str(h)[:40] for h in (t.get("headers") or []))
            # First 3 row strings
            cells = t.get("cells") or []
            ncol = max(1, len(t.get("headers") or []))
            samples = []
            for i in range(0, min(3 * ncol, len(cells)), ncol):
                samples.append(" | ".join(str(c)[:30] for c in cells[i:i+ncol]))
            blob = f"{title} | {headers}" + (" | " + " || ".join(samples) if samples else "")
            texts.append(blob[:800])
        print(f"[tables] embedding {len(texts)} tables...", flush=True)
        e = encode_texts(client, args.model, texts)
        np.save(out_path, e); print(f"[tables] saved {e.shape} → {out_path}", flush=True)
    else:
        print(f"[tables] {out_path} exists, skip", flush=True)

    # 2. Passages
    out_path = args.out_dir / "passage_embeddings_qwen3.npy"
    if not out_path.exists():
        pids = []
        texts = []
        with args.pool.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    pid = (r.get("pid") or "").lower()
                    if pid:
                        pids.append(pid)
                        texts.append((r.get("summary") or "")[:1500])
        print(f"[passages] embedding {len(texts)} passages...", flush=True)
        e = encode_texts(client, args.model, texts)
        np.save(out_path, e); print(f"[passages] saved {e.shape} → {out_path}", flush=True)
    else:
        print(f"[passages] {out_path} exists, skip", flush=True)

    # 3. Dev queries
    out_path = args.out_dir / "dev_query_embeddings_qwen3.npy"
    if not out_path.exists():
        queries = [json.loads(l) for l in args.queries_ibm.read_text().splitlines() if l.strip()]
        texts = [q["text"] for q in queries]
        print(f"[dev-q] embedding {len(texts)} dev queries...", flush=True)
        e = encode_texts(client, args.model, texts)
        np.save(out_path, e); print(f"[dev-q] saved {e.shape} → {out_path}", flush=True)
    else:
        print(f"[dev-q] {out_path} exists, skip", flush=True)

    # 4. Train queries (for GNN training)
    out_path = args.out_dir / "train_query_embeddings_qwen3.npy"
    if not out_path.exists():
        with zipfile.ZipFile(args.train_linked_zip) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            with zf.open(names[0]) as fz:
                train = json.load(fz)
        texts = [q.get("question", "") for q in train]
        print(f"[train-q] embedding {len(texts)} train queries...", flush=True)
        e = encode_texts(client, args.model, texts)
        np.save(out_path, e); print(f"[train-q] saved {e.shape} → {out_path}", flush=True)
        # Also save train query IDs to align with table_id labels later
        train_meta = [{"question_id": q.get("question_id"),
                        "table_id": q.get("table_id"),
                        "question": q.get("question", "")} for q in train]
        (args.out_dir / "train_query_meta.jsonl").write_text(
            "\n".join(json.dumps(m) for m in train_meta) + "\n"
        )
    else:
        print(f"[train-q] {out_path} exists, skip", flush=True)


if __name__ == "__main__":
    main()
