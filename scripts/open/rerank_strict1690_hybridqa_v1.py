#!/usr/bin/env python3
"""Run HybridQA-trained reranker (XLM-RoBERTa cross-encoder) over the
strict-1690 hier-candidate sets, score recall@K_pas.

Two-step:
  Step A: build per-query candidate JSONL (qid, question, gold_pids, cands[(pid, summary)])
          - cands = union of cell-linked URLs across top-K_tab=10 tables (~271 avg)
  Step B: load reranker on cuda 7 (assumes the host has the model checkpoint
          and free VRAM); score every (q, passage) pair in batches.
  Step C: report recall@K_pas after reranker re-ordering.

Usage:
  # Local: build candidates
  python3 scripts/rerank_strict1690_hybridqa_v1.py build-cands \
      --out /tmp/strict1690_cands_K10.jsonl

  # On 12.43 (after rsync of cands + model):
  python3 scripts/rerank_strict1690_hybridqa_v1.py score \
      --cands /tmp/strict1690_cands_K10.jsonl \
      --model-dir /data/home/wangys/models/hybridqa_reranker_v1 \
      --out /tmp/strict1690_rerank_scores.jsonl \
      --device cuda:7 --batch-size 64

  # Local: evaluate
  python3 scripts/rerank_strict1690_hybridqa_v1.py eval \
      --cands /tmp/strict1690_cands_K10.jsonl \
      --scores /tmp/strict1690_rerank_scores.jsonl \
      --out analysis/ottqa_strict1690/recall_rerank.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Optional

import numpy as np


def cmd_build_cands(args):
    """Use existing hier candidates: top-K_tab tables × cell-linked URLs ∩ pool."""
    subset = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    queries_ibm = [json.loads(l) for l in args.queries_ibm.read_text().splitlines() if l.strip()]
    ibm_idx = {q["_id"]: i for i, q in enumerate(queries_ibm)}
    aligned = [(ibm_idx[r["question_id"]], r) for r in subset if r["question_id"] in ibm_idx]
    pr_rows = [r for _, r in aligned if r["kind"] == "passage_recoverable"]
    pr_qidx = np.array([ibm_idx[r["question_id"]] for r in pr_rows])

    ibm_corpus = [json.loads(l) for l in args.ibm_corpus.read_text().splitlines() if l.strip()]
    table_ids = [t["_id"] for t in ibm_corpus]
    tid_to_idx = {t: i for i, t in enumerate(table_ids)}
    ottqa_tables = json.loads(args.ottqa_tables.read_text())

    # 240K pool: pid → summary
    pool: dict[str, str] = {}
    with args.pool.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pid = (r.get("pid") or "").lower()
                if pid: pool[pid] = (r.get("summary") or "").strip()
    print(f"[load] pool: {len(pool)}", flush=True)

    # table_urls
    table_urls = {}
    for tid, t in ottqa_tables.items():
        if tid not in tid_to_idx: continue
        urls = set()
        for row in t.get("data", []):
            for cell in row:
                if isinstance(cell, list) and len(cell) >= 2:
                    for u in (cell[1] or []):
                        pid = u.lower()
                        if pid in pool: urls.add(pid)
        table_urls[tid_to_idx[tid]] = urls

    # Stage 1: table retrieval
    te = np.load(args.table_emb); te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-9)
    qe = np.load(args.query_emb); qe = qe / (np.linalg.norm(qe, axis=1, keepdims=True) + 1e-9)
    scores_t = qe[pr_qidx] @ te.T
    ranks_t = np.argsort(-scores_t, axis=1)[:, : args.K_tab]
    print(f"[stage1] table top-{args.K_tab} for {len(pr_rows)} queries", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0; cand_sizes = []
    with args.out.open("w") as f:
        for i, r in enumerate(pr_rows):
            top_tabs = ranks_t[i].tolist()
            cands = set()
            for tidx in top_tabs:
                cands |= table_urls.get(tidx, set())
            cand_list = sorted(cands)
            cand_sizes.append(len(cand_list))
            f.write(json.dumps({
                "qid": r["question_id"],
                "question": r["question"],
                "gold_pids": [p.lower() for p in r.get("gold_passage_pids", [])],
                "cands": [[c, pool[c][: args.passage_chars]] for c in cand_list],
            }, ensure_ascii=False) + "\n")
            n_total += len(cand_list)
    print(f"[done] {len(pr_rows)} queries, {n_total} total cand pairs, "
          f"avg {n_total/len(pr_rows):.0f}, med {int(np.median(cand_sizes))}", flush=True)


def cmd_score(args):
    """Load reranker, score each (q, passage) pair, write per-query top scores."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = torch.device(args.device)
    print(f"[load] tokenizer + model from {args.model_dir} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, torch_dtype=torch.float16
    ).to(device).eval()
    print(f"[load] model on {device}, params={sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)

    cands_lines = [json.loads(l) for l in args.cands.read_text().splitlines() if l.strip()]
    print(f"[load] queries: {len(cands_lines)}", flush=True)
    n_total = sum(len(r["cands"]) for r in cands_lines)
    print(f"[load] total pairs: {n_total}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out.open("w")
    t0 = time.time(); n_done = 0
    with torch.no_grad():
        for r in cands_lines:
            q = r["question"]
            pairs = [(q, c[1]) for c in r["cands"]]
            scores = []
            for s in range(0, len(pairs), args.batch_size):
                batch = pairs[s: s + args.batch_size]
                enc = tok([p[0] for p in batch], [p[1] for p in batch],
                          padding=True, truncation=True, max_length=512,
                          return_tensors="pt").to(device)
                logits = model(**enc).logits.squeeze(-1)
                scores.extend(logits.float().cpu().tolist())
            fout.write(json.dumps({
                "qid": r["qid"],
                "pids": [c[0] for c in r["cands"]],
                "scores": scores,
            }) + "\n"); fout.flush()
            n_done += len(pairs)
            if (cands_lines.index(r) + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  [{cands_lines.index(r)+1}/{len(cands_lines)}] "
                      f"{n_done}/{n_total} pairs  {el:.0f}s "
                      f"({n_done/el:.0f} pair/s)", flush=True)
    fout.close()
    print(f"[done] wrote scores to {args.out}", flush=True)


def cmd_eval(args):
    """Compute recall@K_pas using reranker scores."""
    cands_lines = {json.loads(l)["qid"]: json.loads(l)
                   for l in args.cands.read_text().splitlines() if l.strip()}
    scores_lines = [json.loads(l) for l in args.scores.read_text().splitlines() if l.strip()]
    Ks = [1, 3, 5, 10, 20]
    hits = {K: 0 for K in Ks}
    n = 0
    for r in scores_lines:
        c = cands_lines.get(r["qid"])
        if not c: continue
        n += 1
        gold = set(c["gold_pids"])
        if not gold: continue
        order = np.argsort(-np.array(r["scores"]))
        pids_ranked = [r["pids"][i] for i in order]
        for K in Ks:
            if set(pids_ranked[:K]) & gold:
                hits[K] += 1
    out = {
        "n_scored": n,
        "K_pas_list": Ks,
        "recall_at_K": {K: round(hits[K] / max(1, n), 4) for K in Ks},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    bc = sub.add_parser("build-cands")
    bc.add_argument("--subset", default="analysis/ottqa_dev_strict1690.jsonl", type=Path)
    bc.add_argument("--queries-ibm", default="data/ottqa_raw_full/dev_queries.jsonl", type=Path)
    bc.add_argument("--ibm-corpus", default="data/ottqa_raw_full/corpus_structure.jsonl", type=Path)
    bc.add_argument("--ottqa-tables", default="data/ottqa_repo/data/traindev_tables.json", type=Path)
    bc.add_argument("--table-emb", default="analysis/ottqa_small/table_embeddings.npy", type=Path)
    bc.add_argument("--query-emb", default="analysis/ottqa_small/query_embeddings.npy", type=Path)
    bc.add_argument("--pool", default="analysis/open_pool_full/passages.jsonl", type=Path)
    bc.add_argument("--K-tab", type=int, default=10)
    bc.add_argument("--passage-chars", type=int, default=1500)
    bc.add_argument("--out", required=True, type=Path)
    bc.set_defaults(fn=cmd_build_cands)

    sc = sub.add_parser("score")
    sc.add_argument("--cands", required=True, type=Path)
    sc.add_argument("--model-dir", required=True, type=Path)
    sc.add_argument("--out", required=True, type=Path)
    sc.add_argument("--device", default="cuda:0")
    sc.add_argument("--batch-size", type=int, default=64)
    sc.set_defaults(fn=cmd_score)

    ev = sub.add_parser("eval")
    ev.add_argument("--cands", required=True, type=Path)
    ev.add_argument("--scores", required=True, type=Path)
    ev.add_argument("--out", default="analysis/ottqa_strict1690/recall_rerank.json", type=Path)
    ev.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
