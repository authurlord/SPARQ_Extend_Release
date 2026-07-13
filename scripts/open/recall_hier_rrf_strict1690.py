#!/usr/bin/env python3
"""Hier passage retrieval with multi-leg RRF on strict-1690 subset.

Stage 1: top-K_tab IBM tables via dense bge-m3 (over 8,891 IBM corpus)
Stage 2: candidate passages = union of cell-linked URLs across top-K_tab,
         intersected with 240K HybridQA pool
Stage 3: rerank candidates by RRF fusion of:
  L1 = BM25  (over passage summary text)
  L2 = dense bge-m3 cos(q_emb, passage_emb)
  L3 = GNN   cos(q_lin(q_emb), passage_emb_gnn)
Report recall@K_pas for K_pas in {1, 3, 5, 10, 20}, target 80%/95%.

Per-leg ablation (single leg vs RRF) for paper table.
"""
from __future__ import annotations
import argparse, json, math, re, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def build_bm25_index(token_lists: list[list[str]], k1: float = 1.5, b: float = 0.75
                     ) -> tuple[csr_matrix, dict[str, int], np.ndarray]:
    """Returns (doc-token weight matrix (n_docs x n_vocab), vocab map, idf vector)."""
    N = len(token_lists)
    dl = np.array([len(d) for d in token_lists], dtype=np.float64)
    avgdl = float(dl.mean() or 1.0)
    vocab: dict[str, int] = {}
    df = Counter()
    for d in token_lists:
        for t in set(d): df[t] += 1
        for t in d:
            if t not in vocab: vocab[t] = len(vocab)
    V = len(vocab)
    idf = np.zeros(V, dtype=np.float64)
    for t, dft in df.items():
        idf[vocab[t]] = math.log((N - dft + 0.5) / (dft + 0.5) + 1.0)
    rows, cols, data = [], [], []
    for di, d in enumerate(token_lists):
        if not d: continue
        denom = k1 * (1 - b + b * len(d) / avgdl)
        tf_c = Counter(d)
        for t, tf in tf_c.items():
            ti = vocab[t]
            rows.append(di); cols.append(ti)
            data.append(tf * (k1 + 1) / (tf + denom))
    mat = csr_matrix((data, (rows, cols)), shape=(N, V), dtype=np.float64)
    return mat, vocab, idf


def rrf_fuse(rank_lists: list[list[int]], rrf_k: int = 60) -> list[int]:
    """Take a list of ranking lists, return a single fused ranking (best-first)."""
    score = defaultdict(float)
    for ranks in rank_lists:
        for k, idx in enumerate(ranks):
            score[idx] += 1.0 / (rrf_k + k + 1)
    return sorted(score.keys(), key=lambda x: -score[x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="analysis/ottqa_dev_strict1690.jsonl", type=Path)
    ap.add_argument("--ibm-corpus", default="data/ottqa_raw_full/corpus_structure.jsonl", type=Path)
    ap.add_argument("--ottqa-tables", default="data/ottqa_repo/data/traindev_tables.json", type=Path)
    ap.add_argument("--queries-ibm", default="data/ottqa_raw_full/dev_queries.jsonl", type=Path)
    ap.add_argument("--table-emb", default="analysis/ottqa_small/table_embeddings.npy", type=Path)
    ap.add_argument("--query-emb", default="analysis/ottqa_small/query_embeddings.npy", type=Path)
    ap.add_argument("--passage-emb", default="analysis/open_pool_full/passage_embeddings.npy", type=Path)
    ap.add_argument("--passage-emb-gnn", default="analysis/open_pool_full/passage_embeddings_gnn.npy", type=Path)
    ap.add_argument("--q-lin-w", default="analysis/open_pool_full/q_lin_weight.npy", type=Path)
    ap.add_argument("--q-lin-b", default="analysis/open_pool_full/q_lin_bias.npy", type=Path)
    ap.add_argument("--pool", default="analysis/open_pool_full/passages.jsonl", type=Path)
    ap.add_argument("--K-tabs", type=str, default="5,10,20",
                     help="comma-separated K_tab values")
    ap.add_argument("--out", default="analysis/ottqa_strict1690/recall_hier_rrf.json", type=Path)
    args = ap.parse_args()

    t0 = time.time()
    K_pas_list = [1, 3, 5, 10, 20]
    K_tab_list = [int(k) for k in args.K_tabs.split(",")]

    # ===== Load subset =====
    subset = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    queries_ibm = [json.loads(l) for l in args.queries_ibm.read_text().splitlines() if l.strip()]
    ibm_idx = {q["_id"]: i for i, q in enumerate(queries_ibm)}
    aligned = [(ibm_idx[r["question_id"]], r) for r in subset if r["question_id"] in ibm_idx]
    aligned_idx = np.array([i for i, _ in aligned])
    aligned_rows = [r for _, r in aligned]
    pr_rows = [r for r in aligned_rows if r["kind"] == "passage_recoverable"]
    pr_local_idx = [i for i, r in enumerate(aligned_rows) if r["kind"] == "passage_recoverable"]
    pr_global_qidx = np.array([ibm_idx[r["question_id"]] for r in pr_rows])
    print(f"[load] passage_recoverable: {len(pr_rows)}", flush=True)

    # ===== Tables + pool =====
    ibm_corpus = [json.loads(l) for l in args.ibm_corpus.read_text().splitlines() if l.strip()]
    table_ids = [t["_id"] for t in ibm_corpus]
    tid_to_idx = {t: i for i, t in enumerate(table_ids)}
    ottqa_tables = json.loads(args.ottqa_tables.read_text())

    pool_pids = []
    pool_summaries = []
    with args.pool.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pid = (r.get("pid") or "").lower()
                if not pid: continue
                pool_pids.append(pid)
                pool_summaries.append(r.get("summary", ""))
    pid_to_idx = {p: i for i, p in enumerate(pool_pids)}
    print(f"[load] ibm={len(ibm_corpus)} pool={len(pool_pids)}  {time.time()-t0:.0f}s", flush=True)

    # ===== Embeddings =====
    te = np.load(args.table_emb); te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-9)
    qe = np.load(args.query_emb); qe = qe / (np.linalg.norm(qe, axis=1, keepdims=True) + 1e-9)
    pe = np.load(args.passage_emb); pe = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-9)
    peg = np.load(args.passage_emb_gnn); peg = peg / (np.linalg.norm(peg, axis=1, keepdims=True) + 1e-9)
    q_lin_w = np.load(args.q_lin_w)
    q_lin_b = np.load(args.q_lin_b)
    print(f"[load] pe={pe.shape} peg={peg.shape} q_lin_w={q_lin_w.shape}  {time.time()-t0:.0f}s", flush=True)

    # ===== Stage 1: table retrieval =====
    pr_q = qe[pr_global_qidx]
    scores_t = pr_q @ te.T
    ranks_t = np.argsort(-scores_t, axis=1)[:, :max(K_tab_list)]
    print(f"[stage1] table top-{max(K_tab_list)}  {time.time()-t0:.0f}s", flush=True)

    # ===== Pre-extract table → cell-linked URLs (in pool) =====
    table_urls = {}
    for tid, t in ottqa_tables.items():
        if tid not in tid_to_idx: continue
        urls = set()
        for row in t.get("data", []):
            for cell in row:
                if isinstance(cell, list) and len(cell) >= 2:
                    for u in (cell[1] or []):
                        pid = u.lower()
                        if pid in pid_to_idx:
                            urls.add(pid)
        table_urls[tid_to_idx[tid]] = urls
    print(f"[pre] table→urls map: {len(table_urls)} tables  {time.time()-t0:.0f}s", flush=True)

    # ===== Build BM25 over WHOLE 240K pool (we just project later via cand idx) =====
    print(f"[bm25] tokenizing 240K passages...", flush=True)
    tok_re = re.compile(r"\w+")
    tok = lambda s: tok_re.findall(s.lower())
    doc_tokens = [tok(s) for s in pool_summaries]
    bm25_mat, vocab, idf = build_bm25_index(doc_tokens)
    print(f"[bm25] mat ready {bm25_mat.shape}  {time.time()-t0:.0f}s", flush=True)

    # ===== Project queries to GNN space =====
    pr_q_gnn = pr_q @ q_lin_w.T + q_lin_b
    pr_q_gnn = pr_q_gnn / (np.linalg.norm(pr_q_gnn, axis=1, keepdims=True) + 1e-9)

    # ===== Main loop: for each K_tab, score candidates, RRF =====
    out_results = []
    n_pr = len(pr_rows)
    for K_tab in K_tab_list:
        print(f"\n[K_tab={K_tab}] ...", flush=True)
        per_leg = {leg: {K: 0 for K in K_pas_list}
                   for leg in ["bm25", "dense", "gnn", "rrf3"]}
        gold_in_cands = 0
        cand_sizes = []

        for i, r in enumerate(pr_rows):
            top_tabs = ranks_t[i, :K_tab].tolist()
            cands = set()
            for tidx in top_tabs:
                cands |= table_urls.get(tidx, set())
            cand_list = sorted(cands)
            if not cand_list: continue
            cand_idx = np.array([pid_to_idx[c] for c in cand_list])
            cand_sizes.append(len(cand_idx))
            gold_pids = {p.lower() for p in r.get("gold_passage_pids", [])}
            gold_local = {ci for ci, pid in enumerate(cand_list) if pid in gold_pids}
            if gold_local: gold_in_cands += 1
            if not gold_local: continue

            # Leg 1: BM25
            q_toks = [vocab[t] for t in tok(r["question"]) if t in vocab]
            if q_toks:
                q_idf = np.zeros(bm25_mat.shape[1], dtype=np.float64)
                for ti in q_toks: q_idf[ti] = idf[ti]
                bm25_scores = (bm25_mat[cand_idx] @ q_idf).astype(np.float32)
            else:
                bm25_scores = np.zeros(len(cand_idx), dtype=np.float32)
            bm25_rank = np.argsort(-bm25_scores).tolist()
            # Leg 2: dense
            dense_scores = pe[cand_idx] @ pr_q[i]
            dense_rank = np.argsort(-dense_scores).tolist()
            # Leg 3: GNN
            gnn_scores = peg[cand_idx] @ pr_q_gnn[i]
            gnn_rank = np.argsort(-gnn_scores).tolist()
            # RRF 3-way
            rrf_rank = rrf_fuse([bm25_rank, dense_rank, gnn_rank])

            for K in K_pas_list:
                if any(x in gold_local for x in bm25_rank[:K]): per_leg["bm25"][K] += 1
                if any(x in gold_local for x in dense_rank[:K]): per_leg["dense"][K] += 1
                if any(x in gold_local for x in gnn_rank[:K]): per_leg["gnn"][K] += 1
                if any(x in gold_local for x in rrf_rank[:K]): per_leg["rrf3"][K] += 1

        avg_cands = float(np.mean(cand_sizes)) if cand_sizes else 0.0
        rec = {
            "K_tab": K_tab,
            "n_passage_recoverable": n_pr,
            "gold_in_candidates": gold_in_cands / n_pr,
            "avg_cand_pool_size": round(avg_cands, 1),
            "passage_recall": {
                leg: {K: round(per_leg[leg][K] / n_pr, 4) for K in K_pas_list}
                for leg in ["bm25", "dense", "gnn", "rrf3"]
            },
        }
        out_results.append(rec)
        print(f"[K_tab={K_tab}] avg_cands={avg_cands:.0f}  gold_in_cands={rec['gold_in_candidates']:.4f}")
        for leg in ["bm25", "dense", "gnn", "rrf3"]:
            print(f"  {leg:6s}: " + "  ".join(f"@{K}={rec['passage_recall'][leg][K]:.4f}" for K in K_pas_list))

    out = {
        "n_passage_recoverable": n_pr,
        "K_tab_list": K_tab_list,
        "K_pas_list": K_pas_list,
        "results": out_results,
        "wall_sec": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n=== Hier-RRF passage recall on 1690 ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
