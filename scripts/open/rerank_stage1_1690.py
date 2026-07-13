#!/usr/bin/env python3
"""Apply OTT-QA table reranker to top-20 candidates from 3-leg RRF on 1690 subset.

Loads:
  - 3-leg RRF ranks (or recomputes from GNN+Qwen3+BM25 caches)
  - Reranker model (models/ottqa_table_reranker_v1/ or specified)
  - traindev_tables.json + IBM corpus for table text rendering

Process per 1690 query:
  1. Take top-20 candidate table indices from 3-leg RRF
  2. Build (query, table_text) pairs with query-token-selected rows
     (same format as training)
  3. Score with cross-encoder
  4. Re-rank → report recall@1/3/5/10/20

Outputs:
  analysis/ottqa_strict1690/recall_reranked_table_v1.json
  analysis/ottqa_strict1690/reranked_cands_v1.jsonl    (per-query cand list w/ scores)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_query_table_gnn import TableRetrievalGNN
from prepare_ottqa_table_reranker_data import (
    build_table_text, tokenize,
)


def rrf_fuse(ranks_list: list[np.ndarray], n_tables: int, rrf_k: int = 60) -> np.ndarray:
    n_q = ranks_list[0].shape[0]
    fused = np.zeros((n_q, n_tables), dtype=np.float64)
    for ranks in ranks_list:
        for i in range(n_q):
            for k, idx in enumerate(ranks[i]):
                fused[i, idx] += 1.0 / (rrf_k + k + 1)
    return np.argsort(-fused, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="analysis/ottqa_open_pool/bipartite_graph.pt", type=Path)
    ap.add_argument("--gnn-ckpt",
                     default="models/ottqa_query_table_gnn_v2_instruct/best.pt", type=Path,
                     help="which GNN ckpt to use for the GNN leg. "
                          "Default = v2_instruct (retrained 2026-05-20 with instructed train queries "
                          "and dev cache alignment fix). Codex §6 flagged stale default to old ckpt.")
    ap.add_argument("--dev-json", default="/tmp/ottqa_dev.json", type=Path)
    ap.add_argument("--dev-q-emb",
                     default="analysis/ottqa_open_pool/dev_query_embeddings_linked_qwen3_instruct.npy",
                     type=Path)
    ap.add_argument("--bm25-ranks",
                     default="analysis/ottqa_strict1690/bm25_dev_full_ranks.npy", type=Path)
    ap.add_argument("--subset", default="analysis/ottqa_dev_strict1690.jsonl", type=Path)
    ap.add_argument("--ibm-corpus", default="data/ottqa_raw_full/corpus_structure.jsonl",
                     type=Path)
    ap.add_argument("--ottqa-tables", default="data/ottqa_repo/data/traindev_tables.json",
                     type=Path)
    ap.add_argument("--reranker-dir", default="models/ottqa_table_reranker_v1", type=Path)
    ap.add_argument("--topK-rerank", type=int, default=20,
                     help="how many top tables from 3-leg RRF to rerank")
    ap.add_argument("--K-rows", type=int, default=5,
                     help="row count in table text (must match training)")
    ap.add_argument("--max-text-chars", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-summary",
                     default="analysis/ottqa_strict1690/recall_reranked_table_v1.json",
                     type=Path)
    ap.add_argument("--out-jsonl",
                     default="analysis/ottqa_strict1690/reranked_cands_v1.jsonl",
                     type=Path)
    args = ap.parse_args()

    Ks = [1, 3, 5, 10, 20]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ===== Load graph + table_ids + GNN ckpt =====
    saved = torch.load(args.graph, weights_only=False, map_location=device)
    data = saved["data"].to(device)
    table_ids: list = saved["table_ids"]
    tid_to_idx = {t: i for i, t in enumerate(table_ids)}
    n_tab = len(table_ids)
    print(f"[load] graph: tables={n_tab}", flush=True)

    dev = json.load(args.dev_json.open())
    dev_qids = [d["question_id"] for d in dev]
    dev_qid_to_devidx = {q: i for i, q in enumerate(dev_qids)}
    dev_labels = np.array([tid_to_idx[d["table_id"]] for d in dev], dtype=np.int64)
    print(f"[load] dev: {len(dev)}", flush=True)

    dev_q = np.load(args.dev_q_emb).astype(np.float32)
    assert dev_q.shape[0] == len(dev), \
        f"dev_q_emb rows {dev_q.shape[0]} != dev_linked {len(dev)} — codex §6 alignment check"
    dev_q = dev_q / (np.linalg.norm(dev_q, axis=1, keepdims=True) + 1e-9)

    # ===== GNN forward =====
    ckpt = torch.load(args.gnn_ckpt, weights_only=False, map_location=device)
    in_dim = data["table"].x.shape[1]
    margs = ckpt["args"]
    model = TableRetrievalGNN(data.metadata(), in_dim=in_dim,
                              hidden=margs["hidden"],
                              n_layers=margs["n_layers"],
                              n_heads=margs["n_heads"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        table_out = F.normalize(out["table"], dim=-1)
        q_t = torch.from_numpy(dev_q).float().to(device)
        q_proj = F.normalize(model.encode_query(q_t), dim=-1)
        scores_gnn = (q_proj @ table_out.t()).cpu().numpy()
        ranks_gnn = np.argsort(-scores_gnn, axis=1)
    table_x = F.normalize(data["table"].x, dim=-1).cpu().numpy()
    scores_qwen3 = dev_q @ table_x.T
    ranks_qwen3 = np.argsort(-scores_qwen3, axis=1)
    bm25_ranks = np.load(args.bm25_ranks)
    assert bm25_ranks.shape[0] == len(dev), \
        f"bm25_ranks rows {bm25_ranks.shape[0]} != dev_linked {len(dev)} — codex §6 alignment check"

    # ===== 3-leg RRF on full dev =====
    K_input = 100
    print(f"[rrf] fusing 3 legs top-{K_input}...", flush=True)
    ranks_rrf = rrf_fuse(
        [ranks_gnn[:, :K_input], ranks_qwen3[:, :K_input], bm25_ranks[:, :K_input]],
        n_tab
    )

    # ===== Slice to 1690 =====
    subset = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    sub_devidx = np.array(
        [dev_qid_to_devidx[r["question_id"]] for r in subset
         if r["question_id"] in dev_qid_to_devidx], dtype=np.int64
    )
    sub_labels = dev_labels[sub_devidx]
    sub_ranks_rrf = ranks_rrf[sub_devidx]
    print(f"[subset] 1690: {len(sub_devidx)}", flush=True)

    # ===== Load tables and reranker =====
    print(f"[load] ottqa_tables...", flush=True)
    tables_json = json.loads(args.ottqa_tables.read_text())
    print(f"[load] tables_json: {len(tables_json)}", flush=True)

    print(f"[load] reranker from {args.reranker_dir}...", flush=True)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(args.reranker_dir)
    rmodel = AutoModelForSequenceClassification.from_pretrained(
        args.reranker_dir, torch_dtype=torch.float16
    ).to(device).eval()

    # ===== Score top-K per query =====
    K_re = args.topK_rerank
    print(f"[score] reranking top-{K_re} for {len(sub_devidx)} queries...", flush=True)
    t0 = time.time()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out_jsonl.open("w")
    reranked_top1 = []
    reranked_topK_hits = {K: 0 for K in Ks}
    # Recall: also report the pre-rerank baseline on the same 1690 for comparison
    pre_topK_hits = {K: 0 for K in Ks}
    n_missing_table = 0  # codex §7: track missing candidate tables

    with torch.no_grad():
        for i, dev_idx in enumerate(sub_devidx):
            sub_idx = i
            cand_tids = [int(idx) for idx in sub_ranks_rrf[i, :K_re]]
            cand_table_ids = [table_ids[ci] for ci in cand_tids]
            q = dev[dev_idx]["question"]
            q_tokens = tokenize(q)
            # Build table texts (same format as training).
            # Codex §7: count missing tables (should be zero for full coverage)
            texts = []
            for j in range(K_re):
                t = tables_json.get(cand_table_ids[j])
                if t is None:
                    n_missing_table += 1
                    t = {}
                texts.append(
                    build_table_text(cand_table_ids[j], t, q_tokens,
                                      K_rows=args.K_rows, max_chars=args.max_text_chars)
                )
            # Pre-rerank counts (RRF order)
            gold_idx = int(sub_labels[i])
            pre_ranked = cand_tids
            for K in Ks:
                if gold_idx in pre_ranked[:K]:
                    pre_topK_hits[K] += 1

            # Reranker scoring
            scores = []
            for s in range(0, K_re, args.batch_size):
                bq = [q] * min(args.batch_size, K_re - s)
                bp = texts[s: s + args.batch_size]
                enc = tok(bq, bp, padding=True, truncation=True, max_length=512,
                          return_tensors="pt").to(device)
                logits = rmodel(**enc).logits.squeeze(-1)
                scores.extend(logits.float().cpu().tolist())
            order = np.argsort(-np.array(scores))
            re_ranked = [cand_tids[j] for j in order]
            for K in Ks:
                if gold_idx in re_ranked[:K]:
                    reranked_topK_hits[K] += 1

            # Save per-query
            fout.write(json.dumps({
                "qid": dev[dev_idx]["question_id"],
                "question": q,
                "gold_table_id": dev[dev_idx]["table_id"],
                "rrf_cand_table_ids": cand_table_ids,
                "rerank_scores": scores,
                "rerank_order_idx": order.tolist(),
                "reranked_table_ids": [cand_table_ids[j] for j in order],
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(sub_devidx)}] {el:.0f}s "
                      f"({(i+1)/max(1,el):.1f} q/s)", flush=True)
    fout.close()

    n = len(sub_devidx)
    summary = {
        "n_total": int(n),
        "topK_rerank": K_re,
        "K_rows": args.K_rows,
        "gnn_ckpt": str(args.gnn_ckpt),
        "reranker_dir": str(args.reranker_dir),
        "n_missing_candidate_table": int(n_missing_table),
        "recall_at_K_pre_rerank_1690": {K: round(pre_topK_hits[K]/n, 4) for K in Ks},
        "recall_at_K_after_rerank_1690": {K: round(reranked_topK_hits[K]/n, 4) for K in Ks},
    }
    if n_missing_table > 0:
        print(f"[warn] codex §7: {n_missing_table} candidate tables were missing "
              f"from traindev_tables.json (built empty text)", flush=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {args.out_summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
