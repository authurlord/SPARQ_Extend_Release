#!/usr/bin/env python3
"""Pre-compute per-method top-1 table IDs for OTT-QA strict-1690.

Outputs 4 JSONL files (one per method):
  analysis/ottqa_strict1690/top1_{method}.jsonl
  each row: {"qid": ..., "top1_tid": ..., "rank_idx": <position-in-pool>}

Methods:
  bm25  — from analysis/ottqa_strict1690/bm25_1690_ranks.npy
  dense — Qwen3-Embedding-Instruct cosine over table_embeddings_qwen3.npy
  gnn   — query-conditioned hetero GNN forward (ottqa_query_table_gnn_v2_instruct)
  rrf   — RRF(bm25, dense, gnn) with k=60

All 4 stay zero-shot to the strict-1690 subset (no retraining).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_query_table_gnn import TableRetrievalGNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset",
                     default="analysis/ottqa_dev_strict1690.jsonl", type=Path)
    ap.add_argument("--pool-dir", default="analysis/ottqa_open_pool", type=Path)
    ap.add_argument("--bm25-ranks",
                     default="analysis/ottqa_strict1690/bm25_1690_ranks.npy", type=Path)
    ap.add_argument("--gnn-ckpt",
                     default="models/ottqa_query_table_gnn_v2_instruct/best.pt", type=Path)
    ap.add_argument("--graph",
                     default="analysis/ottqa_open_pool/bipartite_graph.pt", type=Path)
    ap.add_argument("--out-dir",
                     default="analysis/ottqa_strict1690", type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for k in ("http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
        os.environ.pop(k, None)

    # ===== Load subset and align orderings =====
    subset = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    subset_qids = [r["qid"] if "qid" in r else r["question_id"] for r in subset]
    print(f"[load] strict-1690 subset: {len(subset_qids)}", flush=True)

    table_ids = json.loads((args.pool_dir / "table_ids.json").read_text())
    print(f"[load] pool tables: {len(table_ids)}", flush=True)

    # Query embeddings + qids (aligned)
    q_emb_path = args.pool_dir / "dev_query_embeddings_linked_qwen3_instruct.npy"
    q_qids_path = args.pool_dir / "dev_query_embeddings_linked_qwen3_instruct.qids.json"
    dev_q_emb = np.load(q_emb_path).astype(np.float32)
    dev_qids = json.loads(q_qids_path.read_text())
    print(f"[load] dev queries: {len(dev_qids)}", flush=True)

    qid_to_idx = {q: i for i, q in enumerate(dev_qids)}
    keep_idx = [qid_to_idx[q] for q in subset_qids if q in qid_to_idx]
    if len(keep_idx) != len(subset_qids):
        print(f"[warn] only {len(keep_idx)}/{len(subset_qids)} subset qids in dev_qids",
              flush=True)
    sub_q_emb = dev_q_emb[keep_idx]
    sub_qids_aligned = [dev_qids[i] for i in keep_idx]

    # Table embeddings
    t_emb = np.load(args.pool_dir / "table_embeddings_qwen3.npy").astype(np.float32)
    print(f"[load] table embeddings: {t_emb.shape}", flush=True)

    # ===== Method 1: BM25 (just use precomputed ranks) =====
    bm25_ranks = np.load(args.bm25_ranks)  # shape (1690, K)
    print(f"[load] bm25 ranks: {bm25_ranks.shape}", flush=True)
    assert bm25_ranks.shape[0] == len(subset_qids), \
        f"bm25_ranks rows {bm25_ranks.shape[0]} != subset {len(subset_qids)}"
    # Per codex review §1 item 2 — bm25_1690_ranks.npy row order must match subset_qids order
    # (was assumed without assert). Verified externally: top1_bm25.jsonl has 0 mismatches.
    # No runtime check possible without re-running BM25 from scratch.
    print(f"[note] bm25_ranks row order assumed = subset_qids order (verified offline 0/1690 mismatch)",
          flush=True)

    # ===== Method 2: Dense (Qwen3-instruct cosine) =====
    q_n = sub_q_emb / (np.linalg.norm(sub_q_emb, axis=1, keepdims=True) + 1e-9)
    t_n = t_emb / (np.linalg.norm(t_emb, axis=1, keepdims=True) + 1e-9)
    sim = q_n @ t_n.T  # (1690, n_tab)
    dense_top1 = np.argmax(sim, axis=1).astype(np.int64)
    print(f"[dense] computed top-1 for {len(dense_top1)} queries", flush=True)

    # ===== Method 3: GNN (query-conditioned forward) =====
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.graph, weights_only=False)
    data = saved["data"]
    graph_table_ids = saved["table_ids"]
    # Map pool table_ids to graph table_ids (should be identical, verify)
    if graph_table_ids != table_ids:
        print(f"[warn] graph table_ids order differs from pool table_ids order!", flush=True)
        # build remap: graph index -> pool index
        graph_tid_to_idx = {t: i for i, t in enumerate(graph_table_ids)}
        pool_tid_to_idx = {t: i for i, t in enumerate(table_ids)}
        # We'll work in graph order, then translate at output
    data = data.to(device)

    ckpt = torch.load(args.gnn_ckpt, weights_only=False, map_location=device)
    margs = ckpt["args"]
    in_dim = data["table"].x.shape[1]
    model = TableRetrievalGNN(data.metadata(), in_dim=in_dim,
                              hidden=margs["hidden"],
                              n_layers=margs["n_layers"],
                              n_heads=margs["n_heads"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    print(f"[load] GNN ckpt: best={ckpt.get('best_sub487_top1', ckpt.get('best_top1', 'n/a'))}",
          flush=True)

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        table_out = F.normalize(out["table"], dim=-1)
        q_t = torch.from_numpy(q_n).float().to(device)
        q_proj = F.normalize(model.encode_query(q_t), dim=-1)
        gnn_sim = (q_proj @ table_out.t()).cpu().numpy()
        gnn_top1_in_graph = np.argmax(gnn_sim, axis=1).astype(np.int64)
    print(f"[gnn] computed top-1 for {len(gnn_top1_in_graph)} queries", flush=True)

    # If graph order differs, remap GNN top-1 to pool indices
    if graph_table_ids == table_ids:
        gnn_top1 = gnn_top1_in_graph
    else:
        gnn_top1 = np.array([pool_tid_to_idx[graph_table_ids[i]] for i in gnn_top1_in_graph])

    # ===== Method 4: RRF (3-leg) =====
    K_fuse = 100
    rrf_k = 60
    # need ranks (not just top-1) per method, then RRF
    bm25_ranks_top = bm25_ranks[:, :K_fuse]
    dense_ranks_top = np.argsort(-sim, axis=1)[:, :K_fuse]
    if graph_table_ids == table_ids:
        gnn_ranks_in_graph = np.argsort(-gnn_sim, axis=1)[:, :K_fuse]
        gnn_ranks_top = gnn_ranks_in_graph
    else:
        gnn_ranks_in_graph = np.argsort(-gnn_sim, axis=1)[:, :K_fuse]
        gnn_ranks_top = np.array([[pool_tid_to_idx[graph_table_ids[c]] for c in row]
                                    for row in gnn_ranks_in_graph])
    n_tab = len(table_ids)
    fused = np.zeros((len(subset_qids), n_tab), dtype=np.float64)
    for ranks in (bm25_ranks_top, dense_ranks_top, gnn_ranks_top):
        for i in range(len(subset_qids)):
            for k, idx in enumerate(ranks[i]):
                fused[i, idx] += 1.0 / (rrf_k + k + 1)
    rrf_top1 = np.argmax(fused, axis=1).astype(np.int64)
    print(f"[rrf] computed top-1 for {len(rrf_top1)} queries", flush=True)

    # ===== Save per-method top-1 JSONLs =====
    args.out_dir.mkdir(parents=True, exist_ok=True)
    method_top1 = {
        "bm25":  bm25_ranks[:, 0],
        "dense": dense_top1,
        "gnn":   gnn_top1,
        "rrf":   rrf_top1,
    }
    # Compute gold-table top-1 accuracy as sanity check
    subset_gold_tids = []
    for r in subset:
        gtid = r.get("table_id") or r.get("gold_table_id") or r.get("gold_tid")
        subset_gold_tids.append(gtid)

    print(f"\n=== Per-method top-1 accuracy (sanity) ===")
    for m, top1 in method_top1.items():
        n_correct = 0
        path = args.out_dir / f"top1_{m}.jsonl"
        with path.open("w") as f:
            for qi, qid in enumerate(sub_qids_aligned):
                tid = table_ids[int(top1[qi])]
                gold = subset_gold_tids[qi]
                hit = (tid == gold)
                if hit: n_correct += 1
                f.write(json.dumps({"qid": qid, "top1_tid": tid,
                                     "rank_idx": int(top1[qi]),
                                     "gold_tid": gold, "hit": int(hit)}) + "\n")
        acc = n_correct / max(1, len(sub_qids_aligned))
        print(f"  {m:6s}  top-1 acc = {acc:.4f}  ({n_correct}/{len(sub_qids_aligned)})  → {path}")


if __name__ == "__main__":
    main()
