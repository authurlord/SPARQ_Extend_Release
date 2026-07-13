#!/usr/bin/env python3
"""Properly evaluate trained OTT-QA GNN + Qwen3 baseline on dev_linked queries.

Loads:
  - hetero/bipartite graph
  - trained GNN checkpoint (models/ottqa_query_table_gnn/best.pt)
  - dev_linked-aligned query embeddings
Computes:
  - GNN-only top-K recall
  - Qwen3 dense baseline top-K (same query/table embeddings, no GNN)
  - RRF(GNN, Qwen3) top-K
  - bge-m3 baseline (using existing analysis/ottqa_small/table_embeddings.npy + ottqa_small/query_embeddings.npy
    but the latter is in IBM order — need to re-align with dev_linked)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_query_table_gnn import TableRetrievalGNN


def rrf_fuse_ranks(ranks_list: list[np.ndarray], rrf_k: int = 60) -> np.ndarray:
    """Each ranks: (N_q, K). Return RRF-fused argsort, shape (N_q, max(K))."""
    n_q = ranks_list[0].shape[0]
    n_tables = 8891  # known
    fused = np.zeros((n_q, n_tables), dtype=np.float64)
    for ranks in ranks_list:
        for i in range(n_q):
            for k, idx in enumerate(ranks[i]):
                fused[i, idx] += 1.0 / (rrf_k + k + 1)
    return np.argsort(-fused, axis=1)


def recall_at(ranks: np.ndarray, labels: np.ndarray, Ks: list[int]) -> dict:
    out = {}
    for K in Ks:
        out[K] = float((ranks[:, :K] == labels[:, None]).any(axis=1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="analysis/ottqa_open_pool/bipartite_graph.pt", type=Path)
    ap.add_argument("--ckpt", default="models/ottqa_query_table_gnn/best.pt", type=Path)
    ap.add_argument("--dev-json", default="/tmp/ottqa_dev.json", type=Path)
    ap.add_argument("--dev-q-emb", default="analysis/ottqa_open_pool/dev_query_embeddings_linked_qwen3.npy",
                     type=Path)
    ap.add_argument("--bge-table-emb", default="analysis/ottqa_small/table_embeddings.npy", type=Path)
    ap.add_argument("--bge-query-emb-ibm", default="analysis/ottqa_small/query_embeddings.npy", type=Path)
    ap.add_argument("--queries-ibm", default="data/ottqa_raw_full/dev_queries.jsonl", type=Path)
    ap.add_argument("--out", default="analysis/ottqa_strict1690/recall_stage1_gnn.json", type=Path)
    args = ap.parse_args()

    Ks = [1, 3, 5, 10, 20, 50, 100]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph
    saved = torch.load(args.graph, weights_only=False)
    data = saved["data"].to(device)
    table_ids: list = saved["table_ids"]
    tid_to_idx = {t: i for i, t in enumerate(table_ids)}
    metadata = data.metadata()
    print(f"[load] graph: tables={len(table_ids)}", flush=True)

    # Dev linked
    dev = json.load(args.dev_json.open())
    dev_qids = [d["question_id"] for d in dev]
    dev_labels = np.array([tid_to_idx[d["table_id"]] for d in dev], dtype=np.int64)
    print(f"[load] dev: {len(dev)}", flush=True)

    # Qwen3 dev queries in dev_linked order
    dev_q_qwen3 = np.load(args.dev_q_emb).astype(np.float32)
    dev_q_qwen3 = dev_q_qwen3 / (np.linalg.norm(dev_q_qwen3, axis=1, keepdims=True) + 1e-9)
    print(f"[load] qwen3 dev q: {dev_q_qwen3.shape}", flush=True)

    # ===== GNN =====
    ckpt = torch.load(args.ckpt, weights_only=False)
    in_dim = data["table"].x.shape[1]
    margs = ckpt["args"]
    model = TableRetrievalGNN(metadata, in_dim=in_dim,
                                hidden=margs["hidden"],
                                n_layers=margs["n_layers"],
                                n_heads=margs["n_heads"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    print(f"[load] GNN ckpt: best_top5_in_train={ckpt.get('best_top5', 0):.4f}", flush=True)

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        table_out = F.normalize(out["table"], dim=-1)  # (N_tab, D)
        q_t = torch.from_numpy(dev_q_qwen3).float().to(device)
        q_proj = F.normalize(model.encode_query(q_t), dim=-1)
        scores_gnn = (q_proj @ table_out.t()).cpu().numpy()  # (N_q, N_tab)
        ranks_gnn = np.argsort(-scores_gnn, axis=1)[:, :max(Ks)]

    # ===== Qwen3 dense baseline (table_emb from graph node features) =====
    table_x = F.normalize(data["table"].x, dim=-1).cpu().numpy()
    scores_qwen3 = dev_q_qwen3 @ table_x.T
    ranks_qwen3 = np.argsort(-scores_qwen3, axis=1)[:, :max(Ks)]

    # ===== bge-m3 baseline (existing ottqa_small assets) — need to align dev order =====
    bge_table = np.load(args.bge_table_emb)
    bge_table = bge_table / (np.linalg.norm(bge_table, axis=1, keepdims=True) + 1e-9)
    bge_q_ibm = np.load(args.bge_query_emb_ibm)  # in IBM order
    bge_q_ibm = bge_q_ibm / (np.linalg.norm(bge_q_ibm, axis=1, keepdims=True) + 1e-9)
    queries_ibm = [json.loads(l) for l in args.queries_ibm.read_text().splitlines() if l.strip()]
    ibm_qid_to_idx = {q["_id"]: i for i, q in enumerate(queries_ibm)}
    # Reorder bge dev queries to match dev_linked order
    bge_q_devlinked = np.array(
        [bge_q_ibm[ibm_qid_to_idx[qid]] for qid in dev_qids if qid in ibm_qid_to_idx],
        dtype=np.float32
    )
    # bge tables are in IBM corpus order (8891), which matches our graph table_ids
    # NOTE: graph table_ids built from IBM corpus order too — verify
    # Both should be same order
    scores_bge = bge_q_devlinked @ bge_table.T
    ranks_bge = np.argsort(-scores_bge, axis=1)[:, :max(Ks)]
    # If alignment dropped some queries, handle
    if bge_q_devlinked.shape[0] != len(dev):
        print(f"[warn] bge dev mismatch: {bge_q_devlinked.shape[0]} vs {len(dev)}", flush=True)

    # ===== RRF(GNN, Qwen3) and RRF(GNN, bge-m3) =====
    ranks_rrf_gnn_qwen3 = rrf_fuse_ranks([ranks_gnn, ranks_qwen3])[:, :max(Ks)]
    ranks_rrf_gnn_bge = rrf_fuse_ranks([ranks_gnn, ranks_bge])[:, :max(Ks)]
    ranks_rrf_3way = rrf_fuse_ranks([ranks_gnn, ranks_qwen3, ranks_bge])[:, :max(Ks)]

    # ===== Eval =====
    results = {
        "GNN": recall_at(ranks_gnn, dev_labels, Ks),
        "Qwen3-dense": recall_at(ranks_qwen3, dev_labels, Ks),
        "bge-m3-dense": recall_at(ranks_bge, dev_labels, Ks),
        "RRF(GNN+Qwen3)": recall_at(ranks_rrf_gnn_qwen3, dev_labels, Ks),
        "RRF(GNN+bge-m3)": recall_at(ranks_rrf_gnn_bge, dev_labels, Ks),
        "RRF(GNN+Qwen3+bge-m3)": recall_at(ranks_rrf_3way, dev_labels, Ks),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n=== Stage 1 recall on full 2,214 dev ===")
    print(f"{'':24s}" + "    ".join(f"@{K:3d}" for K in Ks))
    for name, r in results.items():
        print(f"  {name:24s}" + "    ".join(f"{r[K]*100:5.1f}" for K in Ks))

    # Also evaluate on 1690 strict subset
    sub_path = Path("analysis/ottqa_dev_strict1690.jsonl")
    if sub_path.exists():
        sub_qids = {json.loads(l)["question_id"] for l in sub_path.read_text().splitlines() if l.strip()}
        sub_mask = np.array([qid in sub_qids for qid in dev_qids])
        sub_labels = dev_labels[sub_mask]
        all_ranks = {
            "GNN": ranks_gnn, "Qwen3-dense": ranks_qwen3,
            "bge-m3-dense": ranks_bge,
            "RRF(GNN+Qwen3)": ranks_rrf_gnn_qwen3,
            "RRF(GNN+bge-m3)": ranks_rrf_gnn_bge,
            "RRF(GNN+Qwen3+bge-m3)": ranks_rrf_3way,
        }
        sub_results = {name: recall_at(r[sub_mask], sub_labels, Ks)
                        for name, r in all_ranks.items()}
        sub_out_path = args.out.parent / f"{args.out.stem}_1690.json"
        sub_out_path.write_text(json.dumps(sub_results, indent=2))
        print(f"\n=== Stage 1 recall on 1690 strict subset ===")
        print(f"{'':24s}" + "    ".join(f"@{K:3d}" for K in Ks))
        for name, r in sub_results.items():
            print(f"  {name:24s}" + "    ".join(f"{r[K]*100:5.1f}" for K in Ks))


if __name__ == "__main__":
    main()
