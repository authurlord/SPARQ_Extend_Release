#!/usr/bin/env python3
"""HybridQA Stage 1 3-leg RRF v2 with retrained GNN (Qwen3-instruct features).

Legs:
  A: Qwen3-Embed-Instruct dense (existing)
  B: BM25 (existing)
  C: GNN-refined Qwen3 (new: models/hybridqa_table_gnn_qwen3_instruct/best.pt)

Outputs:
  - analysis/open_pool/hybridqa_gnn_v2_table_ranks_dev.npy  (3466, 100)
  - analysis/open_pool/hybridqa_rrf3v2_table_ranks_dev.npy  (3466, 100) — 3-leg
  - analysis/qa_verification_stage1/recall_3leg_v2.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_query_table_gnn import TableRetrievalGNN


def rrf_fuse(ranks_list, n_tab, rrf_k=60, topK=100):
    n_q = ranks_list[0].shape[0]
    fused = np.zeros((n_q, n_tab), dtype=np.float64)
    for ranks in ranks_list:
        for i in range(n_q):
            for k, idx in enumerate(ranks[i]):
                fused[i, idx] += 1.0 / (rrf_k + k + 1)
    return np.argsort(-fused, axis=1)[:, :topK]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="analysis/open_pool/bipartite_graph_qwen3.pt", type=Path)
    ap.add_argument("--ckpt", default="models/hybridqa_table_gnn_qwen3_instruct/best.pt", type=Path)
    ap.add_argument("--pool-dir", default="analysis/open_pool", type=Path)
    ap.add_argument("--dev-q-emb",
                     default="analysis/hybridqa_dev_query_qwen3_instruct.npy", type=Path)
    ap.add_argument("--dev-q-qids",
                     default="analysis/hybridqa_dev_query_qwen3_instruct.qids.json", type=Path)
    ap.add_argument("--dev-json", default="data/hybridqa_raw/dev.json", type=Path)
    ap.add_argument("--subset-487",
                     default="analysis/hybridqa_verifier_subset_487.jsonl", type=Path)
    ap.add_argument("--topK-save", default=100, type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out",
                     default="analysis/qa_verification_stage1/recall_3leg_v2.json",
                     type=Path)
    args = ap.parse_args()

    # ===== Load graph + GNN =====
    saved = torch.load(args.graph, weights_only=False)
    data = saved["data"]
    table_ids = saved["table_ids"]
    tid_to_idx = {t: i for i, t in enumerate(table_ids)}
    n_tab = len(table_ids)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    ckpt = torch.load(args.ckpt, weights_only=False, map_location=device)
    margs = ckpt["args"]
    in_dim = data["table"].x.shape[1]
    model = TableRetrievalGNN(data.metadata(), in_dim=in_dim,
                              hidden=margs["hidden"],
                              n_layers=margs["n_layers"],
                              n_heads=margs["n_heads"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    print(f"[load] GNN ckpt: best_sub487_top1={ckpt.get('best_sub487_top1', 0):.4f}",
          flush=True)

    # ===== GNN forward =====
    dev_q_emb = np.load(args.dev_q_emb).astype(np.float32)
    dev_qids = json.loads(args.dev_q_qids.read_text())
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        table_out = F.normalize(out["table"], dim=-1)
        q_t = torch.from_numpy(
            dev_q_emb / (np.linalg.norm(dev_q_emb, axis=1, keepdims=True) + 1e-9)
        ).float().to(device)
        q_proj = F.normalize(model.encode_query(q_t), dim=-1)
        scores_gnn = (q_proj @ table_out.t()).cpu().numpy()
        ranks_gnn = np.argsort(-scores_gnn, axis=1)[:, :args.topK_save]
    np.save(args.pool_dir / "hybridqa_gnn_v2_table_ranks_dev.npy", ranks_gnn)
    print(f"[save] GNN v2 ranks", flush=True)

    # ===== Existing ranks =====
    ranks_qwen3 = np.load(args.pool_dir / "hybridqa_qwen3_table_ranks_dev.npy")[:, :args.topK_save]
    ranks_bm25 = np.load(args.pool_dir / "hybridqa_bm25_table_ranks_dev.npy")[:, :args.topK_save]

    # ===== Labels =====
    dev = json.load(args.dev_json.open())
    qid_to_gold = {q["question_id"]: q["table_id"] for q in dev}
    gold_idx_arr = np.array([tid_to_idx.get(qid_to_gold.get(q, ""), -1)
                              for q in dev_qids], dtype=np.int64)
    valid = gold_idx_arr >= 0
    sub_qids = {json.loads(l)["question_id"] for l in
                args.subset_487.read_text().splitlines() if l.strip()}
    mask487 = np.array([q in sub_qids for q in dev_qids]) & valid

    Ks = [1, 3, 5, 10, 20, 50, 100]
    def recall_at(ranks, mask):
        r = ranks[mask]; g = gold_idx_arr[mask]
        return {k: float(((r[:, :k] == g[:, None]).any(axis=1)).mean()) for k in Ks}

    # ===== Fusions =====
    combos = {
        "GNN_v2": [ranks_gnn],
        "Qwen3-instruct": [ranks_qwen3],
        "BM25": [ranks_bm25],
        "RRF(Qwen3+BM25)": [ranks_qwen3, ranks_bm25],
        "RRF(Qwen3+GNN_v2)": [ranks_qwen3, ranks_gnn],
        "RRF(BM25+GNN_v2)": [ranks_bm25, ranks_gnn],
        "RRF(Qwen3+BM25+GNN_v2)": [ranks_qwen3, ranks_bm25, ranks_gnn],
    }
    results = {"Ks": Ks, "full_dev_3466": {}, "dev487": {}}
    for name, legs in combos.items():
        if len(legs) == 1:
            ranks = legs[0]
        else:
            ranks = rrf_fuse(legs, n_tab, topK=args.topK_save)
        results["full_dev_3466"][name] = recall_at(ranks, valid)
        results["dev487"][name] = recall_at(ranks, mask487)
        if name == "RRF(Qwen3+BM25+GNN_v2)":
            np.save(args.pool_dir / "hybridqa_rrf3v2_table_ranks_dev.npy", ranks)
            print(f"[save] 3-leg-v2 ranks", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n=== HybridQA Stage 1 3-leg-v2 (GNN retrained) ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
