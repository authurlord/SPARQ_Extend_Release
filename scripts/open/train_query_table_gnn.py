#!/usr/bin/env python3
"""Train query→table retrieval GNN over the HeteroData pool.

Architecture:
    Heterogeneous GAT (HANConv / HGTConv variant) on the 4-typed graph
    (table, row, cell, passage). Output dim 256. Initial node features come
    from bge-m3 (1024) → 256 via per-type linear projection.

    Query encoder: bge-m3(question) → 256 (linear proj, same target space).

Training (contrastive on HybridQA train):
    Positive pair: (question, gold_table_id)
    In-batch negatives: other queries' gold tables in same batch.
    Loss: InfoNCE with temperature 0.05.

Inference: q_emb · table_out_emb → rank tables → top-K.

Bge-m3 question encoding is done offline (one-time precompute) to avoid
hitting the vLLM endpoint per step.

GPU policy: this is a training task. On local: only cuda 2,3 are permitted
(currently occupied). Run on 12.43 cuda 7 (~77GB free; embedder uses ~3GB).
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear as PyGLinear

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_graph_structure_unified import _DenseAPIHandle


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TableRetrievalGNN(nn.Module):
    """Residual heterogeneous GNN. Refines bge-m3 features rather than
    replacing them, so the baseline bge-m3 cosine signal is preserved.

    Output table embedding = bge-m3(table) + λ × GNN_delta(table).
    Query stays in bge-m3 space (no learnable projection on q by default)
    so that initial state recovers the bge-m3 cosine baseline exactly.
    """

    def __init__(self, metadata, in_dim: int = 1024, hidden: int = 256,
                  n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1,
                  init_gate: float = -5.0):  # sigmoid(-5)≈0.0067 → initial out ≈ baseline
        super().__init__()
        self.in_dim = in_dim
        # Down-project for GNN message passing
        self.in_lin = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden) for ntype in metadata[0]
        })
        self.convs = nn.ModuleList([
            HGTConv(hidden, hidden, metadata, heads=n_heads)
            for _ in range(n_layers)
        ])
        # Up-project back to in_dim for residual fusion
        self.out_lin = nn.ModuleDict({
            ntype: nn.Linear(hidden, in_dim) for ntype in metadata[0]
        })
        # Per-type gates (start at 0 so initial table_out == bge-m3 exactly).
        self.gate = nn.ParameterDict({
            ntype: nn.Parameter(torch.tensor(init_gate)) for ntype in metadata[0]
        })
        # Optional query projection (start as identity-like)
        self.q_lin = nn.Linear(in_dim, in_dim)
        nn.init.eye_(self.q_lin.weight)
        nn.init.zeros_(self.q_lin.bias)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        h = {ntype: self.in_lin[ntype](x) for ntype, x in x_dict.items()}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {k: F.gelu(v) for k, v in h.items()}
        delta = {ntype: self.out_lin[ntype](v) for ntype, v in h.items()}
        # Residual fusion: out = x + sigmoid(gate) × delta
        out = {}
        for ntype, x in x_dict.items():
            g = torch.sigmoid(self.gate[ntype])
            out[ntype] = x + g * delta[ntype]
        return out

    def encode_query(self, q_x: torch.Tensor) -> torch.Tensor:
        return self.q_lin(q_x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_train_pairs(train_json_path: Path, tid_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
    """Return list of (question, table_local_idx) pairs from HybridQA train."""
    with train_json_path.open(encoding="utf-8") as f:
        rows = json.load(f)
    pairs = []
    n_skip = 0
    for r in rows:
        q = r.get("question") or ""
        tid = r.get("table_id") or ""
        if tid not in tid_to_idx:
            n_skip += 1
            continue
        pairs.append((q, tid_to_idx[tid]))
    print(f"[data] loaded {len(pairs)} train pairs (skipped {n_skip} unknown-table)",
          flush=True)
    return pairs


def _encode_questions(questions: List[str], api_base: str, api_key: str, model: str,
                       cache_path: Path) -> np.ndarray:
    """Encode questions via vLLM bge-m3 once; cache to disk."""
    if cache_path.exists():
        print(f"[encode-q] reusing {cache_path}", flush=True)
        return np.load(cache_path)
    print(f"[encode-q] encoding {len(questions)} questions ...", flush=True)
    embedder = _DenseAPIHandle(api_base=api_base, model=model, api_key=api_key, batch_size=128)
    t0 = time.time()
    arr = embedder.encode(questions, normalize_embeddings=True, convert_to_numpy=True)
    print(f"[encode-q] done in {time.time()-t0:.1f}s shape={arr.shape}", flush=True)
    np.save(cache_path, arr)
    return arr


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-path", default="analysis/open_pool/hetero_graph.pt", type=Path)
    ap.add_argument("--train-json", default="data/hybridqa_raw/train.json", type=Path)
    ap.add_argument("--dev-json", default="data/hybridqa_raw/dev.json", type=Path)
    ap.add_argument("--api-base", default="http://192.168.12.43:8002/v1")
    ap.add_argument("--api-key", default="embed-key-m3")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--out-model", default="models/query_table_gnn", type=Path)
    ap.add_argument("--cache-dir", default="analysis/q_table_gnn", type=Path)
    ap.add_argument("--hidden", default=256, type=int)
    ap.add_argument("--out-dim", default=256, type=int)
    ap.add_argument("--n-layers", default=2, type=int)
    ap.add_argument("--n-heads", default=4, type=int)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--n-epochs", default=2, type=int)
    ap.add_argument("--temperature", default=0.05, type=float)
    ap.add_argument("--seed", default=1337, type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train", default=0, type=int, help="cap for quick smoke; 0 = full")
    ap.add_argument("--log-every", default=50, type=int)
    ap.add_argument("--hard-negatives", default=None, type=Path,
                    help="If set, path to precomputed (n_train, K) int32 array of "
                         "hard-negative table indices per train query. Augments the "
                         "in-batch contrastive denominator.")
    args = ap.parse_args()

    for k in ("http_proxy", "https_proxy", "all_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(k, None)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_model.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)

    print(f"[load] graph from {args.graph_path} ...", flush=True)
    saved = torch.load(args.graph_path, weights_only=False)
    data: HeteroData = saved["data"]
    table_ids: List[str] = saved["table_ids"]
    tid_to_idx = {t: i for i, t in enumerate(table_ids)}
    metadata = data.metadata()
    print(f"[load] graph metadata: {metadata}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[load] moving graph to {device} ...", flush=True)
    data = data.to(device)

    print(f"[load] training pairs ...", flush=True)
    train_pairs = _load_train_pairs(args.train_json, tid_to_idx)
    if args.max_train > 0:
        train_pairs = train_pairs[:args.max_train]
    train_questions = [q for q, _ in train_pairs]
    train_labels = np.array([t for _, t in train_pairs], dtype=np.int64)

    q_cache = args.cache_dir / "train_question_embeddings.npy"
    train_q_emb = _encode_questions(train_questions, args.api_base, args.api_key,
                                      args.model, q_cache)

    # Optional hard negatives
    train_hard_negs = None
    if args.hard_negatives and args.hard_negatives.exists():
        train_hard_negs = np.load(args.hard_negatives)
        assert train_hard_negs.shape[0] == len(train_pairs), (
            f"hard negs {train_hard_negs.shape} vs train_pairs {len(train_pairs)}"
        )
        print(f"[data] hard negatives loaded: shape={train_hard_negs.shape}", flush=True)

    # Dev pairs (for eval)
    print(f"[load] dev pairs ...", flush=True)
    with args.dev_json.open(encoding="utf-8") as f:
        dev_rows = json.load(f)
    dev_pairs = [(r.get("question") or "", tid_to_idx[r.get("table_id") or ""])
                 for r in dev_rows if r.get("table_id") in tid_to_idx]
    dev_questions = [q for q, _ in dev_pairs]
    dev_labels = np.array([t for _, t in dev_pairs], dtype=np.int64)
    dev_q_cache = args.cache_dir / "dev_question_embeddings.npy"
    dev_q_emb = _encode_questions(dev_questions, args.api_base, args.api_key,
                                    args.model, dev_q_cache)
    print(f"[data] train={len(train_pairs)} dev={len(dev_pairs)} tables={len(table_ids)}",
          flush=True)

    # Model
    in_dim = data["table"].x.shape[1]
    print(f"[model] in_dim={in_dim} hidden={args.hidden} out_dim={args.out_dim} "
          f"layers={args.n_layers} heads={args.n_heads}", flush=True)
    model = TableRetrievalGNN(metadata, in_dim=in_dim, hidden=args.hidden,
                                n_layers=args.n_layers,
                                n_heads=args.n_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    def _eval(q_emb_np: np.ndarray, labels_np: np.ndarray, k: int = 5) -> Dict[str, float]:
        """Top-1 and top-K accuracy on the supplied pairs."""
        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            table_out = F.normalize(out["table"], dim=-1)  # (N_tables, D)
            q_t = torch.from_numpy(q_emb_np).float().to(device)
            q_proj = F.normalize(model.encode_query(q_t), dim=-1)  # (N_q, D)
            scores = q_proj @ table_out.t()  # (N_q, N_tables)
            top_k = scores.topk(k, dim=-1).indices.cpu().numpy()
            labels = labels_np
            top1 = float((top_k[:, 0] == labels).mean())
            topk_acc = float((top_k == labels[:, None]).any(axis=1).mean())
        model.train()
        return {f"top1": round(top1, 4), f"top{k}": round(topk_acc, 4)}

    print(f"[train] starting {args.n_epochs} epochs, batch_size={args.batch_size}", flush=True)
    n_steps_total = args.n_epochs * (len(train_pairs) // args.batch_size)
    step = 0
    started = time.time()
    best_top5 = 0.0
    for epoch in range(args.n_epochs):
        order = np.random.permutation(len(train_pairs))
        for bi in range(0, len(order) - args.batch_size + 1, args.batch_size):
            batch_idx = order[bi: bi + args.batch_size]
            q_np = train_q_emb[batch_idx]
            tids = train_labels[batch_idx]
            q_t = torch.from_numpy(q_np).float().to(device)

            opt.zero_grad()
            out = model(data.x_dict, data.edge_index_dict)
            table_out = F.normalize(out["table"], dim=-1)
            q_proj = F.normalize(model.encode_query(q_t), dim=-1)
            # Scores against positive + in-batch negatives (other batch positions)
            pos_table_emb = table_out[tids]  # (B, D)
            B, D = pos_table_emb.shape
            # In-batch contrastive: each query against all batch positives
            in_batch_logits = q_proj @ pos_table_emb.t() / args.temperature  # (B, B)
            # Hard-negative augmentation (codex-style retrieval training)
            if train_hard_negs is not None:
                hn_idxs = train_hard_negs[batch_idx]  # (B, K) int32
                hn_idxs_t = torch.from_numpy(hn_idxs.astype(np.int64)).to(device)
                hn_emb = table_out[hn_idxs_t.view(-1)].view(B, hn_idxs.shape[1], D)
                # (B, 1, D) × (B, K, D) → (B, K)
                hn_logits = (q_proj.unsqueeze(1) * hn_emb).sum(-1) / args.temperature
                logits = torch.cat([in_batch_logits, hn_logits], dim=1)  # (B, B+K)
            else:
                logits = in_batch_logits
            target = torch.arange(B, device=device)
            loss = F.cross_entropy(logits, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - started
                print(f"[train] ep{epoch} step {step}/{n_steps_total} "
                      f"loss={loss.item():.4f} {elapsed:.0f}s",
                      flush=True)
        # End-of-epoch eval
        m = _eval(dev_q_emb, dev_labels, k=5)
        print(f"[eval] epoch {epoch} dev: {m}", flush=True)
        if m["top5"] > best_top5:
            best_top5 = m["top5"]
            ckpt = {"state_dict": model.state_dict(), "metadata": metadata,
                    "args": vars(args), "best_top5": best_top5,
                    "table_ids": table_ids}
            torch.save(ckpt, args.out_model / "best.pt")
            print(f"[save] best checkpoint top5={best_top5:.4f}", flush=True)

    # Final save + final eval against baseline bge-m3 (no GNN)
    print(f"[done] best dev top5 = {best_top5:.4f}", flush=True)
    # Compare against bge-m3 cosine-only baseline on dev
    table_emb_baseline = data["table"].x.cpu().numpy()
    table_emb_baseline = table_emb_baseline / (np.linalg.norm(table_emb_baseline, axis=1, keepdims=True) + 1e-9)
    q_norm = dev_q_emb / (np.linalg.norm(dev_q_emb, axis=1, keepdims=True) + 1e-9)
    scores = q_norm @ table_emb_baseline.T
    top5_base = (scores.argsort(axis=1)[:, -5:] == dev_labels[:, None]).any(axis=1).mean()
    top1_base = (scores.argmax(axis=1) == dev_labels).mean()
    print(f"[baseline] bge-m3 cosine-only dev top1={top1_base:.4f}  top5={top5_base:.4f}",
          flush=True)
    print(f"[final] GNN top5={best_top5:.4f}  vs bge-m3 top5={top5_base:.4f}  "
          f"Δ={best_top5 - top5_base:+.4f}", flush=True)


if __name__ == "__main__":
    main()
