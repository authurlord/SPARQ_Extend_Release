#!/usr/bin/env python3
"""End-to-end Stage 1 + 2 + 3 + 4 eval on OTT-QA 1690 subset.

Stage 1 input: either
  - 3-leg RRF top-1 from saved ranks (default)
  - top-1 from reranked_cands.jsonl (after table reranker runs)

Stage 2: cell-link union from top-1 table → cand passages (from 240K pool)

Stage 3: BM25 over the cand passages with question → top-K_pas (default; uses
         same render_gold_table BM25 ranker as oracle script). If passage
         reranker provided via --passage-reranker-scores, uses its order instead.

Stage 4: 35B Qwen3.6-A3B-FP8 reader on 12.43:9540 + hybridqa_qa_cot prompt
         + retrieved-table render + ranked passages.

Outputs:
  analysis/ottqa_strict1690/e2e_{tag}.jsonl   per-query result
  analysis/ottqa_strict1690/e2e_{tag}.summary.json   overall + by-kind +
                                                    by-stage1-correct breakdown
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import utils.hybridqa_metrics as _hyb
parse_answer = _hyb.parse_answer
from utils.sparq_eval import sparq_em_f1


PROMPT_TEMPLATE = (REPO_ROOT / "src/schedule_pipeline/hybridqa_qa_cot.txt").read_text() + (
"""

Now answer:
Question: {question}
Selected Table:
{table_text}
SQL Result:
(no SQL run; rely on Selected Table and Linked Passages directly)
Linked Passages:
{passages_text}
""")


def load_pool(path: Path) -> dict[str, str]:
    pool: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pid = (r.get("pid") or "").lower()
            sm = (r.get("summary") or "").strip()
            if pid and sm and (pid not in pool or len(sm) > len(pool[pid])):
                pool[pid] = sm
    return pool


def _bm25_rank(query: str, candidates: list[tuple[str, str]]) -> list[int]:
    if not candidates: return []
    from rank_bm25 import BM25Okapi
    tok = lambda s: re.findall(r"\w+", s.lower())
    docs = [tok(anchor + " " + text) for anchor, text in candidates]
    if not any(docs): return list(range(len(candidates)))
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(tok(query))
    return sorted(range(len(candidates)), key=lambda i: -scores[i])


def render_retrieved_table(t: dict, pool: dict[str, str], question: str,
                            max_linked: int = 30, max_rows: int = 50,
                            passage_chars: int = 2500, cell_chars: int = 200
                            ) -> tuple[str, list[str], int]:
    """Same as oracle render_gold_table but parameterized for retrieved table.
    Returns (table_text, passage_blocks, n_total_cands)."""
    title = (t.get("title") or "")[:300]
    section = (t.get("section_title") or "")[:200]
    caption = (f"# {title}{' — ' + section if section else ''}" if title else "").strip()
    header = t.get("header", [])
    h_names = [h[0] if isinstance(h, list) else str(h) for h in header]
    lines = []
    if caption: lines.append(caption)
    lines.append("col : row_id | " + " | ".join(h_names))

    candidates: list[dict] = []
    seen_urls = set()
    for r_idx, row in enumerate(t.get("data", [])):
        if r_idx >= max_rows: break
        cell_strs = [str(r_idx)]
        for c_idx, cell in enumerate(row):
            if isinstance(cell, list) and len(cell) >= 2:
                cell_text = str(cell[0])[:cell_chars]
                cell_strs.append(cell_text)
                for url in (cell[1] or []):
                    pid = url.lower()
                    if pid in seen_urls: continue
                    seen_urls.add(pid)
                    txt = pool.get(pid, "")
                    if not txt: continue
                    col_name = h_names[c_idx] if c_idx < len(h_names) else f"col{c_idx}"
                    candidates.append({
                        "row": r_idx, "col": col_name,
                        "cell": cell_text[:60], "text": txt[:passage_chars],
                    })
            else:
                cell_strs.append(str(cell)[:cell_chars])
        lines.append(f"row {r_idx} : " + " | ".join(cell_strs))

    anchored = [(c["cell"], c["text"]) for c in candidates]
    order = _bm25_rank(question, anchored)[:max_linked]
    passages_used = []
    for k, idx in enumerate(order):
        c = candidates[idx]
        passages_used.append(
            f"[Passage {k+1}] row={c['row']} col={c['col']} cell={c['cell']}\n{c['text']}"
        )
    return "\n".join(lines), passages_used, len(candidates)


def load_stage1_top1(args, dev: list, dev_qid_to_devidx: dict, table_ids: list) -> dict:
    """Return qid → top1_table_id."""
    if args.reranked_cands_jsonl and args.reranked_cands_jsonl.exists():
        print(f"[stage1] loading from reranker cands: {args.reranked_cands_jsonl}",
              flush=True)
        out = {}
        for line in args.reranked_cands_jsonl.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            top1 = r["reranked_table_ids"][0] if r.get("reranked_table_ids") else \
                   r["rrf_cand_table_ids"][0]
            out[r["qid"]] = top1
        return out
    # Fallback: 3-leg RRF from saved ranks (computed on the fly with current GNN ckpt)
    print(f"[stage1] computing 3-leg RRF on the fly with GNN={args.gnn_ckpt}",
          flush=True)
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from train_query_table_gnn import TableRetrievalGNN

    # Prefer CPU when CUDA is contended by training (e.g., reranker training in progress)
    if args.stage1_device == "auto":
        if torch.cuda.is_available():
            try:
                free, _ = torch.cuda.mem_get_info(0)
                args.stage1_device = "cuda" if free > 3 * (1024 ** 3) else "cpu"
            except Exception:
                args.stage1_device = "cpu"
        else:
            args.stage1_device = "cpu"
    device = torch.device(args.stage1_device)
    print(f"[stage1] device = {device}", flush=True)
    saved = torch.load(args.graph, weights_only=False, map_location=device)
    data = saved["data"].to(device)
    g_table_ids = saved["table_ids"]
    tid_to_idx = {t: i for i, t in enumerate(g_table_ids)}
    n_tab = len(g_table_ids)

    dev_q = np.load(args.dev_q_emb).astype(np.float32)
    dev_q = dev_q / (np.linalg.norm(dev_q, axis=1, keepdims=True) + 1e-9)
    ckpt = torch.load(args.gnn_ckpt, weights_only=False, map_location=device)
    margs = ckpt["args"]
    in_dim = data["table"].x.shape[1]
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
        ranks_gnn = np.argsort(-scores_gnn, axis=1)[:, :100]
    table_x = F.normalize(data["table"].x, dim=-1).cpu().numpy()
    scores_qwen3 = dev_q @ table_x.T
    ranks_qwen3 = np.argsort(-scores_qwen3, axis=1)[:, :100]
    bm25_ranks = np.load(args.bm25_ranks)[:, :100]

    K_in = 100; rrf_k = 60
    fused = np.zeros((len(dev), n_tab), dtype=np.float64)
    for ranks in (ranks_gnn, ranks_qwen3, bm25_ranks):
        for i in range(len(dev)):
            for k, idx in enumerate(ranks[i]):
                fused[i, idx] += 1.0 / (rrf_k + k + 1)
    top1_indices = fused.argmax(axis=1)
    out = {}
    for d, t_idx in zip(dev, top1_indices):
        out[d["question_id"]] = g_table_ids[int(t_idx)]
    # Optionally cache
    if args.save_stage1_jsonl is not None:
        args.save_stage1_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.save_stage1_jsonl.open("w") as f:
            for qid, tid in out.items():
                f.write(json.dumps({"qid": qid, "top1_table_id": tid}) + "\n")
        print(f"[stage1] cached top-1 mapping → {args.save_stage1_jsonl}", flush=True)
    return out


async def call_llm(client, model, prompt, max_tokens=2048):
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return r.choices[0].message.content or ""
    except Exception as exc:
        return f"__ERROR__: {exc!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="analysis/ottqa_dev_strict1690.jsonl", type=Path)
    ap.add_argument("--tables", default="data/ottqa_repo/data/traindev_tables.json", type=Path)
    ap.add_argument("--pool", default="analysis/open_pool_full/passages.jsonl", type=Path)
    ap.add_argument("--dev-json", default="/tmp/ottqa_dev.json", type=Path)
    # Stage 1 inputs (one of)
    ap.add_argument("--reranked-cands-jsonl", type=Path, default=None,
                     help="If given (output of rerank_stage1_1690.py), use top-1 from there")
    ap.add_argument("--graph", default="analysis/ottqa_open_pool/bipartite_graph.pt", type=Path,
                     help="for on-the-fly 3-leg RRF fallback")
    ap.add_argument("--gnn-ckpt",
                     default="models/ottqa_query_table_gnn_v2_instruct/best.pt", type=Path)
    ap.add_argument("--dev-q-emb",
                     default="analysis/ottqa_open_pool/dev_query_embeddings_linked_qwen3_instruct.npy",
                     type=Path)
    ap.add_argument("--bm25-ranks",
                     default="analysis/ottqa_strict1690/bm25_dev_full_ranks.npy", type=Path)
    ap.add_argument("--stage1-device", default="auto",
                     help="cuda/cpu/auto for Stage 1 RRF compute; auto picks CPU if GPU full")
    ap.add_argument("--save-stage1-jsonl", default=None, type=Path,
                     help="if set, save qid → top-1 mapping for reuse")
    # Reader (12.43 zhouyue Qwen3.5-35B-A3B at port 9540; do NOT touch the server)
    ap.add_argument("--api-base", default="http://192.168.12.43:9540/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen3.5-35B-A3B")
    ap.add_argument("--concurrency", default=48, type=int)
    ap.add_argument("--max-linked", default=30, type=int)
    ap.add_argument("--max-queries", default=0, type=int)
    ap.add_argument("--tag", default="3leg_top1", help="output filename tag")
    args = ap.parse_args()

    for k in ("http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None); os.environ.pop(k.upper(), None)

    t0 = time.time()
    queries = [json.loads(l) for l in args.subset.read_text().splitlines() if l.strip()]
    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    print(f"[load] queries: {len(queries)}", flush=True)

    dev = json.load(args.dev_json.open())
    dev_qid_to_devidx = {d["question_id"]: i for i, d in enumerate(dev)}
    saved = __import__("torch").load(args.graph, weights_only=False, map_location="cpu")
    table_ids = saved["table_ids"]
    del saved

    # Stage 1: qid → top1 table_id
    qid_to_top1 = load_stage1_top1(args, dev, dev_qid_to_devidx, table_ids)
    print(f"[stage1] resolved top-1 for {len(qid_to_top1)} qids", flush=True)

    tables = json.loads(args.tables.read_text())
    pool = load_pool(args.pool)
    print(f"[load] tables={len(tables)}  pool={len(pool)}", flush=True)

    out_jsonl = Path(f"analysis/ottqa_strict1690/e2e_{args.tag}.jsonl")
    out_summary = Path(f"analysis/ottqa_strict1690/e2e_{args.tag}.summary.json")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fout = out_jsonl.open("w")

    by_kind = defaultdict(lambda: {"em": 0.0, "f1": 0.0, "n": 0})
    by_stage1_correct = defaultdict(lambda: {"em": 0.0, "f1": 0.0, "n": 0})
    em_sum = f1_sum = 0.0
    n_seen = n_err = n_stage1_correct = 0
    progress = 0
    lock = asyncio.Lock()
    started = time.time()

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.api_base.rstrip("/"), api_key=args.api_key, timeout=180)
    sem = asyncio.Semaphore(args.concurrency)

    async def _one(q):
        nonlocal em_sum, f1_sum, n_seen, n_err, n_stage1_correct, progress
        async with sem:
            qid = q["question_id"]
            gold_tid = q["table_id"]
            top1_tid = qid_to_top1.get(qid)
            stage1_correct = (top1_tid == gold_tid)
            if top1_tid is None or top1_tid not in tables:
                table_text = f"(table {top1_tid} not found)"
                passages = []
                n_cands = 0
            else:
                table_text, passages, n_cands = render_retrieved_table(
                    tables[top1_tid], pool, q["question"],
                    max_linked=args.max_linked
                )
            passages_text = "\n".join(passages) if passages else "(none)"
            prompt = PROMPT_TEMPLATE.format(
                table_text=table_text, passages_text=passages_text, question=q["question"]
            )
            raw = await call_llm(client, args.model, prompt)
        err = raw.startswith("__ERROR__")
        pred = parse_answer(raw) if not err else ""
        em, f1 = sparq_em_f1(pred, q["answer_text"], question=q["question"])
        async with lock:
            fout.write(json.dumps({
                "qid": qid, "question": q["question"], "gold": q["answer_text"],
                "kind": q["kind"], "gold_table_id": gold_tid,
                "retrieved_top1_table_id": top1_tid,
                "stage1_correct": bool(stage1_correct),
                "n_table_cell_link_cands": n_cands, "n_passages": len(passages),
                "predicted": pred, "raw": raw[:1500],
                "em": float(em), "f1": round(f1, 4),
                "error": raw if err else None,
            }, ensure_ascii=False) + "\n"); fout.flush()
            if err: n_err += 1
            else:
                em_sum += em; f1_sum += f1; n_seen += 1
                if stage1_correct: n_stage1_correct += 1
                bk = by_kind[q["kind"]]
                bk["em"] += em; bk["f1"] += f1; bk["n"] += 1
                bs = by_stage1_correct["correct" if stage1_correct else "wrong"]
                bs["em"] += em; bs["f1"] += f1; bs["n"] += 1
            progress += 1
            if progress % 50 == 0:
                el = time.time() - started
                print(f"[e2e] {progress}/{len(queries)} "
                      f"EM={em_sum/max(1,n_seen):.4f} F1={f1_sum/max(1,n_seen):.4f} "
                      f"s1_acc={n_stage1_correct/max(1,n_seen):.4f} "
                      f"err={n_err} {el:.0f}s", flush=True)

    async def _run():
        await asyncio.gather(*[_one(q) for q in queries])
    asyncio.run(_run())
    fout.close()

    n = max(1, n_seen)
    summary = {
        "tag": args.tag,
        "n_total": len(queries), "n_scored": n_seen, "n_errors": n_err,
        "n_stage1_correct": n_stage1_correct,
        "stage1_top1_acc": n_stage1_correct / n,
        "EM": em_sum / n, "F1": f1_sum / n,
        "by_kind": {k: {"n": v["n"], "EM": v["em"] / max(1, v["n"]),
                         "F1": v["f1"] / max(1, v["n"])}
                     for k, v in by_kind.items()},
        "by_stage1_correct": {k: {"n": v["n"], "EM": v["em"] / max(1, v["n"]),
                                    "F1": v["f1"] / max(1, v["n"])}
                                for k, v in by_stage1_correct.items()},
        "wall_sec": round(time.time() - t0, 1),
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== E2E reader on strict-1690 ({args.model}, tag={args.tag}) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
