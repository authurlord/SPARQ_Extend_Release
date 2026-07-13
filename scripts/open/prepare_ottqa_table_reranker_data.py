#!/usr/bin/env python3
"""Build training data for OTT-QA Stage 1 TABLE reranker (cross-encoder).

Per codex error analysis: top-20 reranker with row-aware table text is the
highest-ROI fix for @1 (+6-11 pp), because 273 of 430 failures on 1690 have
gold in rank 2-5 already — they just need better ordering against sibling
tables (Mode A).

Training pairs:
  - For each train query (Qwen3-instruct embedding ↔ table_embeddings_qwen3):
    compute top-20 table indices by cosine.
  - Positive = gold table id (from train_linked.json `table_id`).
  - Hard negatives = top-20 candidates MINUS gold (sibling tables that look
    similar at the dense level — exactly the Mode A failure pattern).
  - Skip queries where gold not in top-20 (~5-10% expected; no useful contrast).

Table text (must match the eval-time format in `rerank_stage1_1690.py`):
  TITLE: {title}
  HEADERS: h1 | h2 | h3 | ...
  ROWS:
    {row_text_1}
    ...
    {row_text_K}
  where the K rows are chosen by query-token overlap with row text (lexical).

Output (FlagEmbedding format):
  {"query": q, "pos": [table_text_gold], "neg": [table_text_n1, ..., table_text_nK]}
"""
from __future__ import annotations
import argparse, json, random, re, zipfile
from pathlib import Path

import numpy as np


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "and", "or", "but", "with", "by", "for", "as", "from", "this", "that",
    "be", "been", "has", "have", "had", "do", "does", "did", "will", "would",
    "what", "who", "where", "when", "which", "how", "why", "?",
}


def tokenize(text: str) -> set[str]:
    text = (text or "").lower().replace("_", " ")
    return {t for t in re.findall(r"[a-z0-9]+", text)
             if len(t) > 1 and t not in STOPWORDS}


def row_to_text(row: list, header: list, max_chars: int = 200) -> str:
    parts = []
    for c_idx, cell in enumerate(row):
        val = str(cell[0])[:60] if (isinstance(cell, list) and cell) else str(cell)[:60]
        h = ""
        if c_idx < len(header):
            hcell = header[c_idx]
            h = hcell[0] if isinstance(hcell, list) else str(hcell)
        parts.append(f"{h}={val}" if h else val)
    return " | ".join(parts)[:max_chars]


def select_top_rows_by_query_overlap(rows: list[list], header: list,
                                       q_tokens: set[str], K: int = 5
                                       ) -> list[str]:
    """Pick top-K rows whose token overlap with query is highest. Falls back
    to the first K rows when overlap is tied or zero."""
    if not rows: return []
    scored = []
    for i, row in enumerate(rows):
        rtxt = row_to_text(row, header)
        r_tokens = tokenize(rtxt)
        overlap = len(q_tokens & r_tokens)
        # Tie-break: rows with lower index come first (preserve table order)
        scored.append((-overlap, i, rtxt))
    scored.sort()
    return [t for _, _, t in scored[:K]]


def build_table_text(table_id: str, ottqa_table: dict, q_tokens: set[str],
                      K_rows: int = 5, max_chars: int = 800) -> str:
    title = (ottqa_table.get("title") or table_id)[:200]
    header = ottqa_table.get("header") or []
    header_str = " | ".join(str(h[0]) if isinstance(h, list) else str(h)
                             for h in header)[:200]
    rows = ottqa_table.get("data") or []
    row_texts = select_top_rows_by_query_overlap(rows, header, q_tokens, K=K_rows)
    rows_block = "\n".join(f"  {r}" for r in row_texts)
    txt = f"TITLE: {title}\nHEADERS: {header_str}\nROWS:\n{rows_block}"
    return txt[:max_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-linked-zip",
                     default="data/ottqa_repo/preprocessed_data/train_linked.json.zip",
                     type=Path)
    ap.add_argument("--ottqa-tables",
                     default="data/ottqa_repo/data/traindev_tables.json",
                     type=Path)
    ap.add_argument("--train-q-emb",
                     default="analysis/ottqa_open_pool/train_query_embeddings_qwen3_instruct.npy",
                     type=Path)
    ap.add_argument("--table-emb",
                     default="analysis/ottqa_open_pool/table_embeddings_qwen3.npy",
                     type=Path)
    ap.add_argument("--ibm-corpus",
                     default="data/ottqa_raw_full/corpus_structure.jsonl", type=Path,
                     help="IBM 8891-table corpus ordering for table_emb rows")
    ap.add_argument("--K-cands", type=int, default=20,
                     help="dense top-K to source hard negs from")
    ap.add_argument("--K-neg", type=int, default=7,
                     help="hard negs kept per (query, gold); pos + N_neg = train group")
    ap.add_argument("--K-rows", type=int, default=5,
                     help="rows kept in each table text serialization")
    ap.add_argument("--max-text-chars", type=int, default=800)
    ap.add_argument("--chunk", type=int, default=4096,
                     help="rows per chunk for query × table matmul")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",
                     default="data/ottqa_table_reranker_training/train_v1.jsonl",
                     type=Path)
    ap.add_argument("--stats-out",
                     default="data/ottqa_table_reranker_training/train_v1.stats.json",
                     type=Path)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    # ===== Load resources =====
    print(f"[load] ottqa_tables ...", flush=True)
    tables = json.loads(args.ottqa_tables.read_text())
    print(f"[load] ottqa_tables: {len(tables)}", flush=True)

    ibm = [json.loads(l) for l in args.ibm_corpus.read_text().splitlines() if l.strip()]
    table_ids = [t["_id"] for t in ibm]
    tid_to_idx = {t: i for i, t in enumerate(table_ids)}
    n_tab = len(table_ids)
    print(f"[load] IBM corpus tables: {n_tab}", flush=True)

    with zipfile.ZipFile(args.train_linked_zip) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".json"))
        with zf.open(names[0]) as f:
            train = json.load(f)
    print(f"[load] train queries: {len(train)}", flush=True)

    qe = np.load(args.train_q_emb).astype(np.float32)
    te = np.load(args.table_emb).astype(np.float32)
    assert qe.shape[0] == len(train), \
        f"q_emb {qe.shape[0]} != train rows {len(train)}"
    assert te.shape[0] == n_tab, f"t_emb {te.shape[0]} != IBM tables {n_tab}"
    qe = qe / (np.linalg.norm(qe, axis=1, keepdims=True) + 1e-9)
    te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-9)

    # ===== Compute top-K cands per train query (chunked) =====
    K = args.K_cands
    n_q = qe.shape[0]
    topk_idx = np.empty((n_q, K), dtype=np.int32)
    print(f"[retrieval] computing top-{K} cands per query ...", flush=True)
    for s in range(0, n_q, args.chunk):
        e = min(n_q, s + args.chunk)
        scores = qe[s:e] @ te.T
        part = np.argpartition(-scores, K, axis=1)[:, :K]
        row_idx = np.arange(part.shape[0])[:, None]
        ord_ = np.argsort(-scores[row_idx, part], axis=1)
        topk_idx[s:e] = part[row_idx, ord_]
        if (s // args.chunk) % 4 == 0:
            print(f"  [retrieval] {e}/{n_q}", flush=True)

    # ===== Build training rows =====
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skip_no_gold_in_top = 0
    n_skip_no_tid_in_graph = 0
    n_skip_gold_not_in_tables = 0
    n_with_rich_negs = 0
    fout = args.out.open("w")
    for ri, r in enumerate(train):
        if ri % 5000 == 0:
            print(f"  [{ri}/{n_q}] written={n_written} "
                  f"skip(notop={n_skip_no_gold_in_top}, notid={n_skip_no_tid_in_graph}, "
                  f"nogtbl={n_skip_gold_not_in_tables})", flush=True)
        q = (r.get("question") or "").strip()
        gold_tid = r.get("table_id") or ""
        if not q or not gold_tid: continue
        if gold_tid not in tid_to_idx:
            n_skip_no_tid_in_graph += 1; continue
        gold_idx = tid_to_idx[gold_tid]
        cands = topk_idx[ri].tolist()
        if gold_idx not in cands:
            n_skip_no_gold_in_top += 1; continue

        # Hard negs: top-K cands minus gold, in rank order, capped at K_neg
        neg_idxs = [i for i in cands if i != gold_idx][: args.K_neg]
        if not neg_idxs: continue

        q_tokens = tokenize(q)

        # Gold table text
        gold_table = tables.get(gold_tid)
        if gold_table is None:
            n_skip_gold_not_in_tables += 1; continue
        pos_text = build_table_text(gold_tid, gold_table, q_tokens,
                                     K_rows=args.K_rows, max_chars=args.max_text_chars)

        # Neg table texts
        neg_texts = []
        for ni in neg_idxs:
            ntid = table_ids[ni]
            ntable = tables.get(ntid)
            if ntable is None: continue
            ntext = build_table_text(ntid, ntable, q_tokens,
                                      K_rows=args.K_rows, max_chars=args.max_text_chars)
            neg_texts.append(ntext)
        if not neg_texts: continue
        if len(neg_texts) >= args.K_neg: n_with_rich_negs += 1

        fout.write(json.dumps({
            "query": q,
            "pos": [pos_text],
            "neg": neg_texts,
        }, ensure_ascii=False) + "\n")
        n_written += 1
    fout.close()

    stats = {
        "n_written": n_written,
        "n_train_queries": n_q,
        "skip_no_gold_in_top_K": n_skip_no_gold_in_top,
        "skip_no_tid_in_graph": n_skip_no_tid_in_graph,
        "skip_gold_not_in_tables_json": n_skip_gold_not_in_tables,
        "n_with_full_K_neg": n_with_rich_negs,
        "K_cands": args.K_cands,
        "K_neg": args.K_neg,
        "K_rows": args.K_rows,
        "max_text_chars": args.max_text_chars,
        "hard_neg_source": "qwen3_instruct_dense_top_K_minus_gold",
    }
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, indent=2))
    print(f"\n[done] wrote {n_written} samples → {args.out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
