#!/usr/bin/env python3
"""HybridQA TRUE open-domain E2E:
  Stage 1: RRF(Qwen3+BM25) → v1 table reranker → top-1 retrieved table
  Stage 2: reuse Phase 7 rrf top-12 open-pool passages (independent of table)
  Stage 3: 35B reader (Qwen3.5-35B-A3B) + hybridqa_qa_cot.txt

Compares to:
  - Phase 7 gold-table baseline (4B): rrf EM = 0.4887
  - Phase 7 gold-table baseline (35B): rrf EM = 0.6140
  - This run = how much loss when we drop gold table → retrieved top-1
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from utils.hybridqa_local import load_hybridqa_dataset
from utils.hybridqa_metrics import parse_answer
from utils.sparq_eval import sparq_em_f1
from prepare_hybridqa_table_reranker_data import build_table_text, tokenize
from rerun_hybridqa_487_phase7_35b import format_table_text, load_pool_summaries, build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrf-cache",
                     default="analysis/qa_verification/hybridqa_rrf_topk12.jsonl",
                     type=Path, help="Phase 7 open-pool rrf retrieval (gives evidence_pids)")
    ap.add_argument("--rrf-table-ranks",
                     default="analysis/open_pool/hybridqa_rrf2_table_ranks_dev.npy", type=Path,
                     help="Stage 1 RRF table ranks (3466 × 100)")
    ap.add_argument("--dev-qids",
                     default="analysis/hybridqa_dev_query_qwen3_instruct.qids.json", type=Path,
                     help="alignment for rrf-table-ranks rows")
    ap.add_argument("--pool-dir", default="analysis/open_pool", type=Path)
    ap.add_argument("--reranker-dir", default="models/hybridqa_table_reranker_v1", type=Path)
    ap.add_argument("--hybridqa-dir", default="data/hybridqa_raw", type=Path)
    ap.add_argument("--passage-pool", default="analysis/open_pool/passages.jsonl", type=Path)
    ap.add_argument("--topK-rerank", type=int, default=20)
    ap.add_argument("--K-rows", type=int, default=5)
    ap.add_argument("--rerank-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--api-base", default="http://192.168.12.43:9540/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen3.5-35B-A3B")
    ap.add_argument("--concurrency", default=48, type=int)
    ap.add_argument("--max-prompt-chars", default=50000, type=int)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out-dir",
                     default="analysis/qa_verification_open_e2e", type=Path)
    args = ap.parse_args()

    for k in ("http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
        os.environ.pop(k, None)

    # ===== Load Phase 7 RRF cache (for passages + qid alignment) =====
    cache = [json.loads(l) for l in args.rrf_cache.read_text().splitlines() if l.strip()]
    qid_to_cache = {r["qid"]: r for r in cache}
    print(f"[load] phase7 rrf cache: {len(cache)} qids (487 subset)", flush=True)

    # ===== Load Stage 1 RRF table ranks (3466 × 100) =====
    table_ranks = np.load(args.rrf_table_ranks)
    dev_qids = json.loads(args.dev_qids.read_text())
    qid_to_rank_idx = {q: i for i, q in enumerate(dev_qids)}
    print(f"[load] stage 1 table_ranks: {table_ranks.shape}", flush=True)

    table_ids = json.loads((args.pool_dir / "table_ids.json").read_text())

    # ===== Pool tables =====
    print(f"[load] HybridQA tables ...", flush=True)
    ds_dev = load_hybridqa_dataset("validation", args.hybridqa_dir)
    qid_to_gold_tid = {}
    pool_tid_to_table = {}
    for i in range(len(ds_dev)):
        rec = ds_dev[i]
        qid_to_gold_tid[rec["question_id"]] = rec["table_id"]
        if rec["table_id"] not in pool_tid_to_table:
            pool_tid_to_table[rec["table_id"]] = rec["table"]
    print(f"[load] dev unique tables: {len(pool_tid_to_table)}", flush=True)

    # ===== Reranker =====
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.reranker_dir)
    rmodel = AutoModelForSequenceClassification.from_pretrained(
        args.reranker_dir, torch_dtype=torch.float16
    ).to(device).eval()
    print(f"[load] reranker: {args.reranker_dir}", flush=True)

    # ===== Stage 1 reranker: rerank top-20 from RRF, take top-1 table per query =====
    print(f"[stage1] reranking top-{args.topK_rerank} for {len(cache)} queries ...", flush=True)
    K_re = args.topK_rerank
    qid_to_retrieved_top1_tid = {}
    qid_to_pre_rerank_top1 = {}
    t0 = time.time()
    with torch.no_grad():
        for ci, r in enumerate(cache):
            qid = r["qid"]
            q = r["question"]
            ri = qid_to_rank_idx.get(qid)
            if ri is None: continue
            cand_idxs = table_ranks[ri, :K_re].tolist()
            cand_tids = [table_ids[c] for c in cand_idxs]
            qid_to_pre_rerank_top1[qid] = cand_tids[0]
            q_tokens = set(tokenize(q))
            texts = [build_table_text(pool_tid_to_table.get(ct, {}), q_tokens,
                                       args.K_rows, 800)
                     for ct in cand_tids]
            scores = []
            for s in range(0, K_re, args.rerank_batch):
                bq = [q] * min(args.rerank_batch, K_re - s)
                bp = texts[s: s + args.rerank_batch]
                enc = tok(bq, bp, padding=True, truncation=True, max_length=512,
                          return_tensors="pt").to(device)
                logits = rmodel(**enc).logits.squeeze(-1)
                scores.extend(logits.float().cpu().tolist())
            order = np.argsort(-np.array(scores))
            qid_to_retrieved_top1_tid[qid] = cand_tids[int(order[0])]
            if (ci + 1) % 100 == 0:
                print(f"  [rerank {ci+1}/{len(cache)}] {time.time()-t0:.0f}s", flush=True)
    del rmodel; del tok
    torch.cuda.empty_cache()
    print(f"[stage1] done {time.time()-t0:.0f}s", flush=True)

    # Stage 1 accuracy
    n_s1_correct = sum(1 for q, t in qid_to_retrieved_top1_tid.items()
                       if t == qid_to_gold_tid.get(q))
    n_s1_pre_correct = sum(1 for q, t in qid_to_pre_rerank_top1.items()
                           if t == qid_to_gold_tid.get(q))
    print(f"[stage1] reranker top-1 acc: {n_s1_correct}/{len(cache)} = {n_s1_correct/len(cache):.4f}",
          flush=True)
    print(f"[stage1] pre-rerank top-1 acc: {n_s1_pre_correct}/{len(cache)} = {n_s1_pre_correct/len(cache):.4f}",
          flush=True)

    # ===== Pool summaries for passage rendering =====
    pool_passages = load_pool_summaries(args.passage_pool)
    print(f"[load] passage pool: {len(pool_passages)}", flush=True)

    # ===== Stage 2+3: Reader =====
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / f"e2e_{args.tag}_487.jsonl"
    out_summary = args.out_dir / f"e2e_{args.tag}_487.summary.json"
    fout = out_jsonl.open("w")
    by_bucket = defaultdict(lambda: {"em": 0.0, "f1": 0.0, "n": 0})
    by_s1 = defaultdict(lambda: {"em": 0.0, "f1": 0.0, "n": 0})
    em_sum = f1_sum = 0.0
    n_seen = n_err = progress = 0
    started = time.time()
    lock = asyncio.Lock()

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.api_base.rstrip("/"),
                         api_key=args.api_key, timeout=180)
    sem = asyncio.Semaphore(args.concurrency)

    async def _one(r):
        nonlocal em_sum, f1_sum, n_seen, n_err, progress
        async with sem:
            qid = r["qid"]
            q = r["question"]
            gold = r["gold_answer"]
            evid = r.get("evidence_pids", [])
            bucket = r.get("subset_bucket", "")
            top1_tid = qid_to_retrieved_top1_tid.get(qid)
            s1_correct = (top1_tid == qid_to_gold_tid.get(qid))
            try:
                table = pool_tid_to_table.get(top1_tid, {})
                table_text, table_title = format_table_text(table)
                passages_used = [pool_passages.get(pid, "") for pid in evid]
                passages_used = [p for p in passages_used if p]
                prompt = build_prompt(q, table_text, table_title, passages_used,
                                       max_chars=args.max_prompt_chars)
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, top_p=1.0, max_tokens=512,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = (resp.choices[0].message.content or "").strip()
                pred = parse_answer(raw)
                em, f1 = sparq_em_f1(pred, gold, question=q)
                err = None
            except Exception as exc:
                pred = ""; raw = f"__ERROR__: {exc!r}"
                em, f1 = 0.0, 0.0; err = raw
        async with lock:
            fout.write(json.dumps({
                "qid": qid, "question": q, "gold": gold,
                "retrieved_top1_tid": top1_tid,
                "gold_tid": qid_to_gold_tid.get(qid),
                "s1_correct": bool(s1_correct),
                "predicted": pred, "raw": raw[:1500],
                "em": float(em), "f1": round(f1, 4),
                "subset_bucket": bucket, "error": err,
            }, ensure_ascii=False) + "\n"); fout.flush()
            if err: n_err += 1
            else:
                em_sum += em; f1_sum += f1; n_seen += 1
                by_bucket[bucket]["em"] += em
                by_bucket[bucket]["f1"] += f1
                by_bucket[bucket]["n"] += 1
                key = "s1_correct" if s1_correct else "s1_wrong"
                by_s1[key]["em"] += em
                by_s1[key]["f1"] += f1
                by_s1[key]["n"] += 1
            progress += 1
            if progress % 50 == 0:
                el = time.time() - started
                print(f"[e2e] {progress}/{len(cache)} "
                      f"EM={em_sum/max(1,n_seen):.4f} F1={f1_sum/max(1,n_seen):.4f} "
                      f"err={n_err} {el:.0f}s", flush=True)

    async def _run():
        await asyncio.gather(*[_one(r) for r in cache])
    asyncio.run(_run())
    fout.close()

    n = max(1, n_seen)
    summary = {
        "tag": args.tag,
        "n_total": len(cache), "n_scored": n_seen, "n_errors": n_err,
        "EM": em_sum / n, "F1": f1_sum / n,
        "stage1_top1_acc": n_s1_correct / len(cache),
        "stage1_pre_rerank_top1_acc": n_s1_pre_correct / len(cache),
        "by_bucket": {k: {"n": v["n"], "EM": v["em"]/max(1,v["n"]),
                           "F1": v["f1"]/max(1,v["n"])}
                       for k, v in by_bucket.items()},
        "by_s1": {k: {"n": v["n"], "EM": v["em"]/max(1,v["n"]),
                       "F1": v["f1"]/max(1,v["n"])}
                   for k, v in by_s1.items()},
        "reranker": str(args.reranker_dir),
        "model": args.model,
    }
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"\n=== HybridQA TRUE open-domain E2E (487 verifier subset) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
