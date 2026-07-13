#!/usr/bin/env python3
"""
Pilot for HybridQA Exp A:
1. table + all linked passages
2. table + BM25 top-k linked passages

Run with the hstar env because it requires datasets/openai/rank_bm25.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

for var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(var, None)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
os.environ.setdefault("NO_PROXY", os.environ["no_proxy"])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.hybridqa_metrics import (
    compute_bertscore_f1,
    exact_match,
    normalize_answer,
    table_to_rows,
    token_precision_recall_f1,
    tokenized,
)

from openai import AsyncOpenAI, BadRequestError
from rank_bm25 import BM25Okapi
from utils.hybridqa_local import load_hybridqa_dataset


EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
DEFAULT_OUT_DIR = Path("/data/workspace/yanmy/SPARQ_Extend/results/qwen35_35b/hybridqa_exp_a")
DEFAULT_BERTSCORE_MODEL = Path("/data/workspace/yanmy/models/deberta-v3-large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HybridQA Exp A pilot")
    parser.add_argument("--model", default="qwen3.5-35b")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="api-key-qwen3")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--hybridqa-raw-dir",
        type=Path,
        default=None,
        help="Directory containing train/dev/test.json and WikiTables-WithLinks*.zip.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--first-n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-prompt-chars", type=int, default=70000)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--healthcheck-timeout", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bertscore-model-path", type=Path, default=DEFAULT_BERTSCORE_MODEL)
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument("--bertscore-device", default="cpu")
    parser.add_argument("--disable-bertscore", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["all_passages", "bm25_topk"],
        choices=["all_passages", "bm25_topk"],
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def check_vllm_ready(api_base: str, api_key: str, timeout_sec: float) -> None:
    models_url = api_base.rstrip("/") + "/models"
    req = Request(models_url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                raise RuntimeError(f"vLLM health check failed: HTTP {resp.status}")
    except URLError as exc:
        raise RuntimeError(f"vLLM is not reachable at {models_url}: {exc}") from exc


def save_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def print_progress(mode: str, done: int, total: int, started: float, results: Sequence[Dict[str, Any]]) -> None:
    elapsed = time.time() - started
    em = sum(1 for r in results if r["exact_match"])
    f1_avg = sum(r["f1"] for r in results) / len(results) if results else 0.0
    errors = sum(1 for r in results if r["error"])
    print(
        f"[progress][{mode}] {done}/{total} "
        f"elapsed={elapsed:.1f}s "
        f"EM={em / len(results) * 100:.2f} "
        f"F1={f1_avg * 100:.2f} "
        f"errors={errors}",
        flush=True,
    )




def table_to_text(table: Dict[str, Any]) -> str:
    header, rows = table_to_rows(table)
    lines = ["col : " + " | ".join(header)]
    for row_idx, row in enumerate(rows):
        values = [str(cell.get("value", "")) for cell in row]
        lines.append(f"row {row_idx} : " + " | ".join(values))
    return "\n".join(lines)


def extract_passages(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    header, rows = table_to_rows(table)
    passages: List[Dict[str, Any]] = []
    seen = set()
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            for url_info in cell.get("urls") or []:
                summary = (url_info.get("summary") or "").strip()
                url = (url_info.get("url") or "").strip()
                if not summary:
                    continue
                key = (url, summary)
                if key in seen:
                    continue
                seen.add(key)
                passages.append(
                    {
                        "row_idx": row_idx,
                        "col_name": header[col_idx],
                        "cell_value": cell.get("value", ""),
                        "url": url,
                        "summary": summary,
                        "text": (
                            f"row {row_idx}, column {header[col_idx]}, cell value {cell.get('value', '')}\n"
                            f"passage: {summary}"
                        ),
                    }
                )
    return passages


def select_passages(question: str, passages: Sequence[Dict[str, Any]], mode: str, top_k: int) -> List[Dict[str, Any]]:
    if mode == "all_passages":
        return list(passages)
    if mode != "bm25_topk":
        raise ValueError(f"Unsupported mode: {mode}")
    if not passages:
        return []

    corpus = [tokenized(p["text"]) for p in passages]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized(question))
    ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
    return [passage for _, passage in ranked[:top_k]]


def build_prompt(question: str, table: Dict[str, Any], selected_passages: Sequence[Dict[str, Any]]) -> str:
    table_text = table_to_text(table)
    passage_chunks = []
    for idx, passage in enumerate(selected_passages, start=1):
        passage_chunks.append(
            f"[Passage {idx}] row={passage['row_idx']} col={passage['col_name']} "
            f"cell={passage['cell_value']}\n{passage['summary']}"
        )
    passages_text = "\n\n".join(passage_chunks) if passage_chunks else "(no linked passages selected)"

    return (
        "Answer the question using the table and linked Wikipedia passages.\n"
        "Return only the final answer span, with no explanation.\n\n"
        f"Table Title: {table.get('title', '')}\n"
        f"Section Title: {table.get('section_title', '')}\n"
        f"Table:\n{table_text}\n\n"
        f"Linked Passages:\n{passages_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def fit_prompt_to_budget(
    question: str,
    table: Dict[str, Any],
    selected_passages: Sequence[Dict[str, Any]],
    max_prompt_chars: int,
) -> Tuple[str, List[Dict[str, Any]], bool]:
    fitted = list(selected_passages)
    prompt = build_prompt(question, table, fitted)
    if max_prompt_chars <= 0:
        return prompt, fitted, False

    truncated = False
    while fitted and len(prompt) > max_prompt_chars:
        fitted.pop()
        prompt = build_prompt(question, table, fitted)
        truncated = True
    return prompt, fitted, truncated


async def infer_answer(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: str,
) -> Tuple[str, str]:
    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1,
                max_tokens=48,
                extra_body=EXTRA_BODY,
            )
        answer = (response.choices[0].message.content or "").strip()
        answer = re.sub(r"^Answers?\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
        return answer, ""
    except BadRequestError as exc:
        return "", f"BadRequestError: {exc}"
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def gold_in_passages(gold_answer: str, passages: Sequence[Dict[str, Any]]) -> bool:
    norm_gold = normalize_answer(gold_answer)
    if not norm_gold:
        return False
    return any(norm_gold in normalize_answer(p["summary"]) for p in passages)


async def run_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    sample_idx: int,
    sample: Dict[str, Any],
    mode: str,
    top_k: int,
    max_prompt_chars: int,
) -> Dict[str, Any]:
    table = sample["table"]
    passages = extract_passages(table)
    selected = select_passages(sample["question"], passages, mode, top_k)
    prompt, selected, truncated = fit_prompt_to_budget(
        question=sample["question"],
        table=table,
        selected_passages=selected,
        max_prompt_chars=max_prompt_chars,
    )
    pred_answer, error = await infer_answer(client, semaphore, model, prompt)
    gold_answer = sample["answer_text"]
    precision, recall, f1 = token_precision_recall_f1(pred_answer, gold_answer)

    result = {
        "idx": sample_idx,
        "question_id": sample["question_id"],
        "mode": mode,
        "question": sample["question"],
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "exact_match": exact_match(pred_answer, gold_answer),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "error": error or None,
        "num_all_passages": len(passages),
        "num_selected_passages": len(selected),
        "truncated_for_budget": truncated,
        "gold_in_all_passages": gold_in_passages(gold_answer, passages),
        "gold_in_selected_passages": gold_in_passages(gold_answer, selected),
        "prompt_chars": len(prompt),
        "selected_passages": [
            {
                "row_idx": p["row_idx"],
                "col_name": p["col_name"],
                "cell_value": p["cell_value"],
                "url": p["url"],
            }
            for p in selected
        ],
    }
    status = "EM" if result["exact_match"] else "MISS"
    print(
        f"[{mode}] {sample_idx}: {status} "
        f"sel={len(selected):02d}/{len(passages):02d} "
        f"f1={result['f1']:.2f} q={sample['question'][:60]}"
        ,
        flush=True,
    )
    return result


def summarize(mode: str, results: Sequence[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    f1_avg = sum(r["f1"] for r in results) / total if total else 0.0
    p_avg = sum(r.get("precision", 0.0) for r in results) / total if total else 0.0
    r_avg = sum(r.get("recall", 0.0) for r in results) / total if total else 0.0
    errors = sum(1 for r in results if r["error"])
    truncated = sum(1 for r in results if r["truncated_for_budget"])
    gold_in_all = sum(1 for r in results if r["gold_in_all_passages"])
    gold_in_selected = sum(1 for r in results if r["gold_in_selected_passages"])
    return {
        "mode": mode,
        "total": total,
        "top_k": top_k if mode == "bm25_topk" else None,
        "max_prompt_chars": None,
        "exact_match": round(exact / total * 100, 2) if total else 0.0,
        "precision": round(p_avg * 100, 2),
        "recall": round(r_avg * 100, 2),
        "f1": round(f1_avg * 100, 2),
        "bertscore_f1": None,
        "errors": errors,
        "truncated_for_budget": round(truncated / total * 100, 2) if total else 0.0,
        "avg_prompt_chars": round(sum(r["prompt_chars"] for r in results) / total, 2) if total else 0.0,
        "avg_selected_passages": round(sum(r["num_selected_passages"] for r in results) / total, 2) if total else 0.0,
        "gold_in_all_passages": round(gold_in_all / total * 100, 2) if total else 0.0,
        "gold_in_selected_passages": round(gold_in_selected / total * 100, 2) if total else 0.0,
    }


def finalize_summary(
    mode: str,
    results: Sequence[Dict[str, Any]],
    top_k: int,
    bertscore_model_path: Path,
    bertscore_batch_size: int,
    bertscore_device: str,
    disable_bertscore: bool,
) -> Dict[str, Any]:
    summary = summarize(mode, results, top_k)
    if not disable_bertscore:
        summary["bertscore_f1"] = compute_bertscore_f1(
            results,
            bertscore_model_path,
            bertscore_batch_size,
            bertscore_device,
        )
    return summary


async def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    check_vllm_ready(args.api_base, args.api_key, args.healthcheck_timeout)
    client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key, timeout=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    dataset = load_hybridqa_dataset(args.split, args.hybridqa_raw_dir)
    end = min(args.start + args.first_n, len(dataset))
    samples = [dataset[i] for i in range(args.start, end)]

    started = time.time()
    payload: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "split": args.split,
            "start": args.start,
            "first_n": args.first_n,
            "top_k": args.top_k,
            "concurrency": args.concurrency,
            "max_prompt_chars": args.max_prompt_chars,
            "request_timeout": args.request_timeout,
            "healthcheck_timeout": args.healthcheck_timeout,
            "save_every": args.save_every,
            "progress_every": args.progress_every,
            "batch_size": args.batch_size,
            "modes": args.modes,
        },
        "summaries": {},
        "results": {},
    }
    out_file = args.out_dir / f"pilot_start{args.start}_n{len(samples)}_k{args.top_k}.json"
    partial_file = out_file.with_suffix(".partial.json")

    for mode in args.modes:
        mode_started = time.time()
        results: List[Dict[str, Any]] = []
        total = len(samples)
        for batch_start in range(0, total, args.batch_size):
            batch = samples[batch_start : batch_start + args.batch_size]
            tasks = [
                run_one(
                    client=client,
                    semaphore=semaphore,
                    model=args.model,
                    sample_idx=args.start + batch_start + local_idx,
                    sample=sample,
                    mode=mode,
                    top_k=args.top_k,
                    max_prompt_chars=args.max_prompt_chars,
                )
                for local_idx, sample in enumerate(batch)
            ]
            for future in asyncio.as_completed(tasks):
                results.append(await future)
                done = len(results)
                payload["results"][mode] = results
                if done % args.progress_every == 0 or done == total:
                    print_progress(mode, done, total, mode_started, results)
                if done % args.save_every == 0 or done == total:
                    payload["summaries"][mode] = summarize(mode, results, args.top_k)
                    payload["summaries"][mode]["max_prompt_chars"] = args.max_prompt_chars
                    payload["elapsed_sec"] = round(time.time() - started, 2)
                    save_payload(partial_file, payload)
        payload["results"][mode] = results
        payload["summaries"][mode] = finalize_summary(
            mode,
            results,
            args.top_k,
            args.bertscore_model_path,
            args.bertscore_batch_size,
            args.bertscore_device,
            args.disable_bertscore,
        )
        payload["summaries"][mode]["max_prompt_chars"] = args.max_prompt_chars

    payload["elapsed_sec"] = round(time.time() - started, 2)
    save_payload(out_file, payload)
    if partial_file.exists():
        partial_file.unlink()

    print("\nSummary", flush=True)
    for mode, summary in payload["summaries"].items():
        print(
            f"  {mode}: EM={summary['exact_match']} F1={summary['f1']} "
            f"errors={summary['errors']} avg_passages={summary['avg_selected_passages']} "
            f"gold_in_selected={summary['gold_in_selected_passages']}"
            ,
            flush=True,
        )
    print(f"  saved: {out_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
