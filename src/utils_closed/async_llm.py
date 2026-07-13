import argparse
import asyncio
import json
import os
from typing import List, Optional
import asyncio
import nest_asyncio

# Apply the patch, which only needs to be done once in a Notebook
nest_asyncio.apply()
import pandas as pd
import time
from tqdm import tqdm

# OpenAI-compatible async client (works with vLLM api_server)
try:
    from openai import AsyncOpenAI, APIConnectionError, RateLimitError, BadRequestError, APIError
except Exception:  # fallback for older openai package name
    from openai import AsyncOpenAI  # type: ignore
    APIConnectionError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async client for vLLM OpenAI-compatible API, mirroring vllm_query_qwen_v0.py I/O.")

    # Mirror key args
    parser.add_argument("--llm_path", type=str, default="/public/Qwen3-4B-Instruct-2507/", help="Model name as seen by API server (e.g., Qwen3-4B-Instruct-2507)")
    parser.add_argument("--input_file", type=str, required=True, default="datasets/wikitq_test.csv", help="CSV or JSON file with an 'instruction' column")
    parser.add_argument("--output_path", type=str, required=True, default="datasets/wikitq_test_output_demo.csv", help="Output CSV path (will contain a 'predict' column)")
    parser.add_argument("--sample_num", type=int, default=1, help="Number of candidates n per sample")

    # API specific
    parser.add_argument("--api_base", type=str, default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key (vLLM ignores if set to EMPTY)")
    parser.add_argument("--concurrency", type=int, default=128, help="Max concurrent requests")
    parser.add_argument("--request_timeout", type=int, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries per request")

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty for vLLM API (-2.0 to 2.0)")

    # Metrics / logging
    parser.add_argument("--metrics_csv", type=str, default="", help="Optional CSV path to write per-request metrics")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar")

    return parser


async def infer_one(client: "AsyncOpenAI", model: str, prompt: str, sem: "asyncio.Semaphore", max_retries: int, request_timeout: int, temperature: float, top_p: float, max_tokens: int, n: int, presence_penalty: float):
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        async with sem:
            try:
                start_ts = time.perf_counter()
                # The parameter is added to both branches of the if/else statement
                if temperature == 0:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        n=n,
                        presence_penalty=presence_penalty, # Added parameter
                    )
                else:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        n=n,
                        presence_penalty=presence_penalty, # Added parameter
                    )
                latency_s = time.perf_counter() - start_ts
                usage = getattr(resp, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0)) if usage is not None else 0
                completion_tokens = int(getattr(usage, "completion_tokens", 0)) if usage is not None else 0
                return {
                    "texts": [c.message.content for c in resp.choices],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_s": latency_s,
                }
            except (APIConnectionError, RateLimitError, BadRequestError, asyncio.TimeoutError) as e:
                if attempt == max_retries:
                    return {
                        "texts": [f"[ERROR after {attempt} tries]: {type(e).__name__}: {e}"],
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_s": 0.0,
                    }
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)


async def run_inference(
    prompts: List[str],
    model: str,
    api_base: str,
    api_key: str,
    concurrency: int,
    request_timeout: int,
    max_retries: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int,
    presence_penalty: float, # Added parameter
    show_progress: bool,
):
    client = AsyncOpenAI(base_url=api_base, api_key=api_key, timeout=request_timeout)
    sem = asyncio.Semaphore(concurrency)

    async def wrapped_task(i: int, p: str):
        res = await infer_one(client, model, p, sem, max_retries, request_timeout, temperature, top_p, max_tokens, n, presence_penalty)
        return i, res

    batch_start = time.perf_counter()
    pbar = tqdm(total=len(prompts), desc="Processed prompts", disable=not show_progress)

    tasks = [asyncio.create_task(wrapped_task(i, p)) for i, p in enumerate(prompts)]

    results_by_idx: dict[int, List[Optional[str]]] = {}
    metrics_rows = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for coro in asyncio.as_completed(tasks):
        i, res = await coro
        texts = res.get("texts", [])
        pt = int(res.get("prompt_tokens", 0))
        ct = int(res.get("completion_tokens", 0))
        latency_s = float(res.get("latency_s", 0.0))

        results_by_idx[i] = texts
        metrics_rows.append({
            "index": i,
            "latency_s": latency_s,
            "prompt_tokens": pt,
            "completion_tokens": ct,
        })
        total_prompt_tokens += pt
        total_completion_tokens += ct

        elapsed = time.perf_counter() - batch_start
        in_tps = (total_prompt_tokens / elapsed) if elapsed > 0 else 0.0
        out_tps = (total_completion_tokens / elapsed) if elapsed > 0 else 0.0
        pbar.set_postfix({
            "est. speed input": f"{in_tps:.2f} toks/s",
            "output": f"{out_tps:.2f} toks/s",
        })
        pbar.update(1)

    pbar.close()

    batch_dur = time.perf_counter() - batch_start
    ordered_results: List[List[Optional[str]]] = [results_by_idx[i] for i in range(len(prompts))]

    return ordered_results, metrics_rows, batch_dur, total_prompt_tokens, total_completion_tokens


def infer_prompts(
    prompts: List[str],
    *,
    llm_path: str = "/data/workspace/yanmy/models/Qwen3-4B-Instruct-2507",
    llm_name: str = os.environ.get('SPARQX_LLM_NAME', 'qwen3-4b'),
    api_base: str = os.environ.get('SPARQX_API_BASE', 'http://127.0.0.1:8000/v1'),
    api_key: str = os.environ.get('SPARQX_API_KEY', 'EMPTY'),
    concurrency: int = 128,
    request_timeout: int = 120,
    max_retries: int = 5,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    sample_num: int = 1,
    presence_penalty: float = 1.0, # Added parameter
    show_progress: bool = True,
):
    """Synchronous helper to run inference as a library function.

    Returns: (results, metrics_rows, summary_dict)
      - results: List[List[str | None]] per prompt (length=sample_num)
      - metrics_rows: per-request metrics dicts
      - summary_dict: {batch_dur, total_prompt_tokens, total_completion_tokens, qps, in_tps, out_tps}
    """
    results, metrics_rows, batch_dur, total_prompt_tokens, total_completion_tokens = asyncio.run(
        run_inference(
            prompts=prompts,
            model=llm_name,
            api_base=api_base,
            api_key=api_key,
            concurrency=concurrency,
            request_timeout=request_timeout,
            max_retries=max_retries,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=sample_num,
            presence_penalty=presence_penalty, # Pass parameter
            show_progress=show_progress,
        )
    )

    qps = (len(prompts) / batch_dur) if batch_dur > 0 else 0.0
    in_tps = (total_prompt_tokens / batch_dur) if batch_dur > 0 else 0.0
    out_tps = (total_completion_tokens / batch_dur) if batch_dur > 0 else 0.0

    summary = {
        "batch_dur": batch_dur,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "qps": qps,
        "in_tps": in_tps,
        "out_tps": out_tps,
    }

    return results, metrics_rows, summary

def load_instructions(input_file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(input_file, index_col=0)
        if "instruction" not in df.columns:
            raise ValueError("'instruction' column not found in CSV")
        return df
    except Exception:
        df = pd.read_json(input_file)
        if "instruction" not in df.columns:
            raise ValueError("'instruction' column not found in JSON")
        return df

