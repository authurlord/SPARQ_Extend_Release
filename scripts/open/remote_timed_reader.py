#!/usr/bin/env python3
"""Universal timed QA reader for REMOTE latency re-measurement (retrieval frozen).

Reads a prompts file from experiments/exports/reader_prompts/<name>.jsonl.gz
({qid, question, gold, prompt} per line; meta in <name>.meta.json gives max_tokens + parse mode),
sends every prompt to a local/remote vLLM endpoint at fixed async concurrency, and reports BOTH
timing views + EM/F1 (SPARQ-parity scoring):
  - wall_total / n          : concurrent throughput (the latency-table "Infer s/q" number)
  - serial_sum  / n         : sum of per-request latencies / n (hardware-concurrency-independent)

Usage (on the 2x4090 server, after starting vLLM):
  python3 scripts/remote_timed_reader.py experiments/exports/reader_prompts/sparqx_cot_ottqa.jsonl.gz \
      --api-base http://localhost:9543/v1 --model <served-name> --concurrency 64
Outputs experiments/results/remote_latency/<name>.<tag>.summary.json
"""
from __future__ import annotations
import argparse, asyncio, gzip, json, os, socket, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import utils.hybridqa_metrics as _hyb
parse_answer = _hyb.parse_answer
from utils.sparq_eval import sparq_em_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", type=Path)
    ap.add_argument("--api-base", default="http://localhost:9543/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = take from meta.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(k, None)

    meta_p = args.prompts.with_suffix("").with_suffix("")  # strip .jsonl.gz
    meta = json.loads(Path(str(meta_p) + ".meta.json").read_text())
    max_tokens = args.max_tokens or meta["max_tokens"]
    parse_mode = meta.get("parse", "direct")
    name = args.prompts.name.replace(".jsonl.gz", "")

    rows = []
    with gzip.open(args.prompts, "rt") as f:
        for l in f:
            if l.strip():
                rows.append(json.loads(l))
    if args.max_queries:
        rows = rows[: args.max_queries]
    print(f"[load] {name}: {len(rows)} prompts  max_tokens={max_tokens} parse={parse_mode}", flush=True)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.api_base.rstrip("/"), api_key=args.api_key, timeout=600)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    recs = []
    em_sum = f1_sum = 0.0
    n_ok = n_err = prog = 0
    t_start = time.time()

    async def one(r):
        nonlocal em_sum, f1_sum, n_ok, n_err, prog
        async with sem:
            t0 = time.time()
            try:
                rr = await client.chat.completions.create(
                    model=args.model, messages=[{"role": "user", "content": r["prompt"]}],
                    max_tokens=max_tokens, temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                raw = rr.choices[0].message.content or ""
                usage = {"pt": rr.usage.prompt_tokens, "ct": rr.usage.completion_tokens}
                err = None
            except Exception as e:
                raw, usage, err = "", {}, repr(e)[:160]
            dt = time.time() - t0
        pred = (parse_answer(raw) if parse_mode == "cot" else raw.strip()) if not err else ""
        em, f1 = (sparq_em_f1(pred, r["gold"], question=r["question"]) if not err else (0.0, 0.0))
        async with lock:
            recs.append({"qid": r["qid"], "t_read": round(dt, 4), **usage,
                         "em": float(em), "f1": round(f1, 4), "err": err})
            if err:
                n_err += 1
            else:
                em_sum += em; f1_sum += f1; n_ok += 1
            prog += 1
            if prog % 200 == 0:
                print(f"  [{prog}/{len(rows)}] EM={em_sum/max(1,n_ok):.4f} err={n_err} "
                      f"{time.time()-t_start:.0f}s", flush=True)

    _run(rows, one)

    wall = time.time() - t_start
    n = len(rows)
    serial = sum(x["t_read"] for x in recs)
    cts = [x.get("ct", 0) for x in recs if not x["err"]]
    summary = {
        "name": name, "tag": args.tag, "host": socket.gethostname(),
        "model": args.model, "api_base": args.api_base, "concurrency": args.concurrency,
        "max_tokens": max_tokens, "parse": parse_mode, "n": n, "n_errors": n_err,
        "EM": round(em_sum / max(1, n_ok), 4), "F1": round(f1_sum / max(1, n_ok), 4),
        "reading_wall_total_sec": round(wall, 2),
        "reading_wall_per_query_sec": round(wall / n, 4),
        "reading_serial_sum_sec": round(serial, 2),
        "reading_serial_per_query_sec": round(serial / n, 4),
        "avg_completion_tokens": round(sum(cts) / max(1, len(cts)), 1),
        "_recombine": "new Total s/q = (frozen index s/q) + (frozen retrieval s/q) + reading_wall_per_query_sec; see EFFICIENCY/LATENCY_TABLE.md",
    }
    outdir = REPO / "experiments/results/remote_latency"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{name}{('.' + args.tag) if args.tag else ''}.summary.json"
    out.write_text(json.dumps(summary, indent=2))
    (outdir / f"{name}{('.' + args.tag) if args.tag else ''}.per_query.jsonl").write_text(
        "\n".join(json.dumps(x) for x in recs))
    print(json.dumps(summary, indent=2))


def _run(rows, one):
    async def go():
        await asyncio.gather(*[one(r) for r in rows])
    asyncio.run(go())


if __name__ == "__main__":
    main()
