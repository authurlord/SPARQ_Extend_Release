#!/usr/bin/env python3
"""
HybridQA pipeline scaffold on top of the current SPARQ-style decomposition:
1. SQL grounding over the table
2. linked-passage retrieval from selected rows
3. optional reranking
4. final direct QA or CoT QA
"""

import argparse
import asyncio
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import httpx
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from openai import AsyncOpenAI, OpenAI
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.hybridqa_metrics import (
    compute_bertscore_f1,
    exact_match,
    normalize_answer,
    parse_answer as _parse_answer_base,
    table_to_rows,
    token_precision_recall_f1,
    tokenized,
)

for var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(var, None)


EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
DEFAULT_OUT_DIR = Path("/data/workspace/yanmy/SPARQ_Extend/results/qwen35_35b/hybridqa_sql_rag")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SQL_TEMPLATE = SCRIPT_DIR / "sql_reason_hybridqa_direct.txt"
DEFAULT_COT_TEMPLATE = SCRIPT_DIR / "hybridqa_qa_cot.txt"
DEFAULT_REWRITE_TEMPLATE = SCRIPT_DIR / "hybridqa_rewrite_query.txt"
DEFAULT_BERTSCORE_MODEL = Path("/data/workspace/yanmy/models/deberta-v3-large")
SQL_LITERAL_END_CHARS = set(" ,);\n\r\t")


def api_root(api_base: str) -> str:
    return api_base[:-3] if api_base.endswith("/v1") else api_base.rstrip("/")


def auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _check_is_sleeping(base: str, headers: Dict[str, str]) -> bool:
    try:
        status = requests.get(f"{base}/is_sleeping", headers=headers, timeout=5)
        if status.status_code == 200 and status.json().get("is_sleeping"):
            print(f"  [vLLM] sleep status -> {status.text.strip()}", flush=True)
            return True
    except Exception:
        pass
    return False


def vllm_sleep(api_base: str = "http://localhost:8000/v1", api_key: str = "") -> None:
    base = api_root(api_base)
    headers = auth_headers(api_key)
    try:
        response = requests.post(
            f"{base}/sleep",
            headers=headers,
            params={"level": 1, "mode": "abort"},
            timeout=30,
        )
        print(f"  [vLLM] sleep -> {response.status_code} @ {base}", flush=True)
        for _ in range(12):
            time.sleep(1)
            if _check_is_sleeping(base, headers):
                return
    except Exception as exc:
        if _check_is_sleeping(base, headers):
            return
        print(f"  [vLLM] sleep failed @ {base}: {exc}", flush=True)


def vllm_wake(api_base: str = "http://localhost:8000/v1", api_key: str = "api-key-qwen3") -> None:
    base = api_root(api_base)
    headers = auth_headers(api_key)
    try:
        response = requests.post(f"{base}/wake_up", headers=headers, timeout=30)
        print(f"  [vLLM] wake_up -> {response.status_code} @ {base}", flush=True)
        for _ in range(24):
            time.sleep(3)
            try:
                sleep_status = requests.get(f"{base}/is_sleeping", headers=headers, timeout=5)
                if sleep_status.status_code == 200 and sleep_status.json().get("is_sleeping"):
                    continue
                ready = requests.get(f"{base}/v1/models", headers=headers, timeout=5)
                if ready.status_code == 200:
                    print(f"  [vLLM] server ready @ {base}", flush=True)
                    return
            except Exception:
                pass
        print(f"  [vLLM] WARNING: server may not be fully ready @ {base}", flush=True)
    except Exception as exc:
        print(f"  [vLLM] wake_up failed @ {base}: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HybridQA SQL + linked passage pipeline")
    parser.add_argument("--model", default="qwen3.5-35b")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="api-key-qwen3")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--first-n", type=int, default=100)
    parser.add_argument("--indices-file", type=Path, default=None)
    parser.add_argument("--subset-tag", default="")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--healthcheck-timeout", type=float, default=5.0)
    parser.add_argument("--sql-max-tokens", type=int, default=256)
    parser.add_argument("--qa-max-tokens", type=int, default=2048)
    parser.add_argument("--sql-repair-rounds", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--retrieval-pool-size", type=int, default=24)
    parser.add_argument("--retrieval-mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--row-selection-mode", choices=["sql", "all"], default="sql")
    parser.add_argument("--passage-selection-mode", choices=["topk", "all"], default="topk")
    parser.add_argument(
        "--final-table-filter-mode",
        choices=["none", "evidence_rows", "evidence_rows_cols"],
        default="none",
    )
    parser.add_argument("--max-prompt-chars", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stage-group-size", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--sql-template", type=Path, default=DEFAULT_SQL_TEMPLATE)
    parser.add_argument("--cot-template", type=Path, default=DEFAULT_COT_TEMPLATE)
    parser.add_argument("--rewrite-template", type=Path, default=DEFAULT_REWRITE_TEMPLATE)
    parser.add_argument("--qa-mode", choices=["direct", "cot"], default="direct")
    parser.add_argument("--rewrite-mode", choices=["none", "llm_prf"], default="none")
    parser.add_argument("--rewrite-max-tokens", type=int, default=192)
    parser.add_argument("--rewrite-prefetch-size", type=int, default=8)
    parser.add_argument("--rewrite-prefetch-mode", choices=["bm25", "dense", "hybrid"], default="dense")
    parser.add_argument("--embedding-api-base", default="")
    parser.add_argument("--embedding-api-key", default="embed-key-m3")
    parser.add_argument("--embedding-model", default="/data/workspace/yanmy/models/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-dimensions", type=int, default=512)
    parser.add_argument("--reranker-model-path", default="")
    parser.add_argument("--reranker-api-key", default="reranker-key-m3")
    parser.add_argument("--reranker-devices", default="")
    parser.add_argument("--reranker-urls", default="")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--rerank-top-n", type=int, default=12)
    parser.add_argument("--vllm-sleep-for-rerank", action="store_true")
    parser.add_argument("--bertscore-model-path", type=Path, default=DEFAULT_BERTSCORE_MODEL)
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument("--bertscore-device", default="cpu")
    parser.add_argument("--disable-bertscore", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()



def sanitize_column_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "col"


def hybridqa_table_to_df(table: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, str], List[List[Dict[str, Any]]]]:
    header, rows = table_to_rows(table)
    clean_header = []
    seen: Dict[str, int] = {}
    for name in header:
        clean = sanitize_column_name(name)
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        clean_header.append(clean)

    records = []
    for row_idx, row in enumerate(rows):
        record = {"row_id": row_idx}
        for clean_name, cell in zip(clean_header, row):
            record[clean_name] = str(cell.get("value", ""))
        records.append(record)
    df = pd.DataFrame(records)
    return df, dict(zip(clean_header, header)), rows


def df_to_table_text(df: pd.DataFrame, limit_rows: Optional[int] = None) -> str:
    if df.empty:
        return "(empty)"
    view = df if limit_rows is None else df.head(limit_rows)
    lines = ["col : " + " | ".join(map(str, view.columns.tolist()))]
    for _, row in view.iterrows():
        values = [str(v) for v in row.tolist()]
        row_id = row["row_id"] if "row_id" in view.columns else _
        lines.append(f"row {row_id} : " + " | ".join(values))
    return "\n".join(lines)


def build_create_table_prompt(df: pd.DataFrame, table_name: str) -> str:
    lines = [f"CREATE TABLE {table_name}("]
    for col in df.columns:
        col_type = "int" if col == "row_id" else "text"
        lines.append(f"\t`{col}` {col_type},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append(")")
    return "\n".join(lines)


def build_sql_prompt(sql_template: str, question: str, table_title: str, df: pd.DataFrame) -> str:
    create_stmt = build_create_table_prompt(df, "w")
    table_text = df.to_string(index=False)
    columns = repr(df.columns.tolist())
    return (
        f"{sql_template}\n\n"
        "<input>\n"
        f"{create_stmt}\n"
        "/*\n"
        "SELECT * FROM w;\n"
        f"{table_text}\n"
        "*/\n"
        f"table title: {table_title}\n"
        f"columns: {columns}\n"
        f"Q: {question}\n"
        "<output>\n"
        "SQL:"
    )


def extract_sql(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    code_blocks = re.findall(r"```(?:sql)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if code_blocks:
        raw_sql = code_blocks[-1].strip()
    else:
        sql_lines = re.findall(r"(?im)^\s*SQL\s*:\s*(.*)$", text)
        if sql_lines:
            raw_sql = sql_lines[-1].strip()
        else:
            match = re.search(r"(?is)(select\b.*)", text)
            raw_sql = match.group(1).strip() if match else text

    raw_sql = re.sub(r"^\s*SQL\s*:\s*", "", raw_sql, flags=re.IGNORECASE).strip()
    if not raw_sql:
        return ""

    select_match = re.search(r"(?is)(select\b.*)", raw_sql)
    if select_match:
        raw_sql = select_match.group(1).strip()

    last_semicolon = raw_sql.rfind(";")
    if last_semicolon != -1:
        raw_sql = raw_sql[: last_semicolon + 1].strip()

    return raw_sql


def escape_inner_apostrophes(sql: str) -> str:
    if not sql or "'" not in sql:
        return sql

    pieces: List[str] = []
    in_string = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch != "'":
            pieces.append(ch)
            i += 1
            continue

        if not in_string:
            in_string = True
            pieces.append(ch)
            i += 1
            continue

        next_char = sql[i + 1] if i + 1 < len(sql) else ""
        if next_char == "'":
            pieces.append("''")
            i += 2
            continue

        if next_char and next_char not in SQL_LITERAL_END_CHARS:
            pieces.append("''")
            i += 1
            continue

        in_string = False
        pieces.append(ch)
        i += 1

    return "".join(pieces)


def normalize_sql_for_duckdb(sql: str) -> str:
    sql = (sql or "").replace("\r", " ").replace("\n", " ").strip()
    if not sql:
        return ""
    sql = sql.replace("\\'", "''")
    sql = re.sub(r"([A-Za-z0-9])\s+'([A-Za-z])", r"\1'\2", sql)
    sql = re.sub(r"`([^`]+)`", r'"\1"', sql)
    sql = escape_inner_apostrophes(sql)
    if not sql.endswith(";"):
        sql += ";"
    return sql


def build_sql_retry_prompt(original_prompt: str, failed_sql: str, error_msg: str, columns: Sequence[str]) -> str:
    return (
        f"{original_prompt}\n\n"
        "Previous SQL attempt failed.\n"
        f"SQL: {failed_sql}\n"
        f"Error: {error_msg}\n"
        f"Available columns: {list(columns)}\n"
        "Please write a corrected SQL query. Output ONLY the SQL query, nothing else.\n"
    )


def execute_sql(df: pd.DataFrame, sql: str) -> Tuple[pd.DataFrame, str]:
    if not sql:
        return pd.DataFrame(), "empty_sql"
    conn = duckdb.connect()
    try:
        conn.register("w", df)
        out_df = conn.execute(normalize_sql_for_duckdb(sql)).fetch_df()
        return out_df, ""
    except Exception as exc:
        return pd.DataFrame(), str(exc)
    finally:
        conn.close()


def collect_row_ids(sql_df: pd.DataFrame) -> List[int]:
    if sql_df.empty:
        return []
    for col in sql_df.columns:
        if str(col).lower() == "row_id":
            ids = []
            for value in sql_df[col].tolist():
                try:
                    ids.append(int(value))
                except Exception:
                    continue
            return ids
    return []


def extract_passages_from_rows(
    rows: Sequence[Sequence[Dict[str, Any]]],
    headers: Sequence[str],
    row_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    selected_ids = set(row_ids) if row_ids else set(range(len(rows)))
    passages: List[Dict[str, Any]] = []
    seen = set()
    for row_idx, row in enumerate(rows):
        if row_idx not in selected_ids:
            continue
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
                        "col_name": headers[col_idx],
                        "cell_value": cell.get("value", ""),
                        "url": url,
                        "summary": summary,
                        "text": (
                            f"row {row_idx}, column {headers[col_idx]}, cell value {cell.get('value', '')}\n"
                            f"passage: {summary}"
                        ),
                    }
                )
    return passages


def retrieve_passages(question: str, passages: Sequence[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not passages:
        return []
    corpus = [tokenized(p["text"]) for p in passages]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized(question))
    ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:top_n]]


def retrieve_passages_multi_query(
    queries: Sequence[Tuple[str, str]],
    passages: Sequence[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    if not passages:
        return []
    clean_queries = []
    seen = set()
    for label, query in queries:
        text = re.sub(r"\s+", " ", str(query or "").strip())
        if not text:
            continue
        dedupe_key = (label, text.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        clean_queries.append((label, text))
    if not clean_queries:
        return list(passages[:top_n])

    corpus = [tokenized(p["text"]) for p in passages]
    bm25 = BM25Okapi(corpus)
    weights = {
        "original": 1.0,
        "general": 0.9,
        "balanced": 1.0,
        "specific": 1.15,
    }
    rrf_k = 60.0
    aggregate = np.zeros(len(passages), dtype=np.float32)
    for label, query in clean_queries:
        scores = bm25.get_scores(tokenized(query))
        ranked_indices = np.argsort(scores)[::-1]
        weight = weights.get(label, 1.0)
        for rank, idx in enumerate(ranked_indices, start=1):
            aggregate[idx] += weight / (rrf_k + rank)
    ranked = sorted(zip(aggregate.tolist(), passages), key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:top_n]]


def minmax_scale(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return arr
    low = float(arr.min())
    high = float(arr.max())
    if high - low <= 1e-12:
        if high > 0:
            return np.ones_like(arr, dtype=np.float32)
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - low) / (high - low)


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    return matrix / norm


def _rank_passages_by_score(
    candidate_passages: Sequence[Dict[str, Any]],
    question: str,
    dense_scores: Optional[np.ndarray],
    mode: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    """Rank passages using bm25, dense, or hybrid scores."""
    if not candidate_passages:
        return []
    if mode == "bm25" or dense_scores is None:
        bm25 = BM25Okapi([tokenized(p["text"]) for p in candidate_passages])
        scores = bm25.get_scores(tokenized(question))
        ranked = sorted(zip(scores.tolist(), candidate_passages), key=lambda x: x[0], reverse=True)
        return [p for _, p in ranked[:top_n]]
    if mode == "dense":
        combined = dense_scores
    else:
        bm25 = BM25Okapi([tokenized(p["text"]) for p in candidate_passages])
        bm25_scores = bm25.get_scores(tokenized(question))
        combined = 0.5 * minmax_scale(bm25_scores) + 0.5 * minmax_scale(dense_scores)
    ranked = sorted(zip(combined.tolist(), candidate_passages), key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:top_n]]


def rerank_passages(
    question: str,
    passages: Sequence[Dict[str, Any]],
    reranker: Any,
    top_k: int,
    reranker_batch_size: int,
) -> List[Dict[str, Any]]:
    if not passages or reranker is None:
        return list(passages[:top_k])
    pairs = [[question, p["text"]] for p in passages]
    scores = reranker.compute_score(
        pairs,
        batch_size=min(len(pairs), reranker_batch_size),
        normalize=True,
    )
    if not isinstance(scores, list):
        scores = [scores]
    ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:top_k]]


def get_sync_openai_client(api_base: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(trust_env=False),
    )


def batch_embed_texts(
    client: OpenAI,
    model: str,
    texts: Sequence[str],
    batch_size: int,
    dimensions: Optional[int] = None,
) -> np.ndarray:
    """
    KGQA-style embedding helper:
    1. send a list of texts in batches
    2. restore response order using the API-returned index

    This is not used by the current BM25 baseline yet, but it is the intended
    building block for the next dense / hybrid retrieval stage.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    all_embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        request_kwargs: Dict[str, Any] = {
            "input": batch,
            "model": model,
        }
        if dimensions is not None:
            request_kwargs["dimensions"] = dimensions
        response = client.embeddings.create(**request_kwargs)
        ordered_embeddings: List[Optional[List[float]]] = [None] * len(batch)
        for item in response.data:
            ordered_embeddings[item.index] = item.embedding
        all_embeddings.extend([embedding for embedding in ordered_embeddings if embedding is not None])
    return np.array(all_embeddings, dtype=np.float32)


def build_final_prompt(
    qa_mode: str,
    cot_template: str,
    question: str,
    table_title: str,
    selected_df: pd.DataFrame,
    sql_df: pd.DataFrame,
    passages: Sequence[Dict[str, Any]],
) -> str:
    selected_text = df_to_table_text(selected_df, limit_rows=20)
    sql_text = df_to_table_text(sql_df, limit_rows=20)
    passage_blocks = []
    for idx, passage in enumerate(passages, start=1):
        passage_blocks.append(
            f"[Passage {idx}] row={passage['row_idx']} col={passage['col_name']} "
            f"cell={passage['cell_value']}\n{passage['summary']}"
        )
    passages_text = "\n\n".join(passage_blocks) if passage_blocks else "(no linked passages selected)"

    if qa_mode == "cot":
        return (
            f"{cot_template}\n\n"
            f"Table Title: {table_title}\n"
            f"Question: {question}\n"
            f"Selected Table:\n{selected_text}\n\n"
            f"SQL Result:\n{sql_text}\n\n"
            f"Linked Passages:\n{passages_text}\n"
        )

    return (
        "Answer the question using the selected table rows, SQL result, and linked passages.\n"
        "Return only the final answer span.\n\n"
        f"Table Title: {table_title}\n"
        f"Question: {question}\n"
        f"Selected Table:\n{selected_text}\n\n"
        f"SQL Result:\n{sql_text}\n\n"
        f"Linked Passages:\n{passages_text}\n\n"
        "Answer:"
    )


def fit_prompt_to_budget(prompt: str, passages: Sequence[Dict[str, Any]], max_prompt_chars: int) -> Tuple[str, List[Dict[str, Any]], bool]:
    if max_prompt_chars <= 0 or len(prompt) <= max_prompt_chars:
        return prompt, list(passages), False
    trimmed = list(passages)
    truncated = False
    while trimmed:
        trimmed.pop()
        truncated = True
        passage_blocks = []
        for idx, passage in enumerate(trimmed, start=1):
            passage_blocks.append(
                f"[Passage {idx}] row={passage['row_idx']} col={passage['col_name']} "
                f"cell={passage['cell_value']}\n{passage['summary']}"
            )
        head = prompt.split("Linked Passages:\n", 1)[0]
        new_passages = "\n\n".join(passage_blocks) if passage_blocks else "(no linked passages selected)"
        prompt = head + "Linked Passages:\n" + new_passages
        if len(prompt) <= max_prompt_chars:
            break
    return prompt, trimmed, truncated


def trim_text(text: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= max_chars:
        return clean
    clipped = clean[: max_chars - 3].rsplit(" ", 1)[0].strip()
    return (clipped or clean[: max_chars - 3]).strip() + "..."


def build_rewrite_prompt(
    rewrite_template: str,
    question: str,
    table_title: str,
    selected_df: pd.DataFrame,
    support_passages: Sequence[Dict[str, Any]],
) -> str:
    table_text = df_to_table_text(selected_df, limit_rows=20)
    columns = repr(selected_df.columns.tolist())
    if support_passages:
        hints = []
        for idx, passage in enumerate(support_passages, start=1):
            hints.append(
                f"- [{idx}] row={passage['row_idx']} col={passage['col_name']} "
                f"cell={passage['cell_value']} | {trim_text(passage['summary'], 220)}"
            )
        evidence_text = "\n".join(hints)
    else:
        evidence_text = "- (no passage hints)"

    return (
        f"{rewrite_template}\n\n"
        "<input>\n"
        f"table caption: {table_title}\n"
        "/*\n"
        f"{table_text}\n"
        "*/\n"
        f"columns: {columns}\n"
        f"evidence hints:\n{evidence_text}\n"
        f"Q: {question}\n"
        "</input>\n"
        "<output>\n"
        "[GENERAL]:"
    )


def parse_rewrite_query_dict(text: str, original_question: str) -> Dict[str, str]:
    variants = {"original": re.sub(r"\s+", " ", str(original_question or "").strip())}
    for label in ["GENERAL", "BALANCED", "SPECIFIC"]:
        match = re.search(rf"(?im)^\s*\[{label}\]\s*:\s*(.+)$", text or "", flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        value = match.group(1).strip()
        if "\n" in value:
            value = next((line.strip() for line in value.splitlines() if line.strip()), "")
        value = re.sub(r"\s+", " ", value).strip()
        if value:
            variants[label.lower()] = value
    return variants


def dedupe_query_variants(query_dict: Dict[str, str]) -> List[Tuple[str, str]]:
    ordered: List[Tuple[str, str]] = []
    seen = set()
    for label in ["original", "general", "balanced", "specific"]:
        text = re.sub(r"\s+", " ", str(query_dict.get(label, "")).strip())
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append((label, text))
    return ordered


def build_rerank_query_text(context: Dict[str, Any]) -> str:
    query_dict = context.get("rewrite_query_dict") or {"original": context["sample"]["question"]}
    variants = dedupe_query_variants(query_dict)
    if len(variants) <= 1:
        return variants[0][1] if variants else context["sample"]["question"]
    lines = []
    label_map = {
        "original": "Original Question",
        "general": "General Retrieval Intent",
        "balanced": "Balanced Retrieval Query",
        "specific": "Specific Retrieval Keywords",
    }
    for label, text in variants:
        lines.append(f"{label_map.get(label, label.title())}: {text}")
    return "\n".join(lines)


async def infer_text(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Tuple[str, str]:
    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1,
                max_tokens=max_tokens,
                extra_body=EXTRA_BODY,
            )
        message = response.choices[0].message
        text = (
            getattr(message, "content", None)
            or getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        return str(text).strip(), ""
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


async def run_rewrite_stage(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    rewrite_template: str,
    args: argparse.Namespace,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = build_rewrite_prompt(
        rewrite_template,
        context["sample"]["question"],
        context["table_title"],
        context["selected_df"],
        context.get("rewrite_support_passages", []),
    )
    rewrite_text, rewrite_error = await infer_text(
        client,
        semaphore,
        args.model,
        prompt,
        args.rewrite_max_tokens,
    )
    context["rewrite_prompt"] = prompt
    context["raw_rewrite_text"] = rewrite_text
    context["rewrite_error"] = rewrite_error
    query_dict = (
        parse_rewrite_query_dict(rewrite_text, context["sample"]["question"])
        if rewrite_text and not rewrite_error
        else {"original": context["sample"]["question"]}
    )
    context["rewrite_query_dict"] = query_dict
    context["rewrite_queries"] = [text for _, text in dedupe_query_variants(query_dict)]
    context["rerank_query_text"] = build_rerank_query_text(context)
    return context


def parse_answer(qa_mode: str, text: str) -> str:
    return _parse_answer_base(text)


def parse_reranker_devices(devices_text: str) -> Optional[Any]:
    text = (devices_text or "").strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None
    lowered = [part.lower() for part in parts]
    if all(part == "cpu" for part in lowered):
        return "cpu"
    parsed: List[Any] = []
    for part in parts:
        if part.isdigit():
            parsed.append(int(part))
        else:
            parsed.append(part)
    return parsed if len(parsed) > 1 else parsed[0]


def parse_url_list(urls_text: str) -> List[str]:
    return [part.strip() for part in (urls_text or "").split(",") if part.strip()]


def url_to_service_base(url: str) -> str:
    text = (url or "").strip().rstrip("/")
    for suffix in ["/v1/score", "/score", "/v1/rerank", "/rerank", "/v1/embeddings", "/embeddings"]:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return api_root(text)


def get_sidecar_targets(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    targets: List[Tuple[str, str, str]] = []
    if args.embedding_api_base:
        targets.append((args.embedding_api_base, args.embedding_api_key, "embedding"))
    seen_bases = {url_to_service_base(args.embedding_api_base)} if args.embedding_api_base else set()
    for idx, url in enumerate(parse_url_list(args.reranker_urls)):
        base = url_to_service_base(url)
        if base in seen_bases:
            continue
        seen_bases.add(base)
        targets.append((base, args.reranker_api_key, f"rerank[{idx}]"))
    return targets


def switch_to_retrieve_mode(args: argparse.Namespace) -> None:
    print("[mode] generation -> retrieve", flush=True)
    vllm_sleep(args.api_base, args.api_key)
    for base, api_key, name in get_sidecar_targets(args):
        print(f"  [mode] wake {name}", flush=True)
        vllm_wake(base, api_key)


def switch_to_generation_mode(args: argparse.Namespace) -> None:
    print("[mode] retrieve -> generation", flush=True)
    for base, api_key, name in get_sidecar_targets(args):
        print(f"  [mode] sleep {name}", flush=True)
        vllm_sleep(base, api_key)
    vllm_wake(args.api_base, args.api_key)


def load_reranker(model_path: str, devices_text: str, batch_size: int) -> Any:
    if not model_path:
        return None
    from FlagEmbedding import FlagReranker

    devices = parse_reranker_devices(devices_text)
    use_fp16 = devices != "cpu"
    return FlagReranker(
        model_path,
        use_fp16=use_fp16,
        devices=devices,
        batch_size=batch_size,
    )


def _rerank_vllm_worker(args: Tuple[List[str], List[str], str, str, int, str]) -> List[float]:
    questions, texts, url, model_name, batch_size, api_key = args
    scores: List[float] = []
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for start in range(0, len(questions), batch_size):
        batch_q = questions[start : start + batch_size]
        batch_t = texts[start : start + batch_size]
        try:
            response = requests.post(
                url,
                headers=headers,
                json={
                    "model": model_name,
                    "text_1": batch_q,
                    "text_2": batch_t,
                    "truncate_prompt_tokens": -1,
                },
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            if "data" in payload:
                ordered_scores: List[Optional[float]] = [None] * len(batch_q)
                for item in payload["data"]:
                    index = int(item.get("index", -1))
                    if 0 <= index < len(ordered_scores):
                        ordered_scores[index] = float(item["score"])
                scores.extend([score if score is not None else -999.0 for score in ordered_scores])
            else:
                scores.extend([-999.0] * len(batch_q))
        except Exception:
            scores.extend([-999.0] * len(batch_q))
    return scores


def rerank_scores_vllm_parallel(
    questions: Sequence[str],
    texts: Sequence[str],
    urls_text: str,
    model_name: str,
    batch_size: int,
    api_key: str,
) -> List[float]:
    urls = parse_url_list(urls_text)
    if not urls:
        return []
    total_items = len(questions)
    if total_items == 0:
        return []

    chunk_size = int(np.ceil(total_items / len(urls)))
    futures = []
    results_map: Dict[int, List[float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as executor:
        for worker_idx, url in enumerate(urls):
            start = worker_idx * chunk_size
            end = min((worker_idx + 1) * chunk_size, total_items)
            if start >= total_items:
                break
            futures.append(
                (
                    worker_idx,
                    executor.submit(
                        _rerank_vllm_worker,
                        (
                            list(questions[start:end]),
                            list(texts[start:end]),
                            url,
                            model_name,
                            batch_size,
                            api_key,
                        ),
                    ),
                )
            )
        for worker_idx, future in futures:
            try:
                results_map[worker_idx] = future.result()
            except Exception:
                results_map[worker_idx] = []

    final_scores: List[float] = []
    for worker_idx in range(len(results_map)):
        final_scores.extend(results_map[worker_idx])
    return final_scores


def serialize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serialize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_json(item) for item in value]
    return value


def save_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(serialize_for_json(payload), indent=2, ensure_ascii=False))


def load_payload_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[resume] failed to load {path}: {type(exc).__name__}: {exc}", flush=True)
        return None


def print_progress(done: int, total: int, started: float, results: Sequence[Dict[str, Any]]) -> None:
    elapsed = time.time() - started
    em = sum(1 for r in results if r["exact_match"])
    f1_avg = sum(r["f1"] for r in results) / len(results) if results else 0.0
    errors = sum(1 for r in results if r["error"])
    print(
        f"[progress][sql_rag] {done}/{total} elapsed={elapsed:.1f}s "
        f"EM={em / len(results) * 100:.2f} F1={f1_avg * 100:.2f} errors={errors}",
        flush=True,
    )


def reranker_uses_gpu(devices_text: str) -> bool:
    devices = parse_reranker_devices(devices_text)
    return devices not in (None, "cpu")


def using_rewrite_prefetch_sidecar(args: argparse.Namespace, embedding_client: Optional[OpenAI]) -> bool:
    return (
        args.rewrite_mode != "none"
        and args.rewrite_prefetch_mode in {"dense", "hybrid"}
        and embedding_client is not None
    )


def load_index_list(path: Path) -> List[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        indices: List[int] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, int):
                indices.append(int(item))
            elif isinstance(item, dict):
                if "idx" in item:
                    indices.append(int(item["idx"]))
                elif "index" in item:
                    indices.append(int(item["index"]))
        return indices
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            indices = []
            for item in payload:
                if isinstance(item, int):
                    indices.append(int(item))
                elif isinstance(item, dict):
                    if "idx" in item:
                        indices.append(int(item["idx"]))
                    elif "index" in item:
                        indices.append(int(item["index"]))
            return indices
    return [int(line.strip()) for line in text.splitlines() if line.strip()]


def effective_subset_tag(args: argparse.Namespace) -> str:
    if args.subset_tag:
        return args.subset_tag.strip()
    if args.indices_file:
        return args.indices_file.stem
    return ""


def build_sample_context(sample_idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
    table = sample["table"]
    df, clean_to_original, rows = hybridqa_table_to_df(table)
    headers, _ = table_to_rows(table)
    original_to_clean = {original: clean for clean, original in clean_to_original.items()}
    return {
        "sample_idx": sample_idx,
        "sample": sample,
        "table_title": table.get("title", ""),
        "df": df,
        "rows": rows,
        "headers": headers,
        "clean_to_original": clean_to_original,
        "original_to_clean": original_to_clean,
    }


async def run_sql_stage(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    sql_template: str,
    args: argparse.Namespace,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    sql_prompt = build_sql_prompt(
        sql_template,
        context["sample"]["question"],
        context["table_title"],
        context["df"],
    )
    sql_text, sql_error = await infer_text(
        client,
        semaphore,
        args.model,
        sql_prompt,
        args.sql_max_tokens,
    )
    sql = extract_sql(sql_text)
    sql_df, sql_exec_error = execute_sql(context["df"], sql)
    sql_retry_count = 0

    while sql_exec_error and not sql_error and sql_retry_count < args.sql_repair_rounds:
        sql_retry_count += 1
        retry_prompt = build_sql_retry_prompt(
            sql_prompt,
            sql,
            sql_exec_error,
            context["df"].columns.tolist(),
        )
        retry_text, retry_error = await infer_text(
            client,
            semaphore,
            args.model,
            retry_prompt,
            args.sql_max_tokens,
        )
        if retry_error:
            sql_error = retry_error
            break
        sql_text = retry_text
        sql = extract_sql(retry_text)
        sql_df, sql_exec_error = execute_sql(context["df"], sql)

    sql_selected_row_ids = collect_row_ids(sql_df)
    if args.row_selection_mode == "all" or not sql_selected_row_ids:
        selected_row_ids = context["df"]["row_id"].astype(int).tolist()
    else:
        selected_row_ids = sql_selected_row_ids

    selected_df = context["df"][context["df"]["row_id"].isin(selected_row_ids)].copy()

    context.update(
        {
            "sql_prompt": sql_prompt,
            "raw_sql_text": sql_text,
            "sql": sql,
            "sql_error": sql_error,
            "sql_exec_error": sql_exec_error,
            "sql_retry_count": sql_retry_count,
            "sql_df": sql_df,
            "sql_selected_row_ids": sql_selected_row_ids,
            "selected_row_ids": selected_row_ids,
            "selected_df": selected_df,
        }
    )
    return context



def prepare_rewrite_support_batch(
    contexts: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    embedding_client: Optional[OpenAI],
) -> None:
    if not contexts:
        return

    all_questions: List[str] = []
    all_passage_texts: List[str] = []
    spans: List[Tuple[int, int]] = []

    for context in contexts:
        candidate_passages = extract_passages_from_rows(
            context["rows"],
            context["headers"],
            context["selected_row_ids"],
        )
        context["candidate_passages"] = candidate_passages
        start = len(all_passage_texts)
        all_questions.append(context["sample"]["question"])
        all_passage_texts.extend([passage["text"] for passage in candidate_passages])
        spans.append((start, len(all_passage_texts)))

    use_dense = (
        args.rewrite_prefetch_mode in {"dense", "hybrid"}
        and embedding_client is not None
        and all_questions
        and all_passage_texts
    )
    question_embeddings = np.zeros((0, 0), dtype=np.float32)
    passage_embeddings = np.zeros((0, 0), dtype=np.float32)

    if use_dense:
        dimensions = args.embedding_dimensions if args.embedding_dimensions > 0 else None
        question_embeddings = l2_normalize_rows(
            batch_embed_texts(
                embedding_client,
                args.embedding_model,
                all_questions,
                args.embedding_batch_size,
                dimensions,
            )
        )
        passage_embeddings = l2_normalize_rows(
            batch_embed_texts(
                embedding_client,
                args.embedding_model,
                all_passage_texts,
                args.embedding_batch_size,
                dimensions,
            )
        )

    for context_idx, context in enumerate(contexts):
        candidate_passages = context.get("candidate_passages", [])
        if not candidate_passages:
            context["rewrite_support_passages"] = []
            continue

        question = context["sample"]["question"]
        start, end = spans[context_idx]
        dense = passage_embeddings[start:end] @ question_embeddings[context_idx] if use_dense else None
        context["rewrite_support_passages"] = _rank_passages_by_score(
            candidate_passages, question, dense, args.rewrite_prefetch_mode, args.rewrite_prefetch_size,
        )


def prepare_retrieval_context_batch(
    contexts: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    embedding_client: Optional[OpenAI],
) -> None:
    if not contexts:
        return

    all_question_texts: List[str] = []
    all_passage_texts: List[str] = []
    spans: List[Tuple[int, int]] = []

    for context in contexts:
        candidate_passages = context.get("candidate_passages") or extract_passages_from_rows(
            context["rows"],
            context["headers"],
            context["selected_row_ids"],
        )
        context["candidate_passages"] = candidate_passages
        start = len(all_passage_texts)
        query_dict = context.get("rewrite_query_dict") or {"original": context["sample"]["question"]}
        retrieval_queries = dedupe_query_variants(query_dict)
        all_question_texts.append((retrieval_queries[0][1] if retrieval_queries else context["sample"]["question"]))
        all_passage_texts.extend([passage["text"] for passage in candidate_passages])
        spans.append((start, len(all_passage_texts)))

    use_dense = args.retrieval_mode in {"dense", "hybrid"} and embedding_client is not None
    question_embeddings = np.zeros((0, 0), dtype=np.float32)
    passage_embeddings = np.zeros((0, 0), dtype=np.float32)

    if use_dense and all_question_texts and all_passage_texts:
        dimensions = args.embedding_dimensions if args.embedding_dimensions > 0 else None
        question_embeddings = l2_normalize_rows(
            batch_embed_texts(
                embedding_client,
                args.embedding_model,
                all_question_texts,
                args.embedding_batch_size,
                dimensions,
            )
        )
        passage_embeddings = l2_normalize_rows(
            batch_embed_texts(
                embedding_client,
                args.embedding_model,
                all_passage_texts,
                args.embedding_batch_size,
                dimensions,
            )
        )

    for context_idx, context in enumerate(contexts):
        candidate_passages = context.get("candidate_passages", [])
        if not candidate_passages:
            context["retrieved_passages"] = []
            continue

        question = context["sample"]["question"]
        query_dict = context.get("rewrite_query_dict") or {"original": question}
        retrieval_queries = dedupe_query_variants(query_dict)
        start, end = spans[context_idx]

        if args.retrieval_mode == "bm25" or not use_dense:
            context["retrieved_passages"] = retrieve_passages_multi_query(
                retrieval_queries,
                candidate_passages,
                args.retrieval_pool_size,
            )
            continue

        dense = passage_embeddings[start:end] @ question_embeddings[context_idx]
        context["retrieved_passages"] = _rank_passages_by_score(
            candidate_passages, question, dense, args.retrieval_mode, args.retrieval_pool_size,
        )


def flatten_rerank_pairs(contexts: Sequence[Dict[str, Any]]) -> Tuple[List[List[str]], List[Tuple[int, int]]]:
    """
    Mirror the KGQA batching pattern:
    flatten all (question, doc) pairs once, then slice the flat score list back
    to each sample using spans.
    """
    all_pairs: List[List[str]] = []
    spans: List[Tuple[int, int]] = []
    for context in contexts:
        start = len(all_pairs)
        retrieved = context.get("retrieved_passages", [])
        rerank_query = context.get("rerank_query_text") or context["sample"]["question"]
        all_pairs.extend([[rerank_query, passage["text"]] for passage in retrieved])
        spans.append((start, len(all_pairs)))
    return all_pairs, spans


def assign_rerank_results(
    contexts: Sequence[Dict[str, Any]],
    spans: Sequence[Tuple[int, int]],
    scores: Sequence[float],
    top_k: int,
) -> None:
    for context, (start, end) in zip(contexts, spans):
        retrieved = context.get("retrieved_passages", [])
        if end <= start:
            context["selected_passages"] = []
            continue
        ranked = sorted(
            zip(scores[start:end], retrieved),
            key=lambda item: item[0],
            reverse=True,
        )
        context["selected_passages"] = [passage for _, passage in ranked[:top_k]]


def rerank_context_batch(
    contexts: Sequence[Dict[str, Any]],
    reranker: Any,
    args: argparse.Namespace,
    manage_vllm_sleep: bool = True,
) -> None:
    if not contexts:
        return

    if args.passage_selection_mode == "all":
        for context in contexts:
            context["selected_passages"] = list(context.get("candidate_passages", []))
        return

    all_pairs, spans = flatten_rerank_pairs(contexts)
    total_pairs = len(all_pairs)
    use_vllm_rerank = bool(parse_url_list(args.reranker_urls))
    use_gpu_rerank = manage_vllm_sleep and args.vllm_sleep_for_rerank and total_pairs > 0 and (
        use_vllm_rerank or (reranker is not None and reranker_uses_gpu(args.reranker_devices))
    )

    if use_gpu_rerank and total_pairs > 0:
        print(
            f"[stage][rerank] sleeping vLLM once for {len(contexts)} samples, {total_pairs} pairs",
            flush=True,
        )
        vllm_sleep(args.api_base)

    try:
        if reranker is None:
            if use_vllm_rerank and all_pairs:
                questions = [pair[0] for pair in all_pairs]
                texts = [pair[1] for pair in all_pairs]
                scores = rerank_scores_vllm_parallel(
                    questions,
                    texts,
                    args.reranker_urls,
                    args.reranker_model_path,
                    args.reranker_batch_size,
                    args.reranker_api_key,
                )
                assign_rerank_results(contexts, spans, scores, args.top_k)
                return
            for context in contexts:
                context["selected_passages"] = list(context.get("retrieved_passages", [])[: args.top_k])
            return

        scores: List[float] = []
        if all_pairs:
            computed = reranker.compute_score(
                all_pairs,
                batch_size=min(len(all_pairs), args.reranker_batch_size),
                normalize=True,
            )
            if isinstance(computed, list):
                scores = computed
            else:
                scores = [computed]
        assign_rerank_results(contexts, spans, scores, args.top_k)
    finally:
        if use_gpu_rerank and total_pairs > 0:
            print(
                f"[stage][rerank] waking vLLM after {len(contexts)} samples",
                flush=True,
            )
            vllm_wake(args.api_base, args.api_key)


def filter_final_table_df(context: Dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    base_df = context["selected_df"].copy()
    if args.final_table_filter_mode == "none" or base_df.empty:
        return base_df

    evidence_row_ids = {
        int(p["row_idx"])
        for p in context.get("selected_passages", [])
        if str(p.get("row_idx", "")).isdigit()
    }
    evidence_row_ids.update(int(row_id) for row_id in context.get("sql_selected_row_ids", []) if row_id is not None)
    if not evidence_row_ids:
        return base_df

    filtered = base_df[base_df["row_id"].isin(sorted(evidence_row_ids))].copy()
    if args.final_table_filter_mode != "evidence_rows_cols" or filtered.empty:
        return filtered

    keep_cols = ["row_id"]
    original_to_clean = context.get("original_to_clean", {})
    for passage in context.get("selected_passages", []):
        clean_col = original_to_clean.get(passage.get("col_name"))
        if clean_col and clean_col in filtered.columns and clean_col not in keep_cols:
            keep_cols.append(clean_col)
    sql_df = context.get("sql_df")
    if isinstance(sql_df, pd.DataFrame):
        for col in sql_df.columns.tolist():
            if col in filtered.columns and col not in keep_cols:
                keep_cols.append(col)
    if len(keep_cols) == 1:
        return filtered
    return filtered[keep_cols].copy()


def prepare_final_prompt(
    context: Dict[str, Any],
    cot_template: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    final_selected_df = filter_final_table_df(context, args)
    context["final_selected_df"] = final_selected_df
    final_prompt = build_final_prompt(
        qa_mode=args.qa_mode,
        cot_template=cot_template,
        question=context["sample"]["question"],
        table_title=context["table_title"],
        selected_df=final_selected_df,
        sql_df=context["sql_df"],
        passages=context.get("selected_passages", []),
    )
    final_prompt, selected_passages, truncated = fit_prompt_to_budget(
        final_prompt,
        context.get("selected_passages", []),
        args.max_prompt_chars,
    )
    context["final_prompt"] = final_prompt
    context["selected_passages"] = selected_passages
    context["truncated_for_budget"] = truncated
    return context


async def run_qa_stage(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    qa_text, qa_error = await infer_text(
        client,
        semaphore,
        args.model,
        context["final_prompt"],
        args.qa_max_tokens,
    )
    pred_answer = parse_answer(args.qa_mode, qa_text)
    error = context["sql_error"] or context["sql_exec_error"] or qa_error or None
    gold_answer = context["sample"]["answer_text"]
    precision, recall, f1 = token_precision_recall_f1(pred_answer, gold_answer)

    result = {
        "idx": context["sample_idx"],
        "question_id": context["sample"]["question_id"],
        "question": context["sample"]["question"],
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "exact_match": exact_match(pred_answer, gold_answer),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "error": error,
        "raw_sql_text": context["raw_sql_text"],
        "sql": context["sql"],
        "sql_retry_count": context["sql_retry_count"],
        "raw_rewrite_text": context.get("raw_rewrite_text", ""),
        "rewrite_queries": context.get("rewrite_queries", [context["sample"]["question"]]),
        "rewrite_support_count": len(context.get("rewrite_support_passages", [])),
        "raw_qa_text": qa_text,
        "sql_prompt_chars": len(context["sql_prompt"]),
        "qa_prompt_chars": len(context["final_prompt"]),
        "sql_selected_row_ids": context.get("sql_selected_row_ids", []),
        "selected_row_ids": context["selected_row_ids"],
        "original_selected_row_count": len(context["selected_df"]),
        "selected_row_count": len(context.get("final_selected_df", context["selected_df"])),
        "candidate_passage_count": len(context.get("candidate_passages", [])),
        "selected_passage_count": len(context.get("selected_passages", [])),
        "truncated_for_budget": context["truncated_for_budget"],
        "selected_passages": [
            {
                "row_idx": p["row_idx"],
                "col_name": p["col_name"],
                "cell_value": p["cell_value"],
                "url": p["url"],
            }
            for p in context.get("selected_passages", [])
        ],
    }
    status = "EM" if result["exact_match"] else "MISS"
    print(
        f"[sql_rag][{args.qa_mode}] {context['sample_idx']}: {status} rows={result['selected_row_count']:02d} "
        f"passages={result['selected_passage_count']:02d}/{result['candidate_passage_count']:02d} "
        f"f1={result['f1']:.2f} q={context['sample']['question'][:60]}",
        flush=True,
    )
    return result


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    f1_avg = sum(r["f1"] for r in results) / total if total else 0.0
    p_avg = sum(r.get("precision", 0.0) for r in results) / total if total else 0.0
    r_avg = sum(r.get("recall", 0.0) for r in results) / total if total else 0.0
    errors = sum(1 for r in results if r["error"])
    truncated = sum(1 for r in results if r["truncated_for_budget"])
    return {
        "total": total,
        "exact_match": round(exact / total * 100, 2) if total else 0.0,
        "precision": round(p_avg * 100, 2),
        "recall": round(r_avg * 100, 2),
        "f1": round(f1_avg * 100, 2),
        "bertscore_f1": None,
        "errors": errors,
        "truncated_for_budget": round(truncated / total * 100, 2) if total else 0.0,
        "avg_selected_rows": round(sum(r["selected_row_count"] for r in results) / total, 2) if total else 0.0,
        "avg_candidate_passages": round(sum(r["candidate_passage_count"] for r in results) / total, 2) if total else 0.0,
        "avg_selected_passages": round(sum(r["selected_passage_count"] for r in results) / total, 2) if total else 0.0,
        "avg_qa_prompt_chars": round(sum(r["qa_prompt_chars"] for r in results) / total, 2) if total else 0.0,
    }


def finalize_summary(results: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    summary = summarize(results)
    if not args.disable_bertscore:
        summary["bertscore_f1"] = compute_bertscore_f1(
            results,
            args.bertscore_model_path,
            args.bertscore_batch_size,
            args.bertscore_device,
        )
    return summary


def using_embedding_sidecar(args: argparse.Namespace, embedding_client: Optional[OpenAI]) -> bool:
    return args.retrieval_mode in {"dense", "hybrid"} and embedding_client is not None


def using_rerank_sidecar(args: argparse.Namespace, reranker: Any) -> bool:
    return bool(parse_url_list(args.reranker_urls)) or (reranker is not None and reranker_uses_gpu(args.reranker_devices))


async def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sql_template = args.sql_template.read_text(encoding="utf-8")
    cot_template = args.cot_template.read_text(encoding="utf-8")
    rewrite_template = ""
    if args.rewrite_mode != "none":
        rewrite_template = args.rewrite_template.read_text(encoding="utf-8")
    reranker = None
    if not parse_url_list(args.reranker_urls):
        reranker = load_reranker(args.reranker_model_path, args.reranker_devices, args.reranker_batch_size)
    embedding_client: Optional[OpenAI] = None
    if (
        args.embedding_api_base
        and (
            args.retrieval_mode in {"dense", "hybrid"}
            or args.rewrite_mode != "none"
        )
    ):
        embedding_client = get_sync_openai_client(args.embedding_api_base, args.embedding_api_key)

    client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key, timeout=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    dataset = load_dataset("hybrid_qa", split=args.split)
    if args.indices_file:
        sample_indices = load_index_list(args.indices_file)
    else:
        end = min(args.start + args.first_n, len(dataset))
        sample_indices = list(range(args.start, end))
    samples = [dataset[i] for i in sample_indices]

    subset_tag = effective_subset_tag(args)
    if subset_tag:
        out_file = args.out_dir / f"hybridqa_sql_rag_{subset_tag}_n{len(samples)}_{args.qa_mode}.json"
    else:
        out_file = args.out_dir / f"hybridqa_sql_rag_start{args.start}_n{len(samples)}_{args.qa_mode}.json"
    partial_file = out_file.with_suffix(".partial.json")
    final_payload = load_payload_if_exists(out_file)
    if final_payload is not None and not args.no_resume:
        print(f"[resume] final output already exists: {out_file}", flush=True)
        print(json.dumps(final_payload.get("summary", {}), indent=2, ensure_ascii=False), flush=True)
        return

    started = time.time()
    payload: Dict[str, Any]
    done_before = 0
    elapsed_before = 0.0

    if not args.no_resume:
        existing_payload = load_payload_if_exists(partial_file)
        if existing_payload is not None:
            payload = existing_payload
            payload["config"] = serialize_for_json(vars(args))
            payload.setdefault("summary", {})
            payload.setdefault("results", [])
            done_before = len(payload["results"])
            elapsed_before = float(payload.get("elapsed_sec") or 0.0)
            print(
                f"[resume] loaded {done_before} completed samples from {partial_file}",
                flush=True,
            )
        else:
            payload = {
                "config": serialize_for_json(vars(args)),
                "summary": {},
                "results": [],
            }
    else:
        payload = {
            "config": serialize_for_json(vars(args)),
            "summary": {},
            "results": [],
        }

    if done_before >= len(samples):
        payload["summary"] = finalize_summary(payload["results"], args)
        payload["elapsed_sec"] = round(elapsed_before, 2)
        save_payload(out_file, payload)
        if partial_file.exists():
            partial_file.unlink()
        print(f"[resume] nothing to do; all {len(samples)} samples are already completed", flush=True)
        return

    for group_start in range(done_before, len(samples), args.stage_group_size):
        group = samples[group_start : group_start + args.stage_group_size]
        contexts = [
            build_sample_context(args.start + group_start + i, sample)
            for i, sample in enumerate(group)
        ]

        print(
            f"[stage][sql] group_start={args.start + group_start} samples={len(contexts)}",
            flush=True,
        )
        for batch_start in range(0, len(contexts), args.batch_size):
            batch_contexts = contexts[batch_start : batch_start + args.batch_size]
            await asyncio.gather(
                *[
                    run_sql_stage(client, semaphore, sql_template, args, context)
                    for context in batch_contexts
                ]
            )

        if args.rewrite_mode != "none":
            rewrite_prefetch_uses_sidecar = args.vllm_sleep_for_rerank and using_rewrite_prefetch_sidecar(args, embedding_client)
            if rewrite_prefetch_uses_sidecar:
                print(
                    f"[stage][rewrite_prefetch] switching to retrieve mode for {len(contexts)} samples",
                    flush=True,
                )
                switch_to_retrieve_mode(args)
            try:
                print(
                    f"[stage][rewrite_prefetch] group_start={args.start + group_start} samples={len(contexts)} "
                    f"mode={args.rewrite_prefetch_mode}",
                    flush=True,
                )
                prepare_rewrite_support_batch(contexts, args, embedding_client)
            finally:
                if rewrite_prefetch_uses_sidecar:
                    print(
                        f"[stage][rewrite_prefetch] switching back to generation mode after support retrieval for {len(contexts)} samples",
                        flush=True,
                    )
                    switch_to_generation_mode(args)

            print(
                f"[stage][rewrite] group_start={args.start + group_start} samples={len(contexts)}",
                flush=True,
            )
            for batch_start in range(0, len(contexts), args.batch_size):
                batch_contexts = contexts[batch_start : batch_start + args.batch_size]
                await asyncio.gather(
                    *[
                        run_rewrite_stage(client, semaphore, rewrite_template, args, context)
                        for context in batch_contexts
                    ]
                )

        use_sidecar_gpu = args.vllm_sleep_for_rerank and (
            using_embedding_sidecar(args, embedding_client) or using_rerank_sidecar(args, reranker)
        )
        if use_sidecar_gpu:
            print(
                f"[stage][retrieve] switching to retrieve mode for {len(contexts)} samples",
                flush=True,
            )
            switch_to_retrieve_mode(args)

        try:
            print(
                f"[stage][retrieve] group_start={args.start + group_start} samples={len(contexts)} mode={args.retrieval_mode}",
                flush=True,
            )
            prepare_retrieval_context_batch(contexts, args, embedding_client)

            print(
                f"[stage][rerank] group_start={args.start + group_start} samples={len(contexts)}",
                flush=True,
            )
            rerank_context_batch(contexts, reranker, args, manage_vllm_sleep=False)
        finally:
            if use_sidecar_gpu:
                print(
                    f"[stage][retrieve] switching back to generation mode after retrieval/rerank for {len(contexts)} samples",
                    flush=True,
                )
                switch_to_generation_mode(args)

        for context in contexts:
            prepare_final_prompt(context, cot_template, args)

        print(
            f"[stage][qa] group_start={args.start + group_start} samples={len(contexts)}",
            flush=True,
        )
        for batch_start in range(0, len(contexts), args.batch_size):
            batch_contexts = contexts[batch_start : batch_start + args.batch_size]
            batch_results = await asyncio.gather(
                *[
                    run_qa_stage(client, semaphore, args, context)
                    for context in batch_contexts
                ]
            )
            payload["results"].extend(batch_results)
            done = len(payload["results"])
            if done % args.progress_every == 0 or done == len(samples):
                print_progress(done, len(samples), started - elapsed_before, payload["results"])
            if done % args.save_every == 0 or done == len(samples):
                payload["summary"] = summarize(payload["results"])
                payload["elapsed_sec"] = round(elapsed_before + (time.time() - started), 2)
                save_payload(partial_file, payload)

    payload["summary"] = finalize_summary(payload["results"], args)
    payload["elapsed_sec"] = round(elapsed_before + (time.time() - started), 2)
    save_payload(out_file, payload)
    if partial_file.exists():
        partial_file.unlink()

    print("\nSummary", flush=True)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False), flush=True)
    print(f"saved: {out_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
