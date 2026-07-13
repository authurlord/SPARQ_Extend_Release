"""Shared HybridQA evaluation metrics and BERTScore utilities."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_BERT_SCORER_CACHE: Dict[Tuple[str, int, str], Any] = {}


def normalize_answer(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_precision_recall_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    common: Dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def get_bert_scorer(model_path: Path, batch_size: int, device: str) -> Any:
    from transformers import AutoConfig

    key = (str(model_path), batch_size, device)
    if key not in _BERT_SCORER_CACHE:
        from bert_score import BERTScorer

        num_layers = None
        try:
            cfg = AutoConfig.from_pretrained(str(model_path))
            num_layers = int(getattr(cfg, "num_hidden_layers"))
        except Exception:
            num_layers = None
        _BERT_SCORER_CACHE[key] = BERTScorer(
            model_type=str(model_path),
            num_layers=num_layers,
            batch_size=batch_size,
            lang="en",
            idf=False,
            rescale_with_baseline=False,
            device=device,
        )
    return _BERT_SCORER_CACHE[key]


def compute_bertscore_f1(
    results: Sequence[Dict[str, Any]],
    model_path: Path,
    batch_size: int,
    device: str,
) -> Optional[float]:
    if not results:
        return None
    try:
        scorer = get_bert_scorer(model_path, batch_size, device)
        preds = [str(r.get("pred_answer", "") or "") for r in results]
        refs = [str(r.get("gold_answer", "") or "") for r in results]
        _, _, f1 = scorer.score(preds, refs, batch_size=batch_size)
        return round(float(f1.mean().item()) * 100, 2)
    except ModuleNotFoundError as exc:
        print(f"[bertscore] skipped: missing dependency ({exc})", flush=True)
        return None
    except Exception as exc:
        print(f"[bertscore] failed: {exc}", flush=True)
        return None


def parse_answer(text: str) -> str:
    text = (text or "").strip()
    # 1. SPARQ-style `therefore, the answer is: "X"` (priority over generic Answer: prefix)
    sparq_pat = re.compile(
        r"(?:therefore|thus)[,:]?\s*the\s+answer\s+is\s*:?\s*(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    m = sparq_pat.search(text)
    if m:
        rest = m.group(1).strip()
        quoted = re.findall(r'["\']([^"\']+)["\']', rest)
        if quoted:
            return quoted[0].strip().strip('".\'')
        first = rest.split("\n")[0].strip().strip('".\' ')
        if 0 < len(first) < 200:
            return first
    # 2. Look for explicit Answer: / Final Answer: tag
    for pattern in [
        r"(?im)^\s*Final\s+Answers?\s*:\s*(.+)$",
        r"(?im)^\s*Answers?\s*:\s*(.+)$",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if "\n" in candidate:
                candidate = next((line.strip() for line in candidate.splitlines() if line.strip()), "")
            return candidate
    # 3. (legacy) SPARQ-style fallback
    sparq_pat = re.compile(
        r"(?:therefore|thus)[,:]?\s*the\s+answer\s+is\s*:?\s*(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    m = sparq_pat.search(text)
    if m:
        rest = m.group(1).strip()
        # Extract first quoted span (handles `"X", "Y"` lists)
        quoted = re.findall(r'["\']([^"\']+)["\']', rest)
        if quoted:
            return quoted[0].strip().strip('".\'')
        # Otherwise first line, stripped
        first = rest.split("\n")[0].strip().strip('".\' ')
        if 0 < len(first) < 200:
            return first
    # 3. Look for plain inline "the answer is X" / "Therefore, X"
    inline_patterns = [
        r"the\s+answer\s+is\s*['\"]?([^.\n\"]+?)['\"]?[.\n]",
        r"the\s+answer\s+is\s*['\"]?([^.\n\"]+?)['\"]?\Z",
        r"Therefore[,:\s]+(?:the\s+answer\s+is\s+)?['\"]?([^.\n\"]+?)['\"]?[.\n]",
        r"final\s+answer[^\w]*['\"]?([^.\n\"]+?)['\"]?[.\n]",
    ]
    for pattern in inline_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip().strip('"\' .')
            cand_lower = candidate.lower()
            if cand_lower.startswith(("the answer is", "therefore", "final answer")):
                continue
            if 0 < len(candidate) < 200:
                return candidate
    # 3. Fallback: when output is just reasoning, take the LAST non-empty short line
    #    (last meaningful token sequence; reasoning blobs usually end with the answer)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        # Drop "Reasoning:" prefix that may sneak into a one-line response
        last = re.sub(r"^(Reasoning|Step\s*\d*)\s*:\s*", "", last, flags=re.IGNORECASE)
        if 0 < len(last) < 200:
            return last
    return re.sub(r"^Answers?\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def tokenized(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def table_to_rows(table: Dict[str, Any]) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    header = table["header"]
    cells = table["data"]
    n_cols = len(header)
    rows = []
    for i in range(0, len(cells), n_cols):
        row = cells[i : i + n_cols]
        if len(row) == n_cols:
            rows.append(row)
    return header, rows
