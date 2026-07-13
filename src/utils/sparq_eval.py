"""Unified SPARQ-style EM/F1 evaluation for all journal datasets.

Wraps /home/yanmy/SPARQ utils.evaluator.Evaluator with a single
`sparq_em_f1(pred, gold_list, question="")` entry point.
"""
from __future__ import annotations
import os, sys
from contextlib import contextmanager
from typing import List


@contextmanager
def _sparq_path():
    saved_cwd = os.getcwd()
    os.chdir("/home/yanmy/SPARQ")
    sys.path.insert(0, "/home/yanmy/SPARQ")
    for k in [k for k in sys.modules if k.startswith("utils") and "hybridqa" not in k]:
        del sys.modules[k]
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        if "/home/yanmy/SPARQ" in sys.path:
            sys.path.remove("/home/yanmy/SPARQ")
        for k in [k for k in sys.modules if k.startswith("utils") and "hybridqa" not in k]:
            del sys.modules[k]


_EVALUATOR = None


def _get_evaluator():
    global _EVALUATOR
    if _EVALUATOR is not None:
        return _EVALUATOR
    with _sparq_path():
        from utils.evaluator import Evaluator
        _EVALUATOR = Evaluator()
    return _EVALUATOR


def sparq_em(pred: str, gold_or_list, question: str = "") -> int:
    if not pred or not gold_or_list:
        return 0
    golds = gold_or_list if isinstance(gold_or_list, (list, tuple)) else [gold_or_list]
    ev = _get_evaluator()
    for g in golds:
        if not g:
            continue
        try:
            if ev.eval_ex_match(pred, g, allow_semantic=True, question=question or ""):
                return 1
        except Exception:
            if str(pred).strip().lower() == str(g).strip().lower():
                return 1
    return 0


def _tokens(s: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z0-9]+", str(s).lower())


def sparq_token_f1(pred: str, gold_or_list) -> float:
    if not pred or not gold_or_list:
        return 0.0
    golds = gold_or_list if isinstance(gold_or_list, (list, tuple)) else [gold_or_list]
    best = 0.0
    p_tok = _tokens(pred)
    if not p_tok:
        return 0.0
    p_set = set(p_tok)
    for g in golds:
        g_tok = _tokens(g)
        if not g_tok:
            continue
        g_set = set(g_tok)
        common = p_set & g_set
        if not common:
            continue
        prec = len(common) / len(p_set)
        rec = len(common) / len(g_set)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best:
            best = f1
    return best


def sparq_em_f1(pred: str, gold_or_list, question: str = ""):
    return sparq_em(pred, gold_or_list, question), sparq_token_f1(pred, gold_or_list)
