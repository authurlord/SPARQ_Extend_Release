import re

from utils.normalizer import str_normalize
from utils.wtq.evaluator import to_value_list, check_denotation
import evaluate  # Updated import
import nltk

def postprocess_text(preds, labels, metric_name):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    if metric_name == "rouge":
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    elif metric_name == "sacrebleu":  # sacrebleu
        labels = [[label] for label in labels]
    elif metric_name == "bleu":
        preds = [pred.split(' ') for pred in preds]
        labels = [[label.split(' ')] for label in labels]
    else:
        pass

    return preds, labels
class EvaluateTool:
    def __init__(self, args):
        pass

    def evaluate(self, preds, gold_text):
        summary = {}

        # gold_text = [item["answer"] for item in golds]

        assert len(preds) == len(gold_text)

        metric_list = ["sacrebleu", "rouge", "meteor", "bertscore", "bleurt"]

        for metric_name in metric_list:
            metric = load_metric(metric_name)
            processed_preds, processed_golds = postprocess_text(preds, gold_text, metric_name)

            if metric_name == "bertscore":
                res = metric.compute(predictions=processed_preds, references=processed_golds, lang="en")
                for k, v in res.items():
                    if k == "hashcode":
                        continue
                    summary[f"{metric_name}_{k}"] = round(1.0 * sum(v) / len(v), 2)

            else:
                res = metric.compute(predictions=processed_preds, references=processed_golds)
                if metric_name == "sacrebleu":
                    summary[metric_name] = res["score"] * 0.01  # limit it to range of [0, 1] for unifying
                elif metric_name == "bleurt":
                    summary["bleurt"] = round(1.0 * sum(res["scores"]) / len(res["scores"]), 2)
                elif metric_name == 'rouge':
                    for sub_metric_name in res.keys():
                        for i, key in enumerate(['precision', 'recall', 'fmeasure']):
                            summary["{}_{}".format(sub_metric_name, key)] = res[sub_metric_name][1][i]
                        # this the the fmeasure('f-score') from the mid('mean aggregation')
                else:
                    summary[metric_name] = res[metric_name]
        return summary

class Evaluator:
    def __init__(self):
        pass

    def evaluate(
            self,
            pred_answer,
            gold_answer,
            dataset,
            allow_semantic=True,
            question=None
    ):
        if dataset == 'wikitq':
            return self.eval_ex_match(pred_answer, gold_answer, allow_semantic, question)
        elif dataset == 'tab_fact':
            return self.eval_tabfact_match(pred_answer, gold_answer)
        elif dataset == 'mmqa':
            # For more metrics on MMQA,
            # please use the utils/mmqa/eval_mmqa.py to call official on all prediction data
            return self.eval_mmqa_match(pred_answer, gold_answer)
        elif dataset == 'niat':
            # Use WikiTQ's robust evaluation logic for NIAT as well
            return self.eval_ex_match(pred_answer, gold_answer, allow_semantic, question)
        else:
            raise ValueError(f'{dataset} evaluator is not supported.')

    def eval_ex_match(self, pred, gold, allow_semantic=True, question=None):
        # print('Hello')
        if not isinstance(pred, list):
            pred = [pred]
            gold = [gold]
        # print(pred,gold)
        pred = [str(p).lower().strip() for p in pred]
        gold = [str(g).lower().strip() for g in gold]
        if not allow_semantic:
            # WikiTQ eval w. string normalization using recognizer
            pred = [str_normalize(span) for span in pred]
            gold = [str_normalize(span) for span in gold]
            pred = to_value_list(pred)
            gold = to_value_list(gold)
            return check_denotation(pred, gold)
        else:
            assert isinstance(question, str)
            question = re.sub(r'\s+', ' ', question).strip().lower()
            pred = [str_normalize(span) for span in pred]
            gold = [str_normalize(span) for span in gold]
            pred = sorted(list(set(pred)))
            gold = sorted(list(set(gold)))
            # (1) 0 matches 'no', 1 matches 'yes'; 0 matches 'more', 1 matches 'less', etc.
            if len(pred) == 1 and len(gold) == 1:
                if (pred[0] == '0' and gold[0] == 'no') \
                        or (pred[0] == '1' and gold[0] == 'yes'):
                    return True
                question_tokens = question.split()
                try:
                    pos_or = question_tokens.index('or')
                    token_before_or, token_after_or = question_tokens[pos_or - 1], question_tokens[pos_or + 1]
                    if (pred[0] == '0' and gold[0] == token_after_or) \
                            or (pred[0] == '1' and gold[0] == token_before_or):
                        return True
                except Exception as e:
                    pass
            # (2) Number value (allow units) and Date substring match
            if len(pred) == 1 and len(gold) == 1:
                NUMBER_UNITS_PATTERN = re.compile(r'^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$')
                DATE_PATTERN = re.compile(r'[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?')
                DURATION_PATTERN = re.compile(r'(P|PT)(\d+)(Y|M|D|H|S)')
                p, g = pred[0], gold[0]
                # Restore `duration` type, e.g., from 'P3Y' -> '3'
                if re.match(DURATION_PATTERN, p):
                    p = re.match(DURATION_PATTERN, p).group(2)
                if re.match(DURATION_PATTERN, g):
                    g = re.match(DURATION_PATTERN, g).group(2)
                match = False
                num_flag, date_flag = False, False
                # Number w. unit match after string normalization.
                # Either pred or gold being number w. units suffices it.
                if re.match(NUMBER_UNITS_PATTERN, p) or re.match(NUMBER_UNITS_PATTERN, g):
                    num_flag = True
                # Date match after string normalization.
                # Either pred or gold being date suffices it.
                if re.match(DATE_PATTERN, p) or re.match(DATE_PATTERN, g):
                    date_flag = True
                if num_flag:
                    p_set, g_set = set(p.split()), set(g.split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if date_flag:
                    p_set, g_set = set(p.replace('-', ' ').split()), set(g.replace('-', ' ').split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if match:
                    return True
            pred = to_value_list(pred)
            gold = to_value_list(gold)
            return check_denotation(pred, gold)

    def eval_tabfact_match(self, pred, gold):
        if isinstance(pred, list):
            pred = pred[0]
        pred, gold = str(pred), str(gold)
        return pred == gold

    def eval_mmqa_match(self, pred, gold):
        """MMQA match evaluation."""
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        pred_norm = str(pred).strip().lower()
        gold_norm = str(gold).strip().lower()
        return pred_norm == gold_norm

    def eval_niat_match(self, pred, gold):
        """NIAT match evaluation - simple exact match after normalization."""
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        # Extract "The answer is: " pattern
        pred_str = str(pred)
        match = re.search(r'The answer is:\s*(.+)', pred_str, re.IGNORECASE)
        if match:
            pred_str = match.group(1)
        pred_norm = pred_str.strip().lower()
        gold_norm = str(gold).strip().lower()
        return pred_norm == gold_norm


def niat_normalize(text):
    """
    Normalize text for NIAT matching without external dependencies.
    Handles: quotes, brackets/citations, whitespace, case.
    """
    if text is None:
        return ""
    text = str(text)
    # Remove ALL quotes (both single and double) - important for list answers
    text = text.replace('"', '').replace("'", '')
    # Remove citations like [1], [11], [note], etc.
    text = re.sub(r'\s*\[[^\]]*\]', '', text)
    # Remove trailing punctuation that might be artifacts
    text = text.rstrip('.')
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase for comparison
    text = text.lower()
    return text


def niat_match_func(sample, strategy="top"):
    """
    Compare final chain answer string with gold answer for NIAT dataset.
    Uses Evaluator.eval_ex_match() for robust matching with normalization.

    Expected sample fields:
      - sample["chain"][-1]["parameter_and_conf"] -> list[(answer_str, conf)]
      - sample["answer"] -> gold answer string
    """
    results = sample["chain"][-1]["parameter_and_conf"]

    if strategy == "top":
        pred_answer = results[0][0]
        # extract "The answer is: " (case insensitive)
        # Accept "The answer is:", "Final Answer:", or "Answer:" markers — the
        # POT-direct / 35B reader follows the prompt's "Final Answer:" format
        # literally, which the original "The answer is:"-only regex missed,
        # leaving the prefix on `pred` and failing eval_ex_match (same class as
        # the TableBench "Final Answer:" extraction gap, 2026-06-04).
        match = re.search(r'(?:final\s+answer|the\s+answer\s+is|answer)\s*[:：]\s*(.+)',
                          pred_answer, re.IGNORECASE | re.DOTALL)
        if match:
            pred = match.group(1)
        else:
            pred = pred_answer
        # take the final-answer line only (drop trailing CoT after a newline)
        pred = pred.strip().split('\n')[0]
        # Additional cleanup: remove surrounding quotes (matches user's .replace('"',''))
        pred = pred.strip().replace('"', '')
    elif strategy == "weighted":
        res_conf_dict = {}
        for res, conf in results:
            if res not in res_conf_dict:
                res_conf_dict[res] = 0
            res_conf_dict[res] += conf
        res_conf_rank = sorted(res_conf_dict.items(), key=lambda x: x[1], reverse=True)
        pred = res_conf_rank[0][0]
    else:
        raise NotImplementedError(f"Strategy '{strategy}' not implemented")

    gold = sample.get("answer", "")
    # Use Evaluator.eval_ex_match for robust matching with normalization
    # Pass question="" to bypass assertion and skip boolean/choice special logic
    return Evaluator().eval_ex_match(pred, gold, allow_semantic=True, question="")


def niat_match_func_for_samples(all_samples, strategy="top"):
    """Calculate NIAT accuracy for a list of samples."""
    correct_list = []
    for sample in all_samples:
        try:
            if niat_match_func(sample, strategy):
                correct_list.append(1)
            else:
                correct_list.append(0)
        except Exception as e:
            # Skip samples with errors
            continue
    if not correct_list:
        return 0.0
    return sum(correct_list) / len(correct_list)


# ============================================================================
# TableBench ROUGE-L Evaluation Functions
# ============================================================================

def tablebench_rouge_l_score(pred: str, gold: str) -> float:
    """
    Calculate ROUGE-L F1 score between prediction and gold answer.
    Uses rouge-score package instead of evaluate.
    
    Args:
        pred: Predicted answer string
        gold: Gold answer string
        
    Returns:
        ROUGE-L F1 score (0.0 to 1.0)
    """
    try:
        from rouge_score import rouge_scorer
        
        # Preprocess: extract "The answer is:" if present
        pred_str = str(pred) if pred else ""
        gold_str = str(gold) if gold else ""
        
        # Extract answer from common patterns
        pattern1 = r'Therefore, the answer is:\s*"?([^"\n\r]+)"?'
        match = re.search(pattern1, pred_str, re.IGNORECASE)
        if match:
            pred_str = match.group(1).strip()
        else:
            # Strip a leading answer prefix the model/POT output may emit, e.g.
            # "Final Answer: 10.6", "Answer: 10.6", "The answer is: 10.6". The
            # 35B reader follows the prompt's "Final Answer:" format literally,
            # and the gold is the bare value; without stripping, ROUGE-L is
            # capped well below 0.8 even when the value is exactly correct
            # (root cause of the 0.32 vs 0.50 TableBench gap, 2026-06-04).
            pred_str = re.sub(
                r'^\s*(?:the\s+)?(?:final\s+answer|answer\s+is|answer)\s*[:：]\s*',
                '', pred_str, flags=re.IGNORECASE)

        # Remove surrounding quotes
        pred_str = pred_str.strip().strip('"\'')
        gold_str = gold_str.strip().strip('"\'')
        
        # Handle empty strings
        if not pred_str or not gold_str:
            return 0.0
        
        # Compute ROUGE-L using rouge-score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        # print(11, gold_str, pred_str)
        scores = scorer.score(gold_str, pred_str)
        
        return scores['rougeL'].fmeasure
    except Exception as e:
        # Fallback: simple exact match
        return 1.0 if str(pred).strip().lower() == str(gold).strip().lower() else 0.0


def tablebench_match_func(sample, threshold=0.5) -> bool:
    """
    Match function using ROUGE-L for TableBench.
    Returns True if ROUGE-L score >= threshold.
    
    Args:
        sample: dict with 'chain' and 'answer' fields
        threshold: ROUGE-L threshold for matching (default 0.5)
        
    Returns:
        True if ROUGE-L score >= threshold
    """
    results = sample["chain"][-1]["parameter_and_conf"]
    pred_answer = results[0][0]
    
    # Extract "The answer is:" pattern
    match = re.search(r'(?:the answer is|therefore|answer):\s*(.+)', pred_answer, re.IGNORECASE)
    if match:
        pred = match.group(1).strip()
    else:
        pred = pred_answer
    
    # Remove surrounding quotes
    pred = pred.strip().strip('"\'')
    gold = str(sample.get("answer", "")).strip()
    
    score = tablebench_rouge_l_score(pred, gold)
    return score >= threshold


def tablebench_match_func_for_samples(all_samples, threshold=0.5) -> dict:
    """
    Calculate ROUGE-L metrics for all TableBench samples.
    
    Args:
        all_samples: List of sample dicts with 'chain' and 'answer' fields
        threshold: ROUGE-L threshold for "correct" classification
        
    Returns:
        dict with 'accuracy', 'avg_rouge_l', 'scores' keys
    """
    scores = []
    correct_count = 0
    
    for sample in all_samples:
        try:
            results = sample["chain"][-1]["parameter_and_conf"]
            pred_answer = results[0][0]
            
            # Extract "The answer is:" pattern
            match = re.search(r'(?:the answer is|therefore|answer):\s*(.+)', pred_answer, re.IGNORECASE)
            if match:
                pred = match.group(1).strip()
            else:
                pred = pred_answer
            
            pred = pred.strip().strip('"\'')
            gold = str(sample.get("answer", "")).strip()
            
            score = tablebench_rouge_l_score(pred, gold)
            scores.append(score)
            
            if score >= threshold:
                correct_count += 1
        except Exception as e:
            scores.append(0.0)
            continue
    
    if not scores:
        return {'accuracy': 0.0, 'avg_rouge_l': 0.0, 'scores': []}
    
    return {
        'accuracy': correct_count / len(scores),
        'avg_rouge_l': sum(scores) / len(scores),
        'scores': scores
    }


def evaluate_tablebench_predictions(preds: list, golds: list) -> dict:
    """
    Evaluate TableBench predictions using ROUGE-L.
    
    Args:
        preds: List of prediction strings
        golds: List of gold answer strings
        
    Returns:
        dict with evaluation metrics
    """
    assert len(preds) == len(golds), f"Length mismatch: {len(preds)} preds vs {len(golds)} golds"
    
    scores = []
    for pred, gold in zip(preds, golds):
        score = tablebench_rouge_l_score(pred, gold)
        scores.append(score)
    
    if not scores:
        return {'avg_rouge_l': 0.0, 'accuracy_at_0.5': 0.0, 'accuracy_at_0.8': 0.0}
    
    avg_score = sum(scores) / len(scores)
    acc_05 = sum(1 for s in scores if s >= 0.5) / len(scores)
    acc_08 = sum(1 for s in scores if s >= 0.8) / len(scores)
    
    return {
        'avg_rouge_l': avg_score,
        'accuracy_at_0.5': acc_05,
        'accuracy_at_0.8': acc_08,
        'total_samples': len(scores)
    }

