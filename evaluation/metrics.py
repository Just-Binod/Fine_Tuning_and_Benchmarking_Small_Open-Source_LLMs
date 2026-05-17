"""
metrics.py
----------
All evaluation metrics in one place.
BLEU for translation, ROUGE-L for summarization, F1+EM for QA.

Used by both baseline_eval.py and benchmark.py — identical measurement
ensures fair before/after comparison.

Usage:
    from evaluation.metrics import compute_metrics
    scores = compute_metrics("translation", predictions, references)
"""

import re
import string
from collections import Counter


# ── BLEU (Translation) ────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return text.strip().split()


def ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(predictions: list[str], references: list[str], max_n: int = 4) -> dict:
    import math
    clipped_counts = Counter()
    total_counts   = Counter()
    pred_len_total = 0
    ref_len_total  = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens  = tokenize(ref)
        pred_len_total += len(pred_tokens)
        ref_len_total  += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ng = ngrams(pred_tokens, n)
            ref_ng  = ngrams(ref_tokens,  n)
            for gram, count in pred_ng.items():
                clipped_counts[n] += min(count, ref_ng.get(gram, 0))
                total_counts[n]   += count

    if pred_len_total == 0:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n+1)} | {"bleu": 0.0}

    bp = 1.0 if pred_len_total > ref_len_total else math.exp(
        1 - ref_len_total / pred_len_total
    )
    precisions = {
        n: (clipped_counts[n] / total_counts[n] if total_counts[n] else 0.0)
        for n in range(1, max_n + 1)
    }

    if any(p == 0 for p in precisions.values()):
        bleu = 0.0
    else:
        bleu = bp * math.exp(sum(math.log(p) for p in precisions.values()) / max_n)

    return {
        "bleu":   round(bleu * 100, 2),
        "bleu_1": round(precisions[1] * 100, 2),
        "bleu_2": round(precisions[2] * 100, 2),
        "bleu_3": round(precisions[3] * 100, 2),
        "bleu_4": round(precisions[4] * 100, 2),
    }


# ── ROUGE-L (Summarization) ──────────────────────────────────────────────────

def lcs_length(x: list, y: list) -> int:
    m, n  = len(x), len(y)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                table[i][j] = table[i-1][j-1] + 1
            else:
                table[i][j] = max(table[i-1][j], table[i][j-1])
    return table[m][n]


def rouge_l_score(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    lcs  = lcs_length(p, r)
    prec = lcs / len(p)
    rec  = lcs / len(r)
    return (2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0


def rouge_l_corpus(predictions: list[str], references: list[str]) -> dict:
    scores = [rouge_l_score(p, r) for p, r in zip(predictions, references)]
    avg = sum(scores) / max(len(scores), 1)
    return {"rouge_l": round(avg * 100, 2)}


# ── F1 + Exact Match (QA) ─────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()


def token_f1(pred: str, ref: str) -> float:
    p_toks = normalize(pred).split()
    r_toks = normalize(ref).split()
    common = sum((Counter(p_toks) & Counter(r_toks)).values())
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec  = common / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def qa_scores(predictions: list[str], references: list[str]) -> dict:
    f1 = [token_f1(p, r)                              for p, r in zip(predictions, references)]
    em = [float(normalize(p) == normalize(r))          for p, r in zip(predictions, references)]
    return {
        "f1":          round(sum(f1) / max(len(f1), 1) * 100, 2),
        "exact_match": round(sum(em) / max(len(em), 1) * 100, 2),
    }


# ── Unified interface ─────────────────────────────────────────────────────────

def compute_metrics(task: str, predictions: list[str], references: list[str]) -> dict:
    """
    Single entry point for all metrics.

    Args:
        task        : "translation" | "summarization" | "qa"
        predictions : model outputs
        references  : gold references

    Returns dict of scores (all 0-100 scale).
    """
    assert len(predictions) == len(references)

    if task == "translation":
        return bleu_score(predictions, references)
    elif task == "summarization":
        return rouge_l_corpus(predictions, references)
    elif task == "qa":
        return qa_scores(predictions, references)
    else:
        raise ValueError(f"Unknown task: {task!r}. Use: translation, summarization, qa")


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Testing metrics ───────────────────────────────────────\n")

    print("BLEU (translation):")
    s = compute_metrics("translation", ["the cat sat on the mat"], ["the cat sat on the mat"])
    print(f"  Perfect match → {s}")

    s = compute_metrics("translation", ["hello world"], ["hello there"])
    print(f"  Partial match → {s}")

    print("\nROUGE-L (summarization):")
    s = compute_metrics("summarization", ["नेपालमा नयाँ सरकार गठन भयो"], ["नेपालमा नयाँ सरकार बन्यो"])
    print(f"  Nepali test   → {s}")

    print("\nF1 + EM (qa):")
    s = compute_metrics("qa", ["काठमाडौं नेपालको राजधानी हो"], ["काठमाडौं"])
    print(f"  Partial match → {s}")

    s = compute_metrics("qa", ["काठमाडौं"], ["काठमाडौं"])
    print(f"  Exact match   → {s}")

    print("\n✓ All metrics working\n")