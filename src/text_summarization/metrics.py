"""Chỉ số chấm câu trả lời hỏi đáp (đáp án chuẩn ngắn, câu trả lời của LLM có thể dài).

contains   đáp án chuẩn (đã chuẩn hoá) nằm trong câu trả lời            -> chỉ số chính, chịu được trả lời dài
recall     tỉ lệ token của đáp án chuẩn có mặt trong câu trả lời       -> điểm từng phần ('Vũ Ngọc Long' vs 'Long')
f1         token F1 kiểu SQuAD                                         -> phạt trả lời dài dòng, để so độ gọn
em         khớp chính xác sau chuẩn hoá                                -> rất khắt khe với LLM
abstain    LLM từ chối / nói không biết                                -> câu đã có ngữ cảnh đúng mà từ chối = mất điểm

Kiểm định (các LLM trả lời CÙNG bộ câu hỏi): khoảng tin cậy bootstrap, McNemar chính xác từng cặp + hiệu chỉnh Holm.
"""
from collections import Counter
from math import comb

import numpy as np

from .llms import ABSTAIN_RE
from .text import contains_answer, norm_answer


def _tokens(s):
    return norm_answer(s).split()


def token_f1(pred, gold):
    p, g = _tokens(pred), _tokens(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def token_recall(pred, gold):
    p, g = _tokens(pred), _tokens(gold)
    return sum((Counter(p) & Counter(g)).values()) / len(g) if g else 0.0


def exact_match(pred, gold):
    return bool(norm_answer(gold)) and norm_answer(pred) == norm_answer(gold)


def abstained(pred):
    return bool(ABSTAIN_RE.search(pred or ''))


def score(pred, gold):
    return {'contains': contains_answer(pred, gold), 'recall': round(token_recall(pred, gold), 4),
            'f1': round(token_f1(pred, gold), 4), 'em': exact_match(pred, gold), 'abstain': abstained(pred),
            'words': len((pred or '').split())}


METRICS = ('contains', 'recall', 'f1', 'em', 'abstain', 'words')


def aggregate(rows, keys=METRICS):
    """Trung bình các chỉ số trên danh sách dict (bool tính như 0/1)."""
    n = len(rows)
    return {k: (sum(float(r[k]) for r in rows) / n if n else 0.0) for k in keys}


# ----------------------------------------------------------------- kiểm định
def bootstrap_ci(values, n_boot=10000, level=0.95, seed=0):
    """Khoảng tin cậy của trung bình (lấy mẫu lại theo câu hỏi)."""
    v = np.asarray(values, dtype=float)
    if not len(v):
        return 0.0, 0.0
    means = v[np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo = (1 - level) / 2
    return float(np.quantile(means, lo)), float(np.quantile(means, 1 - lo))


def mcnemar(a, b):
    """a, b: đúng/sai từng câu của 2 LLM -> (A đúng B sai, B đúng A sai, p hai phía, kiểm định nhị thức chính xác)."""
    n01 = sum(bool(x) and not y for x, y in zip(a, b))
    n10 = sum(bool(y) and not x for x, y in zip(a, b))
    n = n01 + n10
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p


def holm(pvals):
    """Hiệu chỉnh Holm-Bonferroni cho nhiều phép so sánh (giữ thứ tự đầu vào)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        out[i] = run
    return out
