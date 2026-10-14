"""Chỉ số chấm câu trả lời hỏi đáp (đáp án chuẩn ngắn, câu trả lời của LLM có thể dài).

contains   đáp án chuẩn (đã chuẩn hoá) nằm trong câu trả lời            -> chỉ số chính, chịu được trả lời dài
recall     tỉ lệ token của đáp án chuẩn có mặt trong câu trả lời       -> điểm từng phần ('Vũ Ngọc Long' vs 'Long')
f1         token F1 kiểu SQuAD                                         -> phạt trả lời dài dòng, để so độ gọn
em         khớp chính xác sau chuẩn hoá                                -> rất khắt khe với LLM
abstain    LLM từ chối / nói không biết                                -> câu đã có ngữ cảnh đúng mà từ chối = mất điểm
"""
from collections import Counter

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
