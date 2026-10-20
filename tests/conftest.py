"""Mô hình giả + kho nhỏ để test logic mà không cần GPU hay tải mô hình."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from text_summarization.rag.store import Hit


class FakeModels:
    """Đếm token = đếm từ; embed = vector one-hot theo từ khoá; chat/summarize trả chuỗi dễ kiểm tra."""
    def __init__(self, vocab=None, chat_reply='ĐÁP'):
        self.vocab = vocab or []
        self.chat_reply = chat_reply
        self.calls = []

    def embed_query(self, text):
        return np.array([1.0 if w in text else 0.0 for w in self.vocab] + [1e-3], dtype='float32')

    def rerank(self, query, texts):
        return [sum(w in t for w in query.split()) for t in texts]

    def summarize(self, text):
        return 'TÓM TẮT: ' + text[:20]

    def count_tokens(self, text):
        return len(text.split())

    def clip(self, text, max_tokens):
        return ' '.join(text.split()[:max_tokens])

    def answer(self, question, context, max_new_tokens=256):
        from text_summarization.rag.llms import QA_SYSTEM, QA_USER
        return self.chat(QA_SYSTEM, QA_USER.format(context=context, question=question))

    def chat(self, system, user, max_new_tokens=256, repetition_penalty=1.05):
        self.calls.append(user)
        return self.chat_reply() if callable(self.chat_reply) else self.chat_reply


class FakeStore:
    mode = 'hybrid'

    def __init__(self, hits):
        self._hits = hits
        self.chunks = []

    def search(self, vec, k, text=None, mode=None):
        return self._hits[:k]

    def __len__(self):
        return len(self._hits)


@pytest.fixture
def articles():
    return {
        '1': {'title': 'Bản án Hồ Xuân Huy', 'abstract': 'Huy lừa đảo', 'article': ' '.join(['huy'] * 50)},
        '2': {'title': 'Lừa đảo qua điện thoại', 'abstract': 'mất 4,5 tỷ', 'article': ' '.join(['đt'] * 50)},
        '3': {'title': 'Nữ quái cho thuê căn hộ', 'abstract': '110 triệu', 'article': ' '.join(['ch'] * 50)},
    }


def hit(guid, text='x', score=0.0):
    return Hit(guid=guid, title=f'T{guid}', text=text, score=score, chunk_idx=0)
