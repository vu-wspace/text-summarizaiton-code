"""Vector thưa BM25 cho nhánh từ khoá của hybrid search.

Tiếng Việt viết theo âm tiết ('Hồ Xuân Huy', 'bất động sản') nên lấy cả unigram lẫn bigram âm tiết:
bigram giữ được từ ghép / tên riêng mà unigram làm loãng. Mỗi term băm crc32 -> chỉ số uint32 của Qdrant.

Phần TF (bão hoà k1, chuẩn hoá độ dài b) tính ở đây cho chunk; phần IDF do Qdrant tính trên toàn kho
(Modifier.IDF), nên thêm chunk mới không phải tính lại vector cũ. Câu hỏi: mỗi term trọng số 1.
"""
import re
import unicodedata
import zlib
from collections import Counter

K1 = 1.2
B = 0.75
WORD = re.compile(r'\w+')


def terms(text):
    words = WORD.findall(unicodedata.normalize('NFC', text).lower())
    return words + [f'{a}_{b}' for a, b in zip(words, words[1:])]


def term_id(term):
    return zlib.crc32(term.encode('utf-8'))


def _ids(text):
    # băm có thể trùng: gộp đếm theo chỉ số (Qdrant không nhận chỉ số lặp)
    return Counter(term_id(t) for t in terms(text))


def doc_vector(text, avg_len, k1=K1, b=B):
    """-> (indices, values): trọng số TF kiểu BM25 của một chunk."""
    tf = _ids(text)
    n = sum(tf.values())
    norm = k1 * (1 - b + b * n / avg_len) if avg_len else k1
    return list(tf), [c * (k1 + 1) / (c + norm) for c in tf.values()]


def query_vector(text):
    ids = list(_ids(text))
    return ids, [1.0] * len(ids)


def avg_length(texts):
    lens = [len(terms(t)) for t in texts]
    return sum(lens) / len(lens) if lens else 0.0
