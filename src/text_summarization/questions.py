"""Sinh bộ câu hỏi đánh giá bằng LLM từ kho vector, và chấm RAG bằng chính bộ câu hỏi đó.

Sinh : chọn ngẫu nhiên N chunk (mỗi bài ≤ 1) -> Qwen đặt 1 câu hỏi + đáp án chép từ chunk -> lọc câu kém.
Chấm : với mỗi câu hỏi, chạy RAG như người dùng thật rồi đo
       hit@1    bài nguồn là bài đầu tiên trong ngữ cảnh đưa cho LLM
       hit@3    bài nguồn nằm trong ngữ cảnh (≤ 3 bài)
       contains câu trả lời chứa đáp án chuẩn (đã chuẩn hoá ngày tháng, ngoặc, dấu câu)
"""
import json
import random
import re
import sys

from .metrics import token_recall
from .text import contains_answer

GEN_SYSTEM = 'Bạn là người soạn câu hỏi kiểm tra đọc hiểu tin tức tiếng Việt. Chỉ trả về JSON.'
GEN_USER = (
    'ĐOẠN VĂN (trích từ bài báo "{title}"):\n{text}\n\n'
    'Hãy đặt MỘT câu hỏi dựa trên đoạn văn trên.\n'
    'YÊU CẦU:\n'
    '1. Câu hỏi phải tự hiểu được khi đứng một mình: nêu rõ ít nhất một tên riêng (người, nơi, tổ chức) '
    'hoặc sự kiện cụ thể. KHÔNG dùng "đoạn văn", "bài báo", "bài viết", "ông ấy", "vụ việc này".\n'
    '2. Đáp án ngắn (tối đa 15 từ), CHÉP NGUYÊN VĂN từ đoạn văn (tên, số liệu, ngày tháng...).\n'
    '3. Chỉ trả về đúng một dòng JSON: {{"question": "...", "answer": "..."}}'
)
BAD_PHRASES = re.compile(r'đoạn văn|bài báo|bài viết|văn bản|thông tin trên|vụ việc này', re.I)
YES_NO = re.compile(r'\b(có phải|phải không|đúng không|có đúng|hay không|có không)\b|\bkhông\s*\?$', re.I)


# ----------------------------------------------------------------- lọc (không cần mô hình)
def parse_json(text):
    """Lấy object JSON đầu tiên trong output LLM (LLM nhỏ hay kèm chữ thừa / ```json)."""
    m = re.search(r'\{.*?\}', text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    q, a = str(obj.get('question', '')).strip(), str(obj.get('answer', '')).strip()
    return (q, a) if q and a else None


def has_anchor(question):
    """Câu hỏi phải có 'mỏ neo' để tìm được bài: tên riêng (chữ hoa, không tính từ đầu câu) hoặc con số.
    Loại câu mơ hồ kiểu 'Tên người mẫu được đề cử cho giải thưởng?'."""
    words = question.split()
    return any(w[:1].isupper() for w in words[1:]) or bool(re.search(r'\d', question))


def check(question, answer, chunk_text):
    """-> lý do loại (str) hoặc None nếu đạt."""
    if len(question) < 15:
        return 'câu hỏi quá ngắn'
    if BAD_PHRASES.search(question):
        return 'câu hỏi phụ thuộc đoạn văn'
    if not has_anchor(question):
        return 'câu hỏi mơ hồ (không có tên riêng / con số)'
    if YES_NO.search(question):
        return 'câu hỏi có/không'
    if len(answer.split()) > 20:
        return 'đáp án quá dài'
    if not contains_answer(chunk_text, answer):
        return 'đáp án không có trong đoạn văn (LLM bịa)'
    if contains_answer(question, answer) or (len(answer.split()) >= 2 and token_recall(question, answer) >= 0.5):
        return 'đáp án lộ trong câu hỏi'
    return None


def pick_chunks(chunks, n, seed, min_chars=400):
    """Chỉ số chunk ngẫu nhiên: đủ dài, không phải chunk tiêu đề, mỗi bài tối đa 1 chunk."""
    idx = [i for i, c in enumerate(chunks)
           if len(c.page_content) >= min_chars and not c.metadata.get('is_header')]
    random.Random(seed).shuffle(idx)
    seen, out = set(), []
    for i in idx:
        g = str(chunks[i].metadata['guid'])
        if g not in seen:
            seen.add(g)
            out.append(i)
            if len(out) >= n:
                break
    return out


# ----------------------------------------------------------------- sinh
def generate(store, articles, models, n=10, seed=42, oversample=4):
    rows, rejected = [], 0
    for i in pick_chunks(store.chunks, n * oversample, seed):   # lấy dư để bù câu bị loại
        h = store.hit(i)
        title = articles.get(h.guid, {}).get('title', h.title)
        qa = parse_json(models.chat(GEN_SYSTEM, GEN_USER.format(title=title, text=h.text),
                                   max_new_tokens=160, repetition_penalty=1.0))   # 1.0: cần chép nguyên văn đáp án
        reason = 'không đọc được JSON' if qa is None else check(qa[0], qa[1], h.text)
        if reason:
            rejected += 1
            print(f'  ✗ chunk {i} (bài {h.guid}): {reason}', file=sys.stderr)
            continue
        q, a = qa
        q = q if q.endswith('?') else q + '?'
        rows.append({'id': len(rows) + 1, 'guid': h.guid, 'title': title, 'chunk_idx': i,
                     'question': q, 'answer': a, 'context': h.text})
        print(f'  ✓ [{len(rows)}/{n}] {q}  →  {a}')
        if len(rows) >= n:
            break
    print(f'Đạt {len(rows)}/{n} câu, loại {rejected}.')
    return rows


# ----------------------------------------------------------------- chấm
def evaluate(pipeline, rows):
    res = []
    for r in rows:
        reply = pipeline.answer(r['question'])        # đúng luồng người dùng: retriever -> re-rank -> LLM
        got = reply.sources
        res.append({**r, 'pred': reply.text, 'retrieved': got,
                    'hit@1': bool(got) and got[0] == r['guid'],
                    'hit@3': r['guid'] in got[:3],
                    'contains': contains_answer(reply.text, r['answer'])})
        mark = '✓' if res[-1]['contains'] else '✗'
        print(f"  {mark} [{r['id']}] {r['question']}\n      chuẩn: {r['answer']} | RAG: {reply.text}")
    print(summary(res))
    return res


def rescore(rows):
    """Chấm lại 'contains' của file eval cũ bằng cách so khớp mới (không cần chạy lại mô hình)."""
    return [{**r, 'contains': contains_answer(r['pred'], r['answer'])} for r in rows]


def summary(res):
    n = len(res)
    if not n:
        return 'Không có câu nào.'
    rate = lambda k: sum(bool(x[k]) for x in res) / n  # noqa: E731
    return f"hit@1={rate('hit@1'):.2f}  hit@3={rate('hit@3'):.2f}  contains={rate('contains'):.2f}  (n={n})"
