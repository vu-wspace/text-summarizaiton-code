"""Router: phân loại yêu cầu bằng regex. Mặc định là hỏi đáp."""
import re
import unicodedata

from . import config as C

LIST_RE = re.compile(r'^\s*(hãy\s+)?(liệt\s*kê|trích\s*xuất)\b[\s:,-]*', re.I)
SUM_RE = re.compile(r'^\s*(hãy\s+)?(tóm\s*tắt|tóm\s*lược|tom\s*tat)\b[\s:,-]*', re.I)

LABELS = {'PERSON': 'Người', 'LOCATION': 'Địa điểm', 'ORGANIZATION': 'Tổ chức', 'MISCELLANEOUS': 'Khác'}
TYPE_WORDS = {
    'PERSON': ['người', 'nhân vật', 'tên'],
    'LOCATION': ['địa điểm', 'địa danh', 'nơi'],
    'ORGANIZATION': ['tổ chức', 'cơ quan', 'công ty'],
}
TYPE_STRIP = re.compile(
    r'^(các|những)?\s*(người|nhân vật|tên|địa điểm|địa danh|nơi|tổ chức|cơ quan|công ty|thực thể)'
    r'\s*(có|xuất hiện)?\s*(trong|của|ở)?\s*', re.I)

QA, LIST, SUMMARIZE_TOPIC, SUMMARIZE_TEXT = 'qa', 'list', 'summarize_topic', 'summarize_text'


def route(message, min_text_len=C.SUM_MIN_TEXT):
    """-> (intent, payload)
    'Liệt kê <chủ đề>'          -> list
    'Tóm tắt <chủ đề ngắn>'     -> summarize_topic   (tìm bài rồi tóm tắt)
    'Tóm tắt: <văn bản dài>'    -> summarize_text    (tóm tắt chính văn bản dán vào)
    còn lại                     -> qa"""
    msg = unicodedata.normalize('NFC', message).strip()
    m = LIST_RE.match(msg)
    if m:
        return LIST, msg[m.end():].strip()
    m = SUM_RE.match(msg)
    if m:
        rest = msg[m.end():].strip()
        return (SUMMARIZE_TEXT if len(rest) >= min_text_len else SUMMARIZE_TOPIC), rest
    return QA, msg


def wanted_types(payload):
    """'Liệt kê địa điểm trong ...' -> ['LOCATION']; không nêu loại -> None (lấy hết)."""
    p = payload.lower()
    return [t for t, ws in TYPE_WORDS.items() if any(w in p for w in ws)] or None


def strip_type_words(payload):
    """Bỏ từ chỉ loại ở đầu ('địa điểm trong ...') để câu tìm kiếm chỉ còn chủ đề."""
    return TYPE_STRIP.sub('', payload).strip() or payload


def format_entities(ents):
    if not ents:
        return 'Không tìm thấy thực thể nào trong bài.'
    return '\n'.join(f'  {LABELS.get(t, t)}: {", ".join(names)}' for t, names in ents.items())
