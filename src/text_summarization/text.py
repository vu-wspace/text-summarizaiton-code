"""Xử lý văn bản: chuẩn hoá VietNews, làm sạch input terminal, so khớp đáp án khi đánh giá."""
import re
import sys
import unicodedata

# Không dùng raw string: Python đổi \uXXXX thành ký tự thật -> chạy được với cả `re` lẫn RE2 (pandas 3).
INVISIBLE = '[­​-‏⁠﻿]'
ABBR = r'TP|Tp|BS|TS|ThS|PGS|GS|TT|Jr|[A-ZĐ][a-zđ]?'
VN_LOWER = ('aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệfghiìíỉĩịjklmn'
            'oòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvwxyỳýỷỹỵz')
LOWER_CLS = f'[{VN_LOWER}]'

PLATE_RE = re.compile(
    r'\b(\d{2})\s*'                   # mã tỉnh: 77
    r'([A-Z]{1,2}\d?)\s*'             # sê-ri: AA
    r'-\s*'                           # gạch nối
    r'(\d{3}(?:\s*\.\s*)?\d{1,2})\b'  # số: 12345 | 123.45 | 123 . 45
)


def _fix_plate(m):
    num = re.sub(r'\s+', '', m.group(3))
    return f'{m.group(1)}{m.group(2)}-{num}'


def _join_parts(parts):
    return ' '.join(p if re.search(r'[.!?…"\')\]]$', p) else p + ' .' for p in parts)


def normalize_text(text, drop_caption=True):
    """Chuẩn hoá bài VietNews. Dùng CÙNG hàm này lúc indexing và lúc query."""
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFC', text)    # luôn đầu tiên
    text = re.sub(INVISIBLE, '', text)           # ký tự vô hình
    text = re.sub(r'[̀-ͯ]', '', text)  # dấu tổ hợp sót sau NFC
    text = text.replace('`', "'")

    if drop_caption:                             # chú thích ảnh nằm sau 2+ khoảng trắng
        parts = re.split(r'\s{2,}', text.strip())
        text = _join_parts(parts[:-1]) if len(parts) > 1 else parts[0]

    text = text.replace('_', ' ')                           # bỏ tách từ
    text = PLATE_RE.sub(_fix_plate, text)                   # biển số
    text = re.sub(r'\bv\s*\.\s*v\s*\.?', 'v.v.', text)      # v.v.
    text = re.sub(rf'\b({ABBR})\s+\.(?=\s)', r'\1.', text)  # T . / Th . -> T. / Th.
    text = re.sub(rf'\s\.\s(?={LOWER_CLS})', '. ', text)    # ' . ' + chữ thường
    text = re.sub(r'"\s*(.*?)\s*"', r'"\1"', text)          # nháy kép
    text = re.sub(r'\s+/\s+', '/', text)                    # dấu /
    return re.sub(r'\s+', ' ', text).strip()


def normalize_query(q):
    """Câu hỏi người dùng: không bỏ caption, KHÔNG lower() (giữ chữ hoa cho tên riêng)."""
    return normalize_text(q, drop_caption=False)


def clean_input(text):
    """Terminal gửi byte không phải UTF-8 (CMD mặc định) -> Python giữ ký tự 'surrogate' mà tokenizer
    (Rust) không nhận -> TypeError. Thay byte hỏng bằng '\\ufffd' để không sập, và nhắc người dùng."""
    fixed = text.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')
    if '�' in fixed:
        print('  ! Terminal không gửi UTF-8, tiếng Việt bị lỗi. Trên CMD chạy "chcp 65001" trước,'
              ' hoặc dùng Windows Terminal / PowerShell.', file=sys.stderr)
    return fixed


def norm_answer(s):
    """Chuẩn hoá để so khớp đáp án khi đánh giá (bỏ qua khác biệt trình bày, giữ nội dung):
    '10-11' ~ '10/11',  'Gloucestershire ( Anh Quốc )' ~ 'Gloucestershire (Anh Quốc)',  '"Minh Anh"' ~ 'Minh Anh',
    '14,43 %' ~ '14,43%',  '25/11.' ~ '25/11'."""
    s = unicodedata.normalize('NFC', str(s)).lower()
    s = re.sub(r'(\d)\s*[-–/]\s*(?=\d)', r'\1 / ', s)            # ngày tháng: 10-11, 10/11 -> '10 / 11' (so theo từ)
    s = re.sub(r'(\d)\s+%', r'\1%', s)                            # 14,43 % -> 14,43%
    s = re.sub(r'[()\[\]{}"“”‘’\'«»!?;:…]', ' ', s)              # bỏ ngoặc, nháy, dấu câu cuối
    s = re.sub(r'[.,](?!\d)|(?<!\d)[.,]', ' ', s)                 # bỏ . , trừ khi nằm GIỮA hai số (3,2 tỷ)
    return re.sub(r'\s+', ' ', s).strip()


def contains_answer(prediction, gold):
    """Đáp án chuẩn xuất hiện trọn vẹn (theo ranh giới từ) trong câu trả lời: '5' khớp 'bé gái 5 tuổi'
    nhưng KHÔNG khớp '15 tuổi'."""
    g = norm_answer(gold)
    return bool(g) and f' {g} ' in f' {norm_answer(prediction)} '
