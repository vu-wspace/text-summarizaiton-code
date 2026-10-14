from text_summarization.router import (LIST, QA, SUMMARIZE_TEXT, SUMMARIZE_TOPIC, route,
                                       strip_type_words, wanted_types)
from text_summarization.text import clean_input, contains_answer, norm_answer, normalize_text


# ----------------------------------------------------------------- chuẩn hoá VietNews
def test_normalize_vietnews():
    s = 'Ông Bảo_Th . đưa​ xe 51 G -77256 , " nổ " v . v'
    assert normalize_text(s, drop_caption=False) == 'Ông Bảo Th. đưa xe 51G-77256 , "nổ" v.v.'


def test_drop_caption():
    assert normalize_text('Câu một .  Ảnh minh hoạ') == 'Câu một .'


def test_clean_input_surrogates():
    assert clean_input('T\udcf3m t\udce1t') == 'T�m t�t'
    assert clean_input('Tóm tắt') == 'Tóm tắt'


# ----------------------------------------------------------------- so khớp đáp án (từ kết quả thật)
def test_contains_date_formats():
    assert contains_answer('Ngày 10/11/2017.', '10-11')


def test_contains_parentheses_spacing():
    assert contains_answer('với Đại học Gloucestershire (Anh Quốc).', 'Gloucestershire ( Anh Quốc )')


def test_contains_quotes_and_decimal():
    assert contains_answer('Huy lừa 3,2 tỷ đồng.', '3,2 tỷ đồng')
    assert not contains_answer('Huy lừa 3 tỷ đồng.', '3,2 tỷ đồng')
    assert not contains_answer('Minh Tú được đề cử.', '"Minh Anh"')


def test_norm_answer_empty():
    assert norm_answer('  ()  ') == ''
    assert not contains_answer('bất kỳ', '()')


# ----------------------------------------------------------------- router
def test_route():
    assert route('Liệt kê địa điểm trong vụ tàu') == (LIST, 'địa điểm trong vụ tàu')
    assert route('Tóm tắt vụ 2 tàu đối đầu') == (SUMMARIZE_TOPIC, 'vụ 2 tàu đối đầu')
    assert route('Tóm tắt: ' + 'a' * 300)[0] == SUMMARIZE_TEXT
    assert route('Huy lừa bao nhiêu tiền?') == (QA, 'Huy lừa bao nhiêu tiền?')
    assert route('hãy liệt kê nhân vật')[0] == LIST


def test_type_words():
    assert wanted_types('địa điểm trong vụ tàu') == ['LOCATION']
    assert wanted_types('vụ tàu') is None
    assert strip_type_words('các địa điểm trong vụ tàu') == 'vụ tàu'
    assert strip_type_words('địa điểm') == 'địa điểm'   # không còn gì -> giữ nguyên
