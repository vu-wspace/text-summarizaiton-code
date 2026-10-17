from text_summarization.llms import ROLE_LEAK_RE


def test_role_leak_cut():
    assert ROLE_LEAK_RE.sub('', 'Đội tuyển về nước.\n user\n[2] Nghị').strip() == 'Đội tuyển về nước.'
    assert ROLE_LEAK_RE.sub('', 'A.\n<|im_start|>assistant\nB') == 'A.'


def test_role_word_inside_answer_kept():
    s = 'Người dùng (user) hỏi về tàu.\nDòng 2'
    assert ROLE_LEAK_RE.sub('', s) == s
