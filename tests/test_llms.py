from types import SimpleNamespace

from text_summarization.llms import CHATML, LLM, ROLE_LEAK_RE


class FakeTok:
    def __init__(self, bos, add_bos):
        self.bos_token, self.add_bos_token, self.chat_template = bos, add_bos, None

    def apply_chat_template(self, msgs, chat_template, tokenize, add_generation_prompt):
        return '|'.join(m['content'] for m in msgs) + '|A:'


def prompt_with(tok, template=CHATML):
    llm = LLM('vinallama-2.7b-chat')
    llm._tok = tok
    llm.spec = SimpleNamespace(**{**llm.spec.__dict__, 'chat_template': template})
    return llm._prompt('S', 'U')


def test_bos_added_for_llama_tokenizers():
    assert prompt_with(FakeTok('<s>', True)) == '<s>S|U|A:'


def test_no_bos_when_tokenizer_has_none_or_does_not_add():
    assert prompt_with(FakeTok(None, None)) == 'S|U|A:'
    assert prompt_with(FakeTok('<s>', False)) == 'S|U|A:'


def test_role_leak_cut():
    assert ROLE_LEAK_RE.sub('', 'Đội tuyển về nước.\n user\n[2] Nghị').strip() == 'Đội tuyển về nước.'
    assert ROLE_LEAK_RE.sub('', 'A.\n<|im_start|>assistant\nB') == 'A.'


def test_role_word_inside_answer_kept():
    s = 'Người dùng (user) hỏi về tàu.\nDòng 2'
    assert ROLE_LEAK_RE.sub('', s) == s
