"""So sánh LLM: chỉ số, prompt từng mô hình, và luồng contexts -> answer -> report với LLM giả."""
import csv
import sys
import types
from typing import ClassVar

import pytest
from conftest import FakeModels, FakeStore, hit

from text_summarization.cli import main
from text_summarization.core.data import read_jsonl, write_jsonl
from text_summarization.evaluation import compare
from text_summarization.evaluation.metrics import (
    aggregate,
    bootstrap_ci,
    exact_match,
    holm,
    mcnemar,
    score,
    token_f1,
    token_recall,
)
from text_summarization.rag.llms import LLM, LLMS, PROMPTS, _version_tuple, get_spec
from text_summarization.rag.pipeline import RAGPipeline


# ----------------------------------------------------------------- chỉ số
def test_metrics():
    assert token_recall('Ông Vũ Ngọc Long bị bắt', 'Vũ Ngọc Long') == 1.0
    assert token_recall('Long', 'Vũ Ngọc Long') == pytest.approx(1 / 3)
    assert token_f1('3,2 tỷ đồng', '3,2 tỷ đồng') == 1.0
    assert 0 < token_f1('Huy đã lừa tổng cộng 3,2 tỷ đồng của nhiều người', '3,2 tỷ đồng') < 0.6   # dài -> F1 thấp
    assert exact_match('14,43 %', '14,43%') and not exact_match('Năm 14,43%', '14,43%')
    s = score('Tôi không biết dựa trên các tài liệu được cung cấp.', '3,2 tỷ đồng')
    assert s['abstain'] and not s['contains'] and s['f1'] == 0
    assert aggregate([{'contains': True, 'f1': 1.0}, {'contains': False, 'f1': 0.5}],
                     keys=('contains', 'f1')) == {'contains': 0.5, 'f1': 0.75}


def test_significance():
    a = [1] * 6 + [0] * 14 + [1] * 30                  # A đúng B sai 6 câu, B đúng A sai 14 câu, cùng đúng 30
    b = [0] * 6 + [1] * 14 + [1] * 30
    n01, n10, p = mcnemar(a, b)
    assert (n01, n10) == (6, 14) and p == pytest.approx(0.1153, abs=1e-4)
    assert mcnemar([1, 0], [1, 0]) == (0, 0, 1.0)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    lo, hi = bootstrap_ci([1] * 50 + [0] * 50)
    assert lo < 0.5 < hi and hi - lo < 0.25
    assert bootstrap_ci([]) == (0.0, 0.0)


# ----------------------------------------------------------------- danh sách LLM
def test_registry_and_prompts():
    assert set(LLMS) == {'qwen2.5-3b', 'vi-qwen2-3b-rag', 'qwen3.5-2b-vie',
                         'vinallama-2.7b-chat', 'vistral-7b-chat', 'vietnamese-llama2-7b'}
    assert get_spec('vi-qwen2-3b-rag').prompt == 'vi-rag-concise'
    concise = PROMPTS['vi-rag-concise'][1]                 # khuôn AITeamVN giữ nguyên, chỉ thêm 1 yêu cầu
    assert concise.startswith(PROMPTS['vi-rag'][1].split('- Chỉ cần')[0]) and 'MỘT câu ngắn' in concise
    assert get_spec('qwen3.5-2b-vie').loader == 'image-text'
    with pytest.raises(SystemExit):
        get_spec('gpt-5')
    for _system, user in PROMPTS.values():                  # mọi prompt đều nhận đủ 2 trường
        assert '{' not in user.format(context='C', question='Q')
    assert _version_tuple('5.2.0.dev0') > _version_tuple('5.1.9') > _version_tuple('4.57.6')


def test_answer_uses_model_specific_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(LLM, 'chat', lambda self, s, u, n=256: seen.setdefault(self.spec.key, u))
    LLM('vi-qwen2-3b-rag').answer('Ai?', 'NGỮ')
    LLM('qwen2.5-3b').answer('Ai?', 'NGỮ')
    assert '### Ngữ cảnh :\nNGỮ' in seen['vi-qwen2-3b-rag'] and seen['vi-qwen2-3b-rag'].endswith('### Trả lời :')
    assert seen['qwen2.5-3b'].startswith('NGỮ CẢNH:\nNGỮ')



# ----------------------------------------------------------------- luồng đầy đủ với LLM giả
class FakeLLM:
    REPLIES: ClassVar[dict] = {'qwen2.5-3b': 'Đáp án là 3,2 tỷ đồng.', 'vi-qwen2-3b-rag': 'Không biết.'}

    def __init__(self, key, load_4bit=False, verbose=False):
        self.spec = get_spec(key)
        self.model = types.SimpleNamespace(dtype='bf16')

    def timed_answer(self, q, ctx, n):
        return self.REPLIES[self.spec.key], 0.5

    def unload(self):
        pass


@pytest.fixture
def fake_torch(monkeypatch):
    cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=cuda, __version__='fake'))
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(__version__='4.57.6'))
    monkeypatch.setattr(compare, 'LLM', FakeLLM)


def test_contexts_answer_report(tmp_path, articles, fake_torch):
    store = FakeStore([hit('1', 'huy'), hit('2', 'đt'), hit('3', 'ch')])
    rag = RAGPipeline(store, articles, {}, FakeModels(vocab=['huy']))
    qs = [{'id': 1, 'guid': '1', 'question': 'Hồ Xuân Huy lừa bao nhiêu tiền?', 'answer': '3,2 tỷ đồng'},
          {'id': 2, 'guid': '9', 'question': 'Bài không có trong kho?', 'answer': 'X'}]
    ctx = compare.build_contexts(rag, qs)
    assert ctx[0]['hit@1'] and ctx[0]['context'] and not ctx[1]['hit@3']
    write_jsonl(ctx, compare.contexts_path(tmp_path))

    for key in FakeLLM.REPLIES:
        rows, meta = compare.run_llm(key, read_jsonl(compare.contexts_path(tmp_path)))
        assert meta['repo'] == get_spec(key).repo and meta['peak_vram_gb'] is None
        compare.save_answers(tmp_path, key, rows, meta)

    results = compare.load_results(tmp_path)
    assert list(results) == ['qwen2.5-3b', 'vi-qwen2-3b-rag']        # qwen3.5 chưa chạy -> bỏ qua
    summary = {r['llm']: r for r in compare.write_report(tmp_path, ctx, results)}
    assert summary['qwen2.5-3b']['contains'] == 0.5 and summary['vi-qwen2-3b-rag']['abstain'] == 1.0

    md = (tmp_path / 'compare_report.md').read_text(encoding='utf-8')
    assert '✅' in md and '🚫' in md and 'hit@1 = 0.50' in md
    with open(tmp_path / 'compare_review.csv', encoding='utf-8-sig') as f:
        review = list(csv.DictReader(f))
    assert len(review) == 4 and 'dung_that (0/1)' in review[0]


def test_cli_answer_and_report(tmp_path, fake_torch, capsys):
    write_jsonl([{'id': 1, 'question': 'Q?', 'answer': '3,2 tỷ đồng', 'guid': '1', 'context': 'C',
                  'sources': ['1'], 'hit@1': True, 'hit@3': True}], compare.contexts_path(tmp_path))
    main(['--questions-dir', str(tmp_path), 'answer', '--llm', 'qwen2.5-3b'])
    main(['--questions-dir', str(tmp_path), 'report'])
    assert (tmp_path / 'answers_qwen2.5-3b.meta.json').exists()
    assert '| qwen2.5-3b | 1.00 |' in capsys.readouterr().out
    main(['llms'])
    assert 'alphaedge-ai/Qwen3.5-2B-vie-32768' in capsys.readouterr().out


def test_cli_report_exclude_writes_clean_report(tmp_path, capsys):
    ctx = [{'id': i, 'question': 'Q?', 'answer': 'A', 'guid': '1', 'context': 'C', 'sources': ['1'],
            'hit@1': True, 'hit@3': True} for i in (1, 2)]
    write_jsonl(ctx, compare.contexts_path(tmp_path))
    write_jsonl([{'id': 1, 'pred': 'A', 'seconds': 1.0}, {'id': 2, 'pred': 'B', 'seconds': 1.0}],
                compare.answers_path(tmp_path, 'qwen2.5-3b'))
    write_jsonl([{'id': 2, 'reason': 'malformed'}], tmp_path / 'ambiguous.jsonl')
    main(['--questions-dir', str(tmp_path), 'report', '--exclude', 'ambiguous.jsonl'])
    assert '| qwen2.5-3b | 1.00 |' in capsys.readouterr().out            # câu 2 (sai) đã bị bỏ
    assert (tmp_path / 'compare_report_clean.md').exists() and not (tmp_path / 'compare_report.md').exists()
    assert 'trên 1 câu hỏi' in (tmp_path / 'compare_report_clean.md').read_text(encoding='utf-8')


def test_transformers_version_check(monkeypatch):
    from text_summarization.rag.llms import check_transformers

    spec = get_spec('qwen3.5-2b-vie')                                # cần transformers >= 5.2 -> service llm5
    for have, ok in [('5.2.0', True), ('5.10.1', True), ('4.57.6', False)]:
        monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(__version__=have))
        if ok:
            check_transformers(spec)
        else:
            with pytest.raises(SystemExit, match='service llm5'):
                check_transformers(spec)
    check_transformers(get_spec('vistral-7b-chat'))                  # llama-server: không cần transformers


def test_answer_hits_counts_answer_in_any_article():
    ctx = '[1] Bài A nói về bão . Tâm bão cách bờ 450 km\n\n[2] Bài B . Gió giật cấp 12'
    assert compare.answer_hits(ctx, '450 km') == (True, True)
    assert compare.answer_hits(ctx, 'cấp 12') == (False, True)
    assert compare.answer_hits('', 'x') == (False, False)
