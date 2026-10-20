import math

from conftest import FakeModels

from text_summarization.evaluation.judge import control_items, control_report, judge_one, parse_verdict, summarize


def test_parse_clean_json():
    assert parse_verdict('{"reason": "có trong ngữ cảnh", "score": 1}') == (1.0, 'có trong ngữ cảnh')


def test_parse_snaps_to_scale_and_accepts_comma_decimal():
    assert parse_verdict('{"reason": "x", "score": 0.8}')[0] == 1.0
    assert parse_verdict('{"reason": "x", "score": "0,5"}')[0] == 0.5


def test_parse_truncated_json_falls_back_to_regex():
    score, reason = parse_verdict('```json\n{"reason": "thiếu \\"ngày\\" nhưng đúng", "score": 0.5')
    assert score == 0.5 and reason.startswith('thiếu')


def test_parse_failure_is_none_not_zero():
    score, reason = parse_verdict('Câu trả lời khá tốt.')
    assert score is None and reason.startswith('parse lỗi')


def test_parse_ignores_out_of_range_and_template_placeholder():
    assert parse_verdict('{"reason": "x", "score": 7}')[0] is None
    assert parse_verdict('{"reason": "<nhận xét>", "score": <0 hoặc 0.5 hoặc 1>}')[0] is None


def test_judge_one_empty_answer_skips_llm():
    fake = FakeModels(chat_reply='{"reason": "x", "score": 1}')
    out = judge_one(fake, 'q', 'ctx', '  ')
    assert out['faithfulness'] is None and out['relevancy'] == 0.0 and not fake.calls


def test_summarize_excludes_parse_failures():
    rows = [{'faithfulness': 1.0, 'relevancy': None}, {'faithfulness': None, 'relevancy': 0.5},
            {'faithfulness': 0.0, 'relevancy': 1.0}]
    s = summarize(rows)
    assert s['faithfulness'] == 0.5 and s['relevancy'] == 0.75
    assert math.isclose(s['faithfulness_parse_fail'], 1 / 3)


def test_control_items_pair_each_question_with_foreign_answer():
    ctx = [{'id': i, 'context': f'bài {i} có đáp án A{i}', 'answer': f'A{i}'} for i in range(5)]
    ctx.append({'id': 99, 'context': 'không chứa', 'answer': 'Z'})          # ngữ cảnh không có đáp án -> bỏ
    rows = control_items(ctx, n=10)
    pos = [r for r in rows if r['expect'] == 1.0]
    neg = [r for r in rows if r['expect'] == 0.0]
    assert len(pos) == len(neg) == 5 and 99 not in {r['id'] for r in rows}
    assert all(r['pred'] == f"A{r['id']}" for r in pos)
    assert all(r['pred'] != f"A{r['id']}" for r in neg)


def test_control_report_accuracy():
    rows = [{'expect': 1.0, 'faithfulness': 1.0, 'relevancy': 1.0},
            {'expect': 0.0, 'faithfulness': 0.0, 'relevancy': 1.0},
            {'expect': 0.0, 'faithfulness': None, 'relevancy': 0.0}]
    rep = control_report(rows)
    assert rep['faithfulness']['accuracy'] == 1.0 and rep['relevancy']['accuracy'] == 2 / 3
