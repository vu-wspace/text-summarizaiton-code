"""LLM-as-a-judge: chấm FAITHFULNESS (bám ngữ cảnh) và RELEVANCY (đúng trọng tâm câu hỏi) cho answers_*.jsonl.

Bản cũ (scripts/eval_faithfulness_relevancy.py) cho faithfulness = 0.000 với MỌI mô hình, do lỗi đo chứ không do
mô hình:
  - khuôn trả lời mẫu '{"score": 0.0, "reason": "..."}' -> judge 3B chép nguyên điểm 0.0 và chuỗi '...';
  - điểm đứng TRƯỚC lý do -> judge chấm trước, giải thích sau (lý do "không bịa đặt" nhưng điểm 0);
  - parse lỗi bị tính 0 điểm (8–20 % số câu) thay vì loại khỏi trung bình;
  - rubric trộn "thiếu thông tin" với "bịa thông tin".
Ở đây: lý do trước điểm, không có điểm mẫu, parse lỗi = None (báo tỉ lệ), và có CÂU KIỂM SOÁT để biết judge có
phân biệt được câu đúng / câu bịa hay không trước khi tin con số của nó:
  positive  pred = đáp án chuẩn (có trong ngữ cảnh)              -> kỳ vọng faithfulness ≈ 1, relevancy ≈ 1
  negative  pred = đáp án chuẩn của MỘT CÂU HỎI KHÁC (lạc đề, bịa) -> kỳ vọng faithfulness ≈ 0, relevancy ≈ 0

Relevancy không nhìn đáp án chuẩn (độ đúng đã có contains / recall ở metrics.py).
"""
import json
import random
import re

from ..core.text import contains_answer

SCORES = (0.0, 0.5, 1.0)

FAITH_SYSTEM = (
    'Bạn là giám khảo kiểm tra câu trả lời của hệ thống hỏi đáp có trích dẫn tài liệu.\n'
    'Nhiệm vụ: kiểm tra từng thông tin trong CÂU TRẢ LỜI có được NGỮ CẢNH ủng hộ hay không.\n'
    'Thang điểm:\n'
    '- 1: mọi thông tin trong câu trả lời đều có trong ngữ cảnh (diễn đạt lại, rút gọn vẫn được 1).\n'
    '- 0.5: phần chính có trong ngữ cảnh nhưng có chi tiết phụ không có trong ngữ cảnh.\n'
    '- 0: thông tin chính không có trong ngữ cảnh hoặc mâu thuẫn với ngữ cảnh.\n'
    'Lưu ý: câu trả lời NGẮN hoặc THIẾU ý KHÔNG bị trừ điểm; chỉ trừ khi có thông tin không có trong ngữ cảnh. '
    'Câu trả lời từ chối ("không biết") được 1.'
)
FAITH_USER = (
    'NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}\n\nCÂU TRẢ LỜI: {pred}\n\n'
    'Trước tiên nêu ngắn gọn thông tin chính của câu trả lời và chỗ nó xuất hiện (hoặc không) trong ngữ cảnh, '
    'sau đó cho điểm. Chỉ trả về một dòng JSON theo đúng khuôn:\n'
    '{{"reason": "<nhận xét>", "score": <0 hoặc 0.5 hoặc 1>}}'
)

REL_SYSTEM = (
    'Bạn là giám khảo đánh giá câu trả lời có ĐÚNG TRỌNG TÂM câu hỏi hay không (không cần kiểm tra đúng sai).\n'
    'Thang điểm:\n'
    '- 1: trả lời thẳng vào điều câu hỏi hỏi (hỏi ai -> người, hỏi bao nhiêu -> con số, hỏi khi nào -> thời gian...).\n'
    '- 0.5: có liên quan nhưng lan man, hoặc chỉ trả lời một phần điều được hỏi.\n'
    '- 0: nói chuyện khác, không trả lời điều được hỏi, hoặc từ chối trả lời.'
)
REL_USER = (
    'CÂU HỎI: {question}\n\nCÂU TRẢ LỜI: {pred}\n\n'
    'Trước tiên nêu câu hỏi đang hỏi gì và câu trả lời đáp lại điều gì, sau đó cho điểm. '
    'Chỉ trả về một dòng JSON theo đúng khuôn:\n'
    '{{"reason": "<nhận xét>", "score": <0 hoặc 0.5 hoặc 1>}}'
)

SCORE_RE = re.compile(r'["\']?(?:score|điểm)["\']?\s*[:=]\s*["\']?([01](?:[.,]\d+)?)', re.IGNORECASE)
REASON_RE = re.compile(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)', re.DOTALL)


def _snap(x):
    """Điểm bất kỳ trong [0, 1] -> mức gần nhất của thang 0 / 0.5 / 1."""
    return min(SCORES, key=lambda s: abs(s - x))


def parse_verdict(text):
    """Output judge -> (score | None, reason). Thử JSON trước, rồi regex (judge nhỏ hay viết JSON lỗi /
    bị cắt giữa chừng). None = không đọc được điểm -> KHÔNG tính vào trung bình."""
    text = text or ''
    for m in re.finditer(r'\{[^{}]*\}', text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        try:
            score = float(str(obj.get('score')).replace(',', '.'))
        except (TypeError, ValueError):
            continue
        if 0 <= score <= 1:
            return _snap(score), str(obj.get('reason', ''))
    m = SCORE_RE.search(text)
    reason = REASON_RE.search(text)
    reason = reason.group(1) if reason else text[:200]
    if m:
        score = float(m.group(1).replace(',', '.'))
        if 0 <= score <= 1:
            return _snap(score), reason
    return None, f'parse lỗi: {text[:200]}'


def judge_one(llm, question, context, pred, max_new_tokens=256):
    """-> {'faithfulness', 'faithfulness_reason', 'relevancy', 'relevancy_reason'} (điểm None = parse lỗi)."""
    if not (pred or '').strip():
        return {'faithfulness': None, 'faithfulness_reason': 'câu trả lời rỗng',
                'relevancy': 0.0, 'relevancy_reason': 'câu trả lời rỗng'}
    f, fr = parse_verdict(llm.chat(FAITH_SYSTEM, FAITH_USER.format(context=context, question=question, pred=pred),
                                   max_new_tokens=max_new_tokens, repetition_penalty=1.0))
    r, rr = parse_verdict(llm.chat(REL_SYSTEM, REL_USER.format(question=question, pred=pred),
                                   max_new_tokens=max_new_tokens, repetition_penalty=1.0))
    return {'faithfulness': f, 'faithfulness_reason': fr, 'relevancy': r, 'relevancy_reason': rr}


def control_items(contexts, n=30, seed=0):
    """Câu kiểm soát từ các câu mà ngữ cảnh chứa đáp án chuẩn: n positive (đáp án đúng) + n negative
    (đáp án của câu hỏi khác, xoay vòng 1 bước để không câu nào nhận lại đáp án của chính nó)."""
    ok = [c for c in contexts if c['context'] and contains_answer(c['context'], c['answer'])]
    ok = random.Random(seed).sample(ok, min(n, len(ok)))
    rows = [{'id': c['id'], 'pred': c['answer'], 'expect': 1.0} for c in ok]
    rows += [{'id': c['id'], 'pred': ok[(i + 1) % len(ok)]['answer'], 'expect': 0.0}
             for i, c in enumerate(ok)] if len(ok) > 1 else []
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float('nan')


def summarize(rows):
    """Trung bình faithfulness / relevancy (bỏ câu parse lỗi) + tỉ lệ parse lỗi."""
    n = len(rows)
    out = {'n': n}
    for k in ('faithfulness', 'relevancy'):
        out[k] = _mean(r[k] for r in rows)
        out[f'{k}_parse_fail'] = sum(r[k] is None for r in rows) / n if n else 0.0
    return out


def control_report(rows):
    """Judge có tách được positive / negative không: điểm TB mỗi nhóm và độ chính xác (ngưỡng 0.5).
    Với relevancy, negative chỉ mang tính tham khảo: đáp án lạc của câu "bao nhiêu" vẫn có thể là một con số."""
    out = {}
    for k in ('faithfulness', 'relevancy'):
        pos = [r[k] for r in rows if r['expect'] == 1.0]
        neg = [r[k] for r in rows if r['expect'] == 0.0]
        scored = [(r[k], r['expect']) for r in rows if r[k] is not None]
        acc = sum((s >= 0.5) == (e == 1.0) for s, e in scored) / len(scored) if scored else float('nan')
        out[k] = {'pos_mean': _mean(pos), 'neg_mean': _mean(neg), 'accuracy': acc}
    return out


# ----------------------------------------------------------------- chạy
def _judge_rows(llm, items, contexts, tag):
    out = []
    for a in items:
        c = contexts[a['id']]
        out.append({**a, **judge_one(llm, c['question'], c['context'], a['pred'])})
        r = out[-1]
        print(f"  [{tag}] [{a['id']}] faith={r['faithfulness']} rel={r['relevancy']}  {a['pred'][:60]}")
    return out


def run_judge(qdir, judge_key, answer_keys, load_4bit=True, n_controls=30, verbose=False):
    """Chấm answers_<key>.jsonl bằng LLM judge_key (nạp 1 lần). answer_keys là phần sau 'answers_' của tên file
    (vd. 'vi-qwen2-3b-rag_temp05'). -> ghi judge_<judge>/answers_<key>.jsonl, controls.jsonl, report.md"""
    from pathlib import Path

    from ..core.data import read_jsonl, write_jsonl
    from ..rag.llms import LLM

    qdir = Path(qdir)
    out_dir = qdir / f'judge_{judge_key}'
    out_dir.mkdir(exist_ok=True)
    contexts = {c['id']: c for c in read_jsonl(qdir / 'contexts.jsonl')}
    llm = LLM(judge_key, load_4bit=load_4bit, verbose=verbose)

    controls = _judge_rows(llm, control_items(list(contexts.values()), n_controls), contexts, 'control')
    write_jsonl(controls, out_dir / 'controls.jsonl')
    ctrl = control_report(controls)

    results = {}
    for key in answer_keys:
        rows = _judge_rows(llm, read_jsonl(qdir / f'answers_{key}.jsonl'), contexts, key)
        write_jsonl(rows, out_dir / f'answers_{key}.jsonl')
        results[key] = summarize(rows)
    llm.unload()

    md = [f'# LLM-as-a-judge: {judge_key}', '',
          '## Kiểm soát (judge có phân biệt được câu đúng / câu bịa không?)', '',
          f'{len(controls) // 2} positive (đáp án chuẩn) + {len(controls) // 2} negative (đáp án của câu hỏi khác).',
          '', '| chỉ số | TB positive ↑ | TB negative ↓ | độ chính xác |', '|---|---|---|---|']
    md += [f"| {k} | {v['pos_mean']:.2f} | {v['neg_mean']:.2f} | {v['accuracy']:.2f} |" for k, v in ctrl.items()]
    md += ['', '## Kết quả', '', '| answers | faithfulness | parse lỗi | relevancy | parse lỗi | n |',
           '|---|---|---|---|---|---|']
    md += [f"| {k} | {s['faithfulness']:.3f} | {s['faithfulness_parse_fail']:.0%} | {s['relevancy']:.3f} | "
           f"{s['relevancy_parse_fail']:.0%} | {s['n']} |" for k, s in results.items()]
    if judge_key in {k.split('_')[0] for k in answer_keys}:
        md += ['', f'> ⚠ {judge_key} vừa là judge vừa là mô hình được chấm (thiên vị tự chấm).']
    (out_dir / 'report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))
    print(f'  -> {out_dir / "report.md"}')
    return ctrl, results
