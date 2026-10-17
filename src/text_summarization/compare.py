"""So sánh nhiều LLM trên CÙNG bộ câu hỏi và CÙNG ngữ cảnh.

    1. contexts : retriever + re-ranker chạy MỘT lần -> contexts.jsonl (câu hỏi, đáp án chuẩn, ngữ cảnh, hit@k)
    2. answer   : mỗi LLM đọc contexts.jsonl, sinh câu trả lời -> answers_<llm>.jsonl (+ .meta.json)
    3. report   : chấm + gộp -> compare_report.md, compare_summary.csv, compare_review.csv

Tách 3 bước vì: (a) công bằng — mọi LLM nhận ngữ cảnh giống hệt nhau, khác biệt chỉ do LLM;
(b) VRAM — lúc sinh chỉ có 1 LLM trên GPU; (c) Qwen3.5 cần transformers ≥ 5.2 nên chạy bước 2 ở môi trường riêng.
"""
import csv
import json
import platform
import time
from pathlib import Path

from .data import read_jsonl, write_jsonl
from .llms import LLMS, LLM, get_spec
from .metrics import METRICS, aggregate, score


def contexts_path(qdir):
    return Path(qdir) / 'contexts.jsonl'


def answers_path(qdir, key):
    return Path(qdir) / f'answers_{key}.jsonl'


# ----------------------------------------------------------------- 1. ngữ cảnh
def build_contexts(pipeline, questions):
    rows = []
    for r in questions:
        q, ctx, used = pipeline.prepare(r['question'])
        rows.append({'id': r['id'], 'question': q, 'answer': r['answer'], 'guid': str(r['guid']),
                     'context': ctx, 'sources': used,
                     'hit@1': bool(used) and used[0] == str(r['guid']), 'hit@3': str(r['guid']) in used[:3]})
        print(f"  [{r['id']}] hit@1={int(rows[-1]['hit@1'])} hit@3={int(rows[-1]['hit@3'])}  {q}")
    return rows


# ----------------------------------------------------------------- 2. sinh câu trả lời
def run_llm(key, contexts, load_4bit=False, verbose=False, max_new_tokens=256):
    """Sinh câu trả lời cho mọi câu bằng 1 LLM. Trả về (rows, meta). Giải phóng VRAM khi xong."""
    import torch
    import transformers

    llm = LLM(key, load_4bit=load_4bit, verbose=verbose)
    server = llm.spec.loader == 'llama-server'
    cuda = torch.cuda.is_available() and not server   # llama-server ở tiến trình khác: torch không đo được VRAM
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    _ = llm.model                                    # nạp trước để thời gian sinh không tính thời gian nạp
    load_s = time.perf_counter() - t0
    rows = []
    for c in contexts:
        if c['context']:
            pred, sec = llm.timed_answer(c['question'], c['context'], max_new_tokens)
        else:
            pred, sec = '', 0.0
        rows.append({'id': c['id'], 'pred': pred, 'seconds': round(sec, 3)})
        print(f"  [{key}] [{c['id']}] {sec:5.1f}s  {pred[:100]}")
    meta = {'llm': key, 'repo': llm.spec.repo, 'prompt': llm.spec.prompt,
            'dtype': str(getattr(llm.model, 'dtype', '')), '4bit': load_4bit,
            'transformers': transformers.__version__, 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or 'cpu',
            'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
            'load_seconds': round(load_s, 1)}
    llm.unload()
    empty = sum(1 for r, c in zip(rows, contexts) if c['context'] and not r['pred'].strip())
    if empty:
        print(f'  ⚠ {key}: {empty}/{len(rows)} câu trả lời rỗng, xem dòng "!" ở trên để biết token thô.')
    meta['empty_answers'] = empty
    return rows, meta


def save_answers(qdir, key, rows, meta):
    write_jsonl(rows, answers_path(qdir, key))
    with open(answers_path(qdir, key).with_suffix('.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'  -> {answers_path(qdir, key)}')


# ----------------------------------------------------------------- 3. báo cáo
def load_results(qdir, keys=None):
    """{llm: (answer_rows, meta)} cho các LLM đã có file answers_*.jsonl."""
    out = {}
    for key in (keys or LLMS):
        p = answers_path(qdir, key)
        if p.exists():
            meta_p = p.with_suffix('.meta.json')
            meta = json.loads(meta_p.read_text(encoding='utf-8')) if meta_p.exists() else {}
            out[key] = (read_jsonl(p), meta)
    return out


def score_all(contexts, results):
    """-> {llm: [dict mỗi câu: id, pred, seconds, contains, recall, f1, em, abstain, words]}"""
    gold = {c['id']: c for c in contexts}
    scored = {}
    for key, (rows, _) in results.items():
        scored[key] = [{**r, **score(r['pred'], gold[r['id']]['answer'])} for r in rows if r['id'] in gold]
    return scored


def summary_table(contexts, scored, results):
    lines = ['| LLM | contains ↑ | recall ↑ | F1 ↑ | EM ↑ | từ chối | số từ TB | giây/câu | VRAM (GB) |',
             '|---|---|---|---|---|---|---|---|---|']
    csv_rows = []
    for key, rows in scored.items():
        a = aggregate(rows)
        sec = sum(r['seconds'] for r in rows) / max(len(rows), 1)
        vram = results[key][1].get('peak_vram_gb')
        lines.append(f"| {key} | {a['contains']:.2f} | {a['recall']:.2f} | {a['f1']:.2f} | {a['em']:.2f} | "
                     f"{a['abstain']:.2f} | {a['words']:.1f} | {sec:.2f} | {vram if vram is not None else '—'} |")
        csv_rows.append({'llm': key, **{k: round(a[k], 4) for k in METRICS},
                         'seconds': round(sec, 3), 'peak_vram_gb': vram, 'n': len(rows)})
    return '\n'.join(lines), csv_rows


def write_report(qdir, contexts, results):
    qdir = Path(qdir)
    if not results:
        raise SystemExit('Chưa có file answers_*.jsonl nào. Chạy bước answer trước.')
    scored = score_all(contexts, results)
    table, csv_rows = summary_table(contexts, scored, results)
    n = len(contexts)
    h1 = sum(c['hit@1'] for c in contexts) / max(n, 1)
    h3 = sum(c['hit@3'] for c in contexts) / max(n, 1)

    md = [f'# So sánh LLM trên {n} câu hỏi', '',
          f'Truy xuất (chung cho mọi LLM): **hit@1 = {h1:.2f}**, **hit@3 = {h3:.2f}**. '
          'Mọi LLM nhận cùng một ngữ cảnh, nên khác biệt bên dưới chỉ do LLM.', '', table, '',
          '- **contains**: đáp án chuẩn nằm trọn trong câu trả lời (chỉ số chính).',
          '- **recall**: tỉ lệ từ của đáp án chuẩn có trong câu trả lời (điểm từng phần).',
          '- **F1 / EM**: kiểu SQuAD, phạt câu trả lời dài dòng; LLM trả lời thành câu nên EM thường thấp.',
          '- **từ chối**: tỉ lệ câu LLM nói không biết dù ngữ cảnh có bài đúng (khi hit@3 = 1 là mất điểm).', '',
          '## Cấu hình', '', '| LLM | repo | prompt | dtype | 4-bit | transformers | GPU |',
          '|---|---|---|---|---|---|---|']
    if len({bool(m.get('4bit')) for _, m in results.values()}) > 1:
        md.insert(md.index('## Cấu hình'), '> ⚠ Các LLM chạy khác chế độ (bf16 và 4-bit): chênh lệch có thể do lượng tử hoá.\n')
    for key, (_, meta) in results.items():
        md.append(f"| {key} | {meta.get('repo', get_spec(key).repo)} | {meta.get('prompt', '')} | "
                  f"{meta.get('dtype', '')} | {'có' if meta.get('4bit') else 'không'} | {meta.get('transformers', '')} | "
                  f"{meta.get('gpu', '')} |")
    md += ['', '## Từng câu', '']
    keys = list(scored)
    by_id = {k: {r['id']: r for r in scored[k]} for k in keys}
    review = []
    for c in contexts:
        md.append(f"### [{c['id']}] {c['question']}")
        md.append(f"Đáp án chuẩn: **{c['answer']}** · bài nguồn {c['guid']} · "
                  f"hit@1={int(c['hit@1'])} hit@3={int(c['hit@3'])}")
        md.append('')
        md.append('| LLM | ✓ | câu trả lời |')
        md.append('|---|---|---|')
        for k in keys:
            r = by_id[k].get(c['id'])
            if r is None:
                continue
            mark = '✅' if r['contains'] else ('🚫' if r['abstain'] else '❌')
            pred = r['pred'].replace('\n', ' ').replace('|', '\\|')
            md.append(f'| {k} | {mark} | {pred} |')
            review.append({'id': c['id'], 'question': c['question'], 'gold': c['answer'], 'llm': k,
                           'pred': r['pred'], 'contains': int(r['contains']), 'hit@3': int(c['hit@3']),
                           'dung_that (0/1)': '', 'ghi_chu': ''})
        md.append('')

    (qdir / 'compare_report.md').write_text('\n'.join(md), encoding='utf-8')
    _write_csv(qdir / 'compare_summary.csv', csv_rows)
    _write_csv(qdir / 'compare_review.csv', review)
    print(table)
    print(f"  -> {qdir / 'compare_report.md'}\n  -> {qdir / 'compare_summary.csv'}"
          f"\n  -> {qdir / 'compare_review.csv'}  (cột dung_that để chấm tay)")
    return csv_rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:   # utf-8-sig: Excel đọc đúng tiếng Việt
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
