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
import re
import time
from pathlib import Path

from ..core.data import read_jsonl, write_jsonl
from ..core.text import contains_answer
from ..rag.llms import LLM, LLMS, get_spec
from .metrics import METRICS, aggregate, bootstrap_ci, holm, mcnemar, score


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


def answer_hits(context, gold):
    """-> (ans@1, ans@3): đáp án chuẩn có trong bài đầu / trong cả ngữ cảnh. hit@k so guid bài nguồn nên tính
    trượt khi bài khác cùng sự kiện (VietNews có nhiều bài trùng chủ đề) vẫn chứa đáp án; ans@k đo đúng điều LLM
    cần: ngữ cảnh có đáp án hay không."""
    parts = re.split(r'\n\n(?=\[\d+\] )', context or '')
    return contains_answer(parts[0], gold), contains_answer(context or '', gold)


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
            'dtype': str(getattr(llm.model, 'dtype', '')), '4bit': load_4bit and not server,
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
    lines = ['| LLM | contains ↑ | 95% CI | recall ↑ | F1 ↑ | EM ↑ | từ chối | số từ TB | giây/câu | VRAM (GB) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    csv_rows = []
    for key, rows in sorted(scored.items(), key=lambda kv: -aggregate(kv[1])['contains']):
        a = aggregate(rows)
        lo, hi = bootstrap_ci([r['contains'] for r in rows])
        sec = sum(r['seconds'] for r in rows) / max(len(rows), 1)
        vram = results[key][1].get('peak_vram_gb')
        lines.append(f"| {key} | {a['contains']:.2f} | {lo:.2f}–{hi:.2f} | {a['recall']:.2f} | {a['f1']:.2f} | "
                     f"{a['em']:.2f} | {a['abstain']:.2f} | {a['words']:.1f} | {sec:.2f} | "
                     f"{vram if vram is not None else '—'} |")
        csv_rows.append({'llm': key, **{k: round(a[k], 4) for k in METRICS}, 'contains_ci_lo': round(lo, 4),
                         'contains_ci_hi': round(hi, 4), 'seconds': round(sec, 3), 'peak_vram_gb': vram,
                         'n': len(rows)})
    return '\n'.join(lines), csv_rows


def pairwise_table(scored):
    """McNemar từng cặp trên contains (chỉ những câu cả 2 LLM cùng trả lời), p hiệu chỉnh Holm."""
    keys, pairs = list(scored), []
    by_id = {k: {r['id']: r['contains'] for r in scored[k]} for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ids = sorted(by_id[a].keys() & by_id[b].keys())
            pairs.append((a, b, *mcnemar([by_id[a][x] for x in ids], [by_id[b][x] for x in ids])))
    adj = holm([p[4] for p in pairs])
    lines = ['| A vs B | A đúng, B sai | B đúng, A sai | p | p (Holm) |', '|---|---|---|---|---|']
    lines += [f"| {a} vs {b} | {n01} | {n10} | {p:.4f} | {f'**{q:.4f}**' if q < 0.05 else f'{q:.4f}'} |"
              for (a, b, n01, n10, p), q in zip(pairs, adj)]
    return '\n'.join(lines)


def write_report(qdir, contexts, results, suffix=''):
    """suffix: thêm vào tên file kết quả (vd. '_clean' khi chấm trên tập đã bỏ câu mơ hồ)."""
    qdir = Path(qdir)
    if not results:
        raise SystemExit('Chưa có file answers_*.jsonl nào. Chạy bước answer trước.')
    scored = score_all(contexts, results)
    table, csv_rows = summary_table(contexts, scored, results)
    n = len(contexts)
    h1 = sum(c['hit@1'] for c in contexts) / max(n, 1)
    h3 = sum(c['hit@3'] for c in contexts) / max(n, 1)
    ans = [answer_hits(c['context'], c['answer']) for c in contexts]
    a1, a3 = (sum(x[i] for x in ans) / max(n, 1) for i in (0, 1))

    md = [f'# So sánh LLM trên {n} câu hỏi', '',
          (
              f'Truy xuất (chung cho mọi LLM): **hit@1 = {h1:.2f}**, **hit@3 = {h3:.2f}** (đúng bài nguồn); '
              f'**ans@1 = {a1:.2f}**, **ans@3 = {a3:.2f}** (ngữ cảnh chứa đáp án chuẩn). '
              'Mọi LLM nhận cùng một ngữ cảnh, nên khác biệt bên dưới chỉ do LLM.'
          ), '', table, '',
          '- **contains**: đáp án chuẩn nằm trọn trong câu trả lời (chỉ số chính).',
          '- **recall**: tỉ lệ từ của đáp án chuẩn có trong câu trả lời (điểm từng phần).',
          '- **F1 / EM**: kiểu SQuAD, phạt câu trả lời dài dòng; LLM trả lời thành câu nên EM thường thấp.',
          '- **từ chối**: tỉ lệ câu LLM nói không biết dù ngữ cảnh có bài đúng (khi hit@3 = 1 là mất điểm).',
          '- **95% CI**: khoảng tin cậy bootstrap (10.000 lần, lấy mẫu lại theo câu hỏi).', '']
    if len(scored) > 1:
        md += ['## Kiểm định từng cặp (contains)', '',
               'McNemar chính xác trên cùng bộ câu hỏi; **đậm** = khác biệt có ý nghĩa (p Holm < 0,05).', '',
               pairwise_table(scored), '']
    md += ['## Cấu hình', '', '| LLM | repo | prompt | dtype | 4-bit | transformers | GPU |',
           '|---|---|---|---|---|---|---|']
    # GGUF (llama-server) đã lượng tử hoá sẵn -> không tính là "khác chế độ"
    if len({bool(m.get('4bit')) for _, m in results.values() if m.get('dtype') != 'gguf'}) > 1:
        md.insert(md.index('## Cấu hình'),
                  '> ⚠ Các LLM chạy khác chế độ (bf16 và 4-bit): chênh lệch có thể do lượng tử hoá.\n')
    for key, (_, meta) in results.items():
        quant = 'GGUF' if meta.get('dtype') == 'gguf' else ('có' if meta.get('4bit') else 'không')
        md.append(f"| {key} | {meta.get('repo', get_spec(key).repo)} | {meta.get('prompt', '')} | "
                  f"{meta.get('dtype', '')} | {quant} | {meta.get('transformers', '')} | "
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

    report, summary, review_csv = (qdir / f'compare_{name}{suffix}.{ext}'
                                   for name, ext in (('report', 'md'), ('summary', 'csv'), ('review', 'csv')))
    report.write_text('\n'.join(md), encoding='utf-8')
    _write_csv(summary, csv_rows)
    _write_csv(review_csv, review)
    print(table)
    print(f'  -> {report}\n  -> {summary}\n  -> {review_csv}  (cột dung_that để chấm tay)')
    return csv_rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:   # utf-8-sig: Excel đọc đúng tiếng Việt
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
