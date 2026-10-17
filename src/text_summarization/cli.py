"""Một lệnh cho mọi việc:  python -m text_summarization <lệnh> [tuỳ chọn]

  Xây dựng
    load        tải VietNews, loại trùng, chuẩn hoá       -> OUTPUT_LOAD/
    index       NER + chunk + embed + Qdrant (dense+BM25) -> OUTPUT_INDEX/
  Sử dụng
    chat        chatbot trong terminal (qa / liệt kê thực thể / tóm tắt)
    ask         hỏi một câu rồi thoát
  Đánh giá
    questions   sinh N câu hỏi bằng LLM                  -> OUTPUT_QUESTIONS/questions.jsonl
    evaluate    chấm RAG đầu-cuối bằng file câu hỏi      -> questions_eval.jsonl
    retrieval   hit@k / MRR của dense, sparse, hybrid    -> retrieval_eval.jsonl / retrieval_report.md
  So sánh LLM (cùng câu hỏi, cùng ngữ cảnh)
    llms        liệt kê các LLM có sẵn
    contexts    truy xuất 1 lần cho mọi câu hỏi          -> contexts.jsonl
    answer      1 LLM trả lời trên contexts.jsonl        -> answers_<llm>.jsonl
    report      gộp + chấm mọi answers_*.jsonl           -> compare_report.md / .csv
    compare     contexts + answer (nhiều LLM) + report trong một lệnh
"""
import argparse
import faulthandler
import os
import sys
from pathlib import Path

# Windows: torch và thư viện khác có thể mang thêm một bản OpenMP -> cho phép cùng tồn tại (vô hại trên Linux).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
faulthandler.enable()   # crash ở tầng C vẫn in ra dòng Python gây lỗi

# Windows: stdin/stdout mặc định cp1252 khi không phải console gốc (pipe, Git Bash, terminal IDE) -> tiếng Việt
# nhập vào bị hỏng, router không nhận ra "Tóm tắt" / "Liệt kê". Ép UTF-8; byte lạ giữ dạng surrogate cho clean_input.
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdin.reconfigure(encoding='utf-8', errors='surrogateescape')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from . import config as C                       # noqa: E402
from .llms import DEFAULT_LLM, LLMS             # noqa: E402
from .text import clean_input                   # noqa: E402


def build_pipeline(paths, args):
    from .data import read_json
    from .models import Models
    from .pipeline import RAGPipeline
    from .store import QdrantStore, connect

    store = QdrantStore(connect(paths), paths.chunks, mode=args.search)
    articles = read_json(paths.article)
    entities = read_json(paths.entities)
    models = Models(llm=args.llm, llm_4bit=args.llm_4bit, verbose=args.verbose)
    print(f'Kho: {len(articles)} bài, {len(store)} chunk | tìm: {store.mode} | LLM: {args.llm}'
          f' | device={models.device}')
    return RAGPipeline(store, articles, entities, models, verbose=args.verbose)


# ----------------------------------------------------------------- xây dựng
def cmd_load(args, paths):
    from .data import HF, run_load
    run_load(paths, check=args.check, source=args.source or HF)


def cmd_index(args, paths):
    from .indexing import run_index
    run_index(paths, steps=[s.strip() for s in args.steps.split(',') if s.strip()],
              chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, header=args.header)


# ----------------------------------------------------------------- sử dụng
def cmd_chat(args, paths):
    rag = build_pipeline(paths, args)
    print('Sẵn sàng. Gõ "exit" để thoát.\n')
    while True:
        try:
            msg = clean_input(input('Bạn: ')).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {'exit', 'quit'}:
            break
        if msg:
            print('\n' + rag.format_reply(rag.handle(msg)) + '\n')


def cmd_ask(args, paths):
    rag = build_pipeline(paths, args)
    print(rag.format_reply(rag.handle(clean_input(args.question))))


# ----------------------------------------------------------------- đánh giá
def cmd_questions(args, paths):
    from .data import write_jsonl
    from .questions import evaluate, generate

    rag = build_pipeline(paths, args)
    print(f'Sinh {args.n} câu hỏi bằng {args.llm} (seed={args.seed})...')
    rows = generate(rag.store, rag.articles, rag.models, n=args.n, seed=args.seed)
    out = paths.questions_dir / args.name
    write_jsonl(rows, out)
    print(f'  -> {out}')
    if args.eval:
        _save_eval(evaluate(rag, rows), out)


def cmd_evaluate(args, paths):
    from .data import read_jsonl
    from .questions import evaluate, rescore, summary

    src = paths.questions_dir / args.name
    if args.rescore:                      # chỉ chấm lại cách so khớp, không chạy mô hình
        if not src.stem.endswith('_eval'):
            src = src.with_name(src.stem + '_eval' + src.suffix)
        rows = rescore(read_jsonl(src))
        print(summary(rows))
        _save_eval(rows, src, suffix='')
        return
    rag = build_pipeline(paths, args)
    _save_eval(evaluate(rag, read_jsonl(src)), src)


def cmd_retrieval(args, paths):
    from .data import read_jsonl, write_jsonl
    from .questions import KS, retrieval_eval, retrieval_table

    rag = build_pipeline(paths, args)
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    per_q, summ = retrieval_eval(rag, read_jsonl(paths.questions_dir / args.name), modes,
                                 rerank=not args.no_rerank)
    table = retrieval_table(summ)
    print('\n' + table)
    write_jsonl(per_q, paths.questions_dir / 'retrieval_eval.jsonl')
    report = paths.questions_dir / 'retrieval_report.md'
    report.write_text(f'# Đánh giá truy xuất ({args.name})\n\n'
                      'hit@k tính theo BÀI (các bài khác nhau, xếp theo chunk tốt nhất), '
                      f'top-{max(KS)} chunk từ Qdrant, stage=rerank là sau {C.RERANK_MODEL}. '
                      'retrieval_eval.jsonl ghi hạng bài nguồn từng câu '
                      f'(null = không có trong top-{max(KS)} chunk).\n\n{table}\n', encoding='utf-8')
    print(f'  -> {report}')


def _save_eval(rows, src, suffix='_eval'):
    from .data import write_jsonl
    stem = src.stem[:-len('_eval')] if src.stem.endswith('_eval') else src.stem
    out = src if not suffix else src.with_name(stem + suffix + src.suffix)
    write_jsonl(rows, out)
    print(f'  -> {out}')


# ----------------------------------------------------------------- so sánh LLM
def cmd_probe(args, paths):
    from .llms import LLM
    LLM(args.llm, load_4bit=args.llm_4bit, verbose=True).probe(args.question)


def cmd_llms(args, paths):
    import importlib.metadata as md
    try:
        have = md.version('transformers')
    except md.PackageNotFoundError:
        have = 'chưa cài'
    print(f'transformers trong môi trường này: {have}\n')
    for s in LLMS.values():
        need = f'≥ {s.min_transformers}'
        print(f'  {s.key:16s} {s.repo:36s} prompt={s.prompt:6s} transformers {need:12s} service={s.service}\n'
              f'  {"":16s} {s.note}')


def cmd_contexts(args, paths):
    from .compare import build_contexts, contexts_path
    from .data import read_jsonl, write_jsonl

    rag = build_pipeline(paths, args)
    rows = build_contexts(rag, read_jsonl(paths.questions_dir / args.name))
    write_jsonl(rows, contexts_path(paths.questions_dir))
    print(f'  -> {contexts_path(paths.questions_dir)}')


def cmd_answer(args, paths):
    from .compare import contexts_path, run_llm, save_answers, build_contexts
    from .data import read_jsonl, write_jsonl

    qdir = paths.questions_dir
    if not contexts_path(qdir).exists():
        rag = build_pipeline(paths, args)
        write_jsonl(build_contexts(rag, read_jsonl(qdir / args.name)), contexts_path(qdir))
        print(f'  -> {contexts_path(qdir)}')
        rag.models.unload_all()
        del rag
        _free_gpu()
    ctx = read_jsonl(contexts_path(qdir))

    rows, meta = run_llm(args.llm, ctx, load_4bit=args.llm_4bit, verbose=args.verbose)
    save_answers(qdir, args.llm, rows, meta)


def cmd_report(args, paths):
    from .compare import contexts_path, load_results, write_report
    from .data import read_jsonl

    keys = _keys(args.llms) if args.llms else None
    write_report(paths.questions_dir, read_jsonl(contexts_path(paths.questions_dir)),
                 load_results(paths.questions_dir, keys))


def cmd_compare(args, paths):
    """contexts (nếu chưa có / --refresh) -> answer cho từng LLM chạy được trong môi trường này -> report."""
    from .compare import contexts_path, load_results, run_llm, save_answers, write_report
    from .data import read_jsonl, write_jsonl
    from .llms import check_transformers, get_spec

    qdir = paths.questions_dir
    if args.refresh or not contexts_path(qdir).exists():
        from .compare import build_contexts
        rag = build_pipeline(paths, args)
        write_jsonl(build_contexts(rag, read_jsonl(qdir / args.name)), contexts_path(qdir))
        print(f'  -> {contexts_path(qdir)}')
        rag.models.unload_all()
        del rag                                       # giải phóng embedding + re-ranker trước khi nạp LLM
        _free_gpu()
    ctx = read_jsonl(contexts_path(qdir))

    skipped = []
    for key in _keys(args.llms):
        try:
            check_transformers(get_spec(key))
        except SystemExit as e:
            skipped.append((key, str(e)))
            continue
        print(f'\n=== {key} ===')
        rows, meta = run_llm(key, ctx, load_4bit=args.llm_4bit, verbose=args.verbose)
        save_answers(qdir, key, rows, meta)
    for key, why in skipped:
        print(f'\n! Bỏ qua {key}: {why}')
    print()
    write_report(qdir, ctx, load_results(qdir, _keys(args.llms)))


def _keys(s):
    keys = [k.strip() for k in s.split(',') if k.strip()]
    return list(LLMS) if keys == ['all'] else keys


def _free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------- parser
def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m text_summarization', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--load-dir', default=str(C.Paths().load_dir))
    ap.add_argument('--index-dir', default=str(C.Paths().index_dir))
    ap.add_argument('--questions-dir', default=str(C.Paths().questions_dir))
    sub = ap.add_subparsers(dest='cmd', required=True)

    def llm_opts(p):
        p.add_argument('--llm', default=DEFAULT_LLM, choices=list(LLMS), help='LLM sinh câu trả lời')
        p.add_argument('--llm-4bit', action=argparse.BooleanOptionalAction, default=C.LLM_4BIT,
                       help='nạp LLM 4-bit (GPU NVIDIA; mặc định bật, --no-llm-4bit để nạp bf16)')
        p.add_argument('--verbose', action='store_true', help='in từng bước đang chạy')
        p.add_argument('--search', default=C.SEARCH_MODE, choices=['dense', 'sparse', 'hybrid'],
                       help='kiểu truy xuất trên Qdrant (mặc định hybrid = dense + BM25, gộp RRF)')

    p = sub.add_parser('load', help='tải + làm sạch VietNews')
    p.add_argument('--check', action='store_true', help='chỉ kiểm tra kết quả chuẩn hoá')
    p.add_argument('--source', help='thư mục parquet đã tải sẵn (mặc định đọc từ Hugging Face)')
    p.set_defaults(func=cmd_load)

    p = sub.add_parser('index', help='NER + chunk + embed + Qdrant')
    p.add_argument('--steps', default='ner,chunk,embed,index')
    p.add_argument('--chunk-size', type=int, default=C.CHUNK_SIZE)
    p.add_argument('--chunk-overlap', type=int, default=C.CHUNK_OVERLAP,
                   help='chỉ dùng khi tắt SEMANTIC_CHUNKING (semantic chunking không chồng lấn)')
    p.add_argument('--header', action='store_true', help='thêm chunk "tiêu đề . sapo" mỗi bài')
    p.set_defaults(func=cmd_index)

    p = sub.add_parser('chat', help='chatbot trong terminal')
    llm_opts(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser('ask', help='hỏi một câu rồi thoát')
    p.add_argument('question')
    llm_opts(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser('questions', help='sinh câu hỏi đánh giá bằng LLM')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--seed', type=int, default=42, help='đổi seed để được bộ câu hỏi khác')
    p.add_argument('--name', default='questions.jsonl')
    p.add_argument('--eval', action='store_true', help='sinh xong thì chấm RAG luôn')
    llm_opts(p)
    p.set_defaults(func=cmd_questions)

    p = sub.add_parser('evaluate', help='chấm RAG đầu-cuối bằng file câu hỏi')
    p.add_argument('--name', default='questions.jsonl', help='file trong OUTPUT_QUESTIONS/')
    p.add_argument('--rescore', action='store_true',
                   help='chỉ chấm lại cột contains của file *_eval.jsonl (không nạp mô hình)')
    llm_opts(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser('retrieval', help='hit@k / MRR của dense, sparse, hybrid (không cần LLM)')
    p.add_argument('--name', default='questions.jsonl')
    p.add_argument('--modes', default='dense,sparse,hybrid')
    p.add_argument('--no-rerank', action='store_true', help='chỉ đo retriever, bỏ re-ranker')
    llm_opts(p)
    p.set_defaults(func=cmd_retrieval)

    p = sub.add_parser('llms', help='liệt kê các LLM có sẵn')
    p.set_defaults(func=cmd_llms)

    p = sub.add_parser('probe', help='chẩn đoán 1 LLM: logits NaN?, top-5 token, câu trả lời ngắn')
    p.add_argument('question', nargs='?', default='Thủ đô của Việt Nam là gì?')
    llm_opts(p)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser('contexts', help='truy xuất 1 lần cho mọi câu hỏi -> contexts.jsonl')
    p.add_argument('--name', default='questions.jsonl')
    llm_opts(p)
    p.set_defaults(func=cmd_contexts)

    p = sub.add_parser('answer', help='1 LLM trả lời trên contexts.jsonl -> answers_<llm>.jsonl')
    p.add_argument('--name', default='questions.jsonl', help='file câu hỏi trong OUTPUT_QUESTIONS/ (mặc định questions.jsonl)')
    llm_opts(p)
    p.set_defaults(func=cmd_answer)

    p = sub.add_parser('report', help='gộp + chấm mọi answers_*.jsonl')
    p.add_argument('--llms', default='', help='mặc định: mọi LLM đã có file answers_*.jsonl')
    p.set_defaults(func=cmd_report)

    p = sub.add_parser('compare', help='contexts + answer (nhiều LLM) + report')
    p.add_argument('--llms', default='all', help='danh sách cách nhau bởi dấu phẩy, hoặc "all"')
    p.add_argument('--name', default='questions.jsonl')
    p.add_argument('--refresh', action='store_true', help='truy xuất lại dù đã có contexts.jsonl')
    llm_opts(p)
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    paths = C.Paths(Path(args.load_dir), Path(args.index_dir), Path(args.questions_dir))
    args.func(args, paths)


if __name__ == '__main__':
    main()
