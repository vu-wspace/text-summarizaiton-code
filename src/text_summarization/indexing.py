"""Xây kho: NER -> entities.json; chunk -> chunks.pkl; embed -> embedds.npy; index -> index.faiss.

Các bước chạy độc lập được (--steps), nhưng chunks.pkl, embedds.npy và index.faiss PHẢI cùng một
danh sách chunk (cùng thứ tự). Mỗi bước kiểm tra số lượng để không lệch âm thầm.
"""
import pickle

import numpy as np

from . import config as C
from .data import load_articles
from .store import build_index

SEPARATORS = ['\n\n', '\n', ' . ', ' ? ', ' ! ', ' … ', ' ; ', ' : ', ' , ', ' ', '']
STEPS = ('ner', 'chunk', 'embed', 'index')


def make_splitter(tokenizer, chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP):
    """Đo độ dài bằng TOKEN của bge-m3 (mô hình sẽ embed chunk), cắt ưu tiên tại ranh giới câu."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=SEPARATORS, keep_separator='end', strip_whitespace=True)


def header_chunks(articles):
    """Một chunk 'tiêu đề . sapo' / bài. KHÔNG bật khi đánh giá bằng abstract làm câu hỏi (lộ đáp án)."""
    from langchain_core.documents import Document

    return [Document(page_content=f"{d.metadata['title']} . {d.metadata['abstract']}",
                     metadata={**d.metadata, 'is_header': True}) for d in articles]


def build_chunks(articles, chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP, header=False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.EMB_MODEL)
    chunks = make_splitter(tok, chunk_size, chunk_overlap).split_documents(articles)
    lens = [len(tok.encode(c.page_content, add_special_tokens=False)) for c in chunks]
    print(f'Chunk: {len(chunks)} ({len(chunks) / len(articles):.2f}/bài) | token p50={np.percentile(lens, 50):.0f}'
          f' p90={np.percentile(lens, 90):.0f} max={max(lens)}' + (' | + header' if header else ''))
    return (header_chunks(articles) if header else []) + chunks


def run_index(paths, steps=STEPS, chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP, header=False):
    steps = set(steps)
    unknown = steps - set(STEPS)
    if unknown:
        raise ValueError(f'Bước không hợp lệ: {unknown}. Chọn trong {STEPS}')
    paths.faiss_dir.mkdir(parents=True, exist_ok=True)

    articles = None
    if steps & {'ner', 'chunk'}:
        articles = load_articles(paths.article)
        print(f'Đã nạp {len(articles)} bài')

    if 'ner' in steps:
        from .ner import run_ner
        run_ner(articles, paths.entities)

    chunks = None
    if 'chunk' in steps:
        chunks = build_chunks(articles, chunk_size, chunk_overlap, header)
        with open(paths.chunks, 'wb') as f:
            pickle.dump(chunks, f)
        print(f'  -> {paths.chunks}')
    elif steps & {'embed', 'index'}:
        with open(paths.chunks, 'rb') as f:
            chunks = pickle.load(f)

    vectors = None
    if 'embed' in steps:
        from .models import Models
        vectors = Models().embed_passages([c.page_content for c in chunks])
        np.save(paths.embeddings, vectors)
        print(f'  -> {paths.embeddings} {vectors.shape}')
    elif 'index' in steps:
        vectors = np.load(paths.embeddings)

    if 'index' in steps:
        if len(vectors) != len(chunks):
            raise ValueError(f'embedds.npy có {len(vectors)} vector nhưng chunks.pkl có {len(chunks)} chunk '
                             '— chạy lại --steps chunk,embed,index')
        n = build_index(vectors, paths.index)
        print(f'  -> {paths.index} ({n} vector)')
