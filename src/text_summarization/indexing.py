"""Xây kho: NER -> entities.json; chunk -> chunks.pkl; embed -> embedds.npy; index -> Qdrant (dense + BM25).

Các bước chạy độc lập được (--steps), nhưng chunks.pkl, embedds.npy và collection Qdrant PHẢI cùng một
danh sách chunk (cùng thứ tự). Mỗi bước kiểm tra số lượng để không lệch âm thầm.
"""
import pickle

import numpy as np

from . import config as C
from .data import load_articles
from .store import build_index, connect

SEPARATORS = ['\n\n', '\n', ' . ', ' ? ', ' ! ', ' … ', ' ; ', ' : ', ' , ', ' ', '']
STEPS = ('ner', 'chunk', 'embed', 'index')


def semantic_split(text, tokenizer, embedder=None, threshold=C.SEMANTIC_CHUNK_THRESHOLD,
                   min_tokens=C.SEMANTIC_MIN_CHUNK_TOKENS, max_tokens=C.SEMANTIC_MAX_CHUNK_TOKENS):
    """Semantic chunking: tách theo câu, tính cosine similarity giữa các câu liên tiếp,
    gộp các câu có độ tương đồng cao thành chunk có ý nghĩa hoàn chỉnh."""
    import re
    # Tách theo ranh giới câu tiếng Việt
    sentences = [s.strip() for s in re.split(r'(?<=[.?!…])\s+', text) if s.strip()]
    if not sentences:
        return [text] if text.strip() else []
    if len(sentences) == 1:
        return sentences

    # Đếm token từng câu
    sent_tokens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]

    # Nếu câu đầu đã vượt max_tokens, dùng recursive splitter
    chunks = []
    curr_chunk = [sentences[0]]
    curr_tokens = sent_tokens[0]

    # Nạp embedding để tính similarity nếu có
    if embedder is not None and len(sentences) > 1:
        import numpy as np
        vecs = embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        # Cosine similarity giữa câu i và i+1
        sims = np.sum(vecs[:-1] * vecs[1:], axis=1)

        for i in range(1, len(sentences)):
            s = sentences[i]
            t = sent_tokens[i]
            sim = sims[i - 1]

            # Tách chunk nếu:
            # 1. Similarity thấp hơn ngưỡng VÀ chunk hiện tại đã đủ dài (>= min_tokens)
            # 2. Hoặc thêm câu này làm vượt max_tokens
            should_split = (sim < threshold and curr_tokens >= min_tokens) or (curr_tokens + t > max_tokens)

            if should_split and curr_chunk:
                chunks.append(' '.join(curr_chunk))
                curr_chunk = [s]
                curr_tokens = t
            else:
                curr_chunk.append(s)
                curr_tokens += t
    else:
        # Fallback: gộp theo độ dài token
        for i in range(1, len(sentences)):
            s, t = sentences[i], sent_tokens[i]
            if curr_tokens + t > max_tokens and curr_chunk:
                chunks.append(' '.join(curr_chunk))
                curr_chunk = [s]
                curr_tokens = t
            else:
                curr_chunk.append(s)
                curr_tokens += t

    if curr_chunk:
        chunks.append(' '.join(curr_chunk))
    return chunks


def make_splitter(tokenizer, chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP):
    """Đo độ dài bằng TOKEN của bge-m3/vietnamese-bi-encoder, cắt ưu tiên tại ranh giới câu."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=SEPARATORS, keep_separator='end', strip_whitespace=True)


def header_chunks(articles):
    """Một chunk 'tiêu đề . sapo' / bài. KHÔNG bật khi đánh giá bằng abstract làm câu hỏi (lộ đáp án)."""
    from langchain_core.documents import Document

    return [Document(page_content=f"{d.metadata['title']} . {d.metadata['abstract']}",
                     metadata={**d.metadata, 'is_header': True}) for d in articles]


def build_chunks(articles, chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP, header=False,
                 semantic=C.SEMANTIC_CHUNKING):
    from langchain_core.documents import Document
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.EMB_MODEL)

    if semantic:
        print('Dùng SEMANTIC CHUNKING (embedding similarity)...')
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(C.EMB_MODEL)

        all_chunks = []
        for doc in articles:
            text = f"{doc.metadata.get('title', '')} . {doc.page_content}"
            splits = semantic_split(text, tok, embedder=embedder)
            for s in splits:
                all_chunks.append(Document(page_content=s, metadata=doc.metadata))
        chunks = all_chunks
    else:
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
    paths.vector_dir.mkdir(parents=True, exist_ok=True)

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
        n = build_index(connect(paths), vectors, chunks)
        print(f'  -> Qdrant {C.QDRANT_URL or paths.qdrant_dir} / {C.COLLECTION} ({n} point)')
