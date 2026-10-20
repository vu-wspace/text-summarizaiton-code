"""Xây kho: NER -> entities.json; chunk -> chunks.pkl; embed -> embedds.npy; index -> Qdrant (dense + BM25).

Các bước chạy độc lập được (--steps), nhưng chunks.pkl, embedds.npy và collection Qdrant PHẢI cùng một
danh sách chunk (cùng thứ tự). Mỗi bước kiểm tra số lượng để không lệch âm thầm.
"""
import pickle
import re

import numpy as np

from ..core import config as C
from ..core.data import load_articles
from .store import build_index, connect

SEPARATORS = ['\n\n', '\n', ' . ', ' ? ', ' ! ', ' … ', ' ; ', ' : ', ' , ', ' ', '']
STEPS = ('ner', 'chunk', 'embed', 'index')


def _split_words(sent, n_tok, max_tokens):
    """Câu dài hơn max_tokens -> chia đều theo từ (chừa lề 20% vì số token mỗi từ không đều)."""
    if n_tok <= max_tokens:
        return [sent]
    words = sent.split()
    k = -(-n_tok * 5 // (max_tokens * 4))
    size = max(1, -(-len(words) // k))
    return [' '.join(words[i:i + size]) for i in range(0, len(words), size)]


SENT_RE = re.compile(r'(?<=[.?!…])\s+')


def split_sentences(text, tokenizer, max_tokens=C.SEMANTIC_MAX_CHUNK_TOKENS, segment=list):
    """-> (câu gốc, câu đã tách từ, số token từng câu). Câu dài hơn max_tokens được chia nhỏ.
    segment: hàm tách từ list[str] -> list[str] (PhoBERT cần); token đếm trên câu đã tách từ."""
    sentences = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    if not sentences:
        return [], [], []

    def count(seg):
        return [len(tokenizer.encode(s, add_special_tokens=False)) for s in seg]

    seg = segment(sentences)
    n_tok = count(seg)
    if any(t > max_tokens for t in n_tok):
        sentences = [p for s, t in zip(sentences, n_tok) for p in _split_words(s, t, max_tokens)]
        seg = segment(sentences)
        n_tok = count(seg)
    return sentences, seg, n_tok


def group_sentences(n_tok, sims=None, threshold=C.SEMANTIC_CHUNK_THRESHOLD,
                    min_tokens=C.SEMANTIC_MIN_CHUNK_TOKENS, max_tokens=C.SEMANTIC_MAX_CHUNK_TOKENS):
    """-> list nhóm chỉ số câu. Tách chunk khi (cosine câu i-1, i < ngưỡng VÀ chunk đã đủ min_tokens)
    HOẶC thêm câu sẽ vượt max_tokens. sims=None: chỉ gộp theo độ dài."""
    if not n_tok:
        return []
    groups, cur, cur_tok = [], [0], n_tok[0]
    for i in range(1, len(n_tok)):
        t = n_tok[i]
        low_sim = sims is not None and sims[i - 1] < threshold and cur_tok >= min_tokens
        if low_sim or cur_tok + t > max_tokens:
            groups.append(cur)
            cur, cur_tok = [i], t
        else:
            cur.append(i)
            cur_tok += t
    groups.append(cur)
    return groups


def adjacent_cosine(vecs):
    """Cosine giữa các vector liên tiếp (vector đã chuẩn hoá)."""
    return np.sum(vecs[:-1] * vecs[1:], axis=1)


def make_splitter(tokenizer, chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP):
    """Đo độ dài bằng TOKEN của vietnamese-bi-encoder, cắt ưu tiên tại ranh giới câu."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=SEPARATORS, keep_separator='end', strip_whitespace=True)


def header_chunks(articles):
    """Một chunk 'tiêu đề . sapo' / bài. KHÔNG bật khi đánh giá bằng abstract làm câu hỏi (lộ đáp án)."""
    from langchain_core.documents import Document

    return [Document(page_content=f"{d.metadata['title']} . {d.metadata['abstract']}",
                     metadata={**d.metadata, 'is_header': True}) for d in articles]


def build_chunks(articles, models, chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP, header=False,
                 semantic=C.SEMANTIC_CHUNKING, block=1000):
    """-> (chunks, segmented): segmented[i] là chunks[i] đã tách từ (pyvi chạy 1 lần, bước embed dùng lại).
    Semantic: mỗi khối `block` bài embed mọi câu trong một lô lớn (nhanh hơn nhiều so với từng bài)."""
    from langchain_core.documents import Document
    from transformers import AutoTokenizer

    from .models import word_segment

    tok = AutoTokenizer.from_pretrained(C.EMB_MODEL)
    if semantic:
        print(f'Semantic chunking (ngưỡng cosine {C.SEMANTIC_CHUNK_THRESHOLD}, '
              f'{C.SEMANTIC_MIN_CHUNK_TOKENS}-{chunk_size} token)...')
        from tqdm.auto import tqdm

        chunks, segmented = [], []
        bar = tqdm(total=len(articles), desc='Semantic chunk')
        for start in range(0, len(articles), block):
            docs = articles[start:start + block]
            parts = [split_sentences(f"{d.metadata.get('title', '')} . {d.page_content}", tok, chunk_size,
                                     word_segment) for d in docs]
            vecs = models.embed_passages([s for _, seg, _ in parts for s in seg], batch_size=256, segmented=True)
            pos = 0
            for doc, (sents, seg, n_tok) in zip(docs, parts):
                v, pos = vecs[pos:pos + len(sents)], pos + len(sents)
                sims = adjacent_cosine(v) if len(sents) > 1 else None
                for g in group_sentences(n_tok, sims, max_tokens=chunk_size):
                    chunks.append(Document(page_content=' '.join(sents[i] for i in g), metadata=doc.metadata))
                    segmented.append(' '.join(seg[i] for i in g))
            bar.update(len(docs))
        bar.close()
    else:
        chunks = make_splitter(tok, chunk_size, chunk_overlap).split_documents(articles)
        segmented = word_segment(c.page_content for c in chunks)

    lens = [len(tok.encode(s, add_special_tokens=False)) for s in segmented]
    print(f'Chunk: {len(chunks)} ({len(chunks) / len(articles):.2f}/bài) | token p50={np.percentile(lens, 50):.0f}'
          f' p90={np.percentile(lens, 90):.0f} max={max(lens)}' + (' | + header' if header else ''))
    if header:
        heads = header_chunks(articles)
        chunks, segmented = heads + chunks, word_segment(c.page_content for c in heads) + segmented
    return chunks, segmented


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

    models = None
    if steps & {'chunk', 'embed'}:
        from .models import Models
        models = Models()                                 # 1 bản embedding dùng chung cho chunk + embed

    chunks = segmented = None
    if 'chunk' in steps:
        chunks, segmented = build_chunks(articles, models, chunk_size, chunk_overlap, header)
        with open(paths.chunks, 'wb') as f:
            pickle.dump(chunks, f)
        print(f'  -> {paths.chunks}')
    elif steps & {'embed', 'index'}:
        with open(paths.chunks, 'rb') as f:
            chunks = pickle.load(f)

    vectors = None
    if 'embed' in steps:
        if segmented is not None:
            vectors = models.embed_passages(segmented, batch_size=128, segmented=True)
        else:
            vectors = models.embed_passages([c.page_content for c in chunks], batch_size=128)
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
