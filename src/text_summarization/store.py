"""Kho vector Qdrant: collection 'vietnews' (vector dense vietnamese-bi-encoder + vector thưa BM25) + chunks.pkl.

Point id = vị trí chunk trong chunks.pkl. Ba kiểu tìm:
    dense  : cosine trên vector vietnamese-bi-encoder
    sparse : BM25 (TF tính ở sparse.py, IDF do Qdrant tính)
    hybrid : lấy top-PREFETCH của mỗi nhánh rồi gộp bằng Reciprocal Rank Fusion (RRF) ngay trong Qdrant

Kết nối: biến môi trường RAG_QDRANT_URL (vd. http://localhost:6333, chạy service qdrant trong
docker-compose) hoặc mặc định chế độ nhúng, lưu tại OUTPUT_INDEX/qdrant (không cần server).
"""
import pickle
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from . import config as C
from . import sparse

DENSE, SPARSE = 'dense', 'bm25'
MODES = ('dense', 'sparse', 'hybrid')


@dataclass
class Hit:
    guid: str
    title: str
    text: str
    score: float
    chunk_idx: int


def connect(paths):
    from qdrant_client import QdrantClient

    if C.QDRANT_URL:
        return QdrantClient(url=C.QDRANT_URL, timeout=120)
    paths.qdrant_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(paths.qdrant_dir))


def build_index(client, vectors, chunks, collection=C.COLLECTION, batch_size=512):
    """Tạo lại collection từ vector dense (embedds.npy) + nội dung chunk (để tính BM25)."""
    from qdrant_client import models as qm

    if len(vectors) != len(chunks):
        raise ValueError(f'{len(vectors)} vector nhưng {len(chunks)} chunk')
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={DENSE: qm.VectorParams(size=int(vectors.shape[1]), distance=qm.Distance.COSINE)},
        sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)})

    texts = [c.page_content for c in chunks]
    avg_len = sparse.avg_length(texts)
    for start in tqdm(range(0, len(chunks), batch_size), desc='Qdrant upsert'):
        points = []
        for i in range(start, min(start + batch_size, len(chunks))):
            idx, val = sparse.doc_vector(texts[i], avg_len)
            points.append(qm.PointStruct(
                id=i, payload={'guid': str(chunks[i].metadata['guid'])},
                vector={DENSE: np.asarray(vectors[i], dtype='float32').tolist(),
                        SPARSE: qm.SparseVector(indices=idx, values=val)}))
        client.upsert(collection, points, wait=True)
    return client.count(collection, exact=True).count


class QdrantStore:
    def __init__(self, client, chunks_path, collection=C.COLLECTION, mode=C.SEARCH_MODE,
                 prefetch=C.PREFETCH):
        if mode not in MODES:
            raise ValueError(f'mode={mode!r}, chọn trong {MODES}')
        self.client, self.collection, self.mode, self.prefetch = client, collection, mode, prefetch
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)          # list[langchain Document]
        if not client.collection_exists(collection):
            where = f'Qdrant server {C.QDRANT_URL}' if C.QDRANT_URL else 'Qdrant nhúng (OUTPUT_INDEX/qdrant)'
            raise ValueError(f'{where} chưa có collection {collection!r}. Index dựng bằng Docker thì đặt '
                             'RAG_QDRANT_URL=http://localhost:6333; nếu không, chạy: vietnews index --steps index')
        self.dim = client.get_collection(collection).config.params.vectors[DENSE].size
        n = client.count(collection, exact=True).count
        if n != len(self.chunks):
            where = f'Qdrant server {C.QDRANT_URL}' if C.QDRANT_URL else 'Qdrant nhúng (OUTPUT_INDEX/qdrant)'
            raise ValueError(f'{where} ({n}) và chunks.pkl ({len(self.chunks)}) không khớp. Nếu index đã dựng '
                             'bằng Docker (service qdrant) thì đặt RAG_QDRANT_URL=http://localhost:6333; nếu không, '
                             'nạp lại: vietnews index --steps index (hoặc chunk,embed,index nếu đổi mô hình)')

    def __len__(self):
        return len(self.chunks)

    def hit(self, i, score=0.0):
        c = self.chunks[i]
        return Hit(guid=str(c.metadata['guid']), title=c.metadata.get('title', ''),
                   text=c.page_content, score=float(score), chunk_idx=int(i))

    def search(self, vec, k, text=None, mode=None):
        """vec: vector dense của câu hỏi; text: câu hỏi (cho BM25). Thiếu text -> chỉ dense."""
        from qdrant_client import models as qm

        mode = mode or self.mode
        sv = None
        if text and mode != 'dense':
            idx, val = sparse.query_vector(text)
            sv = qm.SparseVector(indices=idx, values=val) if idx else None
        if sv is None:
            mode = 'dense'
        dense_q = np.asarray(vec, dtype='float32').reshape(-1).tolist()
        if len(dense_q) != self.dim:
            raise ValueError(f'Vector câu hỏi {len(dense_q)} chiều nhưng collection {self.dim} chiều: index dựng bằng '
                             f'mô hình embedding khác {C.EMB_MODEL}. Chạy lại: vietnews index --steps chunk,embed,index')

        if mode == 'dense':
            res = self.client.query_points(self.collection, query=dense_q, using=DENSE, limit=k)
        elif mode == 'sparse':
            res = self.client.query_points(self.collection, query=sv, using=SPARSE, limit=k)
        else:
            n = max(k, self.prefetch)
            res = self.client.query_points(
                self.collection, limit=k, query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                prefetch=[qm.Prefetch(query=dense_q, using=DENSE, limit=n),
                          qm.Prefetch(query=sv, using=SPARSE, limit=n)])
        return [self.hit(p.id, p.score) for p in res.points]
