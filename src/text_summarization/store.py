"""Kho vector FAISS: index.faiss (vector) + chunks.pkl (nội dung chunk, cùng thứ tự)."""
import os
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class Hit:
    guid: str
    title: str
    text: str
    score: float
    chunk_idx: int


def _faiss():
    # Windows: torch và faiss mỗi bên mang một bản OpenMP -> nạp torch TRƯỚC faiss, giới hạn 1 luồng.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    import faiss

    if os.name == 'nt':
        faiss.omp_set_num_threads(1)
    return faiss


def build_index(vectors, path):
    """Vector đã chuẩn hoá L2 + tích vô hướng (IndexFlatIP) = cosine, tìm vét cạn (chính xác)."""
    faiss = _faiss()
    vectors = np.ascontiguousarray(vectors, dtype='float32').copy()
    faiss.normalize_L2(vectors)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    faiss.write_index(idx, str(path))
    return idx.ntotal


class FaissStore:
    def __init__(self, index_path, chunks_path):
        self.faiss = _faiss()
        self.index = self.faiss.read_index(str(index_path))
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)          # list[langchain Document]
        if self.index.ntotal != len(self.chunks):
            raise ValueError(f'index.faiss ({self.index.ntotal}) và chunks.pkl ({len(self.chunks)}) không '
                             'khớp. Chạy lại: python -m text_summarization index --steps chunk,embed,index')

    def __len__(self):
        return len(self.chunks)

    def hit(self, i, score=0.0):
        c = self.chunks[i]
        return Hit(guid=str(c.metadata['guid']), title=c.metadata.get('title', ''),
                   text=c.page_content, score=float(score), chunk_idx=int(i))

    def search(self, vec, k):
        q = np.ascontiguousarray(np.asarray(vec, dtype='float32').reshape(1, -1)).copy()
        self.faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        return [self.hit(i, s) for s, i in zip(D[0], I[0]) if i != -1]
