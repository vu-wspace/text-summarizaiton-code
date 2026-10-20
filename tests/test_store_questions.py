import pickle
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeModels
from langchain_core.documents import Document

from text_summarization.core.data import read_jsonl
from text_summarization.evaluation.questions import (
    article_rank,
    check,
    generate,
    has_anchor,
    parse_json,
    pick_chunks,
    rescore,
    retrieval_eval,
)
from text_summarization.rag import sparse
from text_summarization.rag.indexing import run_index
from text_summarization.rag.store import QdrantStore, build_index

DATA = Path(__file__).parent / 'data'


TEXT = 'Từ tháng 10/2016 , Hồ Xuân Huy lừa nhiều người với tổng số tiền 3,2 tỷ đồng . ' * 6


def make_chunks(n=30):
    chunks = [Document(page_content=TEXT, metadata={'guid': str(i // 3), 'title': f'T{i // 3}'})
              for i in range(n)]
    # chunk 13 là chunk duy nhất nói về "Vinamilk" -> BM25 phải tìm ra nó
    if n > 13:
        chunks[13] = Document(
            page_content='Cổ đông F & N Dairy Investments của Singapore mua thêm cổ phần Vinamilk .',
            metadata={'guid': '4', 'title': 'T4'})
    return chunks


@pytest.fixture
def client():
    from qdrant_client import QdrantClient
    return QdrantClient(':memory:')


@pytest.fixture
def store(tmp_path, client):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(30, 16)).astype('float32')
    chunks = make_chunks()
    with open(tmp_path / 'chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    assert build_index(client, vecs, chunks) == 30
    s = QdrantStore(client, tmp_path / 'chunks.pkl')
    s.vecs = vecs
    return s


# ----------------------------------------------------------------- Qdrant
def test_dense_search_returns_itself_first(store):
    hits = store.search(store.vecs[7] * 5, k=3, mode='dense')      # độ lớn không ảnh hưởng: cosine
    assert hits[0].chunk_idx == 7 and hits[0].guid == '2' and abs(hits[0].score - 1) < 1e-4
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_sparse_search_finds_keyword(store):
    hits = store.search(store.vecs[0], k=3, text='Ai mua cổ phần Vinamilk?', mode='sparse')
    assert hits[0].chunk_idx == 13 and hits[0].guid == '4'


def test_hybrid_combines_both(store):
    # dense chỉ về chunk 7, BM25 chỉ về chunk 13: RRF giữ cả hai ở đầu danh sách
    hits = store.search(store.vecs[7], k=5, text='Vinamilk', mode='hybrid')
    assert {7, 13} <= {h.chunk_idx for h in hits[:2]}


def test_hybrid_without_text_falls_back_to_dense(store):
    assert store.search(store.vecs[7], k=1)[0].chunk_idx == 7
    assert store.search(store.vecs[7], k=1, text='?!')[0].chunk_idx == 7    # không có từ nào -> dense


def test_mismatch_raises(tmp_path, client, store):
    build_index(client, np.ones((5, 16), dtype='float32'), make_chunks(5))
    with pytest.raises(ValueError, match='không khớp'):
        QdrantStore(client, tmp_path / 'chunks.pkl')


def test_bm25_vectors():
    idx, val = sparse.doc_vector('Hồ Xuân Huy , Hồ Xuân Huy', avg_len=5)
    assert len(idx) == len(set(idx)) and all(v > 0 for v in val)
    assert 'hồ_xuân' in sparse.terms('HỒ Xuân')                            # lower + bigram âm tiết
    assert sparse.query_vector('Hồ Xuân')[1] == [1.0, 1.0, 1.0]


def test_invalid_step():
    with pytest.raises(ValueError, match='không hợp lệ'):
        run_index(None, steps=['ner', 'faiss'])


def test_article_rank():
    from conftest import hit
    hits = [hit('5'), hit('5'), hit('2'), hit('9')]
    assert article_rank(hits, '5') == 1 and article_rank(hits, '9') == 3 and article_rank(hits, '7') is None


def test_retrieval_eval(store, articles):
    from text_summarization.rag.pipeline import RAGPipeline

    class M(FakeModels):
        def embed_query(self, text):
            return store.vecs[7]
    rag = RAGPipeline(store, articles, {}, M())
    rows = [{'id': 1, 'question': 'Ai mua cổ phần Vinamilk?', 'guid': '4'}]
    per_q, summ = retrieval_eval(rag, rows, modes=('dense', 'sparse'))
    assert per_q[0]['sparse/retriever'] == 1
    by = {(r['mode'], r['stage']): r for r in summ}
    assert by[('sparse', 'retriever')]['hit@1'] == 1.0 and by[('sparse', 'retriever')]['MRR'] == 1.0


# ----------------------------------------------------------------- sinh câu hỏi
def test_parse_json():
    assert parse_json('```json\n{"question": "Q dài?", "answer": "A"}\n```') == ('Q dài?', 'A')
    assert parse_json('không có json') is None
    assert parse_json('{"question": "", "answer": "A"}') is None


def test_has_anchor():
    assert has_anchor('Hồ Xuân Huy lừa bao nhiêu?')
    assert has_anchor('Khi nào SCIC chào bán cổ phần?')
    assert has_anchor('Vụ sạt lở làm chết bé gái 5 tuổi ở đâu?')
    assert not has_anchor('Tên người mẫu được đề cử cho giải thưởng?')       # câu 8 trong kết quả thật
    assert not has_anchor("Tên người được mệnh danh là nhạc sĩ tạo 'hit'?")   # câu 6


def test_check_reasons():
    ctx = 'Hồ Xuân Huy lừa tổng số tiền 3,2 tỷ đồng .'
    assert check('Hồ Xuân Huy lừa được bao nhiêu tiền?', '3,2 tỷ đồng', ctx) is None
    assert 'phụ thuộc' in check('Theo bài báo, Huy lừa bao nhiêu tiền?', '3,2 tỷ đồng', ctx)
    assert 'bịa' in check('Hồ Xuân Huy lừa được bao nhiêu tiền?', '5 tỷ đồng', ctx)
    assert 'lộ' in check('Hồ Xuân Huy lừa 3,2 tỷ đồng của những ai?', '3,2 tỷ đồng', ctx)
    assert 'có/không' in check('Hồ Xuân Huy có lừa 3,2 tỷ đồng không?', '3,2 tỷ đồng', ctx)


def test_check_real_bad_questions():
    # câu 6 trong bộ câu hỏi thật: đáp án 'Tháng 6 này' gần như nằm sẵn trong câu hỏi
    ctx = 'Tháng 6 này , sinh viên trường đã tham gia học kỳ quốc tế tại Đại học Kobe ( Nhật Bản ) .'
    q = 'UEF có chương trình học kỳ quốc tế với trường Đại học Kobe ở Nhật Bản vào tháng 6?'
    assert 'lộ' in check(q, 'Tháng 6 này', ctx)
    assert check('Sinh viên UEF học kỳ quốc tế tại Đại học Kobe vào thời gian nào?', 'Tháng 6 này', ctx) is None


def test_pick_chunks_one_per_article(store):
    idx = pick_chunks(store.chunks, n=5, seed=1)
    assert len(idx) == 5 and len({store.chunks[i].metadata['guid'] for i in idx}) == 5
    assert idx == pick_chunks(store.chunks, n=5, seed=1)          # cùng seed -> cùng bộ


def test_generate_filters_and_fills(store):
    replies = iter(['{"question": "Theo bài báo, ai lừa?", "answer": "Huy"}', 'rác',
                    '{"question": "Hồ Xuân Huy lừa được tổng cộng bao nhiêu tiền", "answer": "3,2 tỷ đồng"}'] * 10)
    rows = generate(store, {}, FakeModels(chat_reply=lambda: next(replies)), n=3, seed=1)
    assert len(rows) == 3 and all(r['question'].endswith('?') for r in rows)
    assert len({r['guid'] for r in rows}) == 3


# ----------------------------------------------------------------- chấm lại kết quả thật của bạn
def test_rescore_real_results():
    rows = read_jsonl(DATA / 'sample_eval.jsonl')
    old = sum(r['contains'] for r in rows)
    new = rescore(rows)
    fixed = {r['id'] for r, n in zip(rows, new) if n['contains'] and not r['contains']}
    assert old == 7 and fixed == {1, 7}                       # 10-11 ~ 10/11/2017 ; ( Anh Quốc ) ~ (Anh Quốc)
    assert not next(n for n in new if n['id'] == 8)['contains']   # Minh Tú ≠ Minh Anh: vẫn sai


def test_rescore_real_results_v2():
    # bộ 10 câu sinh lại bằng code mới: câu 2 '14,43 %' vs '14,43%' là lỗi chấm, phải được tính đúng
    rows = rescore(read_jsonl(DATA / 'sample_eval_v2.jsonl'))
    by_id = {r['id']: r['contains'] for r in rows}
    assert by_id[2] is True
    assert by_id[1] is False      # Qwen trả lời F & N dù bài ghi rõ "không phải F & N": sai thật
