import pytest
from conftest import FakeModels, FakeStore, hit

from text_summarization.pipeline import NO_ANSWER, NOT_FOUND, RAGPipeline


def make(articles, hits, **kw):
    return RAGPipeline(FakeStore(hits), articles, {'1': {'PERSON': ['Hồ Xuân Huy'], 'LOCATION': ['Đà Nẵng']}},
                       FakeModels(**kw))


def test_pick_articles_distinct_and_skip_missing(articles):
    rag = make(articles, [])
    hits = [hit('1'), hit('1'), hit('999'), hit('2'), hit('1'), hit('3'), hit('2')]
    assert rag.pick_articles(hits, 3) == ['1', '2', '3']


def test_build_context_first_article_too_long_is_clipped(articles):
    rag = make(articles, [])
    ctx, used = rag.build_context(['1', '2'], budget=10)    # bài 1 dài ~55 token > 10
    assert ctx.startswith('[1] ') and len(ctx.split()) == 11   # '[1]' + 10 token
    assert used == ['1']                                        # bài 2 không còn chỗ -> không tính là nguồn


def test_build_context_sources_only_what_fits(articles):
    rag = make(articles, [])
    _, used = rag.build_context(['1', '2', '3'], budget=120)  # mỗi bài ~55 token -> vừa 2 bài
    assert used == ['1', '2']


def test_answer_uses_context_and_reports_used_sources(articles):
    hits = [hit('2', 'đt'), hit('1', 'Huy lừa'), hit('3', 'ch')]
    rag = make(articles, hits, chat_reply='3,2 tỷ đồng')
    reply = rag.answer('Huy lừa bao nhiêu?')
    assert reply.text == '3,2 tỷ đồng'
    assert reply.sources[0] == '1'                     # re-ranker đưa chunk chứa "Huy lừa" lên đầu
    assert 'NGỮ CẢNH' in rag.models.calls[0] and 'CÂU HỎI: Huy lừa bao nhiêu?' in rag.models.calls[0]


def test_prepare_takes_articles_from_all_reranked_chunks(articles):
    # 3 chunk tốt nhất cùng thuộc bài 1 -> vẫn phải lấy được bài 2, 3 từ các chunk sau (5 chunk -> 3 bài)
    hits = [hit('1', 'Huy lừa'), hit('1', 'Huy lừa'), hit('1', 'Huy lừa'), hit('2', 'đt'), hit('3', 'ch')]
    _, _, used = make(articles, hits).prepare('Huy lừa ai?')
    assert used[0] == '1' and sorted(used) == ['1', '2', '3']


def test_answer_no_hits(articles):
    reply = make(articles, []).answer('gì đó')
    assert reply.text == NO_ANSWER and reply.sources == []


def test_list_entities_filters_type(articles):
    rag = make(articles, [hit('1', 'Huy')])
    r = rag.handle('Liệt kê địa điểm trong vụ Huy')
    assert r.intent == 'list' and 'Địa điểm: Đà Nẵng' in r.text and 'Người:' not in r.text
    assert r.sources == ['1']


def test_summarize_topic_uses_whole_article(articles):
    rag = make(articles, [hit('3', 'ch')])
    r = rag.handle('Tóm tắt vụ nữ quái')
    assert r.intent == 'summarize_topic' and r.text.startswith('TÓM TẮT: ch ch') and r.sources == ['3']


def test_summarize_topic_not_found(articles):
    assert make(articles, []).handle('Tóm tắt vụ không có').text == NOT_FOUND


def test_empty_payload(articles):
    assert 'Bạn muốn hỏi gì' in make(articles, []).handle('Tóm tắt').text


def test_empty_entity_key_rejected(articles):
    with pytest.raises(ValueError):
        RAGPipeline(FakeStore([]), articles, {'': {}}, FakeModels())
