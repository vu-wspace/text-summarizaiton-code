import numpy as np

from text_summarization.rag.indexing import adjacent_cosine, group_sentences, split_sentences


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Giả lập tokenizer: mỗi từ là 1 token
        return text.split()


def test_split_single_sentence():
    text = 'Hà Nội là thủ đô của Việt Nam.'
    sentences, seg, n_tok = split_sentences(text, FakeTokenizer())
    assert sentences == seg == [text] and n_tok == [8]
    assert group_sentences(n_tok) == [[0]]


def test_group_by_length():
    text = 'Hà Nội là thủ đô. Đà Nẵng là thành phố biển. TP.HCM là trung tâm kinh tế.'
    sentences, _, n_tok = split_sentences(text, FakeTokenizer(), max_tokens=10)
    assert len(sentences) == 3
    assert group_sentences(n_tok, max_tokens=10) == [[0], [1], [2]]
    assert group_sentences(n_tok, max_tokens=12) == [[0, 1], [2]]


def test_group_splits_on_low_similarity_once_min_tokens_reached():
    n_tok = [2, 2, 2, 2]
    sims = [0.9, 0.1, 0.1]
    assert group_sentences(n_tok, sims, threshold=0.5, min_tokens=2, max_tokens=100) == [[0, 1], [2], [3]]
    # chunk chưa đủ min_tokens -> không tách dù cosine thấp
    assert group_sentences(n_tok, sims, threshold=0.5, min_tokens=5, max_tokens=100) == [[0, 1, 2], [3]]


def test_adjacent_cosine():
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert adjacent_cosine(vecs).tolist() == [1.0, 0.0]


def test_long_sentence_is_cut():
    text = ' '.join(['từ'] * 100) + '.'
    sentences, _, n_tok = split_sentences(text, FakeTokenizer(), max_tokens=30)
    assert len(sentences) >= 4 and all(t <= 30 for t in n_tok)
    assert ' '.join(sentences).split() == text.split()      # không mất từ nào


def test_tokens_counted_on_segmented_text():
    seen = []

    def segment(xs):
        seen.extend(xs)
        return [x.replace('Hà Nội', 'Hà_Nội').replace('Đà Nẵng', 'Đà_Nẵng') for x in xs]

    # đã tách từ: mỗi câu 2 token -> vừa 4 token -> gộp 1 chunk; câu giữ văn bản gốc
    sentences, seg, n_tok = split_sentences('Hà Nội đẹp. Đà Nẵng đẹp.', FakeTokenizer(), 4, segment)
    assert sentences == ['Hà Nội đẹp.', 'Đà Nẵng đẹp.'] and seg == ['Hà_Nội đẹp.', 'Đà_Nẵng đẹp.']
    assert n_tok == [2, 2] and group_sentences(n_tok, max_tokens=4) == [[0, 1]]
    assert seen == sentences
