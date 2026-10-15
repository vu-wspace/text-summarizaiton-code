from text_summarization.indexing import semantic_split


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Giả lập tokenizer: mỗi từ là 1 token
        return text.split()


class FakeEmbedder:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        import numpy as np
        # Trả về vector ngẫu nhiên hoặc cố định để test
        # Nếu texts chứa các câu khác nhau -> trả về ma trận shape (len(texts), 4)
        return np.ones((len(texts), 4), dtype=np.float32) / 2.0


def test_semantic_split_single_sentence():
    tok = FakeTokenizer()
    text = "Hà Nội là thủ đô của Việt Nam."
    chunks = semantic_split(text, tok)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_semantic_split_multiple_sentences():
    tok = FakeTokenizer()
    text = "Hà Nội là thủ đô. Đà Nẵng là thành phố biển. TP.HCM là trung tâm kinh tế."
    chunks = semantic_split(text, tok, max_tokens=10)
    assert len(chunks) >= 2


def test_semantic_split_with_embedder():
    tok = FakeTokenizer()
    emb = FakeEmbedder()
    text = "Câu một. Câu hai. Câu ba. Câu bốn."
    chunks = semantic_split(text, tok, embedder=emb, threshold=0.8, min_tokens=2, max_tokens=6)
    assert len(chunks) >= 1
    assert all(len(c.strip()) > 0 for c in chunks)
