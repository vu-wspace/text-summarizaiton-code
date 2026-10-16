"""Cấu hình tập trung: đường dẫn, tên mô hình, tham số. Mọi module đọc từ đây, không hard-code.

Đường dẫn đổi được bằng biến môi trường (tiện cho Docker / Kaggle):
    RAG_LOAD_DIR, RAG_INDEX_DIR, RAG_QUESTIONS_DIR
    RAG_QDRANT_URL   (vd. http://localhost:6333; để trống = Qdrant nhúng tại OUTPUT_INDEX/qdrant)
"""
import os
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------- mô hình (theo vietnamese_llm_rag_evaluation.md)
# Embedding: vietnamese-bi-encoder (PhoBERT, 768 dim, tối đa 256 token, đầu vào PHẢI tách từ bằng pyvi)
EMB_MODEL = 'bkai-foundation-models/vietnamese-bi-encoder'
# Reranker: Vietnamese_Reranker (AITeamVN finetuned bge-reranker-v2-m3)
RERANK_MODEL = 'AITeamVN/Vietnamese_Reranker'
# Summarization
SUM_MODEL = 'VietAI/vit5-large-vietnews-summarization'
# LLM mặc định (danh sách ở llms.LLMS). Vistral-7B-Chat GGUF chạy qua llama-server -> phải bật server trước
# (xem README); không có server thì dùng --llm qwen2.5-3b.
DEFAULT_LLM = 'vistral-7b-chat'
# NER
NER_MODEL = 'NlpHUST/ner-vietnamese-electra-base'

# ----------------------------------------------------------------- tham số chunking
# vietnamese-bi-encoder chỉ nhận 256 token (gồm 2 token đặc biệt) -> chunk dài hơn sẽ bị cắt mất phần sau khi embed.
EMB_MAX_LENGTH = 256
CHUNK_SIZE = 254        # token (đếm trên văn bản đã tách từ, không gồm token đặc biệt)
CHUNK_OVERLAP = 30
NER_WINDOW = 400        # < 512 (giới hạn cứng của ELECTRA)

# Semantic chunking options (bật mặc định)
SEMANTIC_CHUNKING = True              # bật semantic chunking (dùng embedding similarity)
SEMANTIC_CHUNK_THRESHOLD = 0.75       # ngưỡng cosine similarity để tách chunk (0.7-0.85)
SEMANTIC_MIN_CHUNK_TOKENS = 64        # chunk nhỏ nhất (token)
SEMANTIC_MAX_CHUNK_TOKENS = CHUNK_SIZE  # chunk lớn nhất (token)

# ----------------------------------------------------------------- tham số retrieval (theo evaluation: top-5 -> rerank to 3)
TOP_K = 5                 # retriever lấy 5 chunk (giảm từ 20)
SEARCH_MODE = 'hybrid'    # dense | sparse | hybrid (dense + BM25, gộp RRF)
PREFETCH = 20             # hybrid: mỗi nhánh lấy 20 chunk trước khi gộp RRF
TOP_N_QA = 3              # re-ranker giữ 3 chunk cho nhánh hỏi đáp (theo evaluation)
N_ARTICLES = 3            # small-to-big: 5 chunk -> 3 bài khác nhau
CTX_BUDGET = 3000         # token (tokenizer của LLM) cho ngữ cảnh

QDRANT_URL = os.environ.get('RAG_QDRANT_URL', '')
COLLECTION = 'vietnews'

SUM_MAX_INPUT = 1024      # ViT5 cắt bài ở 1024 token
SUM_MAX_OUTPUT = 256
SUM_MIN_TEXT = 300        # "tóm tắt: <văn bản ≥ 300 ký tự>" -> tóm tắt chính văn bản đó


@dataclass(frozen=True)
class Paths:
    load_dir: Path = Path(os.environ.get('RAG_LOAD_DIR', 'OUTPUT_LOAD'))
    index_dir: Path = Path(os.environ.get('RAG_INDEX_DIR', 'OUTPUT_INDEX'))
    questions_dir: Path = Path(os.environ.get('RAG_QUESTIONS_DIR', 'OUTPUT_QUESTIONS'))

    # --- OUTPUT_LOAD
    @property
    def article(self):            # tập test đã chuẩn hoá = kho bài cho RAG
        return self.load_dir / 'data_article.json'

    # --- OUTPUT_INDEX
    @property
    def entities(self):
        return self.index_dir / 'entities.json'

    @property
    def vector_dir(self):          # tên cũ giữ nguyên để dùng lại chunks.pkl / embedds.npy đã có
        return self.index_dir / 'faiss_index'

    @property
    def chunks(self):
        return self.vector_dir / 'chunks.pkl'

    @property
    def embeddings(self):
        return self.vector_dir / 'embedds.npy'

    @property
    def qdrant_dir(self):          # Qdrant nhúng (khi không đặt RAG_QDRANT_URL)
        return self.index_dir / 'qdrant'

    # --- OUTPUT_QUESTIONS
    @property
    def questions(self):
        return self.questions_dir / 'questions.jsonl'
