"""Cấu hình tập trung: đường dẫn, tên mô hình, tham số. Mọi module đọc từ đây, không hard-code.

Đường dẫn đổi được bằng biến môi trường (tiện cho Docker / Kaggle):
    RAG_LOAD_DIR, RAG_INDEX_DIR, RAG_QUESTIONS_DIR
"""
import os
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------- mô hình
NER_MODEL = 'NlpHUST/ner-vietnamese-electra-base'
EMB_MODEL = 'BAAI/bge-m3'
RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'
SUM_MODEL = 'VietAI/vit5-large-vietnews-summarization'
LLM_MODEL = 'Qwen/Qwen2.5-3B-Instruct'

# ----------------------------------------------------------------- tham số
CHUNK_SIZE = 256        # token của bge-m3
CHUNK_OVERLAP = 40
EMB_MAX_LENGTH = 512    # đủ cho chunk 256 token + câu hỏi
NER_WINDOW = 400        # < 512 (giới hạn cứng của ELECTRA)

TOP_K = 20              # FAISS lấy 20 chunk
TOP_N_QA = 10           # re-ranker giữ 10 chunk cho nhánh hỏi đáp
N_ARTICLES = 3          # small-to-big: 10 chunk -> 3 bài khác nhau
CTX_BUDGET = 3000       # token (tokenizer Qwen) cho ngữ cảnh

SUM_MAX_INPUT = 1024    # ViT5 cắt bài ở 1024 token
SUM_MAX_OUTPUT = 256
SUM_MIN_TEXT = 300      # "tóm tắt: <văn bản ≥ 300 ký tự>" -> tóm tắt chính văn bản đó


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
    def faiss_dir(self):
        return self.index_dir / 'faiss_index'

    @property
    def chunks(self):
        return self.faiss_dir / 'chunks.pkl'

    @property
    def embeddings(self):
        return self.faiss_dir / 'embedds.npy'

    @property
    def index(self):
        return self.faiss_dir / 'index.faiss'

    # --- OUTPUT_QUESTIONS
    @property
    def questions(self):
        return self.questions_dir / 'questions.jsonl'
