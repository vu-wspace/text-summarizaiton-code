"""Các mô hình nặng, nạp LƯỜI (chỉ nạp khi lần đầu dùng) để câu hỏi nào cũng không chiếm VRAM thừa.

    embed_query / embed_passages : vietnamese-bi-encoder (SentenceTransformer) -> retriever + indexing
    rerank                       : AITeamVN/Vietnamese_Reranker (CrossEncoder)        -> re-ranker
    summarize                    : ViT5-large vietnews                              -> tóm tắt
    answer / chat / count_tokens : LLM chọn bằng --llm (mặc định: Vi-Qwen2-3B-RAG) -> sinh câu trả lời

Pipeline chỉ gọi qua các hàm này, nên khi test có thể thay cả lớp bằng mô hình giả.
"""
import sys

import numpy as np

from . import config as C
from .llms import DEFAULT_LLM, LLM


def word_segment(texts):
    """PhoBERT (vietnamese-bi-encoder) cần đầu vào đã tách từ: 'Hà Nội' -> 'Hà_Nội'."""
    from pyvi import ViTokenizer

    return [ViTokenizer.tokenize(t) for t in texts]


class Models:
    def __init__(self, llm=DEFAULT_LLM, llm_4bit=False, verbose=False):
        import torch

        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
        self.llm = LLM(llm, load_4bit=llm_4bit, verbose=verbose)
        self._emb = self._rerank = self._sum = None

    def log(self, msg):
        if self.verbose:
            print(f'  · {msg}', file=sys.stderr, flush=True)

    def _half(self):
        return self.torch.bfloat16 if self.torch.cuda.is_bf16_supported() else self.torch.float16

    # ----------------------------------------------------------- vietnamese-bi-encoder (SentenceTransformer)
    def _embedder(self):
        if self._emb is None:
            self.log(f'nạp embedding: {C.EMB_MODEL}')
            from sentence_transformers import SentenceTransformer

            self._emb = SentenceTransformer(C.EMB_MODEL, device=self.device)
            self._emb.max_seq_length = C.EMB_MAX_LENGTH
        return self._emb

    def embed_query(self, text):
        return self.embed_passages([text], batch_size=1)[0]

    def embed_passages(self, texts, batch_size=32, segmented=False):
        """segmented=True: texts đã tách từ sẵn (word_segment) -> không tách lại."""
        emb = self._embedder()
        texts = list(texts) if segmented else word_segment(texts)
        vecs = emb.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype='float32')

    # ----------------------------------------------------------- re-ranker: Vietnamese_Reranker (CrossEncoder)
    def rerank(self, query, texts):
        if self._rerank is None:
            self.log(f'nạp re-ranker: {C.RERANK_MODEL}')
            from sentence_transformers import CrossEncoder

            # GPU: fp16 (BAAI khuyến nghị cho bge-reranker) -> 1,1 GB thay vì 2,3 GB, vừa VRAM còn lại khi LLM
            # chạy chung GPU 8 GB, và tránh lỗi "paging file is too small" trên Windows
            kw = {'model_kwargs': {'dtype': self._half()}} if self.device == 'cuda' else {}
            self._rerank = CrossEncoder(C.RERANK_MODEL, device=self.device, **kw)
        scores = self._rerank.predict([[query, t] for t in texts], batch_size=64, show_progress_bar=False)
        return [float(s) for s in scores]

    # ----------------------------------------------------------- ViT5
    def summarize(self, text):
        if self._sum is None:
            self.log('nạp ViT5')
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(C.SUM_MODEL)
            # T5 tràn số ở fp16 -> chỉ hạ xuống bf16 khi GPU hỗ trợ, còn lại giữ fp32
            bf16 = self.device == 'cuda' and self.torch.cuda.is_bf16_supported()
            model = AutoModelForSeq2SeqLM.from_pretrained(
                C.SUM_MODEL, dtype=self.torch.bfloat16 if bf16 else self.torch.float32).to(self.device).eval()
            self._sum = (tok, model)
        tok, model = self._sum
        with self.torch.no_grad():
            enc = tok('vietnews: ' + text, max_length=C.SUM_MAX_INPUT, truncation=True,
                      return_tensors='pt').to(self.device)
            out = model.generate(**enc, max_length=C.SUM_MAX_OUTPUT, num_beams=4,
                                 no_repeat_ngram_size=3, early_stopping=True)
        return tok.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # ----------------------------------------------------------- LLM (Vi-Qwen2-RAG / Qwen2.5 / Vistral-7B-Chat / ...)
    def answer(self, question, context, max_new_tokens=256):
        self.log(f'sinh câu trả lời bằng {self.llm.spec.key}')
        return self.llm.answer(question, context, max_new_tokens)

    def chat(self, system, user, max_new_tokens=256, repetition_penalty=1.05):
        return self.llm.chat(system, user, max_new_tokens, repetition_penalty)

    def count_tokens(self, text):
        return self.llm.count_tokens(text)

    def clip(self, text, max_tokens):
        """Cắt văn bản còn tối đa max_tokens token (tokenizer của LLM)."""
        return self.llm.clip(text, max_tokens)
