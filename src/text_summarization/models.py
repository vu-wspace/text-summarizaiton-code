"""Các mô hình nặng, nạp LƯỜI (chỉ nạp khi lần đầu dùng) để câu hỏi nào cũng không chiếm VRAM thừa.

    embed_query / embed_passages : bge-m3 (dense)          -> retriever + indexing
    rerank                       : bge-reranker-v2-m3      -> re-ranker
    summarize                    : ViT5-large vietnews     -> tóm tắt
    answer / chat / count_tokens : LLM chọn bằng --llm     -> sinh câu trả lời, sinh câu hỏi (llms.py)

Pipeline chỉ gọi qua các hàm này, nên khi test có thể thay cả lớp bằng mô hình giả.
"""
import sys

import numpy as np

from . import config as C
from .llms import DEFAULT_LLM, LLM


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

    # ----------------------------------------------------------- bge-m3
    def _embedder(self, all_gpus=False):
        if self._emb is None:
            self.log('nạp bge-m3')
            from FlagEmbedding import BGEM3FlagModel

            cuda = self.device == 'cuda'
            # Indexing: devices=None -> FlagEmbedding chia cho mọi GPU (Kaggle 2×T4). Query: 1 GPU.
            devices = None if (cuda and all_gpus) else (['cuda:0'] if cuda else ['cpu'])
            self._emb = BGEM3FlagModel(C.EMB_MODEL, use_fp16=cuda, devices=devices)
        return self._emb

    def embed_query(self, text):
        vec = self._embedder().encode([text], batch_size=1, max_length=C.EMB_MAX_LENGTH)['dense_vecs']
        return np.asarray(vec, dtype='float32')[0]

    def embed_passages(self, texts, batch_size=32):
        vecs = self._embedder(all_gpus=True).encode(
            list(texts), batch_size=batch_size, max_length=C.EMB_MAX_LENGTH)['dense_vecs']
        return np.asarray(vecs, dtype='float32')

    # ----------------------------------------------------------- re-ranker
    def rerank(self, query, texts):
        if self._rerank is None:
            self.log('nạp re-ranker')
            from sentence_transformers import CrossEncoder

            self._rerank = CrossEncoder(C.RERANK_MODEL, device=self.device)  # fp32: fp16 làm lệch điểm
        scores = self._rerank.predict([[query, t] for t in texts], batch_size=64, show_progress_bar=False)
        return [float(s) for s in scores]

    # ----------------------------------------------------------- ViT5
    def summarize(self, text):
        if self._sum is None:
            self.log('nạp ViT5')
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(C.SUM_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(C.SUM_MODEL).to(self.device).eval()
            self._sum = (tok, model)
        tok, model = self._sum
        with self.torch.no_grad():
            enc = tok('vietnews: ' + text, max_length=C.SUM_MAX_INPUT, truncation=True,
                      return_tensors='pt').to(self.device)
            out = model.generate(**enc, max_length=C.SUM_MAX_OUTPUT, num_beams=4,
                                 no_repeat_ngram_size=3, early_stopping=True)
        return tok.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # ----------------------------------------------------------- LLM (Qwen2.5 / Vi-Qwen2-RAG / Qwen3.5)
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
