"""Luồng truy vấn: Router -> Retriever (Qdrant hybrid top-TOP_K) -> Re-ranker -> nhánh qa / list / tóm tắt."""
import sys
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import router as R
from .llms import NO_ANSWER
from .text import normalize_query, normalize_text

NOT_FOUND = 'Không tìm thấy bài báo phù hợp.'


@dataclass
class Reply:
    intent: str
    text: str
    sources: list = field(default_factory=list)   # guid các bài THỰC SỰ được dùng


class RAGPipeline:
    def __init__(self, store, articles, entities, models, verbose=False):
        """store: QdrantStore | articles: {guid: {title, abstract, article}} | entities: {guid: {TYPE: [..]}}
        models: Models (hoặc mô hình giả khi test)"""
        if '' in entities:
            raise ValueError('entities.json có key rỗng — chạy lại NER (key = guid)')
        self.store, self.articles, self.entities, self.models = store, articles, entities, models
        self.verbose = verbose

    def log(self, msg):
        if self.verbose:
            print(f'  · {msg}', file=sys.stderr, flush=True)

    # ----------------------------------------------------------- retriever + re-ranker
    def retrieve(self, query, k=C.TOP_K, top_n=C.TOP_N_QA, rerank=True, mode=None):
        """Qdrant lấy top-k chunk (dense / BM25 / hybrid RRF), re-ranker chấm lại từng cặp (câu hỏi, chunk),
        giữ top_n."""
        self.log(f'embedding + Qdrant ({mode or self.store.mode})')
        hits = self.store.search(self.models.embed_query(query), k, text=query, mode=mode)
        if not rerank or not hits:
            return hits[:top_n]
        self.log('re-rank')
        scores = self.models.rerank(query, [h.text for h in hits])
        order = np.argsort(-np.asarray(scores), kind='stable')
        return [hits[i] for i in order[:top_n]]

    # ----------------------------------------------------------- hỏi đáp
    def pick_articles(self, hits, n=C.N_ARTICLES):
        """Small-to-big: duyệt chunk theo thứ tự điểm, lấy n bài KHÁC NHAU (bỏ guid không có trong kho)."""
        out = []
        for h in hits:
            if h.guid not in out and h.guid in self.articles:
                out.append(h.guid)
                if len(out) == n:
                    break
        return out

    def build_context(self, guids, budget=C.CTX_BUDGET):
        """Ghép nguyên văn các bài trong ngân sách token. Bài đầu (điểm cao nhất) luôn được giữ, cắt bớt
        nếu quá dài; bài sau không còn chỗ thì bỏ. Trả về (ngữ cảnh, guid thực sự đã dùng)."""
        parts, used = [], []
        left = budget
        for g in guids:
            a = self.articles[g]
            text = f"{a['title']} . {a['article']}"
            n = self.models.count_tokens(text)
            if n > left:
                if parts:                       # đã có bài tốt hơn -> bỏ bài này
                    break
                text = self.models.clip(text, left)   # bài đầu quá dài -> cắt, không để ngữ cảnh rỗng
                n = left
            parts.append(f'[{len(parts) + 1}] {text}')
            used.append(g)
            left -= n
        return '\n\n'.join(parts), used

    def prepare(self, question):
        """Phần truy xuất của nhánh hỏi đáp (không gọi LLM): -> (câu hỏi đã chuẩn hoá, ngữ cảnh, guid đã dùng).
        Tách riêng để so sánh nhiều LLM trên CÙNG một ngữ cảnh."""
        q = normalize_query(question)
        guids = self.pick_articles(self.retrieve(q))
        if not guids:
            return q, '', []
        ctx, used = self.build_context(guids)
        return q, ctx, used

    def answer(self, question):
        q, ctx, used = self.prepare(question)
        if not used:
            return Reply(R.QA, NO_ANSWER, [])
        return Reply(R.QA, self.models.answer(q, ctx), used)

    # ----------------------------------------------------------- tóm tắt / liệt kê
    def top_article(self, topic):
        """Bài chứa chunk top-1 sau re-rank."""
        hits = self.retrieve(normalize_query(topic), top_n=C.TOP_K)
        guids = self.pick_articles(hits, n=1)
        return guids[0] if guids else None

    def summarize_topic(self, topic):
        g = self.top_article(topic)
        if g is None:
            return Reply(R.SUMMARIZE_TOPIC, NOT_FOUND)
        self.log('tóm tắt bằng ViT5')
        return Reply(R.SUMMARIZE_TOPIC, self.models.summarize(self.articles[g]['article']), [g])

    def summarize_text(self, text):
        return Reply(R.SUMMARIZE_TEXT, self.models.summarize(normalize_text(text, drop_caption=False)))

    def list_entities(self, payload):
        g = self.top_article(R.strip_type_words(payload))
        if g is None:
            return Reply(R.LIST, NOT_FOUND)
        ents = self.entities.get(g, {})              # thực thể đã tính sẵn lúc indexing
        types = R.wanted_types(payload)
        if types:
            ents = {t: v for t, v in ents.items() if t in types}
        return Reply(R.LIST, f"Bài: {self.articles[g]['title']}\n{R.format_entities(ents)}", [g])

    # ----------------------------------------------------------- điều phối
    def handle(self, message):
        intent, payload = R.route(message)
        if not payload:
            return Reply(intent, 'Bạn muốn hỏi gì? Hãy nhập nội dung cụ thể.')
        if intent == R.LIST:
            return self.list_entities(payload)
        if intent == R.SUMMARIZE_TOPIC:
            return self.summarize_topic(payload)
        if intent == R.SUMMARIZE_TEXT:
            return self.summarize_text(payload)
        return self.answer(payload)

    def format_reply(self, reply):
        label = 'Tóm tắt' if reply.intent.startswith('summarize') else 'Trả lời'
        lines = [f'[{reply.intent}] {label}: {reply.text}']
        lines += [f"  - [{g}] {self.articles[g]['title']}" for g in reply.sources]
        return '\n'.join(lines)
