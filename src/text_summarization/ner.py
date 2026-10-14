"""NER cả bài bằng NlpHUST/ner-vietnamese-electra-base (giới hạn cứng 512 token -> chia cửa sổ ≤ 400)."""
import re
from collections import Counter, defaultdict

from . import config as C
from .data import read_json, write_json


class NER:
    def __init__(self, model_name=C.NER_MODEL, max_tok=C.NER_WINDOW, device=None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.model_max_length = 512
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipe = pipeline('token-classification', model=model, tokenizer=self.tok,
                             aggregation_strategy='first', device=device)
        self.max_tok = max_tok

    def _n(self, s):
        return len(self.tok.encode(s, add_special_tokens=False))

    def _split_long(self, sent):
        """Câu dài hơn max_tok token thì cắt theo từ."""
        parts, cur = [], []
        for w in sent.split():
            cur.append(w)
            if self._n(' '.join(cur)) > self.max_tok:
                cur.pop()
                if cur:
                    parts.append(' '.join(cur))
                cur = [w]
        if cur:
            parts.append(' '.join(cur))
        return parts

    def windows(self, text):
        """Chia bài theo câu thành các cửa sổ ≤ max_tok token (không cắt giữa câu nếu tránh được)."""
        sents = [s.strip() for s in re.split(r'(?<= \.) ', text) if s.strip()]
        out, cur, cur_len = [], [], 0
        for s in sents:
            for p in (self._split_long(s) if self._n(s) > self.max_tok else [s]):
                m = self._n(p)
                if cur and cur_len + m > self.max_tok:
                    out.append(' '.join(cur))
                    cur, cur_len = [], 0
                cur.append(p)
                cur_len += m
        if cur:
            out.append(' '.join(cur))
        return out

    def __call__(self, text, min_score=0.6, drop_misc=False):
        wins = self.windows(text)
        if not wins:
            return {}
        ents = Counter()
        for win, res in zip(wins, self.pipe(wins, batch_size=16)):
            for e in res:
                name = win[e['start']:e['end']].strip(' ,.')   # lấy từ văn bản gốc (giữ nguyên dấu)
                typ = e['entity_group']
                if e['score'] < min_score or len(name) < 2 or (drop_misc and typ == 'MISCELLANEOUS'):
                    continue
                ents[(name, typ)] += 1
        grouped = defaultdict(list)
        for (name, typ), _ in sorted(merge_aliases(ents).items(), key=lambda x: -x[1]):
            grouped[typ].append(name)
        return dict(grouped)


def merge_aliases(ents):
    """Gộp 'Huy' vào 'Hồ Xuân Huy' nếu cùng loại và là phần cuối của tên dài hơn (cộng dồn số lần)."""
    merged = {}
    for (name, typ), cnt in sorted(ents.items(), key=lambda x: -len(x[0][0])):
        target = next((k for k in merged if k[1] == typ and k[0].endswith(' ' + name)), None)
        if target:
            merged[target] += cnt
        else:
            merged[(name, typ)] = cnt
    return merged


def run_ner(articles, path, save_every=1000):
    """NER từng bài, lưu {guid: {TYPE: [tên, ...]}}. Chạy tiếp được nếu bị ngắt (bỏ qua bài đã có)."""
    from tqdm.auto import tqdm

    entities = read_json(path) if path.exists() else {}
    entities.pop('', None)                         # dọn key rỗng từ lần chạy lỗi cũ
    todo = [d for d in articles if d.metadata['guid'] not in entities]
    print(f'NER: còn {len(todo)}/{len(articles)} bài')
    if todo:
        ner = NER()
        for i, d in enumerate(tqdm(todo, desc='NER'), 1):
            entities[d.metadata['guid']] = ner(d.page_content)
            if i % save_every == 0:
                write_json(entities, path)
        write_json(entities, path)
    missing = {d.metadata['guid'] for d in articles} - entities.keys()
    assert not missing, f'entities.json thiếu {len(missing)} bài'
    print(f'  -> {path} ({len(entities)} bài)')
    return entities
