"""Dữ liệu: tải VietNews, loại trùng, chuẩn hoá, xuất JSON; và các hàm đọc/ghi dùng chung."""
import json
import os
import re
from pathlib import Path

from .text import INVISIBLE, LOWER_CLS, normalize_text

HF = 'hf://datasets/nam194/vietnews/'   # đổi thành thư mục local nếu đã tải sẵn parquet
SPLITS = {
    'train': 'data/train-00000-of-00001-84acb79f6c6547a5.parquet',
    'validation': 'data/validation-00000-of-00001-210cc51bf3cdb90f.parquet',
    'test': 'data/test-00000-of-00001-123f98d55067eb7b.parquet',
}


# ----------------------------------------------------------------- đọc / ghi
def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_json(obj, path):
    """Ghi nguyên tử: ghi ra .tmp rồi đổi tên -> không hỏng file nếu bị ngắt giữa chừng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)


def load_articles(path):
    """data_article.json -> list[Document] (page_content = thân bài, metadata = guid/title/abstract)."""
    from langchain_core.documents import Document

    return [Document(page_content=v['article'], id=k,
                     metadata={'guid': k, 'title': v['title'], 'abstract': v['abstract']})
            for k, v in read_json(path).items()]


# ----------------------------------------------------------------- tải + làm sạch VietNews
def load_split(name, source=HF):
    import pandas as pd

    prefix = source if source.startswith('hf://') else str(Path(source)) + '/'
    df = pd.read_parquet(prefix + SPLITS[name])
    n0 = len(df)
    df = df[~df['title'].duplicated()]
    df = df[~df['abstract'].duplicated()]
    df = df.dropna()
    print(f'{name:10s}: {n0} -> {len(df)} bài (loại trùng title/abstract, bỏ NaN)')
    return df


def normalize_df(df):
    out = df.copy()
    out['title'] = out['title'].apply(lambda t: normalize_text(t, drop_caption=False))
    out['abstract'] = out['abstract'].apply(lambda t: normalize_text(t, drop_caption=False))
    out['article'] = out['article'].apply(normalize_text)
    return out


def df_to_json(df, path):
    data = {str(r.guid): {'title': r.title, 'abstract': r.abstract, 'article': r.article}
            for r in df.itertuples(index=False)}
    write_json(data, path)
    print(f'  -> {path} ({len(data)} bài)')


def check_normalize(df, n=2000, seed=0):
    """Đếm các mẫu lẽ ra phải hết sau chuẩn hoá (dùng `re` của Python, tránh RE2 của pandas 3)."""
    out = df['article'].sample(min(n, len(df)), random_state=seed).apply(normalize_text)
    count = lambda pat: int(out.apply(lambda t: len(re.findall(pat, t))).sum())
    print("Còn ' . ' + chữ thường:", count(rf' \. {LOWER_CLS}'))
    print('Còn ký tự vô hình    :', count(INVISIBLE))
    print("Còn '_'              :", count('_'))
    print('\nVí dụ:', out.iloc[0][:500])


def run_load(paths, check=False, source=HF):
    df_test = load_split('test', source)
    if check:
        check_normalize(df_test)
        return
    out = paths.load_dir
    df_to_json(load_split('train', source), out / 'data_train.json')
    df_to_json(load_split('validation', source), out / 'data_val.json')
    df_to_json(df_test, out / 'data_test.json')
    df_to_json(normalize_df(df_test), paths.article)
