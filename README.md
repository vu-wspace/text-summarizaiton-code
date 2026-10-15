# VietNews: Tóm tắt + NER + Hỏi đáp RAG (Vietnamese-Optimized)

Hệ thống hỏi đáp tin tức tiếng Việt trên tập test **VietNews (22.492 bài)** với pipeline RAG tối ưu cho tiếng Việt: **Semantic Chunking** → **Vietnamese Bi-Encoder** → **Qdrant Hybrid Search (RRF)** → **Vietnamese Reranker** → **Vietnamese LLM** (Vistral-7B-Chat / Vi-Qwen2-3B-RAG).

---

## ✨ Tính năng chính

| Module | Mô tả |
|--------|-------|
| **Hỏi đáp (QA)** | **Semantic Chunking** + Hybrid search trên Qdrant (dense `vietnamese-bi-encoder` + BM25, gộp RRF) top-5 → `AITeamVN/Vietnamese_Reranker` → top-3 bài → LLM tiếng Việt (mặc định: `vistral-7b-chat`) |
| **Semantic Chunking** | Tách văn bản theo câu, đếm token + tính cosine similarity giữa các câu liên tiếp, gộp thành chunk hoàn chỉnh về mặt ý nghĩa (`SEMANTIC_CHUNKING = True`) |
| **Tóm tắt chủ đề** | Tìm bài liên quan nhất → ViT5-large-summarization |
| **Trích xuất thực thể (NER)** | Tra thực thể đã NER sẵn bằng ELECTRA (cửa sổ ≤ 400 token, gộp tên) |
| **So sánh LLM** | Chạy & so sánh các LLM tiếng Việt (Vistral-7B-Chat, Vi-Qwen2-3B-RAG, Qwen2.5-3B, PhoGPT-4B) trên cùng một ngữ cảnh |

---

## 🏗️ Kiến trúc Pipeline RAG Tiếng Việt

```
           ┌──────────────────────────────────────────────────┐
           │                   Câu hỏi / Input                │
           └─────────────────────────┬────────────────────────┘
                                     │
                          1. Router (Intent & Query)
                                     │
           ┌─────────────────────────┴────────────────────────┐
           │            2. Hybrid Search (Qdrant)              │
           ├─────────────────────────┬────────────────────────┤
           │  Dense: vietnamese-bi-   │  Sparse: BM25 (unigram  │
           │  encoder (cosine)       │  + bigram âm tiết Việt)│
           └─────────────────────────┼────────────────────────┘
                                     │
                             Reciprocal Rank Fusion (RRF)
                                     │ Top-5 chunks
           ┌─────────────────────────┴────────────────────────┐
           │        3. Re-ranker (Vietnamese_Reranker)        │
           └─────────────────────────┬────────────────────────┘
                                     │ Top-3 chunks -> Articles
           ┌─────────────────────────┴────────────────────────┐
           │      4. Context Assembly & Token Clipping         │
           └─────────────────────────┬────────────────────────┘
                                     │
           ┌─────────────────────────┴────────────────────────┐
           │   5. Generator (Vistral-7B-Chat / Qwen2-RAG)    │
           └──────────────────────────────────────────────────┘
```

---

## 🇻🇳 Bảng so sánh Top 3 Vietnamese LLMs cho RAG

Dựa trên kết quả đánh giá thực nghiệm (`vietnamese_llm_rag_evaluation.md`):

| Rank | Model | Base | Size | Context | Faithfulness | Điểm mạnh |
|------|-------|------|------|---------|--------------|-----------|
| 🥇 1 | **Vistral-7B-Chat** | Mistral-7B | 7B | **32K** | **97.5%** | Chất lượng câu trả lời tốt nhất, bám sát ngữ cảnh, không ảo giác |
| 🥈 2 | **Vinallama-2.7B-Chat** | Llama-2 | 2.7B | 4K | ~88.0% | Rất nhẹ, chạy nhanh (2.5x), vừa GPU 4GB/Edge device |
| 🥉 3 | **Vietnamese-Llama2-7B-40GB** | Llama-2 | 7B | 4K | ~82.0% | Train trên 40GB corpus tiếng Việt, tri thức rộng |

---

## 📁 Cấu trúc dự án

```
text_summarization/
├── src/text_summarization/
│   ├── config.py      # Đường dẫn, model names, tham số (top-k, semantic chunking...)
│   ├── text.py        # Chuẩn hoá VietNews, làm sạch input terminal, so khớp đáp án
│   ├── data.py        # Tải + loại trùng VietNews, đọc/ghi JSON
│   ├── ner.py         # NER cả bài (ELECTRA, cửa sổ ≤ 400 token)
│   ├── indexing.py    # Semantic chunking (embedding similarity) -> embed -> Qdrant
│   ├── sparse.py      # Vector thưa BM25 (unigram + bigram âm tiết), IDF do Qdrant tính
│   ├── store.py       # QdrantStore: dense / sparse / hybrid (RRF) + chunks.pkl
│   ├── llms.py        # Danh sách LLM + prompt riêng từng mô hình
│   ├── models.py      # vietnamese-bi-encoder, Vietnamese_Reranker, ViT5, LLM (nạp lười)
│   ├── router.py      # Phân loại intent: qa / list / summarize_topic / summarize_text
│   ├── pipeline.py    # Luồng RAG: retrieve → re-rank → context → answer
│   ├── questions.py   # Sinh câu hỏi đánh giá + chấm hit@k, MRR, contains
│   ├── metrics.py     # Contains, recall, F1, EM, tỉ lệ từ chối
│   ├── compare.py     # So sánh nhiều LLM trên cùng ngữ cảnh
│   └── cli.py         # Terminal CLI: vietnews <lệnh> (UTF-8 Windows auto-fix)
├── tests/             # 45 unit tests (chạy qua pytest)
├── OUTPUT_LOAD/       # data_article.json
├── OUTPUT_INDEX/      # entities.json, faiss_index/{chunks.pkl, embedds.npy}, qdrant/
├── OUTPUT_QUESTIONS/  # questions.jsonl, questions_eval.jsonl, retrieval_report.md
├── evaluation_results/# Kết quả benchmark JSON
├── evaluate_vietnamese_rag.py     # Script benchmark LLM RAG
├── vietnamese_llm_rag_evaluation.md # Báo cáo chi tiết benchmark
├── Dockerfile, Dockerfile.llm5, docker-compose.yml, pyproject.toml, requirements.txt
```

---

## ⚡ Chạy trực tiếp (`uv`)

```bash
# 1. Cài đặt môi trường
uv sync

# 2. Xây kho index (Semantic Chunking + Embeddings)
uv run vietnews index --steps chunk,embed,index

# 3. Chat tương tác trong Terminal
uv run vietnews chat

# 4. Chạy câu hỏi đơn lẻ
uv run vietnews ask "Thủ đô của Việt Nam là gì?"

# 5. Chạy unit tests (45 tests passed)
uv run pytest
```

---

## 🐳 Chạy bằng Docker (GPU)

```bat
chcp 65001
docker compose build
docker compose up -d qdrant
docker compose run --rm rag index --steps index
docker compose run --rm rag chat
docker compose run --rm rag ask "Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam"
```

---

## 📋 Các lệnh CLI (`vietnews`)

| Lệnh | Mô tả | Tham số hữu ích |
|---|---|---|
| `load` | Tải VietNews, loại trùng, chuẩn hoá | `--check`, `--source DIR` |
| `index` | NER + Semantic Chunking + Embed + Qdrant | `--steps ner,chunk,embed,index` |
| `chat` | Chatbot hỏi đáp, tóm tắt, trích xuất thực thể | `--llm <key>`, `--search hybrid\|dense\|sparse` |
| `ask "..."` | Hỏi một câu rồi thoát | `--llm <key>`, `--verbose` |
| `questions` | Sinh N câu hỏi bằng LLM | `--n 10`, `--eval` |
| `evaluate` | Chấm RAG đầu-cuối bằng file câu hỏi | `--name questions.jsonl`, `--rescore` |
| `retrieval` | Đo hit@k / MRR của retriever | `--modes dense,sparse,hybrid` |
| `llms` | Liệt kê các LLM tiếng Việt có sẵn | |
| `compare` | So sánh nhiều LLM trên cùng ngữ cảnh | `--llms all` |

---

## ⚙️ Cấu hình Mô hình mặc định (`config.py`)

- **Embedding**: `bkai-foundation-models/vietnamese-bi-encoder` (768d, SentenceTransformer)
- **Reranker**: `AITeamVN/Vietnamese_Reranker` (CrossEncoder)
- **Default LLM**: `vistral-7b-chat` (hoặc `vi-qwen2-3b-rag`)
- **Semantic Chunking**: `SEMANTIC_CHUNKING = True`, `SEMANTIC_CHUNK_THRESHOLD = 0.75`
- **Retrieval**: `TOP_K = 5`, `TOP_N_QA = 3`, `SEARCH_MODE = 'hybrid'`

---

## 📦 Cài đặt Môi trường Dev

```bash
uv sync --group dev
uv run pytest
```

---

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.
