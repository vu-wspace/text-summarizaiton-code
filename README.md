# VietNews: Tóm tắt + NER + Hỏi đáp RAG

Hệ thống hỏi đáp tin tức tiếng Việt trên tập test **VietNews (22.492 bài)**, gồm ba chức năng trong một chatbot terminal:

- **Hỏi đáp (RAG)**: Semantic Chunking → Qdrant Hybrid Search (dense + BM25, RRF) → Reranker → LLM tiếng Việt
- **Tóm tắt**: ViT5-large fine-tune trên VietNews
- **Trích xuất thực thể (NER)**: ELECTRA-base tiếng Việt, NER sẵn toàn bộ kho bài

---

## 🤖 Mô hình sử dụng

| Vai trò | Mô hình | Khai báo |
|---|---|---|
| Embedding | `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT, 768d, tối đa 256 token, đầu vào tách từ bằng `pyvi`) | `config.EMB_MODEL` |
| Reranker | `AITeamVN/Vietnamese_Reranker` (CrossEncoder, fine-tune từ bge-reranker-v2-m3) | `config.RERANK_MODEL` |
| Tóm tắt | `VietAI/vit5-large-vietnews-summarization` (seq2seq, input ≤ 1024 / output ≤ 256 token, beam 4) | `config.SUM_MODEL` |
| NER | `NlpHUST/ner-vietnamese-electra-base` (token classification, cửa sổ ≤ 400 token) | `config.NER_MODEL` |
| LLM sinh câu trả lời | Mặc định `vistral-7b-chat` (cần bật `llama-server`); đổi bằng `--llm <key>` | `config.DEFAULT_LLM` |

### LLM có sẵn (`vietnews llms`)

| Key | Repo | Ghi chú |
|---|---|---|
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` | Đa ngôn ngữ, chạy thẳng bằng transformers (không cần server) |
| `vi-qwen2-3b-rag` | `AITeamVN/Vi-Qwen2-3B-RAG` | Fine-tune cho RAG tiếng Việt, prompt riêng |
| `vistral-7b-chat` | `models/Vistral-7B-ChatML.i1-Q4_K_M.gguf` | **Mặc định**. GGUF chạy qua **llama-server** (xem bên dưới), ngữ cảnh 8192 |
| `vinallama-2.7b-chat` | `vilm/vinallama-2.7b-chat` | ChatML, ngữ cảnh 4096 |
| `vietnamese-llama2-7b` | `bkai-foundation-models/vietnamese-llama2-7b-40GB` | Mô hình nền (không phải chat), repo khoá cần HF token |
| `phogpt-4b-chat` | `vinai/PhoGPT-4B-Chat` | Cần transformers 4.44 → service Docker `phogpt` |
| `qwen3.5-2b-vie` | `alphaedge-ai/Qwen3.5-2B-vie-32768` | Cần transformers ≥ 5.2 → service Docker `llm5` |

**Chạy Vistral-7B-Chat (mặc định)**: tải file GGUF vào `models/`, rồi bật llama.cpp server trước khi chạy `chat` / `ask` / `questions` / `evaluate`:

```bash
llama-server -m models/Vistral-7B-ChatML.i1-Q4_K_M.gguf -ngl 99 -c 8192   # mặc định http://localhost:8080
# đổi địa chỉ server bằng biến môi trường RAG_LLAMA_URL
```

---

## 🏗️ Kiến trúc

```
                         Input người dùng
                                │
                    Router (router.py): phân loại intent
        ┌───────────────┬───────┴────────┬──────────────────┐
        │ qa            │ list           │ summarize_topic  │ summarize_text
        ▼               ▼                ▼                  ▼
 Hybrid Search     Tra entities.json  Tìm bài liên quan   Chuẩn hoá văn bản
 (Qdrant)          (NER sẵn)          nhất                (≥ 300 ký tự)
  dense + BM25                            │                  │
  → RRF, top-5                            └───────┬──────────┘
        │                                         ▼
 Vietnamese_Reranker                        ViT5 tóm tắt
  → 3 bài khác nhau
        │
 Ghép ngữ cảnh (≤ 3000 token)
        │
 LLM sinh câu trả lời
```

---

## 📊 Kết quả đánh giá

Chi tiết trong `vietnamese_llm_rag_evaluation.md` và `evaluation_results/`.

| Mô hình | Latency TB | TPS | Faithfulness | Relevancy | Fluency |
|---|---|---|---|---|---|
| **Vistral-7B-Chat-GGUF** (Q4_K_M) | 8.914 ms | 2,51 | **0,975** | 0,674 | 0,337 |

> Lưu ý: đây là benchmark của `evaluate_vietnamese_rag.py` trên **10 câu hỏi kiến thức chung** (không phải câu hỏi trên kho VietNews), mới đo cho Vistral-7B-Chat. Để so sánh các LLM trên chính kho VietNews, dùng `vietnews compare` (xem bên dưới).

---

## 📁 Cấu trúc dự án

```
text_summarization/
├── src/text_summarization/
│   ├── config.py      # Đường dẫn, tên mô hình, tham số (chunking, top-k, ...)
│   ├── text.py        # Chuẩn hoá VietNews, làm sạch input terminal, so khớp đáp án
│   ├── data.py        # Tải + loại trùng VietNews, đọc/ghi JSON
│   ├── ner.py         # NER cả bài (ELECTRA, cửa sổ ≤ 400 token, gộp tên)
│   ├── indexing.py    # Semantic chunking -> embed -> Qdrant
│   ├── sparse.py      # Vector thưa BM25 (unigram + bigram âm tiết), IDF do Qdrant tính
│   ├── store.py       # QdrantStore: dense / sparse / hybrid (RRF) + chunks.pkl
│   ├── llms.py        # Danh sách LLM + prompt riêng từng mô hình
│   ├── models.py      # Embedding, reranker, ViT5, LLM (nạp lười)
│   ├── router.py      # Phân loại intent: qa / list / summarize_topic / summarize_text
│   ├── pipeline.py    # Luồng RAG: retrieve → re-rank → context → answer
│   ├── questions.py   # Sinh câu hỏi đánh giá + chấm hit@k, MRR, contains
│   ├── metrics.py     # Contains, recall, F1, EM, tỉ lệ từ chối
│   ├── compare.py     # So sánh nhiều LLM trên cùng ngữ cảnh
│   └── cli.py         # CLI: vietnews <lệnh>
├── tests/                          # 45 unit test (pytest)
├── evaluation_results/             # Kết quả benchmark JSON
├── evaluate_vietnamese_rag.py      # Script benchmark LLM RAG
├── vietnamese_llm_rag_evaluation.md
├── Dockerfile, Dockerfile.llm5, Dockerfile.phogpt, docker-compose.yml
├── pyproject.toml, requirements.txt, uv.lock
│
│   # Sinh ra khi chạy (không commit):
├── OUTPUT_LOAD/       # data_article.json
├── OUTPUT_INDEX/      # entities.json, faiss_index/{chunks.pkl, embedds.npy}, qdrant/
├── OUTPUT_QUESTIONS/  # questions.jsonl, questions_eval.jsonl, retrieval_report.md, compare_report.md
└── models/            # file GGUF cho llama-server
```

---

## ⚡ Chạy trực tiếp (`uv`)

Yêu cầu: Python 3.11–3.12, GPU NVIDIA (torch được cài từ index CUDA 12.8 — xem `pyproject.toml` để đổi sang cu126 / cpu).

```bash
# 1. Cài môi trường
uv sync

# 2. Tải + chuẩn hoá VietNews -> OUTPUT_LOAD/
uv run vietnews load

# 3. Xây index: NER + Semantic Chunking + Embedding + Qdrant -> OUTPUT_INDEX/
uv run vietnews index

# 4. Chat trong terminal (thêm --llm-4bit nếu VRAM ít)
uv run vietnews chat
uv run vietnews chat --llm vi-qwen2-3b-rag

# 5. Hỏi một câu rồi thoát
uv run vietnews ask "Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam"
```

> **Đổi mô hình embedding thì phải dựng lại index** (`vietnews index --steps chunk,embed,index`). Nếu số chiều vector câu hỏi khác collection Qdrant, `chat`/`ask` sẽ báo lỗi kèm hướng dẫn này.

Mặc định Qdrant chạy nhúng tại `OUTPUT_INDEX/qdrant`. Đặt `RAG_QDRANT_URL=http://localhost:6333` để dùng Qdrant server.

---

## 🐳 Chạy bằng Docker (GPU)

```bat
chcp 65001
docker compose build
docker compose up -d qdrant
docker compose run --rm rag index --steps chunk,embed,index   # dựng index (~20-25 phút GPU)
docker compose run --rm rag                          # mặc định: chat với vistral-7b-chat (cần llama-server trên host)
docker compose run --rm rag ask "Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam" --llm-4bit
```

- Qdrant trong Docker lưu ở volume `qdrant_data`, **tách biệt** với Qdrant nhúng `OUTPUT_INDEX/qdrant` khi chạy bằng `uv`. Đã có `chunks.pkl` + `embedds.npy` mới thì chỉ cần `index --steps index` để nạp vào Qdrant.
- Vistral-7B-Chat: chạy `llama-server` trên máy host với `--host 0.0.0.0`, container gọi qua `RAG_LLAMA_URL=http://host.docker.internal:8080` (đã đặt sẵn trong `docker-compose.yml`):

```bat
tools\llama-cpp\llama-server -m models\Vistral-7B-ChatML.i1-Q4_K_M.gguf -ngl 99 -c 8192 --host 0.0.0.0
docker compose run --rm rag chat --llm vistral-7b-chat
```

LLM cần phiên bản transformers khác chạy trong service riêng:

```bat
docker compose --profile llm5   run --rm llm5   answer --llm qwen3.5-2b-vie
docker compose --profile phogpt run --rm phogpt answer --llm phogpt-4b-chat --llm-4bit
```

---

## 📋 Các lệnh CLI (`vietnews`)

| Lệnh | Mô tả | Tham số hữu ích |
|---|---|---|
| `load` | Tải VietNews, loại trùng, chuẩn hoá | `--check`, `--source DIR` |
| `index` | NER + chunk + embed + Qdrant | `--steps ner,chunk,embed,index`, `--chunk-size`, `--header` |
| `chat` | Chatbot: hỏi đáp, liệt kê thực thể, tóm tắt | `--llm <key>`, `--llm-4bit`, `--search hybrid\|dense\|sparse`, `--verbose` |
| `ask "..."` | Hỏi một câu rồi thoát | như `chat` |
| `questions` | Sinh N câu hỏi đánh giá bằng LLM | `--n 10`, `--seed`, `--eval` |
| `evaluate` | Chấm RAG đầu-cuối bằng file câu hỏi | `--name questions.jsonl`, `--rescore` |
| `retrieval` | Đo hit@k / MRR của retriever | `--modes dense,sparse,hybrid`, `--no-rerank` |
| `llms` | Liệt kê các LLM có sẵn | |
| `probe` | Chẩn đoán 1 LLM (logits NaN, top-5 token) | `--llm <key>` |
| `contexts` | Truy xuất 1 lần cho mọi câu hỏi → `contexts.jsonl` | `--name` |
| `answer` | 1 LLM trả lời trên `contexts.jsonl` → `answers_<llm>.jsonl` | `--llm <key>` |
| `report` | Gộp + chấm mọi `answers_*.jsonl` → `compare_report.md/.csv` | `--llms` |
| `compare` | `contexts` + `answer` (nhiều LLM) + `report` trong một lệnh | `--llms all`, `--refresh` |

---

## ⚙️ Tham số chính (`config.py`)

| Nhóm | Tham số |
|---|---|
| Chunking | `SEMANTIC_CHUNKING = True`, `SEMANTIC_CHUNK_THRESHOLD = 0.75`, chunk 64–254 token (đếm trên văn bản đã tách từ, vừa giới hạn 256 của PhoBERT); `CHUNK_SIZE = 254`, `CHUNK_OVERLAP = 30` |
| Retrieval | `SEARCH_MODE = 'hybrid'`, `PREFETCH = 20`, `TOP_K = 5`, `TOP_N_QA = 3`, `N_ARTICLES = 3`, `CTX_BUDGET = 3000` |
| Tóm tắt | `SUM_MAX_INPUT = 1024`, `SUM_MAX_OUTPUT = 256`, `SUM_MIN_TEXT = 300` |
| NER | `NER_WINDOW = 400` |

Biến môi trường: `RAG_LOAD_DIR`, `RAG_INDEX_DIR`, `RAG_QUESTIONS_DIR`, `RAG_QDRANT_URL`, `RAG_LLAMA_URL`.

---

## 🧪 Test

```bash
uv sync --group dev
uv run pytest
```
