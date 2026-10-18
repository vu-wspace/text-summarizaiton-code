# VietNews RAG: Hỏi đáp · Tóm tắt · Trích xuất thực thể

Hệ thống RAG tin tức tiếng Việt dựa trên kho kiểm thử **VietNews (22.492 bài)**, tối ưu cho máy có GPU cá nhân (RTX 30/40/50 series) và Docker.

---

## 🎯 Ba chức năng chính

| Câu lệnh / Cú pháp | Hành vi hệ thống | Ví dụ minh họa |
|---|---|---|
| `<Câu hỏi bất kỳ>` | **Hỏi đáp RAG**: Truy xuất bài viết liên quan và dùng LLM trả lời ngắn gọn, chính xác | `Vụ 2 tàu đối đầu ở Quảng Nam xảy ra như thế nào?` |
| `Liệt kê ...` | **Trích xuất thực thể** (người, địa điểm, tổ chức) của bài viết liên quan nhất | `Liệt kê người trong vụ Hồ Xuân Huy lừa đảo` |
| `Tóm tắt <chủ đề>` | Truy xuất bài liên quan nhất và **tóm tắt** bằng ViT5 | `Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam` |
| `Tóm tắt: <văn bản ≥ 300 ký tự>` | Tóm tắt trực tiếp đoạn văn bản do người dùng dán vào | `Tóm tắt: Ngày 15-10, Tòa án nhân dân tỉnh...` |

---

## 🏆 Top-3 Model LLM Tốt Nhất

Dựa trên kết quả thực nghiệm benchmark độc lập trên **100 câu hỏi đánh giá** (10 bộ x 10 câu hỏi sinh từ kho VietNews với cùng ngữ cảnh truy xuất: hit@1 = 0,89, hit@3 = 0,94):

| Thứ hạng | Model LLM | Điểm Contains ↑ | Recall ↑ | Tốc độ | VRAM (4-bit) | Đánh giá & Khuyến nghị |
|:---:|---|:---:|:---:|:---:|:---:|---|
| 🥇 **Top 1** | **`vi-qwen2-3b-rag`** *(Mặc định)* | **67.0%** (0.67) | **0.81** | 6.43s | ~3.98 GB | **Độ chính xác cao nhất**, được fine-tune chuyên sâu cho RAG tiếng Việt bởi AITeamVN. Tỉ lệ trích xuất đúng đáp án tuyệt đối cao nhất. |
| 🥈 **Top 2** | **`qwen2.5-3b`** | **59.0%** (0.59) | **0.74** | 3.09s | ~4.01 GB | **Cân bằng & Trả lời súc tích**, tốc độ sinh nhanh gấp đôi Top 1 (~3.09s/câu), câu trả lời gọn gàng không thừa từ, phù hợp nhu cầu phản hồi nhanh. |
| 🥉 **Top 3** | **`qwen3.5-2b-vie`** | **52.0%** (0.52) | **0.70** | **2.00s** | **~1.54 GB** | **Siêu nhẹ & Nhanh nhất**, F1-score cao (0.51), VRAM chỉ tốn ~1.5 GB. Cực kỳ tối ưu cho các máy có cấu hình phần cứng / GPU khiêm tốn. |

> 📌 *Các mô hình khác đã kiểm thử: `vistral-7b-chat` (GGUF, contains 43%, từ chối 23% câu dù có ngữ cảnh), `vinallama-2.7b-chat` (contains 36%, sinh dài dòng), `vietnamese-llama2-7b` (mô hình nền, không phù hợp hỏi đáp).*

---

## 🏗 Kiến trúc Pipeline

```
                              Câu truy vấn
                                   │
                   Router: qa / list / summarize
        ┌──────────────────────────┼─────────────────────────┐
        ▼                          ▼                         ▼
   [Hỏi đáp RAG]              [Liệt kê thực thể]         [Tóm tắt]
        │                          │                         │
 Qdrant Hybrid Search        Bài viết top-1            Bài viết top-1
 (Dense PhoBERT + BM25)            │                         │
        │                  Đọc entities.json          Chuẩn hóa văn bản
   Top-5 chunks           (NER tính sẵn lúc index)           │
        │                                                    ▼
 Vietnamese_Reranker                                   ViT5 Summarizer
        │
   Top-3 bài viết
 (≤ 3.000 token ngữ cảnh)
        │
        ▼
   LLM tiếng Việt (vi-qwen2-3b-rag 4-bit)
```

### Các thành phần mô hình trong hệ thống:
- **Embedding**: `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT 768 chiều, giới hạn 256 tokens, tách từ qua `pyvi`).
- **Kho Vector**: `Qdrant` hỗ trợ tìm kiếm lai (Dense vector + Sparse BM25 unigram/bigram âm tiết gộp Reciprocal Rank Fusion - RRF).
- **Re-ranker**: `AITeamVN/Vietnamese_Reranker` (bge-reranker-v2-m3 fine-tune tiếng Việt).
- **Tóm tắt (Summarization)**: `VietAI/vit5-large-vietnews-summarization` (chạy bf16 trên GPU).
- **Trích xuất thực thể (NER)**: `NlpHUST/ner-vietnamese-electra-base`.
- **LLM Generator**: Tùy chọn 1 trong Top-3 (`vi-qwen2-3b-rag`, `qwen2.5-3b`, `qwen3.5-2b-vie`), mặc định nạp lượng tử hóa 4-bit (`bitsandbytes`) chỉ tốn ~3-4 GB VRAM.

---

## ⚡ Cài đặt & Chạy trực tiếp (Local uv)

### Yêu cầu:
- Python 3.11 – 3.12, công cụ quản lý gói [uv](https://docs.astral.sh/uv/).
- GPU NVIDIA (PyTorch CUDA 12.8, tương thích RTX 50xx, 40xx, 30xx, 20xx).

### 1. Đồng bộ môi trường và khởi tạo dữ liệu
```bash
# Cài đặt toàn bộ thư viện cần thiết
uv sync

# Tải, chuẩn hóa và loại trùng dữ liệu VietNews -> OUTPUT_LOAD/
uv run vietnews load

# Dựng kho: NER + Chunking + Embedding + Qdrant -> OUTPUT_INDEX/
uv run vietnews index
```

### 2. Tương tác và sử dụng
- **Chạy chatbot tương tác (khuyên dùng):**
  ```bash
  # Windows PowerShell
  $env:RAG_QDRANT_URL = "http://localhost:6333"; uv run vietnews chat

  # Windows CMD
  chcp 65001
  set RAG_QDRANT_URL=http://localhost:6333
  uv run vietnews chat
  ```
- **Tùy chọn LLM:**
  ```bash
  uv run vietnews chat --llm qwen2.5-3b            # Chuyển sang Top-2 LLM nhanh hơn
  uv run vietnews chat --no-llm-4bit              # Dùng full bf16 (GPU >= 12GB)
  uv run vietnews ask "Hà Nội ban hành quy tắc ứng xử khi nào?"  # Hỏi một câu duy nhất
  ```

---

## 🐳 Chạy bằng Docker & Docker Compose (Khuyên dùng)

### 1. Khởi động Qdrant Server
```bash
docker compose up -d qdrant
```

### 2. Xây dựng image và đánh chỉ mục
```bash
docker compose build
docker compose run --rm rag index
```

### 3. Khởi chạy Chatbot
```bash
# Chat với LLM mặc định (Top 1: vi-qwen2-3b-rag)
docker compose run --rm rag

# Đặt câu hỏi trực tiếp qua CLI
docker compose run --rm rag ask "Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam"

# Đổi sang Top 2 (qwen2.5-3b)
docker compose run --rm rag chat --llm qwen2.5-3b

# Chạy Top 3 (qwen3.5-2b-vie qua service profile llm5)
docker compose --profile llm5 run --rm llm5 answer --llm qwen3.5-2b-vie
```

---

## 📊 Kết quả thực nghiệm chi tiết

### 1. Đánh giá Retrieval (Truy xuất bài viết)
Thực nghiệm trên 100 câu hỏi (hit@k tính theo bài viết gốc trong top-20 chunks):

| Cơ chế tìm kiếm | Giai đoạn | hit@1 | hit@3 | hit@5 | hit@20 | MRR |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Dense đơn thuần | Retriever | 0.62 | 0.78 | 0.80 | 0.90 | 0.709 |
| Dense đơn thuần | + Re-ranker | 0.78 | 0.85 | 0.87 | 0.90 | 0.822 |
| Sparse (BM25) | Retriever | 0.87 | 0.93 | 0.95 | 0.99 | 0.906 |
| Sparse (BM25) | + Re-ranker | 0.85 | 0.95 | 0.96 | 0.99 | 0.903 |
| **Hybrid (Dense + BM25)** | Retriever | 0.78 | 0.95 | 0.95 | 0.98 | 0.864 |
| **Hybrid (Dense + BM25)** | **+ Re-ranker** | **0.86** | **0.96** | **0.96** | **0.98** | **0.909** |

*Nhận xét:* BM25 vượt trội dense đối với tin tức tiếng Việt do chứa nhiều tên người, địa danh, số liệu cụ thể. Kết hợp Hybrid + Re-ranker mang lại MRR cao nhất (0.909).

### 2. So sánh Top-3 LLM đầu-cuối (End-to-End Generation)
Kiểm thử trên 100 câu hỏi với **cùng một ngữ cảnh truy xuất**:

| Tiêu chí | 🥇 vi-qwen2-3b-rag | 🥈 qwen2.5-3b | 🥉 qwen3.5-2b-vie |
|---|:---:|:---:|:---:|
| **Điểm Contains (Chính xác)** | **0.67 (67%)** | 0.59 (59%) | 0.52 (52%) |
| **Recall** | **0.81** | 0.74 | 0.70 |
| **F1 Score** | 0.16 | 0.35 | **0.51** |
| **Tỉ lệ từ chối sai (Refusal)** | **0.00%** | **0.00%** | **0.00%** |
| **Thời gian phản hồi TB** | 6.43s / câu | 3.09s / câu | **2.00s / câu** |
| **Số từ trung bình** | 55.4 từ | 17.9 từ | 15.7 từ |
| **VRAM tiêu thụ (4-bit)** | ~3.98 GB | ~4.01 GB | **~1.54 GB** |

- **`vi-qwen2-3b-rag`** vượt trội về khả năng bóc tách đầy đủ dữ kiện ngữ cảnh thành câu trả lời tự nhiên.
- **`qwen2.5-3b`** phản hồi ngắn gọn, tập trung thẳng vào trọng tâm câu hỏi, tiết kiệm 52% thời gian suy luận.
- **`qwen3.5-2b-vie`** cực kỳ tiết kiệm tài nguyên (1.5 GB VRAM), nhanh nhất, phù hợp triển khai edge hoặc server cấu hình thấp.

---

## 🛠 Danh mục lệnh CLI (`vietnews <lệnh>`)

| Lệnh | Ý nghĩa | Tham số tiêu biểu |
|---|---|---|
| `load` | Tải dữ liệu, lọc trùng, làm sạch bài viết | `--source <dir>`, `--check` |
| `index` | Tạo embedding và nạp vào Qdrant | `--steps ner,chunk,embed,index` |
| `chat` | Khởi động phiên chatbot tương tác | `--llm <key>`, `--llm-4bit`, `--search hybrid` |
| `ask "..."` | Hỏi 1 câu đơn lẻ | `--llm <key>`, `--search hybrid` |
| `questions` | Sinh tập câu hỏi trắc nghiệm/đánh giá | `--n 10`, `--seed 42`, `--eval` |
| `retrieval` | Đánh giá độ chuẩn xác truy xuất (hit@k, MRR) | `--modes dense,sparse,hybrid` |
| `contexts` | Thu thập ngữ cảnh chung cho tập câu hỏi | `--name questions.jsonl` |
| `answer` | Cho một LLM cụ thể trả lời trên contexts | `--llm <key>` |
| `report` | Tổng hợp bảng so sánh, vẽ chỉ số và báo cáo | `--llms vi-qwen2-3b-rag,qwen2.5-3b` |
| `compare` | Chạy trọn vẹn quy trình benchmark LLM | `--refresh` |

---

## 📂 Cấu trúc thư mục

```
text_summarization/
├── src/text_summarization/
│   ├── cli.py             # Entrypoint chính cho các câu lệnh vietnews
│   ├── config.py          # Tập trung toàn bộ hằng số, đường dẫn, siêu tham số
│   ├── data.py            # Xử lý tải và làm sạch dữ liệu VietNews
│   ├── indexing.py        # Semantic Chunking và luồng nạp Index
│   ├── llms.py            # Đặc tả Top LLM, template prompt, quản lý VRAM
│   ├── metrics.py         # Đo đạc chỉ số: contains, recall, F1, McNemar
│   ├── models.py          # Quản lý vòng đời nạp lười mô hình PyTorch
│   ├── ner.py             # Trích xuất thực thể tên riêng (ELECTRA)
│   ├── pipeline.py        # Luồng RAG hoàn chỉnh (Retrieve -> Rerank -> QA/Sum)
│   ├── router.py          # Phân loại intent câu hỏi
│   ├── sparse.py          # Tokenizer & Vectorizer BM25 âm tiết
│   └── store.py           # Kết nối và truy vấn Qdrant
├── tests/                 # 57 bài test đơn vị (unit tests)
├── Dockerfile             # Image chạy pipeline RAG với PyTorch CUDA 12.8
├── Dockerfile.llm5        # Image phụ chạy các dòng mô hình transformers 5.x
├── docker-compose.yml     # Khởi động cụm dịch vụ RAG & Qdrant
├── pyproject.toml         # Cấu hình dự án uv và thư viện
└── requirements.txt       # Danh sách dependencies cho Docker build
```

---

## 🧪 Kiểm thử (Unit Tests)

Chạy bộ test tự động (không yêu cầu GPU):
```bash
uv run --group dev pytest
```
*Kết quả:* Toàn bộ 57/57 unit tests đều pass trong ~1.5 giây.
