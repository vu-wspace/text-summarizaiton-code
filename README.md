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

Dựa trên kết quả thực nghiệm benchmark độc lập trên **100 câu hỏi đánh giá** (10 bộ x 10 câu hỏi sinh từ kho VietNews với cùng ngữ cảnh truy xuất: hit@1 = 0,86, hit@3 = 0,96):

| Thứ hạng | Model LLM | Contains ↑ | Recall ↑ | F1 ↑ | EM ↑ | Tốc độ | VRAM (4-bit) | Đánh giá & Khuyến nghị |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 🥇 **Top 1** | **`vi-qwen2-3b-rag`** *(Mặc định)* | **0.67** | **0.81** | 0.16 | 0.00 | 6.10s | ~5.55 GB | **Độ chính xác cao nhất**, được fine-tune chuyên sâu cho RAG tiếng Việt bởi AITeamVN. Tỉ lệ trích xuất đúng đáp án tuyệt đối cao nhất. |
| 🥈 **Top 2** | **`qwen2.5-3b`** | **0.59** | **0.74** | **0.35** | **0.06** | 3.06s | ~4.01 GB | **Cân bằng & Trả lời súc tích**, tốc độ sinh nhanh gấp đôi Top 1 (~3.06s/câu), câu trả lời gọn gàng không thừa từ, phù hợp nhu cầu phản hồi nhanh. |
| 🥉 **Top 3** | **`qwen3.5-2b-vie`** | 0.52 | 0.70 | **0.51** | **0.29** | **1.82s** | **~1.54 GB** | **Siêu nhẹ & Nhanh nhất**, F1-score cao (0.51), VRAM chỉ tốn ~1.5 GB. Cực kỳ tối ưu cho các máy có cấu hình phần cứng / GPU khiêm tốn. |

> 📌 *Các mô hình khác đã kiểm thử: `vistral-7b-chat` (GGUF, contains 0.43, từ chối 23% câu dù có ngữ cảnh), `vinallama-2.7b-chat` (contains 0.36, sinh dài dòng), `vietnamese-llama2-7b` (mô hình nền, không phù hợp hỏi đáp).*

---

### 📈 Đánh giá Faithfulness & Relevancy (LLM-as-a-judge, 100 câu hỏi)

Lệnh: `uv run vietnews --questions-dir OUTPUT_QUESTIONS/compare100 judge --judge vi-qwen2-3b-rag`

- **Faithfulness**: mọi thông tin trong câu trả lời có nằm trong ngữ cảnh truy xuất không (0 / 0.5 / 1). Trả lời ngắn / thiếu ý KHÔNG bị trừ điểm.
- **Relevancy**: câu trả lời có đúng trọng tâm câu hỏi không (không nhìn đáp án chuẩn; độ đúng đã đo bằng contains / recall).

**Kiểm soát judge trước khi tin con số**: 30 câu positive (đáp án chuẩn, có trong ngữ cảnh) + 30 câu negative (đáp án của câu hỏi khác).

| Judge | Faithfulness: TB positive ↑ | TB negative ↓ | Độ chính xác | Relevancy: độ chính xác |
|---|:---:|:---:|:---:|:---:|
| `vi-qwen2-3b-rag` | **0.98** | **0.00** | **1.00** | 0.68 |
| `qwen2.5-3b` | 0.63 | 0.00 | 0.93 | 0.58 |

| Model | Faithfulness (judge vi-qwen2) | Faithfulness (judge qwen2.5) | Relevancy (judge vi-qwen2) |
|---|:---:|:---:|:---:|
| `vi-qwen2-3b-rag` (prompt gốc) | **0.980** | **0.805** | **0.865** |
| `vi-qwen2-3b-rag` (prompt ngắn, mặc định mới) | 0.975 | 0.755 | 0.860 |
| `qwen2.5-3b` | 0.950 | 0.655 | 0.785 |
| `qwen3.5-2b-vie` | 0.945 | 0.630 | 0.685 |

> 📌 Parse lỗi: 0 % ở mọi lần chạy. Hai judge cho **cùng thứ hạng**, kể cả khi `qwen2.5-3b` tự chấm chính nó (thiên vị tự chấm không giải thích được kết quả). Judge `qwen2.5-3b` khắt khe hơn (chỉ cho 0.63 với câu positive) nên giá trị tuyệt đối của nó thấp hơn; dùng `vi-qwen2-3b-rag` làm số chính. **Relevancy độ tin cậy thấp** (độ chính xác kiểm soát 0.58–0.68) — chỉ nên xem như tham khảo.
>
> ⚠ *Kết quả cũ (faithfulness = 0.000 cho mọi model) là lỗi đo, không phải do model*: khuôn mẫu `{"score": 0.0, "reason": "..."}` trong prompt khiến judge 3B chép nguyên điểm 0.0 (57–84 % lý do bắt đầu bằng chuỗi `...`), parse lỗi (8–20 %) bị tính 0 điểm, và rubric phạt câu trả lời *thiếu* thông tin như câu *bịa*. Đã sửa trong `src/text_summarization/evaluation/judge.py`.

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

# Tải tập test VietNews, loại trùng, chuẩn hoá -> OUTPUT_LOAD/data_article.json
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

**Thử tối ưu truy xuất (dev = 100 câu ở trên, held-out = 110 câu mới, seed 2026, không trùng bài):** 23 cấu hình (độ sâu re-rank k = 5–50, RRF có trọng số, thêm tiêu đề bài khi re-rank, gộp điểm re-ranker + retriever). Không cấu hình nào hơn cấu hình hiện tại có ý nghĩa thống kê; cấu hình tốt nhất trên dev (hit@1 0.88) tụt còn 0.83 trên held-out (quá khớp). Cấu hình hiện tại ổn định: hit@1 **0.87 / 0.87**. Re-ranker là thành phần đóng góp rõ nhất (bỏ re-ranker: held-out hit@1 0.87 → 0.74, p = 0.001). → **Giữ nguyên cấu hình.**

Phần lớn các câu "trượt" hit@1 không phải lỗi truy xuất: VietNews có nhiều bài cùng sự kiện, bài top-1 khác bài nguồn nhưng **vẫn chứa đáp án** (6/13 câu trượt ở dev), số còn lại chủ yếu là câu hỏi mơ hồ do LLM sinh (vd. *"Khi nào ông Trump sa thải ông Comey?"* → *"Bốn tháng sau đó"*). Tỉ lệ bài top-1 chứa đáp án chuẩn là **0.93 / 0.94** (dev / held-out). `compare_report.md` nay báo thêm **ans@1 / ans@3** (ngữ cảnh chứa đáp án chuẩn) bên cạnh hit@k.

### 2. So sánh Top-3 LLM đầu-cuối (End-to-End Generation)
Kiểm thử trên 100 câu hỏi với **cùng một ngữ cảnh truy xuất**:

| Tiêu chí | 🥇 vi-qwen2-3b-rag | 🥈 qwen2.5-3b | 🥉 qwen3.5-2b-vie |
|---|:---:|:---:|:---:|
| **Điểm Contains (Chính xác)** | **0.67** | 0.59 | 0.52 |
| **Recall** | **0.81** | 0.74 | 0.70 |
| **F1 Score** | 0.16 | 0.35 | **0.51** |
| **EM (Exact Match)** | 0.00 | 0.06 | **0.29** |
| **Tỉ lệ từ chối (Refusal)** | **0.00** | **0.00** | **0.00** |
| **Thời gian phản hồi TB** | 6.10s / câu | 3.06s / câu | **1.82s / câu** |
| **Số từ trung bình** | 55.4 từ | 17.9 từ | 15.7 từ |
| **VRAM tiêu thụ (4-bit)** | ~5.55 GB | ~4.01 GB | **~1.54 GB** |

- **`vi-qwen2-3b-rag`** vượt trội về khả năng bóc tách đầy đủ dữ kiện ngữ cảnh thành câu trả lời tự nhiên.
- **`qwen2.5-3b`** phản hồi ngắn gọn, tập trung thẳng vào trọng tâm câu hỏi, tiết kiệm 50% thời gian suy luận.
- **`qwen3.5-2b-vie`** cực kỳ tiết kiệm tài nguyên (1.5 GB VRAM), nhanh nhất, phù hợp triển khai edge hoặc server cấu hình thấp.

**Prompt ngắn cho `vi-qwen2-3b-rag` (mặc định mới, `prompt='vi-rag-concise'`):** giữ nguyên khuôn prompt AITeamVN, chỉ thêm yêu cầu *"Trả lời bằng MỘT câu ngắn…"*. Cùng ngữ cảnh, chạy lại cả hai prompt trong cùng điều kiện:

| Tập | Prompt | Contains | Recall | F1 | Số từ TB | Giây/câu |
|---|---|:---:|:---:|:---:|:---:|:---:|
| dev (100) | gốc `vi-rag` | 0.69 | 0.81 | 0.162 | 59.6 | — |
| dev (100) | ngắn `vi-rag-concise` | 0.68 | 0.81 | **0.198** | **42.9** | — |
| held-out (110) | gốc `vi-rag` | 0.60 | 0.75 | 0.141 | 66.1 | 13.0 |
| held-out (110) | ngắn `vi-rag-concise` | 0.58 | 0.74 | **0.167** | **53.8** | **10.8** |

Contains không khác biệt có ý nghĩa (McNemar p = 1.0 / 0.75), faithfulness giữ nguyên. Cải thiện có thật nhưng **khiêm tốn**: mô hình đã fine-tune vẫn hay mở đầu "Theo ngữ cảnh…" và thêm "Cụ thể…". Cắt hậu xử lý phần "Cụ thể…" tăng F1 lên 0.23 nhưng làm mất 1–5 câu trả lời đúng nên **không** áp dụng. EM ≈ 0 là do cấu trúc: mô hình trả lời bằng câu hoàn chỉnh còn đáp án chuẩn là cụm từ ngắn.

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
| `judge` | LLM-as-a-judge: faithfulness + relevancy, kèm câu kiểm soát | `--judge vi-qwen2-3b-rag`, `--answers <keys>` |

---

## 📂 Cấu trúc thư mục

```
text_summarization/
├── src/text_summarization/
│   ├── cli.py                 # Entrypoint chính cho các câu lệnh vietnews
│   ├── core/                  # Nền tảng dùng chung
│   │   ├── config.py          # Tập trung toàn bộ hằng số, đường dẫn, siêu tham số
│   │   ├── data.py            # Tải, làm sạch, đọc/ghi dữ liệu VietNews
│   │   └── text.py            # Chuẩn hoá văn bản, so khớp đáp án
│   ├── rag/                   # Hệ thống RAG
│   │   ├── indexing.py        # Semantic Chunking và luồng nạp Index
│   │   ├── ner.py             # Trích xuất thực thể tên riêng (ELECTRA)
│   │   ├── store.py           # Kết nối và truy vấn Qdrant
│   │   ├── sparse.py          # Tokenizer & Vectorizer BM25 âm tiết
│   │   ├── models.py          # Nạp lười embedding, re-ranker, ViT5, LLM
│   │   ├── llms.py            # Đặc tả LLM, template prompt, quản lý VRAM
│   │   ├── router.py          # Phân loại intent câu hỏi
│   │   └── pipeline.py        # Luồng RAG hoàn chỉnh (Retrieve -> Rerank -> QA/Sum)
│   └── evaluation/            # Đánh giá
│       ├── questions.py       # Sinh câu hỏi, chấm truy xuất (hit@k, MRR)
│       ├── compare.py         # So sánh LLM trên cùng ngữ cảnh (contexts -> answer -> report)
│       ├── metrics.py         # Chỉ số: contains, recall, F1, McNemar
│       └── judge.py           # LLM-as-a-judge: faithfulness / relevancy + câu kiểm soát
├── tests/                 # 68 bài test đơn vị (unit tests)
├── OUTPUT_LOAD/           # data_article.json: kho 22.492 bài (sinh bởi `vietnews load`)
├── OUTPUT_INDEX/          # entities.json + vectors/{chunks.pkl, embedds.npy} (sinh bởi `vietnews index`)
├── OUTPUT_QUESTIONS/      # kết quả đánh giá
│   ├── compare100/        # benchmark chính: 100 câu, câu trả lời 6 LLM, judge_<llm>/report.md
│   ├── heldout100/        # 110 câu held-out (seed 2026) để kiểm chứng, không dùng để tinh chỉnh
│   └── archive/           # các lần chạy cũ (set1–10, 10 câu đầu tiên...)
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
*Kết quả:* Toàn bộ 68/68 unit tests đều pass trong ~1.5 giây.
