# VietNews RAG: Hỏi đáp · Tóm tắt · Trích xuất thực thể

Chatbot terminal cho tin tức tiếng Việt, chạy trên tập test **VietNews (22.492 bài)**. Một câu lệnh, ba chức năng:

| Bạn gõ | Hệ thống làm | Ví dụ |
|---|---|---|
| Câu hỏi bất kỳ | **Hỏi đáp RAG**: tìm bài liên quan rồi để LLM trả lời từ chính các bài đó | `Vụ 2 tàu đối đầu ở Quảng Nam xảy ra như thế nào?` |
| `Liệt kê ...` | **Liệt kê thực thể** (người, địa điểm, tổ chức) của bài liên quan nhất | `Liệt kê người trong vụ Hồ Xuân Huy lừa đảo` |
| `Tóm tắt <chủ đề>` | Tìm bài liên quan nhất rồi **tóm tắt** bằng ViT5 | `Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam` |
| `Tóm tắt: <văn bản ≥ 300 ký tự>` | Tóm tắt chính văn bản dán vào | |

```
Bạn: Liệt kê người trong vụ Hồ Xuân Huy giả danh công an lừa đảo
[list] Trả lời: Bài: Bản án cho đối tượng giả danh công an để lừa đảo
  Người: Hồ Xuân Huy, Lê Quốc Th, Nguyễn Tấn T, Trần Nhật S, Đinh Ngọc H, Bảo Th
  - [1] Bản án cho đối tượng giả danh công an để lừa đảo
```

---

## Kiến trúc

```
                              Câu nhập
                                 │
                 Router (regex): qa / list / summarize_topic / summarize_text
      ┌──────────────────┬───────┴────────────┬─────────────────────┐
      │ qa               │ list               │ summarize_topic     │ summarize_text
      ▼                  ▼                    ▼                     ▼
 Qdrant hybrid       Bài top-1 ─► entities.json   Bài top-1          Chuẩn hoá văn bản
 dense + BM25 (RRF)  (NER tính sẵn lúc index)     │                     │
 top-5 chunk                                      └──────────┬──────────┘
      │                                                      ▼
 Vietnamese_Reranker → 3 bài khác nhau                 ViT5 tóm tắt
      │
 Ghép nguyên văn bài (≤ 3000 token)
      │
 LLM (mặc định Vistral-7B qua llama-server)
```

**Lúc dựng kho (index):**

1. **NER** trên tiêu đề, sapo và thân bài. Tên gọi tắt ("Huy") được gộp vào họ tên đầy đủ ("Hồ Xuân Huy"), vì báo thường chỉ ghi họ tên đầy đủ ở sapo.
2. **Semantic chunking**: tách câu, gộp các câu liền kề có cosine ≥ 0,75 thành một chunk dài 64–254 token. 254 token là vừa giới hạn 256 token của PhoBERT. Kết quả là 132.850 chunk.
3. **Embedding** `vietnamese-bi-encoder`, đầu vào được tách từ bằng `pyvi`.
4. **Qdrant**: mỗi chunk có một vector dense và một vector thưa BM25 (unigram + bigram âm tiết). Phần IDF do Qdrant tính.

## Mô hình

| Vai trò | Mô hình | Cấu hình |
|---|---|---|
| Embedding | `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT, 768 chiều, ≤ 256 token) | `config.EMB_MODEL` |
| Re-ranker | `AITeamVN/Vietnamese_Reranker` (bge-reranker-v2-m3 fine-tune tiếng Việt, fp16 trên GPU) | `config.RERANK_MODEL` |
| Tóm tắt | `VietAI/vit5-large-vietnews-summarization` (≤ 1024 token vào, ≤ 256 ra, beam 4, bf16 trên GPU) | `config.SUM_MODEL` |
| NER | `NlpHUST/ner-vietnamese-electra-base` (cửa sổ ≤ 400 token) | `config.NER_MODEL` |
| LLM | Mặc định `vistral-7b-chat`; đổi bằng `--llm <key>` | `config.DEFAULT_LLM` |

LLM có sẵn (`vietnews llms`):

| Key | Repo | Ghi chú |
|---|---|---|
| `vistral-7b-chat` | `models/Vistral-7B-ChatML.i1-Q4_K_M.gguf` | **Mặc định**. GGUF chạy qua **llama-server** |
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` | Chạy thẳng bằng transformers, không cần server |
| `vi-qwen2-3b-rag` | `AITeamVN/Vi-Qwen2-3B-RAG` | Fine-tune cho RAG tiếng Việt, prompt riêng |
| `vinallama-2.7b-chat` | `vilm/vinallama-2.7b-chat` | ChatML, ngữ cảnh 4096 |
| `vietnamese-llama2-7b` | `bkai-foundation-models/vietnamese-llama2-7b-40GB` | Mô hình nền (không phải chat), cần HF token |
| `phogpt-4b-chat` | `vinai/PhoGPT-4B-Chat` | Cần transformers 4.44 → service Docker `phogpt` |
| `qwen3.5-2b-vie` | `alphaedge-ai/Qwen3.5-2B-vie-32768` | Cần transformers ≥ 5.2 → service Docker `llm5` |

---

## Cài đặt và chạy

Yêu cầu: Python 3.11–3.12, [uv](https://docs.astral.sh/uv/), GPU NVIDIA (torch cài từ index CUDA 12.8; đổi sang `cu126` / `cpu` trong `pyproject.toml`). Đã chạy trên RTX 5060 Laptop 8 GB + 16 GB RAM.

### 1. Cài môi trường và dựng kho

```bash
uv sync
uv run vietnews load      # tải + loại trùng + chuẩn hoá VietNews -> OUTPUT_LOAD/
uv run vietnews index     # NER + chunk + embed + Qdrant         -> OUTPUT_INDEX/
```

Bước `index` mất khoảng 20 phút cho NER và 20–25 phút cho chunk + embed trên GPU. Có thể chạy riêng từng bước bằng `--steps ner,chunk,embed,index`.

> **Qdrant ở đâu?** Mặc định là Qdrant nhúng tại `OUTPUT_INDEX/qdrant`, không cần server. Nhưng kho có 132.850 chunk nên Qdrant khuyên dùng server. Muốn dùng server thì chạy `docker compose up -d qdrant` và đặt `RAG_QDRANT_URL=http://localhost:6333` **cho cả lúc index lẫn lúc chat**.

### 2. Bật llama-server (cho LLM mặc định Vistral)

Tải file GGUF vào `models/` và [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) vào `tools/llama-cpp/`, rồi mở một terminal riêng:

```bat
tools\llama-cpp\llama-server -m models\Vistral-7B-ChatML.i1-Q4_K_M.gguf -ngl 99 -c 8192 --port 8080
```

Cần ghi rõ `--port 8080`, vì bản llama-server mới không còn mặc định cổng này. Muốn đổi địa chỉ thì đặt `RAG_LLAMA_URL`. Nếu không muốn bật server thì chạy với `--llm qwen2.5-3b`.

### 3. Chat

**Windows CMD** (không để dấu cách quanh `=`):

```bat
chcp 65001
set RAG_QDRANT_URL=http://localhost:6333
uv run vietnews chat
```

**PowerShell:**

```powershell
$env:RAG_QDRANT_URL = "http://localhost:6333"; uv run vietnews chat
```

Bỏ dòng `RAG_QDRANT_URL` nếu dùng Qdrant nhúng. Các cách chạy khác:

```bash
uv run vietnews ask "Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam"   # hỏi một câu rồi thoát
uv run vietnews chat --llm vi-qwen2-3b-rag --llm-4bit          # LLM khác, nạp 4-bit
uv run vietnews chat --search dense --verbose                  # chỉ dense, in từng bước
```

---

## Chạy bằng Docker (GPU)

```bat
chcp 65001
docker compose build
docker compose up -d qdrant
docker compose run --rm rag index                       # dựng kho vào service qdrant
docker compose run --rm rag                             # chat (gọi llama-server trên máy host)
docker compose run --rm rag ask "Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam"
```

- Container gọi llama-server trên máy host qua `http://host.docker.internal:8080`. Vì vậy server phải được bật với thêm `--host 0.0.0.0`.
- Index dựng bằng Docker nằm trong volume `qdrant_data`, **không** nằm trong `OUTPUT_INDEX/qdrant`. Muốn chat bằng `uv` trên index này thì đặt `RAG_QDRANT_URL=http://localhost:6333`.
- Các LLM cần phiên bản transformers khác chạy trong service riêng:
  `docker compose --profile llm5 run --rm llm5 answer --llm qwen3.5-2b-vie`,
  `docker compose --profile phogpt run --rm phogpt answer --llm phogpt-4b-chat --llm-4bit`.

---

## Lệnh CLI (`vietnews <lệnh>`)

| Lệnh | Mô tả | Tham số hữu ích |
|---|---|---|
| `load` | Tải VietNews, loại trùng, chuẩn hoá | `--check`, `--source DIR` |
| `index` | NER + chunk + embed + Qdrant | `--steps ner,chunk,embed,index`, `--header` |
| `chat` | Chatbot: hỏi đáp, liệt kê, tóm tắt | `--llm`, `--llm-4bit`, `--search hybrid\|dense\|sparse`, `--verbose` |
| `ask "..."` | Hỏi một câu rồi thoát | như `chat` |
| `questions` | Sinh N câu hỏi đánh giá bằng LLM | `--n 10`, `--seed`, `--eval` |
| `evaluate` | Chấm RAG đầu-cuối trên file câu hỏi | `--name`, `--rescore` |
| `retrieval` | hit@k / MRR của dense, sparse, hybrid | `--modes`, `--no-rerank` |
| `llms` | Liệt kê LLM có sẵn | |
| `probe` | Chẩn đoán 1 LLM (logits NaN, top-5 token) | `--llm` |
| `contexts` → `answer` → `report` | So sánh nhiều LLM trên **cùng** ngữ cảnh | `--llm`, `--llms` |
| `compare` | Chạy cả `contexts` + `answer` + `report` | `--llms all`, `--refresh` |

Thư mục đổi được bằng `--load-dir`, `--index-dir`, `--questions-dir` (đặt trước tên lệnh) hoặc biến môi trường `RAG_LOAD_DIR`, `RAG_INDEX_DIR`, `RAG_QUESTIONS_DIR`.

---

## Kết quả đánh giá

Bộ đánh giá gồm 100 câu hỏi. Các câu được sinh tự động từ 100 bài khác nhau trong kho (10 bộ × 10 câu, seed 1–10), mỗi câu kèm đáp án ngắn chép nguyên văn từ bài.

### Truy xuất

Chỉ số **hit@k** tính theo bài: bài nguồn có nằm trong k bài khác nhau đầu tiên hay không. Mỗi chế độ lấy top-20 chunk.

| Chế độ | Giai đoạn | hit@1 | hit@3 | hit@5 | hit@20 | MRR |
|---|---|---|---|---|---|---|
| dense | retriever | 0,62 | 0,78 | 0,80 | 0,90 | 0,709 |
| dense | + rerank | 0,78 | 0,85 | 0,87 | 0,90 | 0,822 |
| sparse (BM25) | retriever | 0,87 | 0,93 | 0,95 | 0,99 | 0,906 |
| sparse (BM25) | + rerank | 0,85 | 0,95 | 0,96 | 0,99 | 0,903 |
| **hybrid** | retriever | 0,78 | 0,95 | 0,95 | 0,98 | 0,864 |
| **hybrid** | **+ rerank** | **0,86** | **0,96** | 0,96 | 0,98 | **0,909** |

- Dense đơn thuần (vietnamese-bi-encoder) yếu hơn BM25 rất nhiều trên tin tức: câu hỏi chứa tên riêng và con số, nên khớp từ khoá chính xác hơn khớp ngữ nghĩa. Hybrid cộng với rerank cho kết quả tốt nhất.
- Pipeline thật dùng top-5 chunk. Trên 100 câu, ngữ cảnh đưa cho LLM chứa bài đúng ở vị trí đầu 89% số câu, và trong 3 bài 94% số câu. Thử tăng lên top-20 thì hit@3 lên 0,96 nhưng hit@1 giảm xuống 0,86, nên giữ top-5.

### Đầu-cuối (Vistral-7B-Chat Q4_K_M)

| hit@1 | hit@3 | contains | recall | F1 | từ chối | số từ TB |
|---|---|---|---|---|---|---|
| 0,88 | 0,94 | **0,41** | 0,52 | 0,18 | 0,23 | 36 |

Điểm yếu chính là **LLM**, không phải phần truy xuất. Có 94/100 câu mà bài đúng đã nằm trong ngữ cảnh, nhưng Vistral chỉ trả lời đúng 43% trong số đó. Có tới 23% số câu Vistral trả lời "không biết".

Kết quả so sánh LLM trên **cùng 100 câu, cùng ngữ cảnh** (`vietnews compare`, đo ngày 15/10/2026 với bộ truy xuất cũ bge-m3 + bge-reranker-v2-m3, LLM chạy 4-bit, hit@3 = 0,95):

| LLM | contains | recall | từ chối | giây/câu |
|---|---|---|---|---|
| vi-qwen2-3b-rag | **0,68** | 0,80 | 0,01 | 4,6 |
| qwen2.5-3b | 0,60 | 0,74 | 0,01 | 4,5 |
| qwen3.5-2b-vie | 0,49 | 0,66 | 0,00 | 1,9 |
| vistral-7b-chat | 0,42 | 0,55 | 0,20 | 7,4 |
| vinallama-2.7b-chat | 0,32 | 0,56 | 0,00 | 17,2 |

Muốn độ chính xác cao hơn thì chạy `--llm vi-qwen2-3b-rag --llm-4bit`.

Chỉ số **contains** cho biết câu trả lời có chứa trọn đáp án chuẩn hay không (sau khi chuẩn hoá ngày tháng, dấu câu). Kết quả chi tiết từng câu nằm trong `OUTPUT_QUESTIONS/run_20261017/`.

---

## Cấu hình chính (`src/text_summarization/config.py`)

| Nhóm | Tham số |
|---|---|
| Chunking | `SEMANTIC_CHUNKING = True`, `SEMANTIC_CHUNK_THRESHOLD = 0.75`, chunk 64–254 token (`CHUNK_SIZE = 254`) |
| Truy xuất | `SEARCH_MODE = 'hybrid'`, `PREFETCH = 20`, `TOP_K = 5`, `N_ARTICLES = 3`, `CTX_BUDGET = 3000` |
| Tóm tắt | `SUM_MAX_INPUT = 1024`, `SUM_MAX_OUTPUT = 256`, `SUM_MIN_TEXT = 300` |
| NER | `NER_WINDOW = 400` |

| Biến môi trường | Ý nghĩa | Mặc định |
|---|---|---|
| `RAG_QDRANT_URL` | Qdrant server | trống = Qdrant nhúng `OUTPUT_INDEX/qdrant` |
| `RAG_LLAMA_URL` | llama-server | `http://localhost:8080` |
| `RAG_LOAD_DIR`, `RAG_INDEX_DIR`, `RAG_QUESTIONS_DIR` | Thư mục dữ liệu | `OUTPUT_LOAD`, `OUTPUT_INDEX`, `OUTPUT_QUESTIONS` |

**Đổi mô hình embedding thì phải dựng lại index** bằng `vietnews index --steps chunk,embed,index`. **Sửa NER thì xoá `OUTPUT_INDEX/entities.json`** rồi chạy `vietnews index --steps ner`, vì bước NER bỏ qua những bài đã có kết quả.

---

## Xử lý sự cố

| Lỗi | Nguyên nhân / cách sửa |
|---|---|
| `Qdrant ... (71594) và chunks.pkl (132850) không khớp` | Đang đọc nhầm Qdrant. Index dựng bằng Docker thì đặt `RAG_QDRANT_URL=http://localhost:6333`; nếu không thì chạy `vietnews index --steps index`. |
| `Không kết nối được llama-server tại http://localhost:8080` | Server chưa bật, hoặc bật mà thiếu `--port 8080`. |
| `The paging file is too small (os error 1455)` | Windows hết bộ nhớ ảo. Đóng bớt ứng dụng, tăng pagefile, hoặc giảm `-c` của llama-server xuống 4096. |
| Tiếng Việt gõ vào bị lỗi, "Tóm tắt"/"Liệt kê" bị hiểu thành hỏi đáp | CMD chưa chạy `chcp 65001`. |
| `Vector câu hỏi 768 chiều nhưng collection 1024 chiều` | Index dựng bằng mô hình embedding khác. Chạy `vietnews index --steps chunk,embed,index`. |

---

## Cấu trúc dự án

```
src/text_summarization/
├── cli.py         # vietnews <lệnh>
├── config.py      # đường dẫn, tên mô hình, tham số
├── data.py        # tải + loại trùng VietNews, đọc/ghi JSON
├── text.py        # chuẩn hoá văn bản, làm sạch input terminal, so khớp đáp án
├── ner.py         # NER tiêu đề + sapo + thân bài, gộp tên tắt vào họ tên đầy đủ
├── indexing.py    # semantic chunking -> embed -> Qdrant
├── sparse.py      # vector thưa BM25 (unigram + bigram âm tiết)
├── store.py       # QdrantStore: dense / sparse / hybrid (RRF)
├── models.py      # embedding, re-ranker, ViT5, LLM (nạp lười)
├── llms.py        # danh sách LLM, prompt từng mô hình, llama-server
├── router.py      # phân loại intent
├── pipeline.py    # retrieve -> re-rank -> ngữ cảnh -> trả lời / liệt kê / tóm tắt
├── questions.py   # sinh câu hỏi đánh giá, chấm hit@k, MRR, contains
├── metrics.py     # contains, recall, F1, EM, tỉ lệ từ chối
└── compare.py     # so sánh nhiều LLM trên cùng ngữ cảnh
tests/             # pytest, không cần GPU / mô hình thật
Dockerfile, Dockerfile.llm5, Dockerfile.phogpt, docker-compose.yml
pyproject.toml, uv.lock, requirements.txt (cho Docker)

# Sinh ra khi chạy, không commit:
OUTPUT_LOAD/       data_article.json
OUTPUT_INDEX/      entities.json, faiss_index/{chunks.pkl, embedds.npy}, qdrant/
OUTPUT_QUESTIONS/  câu hỏi, kết quả chấm, báo cáo so sánh
models/            file GGUF
tools/llama-cpp/   llama-server
```

## Test

```bash
uv run --group dev pytest
```
