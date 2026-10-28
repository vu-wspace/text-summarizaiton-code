# VietNews RAG: Hỏi đáp · Tóm tắt · Trích xuất thực thể

Hệ thống RAG (Retrieval-Augmented Generation) cho tin tức tiếng Việt, xây trên kho **VietNews (22.492 bài, tập test sau khi loại trùng)**. Chạy hoàn toàn cục bộ trên một GPU NVIDIA cá nhân (8 GB VRAM là đủ), bằng `uv` hoặc Docker.

Một câu lệnh `vietnews` lo toàn bộ vòng đời: tải dữ liệu → dựng chỉ mục → chatbot → sinh bộ câu hỏi đánh giá → đo truy xuất → so sánh nhiều LLM trên cùng ngữ cảnh → chấm LLM-as-a-judge.

---

## 📑 Mục lục

1. [Ba chức năng chính](#-ba-chức-năng-chính)
2. [Bắt đầu nhanh](#-bắt-đầu-nhanh)
3. [Kiến trúc pipeline](#-kiến-trúc-pipeline)
4. [Chi tiết từng thành phần](#-chi-tiết-từng-thành-phần)
5. [Các LLM được hỗ trợ](#-các-llm-được-hỗ-trợ)
6. [Cài đặt & chạy bằng uv](#-cài-đặt--chạy-bằng-uv)
7. [Chạy bằng Docker](#-chạy-bằng-docker--docker-compose)
8. [Cấu hình](#️-cấu-hình)
9. [Danh mục lệnh CLI](#-danh-mục-lệnh-cli)
10. [Quy trình đánh giá & benchmark](#-quy-trình-đánh-giá--benchmark)
11. [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
12. [Cấu trúc thư mục & file sinh ra](#-cấu-trúc-thư-mục--file-sinh-ra)
13. [Kiểm thử](#-kiểm-thử)
14. [Xử lý sự cố](#-xử-lý-sự-cố)

---

## 🎯 Ba chức năng chính

Router phân loại câu nhập bằng regex ở **đầu câu** (không phân biệt hoa thường, chấp nhận tiền tố "Hãy"). Mọi câu không khớp đều là hỏi đáp.

| Câu nhập | Intent | Hành vi | Ví dụ |
|---|---|---|---|
| `<câu hỏi bất kỳ>` | `qa` | Truy xuất → re-rank → ghép tối đa 3 bài → LLM trả lời | `Vụ 2 tàu đối đầu ở Quảng Nam xảy ra như thế nào?` |
| `Liệt kê ...` / `Trích xuất ...` | `list` | Tìm bài liên quan nhất, trả về thực thể (người, địa điểm, tổ chức, khác) đã tính sẵn lúc index | `Liệt kê người trong vụ Hồ Xuân Huy lừa đảo` |
| `Tóm tắt <chủ đề ngắn>` | `summarize_topic` | Tìm bài liên quan nhất rồi tóm tắt bằng ViT5 | `Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam` |
| `Tóm tắt: <văn bản ≥ 300 ký tự>` | `summarize_text` | Tóm tắt trực tiếp đoạn văn bản dán vào (không truy xuất) | `Tóm tắt: Ngày 15-10, Tòa án nhân dân tỉnh...` |

- Với `Liệt kê`, có thể lọc theo loại bằng các từ đứng ngay sau: *người / nhân vật / tên* → PERSON, *địa điểm / địa danh / nơi* → LOCATION, *tổ chức / cơ quan / công ty* → ORGANIZATION; ghép được nhiều loại (`Liệt kê người và tổ chức trong ...`). Không nêu loại thì trả về tất cả. Phần chỉ loại bị bỏ trước khi tìm bài, nên câu tìm kiếm chỉ còn chủ đề.
- `Tóm tắt` cũng nhận `Tóm lược` và `tom tat` (không dấu). Ngưỡng 300 ký tự phân biệt "chủ đề" với "văn bản" (`SUM_MIN_TEXT`).
- Mỗi câu trả lời kèm danh sách bài nguồn **thực sự được dùng** (`[guid] tiêu đề`).

---

## 🚀 Bắt đầu nhanh

```bash
uv sync                      # cài môi trường (PyTorch CUDA 12.8)
uv run vietnews load         # tải + làm sạch VietNews  -> OUTPUT_LOAD/data_article.json
uv run vietnews index        # NER + chunk + embed + Qdrant nhúng -> OUTPUT_INDEX/
uv run vietnews chat         # chatbot (LLM mặc định vi-qwen2-3b-rag, 4-bit)
```

Hoặc với Docker:

```bash
docker compose build
docker compose up -d qdrant
docker compose run --rm rag index
docker compose run --rm rag          # chat
```

> Lần chạy đầu sẽ tải các mô hình từ Hugging Face (tổng ~15 GB) và chạy NER + embedding cho 22.492 bài, mất khá lâu. Các lần sau dùng lại cache.

---

## 🏗 Kiến trúc pipeline

```
                              Câu truy vấn
                                   │
                   Router (regex): qa / list / summarize
        ┌──────────────────────────┼─────────────────────────┐
        ▼                          ▼                         ▼
   [Hỏi đáp RAG]              [Liệt kê thực thể]         [Tóm tắt]
        │                          │                         │
 Qdrant Hybrid Search        Bài viết top-1            Bài viết top-1  ◄── (hoặc văn bản dán vào)
 (Dense PhoBERT + BM25,            │                         │
  mỗi nhánh 20 → RRF)      Đọc entities.json          Chuẩn hóa văn bản
        │                  (NER tính sẵn lúc index)          │
   Top-5 chunks                                              ▼
        │                                              ViT5 Summarizer
 Vietnamese_Reranker                                  (beam 4, ≤ 256 token)
        │
 Small-to-big: 3 bài khác nhau
 (ghép nguyên văn, ≤ 3.000 token)
        │
        ▼
   LLM tiếng Việt (vi-qwen2-3b-rag 4-bit, greedy)
```

Luồng **dựng chỉ mục** (lệnh `index`, 4 bước chạy độc lập được):

```
data_article.json ──► ner   ──► entities.json            {guid: {PERSON: [...], LOCATION: [...], ...}}
                  ──► chunk ──► vectors/chunks.pkl        (semantic chunking, ≤ 254 token)
                  ──► embed ──► vectors/embedds.npy       (768 chiều, đã chuẩn hoá)
                  ──► index ──► Qdrant collection "vietnews" (vector dense + vector thưa BM25)
```

`chunks.pkl`, `embedds.npy` và collection Qdrant phải cùng một danh sách chunk theo cùng thứ tự (point id = vị trí chunk). Mỗi bước kiểm tra số lượng và báo lỗi nếu lệch.

### Các mô hình trong hệ thống

| Vai trò | Mô hình | Ghi chú |
|---|---|---|
| Embedding | `bkai-foundation-models/vietnamese-bi-encoder` | PhoBERT, 768 chiều, tối đa 256 token, đầu vào phải tách từ bằng `pyvi` (`Hà Nội` → `Hà_Nội`) |
| Kho vector | `Qdrant` v1.19 | Dense (cosine) + sparse BM25, gộp Reciprocal Rank Fusion ngay trong Qdrant |
| Re-ranker | `AITeamVN/Vietnamese_Reranker` | bge-reranker-v2-m3 fine-tune tiếng Việt, CrossEncoder, fp16 trên GPU (~1,1 GB) |
| Tóm tắt | `VietAI/vit5-large-vietnews-summarization` | bf16 nếu GPU hỗ trợ, nếu không thì fp32 (T5 tràn số ở fp16) |
| NER | `NlpHUST/ner-vietnamese-electra-base` | Chạy một lần lúc index |
| LLM sinh câu trả lời | Mặc định `AITeamVN/Vi-Qwen2-3B-RAG` | 4-bit NF4 (`bitsandbytes`) mặc định; xem [danh sách LLM](#-các-llm-được-hỗ-trợ) |

Mọi mô hình đều **nạp lười**: chỉ nạp khi lần đầu cần dùng, nên hỏi đáp không chiếm VRAM của ViT5 và ngược lại.

---

## 🔬 Chi tiết từng thành phần

### 1. Tải & chuẩn hoá dữ liệu (`vietnews load`)

- Đọc tập **test** của `nam194/vietnews` (parquet) trực tiếp từ Hugging Face, hoặc từ thư mục local qua `--source`. Tập train/validation không dùng.
- Loại bài trùng `title`, trùng `abstract` và bài thiếu trường → **22.492 bài**.
- Chuẩn hoá (`core/text.py::normalize_text`, dùng **cùng một hàm** lúc index và lúc truy vấn):
  - Unicode NFC, xoá ký tự vô hình (zero-width, soft hyphen, bidi) và dấu tổ hợp sót lại.
  - Bỏ chú thích ảnh ở cuối bài (phần sau 2+ khoảng trắng).
  - Bỏ dấu `_` của bản tách từ gốc, sửa biển số xe (`77 AA - 123 . 45` → `77AA-123.45`), `v.v.`, chữ viết tắt (`TP .` → `TP.`), khoảng trắng quanh dấu câu, nháy kép và `/`.
- Kết quả: `OUTPUT_LOAD/data_article.json` dạng `{guid: {title, abstract, article}}`, ghi nguyên tử (ghi `.tmp` rồi đổi tên).
- `vietnews load --check`: chỉ đếm các mẫu lẽ ra phải hết sau chuẩn hoá trên 2.000 bài ngẫu nhiên, không ghi file.

### 2. Trích xuất thực thể (bước `ner`)

- Văn bản đưa vào NER = **tiêu đề + sapo + thân bài**: báo thường chỉ nêu họ tên đầy đủ ở sapo, thân bài gọi tắt.
- ELECTRA giới hạn 512 token nên bài được chia theo câu thành các cửa sổ ≤ 400 token (`NER_WINDOW`); câu quá dài cắt theo từ. Cửa sổ của nhiều bài gom vào một lô để chạy nhanh.
- Lọc thực thể có điểm < 0,6 hoặc dài < 2 ký tự; **gộp bí danh** cùng loại (`Huy` gộp vào `Hồ Xuân Huy`); sắp theo số lần xuất hiện.
- Lưu định kỳ mỗi 1.000 bài và **chạy tiếp được** nếu bị ngắt (bỏ qua bài đã có trong `entities.json`).

### 3. Chia chunk (bước `chunk`)

Mặc định dùng **semantic chunking** (`SEMANTIC_CHUNKING = True`):

1. Ghép `tiêu đề . thân bài`, tách câu theo `. ? ! …`.
2. Tách từ bằng `pyvi`, đếm token bằng tokenizer của vietnamese-bi-encoder. Câu dài hơn giới hạn được chia đều theo từ.
3. Embed mọi câu (theo khối 1.000 bài), tính cosine giữa hai câu liền kề.
4. Gộp câu liên tiếp thành chunk; **cắt chunk** khi cosine < 0,75 *và* chunk đã ≥ 64 token, hoặc khi thêm câu sẽ vượt 254 token.

Giới hạn 254 token là vì embedding chỉ nhận 256 token (gồm 2 token đặc biệt); chunk dài hơn sẽ bị cắt mất phần sau khi embed. Bản tách từ của chunk được giữ lại để bước `embed` không phải chạy `pyvi` lần nữa.

Tắt semantic chunking thì dùng `RecursiveCharacterTextSplitter` đo theo token, `--chunk-size 254 --chunk-overlap 30`, ưu tiên cắt ở ranh giới câu. Tuỳ chọn `--header` thêm một chunk `tiêu đề . sapo` mỗi bài (**không** bật khi đánh giá bằng câu hỏi sinh từ sapo vì lộ đáp án).

### 4. Đánh chỉ mục (bước `embed` + `index`)

- **Dense**: vector 768 chiều đã chuẩn hoá, khoảng cách cosine.
- **Sparse BM25** (`rag/sparse.py`): term = **unigram + bigram âm tiết** (`bất_động`, `động_sản`) vì tiếng Việt viết theo âm tiết, bigram giữ được từ ghép và tên riêng. Term băm `crc32` thành chỉ số. Phần TF (k1 = 1,2, b = 0,75) tính lúc index, phần IDF để Qdrant tính trên toàn kho (`Modifier.IDF`), nên thêm chunk mới không phải tính lại vector cũ. Câu hỏi: mỗi term trọng số 1.
- Qdrant chạy **nhúng** (lưu tại `OUTPUT_INDEX/qdrant`, không cần server) nếu không đặt `RAG_QDRANT_URL`, hoặc kết nối **server** (service `qdrant` trong docker-compose).

### 5. Truy xuất & re-rank (nhánh hỏi đáp)

1. Chuẩn hoá câu hỏi (giữ chữ hoa cho tên riêng).
2. **Hybrid**: mỗi nhánh (dense, BM25) lấy 20 chunk (`PREFETCH`), gộp RRF, giữ **top-5 chunk** (`TOP_K`). Có thể đổi sang `--search dense` hoặc `--search sparse`.
3. **Re-rank** 5 chunk bằng Vietnamese_Reranker, sắp lại theo điểm.
4. **Small-to-big**: duyệt chunk theo thứ tự, lấy **3 bài khác nhau** (`N_ARTICLES`).
5. **Ghép ngữ cảnh**: nguyên văn `tiêu đề . thân bài` của từng bài, đánh số `[1] [2] [3]`, trong ngân sách **3.000 token** đo bằng tokenizer của LLM (`CTX_BUDGET`). Bài đầu luôn được giữ (cắt bớt nếu quá dài); bài sau không còn chỗ thì bỏ.
6. **Sinh câu trả lời**: giải mã greedy, `repetition_penalty = 1.05`, tối đa 256 token mới. Ngữ cảnh rỗng thì trả về *"Tôi không biết dựa trên các tài liệu được cung cấp."*

Hai nhánh `list` và `summarize_topic` dùng cùng bước 1–3 nhưng chỉ lấy **bài top-1**.

### 6. Tóm tắt

ViT5 nhận `"vietnews: " + văn bản`, cắt đầu vào ở 1.024 token, sinh tối đa 256 token với beam search 4, `no_repeat_ngram_size = 3`.

---

## 🤖 Các LLM được hỗ trợ

Xem danh sách trong môi trường hiện tại: `uv run vietnews llms`. Thêm mô hình mới = thêm một dòng `LLMSpec` vào `LLMS` trong `src/text_summarization/rag/llms.py`.

| Key (`--llm`) | Repo | Prompt | Cách nạp | Ghi chú |
|---|---|---|---|---|
| `vi-qwen2-3b-rag` *(mặc định)* | `AITeamVN/Vi-Qwen2-3B-RAG` | `vi-rag-concise` | transformers 4.x | Qwen2 3B fine-tune cho RAG tiếng Việt; khuôn prompt AITeamVN + yêu cầu trả lời một câu ngắn |
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` | `ours` | transformers 4.x | Đa ngôn ngữ, không fine-tune riêng cho RAG |
| `qwen3.5-2b-vie` | `alphaedge-ai/Qwen3.5-2B-vie-32768` | `ours` | **transformers ≥ 5.2** (service `llm5`) | Qwen3.5-2B rút gọn từ vựng còn tiếng Việt (1,77B tham số), tắt chế độ thinking |
| `vinallama-2.7b-chat` | `vilm/vinallama-2.7b-chat` | `ours` | transformers 4.x | Khuôn ChatML, ngữ cảnh 4.096 token (tự cắt ngữ cảnh) |
| `vistral-7b-chat` | file GGUF `Q4_K_M` | `ours` | **llama-server** (llama.cpp) | Repo gốc bị khoá nên chạy bản GGUF qua HTTP (`RAG_LLAMA_URL`, mặc định `:8080`) |
| `vietnamese-llama2-7b` | `bkai-foundation-models/vietnamese-llama2-7b-40GB` | `ours` | transformers 4.x | Mô hình **nền**, không phải chat; repo khoá (cần HF token) |

Ba prompt có sẵn (`PROMPTS` trong `llms.py`):

- **`ours`**: prompt của dự án, có quy tắc chỉ dùng ngữ cảnh, câu từ chối cố định, giữ nguyên tên riêng / số liệu / ngày tháng.
- **`vi-rag`**: đúng khuôn AITeamVN công bố cho dòng Vi-Qwen2-RAG.
- **`vi-rag-concise`**: `vi-rag` + yêu cầu *"Trả lời bằng MỘT câu ngắn, không mở đầu bằng 'Theo ngữ cảnh'..."*.

Vì sao `transformers < 5` ở môi trường chính: transformers 5 làm hỏng tokenizer ViT5. Qwen3.5 lại cần ≥ 5.2, nên bước `answer` của nó chạy trong image riêng `Dockerfile.llm5` (chỉ đọc `contexts.jsonl`, không cần ViT5 / embedding / Qdrant).

**Chẩn đoán một LLM** (logits có NaN/inf không, top-5 token kế tiếp, câu trả lời ngắn):

```bash
uv run vietnews probe --llm vinallama-2.7b-chat "Thủ đô của Việt Nam là gì?"
```

---

## ⚡ Cài đặt & chạy bằng uv

### Yêu cầu

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.11 – 3.12 |
| Trình quản lý gói | [uv](https://docs.astral.sh/uv/) |
| GPU | NVIDIA, PyTorch CUDA 12.8 (RTX 20/30/40/50). 8 GB VRAM đủ cho cấu hình mặc định (LLM 4-bit) |
| Ổ đĩa | ~15 GB cho mô hình (cache Hugging Face) + dữ liệu và chỉ mục |

Không có GPU NVIDIA vẫn chạy được trên CPU (rất chậm): đổi index PyTorch trong `pyproject.toml` (`[tool.uv.sources]`) sang `cpu`; 4-bit tự tắt khi không có CUDA. Driver cũ với RTX 20/30/40 có thể dùng `cu126`.

### 1. Cài đặt và dựng dữ liệu

```bash
uv sync                                   # cài thư viện
uv run vietnews load                      # -> OUTPUT_LOAD/data_article.json
uv run vietnews index                     # -> OUTPUT_INDEX/ (mặc định Qdrant nhúng)
```

Chạy lại từng phần khi cần:

```bash
uv run vietnews index --steps ner                    # chỉ NER (chạy tiếp nếu bị ngắt)
uv run vietnews index --steps chunk,embed,index      # bắt buộc sau khi đổi mô hình embedding / tham số chunk
uv run vietnews index --steps index                  # chỉ nạp lại Qdrant từ chunks.pkl + embedds.npy có sẵn
```

Muốn index vào Qdrant **server** (vd. container `qdrant`) thì đặt `RAG_QDRANT_URL` *trước khi* chạy `index`, và đặt lại cùng giá trị khi chat.

### 2. Sử dụng

```bash
uv run vietnews chat                                       # chatbot, gõ "exit" để thoát
uv run vietnews ask "Hà Nội ban hành quy tắc ứng xử khi nào?"
uv run vietnews chat --llm qwen2.5-3b                      # LLM nhanh hơn
uv run vietnews chat --no-llm-4bit                         # bf16 đầy đủ (GPU ≥ 12 GB)
uv run vietnews chat --search sparse --verbose             # chỉ BM25, in từng bước
```

Dùng Qdrant server (index đã dựng bằng Docker):

```powershell
# Windows PowerShell
$env:RAG_QDRANT_URL = "http://localhost:6333"; uv run vietnews chat
```

```bat
:: Windows CMD (chcp 65001 để gõ được tiếng Việt)
chcp 65001
set RAG_QDRANT_URL=http://localhost:6333
uv run vietnews chat
```

Ví dụ một phiên chat:

```
Kho: 22492 bài, ... chunk | tìm: hybrid | LLM: vi-qwen2-3b-rag | device=cuda
Sẵn sàng. Gõ "exit" để thoát.

Bạn: Liệt kê người trong vụ Hồ Xuân Huy lừa đảo

[list] Trả lời: Bài: <tiêu đề bài>
  Người: Hồ Xuân Huy, ...
  - [<guid>] <tiêu đề bài>
```

---

## 🐳 Chạy bằng Docker & Docker Compose

`docker-compose.yml` gồm ba service:

| Service | Image | Vai trò |
|---|---|---|
| `qdrant` | `qdrant/qdrant:v1.19.1` | Kho vector, cổng `6333`, dữ liệu ở volume `qdrant_data` |
| `rag` | `Dockerfile` (Python 3.12, torch 2.8 cu128, transformers 4.x) | Toàn bộ pipeline; mặc định chạy `chat` |
| `llm5` | `Dockerfile.llm5` (transformers 5.x), profile `llm5` | Chỉ bước `answer` cho `qwen3.5-2b-vie` |

Mô hình **không** đóng vào image mà tải vào volume `hf_cache`, dùng lại giữa các lần chạy. `OUTPUT_LOAD/`, `OUTPUT_INDEX/`, `OUTPUT_QUESTIONS/` được mount từ máy host. Cần NVIDIA Container Toolkit (hoặc Docker Desktop với WSL2 GPU).

```bash
# 1. Khởi động Qdrant và build image
docker compose up -d qdrant
docker compose build

# 2. Dữ liệu + chỉ mục (load ghi vào ./OUTPUT_LOAD trên host)
docker compose run --rm rag load
docker compose run --rm rag index

# 3. Sử dụng
docker compose run --rm rag                                        # chat với vi-qwen2-3b-rag
docker compose run --rm rag ask "Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam"
docker compose run --rm rag chat --llm qwen2.5-3b
docker compose run --rm rag --help
```

**Vistral-7B (GGUF)** chạy bằng llama-server trên máy host, container gọi qua `host.docker.internal:8080`:

```bash
tools\llama-cpp\llama-server -m models\Vistral-7B-ChatML.i1-Q4_K_M.gguf -ngl 99 -c 8192 --port 8080 --host 0.0.0.0
docker compose run --rm rag chat --llm vistral-7b-chat
```

**Qwen3.5-2B** (cần transformers 5.x) chỉ chạy được bước `answer`, xem [quy trình benchmark](#-quy-trình-đánh-giá--benchmark).

---

## ⚙️ Cấu hình

### Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `RAG_QDRANT_URL` | *(trống)* | URL Qdrant server, vd. `http://localhost:6333`. Trống = Qdrant nhúng tại `OUTPUT_INDEX/qdrant` |
| `RAG_LOAD_DIR` | `OUTPUT_LOAD` | Thư mục dữ liệu bài viết |
| `RAG_INDEX_DIR` | `OUTPUT_INDEX` | Thư mục chỉ mục |
| `RAG_QUESTIONS_DIR` | `OUTPUT_QUESTIONS` | Thư mục câu hỏi và kết quả đánh giá |
| `RAG_LLAMA_URL` | `http://localhost:8080` | llama-server cho `vistral-7b-chat` |

Ba thư mục cũng đổi được bằng tuỳ chọn toàn cục `--load-dir`, `--index-dir`, `--questions-dir` (đặt **trước** tên lệnh).

### Tham số trong `src/text_summarization/core/config.py`

Mọi module đọc hằng số từ đây, không hard-code.

| Nhóm | Hằng số | Giá trị | Ý nghĩa |
|---|---|---|---|
| Mô hình | `DEFAULT_LLM` | `vi-qwen2-3b-rag` | LLM mặc định |
| | `LLM_4BIT` | `True` | Nạp LLM 4-bit (bf16 ~6,5 GB + các mô hình khác vượt GPU 8 GB) |
| Chunking | `CHUNK_SIZE` | 254 | Token tối đa / chunk |
| | `CHUNK_OVERLAP` | 30 | Chỉ dùng khi tắt semantic chunking |
| | `SEMANTIC_CHUNKING` | `True` | Bật semantic chunking |
| | `SEMANTIC_CHUNK_THRESHOLD` | 0,75 | Ngưỡng cosine để cắt chunk |
| | `SEMANTIC_MIN_CHUNK_TOKENS` | 64 | Chunk tối thiểu trước khi được cắt theo ngữ nghĩa |
| | `NER_WINDOW` | 400 | Cửa sổ NER (< 512 của ELECTRA) |
| Truy xuất | `SEARCH_MODE` | `hybrid` | `dense` / `sparse` / `hybrid` |
| | `PREFETCH` | 20 | Số chunk mỗi nhánh trước khi gộp RRF |
| | `TOP_K` | 5 | Chunk đưa vào re-ranker (k = 20 tăng hit@3 nhưng giảm hit@1) |
| | `N_ARTICLES` | 3 | Số bài khác nhau trong ngữ cảnh |
| | `CTX_BUDGET` | 3000 | Ngân sách token ngữ cảnh |
| Tóm tắt | `SUM_MAX_INPUT` / `SUM_MAX_OUTPUT` | 1024 / 256 | Giới hạn token của ViT5 |
| | `SUM_MIN_TEXT` | 300 | Độ dài tối thiểu để `Tóm tắt:` được coi là văn bản |

---

## 🛠 Danh mục lệnh CLI

Cú pháp: `vietnews [--load-dir D] [--index-dir D] [--questions-dir D] <lệnh> [tuỳ chọn]` (tương đương `python -m text_summarization`).

Các lệnh có nạp pipeline (`chat`, `ask`, `questions`, `evaluate`, `retrieval`, `contexts`, `answer`, `compare`, `probe`) dùng chung bộ tuỳ chọn:

| Tuỳ chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--llm <key>` | `vi-qwen2-3b-rag` | LLM sinh câu trả lời |
| `--llm-4bit` / `--no-llm-4bit` | bật | Nạp 4-bit NF4 hoặc bf16 |
| `--search dense\|sparse\|hybrid` | `hybrid` | Kiểu truy xuất |
| `--verbose` | tắt | In từng bước (nạp mô hình, truy xuất, re-rank...) |

| Nhóm | Lệnh | Ý nghĩa | Tuỳ chọn riêng | Đầu ra |
|---|---|---|---|---|
| Xây dựng | `load` | Tải, loại trùng, chuẩn hoá VietNews | `--source <dir>`, `--check` | `OUTPUT_LOAD/data_article.json` |
| | `index` | NER + chunk + embed + Qdrant | `--steps ner,chunk,embed,index`, `--chunk-size`, `--chunk-overlap`, `--header` | `OUTPUT_INDEX/` |
| Sử dụng | `chat` | Chatbot trong terminal | | |
| | `ask "..."` | Hỏi một câu rồi thoát | | |
| Đánh giá | `questions` | Sinh N câu hỏi đánh giá bằng LLM | `--n 10`, `--seed 42`, `--name`, `--eval` | `questions.jsonl` |
| | `evaluate` | Chấm RAG đầu-cuối (hit@1, hit@3, contains) | `--name`, `--rescore` | `questions_eval.jsonl` |
| | `retrieval` | hit@k / MRR của từng chế độ tìm, trước và sau re-rank | `--name`, `--modes dense,sparse,hybrid`, `--no-rerank` | `retrieval_eval.jsonl`, `retrieval_report.md` |
| So sánh LLM | `llms` | Liệt kê LLM và phiên bản transformers cần | | |
| | `probe ["câu hỏi"]` | Chẩn đoán một LLM | | |
| | `contexts` | Truy xuất một lần cho mọi câu hỏi | `--name` | `contexts.jsonl` |
| | `answer` | Một LLM trả lời trên `contexts.jsonl` | `--name` | `answers_<llm>.jsonl` + `.meta.json` |
| | `report` | Gộp và chấm mọi `answers_*.jsonl` | `--llms a,b`, `--exclude ambiguous.jsonl` | `compare_report.md`, `compare_summary.csv`, `compare_review.csv` (`*_clean` khi có `--exclude`) |
| | `compare` | `contexts` + `answer` cho nhiều LLM + `report` | `--llms all`, `--refresh`, `--name` | như trên |
| | `judge` | LLM-as-a-judge: faithfulness + relevancy, kèm câu kiểm soát | `--judge qwen2.5-3b`, `--answers <keys>`, `--controls 30`, `--llm-4bit` | `judge_<llm>/report.md` |

Ghi chú:
- `evaluate --rescore` chỉ chấm lại cột `contains` của file `*_eval.jsonl` có sẵn, không nạp mô hình.
- `compare` tự bỏ qua (và báo) LLM không chạy được trong môi trường hiện tại, vd. `qwen3.5-2b-vie` khi transformers < 5.2.
- `judge` không truyền `--answers` thì chấm mọi file `answers_*.jsonl` trong thư mục.

---

## 📐 Quy trình đánh giá & benchmark

### Sinh bộ câu hỏi (`questions`)

1. Chọn ngẫu nhiên (theo `--seed`) các chunk dài ≥ 400 ký tự, **mỗi bài tối đa 1 chunk**, lấy dư gấp 4 lần để bù câu bị loại.
2. LLM đặt một câu hỏi + đáp án ngắn **chép nguyên văn** từ chunk, trả về JSON.
3. Loại câu: quá ngắn, phụ thuộc đoạn văn ("bài báo", "ông ấy"...), **không có mỏ neo** (tên riêng hoặc con số), câu có/không, đáp án > 20 từ, đáp án không có trong chunk (LLM bịa), đáp án lộ trong câu hỏi.

### So sánh LLM công bằng (`contexts` → `answer` → `report`)

Tách ba bước để (a) mọi LLM nhận **ngữ cảnh giống hệt nhau**, khác biệt chỉ do LLM; (b) lúc sinh chỉ có một LLM trên GPU; (c) Qwen3.5 chạy bước `answer` trong môi trường transformers 5.

```bash
Q=OUTPUT_QUESTIONS/compare100
docker compose run --rm rag --questions-dir $Q contexts                       # truy xuất 1 lần
docker compose run --rm rag --questions-dir $Q answer --llm vi-qwen2-3b-rag
docker compose run --rm rag --questions-dir $Q answer --llm qwen2.5-3b
docker compose --profile llm5 run --rm llm5 --questions-dir $Q answer --llm qwen3.5-2b-vie
docker compose run --rm rag --questions-dir $Q report                         # -> compare_report.md
docker compose run --rm rag --questions-dir $Q report --exclude ambiguous.jsonl   # -> *_clean
docker compose run --rm rag --questions-dir $Q judge --judge vi-qwen2-3b-rag  # -> judge_<llm>/report.md
```

Với uv thay `docker compose run --rm rag` bằng `uv run vietnews`. Hoặc gộp tất cả: `uv run vietnews --questions-dir $Q compare --llms vi-qwen2-3b-rag,qwen2.5-3b`.

### Các chỉ số

| Chỉ số | Định nghĩa | Dùng để |
|---|---|---|
| `hit@k` | Bài nguồn của câu hỏi nằm trong k bài **khác nhau** đầu tiên | Đo truy xuất |
| `MRR` | Trung bình 1/hạng của bài nguồn (0 nếu không có trong top-20 chunk) | Đo truy xuất |
| `ans@1` / `ans@3` | Đáp án chuẩn có trong bài đầu / trong cả ngữ cảnh | Đo truy xuất theo điều LLM thực sự cần (VietNews có nhiều bài cùng sự kiện) |
| `contains` | Đáp án chuẩn (đã chuẩn hoá ngày tháng, ngoặc, dấu câu) nằm trong câu trả lời | **Chỉ số chính**, chịu được câu trả lời dài |
| `recall` | Tỉ lệ token của đáp án chuẩn có trong câu trả lời | Điểm từng phần (`Vũ Ngọc Long` vs `Long`) |
| `F1` | Token F1 kiểu SQuAD | Phạt trả lời dài dòng |
| `EM` | Khớp chính xác sau chuẩn hoá | Rất khắt khe với LLM |
| `abstain` | LLM từ chối / nói không biết | Có ngữ cảnh đúng mà từ chối = mất điểm |
| Faithfulness | Judge: mọi thông tin trong câu trả lời có trong ngữ cảnh không (0 / 0,5 / 1) | Phát hiện bịa |
| Relevancy | Judge: câu trả lời có đúng trọng tâm câu hỏi không (không nhìn đáp án chuẩn) | Tham khảo |

Kiểm định thống kê trong `report`: khoảng tin cậy **bootstrap** 95 % (lấy mẫu lại theo câu hỏi), **McNemar chính xác** từng cặp LLM, hiệu chỉnh **Holm-Bonferroni**.

**Câu kiểm soát của judge**: trước khi tin con số, judge chấm 30 câu *positive* (pred = đáp án chuẩn, có trong ngữ cảnh → kỳ vọng ≈ 1) và 30 câu *negative* (pred = đáp án của một câu hỏi khác → kỳ vọng ≈ 0). Judge viết lý do **trước** điểm, prompt không có điểm mẫu, và kết quả không đọc được bị loại khỏi trung bình (báo tỉ lệ) thay vì tính 0.

---

## 📊 Kết quả thực nghiệm

Bộ benchmark chính: **100 câu hỏi** (10 bộ × 10 câu sinh từ kho VietNews), `OUTPUT_QUESTIONS/compare100`. Bộ kiểm chứng: **110 câu held-out** (seed 2026, không trùng bài), `OUTPUT_QUESTIONS/heldout100`, không dùng để tinh chỉnh.

### 🏆 Top-3 LLM

Cùng ngữ cảnh truy xuất (hit@1 = 0,86, hit@3 = 0,96):

| Thứ hạng | Model LLM | Contains ↑ | Recall ↑ | F1 ↑ | EM ↑ | Tốc độ | VRAM (4-bit) | Đánh giá & Khuyến nghị |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 🥇 **Top 1** | **`vi-qwen2-3b-rag`** *(Mặc định)* | **0.67** | **0.81** | 0.16 | 0.00 | 6.10s | ~5.55 GB | **Độ chính xác cao nhất**, fine-tune cho RAG tiếng Việt bởi AITeamVN. Trả lời bằng câu hoàn chỉnh nên F1/EM thấp. |
| 🥈 **Top 2** | **`qwen2.5-3b`** | **0.59** | **0.74** | **0.35** | **0.06** | 3.06s | ~4.01 GB | **Cân bằng & súc tích**, nhanh gấp đôi Top 1, câu trả lời gọn, phù hợp khi cần phản hồi nhanh. |
| 🥉 **Top 3** | **`qwen3.5-2b-vie`** | 0.52 | 0.70 | **0.51** | **0.29** | **1.82s** | **~1.54 GB** | **Nhẹ & nhanh nhất**, F1 cao nhất, chỉ ~1,5 GB VRAM. Phù hợp GPU khiêm tốn. |

> 📌 *Các mô hình khác đã kiểm thử: `vistral-7b-chat` (GGUF, contains 0.43, từ chối 23% câu dù có ngữ cảnh), `vinallama-2.7b-chat` (contains 0.36, sinh dài dòng), `vietnamese-llama2-7b` (mô hình nền, không phù hợp hỏi đáp).*

### 1. Đánh giá truy xuất

100 câu hỏi, hit@k tính theo bài viết gốc trong top-20 chunk:

| Cơ chế tìm kiếm | Giai đoạn | hit@1 | hit@3 | hit@5 | hit@20 | MRR |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Dense đơn thuần | Retriever | 0.62 | 0.78 | 0.80 | 0.90 | 0.709 |
| Dense đơn thuần | + Re-ranker | 0.78 | 0.85 | 0.87 | 0.90 | 0.822 |
| Sparse (BM25) | Retriever | 0.87 | 0.93 | 0.95 | 0.99 | 0.906 |
| Sparse (BM25) | + Re-ranker | 0.85 | 0.95 | 0.96 | 0.99 | 0.903 |
| **Hybrid (Dense + BM25)** | Retriever | 0.78 | 0.95 | 0.95 | 0.98 | 0.864 |
| **Hybrid (Dense + BM25)** | **+ Re-ranker** | **0.86** | **0.96** | **0.96** | **0.98** | **0.909** |

*Nhận xét:* BM25 vượt trội dense với tin tức tiếng Việt vì chứa nhiều tên người, địa danh, số liệu cụ thể. Hybrid + Re-ranker cho MRR cao nhất (0.909).

**Thử tối ưu truy xuất (dev = 100 câu ở trên, held-out = 110 câu):** 23 cấu hình (độ sâu re-rank k = 5–50, RRF có trọng số, thêm tiêu đề bài khi re-rank, gộp điểm re-ranker + retriever). Không cấu hình nào hơn cấu hình hiện tại có ý nghĩa thống kê; cấu hình tốt nhất trên dev (hit@1 0.88) tụt còn 0.83 trên held-out (quá khớp). Cấu hình hiện tại ổn định: hit@1 **0.87 / 0.87**. Re-ranker là thành phần đóng góp rõ nhất (bỏ re-ranker: held-out hit@1 0.87 → 0.74, p = 0.001). → **Giữ nguyên cấu hình.**

Phần lớn các câu "trượt" hit@1 không phải lỗi truy xuất: VietNews có nhiều bài cùng sự kiện, bài top-1 khác bài nguồn nhưng **vẫn chứa đáp án** (6/13 câu trượt ở dev), số còn lại chủ yếu là câu hỏi mơ hồ do LLM sinh (vd. *"Khi nào ông Trump sa thải ông Comey?"* → *"Bốn tháng sau đó"*). Tỉ lệ bài top-1 chứa đáp án chuẩn là **0.93 / 0.94** (dev / held-out). Vì vậy `compare_report.md` báo thêm **ans@1 / ans@3** bên cạnh hit@k.

### 2. So sánh Top-3 LLM đầu-cuối

100 câu hỏi, **cùng một ngữ cảnh truy xuất**:

| Tiêu chí | 🥇 vi-qwen2-3b-rag | 🥈 qwen2.5-3b | 🥉 qwen3.5-2b-vie |
|---|:---:|:---:|:---:|
| **Contains (chính xác)** | **0.67** | 0.59 | 0.52 |
| **Recall** | **0.81** | 0.74 | 0.70 |
| **F1** | 0.16 | 0.35 | **0.51** |
| **EM** | 0.00 | 0.06 | **0.29** |
| **Tỉ lệ từ chối** | **0.00** | **0.00** | **0.00** |
| **Thời gian TB** | 6.10s / câu | 3.06s / câu | **1.82s / câu** |
| **Số từ TB** | 55.4 từ | 17.9 từ | 15.7 từ |
| **VRAM (4-bit)** | ~5.55 GB | ~4.01 GB | **~1.54 GB** |

- **`vi-qwen2-3b-rag`** bóc tách đầy đủ dữ kiện ngữ cảnh thành câu trả lời tự nhiên.
- **`qwen2.5-3b`** trả lời ngắn gọn, thẳng vào trọng tâm, tiết kiệm 50% thời gian suy luận.
- **`qwen3.5-2b-vie`** tiết kiệm tài nguyên nhất (1.5 GB VRAM), nhanh nhất, phù hợp triển khai edge hoặc server cấu hình thấp.

**Tập sạch (bỏ câu hỏi mơ hồ):** 25/100 câu do LLM sinh được gắn nhãn trong `OUTPUT_QUESTIONS/compare100/ambiguous.jsonl` (kèm lý do từng câu), **chỉ dựa trên câu hỏi + đáp án chuẩn, không nhìn kết quả của hệ thống**: 5 câu đáp án tương đối (*"Bốn tháng sau đó"*, *"hôm nay"*), 11 câu thiếu định danh (*"Thượng viện Mỹ thông qua đạo luật gì?"*), 9 câu hỏng (đáp án không trả lời câu hỏi). Chấm lại: `vietnews --questions-dir OUTPUT_QUESTIONS/compare100 report --exclude ambiguous.jsonl` → `compare_report_clean.md`.

| LLM | Contains (100 câu) | Contains (75 câu sạch) | Recall (sạch) | F1 (sạch) |
|---|:---:|:---:|:---:|:---:|
| `vi-qwen2-3b-rag` | 0.68 | **0.77** | **0.89** | 0.17 |
| `qwen2.5-3b` | 0.60 | 0.72 | 0.85 | 0.36 |
| `qwen3.5-2b-vie` | 0.49 | 0.56 | 0.75 | **0.56** |
| `vistral-7b-chat` | 0.42 | 0.47 | 0.61 | 0.19 |
| `vinallama-2.7b-chat` | 0.32 | 0.33 | 0.60 | 0.06 |

Truy xuất trên tập sạch: hit@1 0.87, **ans@1 0.95, ans@3 1.00** (ngữ cảnh luôn chứa đáp án). Thứ hạng các LLM không đổi; phần còn thiếu của điểm contains chủ yếu nằm ở bước sinh câu trả lời, không phải truy xuất.

### 3. Prompt ngắn cho `vi-qwen2-3b-rag`

Mặc định mới `prompt='vi-rag-concise'`: giữ nguyên khuôn prompt AITeamVN, chỉ thêm yêu cầu *"Trả lời bằng MỘT câu ngắn…"*. Cùng ngữ cảnh, chạy lại cả hai prompt trong cùng điều kiện:

| Tập | Prompt | Contains | Recall | F1 | Số từ TB | Giây/câu |
|---|---|:---:|:---:|:---:|:---:|:---:|
| dev (100) | gốc `vi-rag` | 0.69 | 0.81 | 0.162 | 59.6 | — |
| dev (100) | ngắn `vi-rag-concise` | 0.68 | 0.81 | **0.198** | **42.9** | — |
| held-out (110) | gốc `vi-rag` | 0.60 | 0.75 | 0.141 | 66.1 | 13.0 |
| held-out (110) | ngắn `vi-rag-concise` | 0.58 | 0.74 | **0.167** | **53.8** | **10.8** |

Contains không khác biệt có ý nghĩa (McNemar p = 1.0 / 0.75), faithfulness giữ nguyên. Cải thiện có thật nhưng **khiêm tốn**: mô hình đã fine-tune vẫn hay mở đầu "Theo ngữ cảnh…" và thêm "Cụ thể…". Cắt hậu xử lý phần "Cụ thể…" tăng F1 lên 0.23 nhưng làm mất 1–5 câu trả lời đúng nên **không** áp dụng. EM ≈ 0 là do cấu trúc: mô hình trả lời bằng câu hoàn chỉnh còn đáp án chuẩn là cụm từ ngắn.

### 4. Faithfulness & Relevancy (LLM-as-a-judge, 100 câu)

Lệnh: `uv run vietnews --questions-dir OUTPUT_QUESTIONS/compare100 judge --judge vi-qwen2-3b-rag`

Kiểm soát judge (30 câu positive + 30 câu negative):

| Judge | Faithfulness: TB positive ↑ | TB negative ↓ | Độ chính xác | Relevancy: độ chính xác |
|---|:---:|:---:|:---:|:---:|
| `vi-qwen2-3b-rag` | **0.98** | **0.00** | **1.00** | 0.68 |
| `qwen2.5-3b` | 0.63 | 0.00 | 0.93 | 0.58 |

Kết quả:

| Model | Faithfulness (judge vi-qwen2) | Faithfulness (judge qwen2.5) | Relevancy (judge vi-qwen2) |
|---|:---:|:---:|:---:|
| `vi-qwen2-3b-rag` (prompt gốc) | **0.980** | **0.805** | **0.865** |
| `vi-qwen2-3b-rag` (prompt ngắn, mặc định mới) | 0.975 | 0.755 | 0.860 |
| `qwen2.5-3b` | 0.950 | 0.655 | 0.785 |
| `qwen3.5-2b-vie` | 0.945 | 0.630 | 0.685 |

> 📌 Parse lỗi: 0 % ở mọi lần chạy. Hai judge cho **cùng thứ hạng**, kể cả khi `qwen2.5-3b` tự chấm chính nó (thiên vị tự chấm không giải thích được kết quả). Judge `qwen2.5-3b` khắt khe hơn (chỉ cho 0.63 với câu positive) nên giá trị tuyệt đối thấp hơn; dùng `vi-qwen2-3b-rag` làm số chính. **Relevancy có độ tin cậy thấp** (độ chính xác kiểm soát 0.58–0.68), chỉ nên xem như tham khảo.
>
> ⚠ *Kết quả cũ (faithfulness = 0.000 cho mọi model) là lỗi đo, không phải do model*: khuôn mẫu `{"score": 0.0, "reason": "..."}` trong prompt khiến judge 3B chép nguyên điểm 0.0 (57–84 % lý do bắt đầu bằng chuỗi `...`), parse lỗi (8–20 %) bị tính 0 điểm, và rubric phạt câu trả lời *thiếu* thông tin như câu *bịa*. Đã sửa trong `src/text_summarization/evaluation/judge.py`.

---

## 📂 Cấu trúc thư mục & file sinh ra

```
text_summarization/
├── src/text_summarization/
│   ├── __main__.py            # python -m text_summarization
│   ├── cli.py                 # Entrypoint của mọi lệnh vietnews
│   ├── core/                  # Nền tảng dùng chung
│   │   ├── config.py          # Hằng số, đường dẫn, siêu tham số
│   │   ├── data.py            # Tải, làm sạch, đọc/ghi dữ liệu VietNews
│   │   └── text.py            # Chuẩn hoá văn bản, làm sạch input terminal, so khớp đáp án
│   ├── rag/                   # Hệ thống RAG
│   │   ├── indexing.py        # Semantic chunking và luồng dựng index
│   │   ├── ner.py             # Trích xuất thực thể (ELECTRA), gộp bí danh
│   │   ├── store.py           # Kết nối và truy vấn Qdrant (dense / sparse / hybrid RRF)
│   │   ├── sparse.py          # Vector thưa BM25 unigram + bigram âm tiết
│   │   ├── models.py          # Nạp lười embedding, re-ranker, ViT5, LLM
│   │   ├── llms.py            # Đặc tả LLM, prompt, nạp 4-bit, llama-server
│   │   ├── router.py          # Phân loại intent câu nhập
│   │   └── pipeline.py        # Luồng RAG (Retrieve → Rerank → QA / List / Summarize)
│   └── evaluation/            # Đánh giá
│       ├── questions.py       # Sinh câu hỏi, chấm đầu-cuối, chấm truy xuất (hit@k, MRR)
│       ├── compare.py         # So sánh LLM trên cùng ngữ cảnh (contexts → answer → report)
│       ├── metrics.py         # contains, recall, F1, EM, bootstrap, McNemar, Holm
│       └── judge.py           # LLM-as-a-judge: faithfulness / relevancy + câu kiểm soát
├── tests/                     # 69 unit test (mô hình giả, không cần GPU)
├── OUTPUT_LOAD/               # (gitignore) data_article.json
├── OUTPUT_INDEX/              # (gitignore) entities.json, vectors/{chunks.pkl, embedds.npy}, qdrant/
├── OUTPUT_QUESTIONS/          # (gitignore) kết quả đánh giá
│   ├── compare100/            # benchmark chính: 100 câu, câu trả lời 6 LLM, judge_<llm>/report.md
│   ├── heldout100/            # 110 câu held-out (seed 2026) để kiểm chứng
│   └── archive/               # các lần chạy cũ
├── Dockerfile                 # Image pipeline RAG, PyTorch CUDA 12.8, transformers 4.x
├── Dockerfile.llm5            # Image phụ cho LLM cần transformers 5.x
├── docker-compose.yml         # Service rag, qdrant, llm5
├── pyproject.toml             # Cấu hình dự án uv, script vietnews, ruff, pytest
└── requirements.txt           # Dependencies cho Docker build
```

File trong một thư mục câu hỏi (vd. `OUTPUT_QUESTIONS/compare100/`):

| File | Sinh bởi | Nội dung |
|---|---|---|
| `questions.jsonl` | `questions` | `id, guid, title, chunk_idx, question, answer, context` |
| `questions_eval.jsonl` | `evaluate` / `questions --eval` | thêm `pred, retrieved, hit@1, hit@3, contains` |
| `retrieval_eval.jsonl`, `retrieval_report.md` | `retrieval` | hạng bài nguồn từng câu theo từng chế độ / giai đoạn, bảng hit@k + MRR |
| `contexts.jsonl` | `contexts` | câu hỏi, đáp án chuẩn, ngữ cảnh đã ghép, bài nguồn, hit@k |
| `answers_<llm>.jsonl`, `.meta.json` | `answer` | câu trả lời từng câu + thời gian; meta: phần cứng, phiên bản, VRAM |
| `compare_report.md`, `compare_summary.csv`, `compare_review.csv` | `report` | bảng so sánh, kiểm định, bảng soát từng câu |
| `ambiguous.jsonl` | gắn nhãn thủ công | `id` + lý do của các câu cần loại |
| `judge_<llm>/report.md` | `judge` | điểm faithfulness / relevancy, kết quả câu kiểm soát |

---

## 🧪 Kiểm thử

```bash
uv run --group dev pytest           # 69 test, không cần GPU hay tải mô hình
uv run --group dev pytest -k router # chạy một nhóm
```

Pipeline chỉ gọi mô hình qua lớp `Models`, nên test thay bằng mô hình giả (`tests/conftest.py`). Các test bao phủ router, chuẩn hoá văn bản, semantic chunking, NER (gộp bí danh), Qdrant store, sinh và lọc câu hỏi, chỉ số chấm, so sánh LLM và parse kết quả judge.

Lint: cấu hình `ruff` trong `pyproject.toml` (độ dài dòng 120).

---

## 🩺 Xử lý sự cố

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `... chưa có collection 'vietnews'` hoặc `Qdrant (...) và chunks.pkl (...) không khớp` | Index được dựng ở Qdrant khác chỗ đang kết nối. Index dựng bằng Docker → đặt `RAG_QDRANT_URL=http://localhost:6333`. Nếu không → `vietnews index --steps index` để nạp lại Qdrant nhúng |
| `Vector câu hỏi ... chiều nhưng collection ... chiều` | Index dựng bằng mô hình embedding khác. Chạy lại `vietnews index --steps chunk,embed,index` |
| `embedds.npy có ... vector nhưng chunks.pkl có ... chunk` | Chạy lệch bước. Chạy lại `--steps chunk,embed,index` |
| Gõ tiếng Việt bị lỗi, router không nhận "Tóm tắt" / "Liệt kê" | Windows CMD: chạy `chcp 65001` trước. CLI đã ép stdin/stdout UTF-8 cho Git Bash, terminal IDE, pipe |
| `... cần transformers ≥ 5.2` | Mô hình (Qwen3.5) cần môi trường riêng: `docker compose --profile llm5 run --rm llm5 answer --llm qwen3.5-2b-vie` |
| `Không kết nối được llama-server` | Chạy llama-server trước với `--port` khớp `RAG_LLAMA_URL` (xem lệnh trong thông báo lỗi), hoặc dùng LLM khác |
| Hết VRAM (CUDA OOM) | Giữ 4-bit (mặc định), dùng LLM nhỏ hơn (`qwen3.5-2b-vie`, `qwen2.5-3b`), tránh `--no-llm-4bit` trên GPU < 12 GB |
| Windows: `paging file is too small` | Tăng virtual memory của Windows; re-ranker đã nạp fp16 để giảm bộ nhớ |
| LLM trả về rỗng / ký tự rác | `vietnews probe --llm <key>` để xem logits có NaN không và token đầu tiên mô hình sinh ra |
| Lỗi OpenMP trùng (`libiomp5md.dll`) trên Windows | CLI đã đặt sẵn `KMP_DUPLICATE_LIB_OK=TRUE`; nếu import module trực tiếp thì đặt biến này thủ công |
