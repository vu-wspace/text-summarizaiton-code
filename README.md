# VietNews: Tóm tắt + NER + Hỏi đáp RAG

Hệ thống hỏi đáp tin tức tiếng Việt trên tập test **VietNews (22.492 bài)** với pipeline RAG hoàn chỉnh: truy xuất → rerank → sinh câu trả lời / tóm tắt / trích xuất thực thể.

---

## ✨ Tính năng chính

| Module | Mô tả |
|--------|-------|
| **Hỏi đáp (QA)** | bge-m3 → FAISS top-20 → bge-reranker-v2-m3 → 3 bài gốc → LLM (mặc định Qwen2.5-3B; đổi bằng `--llm`) |
| **Tóm tắt chủ đề** | Tìm bài liên quan nhất → ViT5-large-summarization |
| **Trích xuất thực thể (NER)** | Tìm bài → Tra thực thể đã NER sẵn bằng ELECTRA (cửa sổ ≤ 400 token, gộp tên) |
| **So sánh LLM** | Chạy 4 LLM trên cùng ngữ cảnh, đánh giá hit@k, contains, recall, F1, EM, từ chối, VRAM, latency |

---

## 📁 Cấu trúc dự án

```
text_summarization/
├── src/text_summarization/
│   ├── config.py      # Đường dẫn, tên mô hình, tham số (top-k, chunk size, ngân sách token...)
│   ├── text.py        # Chuẩn hoá VietNews, làm sạch input terminal, so khớp đáp án khi chấm
│   ├── data.py        # Tải + loại trùng VietNews, đọc/ghi JSON
│   ├── ner.py         # NER cả bài (cửa sổ ≤ 400 token), gộp tên
│   ├── indexing.py    # Chunk theo token bge-m3 → embed → FAISS
│   ├── store.py       # FaissStore: index.faiss + chunks.pkl, tìm theo cosine
│   ├── llms.py        # Danh sách LLM + prompt riêng từng mô hình, nạp/giải phóng VRAM
│   ├── models.py      # bge-m3, re-ranker, ViT5 + LLM đang chọn — nạp lười
│   ├── router.py      # Regex: qa / list / summarize_topic / summarize_text
│   ├── pipeline.py    # Luồng truy vấn: retrieve → re-rank → nhánh trả lời
│   ├── questions.py   # Sinh câu hỏi đánh giá bằng LLM + chấm hit@1, hit@3, contains
│   ├── metrics.py     # Contains, recall, F1, EM, tỉ lệ từ chối
│   ├── compare.py     # So sánh nhiều LLM: contexts → answers → report
│   └── cli.py         # python -m text_summarization <lệnh>
├── tests/             # Test logic bằng mô hình giả (không cần GPU)
├── OUTPUT_LOAD/       # data_article.json, ...
├── OUTPUT_INDEX/      # entities.json, faiss_index/{index.faiss, chunks.pkl, embedds.npy}
├── OUTPUT_QUESTIONS/  # questions.jsonl, questions_eval.jsonl, compare_report.md, compare_summary.csv
├── evaluation_results/ # Kết quả đánh giá LLM RAG (JSON)
├── evaluate_vietnamese_rag.py  # Script đánh giá top 3 Vietnamese LLMs
├── vietnamese_llm_rag_evaluation.md  # Báo cáo chi tiết top 3 LLM
├── Dockerfile, Dockerfile.llm5, docker-compose.yml, pyproject.toml, requirements.txt
└── _old/              # Các script cũ trước khi tổ chức lại
```

---

## 🐳 Chạy bằng Docker (GPU - khuyến nghị)

```bat
chcp 65001
docker compose build
docker compose run --rm rag                                                   :: Chat tương tác
docker compose run --rm rag ask "Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam" --llm-4bit
docker compose run --rm rag questions --n 10 --eval --llm-4bit                 :: Sinh + chấm 10 câu hỏi
docker compose run --rm rag evaluate --rescore                                 :: Chấm lại, không nạp mô hình
docker compose run --rm rag --help
```

> `chcp 65001` cho CMD gửi tiếng Việt dạng UTF-8. Khi gõ lệnh con (`ask`, `questions`…), nhớ thêm `--llm-4bit` để Qwen vừa 8 GB VRAM. Các lệnh dùng LLM nhận thêm `--llm KHOÁ`, xem `llms`.

---

## ⚡ Chạy trực tiếp (uv)

```bat
uv sync
uv run vietnews chat --llm-4bit
uv run vietnews ask "Hồ Xuân Huy lừa bao nhiêu tiền?" --llm-4bit
uv run pytest                      :: Cần: uv sync --group dev
```

---

## 📋 Các lệnh CLI

| Lệnh | Việc | Ghi chú |
|---|---|---|
| `load [--check] [--source DIR]` | Tải VietNews, loại trùng, chuẩn hoá → `OUTPUT_LOAD/` | `--source` là thư mục parquet đã tải sẵn |
| `index [--steps ner,chunk,embed,index] [--header]` | Xây kho → `OUTPUT_INDEX/` | NER và embed nên chạy trên GPU. `--steps index` chỉ dựng lại FAISS từ `embedds.npy` |
| `chat` / `ask "..."` | Hỏi đáp, tóm tắt, liệt kê | `--verbose` in từng bước |
| `questions --n N [--seed S] [--eval]` | Sinh N câu hỏi bằng Qwen (mỗi bài ≤ 1 câu) | Ghi đè `questions.jsonl`; dùng `--name` để lưu tên khác |
| `evaluate [--name F] [--rescore]` | Chấm RAG: hit@1, hit@3, contains | `--rescore` chỉ chấm lại cột `contains` |
| `llms` | Liệt kê LLM và phiên bản transformers cần | |
| `contexts` / `answer --llm K` / `report` | So sánh LLM, xem mục dưới | |
| `compare --llms a,b\|all [--refresh]` | Gói cả 3 bước | |

> Đường dẫn dữ liệu đổi bằng `--load-dir/--index-dir/--questions-dir` (đặt **trước** tên lệnh) hoặc biến môi trường `RAG_LOAD_DIR`, `RAG_INDEX_DIR`, `RAG_QUESTIONS_DIR`.

---

## 🔬 So sánh 4 LLM (cùng ngữ cảnh, cùng 10 câu hỏi)

| Khoá | Repo | Prompt |
|---|---|---|
| `qwen2.5-3b` | Qwen/Qwen2.5-3B-Instruct | Prompt của dự án |
| `vi-qwen2-3b-rag` | AITeamVN/Vi-Qwen2-3B-RAG | Prompt gốc AITeamVN (fine-tuned cho RAG) |
| `qwen3.5-2b-vie` | alphaedge-ai/Qwen3.5-2B-vie-32768 | Prompt của dự án, tắt chế độ suy luận |
| `phogpt-4b-chat` | vinai/PhoGPT-4B-Chat | Khuôn "Dựa vào văn bản sau đây…" (template PhoGPT không có vai system) |

### Yêu cầu phiên bản transformers

| Image (service) | transformers | Dùng cho |
|---|---|---|
| `rag` | 4.56 – 4.x | Qwen2.5, Vi-Qwen2-RAG, ViT5, bge-m3 |
| `llm5` | ≥ 5.2 | Qwen3.5 (kiến trúc `qwen3_5` mới từ 5.2) |
| `phogpt` | 4.44.2 | PhoGPT (code MPT riêng dùng RoPE Llama bị xoá ở bản mới) |

### Quy trình so sánh (3 bước)

```bat
docker compose run --rm rag contexts                                      :: 1. Truy xuất 1 lần -> contexts.jsonl
docker compose run --rm rag answer --llm qwen2.5-3b                       :: 2. Mỗi LLM một lượt, GPU chỉ chứa 1 LLM
docker compose run --rm rag answer --llm vi-qwen2-3b-rag
docker compose --profile llm5 build llm5
docker compose --profile llm5 run --rm llm5 answer --llm qwen3.5-2b-vie
docker compose --profile phogpt build phogpt
docker compose --profile phogpt run --rm phogpt answer --llm phogpt-4b-chat --llm-4bit
docker compose run --rm rag report                                        :: 3. Chấm + gộp báo cáo
```

> Qwen 2–3B chạy bf16 vừa 8 GB VRAM. PhoGPT 3.7B ở bf16 cần ~7.5 GB chỉ cho trọng số, nên trên card 8 GB **bắt buộc** `--llm-4bit`. Muốn so sánh công bằng thì chạy **cả bốn** với `--llm-4bit`. Nếu trộn chế độ, báo cáo sẽ ghi cảnh báo và cột 4-bit. `compare --llms all` gói cả 3 bước trong một lệnh, và tự bỏ qua mô hình không chạy được trong môi trường hiện tại.

### Kết quả đầu ra (`OUTPUT_QUESTIONS/`)

- `compare_report.md`: Bảng tổng hợp + câu trả lời từng LLM cho từng câu, đánh dấu ✅ đúng, ❌ sai, 🚫 từ chối
- `compare_summary.csv`: Các chỉ số dạng bảng
- `compare_review.csv`: Điền cột `dung_that` để chấm tay

| Chỉ số | Ý nghĩa |
|---|---|
| hit@1, hit@3 | Của retriever, chung cho mọi LLM |
| contains | **Chỉ số chính**: đáp án chuẩn nằm trọn trong câu trả lời |
| recall | Tỉ lệ từ của đáp án có trong câu trả lời (điểm từng phần) |
| F1, EM | Kiểu SQuAD; phạt câu trả lời dài dòng |
| từ chối | Tỉ lệ câu LLM nói không biết |
| giây/câu, VRAM | Đo trên máy chạy |

---

## 🇻🇳 Đánh giá Top 3 Vietnamese LLMs cho RAG

Dựa trên HuggingFace downloads, adoption cộng đồng và kiến trúc phù hợp cho RAG:

| Rank | Model | Base | Size | License | Điểm mạnh |
|------|-------|------|------|---------|-----------|
| 1 | **Vistral-7B-Chat** | Mistral-7B | 7B | AFL-3.0 | Hiệu suất chat tiếng Việt tốt nhất, instruction following mạnh |
| 2 | **Vinallama-2.7B-Chat** | Llama-2 | 2.7B | Llama-2 | Nhẹ, inference nhanh, tốt cho edge deployment |
| 3 | **Vietnamese-Llama2-7B-40GB** | Llama-2 | 7B | Custom | Corpus pretraining tiếng Việt lớn nhất (40GB) |

📖 **Chi tiết**: Xem [`vietnamese_llm_rag_evaluation.md`](vietnamese_llm_rag_evaluation.md)

### Embedding & Reranker khuyến nghị cho tiếng Việt

| Model | Dim | Downloads | Best For |
|-------|-----|-----------|----------|
| `bkai-foundation-models/vietnamese-bi-encoder` | 768 | 485K+ | Semantic search chung |
| `AITeamVN/Vietnamese_Embedding` (bge-m3 based) | 1024 | 21K+ | Multilingual, dense retrieval |
| `dangvantuan/vietnamese-document-embedding` | 768 | 70K+ | Document-level retrieval |
| `AITeamVN/Vietnamese_Reranker` | - | 73K+ | Re-ranking (cross-encoder) |

### Pipeline RAG khuyến nghị

```
Query → [Vietnamese Embedding] → Vector Search → [Reranker] → Context → [Vietnamese LLM] → Answer
```

**Combo đề xuất:**
- Embedding: `bkai-foundation-models/vietnamese-bi-encoder` (tỷ lệ downloads/chất lượng tốt nhất)
- Reranker: `AITeamVN/Vietnamese_Reranker` (bge-reranker-v2-m3 fine-tuned)
- Generator: `Viet-Mistral/Vistral-7B-Chat` (hiệu suất chat tốt nhất)

---

## 📊 Kết quả đánh giá thực tế (Vistral-7B-Chat-GGUF)

Chạy 10 câu hỏi tiếng Việt trên máy 8GB VRAM (GGUF 4-bit):

| Metric | Giá trị trung bình |
|--------|-------------------|
| **Faithfulness** | 0.975 / 1.0 |
| **Relevancy** | 0.674 / 1.0 |
| **Fluency** | 0.337 / 1.0 |
| **Latency** | 8.9 giây / câu |
| **Throughput** | 2.5 tokens/giây |

> **Lưu ý**: Fluency thấp do tokenizer tiếng Việt (nhiều từ bị tách thành subword lạ). Câu trả lời vẫn chính xác về mặt thông tin (faithfulness cao).

📁 Kết quả chi tiết: `evaluation_results/Vistral-7B-Chat-GGUF_20261013_221207.json`

---

## 🔧 Các thay đổi quan trọng so với bản cũ

- **Ngữ cảnh rỗng**: Trước đây, nếu bài đầu tiên dài > 3000 token thì Qwen nhận ngữ cảnh rỗng. Giờ bài đó được cắt bớt để vừa ngân sách.
- **Nguồn hiển thị**: Chỉ gồm các bài thực sự nằm trong ngữ cảnh. Trước đây hiện cả bài đã bị cắt khỏi ngữ cảnh.
- **Bài không còn trong kho**: Chunk trỏ tới guid không có trong `data_article.json` được bỏ qua thay vì báo `KeyError`.
- **Chấm `contains`**: Bỏ qua khác biệt trình bày như `10-11` vs `10/11`, hay `( Anh Quốc )` vs `(Anh Quốc)`, nhưng vẫn phân biệt `3,2 tỷ` với `3 tỷ`.
- **Sinh câu hỏi**:
  - Loại câu hỏi mơ hồ, không có tên riêng hay con số.
  - Không dùng `repetition_penalty` lúc sinh, để Qwen chép nguyên văn đáp án.

> Vì các thay đổi trên, kết quả `hit@k` và `contains` **không so sánh trực tiếp được** với các lần chạy trước. Nên chạy lại `questions --eval`.

---

## 📦 Cài đặt môi trường dev

```bash
uv sync --group dev
uv run pytest
```

---

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

## 🤝 Đóng góp

Mọi PR/Issue đều được chào đón! Vui lòng đọc CONTRIBUTING.md trước khi đóng góp.
