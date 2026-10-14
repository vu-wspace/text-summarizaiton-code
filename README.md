# VietNews: Tóm tắt + NER + Hỏi đáp RAG

Hệ thống hỏi đáp tin tức tiếng Việt trên tập test VietNews (22.492 bài):

- **Hỏi đáp:** bge-m3 → FAISS top-20 → bge-reranker → 3 bài gốc → LLM (mặc định Qwen2.5-3B; đổi bằng `--llm`)
- **Tóm tắt:** tìm bài liên quan nhất → ViT5-large
- **Liệt kê thực thể:** tìm bài → tra thực thể đã NER sẵn bằng ELECTRA

## Cấu trúc

```
text_summarization/
├── src/text_summarization/
│   ├── config.py      # đường dẫn, tên mô hình, tham số (top-k, chunk size, ngân sách token...)
│   ├── text.py        # chuẩn hoá VietNews, làm sạch input terminal, so khớp đáp án khi chấm
│   ├── data.py        # tải + loại trùng VietNews, đọc/ghi JSON
│   ├── ner.py         # NER cả bài (cửa sổ ≤ 400 token), gộp tên
│   ├── indexing.py    # chunk theo token bge-m3 → embed → FAISS
│   ├── store.py       # FaissStore: index.faiss + chunks.pkl, tìm theo cosine
│   ├── llms.py        # danh sách LLM + prompt riêng từng mô hình, nạp/giải phóng VRAM
│   ├── models.py      # bge-m3, re-ranker, ViT5 + LLM đang chọn — nạp lười
│   ├── router.py      # regex: qa / list / summarize_topic / summarize_text
│   ├── pipeline.py    # luồng truy vấn: retrieve → re-rank → nhánh trả lời
│   ├── questions.py   # sinh câu hỏi đánh giá bằng LLM + chấm hit@1, hit@3, contains
│   ├── metrics.py     # contains, recall, F1, EM, tỉ lệ từ chối
│   ├── compare.py     # so sánh nhiều LLM: contexts → answers → report
│   └── cli.py         # python -m text_summarization <lệnh>
├── tests/             # test logic bằng mô hình giả (không cần GPU)
├── OUTPUT_LOAD/       # data_article.json, ...
├── OUTPUT_INDEX/      # entities.json, faiss_index/{index.faiss, chunks.pkl, embedds.npy}
├── OUTPUT_QUESTIONS/  # questions.jsonl, questions_eval.jsonl
├── Dockerfile, Dockerfile.llm5, docker-compose.yml, pyproject.toml, requirements.txt
└── _old/              # các script cũ trước khi tổ chức lại
```

## Chạy bằng Docker (GPU)

```bat
chcp 65001
docker compose build
docker compose run --rm rag                                                   :: chat
docker compose run --rm rag ask "Tóm tắt vụ 2 tàu đối đầu ở Quảng Nam" --llm-4bit
docker compose run --rm rag questions --n 10 --eval --llm-4bit                 :: sinh + chấm
docker compose run --rm rag evaluate --rescore                                 :: chấm lại, không nạp mô hình
docker compose run --rm rag --help
```

`chcp 65001` cho CMD gửi tiếng Việt dạng UTF-8. Khi gõ lệnh con (`ask`, `questions`…), nhớ thêm `--llm-4bit` để Qwen vừa 8 GB VRAM. Các lệnh dùng LLM nhận thêm `--llm KHOÁ`, xem `llms`.

## Chạy trực tiếp (uv)

```bat
uv sync
uv run vietnews chat --llm-4bit
uv run vietnews ask "Hồ Xuân Huy lừa bao nhiêu tiền?" --llm-4bit
uv run pytest                      :: cần: uv sync --group dev
```

## Các lệnh

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

Đường dẫn dữ liệu đổi bằng `--load-dir/--index-dir/--questions-dir` (đặt **trước** tên lệnh) hoặc biến môi trường `RAG_LOAD_DIR`, `RAG_INDEX_DIR`, `RAG_QUESTIONS_DIR`.

## So sánh LLM

Bốn LLM trả lời **cùng 10 câu hỏi** trong `questions.jsonl`, trên **cùng ngữ cảnh**:

| Khoá | Repo | Prompt |
|---|---|---|
| `qwen2.5-3b` | Qwen/Qwen2.5-3B-Instruct | prompt của dự án |
| `vi-qwen2-3b-rag` | AITeamVN/Vi-Qwen2-3B-RAG | prompt gốc AITeamVN (mô hình được fine-tune theo khuôn này) |
| `qwen3.5-2b-vie` | alphaedge-ai/Qwen3.5-2B-vie-32768 | prompt của dự án, tắt chế độ suy luận |
| `phogpt-4b-chat` | vinai/PhoGPT-4B-Chat | khuôn "Dựa vào văn bản sau đây…" trong lượt người dùng (template PhoGPT không có vai system) |

Mỗi mô hình cần một khoảng phiên bản transformers khác nhau, nên quy trình tách 3 bước và bước 2 chạy ở image phù hợp:

| Image (service) | transformers | Dùng cho |
|---|---|---|
| `rag` | 4.56 – 4.x | Qwen2.5, Vi-Qwen2-RAG, cùng ViT5, bge-m3 |
| `llm5` | ≥ 5.2 | Qwen3.5 (kiến trúc `qwen3_5` mới có từ 5.2) |
| `phogpt` | 4.44.2 | PhoGPT (code MPT riêng dùng lớp RoPE Llama đã bị xoá ở bản mới) |


```bat
docker compose run --rm rag contexts                                      :: 1. truy xuất 1 lần -> contexts.jsonl
docker compose run --rm rag answer --llm qwen2.5-3b                       :: 2. mỗi LLM một lượt, GPU chỉ chứa 1 LLM
docker compose run --rm rag answer --llm vi-qwen2-3b-rag
docker compose --profile llm5 build llm5
docker compose --profile llm5 run --rm llm5 answer --llm qwen3.5-2b-vie
docker compose --profile phogpt build phogpt
docker compose --profile phogpt run --rm phogpt answer --llm phogpt-4b-chat --llm-4bit
docker compose run --rm rag report                                        :: 3. chấm + gộp
```

Qwen 2–3B chạy bf16 vừa 8 GB VRAM. PhoGPT 3,7B ở bf16 cần khoảng 7,5 GB chỉ cho trọng số, nên trên card 8 GB phải dùng `--llm-4bit`. Muốn so sánh công bằng thì chạy **cả bốn** với `--llm-4bit`. Nếu trộn chế độ, báo cáo sẽ ghi cảnh báo và cột 4-bit. `compare --llms all` gói cả 3 bước trong một lệnh, và tự bỏ qua mô hình không chạy được trong môi trường hiện tại.

Kết quả nằm trong `OUTPUT_QUESTIONS/`:

- `compare_report.md`: bảng tổng hợp và câu trả lời của từng LLM cho từng câu, đánh dấu ✅ đúng, ❌ sai, 🚫 từ chối.
- `compare_summary.csv`: các chỉ số dạng bảng.
- `compare_review.csv`: điền cột `dung_that` để chấm tay.

| Chỉ số | Ý nghĩa |
|---|---|
| hit@1, hit@3 | Của retriever, chung cho mọi LLM |
| contains | Đáp án chuẩn nằm trọn trong câu trả lời. Đây là chỉ số chính |
| recall | Tỉ lệ từ của đáp án có trong câu trả lời (điểm từng phần) |
| F1, EM | Kiểu SQuAD; phạt câu trả lời dài dòng |
| từ chối | Tỉ lệ câu LLM nói không biết |
| giây/câu, VRAM | Đo trên máy chạy |

## Các thay đổi so với bản cũ

- **Ngữ cảnh rỗng:** trước đây, nếu bài đầu tiên dài hơn 3000 token thì Qwen nhận ngữ cảnh rỗng. Giờ bài đó được cắt bớt để vừa ngân sách.
- **Nguồn hiển thị:** chỉ gồm các bài thực sự nằm trong ngữ cảnh. Trước đây hiện cả bài đã bị cắt khỏi ngữ cảnh.
- **Bài không còn trong kho:** chunk trỏ tới guid không có trong `data_article.json` được bỏ qua thay vì báo `KeyError`.
- **Chấm `contains`:** bỏ qua khác biệt trình bày như `10-11` và `10/11`, hay `( Anh Quốc )` và `(Anh Quốc)`, nhưng vẫn phân biệt `3,2 tỷ` với `3 tỷ`.
- **Sinh câu hỏi:**
  - Loại câu hỏi mơ hồ, không có tên riêng hay con số.
  - Không dùng `repetition_penalty` lúc sinh, để Qwen chép nguyên văn đáp án.

Vì các thay đổi trên, kết quả `hit@k` và `contains` **không so sánh trực tiếp được** với các lần chạy trước. Nên chạy lại `questions --eval`.
