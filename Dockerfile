# VietNews RAG: tóm tắt + NER + hỏi đáp, chạy trên GPU NVIDIA.
# Mô hình (~15 GB) KHÔNG đóng vào image: tải lần đầu vào volume hf_cache (xem docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONPATH=/app/src

WORKDIR /app

# 1) torch CUDA 12.8 (bắt buộc cho RTX 50xx, chạy được cả RTX 20/30/40). Máy chủ chỉ cần driver NVIDIA.
RUN pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128

# 2) Thư viện còn lại (lớp riêng: sửa code không phải cài lại)
COPY requirements.txt .
RUN pip install -r requirements.txt "numpy>=1.26,<2.3"

# 3) Code
COPY src ./src

ENTRYPOINT ["python", "-m", "text_summarization"]
CMD ["chat", "--llm-4bit"]
