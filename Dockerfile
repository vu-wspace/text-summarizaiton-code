# VietNews RAG: tóm tắt + NER + hỏi đáp, chạy trên GPU NVIDIA.
# Mô hình (~15 GB) KHÔNG đóng vào image: nạp từ volume hf_cache (xem docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONPATH=/app/src

WORKDIR /app

# 1) Cài curl/ca-certificates cho tải gói/kết nối mạng ổn định
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2) torch CUDA 12.8 (tương thích RTX 50xx Blackwell và RTX 20/30/40)
RUN pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128

# 3) Cài dependencies từ requirements.txt (ghim numpy tương thích PyTorch và HuggingFace)
COPY requirements.txt .
RUN pip install -r requirements.txt "numpy>=1.26,<2.3"

# 4) Mã nguồn ứng dụng
COPY src ./src

ENTRYPOINT ["python", "-m", "text_summarization"]
CMD ["chat"]
