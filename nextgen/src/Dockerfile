# =========================================================
# Dockerfile cho HF Spaces (Docker SDK) — gộp 1 nơi:
#   Stage 1: build React (web/) -> app/static
#   Stage 2: FastAPI (CPU) phục vụ cả API + React build trên cổng 7860
# =========================================================

# ---- Stage 1: build frontend React ----
FROM node:20-alpine AS web
WORKDIR /src/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build          # vite outDir=../app/static -> /src/app/static

# ---- Stage 2: backend Python ----
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces khuyến nghị chạy non-root (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Python deps: CPU torch trước, rồi phần còn lại
COPY requirements-app.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-app.txt

# Mã nguồn + dữ liệu + React build
COPY config/ config/
COPY src/ src/
COPY app/ app/
COPY data/ data/
COPY --from=web /src/app/static app/static

RUN chown -R user:user /app /home/user
USER user

# Bake sẵn model + BM25 index vào image -> cold-start nhanh.
# (Nếu build trên HF quá lâu/timeout, có thể XÓA 2 dòng RUN này; app sẽ tự tải & dựng lúc khởi động.)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('AITeamVN/Vietnamese_Embedding'); CrossEncoder('BAAI/bge-reranker-large')"
RUN python -m src.index_bm25

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
