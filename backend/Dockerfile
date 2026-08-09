FROM python:3.11-slim

# System dependencies: tesseract-ocr (for OCR fallback), poppler (pdf2image), libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the BGE reranker weights so they're baked into the image.
# This avoids a slow download on first container start.
# CLOUD-DEPENDENCY AUDIT: This single RUN layer downloads model weights from
# HuggingFace Hub at BUILD TIME. After this, no weights are downloaded at
# runtime — inference is fully offline.
RUN python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)"

# Pre-download fastembed sparse model (BM25)
RUN python -c "from fastembed.sparse.bm25 import Bm25; Bm25('Qdrant/bm25')"

# Copy application code
COPY app/ ./app/

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/query_logs /app/data/eval_logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
