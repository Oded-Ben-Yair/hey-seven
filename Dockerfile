# Stage 1: Build dependencies
FROM python:3.12.8-slim-bookworm AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
# Production build uses requirements-prod.txt which excludes chromadb (~200MB)
# and dev dependencies. For local dev with ChromaDB, use requirements.txt.
RUN pip install --no-cache-dir --target=/build/deps -r requirements-prod.txt

# Stage 2: Production
FROM python:3.12.8-slim-bookworm

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /build/deps /usr/local/lib/python3.12/site-packages/

# Copy application code (least-frequently-changed first for Docker cache)
COPY data/ ./data/
COPY static/ ./static/
COPY src/ ./src/

# Create ChromaDB directory owned by appuser BEFORE switching user
RUN mkdir -p /app/data/chroma && chown -R appuser:appuser /app/data

# Environment
# WEB_CONCURRENCY=1 for demo/single-container deployment.
# Production scaling: set WEB_CONCURRENCY=2-4 based on Cloud Run vCPU
# allocation (1 worker per vCPU). Gunicorn with uvicorn workers:
#   CMD gunicorn src.api.app:app -w ${WEB_CONCURRENCY} -k uvicorn.workers.UvicornWorker
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    CHROMA_PERSIST_DIR=/app/data/chroma \
    WEB_CONCURRENCY=1

# Cloud Run startup CPU boost: use --cpu-boost in deploy command
# to get full CPU allocation during container startup (cold start optimization)

EXPOSE 8080

# Switch to non-root user
USER appuser

# Data ingestion happens at STARTUP (FastAPI lifespan), not build time.
# Build-time ingestion would require GOOGLE_API_KEY baked into the image.

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD python -m uvicorn src.api.app:app \
    --host 0.0.0.0 --port 8080 --workers ${WEB_CONCURRENCY} \
    --timeout-graceful-shutdown 10
