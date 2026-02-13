# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --target=/build/deps -r requirements.txt

# Stage 2: Production
FROM python:3.12-slim

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /build/deps /usr/local/lib/python3.12/site-packages/

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY static/ ./static/

# Create ChromaDB directory owned by appuser BEFORE switching user
RUN mkdir -p /app/data/chroma && chown -R appuser:appuser /app/data

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    CHROMA_PERSIST_DIR=/app/data/chroma

EXPOSE 8080

# Switch to non-root user
USER appuser

# Data ingestion happens at STARTUP (FastAPI lifespan), not build time.
# Build-time ingestion would require GOOGLE_API_KEY baked into the image.

CMD ["python", "-m", "uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1", \
     "--timeout-graceful-shutdown", "10"]
