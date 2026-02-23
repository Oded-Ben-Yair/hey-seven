# Stage 1: Build dependencies
# R37 fix D6-M3: Pin to SHA-256 digest to prevent tag republishing attacks.
# Tag can be republished with different contents; digest is immutable.
# To update: docker pull python:3.12.8-slim-bookworm && docker inspect --format='{{index .RepoDigests 0}}'
FROM python:3.12.8-slim-bookworm@sha256:8ef40398b663cf0a3a4685ad0ffcf924282e4c954283b33b7765eae0856d7e0c AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
# Production build uses requirements-prod.txt which excludes chromadb (~200MB)
# and dev dependencies. For local dev with ChromaDB, use requirements.txt.
#
# R40 fix: --require-hashes for supply chain hardening. Ensures every package
# matches a known SHA-256 hash, preventing dependency confusion and tampering.
# To regenerate hashes after dependency changes:
#   pip-compile --generate-hashes --output-file=requirements-prod.txt requirements-prod.in
# Until hashed requirements are generated, use --no-cache-dir only (TODO: HEYSEVEN-58).
RUN pip install --no-cache-dir --target=/build/deps -r requirements-prod.txt

# Stage 2: Production
FROM python:3.12.8-slim-bookworm@sha256:8ef40398b663cf0a3a4685ad0ffcf924282e4c954283b33b7765eae0856d7e0c

# Install curl for HEALTHCHECK (slim image has neither curl nor wget)
# Security: non-root user
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /build/deps /usr/local/lib/python3.12/site-packages/

# Copy application code (least-frequently-changed first for Docker cache)
COPY data/ ./data/
COPY static/ ./static/
COPY src/ ./src/

# ChromaDB is dev-only (not in requirements-prod.txt). Production uses Vertex AI Vector Search.
# Create data directory owned by appuser for property JSON files.
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

# Environment
# WEB_CONCURRENCY=1 for demo/single-container deployment.
# Production scaling: set WEB_CONCURRENCY=2-4 based on Cloud Run vCPU
# allocation (1 worker per vCPU). Gunicorn with uvicorn workers:
#   CMD gunicorn src.api.app:app -w ${WEB_CONCURRENCY} -k uvicorn.workers.UvicornWorker
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PORT=8080 \
    WEB_CONCURRENCY=1

# Cloud Run startup CPU boost: use --cpu-boost in deploy command
# to get full CPU allocation during container startup (cold start optimization)

EXPOSE 8080

# Switch to non-root user
USER appuser

# Data ingestion happens at STARTUP (FastAPI lifespan), not build time.
# Build-time ingestion would require GOOGLE_API_KEY baked into the image.

# NOTE: Cloud Run ignores Dockerfile HEALTHCHECK — it uses its own HTTP
# startup/liveness probes configured via gcloud deploy flags (see cloudbuild.yaml).
# Kept for local docker-compose / Docker Desktop health monitoring.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Graceful shutdown: 15s allows in-flight SSE streams to complete before
# uvicorn force-terminates connections. Cloud Run sends SIGTERM and allows
# up to --timeout (180s) for graceful shutdown; uvicorn's own timeout is
# the actual bound on connection draining.
CMD ["python", "-m", "uvicorn", "src.api.app:app", \
    "--host", "0.0.0.0", "--port", "8080", "--workers", "1", \
    "--timeout-graceful-shutdown", "15"]
