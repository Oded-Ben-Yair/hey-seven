# Multi-stage Dockerfile for Casino Host Agent on Cloud Run.
# Optimized for fast cold starts, minimal image size, and security.

# ---------------------------------------------------------------------------
# Stage 1: Build dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for compiling Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --target=/build/deps -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS production

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy pre-built dependencies
COPY --from=builder /build/deps /usr/local/lib/python3.12/site-packages/

# Copy application code using proper package layout (no symlinks).
# The hyphenated directory is copied as the underscored Python package name.
COPY langgraph-agent/ ./langgraph_agent/
COPY rag/ ./rag/
COPY api/ ./api/

# Ensure __init__.py exists in each package
RUN touch ./langgraph_agent/__init__.py ./rag/__init__.py ./api/__init__.py 2>/dev/null; true

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    APP_VERSION=0.1.0 \
    ENVIRONMENT=production

EXPOSE 8080

# Switch to non-root BEFORE running anything
USER appuser

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Start with a single worker (Cloud Run scales horizontally via instances)
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--timeout-keep-alive", "65"]
