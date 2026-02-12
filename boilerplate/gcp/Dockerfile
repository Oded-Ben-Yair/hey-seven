# Multi-stage Dockerfile for Casino Host Agent on Cloud Run
# Optimized for fast cold starts and minimal image size.

# ---------------------------------------------------------------------------
# Stage 1: Dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS dependencies

WORKDIR /build

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --target=/build/deps -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Production Image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS production

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy dependencies from build stage
COPY --from=dependencies /build/deps /usr/local/lib/python3.12/site-packages/

# Copy application code
COPY langgraph-agent/ ./langgraph_agent/
COPY rag/ ./rag/
COPY api/ ./api/
COPY requirements.txt .

# Ensure package directories have __init__.py
RUN touch ./langgraph_agent/__init__.py ./rag/__init__.py ./api/__init__.py 2>/dev/null; true

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    APP_VERSION=0.1.0

# Cloud Run expects the container to listen on $PORT
EXPOSE 8080

# Switch to non-root user
USER appuser

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Start the application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
