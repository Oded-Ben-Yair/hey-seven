# Deployment Instructions — Hey Seven

## Prerequisites

- Python 3.12+
- Node.js 18+ (for frontend)
- Docker 20.10+
- Google Cloud SDK (`gcloud`)
- Azure CLI (for development Key Vault access)

## Local Development

```bash
# Clone (GitHub — NOT Azure DevOps)
git clone git@github.com:Oded-Ben-Yair/hey-seven.git
cd hey-seven

# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # Includes ChromaDB for local dev
uvicorn src.api.app:app --reload --port 8080

# Frontend (separate terminal)
cd frontend
npm install
npm run dev

# Run tests
pytest tests/ -v    # 3236 tests, 90%+ coverage
```

## Docker Build

```bash
# Build and run
docker compose up -d --build

# Verify
curl http://localhost:8080/health
```

The Dockerfile uses a 2-stage build with:
- SHA-256 digest-pinned base image (prevents tag republishing attacks)
- `--require-hashes` pip install (supply chain hardening)
- Non-root user (`appuser`)
- No curl in production image (Python urllib for health check)
- `requirements-prod.txt` excludes ChromaDB (~200MB, dev-only)

## GCP Deployment (Target Platform)

**Platform**: GCP Cloud Run
**Infrastructure**: Firestore, Vertex AI Vector Search, Redis Memorystore, KMS

### Deploy to Cloud Run

```bash
# Build and push container
gcloud builds submit --tag gcr.io/<project-id>/hey-seven

# Deploy
gcloud run deploy hey-seven \
  --image gcr.io/<project-id>/hey-seven \
  --platform managed \
  --region us-central1 \
  --set-env-vars ENVIRONMENT=production \
  --set-env-vars WEB_CONCURRENCY=4
```

### CI/CD Pipeline

Cloud Build (GCP-native). See `cloudbuild.yaml` if configured.

## Environment Variables

| Variable | Description | Source |
|----------|-------------|--------|
| `ENVIRONMENT` | `development` or `production` | Set per environment |
| `PORT` | Application port | `8080` |
| `WEB_CONCURRENCY` | Uvicorn workers (1 per vCPU) | Default: `1` |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | GCP Console |
| `GEMINI_API_KEY` | Gemini 2.5 Flash/Pro | Azure Key Vault (dev) / GCP Secret Manager (prod) |
| `FIRESTORE_PROJECT` | Firestore project for state | GCP Console |
| `LANGSMITH_API_KEY` | LangSmith observability | Azure Key Vault (dev) |
| `LANGFUSE_SECRET_KEY` | Langfuse tracing | Azure Key Vault (dev) |

**Development secrets**: Azure Key Vault (`kv-seekapa-apps`)
**Production secrets**: GCP Secret Manager + service accounts

## Health Check

```bash
curl http://localhost:8080/health
# Or in production:
curl https://<cloud-run-url>/health
```

## Architecture

12-node LangGraph StateGraph with:
- Router -> Compliance Gate -> Specialist Dispatch -> 6 Specialist Agents
- Validation loop (generate -> validate -> retry max 1 -> fallback)
- Pre-LLM deterministic guardrails (5 layers)
- Circuit breaker for LLM calls
- SSE streaming for real-time responses
- Guest profiling with 10-dimension evaluation (P1-P10)

## Rollback

```bash
# GitHub (NOT Azure DevOps)
git revert <bad-commit>
git push origin main

# Redeploy previous Cloud Run revision
gcloud run services update-traffic hey-seven --to-revisions=<previous-revision>=100
```

## Troubleshooting

- **ChromaDB import error in production**: Production uses Vertex AI Vector Search, not ChromaDB. Verify `requirements-prod.txt` is used (not `requirements.txt`)
- **Firestore connection failing**: Check `GOOGLE_CLOUD_PROJECT` and service account permissions
- **LLM circuit breaker open**: Check Gemini API quotas and rate limits; circuit breaker opens after consecutive failures
- **RAG retrieval poor**: Verify embedding model version matches between ingestion and retrieval (`text-embedding-004`, pinned)
- **Guardrails blocking legitimate queries**: Check pre-LLM deterministic patterns in `src/agent/guardrails.py`
- **Streaming SSE disconnecting**: Verify client-side EventSource reconnection and backend ASGI middleware configuration
