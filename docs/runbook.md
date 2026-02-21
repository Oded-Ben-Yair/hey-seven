# Hey Seven — Operational Runbook

## Service Overview

| Component | URL | Health |
|-----------|-----|--------|
| Hey Seven API | https://hey-seven-XXXXX.run.app | GET /health |
| Cloud Run Console | console.cloud.google.com/run | Dashboard |
| LangFuse Dashboard | cloud.langfuse.com | Traces |

---

## Cloud Run Service Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Port | 8080 | Configured via `ENV PORT=8080` in Dockerfile, passed to uvicorn `--port 8080` |
| Min instances | 1 | `--min-instances=1` in cloudbuild.yaml -- avoids cold-start latency |
| Max instances | 10 | `--max-instances=10` -- caps scaling to prevent DDoS-driven cost runaway |
| Memory | 2Gi | `--memory=2Gi` -- LangGraph + embeddings + Firestore client overhead |
| CPU | 2 | `--cpu=2` -- async SSE streams benefit from concurrent event loop handling |
| Concurrency | 50 | `--concurrency=50` -- SSE streams hold connections; 50 concurrent per instance with asyncio |
| Request timeout | 180s | `--timeout=180s` -- SSE_TIMEOUT(60s) + graceful_shutdown(15s) + LLM pipeline(6 calls * 15s) + buffer |
| CPU boost | Enabled | `--cpu-boost` -- full CPU allocation during container startup for faster cold start |
| Graceful shutdown | 15s | uvicorn `--timeout-graceful-shutdown 15` -- allows in-flight SSE streams to complete before force-termination |
| Workers | 1 | uvicorn `--workers 1` -- single worker for demo; scale via WEB_CONCURRENCY env var for production |
| Python | 3.12.8 | `python:3.12.8-slim-bookworm` (multi-stage build) |
| User | appuser | Non-root user (security hardening) |
| Traffic routing | `--no-traffic` then `--to-latest` | Deploy without routing; smoke test validates before traffic switch |

---

## Probe Configuration

### As Deployed (cloudbuild.yaml Step 6)

```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8080
  periodSeconds: 10
  failureThreshold: 6

livenessProbe:
  httpGet:
    path: /live
    port: 8080
  periodSeconds: 30
```

Cloud Run does not support configuring a separate `readinessProbe` via `gcloud run deploy` flags. The startup probe uses `/health` (which checks agent, RAG, property data, and circuit breaker state). The liveness probe uses `/live` (always returns 200).

### Why /live vs /health?

| Endpoint | Returns | Purpose | When it returns 503 |
|----------|---------|---------|---------------------|
| `/live` | `{"status": "alive"}` (always 200) | Confirms process is alive and event loop is responsive | Never -- always 200 |
| `/health` | `HealthResponse` (200 or 503) | Full readiness check: agent, RAG, property data, CB state | When `ready=False`, `agent=None`, `property_loaded=False`, or `circuit_breaker_state="open"` |

**Critical design decision**: `/live` is used for liveness, not `/health`. When the circuit breaker opens (LLM API outage), `/health` returns 503. If `/health` were the liveness probe, Cloud Run would replace the instance in a loop, amplifying the outage. `/live` avoids this -- the instance stays alive and auto-recovers when the LLM API returns.

### Health Response Schema

```json
{
  "status": "healthy | degraded",
  "version": "1.0.0 | <commit_sha>",
  "agent_ready": true,
  "property_loaded": true,
  "rag_ready": true,
  "observability_enabled": false,
  "circuit_breaker_state": "closed | open | half_open | unknown",
  "environment": "development | production"
}
```

`status` is `"healthy"` when ALL of: `ready=True`, `agent_ready=True`, `property_loaded=True`, AND `circuit_breaker_state != "open"`. Otherwise `"degraded"`.

### Dockerfile HEALTHCHECK (local only)

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```

Cloud Run ignores Dockerfile HEALTHCHECK. This is kept for local `docker-compose` / Docker Desktop health monitoring.

---

## Build Pipeline (cloudbuild.yaml)

| Step | Description | Image |
|------|-------------|-------|
| 1 | Install deps, run `ruff check`, `mypy`, `pytest` (90% coverage gate) | `python:3.12.8-slim-bookworm` |
| 2 | Docker build to Artifact Registry (`us-central1-docker.pkg.dev`) | `gcr.io/cloud-builders/docker` |
| 3 | Trivy vulnerability scan (CRITICAL + HIGH severity, pinned `0.58.2`) | `aquasec/trivy:0.58.2` |
| 4 | Push image to Artifact Registry | `gcr.io/cloud-builders/docker` |
| 5 | Capture current revision for rollback | `cloud-sdk` |
| 6 | Deploy to Cloud Run with `--no-traffic` (canary-safe) | `cloud-sdk` |
| 7 | Smoke test: health check with 3 retries (15s apart), version assertion | `cloud-sdk` |
| 8 | Route 100% traffic to new revision (`--to-latest`) | `cloud-sdk` |

**Automatic rollback**: If the smoke test in Step 7 fails (health endpoint does not return 200 after 3 attempts), the pipeline rolls back to the previous revision captured in Step 5 and exits with failure.

---

## Deployment Playbook

### Standard Deployment (via Cloud Build)

1. Merge to main / push tag: Cloud Build triggers automatically
2. Pipeline runs Steps 1-8 (test, build, scan, deploy, smoke, route)
3. Verify health: `curl https://hey-seven-XXXXX.run.app/health`
4. Verify version: response `version` field must match deployed commit SHA
5. Smoke test: `curl -X POST https://hey-seven-XXXXX.run.app/chat -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{"message":"hello"}'`

### Manual Deployment

```bash
# Build and push
gcloud builds submit --config=cloudbuild.yaml

# Or deploy a specific image directly
gcloud run deploy hey-seven \
  --image=us-central1-docker.pkg.dev/$PROJECT_ID/hey-seven/hey-seven:$COMMIT_SHA \
  --region=us-central1
```

### Secrets Configuration

Secrets are mounted from GCP Secret Manager via `--set-secrets`:

| Secret Name | Env Variable | Purpose |
|-------------|-------------|---------|
| `google-api-key` | `GOOGLE_API_KEY` | Gemini LLM access |
| `hey-seven-api-key` | `API_KEY` | Client API key authentication |
| `cms-webhook-secret` | `CMS_WEBHOOK_SECRET` | CMS webhook HMAC-SHA256 verification |
| `telnyx-public-key` | `TELNYX_PUBLIC_KEY` | SMS webhook signature verification |

### Environment Variables (set via `--set-env-vars`)

| Variable | Production Value | Purpose |
|----------|-----------------|---------|
| `ENVIRONMENT` | `production` | Enables production secret validation |
| `LOG_LEVEL` | `INFO` | Structured JSON logging level |
| `VERSION` | `$COMMIT_SHA` | Deployed version for health endpoint |

---

## Rollback

### Quick Rollback (Traffic Routing)

```bash
# List available revisions
gcloud run revisions list --service=hey-seven --region=us-central1

# Route 100% traffic to a specific previous revision
gcloud run services update-traffic hey-seven \
  --to-revisions=REVISION_NAME=100 \
  --region=us-central1

# Verify rollback
curl https://hey-seven-XXXXX.run.app/health
# Confirm version field shows previous commit SHA
```

### Full Rollback (Redeploy Previous Image)

```bash
gcloud run deploy hey-seven \
  --image=us-central1-docker.pkg.dev/$PROJECT_ID/hey-seven/hey-seven:PREVIOUS_SHA \
  --region=us-central1
```

---

## Incident Response

### LLM API Outage

- **Signal**: Circuit breaker opens -> `/health` returns 503 (`"circuit_breaker_state": "open"`)
- **Impact**: All chat requests return safe fallback messages; agent is functionally degraded but not down
- **Cloud Run behavior**: Liveness probe (`/live`) still returns 200, so instances are NOT replaced. New traffic may not be routed if startup probe uses `/health`.
- **Recovery**: Auto-recovers when LLM API returns. CB transitions to `half_open` after 60s cooldown, allows one probe request. On success, transitions to `closed`.
- **Action**: Monitor CB state via `/health`. Check Gemini API status: https://status.cloud.google.com/. No manual intervention needed unless API key expired.
- **CB parameters**: `CB_FAILURE_THRESHOLD=5` failures in `CB_ROLLING_WINDOW_SECONDS=300s` window triggers open state. `CB_COOLDOWN_SECONDS=60s` before half-open probe.

### RAG Pipeline Failure

- **Signal**: `/health` returns `"rag_ready": false`. Retriever returns empty results, log WARNING.
- **Impact**: Generate node produces response without context (lower quality but not broken)
- **Recovery for ChromaDB (local dev)**: Restart instance -- lifespan triggers auto-ingestion if `VECTOR_DB=chroma` and `chroma.sqlite3` is missing
- **Recovery for Firestore (production)**: Re-ingest data via CMS webhook or manual `ingest_property()`. Production does NOT auto-ingest on startup (prevents race conditions with multiple Cloud Run instances).
- **Action**: Check Firestore/Vertex AI connectivity. Verify embedding model version matches ingestion model (`gemini-embedding-001` pinned in config).

### High Error Rate

- **Signal**: 5xx rate > 1% in Cloud Monitoring
- **Action**: Check structured logs in Cloud Logging: filter `severity>=ERROR`
- **Common causes**:
  - API key rotation missed (Secret Manager version not updated)
  - Dependency version mismatch
  - Agent initialization failure (check startup logs for `"Failed to initialize agent"`)
- **Structured log fields**: `request_id`, `method`, `path`, `status`, `duration_ms` (emitted as JSON by `RequestLoggingMiddleware`)

### Memory Pressure / OOMKilled

- **Signal**: `OOMKilled` in Cloud Run logs
- **Action**: Increase memory limit (currently 2Gi) or reduce concurrency (currently 50)
- **Prevention**:
  - Rate limiter uses `OrderedDict` with LRU eviction capped at `RATE_LIMIT_MAX_CLIENTS=10000`
  - Circuit breaker failure timestamps in `collections.deque` are pruned by rolling window (no maxlen -- bounded by prune frequency)
  - Message history bounded by `MAX_MESSAGE_LIMIT=40` per conversation

### SSE Stream Timeout

- **Signal**: Client receives `{"event": "error", "data": {"error": "Response timed out. Please try again."}}`
- **Cause**: LLM generation took longer than `SSE_TIMEOUT_SECONDS=60s`
- **Action**: Check LLM latency in traces. If systemic, increase `SSE_TIMEOUT_SECONDS` (must be less than Cloud Run `--timeout=180s`)
- **Heartbeat**: SSE sends `event: ping` every 15s during long generations to prevent client-side EventSource timeouts

### Validation Loop Stuck

- **Signal**: High latency, graph reaches recursion limit
- **Diagnosis**: Check LangFuse traces for generate -> validate -> generate loops. `retry_count` in state should max at 1.
- **Remediation**: `GRAPH_RECURSION_LIMIT=10` prevents infinite loops (hard bound). Review validator prompt for overly strict criteria.
- **Degraded-pass behavior**: If validator LLM fails on first attempt, response is served unvalidated (availability over safety -- deterministic guardrails already ran). If validator fails on retry attempt, response is blocked (safety over availability).

### Cold Start Latency > 10s

- **Diagnosis**: Check Cloud Run instance count. `--min-instances=1` should prevent scale-to-zero.
- **Remediation**: `--cpu-boost` is already enabled. If still slow, check container startup logs for ingestion time.
- **Note**: Production (`VECTOR_DB=firestore`) skips RAG ingestion on startup. Only local dev (`VECTOR_DB=chroma`) triggers startup ingestion.

### Rate Limit Exhaustion

- **Signal**: Legitimate users getting 429 responses with `Retry-After: 60` header
- **Diagnosis**: Rate limiting is per-instance, in-memory. Effective limit across N instances = `RATE_LIMIT_CHAT * N`.
- **Rate-limited endpoints**: `/chat` and `/feedback` only. `/health`, `/live`, `/property`, `/graph` are exempt.
- **Remediation**: Increase `RATE_LIMIT_CHAT` env var (default: 20 req/min). Block abusive IPs at Cloud Run ingress (Cloud Armor / IAP).

### PII in Logs

- **Signal**: PII detected in Cloud Logging output
- **Diagnosis**: Search Cloud Logging for phone patterns: `textPayload=~"\+1\d{10}"`
- **Remediation**: Verify `pii_redaction.redact_pii()` is called in feedback endpoint logging. Add missing PII patterns. Purge affected log entries from Cloud Logging.
- **PII redaction behavior**: Fails CLOSED on error -- returns `[PII_REDACTION_ERROR]` safe placeholder, never passes through original text.

---

## Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Retrieval relevance | < 80% | < 70% | Review query patterns, re-chunk data |
| Validation pass rate | < 90% | < 85% | Investigate prompt drift, check LLM changes |
| "I don't know" rate | > 20% | > 30% | Add data to knowledge base |
| Response latency P95 | > 3s | > 5s | Check LLM latency, review prompt length |
| Error rate | > 1% | > 2% | Check Gemini API status page |
| Circuit breaker opens | > 1/hour | > 3/hour | Gemini API degradation -- check status |

---

## Observability

### Structured Logging (Cloud Logging)

`RequestLoggingMiddleware` emits structured JSON access logs compatible with Cloud Logging:

```json
{
  "severity": "INFO",
  "request_id": "a1b2c3d4",
  "method": "POST",
  "path": "/chat",
  "status": 200,
  "duration_ms": 1234.5
}
```

- `X-Request-ID` injected into every response header for correlation
- `X-Response-Time-Ms` header on every response
- `CancelledError` (client disconnect) logged at INFO, not ERROR

### LangFuse

- Enabled when `LANGFUSE_PUBLIC_KEY` is set
- Sampling: 10% in production, 100% in development
- Traces include: thread_id (session), request_id (HTTP correlation), query_type tags
- Callback handler creates trace hierarchies: Trace -> Span (per node) -> Generation (LLM call)

### LangSmith

- Configured via `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` env vars
- Requires app-level sampling code (not just env var config)

---

## Security Architecture

### Middleware Stack (Execution Order)

Starlette executes middleware in reverse add order. Actual execution from outermost to innermost:

1. **RequestBodyLimitMiddleware** (outermost) -- rejects payloads > 64KB before processing
2. **ErrorHandlingMiddleware** -- catches unhandled exceptions, returns structured 500 JSON
3. **RequestLoggingMiddleware** -- injects X-Request-ID, emits structured access logs
4. **SecurityHeadersMiddleware** -- CSP (nonce-based), HSTS, X-Frame-Options, X-Content-Type-Options
5. **ApiKeyMiddleware** -- `hmac.compare_digest()` on X-API-Key for protected endpoints
6. **RateLimitMiddleware** (innermost) -- sliding-window per IP on /chat and /feedback
7. **CORSMiddleware** -- configured per environment via `ALLOWED_ORIGINS`

### API Key Authentication

- Protected endpoints: `/chat`, `/graph`, `/property`, `/feedback`
- Unprotected: `/health`, `/live`, `/sms/webhook`, `/cms/webhook`
- API key refreshed from settings every 60s (supports secret rotation without restart)
- When `API_KEY` is empty (development default), authentication is disabled
- Production validation: `Settings.validate_production_secrets()` hard-fails if `API_KEY` is empty when `ENVIRONMENT != 'development'`

### Webhook Security

- `/sms/webhook`: Telnyx signature verification via `ed25519` headers; returns 404 when `SMS_ENABLED=False`
- `/cms/webhook`: HMAC-SHA256 signature verification via `X-Webhook-Signature` header

### Guardrails (Pre-LLM)

Five deterministic guardrail layers run before any LLM call:
1. Prompt injection detection (regex + optional semantic classifier)
2. Responsible gaming detection
3. Age verification detection
4. BSA/AML detection
5. Patron privacy detection

---

## Environment Variables (Complete Reference)

| Variable | Default | Production | Purpose |
|----------|---------|------------|---------|
| `ENVIRONMENT` | `development` | `production` | Enables production secret validation |
| `LOG_LEVEL` | `INFO` | `INFO` | Logging level |
| `VERSION` | `1.0.0` | `$COMMIT_SHA` | Version reported by /health |
| `GOOGLE_API_KEY` | (empty) | Secret Manager | Gemini LLM access |
| `API_KEY` | (empty) | Secret Manager | Client authentication (required in production) |
| `CMS_WEBHOOK_SECRET` | (empty) | Secret Manager | CMS webhook verification (required in production) |
| `TELNYX_PUBLIC_KEY` | (empty) | Secret Manager | SMS webhook verification |
| `MODEL_NAME` | `gemini-2.5-flash` | `gemini-2.5-flash` | Primary LLM model |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | `gemini-embedding-001` | Embedding model (pinned) |
| `VECTOR_DB` | `chroma` | `firestore` | Vector store backend |
| `CASINO_ID` | `mohegan_sun` | Per-tenant | Multi-tenant casino identifier |
| `PROPERTY_DATA_PATH` | `data/mohegan_sun.json` | Per-tenant | Casino property JSON path |
| `RATE_LIMIT_CHAT` | `20` | `20-30` | Requests per minute per IP |
| `SSE_TIMEOUT_SECONDS` | `60` | `60` | Max SSE stream duration |
| `CB_FAILURE_THRESHOLD` | `5` | `5` | Failures to trip circuit breaker |
| `CB_COOLDOWN_SECONDS` | `60` | `60` | Seconds before half-open probe |
| `CB_ROLLING_WINDOW_SECONDS` | `300` | `300` | Failure counting window |
| `GRAPH_RECURSION_LIMIT` | `10` | `10` | Max graph recursion (validation loop bound) |
| `MAX_MESSAGE_LIMIT` | `40` | `40` | Max messages per conversation |
| `LANGFUSE_PUBLIC_KEY` | (empty) | Set in Secret Manager | LangFuse observability |
| `LANGFUSE_SECRET_KEY` | (empty) | Set in Secret Manager | LangFuse authentication |
| `SMS_ENABLED` | `False` | Per-tenant | Enable SMS channel |

---

## Escalation Matrix

| Severity | Response Time | Who | Channel |
|----------|--------------|-----|---------|
| P0 (service down) | 15 min | On-call engineer | Slack #hey-seven-alerts |
| P1 (degraded) | 1 hour | Engineering team | Slack #hey-seven-ops |
| P2 (non-urgent) | 4 hours | Engineering team | Jira ticket |
| P3 (cosmetic) | Next sprint | Product team | Jira backlog |

---

## Responsible Gaming Escalation

These are handled automatically by deterministic guardrails (NO human judgment needed):

- **Self-exclusion mentions**: Auto-provides DMHAS 1-888-789-7777 (Connecticut)
- **Problem gambling**: Auto-provides 1-800-522-4700 (NCPG)
- **BSA/AML suspicious**: Auto-redirects to appropriate authorities
- **Underage mentions**: Auto-provides age verification info

---

## API Endpoints Reference

| Method | Path | Auth | Rate Limited | Purpose |
|--------|------|------|-------------|---------|
| POST | `/chat` | API Key | Yes (20/min) | SSE streaming chat with agent |
| GET | `/health` | None | No | Readiness check (200 or 503) |
| GET | `/live` | None | No | Liveness check (always 200) |
| GET | `/property` | API Key | No | Property metadata |
| GET | `/graph` | API Key | No | Graph structure for visualization |
| POST | `/feedback` | API Key | Yes (20/min) | User feedback on responses |
| POST | `/sms/webhook` | Telnyx signature | No | Inbound SMS webhook |
| POST | `/cms/webhook` | HMAC signature | No | CMS content update webhook |
| GET | `/*` | None | No | Static files (frontend) |
