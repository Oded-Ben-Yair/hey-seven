# Round 8 — Grok Deployment Readiness Review

**Date**: 2026-02-20
**Reviewer**: Grok 4 (hostile mode)
**Spotlight**: Deployment Readiness
**Model**: grok-4

---

## Scoring Summary

| # | Dimension | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Docker Best Practices | 8 | knowledge-base/ not in image for RAG; HEALTHCHECK ignored by Cloud Run |
| 2 | CI/CD Pipeline | 7 | No post-deploy smoke test; no rollback automation; no deploy.sh |
| 3 | Health Checks & Readiness | 6 | No Cloud Run startup/readiness probe config; Dockerfile HEALTHCHECK irrelevant |
| 4 | Secrets Management | 9 | Strong — prod validators, --set-secrets, no .env in image |
| 5 | Cloud Run Configuration | 7 | Missing --cpu flag; missing --concurrency flag |
| 6 | Graceful Shutdown & Cold Start | 8 | Exec-form CMD, --timeout-graceful-shutdown, --cpu-boost |
| 7 | Observability in Production | 7 | langfuse and cryptography unpinned in prod requirements |
| 8 | Security Hardening | 9 | Robust middleware stack; CSP unsafe-inline is documented trade-off |
| 9 | Configuration Management | 8 | get_settings() @lru_cache inconsistent with TTLCache pattern elsewhere |
| 10 | Disaster Recovery & Rollback | 4 | No rollback strategy; no smoke test; no DR plan |

**Total: 73/100**

---

## Findings

### F1: No post-deploy smoke test in CI/CD (HIGH)

**Dimension**: CI/CD Pipeline
**Evidence**: `cloudbuild.yaml` Step 4 deploys but never validates the deployment succeeded functionally.
**Impact**: Broken deployments go undetected until users hit errors. For a casino AI agent, this is unacceptable.
**Fix**: Add Step 5 in `cloudbuild.yaml`:
```yaml
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        sleep 30
        SERVICE_URL=$(gcloud run services describe hey-seven --region=us-central1 --format='value(status.url)')
        STATUS=$(curl -s -o /dev/null -w '%{http_code}' "$SERVICE_URL/health")
        if [ "$STATUS" != "200" ]; then
          echo "SMOKE TEST FAILED: /health returned $STATUS"
          exit 1
        fi
```

### F2: No rollback strategy (CRITICAL)

**Dimension**: Disaster Recovery & Rollback
**Evidence**: No documented or automated rollback mechanism anywhere in `cloudbuild.yaml`, no `deploy.sh` script.
**Impact**: Failed deploy with data corruption or broken agent requires manual intervention with no documented procedure. Casino AI downtime = revenue loss + regulatory risk.
**Fix**:
1. Document rollback in README or ops runbook:
   ```bash
   # List revisions
   gcloud run revisions list --service=hey-seven --region=us-central1
   # Roll back to previous
   gcloud run services update-traffic hey-seven --to-revisions=REVISION_NAME=100 --region=us-central1
   ```
2. Auto-rollback on smoke test failure in cloudbuild.yaml (capture previous revision before deploy, revert if Step 5 fails).

### F3: Cloud Run probes not configured (HIGH)

**Dimension**: Health Checks & Readiness
**Evidence**: `Dockerfile:54-55` has `HEALTHCHECK` which Cloud Run ignores entirely. `cloudbuild.yaml` Step 4 has no `--startup-probe` or `--liveness-probe` flags.
**Impact**: Cloud Run uses default TCP probe on port 8080. During cold start with agent initialization + RAG ingestion, the container may receive traffic before the agent is ready (lifespan hasn't completed). /health returns 503 in this state, but Cloud Run doesn't know to wait.
**Fix**: Add to `cloudbuild.yaml` deploy step:
```
--startup-cpu-boost
--startup-probe-path=/health
--startup-probe-initial-delay=10
--startup-probe-period=5
--startup-probe-failure-threshold=12
--liveness-probe-path=/health
--liveness-probe-period=30
```
Remove or comment out the Dockerfile `HEALTHCHECK` with a note that Cloud Run manages probes.

### F4: Unpinned production dependencies (HIGH)

**Dimension**: Observability in Production
**Evidence**: `requirements-prod.txt:19` has `langfuse>=2.0` and `requirements-prod.txt:29` has `cryptography>=43.0`. Every other dependency is pinned.
**Impact**: A minor Langfuse or cryptography release could break the build or introduce behavioral changes silently. In a casino AI with webhook signature verification (cryptography) and observability (langfuse), this is high-risk.
**Fix**: Pin both to exact versions:
```
langfuse==2.58.0  # or whatever is currently resolved
cryptography==43.0.3
```

### F5: Missing --cpu flag in Cloud Run deploy (MEDIUM)

**Dimension**: Cloud Run Configuration
**Evidence**: `cloudbuild.yaml` Step 4 omits `--cpu` flag. Cloud Run defaults to 1 vCPU.
**Impact**: With `--memory=2Gi` and 1 vCPU, the container is memory-heavy but CPU-starved. LLM API calls are I/O-bound (OK), but RAG ingestion at startup and concurrent SSE streams may benefit from more CPU. The `--cpu-boost` flag only helps during startup.
**Fix**: Add `--cpu=2` to deploy step. Alternatively, add `--cpu-throttling` flag to allow CPU allocation only during request processing (cost optimization).

### F6: Missing --concurrency flag (MEDIUM)

**Dimension**: Cloud Run Configuration
**Evidence**: `cloudbuild.yaml` Step 4 omits `--concurrency`. Cloud Run defaults to 80 concurrent requests per container.
**Impact**: With SSE streaming endpoints that hold connections open for up to 60 seconds, 80 concurrent connections per container could exhaust resources. With 1 uvicorn worker (no parallelism), this is especially risky.
**Fix**: Add `--concurrency=50` (or lower) to deploy step. Tune based on load testing. Consider: each SSE stream holds a connection + LLM API call for ~5-30 seconds. With 1 worker and asyncio, 50 is aggressive but manageable if I/O-bound.

### F7: get_settings() uses @lru_cache, not TTLCache (LOW)

**Dimension**: Configuration Management
**Evidence**: `src/config.py:175` uses `@lru_cache(maxsize=1)` for `get_settings()`, while `src/agent/memory.py:32` uses `TTLCache(maxsize=1, ttl=3600)` for checkpointer cache, and other singletons (LLM, validator) also use TTLCache.
**Impact**: Settings never refresh after first load. If Cloud Run updates env vars (e.g., secret rotation via `--set-secrets=SECRET:latest`), the container must restart to pick up changes. This is inconsistent with the TTLCache pattern used elsewhere for credential rotation.
**Fix**: Either:
1. Accept this as intentional (settings are immutable per container lifecycle — documented trade-off), or
2. Switch to TTLCache for consistency:
```python
from cachetools import TTLCache
_settings_cache = TTLCache(maxsize=1, ttl=3600)
def get_settings():
    if "s" not in _settings_cache:
        _settings_cache["s"] = Settings()
    return _settings_cache["s"]
```

### F8: knowledge-base/ directory not available in container (MEDIUM)

**Dimension**: Docker Best Practices
**Evidence**: `.dockerignore` excludes `knowledge-base/` (line 24). `Dockerfile` does not `COPY knowledge-base/`. However, `src/api/app.py:74-81` only runs ingestion for `VECTOR_DB=chroma` (dev mode), and production uses Firestore/Vertex AI with pre-built indexes.
**Impact**: LOW in practice — production skips local ingestion entirely. But if someone deploys with `VECTOR_DB=chroma` for a staging environment, knowledge-base/ won't be available. The app would start with an empty RAG index.
**Fix**: Document explicitly in Dockerfile comments: "knowledge-base/ excluded intentionally — production uses pre-built Firestore indexes. For staging with ChromaDB, mount knowledge-base/ as a volume or use VECTOR_DB=firestore."

### F9: No deploy.sh script (LOW)

**Dimension**: CI/CD Pipeline
**Evidence**: No `deploy.sh` found in project root. All deploy logic is embedded in `cloudbuild.yaml`.
**Impact**: Local or manual deployments require copy-pasting gcloud commands. Low severity since Cloud Build is the primary deployment mechanism.
**Fix**: Create a `scripts/deploy.sh` wrapping the `gcloud run deploy` command from cloudbuild.yaml for local/manual use.

### F10: Dockerfile HEALTHCHECK redundant (LOW)

**Dimension**: Docker Best Practices
**Evidence**: `Dockerfile:54-55` defines `HEALTHCHECK` but Cloud Run ignores Dockerfile HEALTHCHECK directives entirely, using its own probe system.
**Impact**: No functional impact — it's dead config. Could mislead developers into thinking health checks are active.
**Fix**: Remove or comment with `# NOTE: Cloud Run ignores Dockerfile HEALTHCHECK — configure via --startup-probe and --liveness-probe flags`.

### F11: No disaster recovery plan (MEDIUM)

**Dimension**: Disaster Recovery & Rollback
**Evidence**: No DR documentation. Firestore data (conversation state) has no backup strategy mentioned. Vector store (Vertex AI) index rebuilding not documented.
**Impact**: If Firestore data is corrupted or Vertex AI index is deleted, there's no documented recovery path. For a casino AI handling guest conversations, data loss could mean compliance issues (conversation audit trail).
**Fix**: Document DR plan covering:
1. Firestore backup: Enable automated Firestore exports to GCS (`gcloud firestore export gs://backup-bucket`)
2. Vector index: Document rebuild procedure from knowledge-base/ source data
3. Conversation audit: Firestore point-in-time recovery or scheduled exports

---

## Strengths (What's Done Right)

1. **Multi-stage Docker build** with production-only deps — excludes ChromaDB (~200MB), clean separation
2. **Exec-form CMD** — PID 1 receives SIGTERM directly, proper graceful shutdown
3. **Non-root user** (appuser) — follows container security best practices
4. **Pure ASGI middleware** — no BaseHTTPMiddleware, SSE streaming preserved
5. **Production secret validation** — hard-fail on missing API_KEY, CMS_WEBHOOK_SECRET
6. **Trivy vulnerability scan** in CI with exit-code=1 — blocks deploys with CRITICAL/HIGH CVEs
7. **--min-instances=1** — avoids cold-start latency for first request
8. **Structured JSON logging** — Cloud Logging compatible, includes request_id correlation
9. **Defense-in-depth auth** — Cloud Run public + app-level ApiKey middleware
10. **hmac.compare_digest** for all secret comparisons — timing attack prevention

---

## Score: 73/100

| Finding | Severity | Dimension |
|---------|----------|-----------|
| F1: No post-deploy smoke test | HIGH | CI/CD |
| F2: No rollback strategy | CRITICAL | DR & Rollback |
| F3: Cloud Run probes not configured | HIGH | Health Checks |
| F4: Unpinned prod dependencies | HIGH | Observability |
| F5: Missing --cpu flag | MEDIUM | Cloud Run Config |
| F6: Missing --concurrency flag | MEDIUM | Cloud Run Config |
| F7: get_settings() @lru_cache | LOW | Config Management |
| F8: knowledge-base/ not in container | MEDIUM | Docker |
| F9: No deploy.sh script | LOW | CI/CD |
| F10: Dockerfile HEALTHCHECK redundant | LOW | Docker |
| F11: No disaster recovery plan | MEDIUM | DR & Rollback |

**Critical: 1 | High: 3 | Medium: 4 | Low: 3 | Total: 11 findings**
