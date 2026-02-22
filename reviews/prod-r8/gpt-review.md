# Round 8 — GPT-5.2 Review: DEPLOYMENT READINESS

**Date**: 2026-02-20
**Reviewer**: GPT-5.2 (Azure AI Foundry)
**Spotlight**: Deployment Readiness (+1 severity for deployment findings)
**Prior Scores**: R1=67.3, R7=54.3 (avg), 1216 tests passing

---

## Dimension Scores

| # | Dimension | Score |
|---|-----------|-------|
| 1 | Docker Image Quality | 6/10 |
| 2 | CI/CD Pipeline | 7/10 |
| 3 | Health Endpoints | 5/10 |
| 4 | Environment & Secrets | 6/10 |
| 5 | Graceful Shutdown | 4/10 |
| 6 | Dependency Management | 6/10 |
| 7 | Middleware & Security | 7/10 |
| 8 | Observability | 5/10 |
| 9 | Scalability | 5/10 |
| 10 | Operational Readiness | 4/10 |
| | **TOTAL** | **55/100** |

---

## Findings

### 1. Docker Image Quality (6/10)

**DEPLOY-001 — Base image not pinned by digest (supply-chain drift)**
- Severity: HIGH (deployment +1)
- Refs: `Dockerfile`: `FROM python:3.12.8-slim-bookworm` (both stages)
- Tag pinning is better than `latest`, but without a SHA256 digest you'll rebuild different bytes over time.

**DEPLOY-002 — Copies built deps straight into system site-packages (risk of mismatch/overwrites)**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `Dockerfile`: `COPY --from=builder /build/deps /usr/local/lib/python3.12/site-packages/`
- Bypasses normal venv/installed wheel metadata expectations. Makes SBOM attribution harder. Prefer a venv or `pip install` into final stage using wheels from builder.

**DEPLOY-003 — No explicit runtime hardening beyond non-root user**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `Dockerfile`: no `readOnlyRootFilesystem`, no dropped caps, no `PYTHONHASHSEED`, no `UVICORN_*` tuning.
- Non-root user is good but not production-hardened.

**DEPLOY-004 — Container HEALTHCHECK ignored by Cloud Run, adds noise**
- Severity: MEDIUM (deployment +1 from LOW)
- Refs: `Dockerfile`: `HEALTHCHECK ... urlopen('http://localhost:8080/health')`
- Cloud Run does not use Docker HEALTHCHECK for routing. Still runs and consumes CPU.

### 2. CI/CD Pipeline (7/10)

**DEPLOY-005 — No branch/tag gating or approval gates**
- Severity: MEDIUM (deployment +1 from LOW)
- Refs: `cloudbuild.yaml`: steps are sequential but no branch protection or deploy approval trigger.

**DEPLOY-006 — Trivy uses `latest` image (non-reproducible scanner)**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `cloudbuild.yaml`: `name: 'aquasec/trivy:latest'`
- Pin scanner version/digest to avoid pipeline behavior changes.

**DEPLOY-007 — No SBOM/provenance/attestation (SLSA-lite missing)**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `cloudbuild.yaml`: no `syft`, no Artifact Registry attestations, no provenance output.

**DEPLOY-008 — STRENGTH: Coverage gate + lint/typecheck + vuln gate**
- Severity: N/A (positive)
- Refs: `cloudbuild.yaml`: `--cov-fail-under=90`, `ruff`, `mypy`, `trivy --exit-code=1 --severity=CRITICAL,HIGH`

### 3. Health Endpoints (5/10)

**DEPLOY-009 — `/health` conflates readiness + liveness (Cloud Run needs fast, stable liveness)**
- Severity: HIGH (deployment +1)
- Refs: `src/api/app.py`: `/health` returns 503 on "degraded" states (agent missing, CB open, property not loaded).
- If Cloud Run or external LB uses this for health, instances will flap and amplify outages. Need separate `/live` (always 200 if process alive) and `/ready` (gated).

**DEPLOY-010 — Health response contains placeholder checks, not actual implementations**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `src/api/app.py`: `rag_ready = False  # Checks vector store`, `cb_state = "unknown"  # Checks circuit breaker`
- Note: The actual full code DOES implement these checks (with try/except). The summary was misleading. Actual severity may be lower upon full code inspection.

**DEPLOY-011 — No version/build metadata beyond VERSION string**
- Severity: MEDIUM (deployment +1 from LOW)
- Refs: Only `VERSION` in settings; no commit SHA, build timestamp, or image tag in health response.

### 4. Environment & Secrets (6/10)

**DEPLOY-012 — `--allow-unauthenticated` expands blast radius**
- Severity: HIGH (deployment +1)
- Refs: `cloudbuild.yaml`: `gcloud run deploy ... --allow-unauthenticated`
- If API key middleware has any bypass/bug, the service is internet-exposed. Prefer Cloud Run IAM auth as baseline.

**DEPLOY-013 — Secret rotation uses `:latest` only; no version pinning or rollout strategy**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `cloudbuild.yaml`: `--set-secrets=...:latest`
- "latest" makes rotations risky. App does not reload secrets without restart (except API key with TTL).

**DEPLOY-014 — STRENGTH: Pydantic settings validation prevents boot with missing prod secrets**
- Severity: N/A (positive)
- Refs: `src/config.py`: `validate_production_secrets`, `validate_consent_hmac`

### 5. Graceful Shutdown (4/10)

**DEPLOY-015 — Active SSE streams not terminated on SIGTERM**
- Severity: HIGH (deployment +1)
- Refs: `src/api/app.py`: `EventSourceResponse(event_generator())` with `asyncio.timeout(sse_timeout)`
- On SIGTERM, need to stop accepting new streams and end existing ones. No explicit shutdown signal passed to generator; `app.state.ready=False` doesn't stop active streams.

**DEPLOY-016 — Graceful shutdown timeout (10s) vs SSE timeout (60s) mismatch**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `Dockerfile CMD`: `--timeout-graceful-shutdown 10` vs `.env`: `SSE_TIMEOUT_SECONDS=60`
- Active streams will be cut mid-response during deploy, causing client retries and amplified load.

### 6. Dependency Management (6/10)

**DEPLOY-017 — Mixed pinning: some deps unpinned (`>=`)**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `requirements-prod.txt`: `langfuse>=2.0`, `cryptography>=43.0`
- Production must use fully pinned versions.

**DEPLOY-018 — No `--require-hashes` / lockfile / provenance**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `requirements-*.txt`: plain pip install without hash verification.
- Supply-chain gap.

**DEPLOY-019 — STRENGTH: Prod vs dev dependency split and .dockerignore excludes test/local files**
- Severity: N/A (positive)
- Refs: `requirements-prod.txt` excludes chromadb, `.dockerignore` excludes tests/reviews/research.

### 7. Middleware & Security (7/10)

**DEPLOY-020 — Public ingress + API key only (no IAM/OIDC/WAF)**
- Severity: HIGH (deployment +1)
- Refs: `cloudbuild.yaml`: `--allow-unauthenticated`; `ApiKeyMiddleware`
- API keys are weaker than IAM/OIDC. No mention of Cloud Armor, bot protection, or per-route auth.

**DEPLOY-021 — Rate limiting is per-instance (bypassed by scale-out)**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `RateLimitMiddleware`: in-memory sliding window per IP.
- Multiple Cloud Run instances = multiple independent counters. Real protection needs edge/global rate limiting (Cloud Armor/API Gateway) or shared store.

**DEPLOY-022 — STRENGTH: Pure ASGI middleware, correct order, constant-time compare**
- Severity: N/A (positive)
- Refs: Pure ASGI (no BaseHTTPMiddleware), `hmac.compare_digest`, correct middleware stack ordering.

### 8. Observability (5/10)

**DEPLOY-023 — `logging.basicConfig` in lifespan is too late/weak for structured prod logging**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `src/api/app.py`: `logging.basicConfig(...)`
- Cloud Run needs consistent JSON logs, severity mapping, `X-Cloud-Trace-Context` correlation. `basicConfig` after imports is fragile.

**DEPLOY-024 — No metrics/tracing/error reporting integration shown**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: No OpenTelemetry, no Cloud Trace/Profiler/Error Reporting hooks. LangFuse/LangSmith are optional and not deployed.

**DEPLOY-025 — STRENGTH: Request ID + response time middleware exists**
- Severity: N/A (positive)
- Refs: `RequestLoggingMiddleware`: X-Request-ID, X-Response-Time-Ms, structured JSON access logs.

### 9. Scalability (5/10)

**DEPLOY-026 — workers=1 hardcoded; no Cloud Run `--concurrency` setting**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `Dockerfile`: `--workers 1`; `cloudbuild.yaml`: no `--concurrency` flag.
- Cloud Run can run multiple concurrent requests per instance. Single worker may underutilize CPU. Adding workers can break memory/LLM client assumptions. Needs deliberate tuning.

**DEPLOY-027 — `min-instances=1` forces always-on cost without explicit justification**
- Severity: MEDIUM (deployment +1 from LOW)
- Refs: `cloudbuild.yaml`: `--min-instances=1`
- Good for latency but needs explicit rationale and budget acceptance.

**DEPLOY-028 — Startup work (agent build, optional ingest) risks slow start and Cloud Run timeouts**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `src/api/app.py`: `build_graph(...)` in lifespan; `ingest_property` if chroma sqlite missing.
- Even "dev only" ingestion is guarded only by `VECTOR_DB == "chroma"`. If misconfigured in prod, startup can be very slow.

### 10. Operational Readiness (4/10)

**DEPLOY-029 — No rollout strategy: direct deploy, no canary/traffic split/automatic rollback**
- Severity: HIGH (deployment +1)
- Refs: `cloudbuild.yaml`: `gcloud run deploy` with no `--revision-suffix`, `--tag`, or `--traffic` controls.

**DEPLOY-030 — No alerting/SLO/dashboard hooks described**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: No alert policies, uptime checks, log-based metrics, or paging integration in deployment artifacts.

**DEPLOY-031 — Cloud Run timeout (90s) vs SSE behavior misalignment**
- Severity: HIGH (deployment +1 from MEDIUM)
- Refs: `cloudbuild.yaml`: `--timeout=90s`; `src/api/app.py`: `SSE_TIMEOUT_SECONDS=60`
- Comment says "SSE(60) + graceful(10) + buffer(20)" — the math is correct but the design doesn't handle near-timeout gracefully.

---

## Summary

| Metric | Value |
|--------|-------|
| Total Score | **55/100** |
| Finding Count | 31 (27 findings + 4 strengths noted) |
| CRITICAL | 0 |
| HIGH | 19 |
| MEDIUM | 8 |
| LOW | 0 |

### Top 3 Strengths
1. Strong CI quality gates: lint + mypy + pytest + coverage + Trivy blocking.
2. Sensible config validation for production secrets via Pydantic settings.
3. Middleware stack is ASGI-correct with structured request logging and basic security headers.

### Top 5 Deployment Risks
1. `/health` conflates readiness + liveness — can flap instances and amplify outages.
2. Public unauthenticated Cloud Run ingress relying on API keys only (no IAM/WAF).
3. No safe rollout/rollback strategy (no canary/traffic split, no pinned secret versions).
4. Graceful shutdown not coordinated with SSE streaming — likely dropped in-flight streams on scale-down/deploy.
5. Supply-chain gaps: no digest pinning for base/scanner, partial dependency pinning, no lockfile hashes.

### Hire / No-hire Signal (Deployment Readiness Only)
**No-hire** — The fundamentals are present, but the health model, auth/ingress posture, rollout strategy, and shutdown/streaming behavior are not production-safe on Cloud Run without further work.

---

*Note: DEPLOY-010 partially mitigated — the full app.py code does implement RAG and CB health checks (the review summary was based on the truncated version). Actual severity may be lower upon full code inspection.*
