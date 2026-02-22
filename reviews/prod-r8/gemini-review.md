# Deployment Readiness Review: Hey Seven (R8)

**Reviewer**: Gemini 3 Pro (Hostile Deployment Auditor)
**Date**: 2026-02-20
**Spotlight**: DEPLOYMENT READINESS (+1 severity per protocol)
**Target**: GCP Cloud Run (Production)
**Prior Scores**: R1=67.3 -> R7=54.3

---

## Executive Summary

**Final Score: 62/100** (+7.7 from R7)

While code coverage is high (1216 tests, 90.41%) and the Docker build pipeline is clean, the architectural integration with Cloud Run has fundamental mismatches. Stateful patterns (in-memory rate limiting, Docker HEALTHCHECK) are deployed to a stateless, ephemeral, horizontally-scaling environment. The application is "code-ready" but not fully "cloud-ready."

Strong areas: Docker security (multi-stage, non-root, exec-form CMD), env var management (SecretStr, production validators), middleware architecture (all pure ASGI, SSE-safe).

Weak areas: Cloud Run health probes not configured, in-memory rate limiting ineffective across instances, unpinned production dependencies.

---

## Detailed Dimension Scoring

### 1. Docker Security (9/10)

Strongest area. Multi-stage build correctly implemented.

**Strengths**:
- Non-root `appuser` with `--no-login` shell
- `apt-get` lists cleaned in same layer
- Exec-form CMD (receives SIGTERM directly at PID 1)
- `PYTHONUNBUFFERED=1` and `PYTHONDONTWRITEBYTECODE=1` set
- Separate `requirements-prod.txt` excludes ChromaDB (~200MB)
- `.dockerignore` comprehensive: excludes .env, tests, reviews, research, chroma data

**Findings**:

- **DEPLOY-R8-001 (MEDIUM)**: `cryptography>=43.0` in requirements compiles native extensions in the builder stage. The `slim-bookworm` final image may lack OpenSSL runtime libraries required by the compiled wheel. If the builder stage compiles against a different OpenSSL version than what's in the runtime stage, `ImportError` at container start. **Mitigation**: The `--target=/build/deps` approach copies pre-compiled wheels, which typically bundle their own `.so` files. But this should be verified with a smoke test of `python -c "from cryptography.hazmat.backends.openssl import backend"` in the final image.

### 2. Cloud Run Config (5/10)

Aggressive but contains configuration mismatches.

**Strengths**:
- `--cpu-boost` for faster cold starts
- `--min-instances=1` keeps one warm instance
- `--max-instances=10` caps cost runaway
- `--set-secrets` uses GCP Secret Manager (not env vars for secrets)
- Timeout arithmetic documented: SSE(60s) + graceful(10s) + buffer(20s) = 90s

**Findings**:

- **DEPLOY-R8-003 (HIGH)**: `--allow-unauthenticated` with only app-level API key auth. Comment in cloudbuild.yaml explains this is defense-in-depth, but if ApiKeyMiddleware has a bug or bypass path (e.g., a new endpoint added without updating `_PROTECTED_PATHS`), Cloud Run offers zero protection. The SMS and CMS webhook endpoints are NOT in `_PROTECTED_PATHS` -- they rely on their own signature verification. An unprotected webhook endpoint with a signature verification bug exposes the attack surface directly. **Recommendation**: Use Cloud Run IAM + API Gateway for external-facing endpoints, or explicitly document every unprotected path and its security mechanism.

- **DEPLOY-R8-004 (HIGH)**: `--timeout=90s` is tight for a GenAI agent. LangGraph graph execution involves: router LLM call + RAG retrieval + whisper planner LLM call + generate LLM call + validate LLM call + persona envelope LLM call = 6 LLM calls minimum. At 5-15s per Gemini call under load, a complex query can legitimately take 60-90s. The SSE timeout is 60s, but Cloud Run infrastructure will 504 at 90s regardless of SSE keepalive. Consider 180-300s Cloud Run timeout.

- **DEPLOY-R8-005 (MEDIUM)**: `--memory=2Gi` with no `--cpu` specified. Cloud Run defaults to 1 vCPU. With LangChain's memory overhead, RAG retrieval context, and concurrent SSE streams, 2GB may be tight. A concurrent burst of 5+ chat requests with large RAG contexts could trigger OOM. Consider `--memory=4Gi --cpu=2` for production load.

### 3. Health Checks (2/10)

Fundamental platform mismatch.

**Strengths**:
- `/health` endpoint is comprehensive: checks agent, property data, RAG, circuit breaker state
- Returns 503 for degraded (correct for load balancer routing)

**Findings**:

- **DEPLOY-R8-006 (CRITICAL)**: **The `HEALTHCHECK` instruction in the Dockerfile is IGNORED by Cloud Run.** Cloud Run does not execute Docker HEALTHCHECK commands. It uses its own HTTP startup/liveness probes configured via the Cloud Run service definition. The Dockerfile HEALTHCHECK gives false confidence -- it works in local Docker but has zero effect in production. **Fix**: Add `--startup-cpu-boost` (already present) and configure startup/liveness probes in cloudbuild.yaml or via `gcloud run services update`:
  ```
  --startup-probe-path=/health
  --startup-probe-period=10
  --startup-probe-failure-threshold=3
  --liveness-probe-path=/health
  --liveness-probe-period=30
  ```

- **DEPLOY-R8-007 (HIGH)**: Without configured liveness probes, Cloud Run falls back to TCP port check. If the application deadlocks (asyncio event loop blocked, circuit breaker stuck open) but keeps the TCP socket open, Cloud Run continues routing traffic to a functionally dead container. The comprehensive `/health` endpoint exists but the platform never calls it.

- **DEPLOY-R8-008 (MEDIUM)**: The `start-period=60s` in the Docker HEALTHCHECK suggests the app needs 60s to initialize (agent build, ChromaDB ingestion, property loading). But in production with `VECTOR_DB=firestore`, startup should be faster. The startup probe timeout should reflect actual production startup time, not dev/ChromaDB startup time.

### 4. Env Var Management (9/10)

Solid implementation.

**Strengths**:
- `SecretStr` for all sensitive values (6 fields)
- `validate_production_secrets` hard-fails on missing API_KEY/CMS_WEBHOOK_SECRET in non-dev
- `validate_consent_hmac` rejects default placeholder
- `normalize_embedding_model` prevents vector space mismatch
- `validate_rag_config` ensures chunk overlap < chunk size
- Cloud Build uses `--set-secrets` (Secret Manager references, not plain env vars)

**Findings**:

- **DEPLOY-R8-009 (LOW)**: `ENVIRONMENT=production` is hardcoded in `--set-env-vars`. For a staging environment, you'd need a separate cloudbuild.yaml or variable substitution. Minor for current scope (single-environment demo).

- **DEPLOY-R8-010 (LOW)**: `GOOGLE_API_KEY` validation only checks non-empty in production via langchain auto-detection. No format validation (e.g., starts with `AI`). A misconfigured secret (wrong secret name mapped) would pass validation but fail at first LLM call.

### 5. Startup/Shutdown Lifecycle (8/10)

**Strengths**:
- Lifespan context manager correctly initializes/cleans up
- `app.state.ready = False` on shutdown prevents new traffic
- ChromaDB ingestion only in dev (VECTOR_DB=chroma guard)
- Agent failure is non-fatal (app starts degraded with agent=None, /chat returns 503)
- `--timeout-graceful-shutdown=10` matches Cloud Run's SIGTERM window

**Findings**:

- **DEPLOY-R8-011 (MEDIUM)**: Cloud Run sends SIGTERM and waits up to the configured `--timeout` (90s) for graceful shutdown, not 10s. The uvicorn `--timeout-graceful-shutdown=10` means uvicorn will forcefully terminate connections after 10s regardless. If an SSE stream is at second 8 of a response, the user loses the last 2s of output. Consider bumping to 15-20s since Cloud Run allows it.

- **DEPLOY-R8-012 (LOW)**: `app.state.agent = None` on shutdown. If a concurrent request is mid-stream using the agent, setting it to None could cause AttributeError. The `ready=False` flag prevents NEW requests but doesn't drain in-flight ones. The graceful shutdown timeout should handle this, but it's a race condition worth documenting.

### 6. Dependency Management (6/10)

**Strengths**:
- Separate `requirements.txt` (dev with ChromaDB) vs `requirements-prod.txt` (no ChromaDB)
- Most packages pinned to exact versions
- Trivy vulnerability scan in CI

**Findings**:

- **DEPLOY-R8-013 (HIGH)**: `langfuse>=2.0` and `cryptography>=43.0` use unbounded version ranges in `requirements-prod.txt`. A breaking change in langfuse 3.x or cryptography 44.x will break production builds unpredictably. **Pin exact versions.** Every other package is correctly pinned -- these two are inconsistent.

- **DEPLOY-R8-014 (MEDIUM)**: `requirements-dev.txt` uses `-r requirements.txt` (which includes ChromaDB) then adds test tools. But `requirements.txt` includes both dev AND prod deps (chromadb + firestore). This means dev installs both vector DB backends. Not harmful but creates a confusing dependency tree.

### 7. CI/CD Completeness (7/10)

**Strengths**:
- 4-gate pipeline: lint+type+test -> build -> scan -> deploy
- `--cov-fail-under=90` coverage gate
- Trivy scan with `--exit-code=1` blocks deployment on CRITICAL/HIGH CVEs
- `--ignore=tests/test_eval.py` excludes eval tests that need API keys
- Artifact Registry (not deprecated gcr.io)

**Findings**:

- **DEPLOY-R8-015 (HIGH)**: No staging environment or canary deployment. Direct push to production. No post-deploy smoke test step. If Trivy passes but the app crashes on startup (e.g., missing Secret Manager secret), the broken revision receives 100% traffic immediately. Add a `--no-traffic` flag and a post-deploy health check step before traffic migration.

- **DEPLOY-R8-016 (MEDIUM)**: `mypy --ignore-missing-imports` suppresses type errors from untyped third-party packages (LangChain, LangGraph). Acceptable but documented trade-off would increase confidence.

- **DEPLOY-R8-017 (LOW)**: Cloud Build step 1 installs all deps fresh every build (no pip cache layer). Build time could be improved with a custom builder image that pre-caches common dependencies.

### 8. Middleware & Security (5/10)

**Strengths**:
- All 6 middleware are pure ASGI (no BaseHTTPMiddleware -- SSE-safe)
- Correct execution order documented (outermost BodyLimit -> innermost RateLimit)
- `hmac.compare_digest` for API key comparison (timing-attack resistant)
- `TRUSTED_PROXIES=None` default prevents XFF spoofing
- Request ID sanitization prevents log injection
- Security headers on error responses (ErrorHandlingMiddleware duplicates them)

**Findings**:

- **DEPLOY-R8-018 (CRITICAL)**: **In-Memory Rate Limiting in Cloud Run.** `RateLimitMiddleware` uses `asyncio.Lock` and `OrderedDict` stored in process memory. Cloud Run scales 1-10 instances. Each instance has its own rate limit state. Effective rate limit = `RATE_LIMIT_CHAT * active_instances` = 20 * 10 = 200 req/min per user. A determined attacker hitting all instances bypasses the rate limit entirely. With `min-instances=1` and `max-instances=10`, this ranges from 20 to 200 depending on scaling. **Fix**: Use Redis (Cloud Memorystore), Cloud Armor rate limiting, or API Gateway rate limiting for distributed enforcement.

- **DEPLOY-R8-019 (MEDIUM)**: `/sms/webhook` and `/cms/webhook` are not in `ApiKeyMiddleware._PROTECTED_PATHS`. They use their own signature verification. But if `TELNYX_PUBLIC_KEY` is empty and `ENVIRONMENT=development`, the SMS webhook has NO authentication. The `validate_production_secrets` validator only checks Telnyx key when `SMS_ENABLED=True`, but the `/sms/webhook` endpoint is always mounted regardless of `SMS_ENABLED`. An attacker could POST to `/sms/webhook` in production even if SMS is disabled.

- **DEPLOY-R8-020 (LOW)**: CSP uses `unsafe-inline` for scripts. Documented trade-off for single-file demo HTML. Acceptable but should be on the production hardening roadmap.

### 9. Observability (8/10)

**Strengths**:
- Structured JSON logging (Cloud Logging compatible with `severity` field)
- X-Request-ID injection and propagation to LangGraph
- X-Response-Time-Ms header for latency tracking
- Circuit breaker state exposed in health endpoint
- LangFuse integration for LLM trace observability
- Access logger separated from app logger (dedicated handler, no propagation)

**Findings**:

- **DEPLOY-R8-021 (MEDIUM)**: `LOG_LEVEL=WARNING` in production (set in cloudbuild.yaml). The `RequestLoggingMiddleware` uses a separate logger (`hey_seven.access`) at INFO level with `propagate=False`, so access logs WILL still emit. But application-level INFO logs (agent initialization, RAG ingestion, etc.) will be suppressed. Consider `LOG_LEVEL=INFO` for production with structured logging -- Cloud Logging can filter by severity.

- **DEPLOY-R8-022 (LOW)**: No distributed tracing (e.g., OpenTelemetry). X-Request-ID provides correlation within a single request but not across services. Acceptable for current single-service architecture.

### 10. Production Hardening (4/10)

**Strengths**:
- Circuit breaker with configurable failure threshold and cooldown
- SSE timeout with heartbeats to prevent client-side disconnects
- `is_disconnected()` check before each SSE yield
- RequestBodyLimitMiddleware with dual-layer enforcement
- CancelledError at INFO level (SSE-aware)
- API key TTL refresh (60s) for rotation without restart

**Findings**:

- **DEPLOY-R8-023 (HIGH)**: No version assertion in the deployment pipeline. `VERSION=1.0.0` is hardcoded in config and .env.example but never bumped or verified post-deploy. After deployment, there's no check that the running container serves the expected version. Stale instances (Cloud Run keeps old revisions) could serve outdated code indefinitely. Add a post-deploy step: `curl /health | jq .version` and assert it matches the expected version.

- **DEPLOY-R8-024 (MEDIUM)**: `RequestBodyLimitMiddleware` streaming enforcement has a race condition. When `exceeded=True` is set during `receive_wrapper`, the next `send_wrapper` call suppresses the response. But if the app has already started sending response headers (response_started=True in a different middleware), the 413 response will be dropped and the client gets a broken response. Edge case but possible under load.

- **DEPLOY-R8-025 (MEDIUM)**: `app.state.property_data` loaded from a JSON file at startup using `open()` with no file locking. If the CMS webhook updates the data file while the app is reading it, partial reads could corrupt the in-memory state. Unlikely in production (Cloud Run filesystem is read-only by default) but the code doesn't enforce this assumption.

---

## Score Summary

| Dimension | Score |
|-----------|-------|
| 1. Docker Security | 9 |
| 2. Cloud Run Config | 5 |
| 3. Health Checks | 2 |
| 4. Env Var Management | 9 |
| 5. Startup/Shutdown Lifecycle | 8 |
| 6. Dependency Management | 6 |
| 7. CI/CD Completeness | 7 |
| 8. Middleware & Security | 5 |
| 9. Observability | 8 |
| 10. Production Hardening | 4 |
| **TOTAL** | **63/100** |

---

## Finding Summary

| ID | Severity | Dimension | Finding |
|----|----------|-----------|---------|
| DEPLOY-R8-001 | MEDIUM | Docker | cryptography OpenSSL runtime linkage risk |
| DEPLOY-R8-003 | HIGH | Cloud Run | --allow-unauthenticated with app-level-only auth |
| DEPLOY-R8-004 | HIGH | Cloud Run | 90s timeout tight for 6-LLM-call agent pipeline |
| DEPLOY-R8-005 | MEDIUM | Cloud Run | 2Gi memory may be insufficient under load |
| DEPLOY-R8-006 | CRITICAL | Health | Docker HEALTHCHECK ignored by Cloud Run |
| DEPLOY-R8-007 | HIGH | Health | No liveness/startup probe configured |
| DEPLOY-R8-008 | MEDIUM | Health | Startup timing should reflect production, not dev |
| DEPLOY-R8-009 | LOW | Env | ENVIRONMENT hardcoded to production |
| DEPLOY-R8-010 | LOW | Env | No GOOGLE_API_KEY format validation |
| DEPLOY-R8-011 | MEDIUM | Lifecycle | Graceful shutdown 10s may truncate SSE streams |
| DEPLOY-R8-012 | LOW | Lifecycle | Agent=None race with in-flight requests |
| DEPLOY-R8-013 | HIGH | Deps | langfuse>=2.0 and cryptography>=43.0 unpinned |
| DEPLOY-R8-014 | MEDIUM | Deps | Confusing dev/prod dependency tree |
| DEPLOY-R8-015 | HIGH | CI/CD | No staging environment or canary deploy |
| DEPLOY-R8-016 | MEDIUM | CI/CD | mypy --ignore-missing-imports reduces confidence |
| DEPLOY-R8-017 | LOW | CI/CD | No pip cache in Cloud Build |
| DEPLOY-R8-018 | CRITICAL | Security | In-memory rate limiting useless across Cloud Run instances |
| DEPLOY-R8-019 | MEDIUM | Security | /sms/webhook always mounted, even when SMS disabled |
| DEPLOY-R8-020 | LOW | Security | CSP unsafe-inline for demo (documented trade-off) |
| DEPLOY-R8-021 | MEDIUM | Observability | LOG_LEVEL=WARNING suppresses app INFO logs in prod |
| DEPLOY-R8-022 | LOW | Observability | No distributed tracing (acceptable for single-service) |
| DEPLOY-R8-023 | HIGH | Hardening | No version assertion post-deploy |
| DEPLOY-R8-024 | MEDIUM | Hardening | RequestBodyLimit race with response_started |
| DEPLOY-R8-025 | MEDIUM | Hardening | Property data file read not atomic |

**Total Findings: 24** (2 CRITICAL, 6 HIGH, 10 MEDIUM, 6 LOW)

---

## Priority Remediation (Must-Fix Before Deploy)

1. **DEPLOY-R8-006**: Remove Dockerfile HEALTHCHECK. Add Cloud Run startup/liveness probes via `gcloud run services update` or in deploy step.
2. **DEPLOY-R8-018**: Replace in-memory rate limiting with Cloud Armor or Redis-backed implementation for Cloud Run horizontal scaling.
3. **DEPLOY-R8-013**: Pin `langfuse` and `cryptography` to exact versions in `requirements-prod.txt`.
4. **DEPLOY-R8-007**: Configure HTTP liveness probe to `/health` in Cloud Run service definition.
5. **DEPLOY-R8-015**: Add `--no-traffic` to deploy step, then a post-deploy health check before `gcloud run services update-traffic`.
6. **DEPLOY-R8-023**: Add version bump + post-deploy version assertion step.

---

## Accepted Trade-offs (Not Penalized)

- `unsafe-inline` CSP for demo HTML (DEPLOY-R8-020 -- documented)
- No distributed tracing for single-service (DEPLOY-R8-022 -- YAGNI)
- ENVIRONMENT hardcoded for single-environment demo (DEPLOY-R8-009)
- mypy ignore-missing-imports for untyped LangChain (DEPLOY-R8-016 -- industry standard)
