# Component 9: API Layer — Architecture Audit

**Auditor**: auditor-api
**Date**: 2026-03-05
**Scope**: FastAPI backend, ASGI middleware, SSE streaming, PII redaction, error handling

---

## 1. Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/api/app.py` | 938 | Main FastAPI app: lifespan, endpoints, SSE streaming, SIGTERM drain |
| `src/api/middleware.py` | 835 | 6 pure ASGI middleware classes (logging, errors, security, auth, rate limit, body limit) |
| `src/api/models.py` | 163 | Pydantic v2 request/response models + SSE event schemas |
| `src/api/errors.py` | 59 | RFC 7807 error taxonomy (ErrorCode enum, 9 codes) |
| `src/api/pii_redaction.py` | 165 | Regex PII redaction (phone, email, CC, SSN, player card, names). Fails closed. |
| `src/agent/streaming_pii.py` | 134 | StreamingPIIRedactor with lookahead buffering (120-char pattern, 500-char buffer) |

**Total**: 6 files, 2,294 lines

---

## 2. Wiring Verification

All modules are **actively wired** from production entry points:

- `src/api/app.py` — IS the entry point (`create_app()` factory called by uvicorn)
- `src/api/middleware.py` — imported and applied in `app.py` lifespan/middleware stack
- `src/api/models.py` — imported by `app.py` for request/response validation
- `src/api/errors.py` — imported by `app.py` for error responses
- `src/api/pii_redaction.py` — imported by `agent/persona.py`, `agent/graph.py`, `agent/streaming_pii.py`
- `src/agent/streaming_pii.py` — imported by `agent/graph.py` (SSE streaming path)

**Grep proof**:
```
src/api/app.py:from src.api.middleware import ...
src/api/app.py:from src.api.models import ...
src/api/app.py:from src.api.errors import ...
src/agent/persona.py:from src.api.pii_redaction import redact_pii
src/agent/graph.py:from src.api.pii_redaction import redact_pii
src/agent/streaming_pii.py:from src.api.pii_redaction import redact_pii
src/agent/graph.py:from src.agent.streaming_pii import StreamingPIIRedactor
```

**Verdict**: All 6 files are REAL production code, fully wired.

---

## 3. Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `tests/test_api.py` | 56 | All endpoints (/chat, /health, /live, /property, /graph, /feedback, /metrics, /sms/webhook, /cms/webhook), SSE streaming, error handling, CORS |
| `tests/test_middleware.py` | 36 | All 6 middleware classes: logging, error handling, security headers, auth, rate limiting, body limit |
| `tests/test_streaming_pii.py` | 21 | StreamingPIIRedactor: buffering, flush, pattern splitting across chunks, all PII types |
| `tests/test_auth_e2e.py` | 14 | API key auth end-to-end: valid/invalid/missing keys, middleware bypass in dev |

**Total**: 127 tests covering the API layer

Additional coverage from integration tests:
- `tests/test_phase2_integration.py`, `test_phase3_integration.py`, `test_phase4_integration.py` exercise API endpoints indirectly

---

## 4. Live vs Mock Assessment

| Test File | Mock Count | Live LLM Calls | Assessment |
|-----------|-----------|----------------|------------|
| `test_api.py` | 145 | 0 | **Heavily mocked** — LLM, RAG, graph all mocked. Tests validate HTTP wiring, not agent behavior. |
| `test_middleware.py` | 68 | 0 | **Appropriately mocked** — middleware is deterministic logic, mocks isolate ASGI app layer. |
| `test_streaming_pii.py` | 0 | 0 | **No mocks needed** — pure regex/buffer logic, fully deterministic. |
| `test_auth_e2e.py` | 9 | 0 | **Lightly mocked** — mocks LLM only, exercises real middleware stack end-to-end. |

**Summary**: API layer tests are overwhelmingly mocked. This is **structurally appropriate** for HTTP endpoint testing (you don't need live LLM to test that POST /chat returns SSE events). However, it means no test validates the full request->LLM->SSE->client path with a real model. Live behavioral quality is measured by the evaluation framework (outside pytest).

---

## 5. Known Gaps

### GAP-1: In-Memory Rate Limiting (MEDIUM)
`RateLimitMiddleware` uses in-memory sliding window with optional Redis backend. In a multi-instance Cloud Run deployment, in-memory rate limiting is per-instance only. Redis backend exists but requires `REDIS_URL` configuration.

**Impact**: A client could send N * instances requests/minute instead of N.
**Mitigation**: Redis backend is wired and ready. Also, Cloud Armor WAF provides L7 rate limiting as an additional layer.

### GAP-2: Feedback Endpoint Not Forwarded to LangFuse (LOW)
`POST /feedback` stores feedback in `app.state.feedback_store` (in-memory list). It is NOT forwarded to LangFuse or any persistent store. Feedback is lost on container restart.

**Impact**: User feedback for training/improvement is not persisted.
**Mitigation**: Acceptable for MVP. LangFuse integration path is clear.

### GAP-3: Deprecation Annotations Scaffolded (LOW)
`models.py` has SSE event type schemas but no formal API versioning or deprecation headers. The infrastructure supports it but it's not active.

**Impact**: None currently. Will matter when v2 API is introduced.

### GAP-4: No Request Tracing Correlation (LOW)
Requests get a UUID in `RequestLoggingMiddleware` but there's no `X-Request-ID` propagation through to LangFuse traces or agent graph invocations.

**Impact**: Debugging production issues requires correlating logs manually.

---

## 6. Confidence: 88%

**Strengths**:
- Pure ASGI middleware (correct for SSE — BaseHTTPMiddleware would break streaming)
- SIGTERM graceful drain with `_active_streams` tracking (Cloud Run aware)
- RFC 7807 structured errors with 9 error codes
- Security headers (HSTS, X-Frame-Options, X-Content-Type-Options, CSP, Permissions-Policy)
- PII redaction fails closed (safe placeholder, never pass-through)
- Streaming PII redactor with lookahead buffering (handles patterns split across SSE chunks)
- Rate limiter with LRU eviction + background sweep (memory-bounded)
- Cache-Control + ETag on cacheable endpoints
- P50/P95/P99 latency metrics via deque-based sampling
- Production secret validation (hard-fail for empty API_KEY in non-dev)

**Weaknesses**:
- Rate limiting is per-instance without Redis (multi-instance gap)
- Feedback not persisted
- No request-to-trace correlation ID

---

## 7. Verdict: PRODUCTION-READY

The API layer is the most mature component in the system. Pure ASGI middleware, graceful shutdown, fail-closed PII, structured errors, and security headers are all production-grade patterns correctly implemented. The gaps are minor and have clear mitigation paths. The only real production concern is ensuring Redis is configured for rate limiting in multi-instance deployment.
