# Production Review R13 -- Gemini Focus

**Reviewer**: Claude Opus 4.6 (simulating Gemini reviewer perspective)
**Commit**: fda95e3
**Focus Areas**: Architecture coherence, dead code, documentation honesty, design pattern consistency
**Spotlight**: API Design + Testing (+1 severity bump)

---

## Dimension Scores

| # | Dimension | Score | Justification |
|---|-----------|-------|---------------|
| 1 | Graph Architecture | 9/10 | Mature 11-node StateGraph with well-documented dual-layer feature flags, deterministic tie-breaking, and structured dispatch -- no architectural regressions. |
| 2 | RAG Pipeline | 8/10 | Per-item chunking, SHA-256 idempotent IDs, version-stamp purging, and property_id isolation are all solid; retriever cache uses ad-hoc dict-based TTL instead of cachetools.TTLCache (inconsistent with LLM caches). |
| 3 | Data Model | 9/10 | PropertyQAState is clean with proper reducers (_keep_max for responsible_gaming_count), RetrievedChunk TypedDict, and parity assertion at import time -- no drift risk. |
| 4 | **API Design** (SPOTLIGHT) | 7/10 | SSE streaming contract is well-defined with heartbeat and PII redaction; however, the /feedback endpoint lacks rate-limiting, /cms/webhook has no HMAC replay protection, and the heartbeat implementation has a timing gap (see F-001). |
| 5 | **Testing Strategy** (SPOTLIGHT) | 7/10 | E2E tests through real graph are excellent (TestEndToEndGraphIntegration covers 5 guardrail paths + full happy path); however, /feedback and /cms/webhook have zero HTTP-level tests in test_api.py, and the SSE heartbeat is untested (see F-002, F-003). |
| 6 | Docker & DevOps | 8/10 | Not a focus area for this round; lifespan, VECTOR_DB=chroma guard, and production secret validators are solid. |
| 7 | Prompts & Guardrails | 9/10 | 5-layer deterministic guardrails, structured output routing (Literal types), VALIDATION_PROMPT with 6 criteria and examples -- mature and well-documented. |
| 8 | Scalability & Production | 8/10 | LLM semaphore backpressure (20 concurrent), TTL-cached singletons, circuit breaker per-agent, and in-memory rate limiter with documented Cloud Run limitation -- honest about trade-offs. |
| 9 | Trade-off Documentation | 9/10 | Dual-layer feature flag comment block in build_graph() is exemplary; degraded-pass strategy documented inline; every accepted limitation has a rationale. |
| 10 | Domain Intelligence | 9/10 | Responsible gaming escalation counter (_keep_max reducer), BSA/AML deflection, patron privacy, age verification with state-specific law reference -- regulated domain handled correctly. |

**Total: 83/100**

---

## Findings

### F-001: SSE Heartbeat Timing Gap -- Heartbeat Only Fires After an Event is Already Available

**Severity**: HIGH (SPOTLIGHT +1: would be MEDIUM otherwise)
**Location**: `src/api/app.py:174-188`
**Problem**: The heartbeat check (`now - last_event_time >= _HEARTBEAT_INTERVAL`) runs _inside_ the `async for event in chat_stream(...)` loop. This means heartbeats are only sent when a new event is already yielded by the graph. If the LLM takes 45 seconds to generate the first token after `metadata` is sent, no heartbeat fires during that gap because the loop is blocked awaiting the next event from `chat_stream`. The heartbeat only triggers retroactively when the next real event finally arrives.

**Impact**: Browser EventSource clients default to reconnect after ~45-60s of silence. During long LLM generations (router + retrieval + whisper + first specialist token can take 15-30s), the client may time out and auto-reconnect, creating duplicate graph executions. The heartbeat is essentially a no-op for its intended purpose.

**Fix**: Use `asyncio.wait` with a heartbeat timer running concurrently alongside the event generator:

```python
async def event_generator():
    async with asyncio.timeout(sse_timeout):
        event_iter = chat_stream(agent, body.message, body.thread_id, request_id=request_id).__aiter__()
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(event_iter.__anext__(), timeout=_HEARTBEAT_INTERVAL)
                yield event
            except TimeoutError:
                yield {"event": "ping", "data": ""}
            except StopAsyncIteration:
                break
```

---

### F-002: /feedback Endpoint Has No HTTP-Level Integration Test

**Severity**: HIGH (SPOTLIGHT +1: would be MEDIUM otherwise)
**Location**: `tests/test_api.py` (missing)
**Problem**: The `/feedback` endpoint at `src/api/app.py:481-496` handles user feedback with PII redaction on comments. There is exactly one test for `FeedbackRequest` model validation in `test_phase4_integration.py:426` but zero tests in `test_api.py` that exercise the actual HTTP endpoint. No test verifies: (a) successful 200 response, (b) PII redaction of comment in logs, (c) validation rejection (rating out of 1-5 range), (d) missing thread_id rejection.

**Impact**: The endpoint could return 500, log unredacted PII, or accept malformed input with no test coverage catching it. For a production endpoint in a regulated casino environment, PII leaking through feedback comments is a compliance risk.

**Fix**: Add at minimum:
```python
class TestFeedbackEndpoint:
    def test_feedback_returns_200(self):
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/feedback", json={
                "thread_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "rating": 4,
                "comment": "Great help!"
            })
            assert resp.status_code == 200
            assert resp.json()["status"] == "received"

    def test_feedback_invalid_rating_returns_422(self):
        ...

    def test_feedback_invalid_thread_id_returns_422(self):
        ...
```

---

### F-003: /cms/webhook Lacks Timestamp Replay Protection

**Severity**: HIGH (SPOTLIGHT +1: would be MEDIUM otherwise)
**Location**: `src/api/app.py:449-476`, `src/cms/webhook.py`
**Problem**: The CMS webhook accepts `X-Webhook-Timestamp` from the request headers and passes it to `handle_cms_webhook()`, but there is no visible replay protection. An attacker who captures a valid webhook request (with valid HMAC signature) can replay it indefinitely. The timestamp is passed through but never validated against a staleness window (e.g., reject if >5 minutes old).

**Impact**: An attacker could replay a legitimate CMS update webhook to revert content to a previous version, potentially restoring outdated restaurant hours or promotion details. In a casino context, stale promotion information could create legal liability.

**Fix**: Add timestamp staleness validation:
```python
import time
if timestamp:
    try:
        ts = int(timestamp)
        if abs(time.time() - ts) > 300:  # 5 minute window
            return {"status": "rejected", "error": "Timestamp too old"}
    except ValueError:
        return {"status": "rejected", "error": "Invalid timestamp"}
```

---

### F-004: StreamingPIIRedactor Lookahead Uses Original Buffer But Emits from Redacted Buffer

**Severity**: MEDIUM
**Location**: `src/agent/streaming_pii.py:111-118`
**Problem**: In `_scan_and_release(force=False)`, the method redacts the full buffer, emits `redacted[:-_MAX_PATTERN_LEN]` as "safe", but retains `self._buffer[-_MAX_PATTERN_LEN:]` from the ORIGINAL (unredacted) buffer. This is documented as intentional ("not the redacted text, since redaction may change lengths"), but creates a subtle issue: if a PII pattern falls entirely within the safe prefix of the redacted text but the corresponding original chars span the safe/lookahead boundary, the PII will be correctly redacted in the safe emission. However, if the PII pattern falls entirely within the lookahead of the original buffer, it will be redacted on the next `feed()` or `flush()`. The edge case is: what if a PII pattern starts in the safe prefix of the redacted text but the redaction changes the boundary? The safe prefix could be truncated mid-redaction-token (e.g., "[PHO" emitted, "NE]" in next flush).

**Impact**: A redaction placeholder like `[PHONE]` could be split across two SSE chunks: `[PHO` in one token event, `NE]` in the next. This is cosmetically wrong but not a PII leak (the actual phone number is still redacted). Low real-world impact but confusing to the client.

**Fix**: After redaction, find the last complete redaction token boundary in the safe prefix. If the safe text ends mid-token (e.g., `[PHO`), move the incomplete token back to the buffer.

---

### F-005: Retriever Cache Is Not Thread-Safe

**Severity**: MEDIUM
**Location**: `src/rag/pipeline.py:856-934`
**Problem**: The retriever cache uses plain `dict` objects (`_retriever_cache`, `_retriever_cache_time`) without any locking mechanism. In an async context with multiple concurrent requests, two coroutines could simultaneously detect cache expiry, both create new retriever instances, and both write to the dict. While Python's GIL prevents dict corruption, this results in redundant Firestore/ChromaDB client creation and potential resource waste. Compare this to the LLM caches in `nodes.py` which correctly use `asyncio.Lock`.

**Impact**: Under concurrent startup load (multiple /chat requests arriving simultaneously after TTL expiry), multiple ChromaDB/Firestore clients could be created, consuming extra memory and connections. Not a correctness bug but an inconsistency with the established pattern.

**Fix**: Add an `asyncio.Lock` to `_get_retriever_cached()`, consistent with `_get_llm()`:
```python
_retriever_lock = asyncio.Lock()

async def _get_retriever_cached() -> AbstractRetriever:
    async with _retriever_lock:
        # ... existing cache check and creation logic
```
Note: this requires making `_get_retriever_cached` async and updating callers.

---

### F-006: _get_retriever_cached Is Sync But Called from Sync get_retriever

**Severity**: MEDIUM
**Location**: `src/rag/pipeline.py:861, 937-958`
**Problem**: `_get_retriever_cached()` is a synchronous function that imports `time` locally and does synchronous Firestore/ChromaDB client creation. However, it is called from `get_retriever()` which is also synchronous. The tools module (`search_knowledge_base`, `search_hours`) call `get_retriever()` synchronously, and `retrieve_node` wraps these in `asyncio.to_thread()`. This means retriever creation (including Firestore client init, which may make network calls) runs in the thread pool. This is functional but inconsistent: LLM singletons are async with `asyncio.Lock`, but the retriever singleton is sync without any lock.

**Impact**: The inconsistency makes the codebase harder to reason about. If Firestore retriever creation does blocking I/O (which Google Cloud client libraries do), it blocks a thread pool thread but not the event loop -- acceptable but not ideal. The real issue is the missing lock (F-005).

**Fix**: Consider making `get_retriever` async and adding `asyncio.Lock`, or document why the sync path is intentional (thread pool isolation).

---

### F-007: Missing Test for /sms/webhook When SMS_ENABLED=False Returns 404

**Severity**: MEDIUM (SPOTLIGHT +1: would be LOW otherwise)
**Location**: `tests/test_api.py` (missing)
**Problem**: The SMS webhook endpoint has a feature-flag guard (`SMS_ENABLED=False` returns 404), but `test_api.py` has no test for this endpoint at all. The SMS webhook tests exist in `test_cms.py` and `test_deployment.py` but those files test the underlying `handle_inbound_sms` function, not the HTTP endpoint. The 404-on-disabled behavior, the signature verification path, and the keyword-response path have no HTTP-level coverage.

**Impact**: Regression risk for the feature-flag guard. If someone accidentally removes the `SMS_ENABLED` check, no API-level test would catch it.

**Fix**: Add:
```python
class TestSmsWebhookEndpoint:
    def test_sms_disabled_returns_404(self):
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/sms/webhook", json={"data": {}})
            assert resp.status_code == 404
```

---

### F-008: graph.py Imports StreamingPIIRedactor But Module Path Is Under agent/, Not api/

**Severity**: LOW
**Location**: `src/agent/graph.py:32`, `src/agent/streaming_pii.py`
**Problem**: `StreamingPIIRedactor` lives at `src/agent/streaming_pii.py` but it wraps and delegates to `src/api/pii_redaction.redact_pii`. This creates a bidirectional dependency between the `agent` and `api` packages: `graph.py` (agent) imports from `streaming_pii.py` (agent) which imports from `pii_redaction.py` (api), while `graph.py` (agent) also directly imports `contains_pii` and `redact_pii` from `api/pii_redaction.py`. The module placement is surprising -- streaming PII redaction is a data pipeline concern (agent layer), but its implementation details (regex patterns) live in the API layer.

**Impact**: No runtime issue, but the layering is inverted. If the API layer is ever separated into a distinct service, the agent layer has an import dependency on it.

**Fix**: Consider moving the core PII patterns and `redact_pii`/`contains_pii` to a shared module (e.g., `src/data/pii.py` or `src/shared/pii.py`) that both `agent` and `api` import from. This maintains the single source of truth for patterns while fixing the layering inversion.

---

### F-009: Persona Envelope PII Redaction Always Calls redact_pii Even When No PII Detected

**Severity**: LOW
**Location**: `src/agent/persona.py:40-43`
**Problem**: `_validate_output()` calls `redact_pii(response_text)` unconditionally on every response, regardless of whether PII exists. The `redact_pii` function applies 7+ regex patterns and 2 name patterns on every call. For the happy path (no PII in LLM output), this is wasted computation. Compare to the SSE streaming path in `graph.py:660` which uses `contains_pii(content)` as a fast pre-check before calling `redact_pii(content)`.

**Impact**: Marginal latency (microseconds per response). Not a real performance issue given LLM latency dominates, but it is an inconsistency in the PII redaction strategy between persona_envelope (always redact) and SSE replace events (check-then-redact).

**Fix**: Add the same `contains_pii` pre-check used in `graph.py:660`:
```python
def _validate_output(response_text: str) -> str:
    if not contains_pii(response_text):
        return response_text
    redacted = redact_pii(response_text)
    if redacted != response_text:
        logger.warning("Output guardrail: PII detected in LLM response, redacting")
    return redacted
```

---

### F-010: CORSMiddleware Allow-Credentials Not Explicitly Set

**Severity**: LOW
**Location**: `src/api/app.py:118-123`
**Problem**: The CORSMiddleware is configured with specific origins and headers, but `allow_credentials` is not set (defaults to `False`). This means cookies and `Authorization` headers are not sent in cross-origin requests. For the current API-key-based auth (`X-API-Key` header), this is fine. However, the `X-API-Key` header IS listed in `allow_headers`, which means the middleware allows it in CORS preflight -- but if someone later adds cookie-based auth or JWT bearer tokens, they would need to add `allow_credentials=True`. This is not a bug today but a documentation gap.

**Impact**: No current impact. Future risk if auth mechanism changes.

**Fix**: Add a comment documenting the intentional omission:
```python
# allow_credentials=False (default): API uses X-API-Key header auth,
# not cookies. Set to True only if cookie-based auth is added.
```

---

## Summary

### Strengths
- The E2E test suite in `test_api.py::TestEndToEndGraphIntegration` is genuinely impressive: tests exercise the real compiled graph through HTTP, verify graph_node lifecycle events with duration_ms, and assert on SSE event sequences. This is the highest-quality integration test suite I have seen in this codebase across 13 rounds.
- The dual-layer feature flag architecture (build-time topology vs runtime behavior) with a 35-line comment block explaining the rationale is exemplary technical documentation.
- Structured dispatch with `DispatchOutput` Pydantic model + keyword fallback is a sound dual-path architecture. The priority tie-breaking with `_CATEGORY_PRIORITY` is production-grade.
- The `_initial_state` parity check at import time (`_EXPECTED_FIELDS != _INITIAL_FIELDS -> ValueError`) prevents state schema drift between `PropertyQAState` and `_initial_state()`. This is better than `assert` (which vanishes with `-O`).

### Weaknesses
- The API spotlight reveals three endpoints with insufficient test coverage: `/feedback`, `/sms/webhook`, and `/cms/webhook` have no HTTP-level tests in `test_api.py`. The E2E tests for `/chat` are excellent, but the other endpoints are orphaned.
- The SSE heartbeat implementation is structurally broken -- it cannot fire during long inter-event gaps, which defeats its stated purpose of preventing EventSource timeouts.
- The retriever cache pattern deviates from the established `asyncio.Lock` + `TTLCache` pattern used by all three LLM singletons. This inconsistency is a maintenance hazard.

### Score Trajectory
| Model | R11 | R12 | R13 |
|-------|-----|-----|-----|
| Gemini | 86 | 84 | 83 |

Score down 1 point from R12 due to spotlight severity bump on API/testing gaps. The codebase is architecturally mature, but the testing coverage is uneven -- the chat/guardrail paths are thoroughly tested while auxiliary endpoints are not.
