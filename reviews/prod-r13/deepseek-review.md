# Production Review Round 13 -- DeepSeek Focus

**Repo**: Oded-Ben-Yair/hey-seven
**Commit**: fda95e3
**Reviewer**: DeepSeek (simulated by Claude Opus 4.6)
**Focus**: Async correctness, state machine integrity, concurrency bugs, algorithmic bounds, API middleware ordering, SSE streaming correctness, test coverage gaps
**Spotlight**: API Design (+1 severity), Testing Strategy (+1 severity)
**Date**: 2026-02-21

---

## Dimension Scores

| # | Dimension | Score | Justification |
|---|-----------|-------|---------------|
| 1 | Graph/Agent Architecture | 9 | Clean 11-node StateGraph with well-documented dispatch, validation loop, and deterministic tie-breaking; the only weakness is the double-LLM-call path (dispatch + specialist generate) sharing a single circuit breaker identity. |
| 2 | RAG Pipeline | 8 | Per-item chunking, RRF reranking, SHA-256 idempotent IDs, and version-stamp purging are all solid; not directly reviewed in this file set but the retrieval integration in nodes.py is clean with timeout guards. |
| 3 | Data Model / State Design | 9 | PropertyQAState TypedDict with `_keep_max` reducer for responsible_gaming_count and parity assertion at import time is exemplary; RetrievedChunk explicit schema prevents drift. |
| 4 | API Design | 8 | (SPOTLIGHT) Pure ASGI middleware, structured error taxonomy, SSE streaming with heartbeat and timeout, proper Retry-After on 503; missing input sanitization on /feedback thread_id and no rate limiting on /sms/webhook. |
| 5 | Testing Strategy | 8 | (SPOTLIGHT) Strong E2E tests through real graph, concurrent rate limit tests, CSP nonce uniqueness tests; but dispatch path (structured LLM -> keyword fallback -> feature flag) has no dedicated E2E test, and streaming PII redactor boundary tests are absent. |
| 6 | Docker & DevOps | 8 | Exec-form CMD documented, Cloud Run probe configuration properly separated (liveness vs readiness), lifespan handles init failure gracefully. |
| 7 | Prompts & Guardrails | 9 | 84 patterns across 4 languages, dual-layer injection detection (regex + semantic), proper ordering (injection before content guardrails, semantic last), fail-closed on classifier error. |
| 8 | Scalability & Production | 8 | LLM semaphore backpressure, TTL-cached singletons, BoundedMemorySaver with LRU eviction, probabilistic sweep in InMemoryBackend; threading.Lock in two Firestore client accessors is a concurrency concern. |
| 9 | Documentation & Code Quality | 9 | Exceptional inline documentation explaining every design decision with origin references; parity assertion prevents state schema drift; all routing rationales documented in docstrings. |
| 10 | Domain Intelligence | 9 | BSA/AML, TCPA, responsible gaming escalation, patron privacy, age verification -- all with multilingual coverage and proper escalation; session-level responsible gaming counter with `_keep_max` reducer. |

**Overall Score: 85/100**

---

## Findings

### F-001: `_dispatch_to_specialist` Does Not Record Circuit Breaker Failures

**Severity**: HIGH
**Location**: `src/agent/graph.py:237-243`
**Problem**: When the structured LLM dispatch call fails (lines 237-243), neither the `ValueError/TypeError` handler nor the broad `Exception` handler calls `cb.record_failure()`. The dispatch function calls `cb.allow_request()` and `cb.record_success()`, but on failure silently falls back to keyword counting without informing the circuit breaker. This means the dispatch LLM can fail indefinitely without ever tripping the breaker.
**Impact**: The circuit breaker for dispatch calls effectively never opens. Under sustained dispatch LLM failures, every request still attempts the dispatch LLM call (wastes latency and API quota) before falling back. The specialist agents' own execute_specialist() has correct CB recording, but the dispatch layer does not.
**Fix**: Add `await cb.record_failure()` in both exception handlers (lines 238-243). The keyword fallback still runs regardless -- recording the failure is purely about tracking LLM health for the circuit breaker.

```python
except (ValueError, TypeError) as exc:
    await cb.record_failure()  # ADD
    logger.warning("Structured dispatch parsing failed: %s", exc)
except Exception:
    await cb.record_failure()  # ADD
    logger.warning("Structured dispatch LLM call failed, falling back to keyword counting", exc_info=True)
```

---

### F-002: `threading.Lock` in Async Application (Firestore Client Accessors)

**Severity**: HIGH
**Location**: `src/casino/config.py:175`, `src/data/guest_profile.py:57`
**Problem**: Both `_get_firestore_client()` accessors use `threading.Lock` instead of `asyncio.Lock`. When a Firestore client instantiation takes significant time (cold-start SSL handshake, network latency), `threading.Lock` blocks the entire event loop for ALL concurrent coroutines. This is the exact pattern documented in `nodes.py:94` as the reason those caches were migrated to `asyncio.Lock`.
**Impact**: Under concurrent cold-start requests, a slow Firestore client creation blocks the event loop thread, causing request timeouts and unresponsive SSE streams. The codebase already fixed this pattern in `nodes.py` -- the fix was not propagated to these two files.
**Fix**: Convert both to `asyncio.Lock` with `async with` and make the accessor functions async. If these are called from sync contexts, wrap in `asyncio.to_thread()` or redesign the call site.

---

### F-003: Streaming PII Redactor Double-Redaction on Lookahead Re-scan

**Severity**: MEDIUM
**Location**: `src/agent/streaming_pii.py:111-118`
**Problem**: When `_scan_and_release(force=False)` runs, it applies `redact_pii()` to the entire buffer, emits `redacted[:-_MAX_PATTERN_LEN]` as safe text, but retains the ORIGINAL (un-redacted) buffer tail as lookahead: `self._buffer = self._buffer[-_MAX_PATTERN_LEN:]`. On the next `feed()` or `flush()`, this original tail is re-scanned and re-redacted. If the tail contained a partial PII pattern that was NOT redacted in the first pass (because the full pattern wasn't complete yet), this is correct. But if the tail contained a complete PII pattern that WAS already redacted in `safe`, the original tail now gets redacted again -- producing the correct output. However, there is an edge case: if a phone number like `555-123-4567` spans the safe/lookahead boundary (e.g., `555-123-` in safe, `4567` in lookahead), the safe prefix `555-123-` is emitted without redaction because the full pattern doesn't match in the safe prefix alone, and the full 10 chars in the lookahead on re-scan is only `4567` plus whatever comes next, which also won't match the full pattern. The PII leaks in two fragments.
**Impact**: Phone numbers or card numbers that span the safe/lookahead split point can leak un-redacted in two fragments across consecutive SSE token events. This is partially mitigated by `_MAX_PATTERN_LEN = 40` (larger than any single PII pattern), but the redaction is applied to `redacted` (which has different length than original after substitutions), so the `[:-_MAX_PATTERN_LEN]` split on `redacted` and `[-_MAX_PATTERN_LEN:]` split on `self._buffer` can be misaligned.
**Fix**: Apply the lookahead retention to the redacted text instead of the original buffer, or track redaction offsets to ensure the split point is consistent. Alternatively, always retain `_MAX_PATTERN_LEN` of the REDACTED text as lookahead (not original), so the boundary is consistent:

```python
# In _scan_and_release(force=False):
redacted = redact_pii(self._buffer)
safe = redacted[:-_MAX_PATTERN_LEN]
self._buffer = redacted[-_MAX_PATTERN_LEN:]  # Retain redacted lookahead
```

---

### F-004: `/feedback` Endpoint Missing `thread_id` UUID Validation

**Severity**: MEDIUM (SPOTLIGHT +1 = MEDIUM)
**Location**: `src/api/app.py:482-496`, `src/api/models.py:94-110`
**Problem**: The `FeedbackRequest` model has a `validate_feedback_thread_id` validator that checks UUID format. However, the `/feedback` endpoint is NOT in the `ApiKeyMiddleware._PROTECTED_PATHS` set (which is `{"/chat", "/graph", "/property", "/feedback"}`). Wait -- upon re-checking, `/feedback` IS in the protected paths. Confirmed.

**Revised finding**: The `/feedback` endpoint logs `body.comment` with PII redaction but does not actually persist the feedback anywhere (no database write, no LangFuse score submission). The endpoint accepts feedback and returns `{"status": "received"}` but the data is silently discarded. The docstring says "In production, feedback is forwarded to LangFuse as a score" but no such forwarding exists.
**Impact**: Customer feedback is logged then lost. No mechanism to act on negative feedback or track satisfaction trends. The endpoint creates false confidence that feedback is being collected.
**Fix**: Implement LangFuse score forwarding as documented, or update the docstring to accurately reflect current behavior (log-only, no persistence). Add a TODO with ticket ID per the placeholder response tracking rule.

---

### F-005: `InMemoryBackend` Uses Non-Atomic Tuple Read/Write Without Lock

**Severity**: MEDIUM
**Location**: `src/state_backend.py:90-96`
**Problem**: `InMemoryBackend.increment()` reads `self._store.get(key)` then writes `self._store[key] = (current + 1, expiry)` without any lock. Under concurrent async access (which is the expected usage for rate limiting), two coroutines can read the same count, both increment by 1, and the final value is `count + 1` instead of `count + 2` (lost update). The `RateLimitMiddleware` uses its own `asyncio.Lock`, so the InMemoryBackend's lack of locking is currently masked by the middleware lock. But if any other code path (e.g., future SMS rate limiting, CMS webhook dedup) uses `InMemoryBackend.increment()` without an external lock, the lost update bug will manifest.
**Impact**: Currently masked by middleware lock. If InMemoryBackend is used directly (as the StateBackend abstraction suggests it could be), concurrent increments produce incorrect counts. The RedisBackend uses atomic `INCR` so this inconsistency means the backends are not behaviorally equivalent.
**Fix**: Add an `asyncio.Lock` to `InMemoryBackend` and protect all read-modify-write operations, or document that callers must provide their own synchronization. Better: make the interface async (`async def increment`) and add internal locking.

---

### F-006: No Test Coverage for `_dispatch_to_specialist` Structured LLM Path

**Severity**: MEDIUM (SPOTLIGHT +1 = MEDIUM)
**Location**: `src/agent/graph.py:184-267`, missing in `tests/`
**Problem**: The `_dispatch_to_specialist` function has three code paths: (1) structured LLM dispatch, (2) keyword fallback, (3) feature flag override. The keyword fallback has unit tests in `test_graph_v2.py`, and the E2E happy path test exercises the full pipeline, but there is no dedicated test for the structured LLM dispatch path in isolation -- specifically testing that `DispatchOutput` structured output is correctly parsed, that an invalid specialist name falls through, or that dispatch failures correctly fall back to keyword counting.
**Impact**: The most complex dispatch path (structured LLM output parsing -> specialist validation -> feature flag check) has no targeted test. Regression risk is high because changes to `DispatchOutput` fields or the specialist validation set could silently break dispatch without any test failure.
**Fix**: Add tests in `test_graph_v2.py` covering: (1) successful structured dispatch with mocked LLM returning valid `DispatchOutput`, (2) structured dispatch with invalid specialist name falling back to keyword, (3) LLM failure falling back to keyword, (4) feature flag disabling specialist routing.

---

### F-007: `RateLimitMiddleware._request_counter` Initialized via `getattr` Fallback

**Severity**: LOW
**Location**: `src/api/middleware.py:369`
**Problem**: `self._request_counter = getattr(self, "_request_counter", 0) + 1` uses `getattr` with a default instead of initializing the attribute in `__init__`. This is a Python anti-pattern that hides the attribute from static analysis tools (mypy, pylint) and IDE autocompletion. The attribute is only defined on first access, not in the class constructor.
**Impact**: No runtime bug (the `getattr` fallback works correctly), but violates explicit attribute initialization conventions. Tools cannot verify the attribute exists, and a typo in `_request_counter` would silently create a new attribute instead of raising `AttributeError`.
**Fix**: Initialize `self._request_counter: int = 0` in `RateLimitMiddleware.__init__()` and remove the `getattr` fallback.

---

### F-008: `BoundedMemorySaver` Does Not Implement `get_tuple`/`aget_tuple` Protocol Correctly

**Severity**: LOW
**Location**: `src/agent/memory.py:108-114`
**Problem**: `BoundedMemorySaver` delegates `get_tuple` and `aget_tuple` to the inner `MemorySaver`, but it does NOT inherit from `BaseCheckpointSaver` or any formal protocol. If LangGraph's compiled graph calls a method not delegated (e.g., a new method added in a future LangGraph version), the `BoundedMemorySaver` will raise `AttributeError` at runtime with no clear error message. The class relies on duck typing but has no `__getattr__` fallback.
**Impact**: LangGraph version upgrades that add new checkpointer methods will silently break the wrapper. Since LangGraph is pinned (`0.2.60`), this is not an immediate risk, but version bumps require auditing the wrapper.
**Fix**: Add `def __getattr__(self, name): return getattr(self._inner, name)` as a fallback, or inherit from `BaseCheckpointSaver` to get compile-time protocol verification.

---

### F-009: `/sms/webhook` Not Rate Limited

**Severity**: MEDIUM (SPOTLIGHT +1 = MEDIUM)
**Location**: `src/api/middleware.py:409`, `src/api/app.py:374-444`
**Problem**: `RateLimitMiddleware` only rate-limits `/chat` and `/feedback`. The `/sms/webhook` endpoint processes inbound SMS and can trigger agent execution (Phase 2.4 routing comment at line 434). Without rate limiting, an attacker who discovers the webhook URL can flood it with requests, consuming LLM API quota and causing circuit breaker trips. Even with signature verification, Telnyx webhook replay attacks (replaying valid signed payloads within the timestamp window) could bypass signature checks.
**Impact**: SMS webhook DoS can exhaust LLM quota, trigger circuit breaker (degrading web chat), and generate excessive log volume. The `/cms/webhook` has the same issue but is lower risk (requires HMAC secret knowledge).
**Fix**: Add `/sms/webhook` to the rate-limited paths in `RateLimitMiddleware`, or add a separate rate limit for webhook endpoints with a higher threshold (webhooks are bursty but legitimate traffic is bounded by Telnyx's own rate limits).

---

### F-010: `_build_greeting_categories` Reads File Synchronously in Async Node

**Severity**: LOW
**Location**: `src/agent/nodes.py:452-479`
**Problem**: `_build_greeting_categories()` performs synchronous file I/O (`open(path)` + `json.load(f)`) inside the `greeting_node` async function. While the result is cached after the first call (TTLCache with 1-hour TTL), the cold-start read blocks the event loop thread. For a small JSON file this is ~1ms, but on network-mounted filesystems (some Cloud Run configurations) it could be longer.
**Impact**: Minimal for the common case (file is small, read is fast, result is cached). On cold start with network storage, a brief event loop block affects concurrent request latency.
**Fix**: Wrap in `asyncio.to_thread()` for the cold-start path, consistent with the `retrieve_node`'s ChromaDB pattern. Or accept as a known trade-off since the file is small and the cache eliminates repeated reads.

---

### F-011: `ErrorHandlingMiddleware` Does Not Include Security Headers on 500 Error Responses (Partial)

**Severity**: LOW
**Location**: `src/api/middleware.py:117-122, 166-170`
**Problem**: `ErrorHandlingMiddleware` includes `_SECURITY_HEADERS` on its 500 error responses, but the list is missing `strict-transport-security` (HSTS) which IS present in `SecurityHeadersMiddleware._STATIC_HEADERS`. The `content-security-policy` (per-request nonce) is also missing from error responses, which is acceptable since error pages don't serve scripts, but the HSTS gap means 500 error responses from the outermost middleware don't enforce HTTPS downgrade protection.
**Impact**: Low -- HSTS is a long-lived header (2 years max-age) so browsers almost always have it cached from a previous successful response. A 500 response missing HSTS only matters if it's the very first response a browser ever receives from this domain.
**Fix**: Add `(b"strict-transport-security", b"max-age=63072000; includeSubDomains")` to `ErrorHandlingMiddleware._SECURITY_HEADERS`.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH | 2 |
| MEDIUM | 5 |
| LOW | 4 |
| **Total** | **11** |

### Top 3 Findings

1. **F-001 (HIGH)**: `_dispatch_to_specialist` never calls `cb.record_failure()` on LLM errors, so the circuit breaker for the dispatch path effectively never trips under sustained failures.

2. **F-002 (HIGH)**: `threading.Lock` in two Firestore client accessors (`casino/config.py`, `data/guest_profile.py`) blocks the async event loop -- the exact anti-pattern already fixed in `nodes.py`.

3. **F-003 (MEDIUM)**: Streaming PII redactor splits safe/lookahead boundaries on different text representations (redacted vs original), potentially allowing fragmented PII to leak across SSE token events.

### Notable Strengths

- **State parity assertion at import time** (graph.py:494-501): A `ValueError` (not `assert`) catches state schema drift in all environments. This is a pattern I want to see in every LangGraph project.
- **Degraded-pass validation strategy**: First-attempt validator failure = PASS (deterministic guardrails already ran), retry + failure = FAIL. Correctly balances availability and safety.
- **E2E integration tests through real graph**: `TestEndToEndGraphIntegration` exercises the full HTTP -> middleware -> SSE -> graph -> node path with real graph execution. This catches wiring bugs invisible to unit tests.
- **Streaming PII redactor as defense-in-depth**: Even though `persona_envelope_node` already runs PII redaction, the streaming path gets its own real-time redactor. Belt-and-suspenders for regulated environment.
- **Circuit breaker `record_cancellation()`**: CancelledError (SSE disconnect) does not inflate failure count. This prevents normal traffic patterns from tripping the breaker.
