# Production Review Round 13 -- GPT Focus (Calibrated)

**Reviewer**: GPT-5.2 (simulated by Claude Opus 4.6)
**Repo**: Oded-Ben-Yair/hey-seven | **Commit**: fda95e3
**Date**: 2026-02-21
**Focus**: API Design + Testing (Spotlight, +1 severity)

---

## Scoring Summary

| # | Dimension | Score | Justification |
|---|-----------|-------|---------------|
| 1 | Graph Architecture | 9 | 11-node StateGraph with validation loop, structured dispatch, per-turn state reset with compile-time parity check. Dual-layer routing (compliance_gate + router) is defense-in-depth done right. Feature flag architecture (build-time topology vs runtime behavior) is well-documented. Minor: `_dispatch_to_specialist` catches `Exception` broadly -- acceptable given google-genai version variance, but typed catch would be safer. |
| 2 | RAG Pipeline | 8 | Per-item chunking, RRF reranking, SHA-256 idempotent IDs, version-stamp purging all present. Retrieval timeout guard with `asyncio.wait_for` is good. `asyncio.to_thread` wrapper for sync ChromaDB is correct. Not fully evaluated in this round (file set). |
| 3 | Data Model | 9 | `PropertyQAState` TypedDict with `Annotated` reducers for `messages` and `responsible_gaming_count` is clean. `_keep_max` reducer prevents counter reset. `RetrievedChunk` TypedDict documents the implicit dict contract. `DispatchOutput` with `Literal` types prevents invalid specialist names. Parity check at import time catches state schema drift. |
| 4 | **API Design (SPOTLIGHT)** | 8 | Pure ASGI middleware stack is correctly ordered. SSE streaming with heartbeat, timeout, error recovery all present. CMS and SMS webhook handlers have signature verification and replay protection. **Deductions**: (F-001) heartbeat timing logic has a subtle gap, (F-003) CMS webhook handler mixes sync signature verification with async handler, (F-006) `/feedback` endpoint does not validate thread_id existence, (F-007) static fallback graph structure could drift from real graph. Structured error taxonomy with `ErrorCode` enum is clean. Health endpoint correctly separates liveness from readiness. |
| 5 | **Testing Strategy (SPOTLIGHT)** | 8 | 1013 lines of test code across the 4 test files reviewed. Full E2E graph integration tests exercising real compiled StateGraph through HTTP -- this is rare and excellent. SSE event sequence contract tests verify metadata-first/done-last invariant. Concurrent request tests validate middleware under async load. **Deductions**: (F-002) conftest clears `_greeting_cache` from wrong module, (F-004) no test for CMS webhook endpoint, (F-005) no test for `/feedback` rate limiting, (F-009) E2E happy path test mocks too many specialist agents individually rather than through a unified mock. |
| 6 | Docker & DevOps | 8 | Single-container deployment with MemorySaver for dev, FirestoreSaver for prod. Cloud Run probe separation (liveness vs readiness) correctly prevents instance flapping. `BoundedMemorySaver` with LRU eviction prevents OOM in long sessions. Not all DevOps files were in scope. |
| 7 | Prompts & Guardrails | 9 | 5-layer deterministic guardrails (prompt injection, responsible gaming, age verification, BSA/AML, patron privacy) execute before any LLM call. Responsible gaming escalation after 3+ triggers is thoughtful. `string.Template.safe_substitute` prevents format-string injection. Structured output routing via Pydantic + Literal types (no substring matching). |
| 8 | Scalability & Production | 8 | TTL-cached LLM singletons with separate locks for main/validator. Circuit breaker with rolling window, half-open probing, cancellation handling. Rate limiter with LRU eviction, async lock, stale-client sweep. `InMemoryBackend` with probabilistic sweep and max store size. Documented Cloud Run scaling limitations (per-instance rate limiting). Redis backend ready. |
| 9 | Trade-off Documentation | 9 | Feature flag architecture comment block (lines 375-409 in graph.py) is exemplary -- explains why build-time, why not all runtime, and emergency disable procedure. Degraded-pass validation strategy documented inline with rationale. BoundedMemorySaver purpose and limits documented. Consent hash chain HMAC rationale is clear. |
| 10 | Domain Intelligence | 9 | TCPA compliance with quiet hours, area-code timezone mapping with MNP caveat, consent hash chain with HMAC authentication. Three-tier consent hierarchy (none/express/written). Responsible gaming escalation at 3+ triggers. BSA/AML, patron privacy, age verification all handled as distinct query types with appropriate responses. |

**Total: 85/100**

---

## Findings

### F-001: SSE Heartbeat Timer Does Not Account for Time Spent Yielding Events

**Severity**: MEDIUM (spotlight +1 = MEDIUM)
**Location**: `src/api/app.py` lines 174-188
**Problem**: The heartbeat logic checks `now - last_event_time >= _HEARTBEAT_INTERVAL` and sends a ping *before* yielding the actual event. But `last_event_time` is updated *after* yielding, not before the heartbeat check. If a sequence of rapidly-yielded events takes longer than 15 seconds due to backpressure (slow client), multiple heartbeats could be emitted back-to-back before any real event. Additionally, the heartbeat is emitted *before* the event, then `last_event_time` is reset -- meaning the heartbeat itself does not reset the timer before yielding the event, so the event immediately after a heartbeat could trigger another heartbeat check as stale.
**Impact**: Under slow clients or backpressure, unnecessary heartbeat events. Not a crash, but violates the "1 ping per 15s" contract.
**Fix**:
```python
now = asyncio.get_event_loop().time()
if now - last_event_time >= _HEARTBEAT_INTERVAL:
    yield {"event": "ping", "data": ""}
    last_event_time = now  # Reset after ping
yield event
last_event_time = asyncio.get_event_loop().time()  # Reset after event
```

---

### F-002: Conftest Clears `_greeting_cache` from `src.agent.nodes` but It Is Defined There

**Severity**: LOW (spotlight +1 = LOW)
**Location**: `tests/conftest.py` lines 43-48
**Problem**: The conftest imports `_greeting_cache` from `src.agent.nodes`, which is correct since the cache is defined at module level in `nodes.py` line 436. However, the comment says "greeting cache" but this cache is keyed by `casino_id` and caches the result of `_build_greeting_categories()`. The conftest does NOT clear the `_content_hashes` cache in `src.cms.webhook` -- wait, it does (line 116-120). This finding is actually about the fact that the conftest does not clear `_DELIVERY_LOG` and `_sms_webhook._idempotency_tracker` atomically -- if a test imports one but not the other, partial state may leak. Actually, upon re-reading, both are cleared independently. **Revised**: This is a non-issue. The conftest correctly clears `_greeting_cache`. Withdrawing this finding.

**WITHDRAWN** -- conftest correctly clears `_greeting_cache` from `src.agent.nodes`.

---

### F-003: CMS Webhook `verify_webhook_signature` Is Synchronous but Called from Async Handler

**Severity**: LOW
**Location**: `src/cms/webhook.py` lines 32-81, `src/api/app.py` line 461
**Problem**: `verify_webhook_signature` in `cms/webhook.py` is a synchronous function doing HMAC computation. It is called from the async `handle_cms_webhook` coroutine via `src/api/app.py`. While HMAC-SHA256 on a small payload is fast (microseconds), the function also calls `time.time()` (fine) and does string operations (fine). This is acceptable for the current scale, but inconsistent with the SMS webhook's `verify_webhook_signature` which is `async`. The CMS version should be `async` for API consistency, even though the implementation is CPU-bound.
**Impact**: No production impact at current scale. API inconsistency between CMS and SMS webhook verification functions.
**Fix**: Make `verify_webhook_signature` in `cms/webhook.py` an `async def` for consistency. The caller already `await`s `handle_cms_webhook`, so the change is backward-compatible.

---

### F-004: No Test Coverage for CMS Webhook Endpoint

**Severity**: HIGH (spotlight +1 = HIGH)
**Location**: `tests/test_api.py` -- missing
**Problem**: The `/cms/webhook` endpoint (app.py lines 449-476) has zero test coverage in `test_api.py`. The endpoint handles signature verification, payload parsing, validation, content hashing, and re-indexing -- all critical for content integrity in a casino-regulated environment. There is no test for:
- Successful webhook processing (indexed)
- Unchanged content (hash match)
- Invalid signature rejection (403)
- Missing required fields (rejected)
- Quarantined items (validation failure)
- Re-indexing failure (non-critical, should still return indexed)
**Impact**: CMS webhook bugs would ship to production undetected. A malformed payload could corrupt the knowledge base without any test catching it.
**Fix**: Add a `TestCmsWebhookEndpoint` class with at least 5 test cases covering the happy path, signature rejection, missing fields, quarantine, and unchanged content.

---

### F-005: `/feedback` Endpoint Does Not Rate-Limit Effectively and Has No Persistence

**Severity**: MEDIUM (spotlight +1 = MEDIUM)
**Location**: `src/api/app.py` lines 481-496, `src/api/middleware.py` line 409
**Problem**: The `/feedback` endpoint is rate-limited (middleware.py line 409 includes `/feedback` in rate-limited paths), but shares the same rate limit counter as `/chat`. A user who exhausts their 20 req/min quota on `/chat` cannot submit feedback, and vice versa. More critically, feedback is only logged (`logger.info`) -- there is no persistence layer. The endpoint returns `FeedbackResponse(status="received")` but the feedback data is discarded after logging. The docstring claims "In production, feedback is forwarded to LangFuse as a score" but this forwarding is not implemented.
**Impact**: Feedback data is lost on container recycle. The endpoint is effectively a no-op beyond logging. Shared rate limit with `/chat` creates UX friction.
**Fix**:
1. Implement LangFuse score forwarding as the docstring claims, or add a TODO with ticket ID.
2. Consider separate rate limit buckets for `/chat` and `/feedback`.

---

### F-006: `StreamingPIIRedactor._scan_and_release` Applies Redaction to Buffer but Keeps Original Buffer Tail

**Severity**: MEDIUM
**Location**: `src/agent/streaming_pii.py` lines 101-118
**Problem**: When `force=False`, the method applies `redact_pii()` to the full buffer, yields `redacted[:-_MAX_PATTERN_LEN]`, then keeps `self._buffer[-_MAX_PATTERN_LEN:]` (the ORIGINAL buffer tail). This means the lookahead window retains the original (un-redacted) text for re-scanning on the next `feed()`. If PII was partially in the safe prefix and partially in the lookahead, the PII is redacted in the yielded prefix, but the lookahead tail still contains the original PII fragment. On the next `feed()`, the PII fragment in the tail will be re-scanned with new content appended. This is actually correct -- the re-scan will catch the PII again if it forms a complete pattern with new content. However, if the PII pattern was entirely within the yielded safe prefix, the tail is clean. If the PII spans the boundary, keeping the original tail ensures the pattern is caught on re-scan. **Revised**: This is correct behavior. The original buffer tail is retained precisely because redaction may change string lengths, making index-based slicing of the redacted text unreliable for determining the lookahead boundary. Withdrawing.

**WITHDRAWN** -- The original-buffer-tail retention is the correct design choice.

---

### F-007: Static Graph Structure Fallback Can Drift from Actual Graph

**Severity**: LOW (spotlight +1 = LOW)
**Location**: `src/api/app.py` lines 310-336
**Problem**: `_STATIC_GRAPH_STRUCTURE` is a manually maintained dict that serves as fallback when the agent is not initialized or introspection fails. If the graph topology changes (nodes added/removed, edges modified), the static structure must be manually updated. There is no automated check that `_STATIC_GRAPH_STRUCTURE` matches the actual compiled graph.
**Impact**: Frontend graph visualization could show an incorrect topology when the agent is degraded.
**Fix**: Add a test that compiles the real graph, introspects it, and asserts the static fallback matches (at minimum, same node count and edge count).

---

### F-008: `BoundedMemorySaver._track_thread` Accesses `self._inner.storage` Without API Guarantee

**Severity**: MEDIUM
**Location**: `src/agent/memory.py` lines 68-74
**Problem**: The LRU eviction logic accesses `self._inner.storage` via `hasattr` check. `MemorySaver.storage` is an implementation detail of LangGraph's `MemorySaver` class, not a documented public API. If LangGraph upgrades from 0.2.60 to a newer version that renames or restructures `storage`, eviction silently stops working (no error, just no eviction), and memory grows unbounded.
**Impact**: After a LangGraph version upgrade, `BoundedMemorySaver` could silently stop evicting threads, leading to OOM in long-running containers.
**Fix**:
1. Pin LangGraph version (already done: `langgraph==0.2.60`).
2. Add a startup assertion: `assert hasattr(MemorySaver(), "storage"), "LangGraph MemorySaver API changed -- BoundedMemorySaver eviction will not work"`.
3. Document this dependency in the `BoundedMemorySaver` docstring.

---

### F-009: E2E Happy Path Test Mocks 8+ Individual Patch Targets

**Severity**: LOW (spotlight +1 = LOW)
**Location**: `tests/test_api.py` lines 932-958
**Problem**: `test_happy_path_property_qa_e2e` uses 9 separate `patch()` context managers to mock individual LLMs, circuit breakers, and retrievers across different agent modules (`nodes`, `host_agent`, `hotel_agent`, `whisper_planner`, `compliance_gate`). This test is fragile: any refactor that moves `_get_llm` or `_get_circuit_breaker` to a different module path breaks the test without changing behavior. The test is also not truly E2E -- it is a "middleware + graph wiring" test with mocked internals.
**Impact**: High maintenance burden. Refactoring agent modules requires updating 8+ patch paths in this single test.
**Fix**: Create a test fixture that provides a pre-configured mock graph (or a unified mock factory) to reduce patch sprawl. Alternatively, create a dedicated `_make_e2e_app_with_mocked_llm()` helper that centralizes mock setup.

---

### F-010: `InMemoryBackend._maybe_sweep` Uses Non-Deterministic Random for Sweep Probability

**Severity**: LOW
**Location**: `src/state_backend.py` lines 64-88
**Problem**: `_maybe_sweep()` uses `random.random()` to probabilistically trigger cleanup. In testing, this means sweep behavior is non-deterministic -- tests cannot reliably trigger or prevent a sweep. The `_sweep_counter` is used for counting but not for deterministic triggering (e.g., "sweep every 100th write").
**Impact**: Tests that depend on specific store sizes may flake due to random sweeps. No production impact.
**Fix**: Use counter-based sweep (`self._sweep_counter % 100 == 0`) instead of random probability. The amortized cost is identical, but behavior becomes deterministic and testable.

---

### F-011: `_dispatch_to_specialist` Does Not Record CB Failure on LLM Error

**Severity**: MEDIUM
**Location**: `src/agent/graph.py` lines 237-243
**Problem**: When the structured dispatch LLM call fails (lines 237-243), the exception is caught and the function falls back to keyword counting. However, `cb.record_failure()` is never called -- only `cb.record_success()` is called on success (line 223). This means LLM failures in the dispatch path do not contribute to circuit breaker trip counts. If the LLM is consistently failing for dispatch but succeeding for generation (or vice versa), the circuit breaker never opens for the failing path.
**Impact**: Circuit breaker cannot detect sustained dispatch LLM failures. The keyword fallback silently masks the failures, but without CB tracking, the system cannot surface the degradation via the health endpoint.
**Fix**: Add `await cb.record_failure()` in both exception handlers (lines 238 and 241):
```python
except (ValueError, TypeError) as exc:
    logger.warning("Structured dispatch parsing failed: %s", exc)
    await cb.record_failure()
except Exception:
    logger.warning("Structured dispatch LLM call failed", exc_info=True)
    await cb.record_failure()
```

---

### F-012: PII Redaction Phone Pattern Has High False Positive Rate on 7-Digit Sequences

**Severity**: LOW
**Location**: `src/api/pii_redaction.py` line 35
**Problem**: The phone regex `\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b` matches 10-digit sequences that could be legitimate content in a casino context. For example, "The arena seats 10,000 fans" would not match (comma separator), but "Room 305 has 555 sqft and 1234 thread count sheets" could partially match if digits align. More relevant: gaming statistics like "over 5000 slots and 300 table games" are safe, but promotional codes or reference numbers could false-positive.
**Impact**: Low in practice -- casino content rarely has 10-digit sequences without separators. But edge cases exist.
**Fix**: Accept as-is for current scale. Consider adding a casino-domain allowlist for known numeric patterns (room numbers, game counts) if false positives are reported.

---

## Summary

### Strengths
1. **E2E integration tests through real compiled graph** -- This is the most impressive testing pattern in the codebase. Five compliance guardrail paths tested end-to-end from HTTP POST through middleware to SSE output, plus a full happy-path test exercising all 8 nodes. This is rare in LangGraph projects.
2. **Middleware isolation tests** -- Each ASGI middleware is tested independently with Starlette test utilities, plus concurrent access tests validating async safety. CSP nonce uniqueness test is thorough.
3. **Defense-in-depth architecture** -- Dual-layer routing (compliance_gate + router), streaming PII redaction as defense-in-depth on top of persona_envelope redaction, feature flag dual-layer (build-time topology + runtime behavior), and fail-closed PII redaction.
4. **Documentation quality** -- Feature flag architecture comment block, degraded-pass validation rationale, circuit breaker concurrency warnings, and BoundedMemorySaver purpose are all production-grade inline documentation.

### Weaknesses
1. **CMS webhook has zero E2E test coverage** (F-004, HIGH) -- Critical for content integrity in a regulated environment.
2. **Feedback endpoint is effectively a no-op** (F-005, MEDIUM) -- Claims LangFuse forwarding but does not implement it.
3. **Dispatch CB failures not tracked** (F-011, MEDIUM) -- Circuit breaker blind spot in specialist dispatch path.
4. **E2E happy path test has high mock fragility** (F-009, LOW) -- 9 separate patches create maintenance burden.

### Score Context
R12 average was 80.0. This codebase has matured significantly in API design and testing since R11. The E2E integration tests through real compiled graph are a standout -- most LangGraph projects stop at unit testing individual nodes. The middleware stack is well-tested with concurrent access, LRU eviction, and TTL refresh scenarios. The main gaps are CMS webhook coverage and the feedback endpoint's unfulfilled LangFuse promise.

**R13 Score: 85/100** (+5 from R12 average)

---

## Finding Severity Counts

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH | 1 |
| MEDIUM | 4 |
| LOW | 5 |
| **Total** | **10** (2 withdrawn) |
