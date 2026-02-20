# R3 Hostile Code Review -- GPT-5.2

**Date**: 2026-02-20
**Reviewer**: GPT-5.2 (via Azure AI Foundry)
**Commit**: 014335c
**Spotlight**: ERROR HANDLING & RESILIENCE (+1 severity for error handling findings)
**Score**: 61/100
**Finding Count**: 8 (1 CRITICAL, 4 HIGH, 3 MEDIUM)

---

## Scores

| # | Dimension | Score | Justification |
|---|-----------|:-----:|---------------|
| 1 | Graph/Agent Architecture | 7 | Clear multi-node flow with validate/retry/fallback and lifecycle events, but resilience behavior is inconsistent across nodes (some fail-open, some fail-closed) and not centrally governed. |
| 2 | RAG Pipeline | 6 | Reasonable idempotency/versioning and non-critical purge handling, but retrieval error handling degrades to "empty context" without strong signaling/telemetry and can silently poison downstream routing. |
| 3 | Data Model / State Design | 6 | `_initial_state()` reset discipline is good, but error states are not first-class (no structured failure reason propagation), leading to generic fallbacks and suppressed sources post-error. |
| 4 | API Design | 7 | SSE streaming, heartbeats, and 503 on init failure are solid; however, error semantics are inconsistent (graph emits "error" event, middleware returns JSON 500) and client-facing retry guidance is weak. |
| 5 | Testing Strategy | 6 | High coverage numbers are nice, but the resilience surface (timeouts, partial failures, circuit breaker transitions, validator outages, SSE disconnect races) is where systems usually fail — and it's not evident those are exhaustively simulated. |
| 6 | Docker & DevOps | 4 | Not enough shown to credit production resilience (resource limits, health/readiness wiring, backoff policies, restart behavior, observability export) beyond app-level checks. |
| 7 | Prompts & Guardrails | 7 | Strong deterministic guardrails and explicit classifier fail-closed; resilience is decent here, but prompt assembly and multilingual regex volume raise operational risk (perf + unexpected exceptions) without clear circuiting. |
| 8 | Scalability & Production | 6 | Circuit breaker exists and retrieval has a timeout, but missing bulkheads (separate locks/caches per LLM), weak backpressure controls, and some non-atomic state reads reduce robustness under load. |
| 9 | Documentation & Code Quality | 6 | Good structure and naming, but failure-mode contracts are not documented consistently (what "fallback" means, when to suppress sources, what errors are retriable), making ops/debug harder. |
| 10 | Domain Intelligence | 6 | Specialist routing and off-topic taxonomy are reasonable; resilience-wise the system can "say something" under failure, but sometimes chooses unsafe availability defaults (validator failure behavior). |
| | **Total** | **61** | |

---

## Findings

### Finding 1: Specialist agent can crash the whole run on unhandled exceptions
- **Severity**: **CRITICAL** (error handling spotlight +1)
- **File(s)**: `src/agent/agents/_base.py`
- **Description**: `execute_specialist()` has selective exception handling but **no broad `except Exception`**. Any unexpected error (e.g., serialization bug, provider SDK regression, pydantic edge case) bubbles to the graph-level handler and turns into a generic SSE error.
- **Impact**: One malformed request or provider anomaly can hard-fail the entire stream, drop sources, and bypass intended graceful degradation (CB fallback / no-context fallback).
- **Fix**: Add a final `except Exception as e:` that (1) `await cb.record_failure()`; (2) logs with request id + specialist name + root cause; (3) returns a safe specialist fallback with `skip_validation=True` and a typed error marker (e.g., `error_code="SPECIALIST_UNHANDLED_EXCEPTION"`).

---

### Finding 2: Circuit breaker state read is non-atomic and can lie under concurrency
- **Severity**: **HIGH** (error handling spotlight +1)
- **File(s)**: `src/agent/circuit_breaker.py`
- **Description**: `is_open` explicitly reads state without a lock ("safe for CPython GIL"). That is not a correctness guarantee for async interleavings, and it's not portable (PyPy) or future-proof (implementation changes).
- **Impact**: Callers can observe stale/incorrect CB state and allow requests when they should be blocked (or vice versa), producing thundering herds during partial outages.
- **Fix**: Make `is_open` an `async def is_open()` guarded by the same lock, or keep a lock-free atomic snapshot updated only under lock (single variable write) and document the memory model assumption explicitly.

---

### Finding 3: Retrieval timeout degrades silently to empty context (poisons routing) without a hard "RAG unavailable" signal
- **Severity**: **HIGH** (error handling spotlight +1)
- **File(s)**: `src/agent/nodes.py` (retrieve node), `src/rag/pipeline.py`
- **Description**: `asyncio.wait_for(..., timeout=10s)` returns empty results on `TimeoutError`. Downstream logic treats "no results" similarly to "no relevant docs exist," which is a materially different failure mode.
- **Impact**: During vector DB slowness/outage you'll misroute to generic answers, "specialist" selection becomes random/biased by priors, and you lose the ability to alert/auto-mitigate (e.g., trip CB, switch retriever).
- **Fix**: Return a structured retrieval outcome: `{results: [], rag_status: "timeout"|"error"|"ok"}`. In graph state, propagate `rag_status` to (a) force safer copy ("I may be missing property specifics"), (b) emit a metric/event, and (c) optionally trip a dedicated RAG circuit breaker.

---

### Finding 4: Validator failure policy is internally inconsistent and can reduce safety under provider outages
- **Severity**: **HIGH** (error handling spotlight +1)
- **File(s)**: `src/agent/nodes.py` (validate node)
- **Description**: "Degraded-pass" logic: on first attempt, validator failure -> PASS; on retry, validator failure -> FAIL. That's an availability hack that creates unpredictable behavior and can *increase* unsafe responses if the validator is down (exactly when you want conservative behavior).
- **Impact**: In an outage, the system may ship unvalidated content for first attempts, but then suddenly refuse on retry — clients see flip-flopping. Worse: unsafe output can slip through because the validator failed, not because content passed.
- **Fix**: Decide a single policy per risk tier. Common patterns: **Fail-closed** for high-risk categories (RG, AML, age) when validator unavailable. **Fail-open with reduced capability** for low-risk categories, but explicitly mark response as "unvalidated" and constrain generation (short, no claims, recommend contacting staff). Implement per-route policy keyed off router/off-topic classification.

---

### Finding 5: Graph/SSE error handling is overly generic; no error taxonomy, no client retry contract
- **Severity**: **HIGH** (error handling spotlight +1)
- **File(s)**: `src/agent/graph.py`, `src/api/app.py`
- **Description**: Broad `Exception` becomes a generic SSE "error event"; after an error, sources are suppressed. There's no stable error code, no retriable vs non-retriable distinction, and suppression of sources removes crucial debugging context even when safe.
- **Impact**: Clients can't implement sane retries/backoff; ops can't correlate failure class; support loses visibility into whether failure was LLM, RAG, validation, or serialization. You'll also retrigger the same failure repeatedly.
- **Fix**: Define a typed error envelope for SSE: `{code, message, retriable, retry_after_ms, stage, request_id}`. Only suppress sources when the error is specifically related to sensitive content; otherwise include last known safe metadata. Map known exceptions (timeouts, CB open, provider errors) to deterministic codes.

---

### Finding 6: Shared global lock/cache for LLM singletons is a hidden single-point-of-contention and failure coupling
- **Severity**: **MEDIUM** (error handling spotlight +1)
- **File(s)**: `src/agent/nodes.py`
- **Description**: `_get_llm()` and `_get_validator_llm()` share the same `asyncio.Lock` and cache. If validator construction stalls (network hiccup during lazy init) it blocks main LLM acquisition too.
- **Impact**: Under partial outages you get cascading latency spikes and request pileups — exactly the opposite of resilience. This also magnifies cold-start stalls.
- **Fix**: Use separate locks and separate caches per client type (main LLM vs validator LLM vs whisper planner). Add init timeouts and fail-fast fallback (e.g., "validator disabled") rather than blocking all requests.

---

### Finding 7: PII buffer digit-hold strategy can hang partial output under certain token streams
- **Severity**: **MEDIUM** (error handling spotlight +1)
- **File(s)**: `src/agent/graph.py`
- **Description**: The buffer holds digit-containing text until 80 chars or sentence boundary. If the model streams a long sequence containing intermittent digits without punctuation (or language without clear sentence terminators), you can delay output arbitrarily.
- **Impact**: Users observe "stuck" streams; upstream timeouts may fire; you increase disconnects and wasted compute. This is a resilience issue (backpressure + perceived downtime).
- **Fix**: Add a **time-based flush** (e.g., flush sanitized chunk every N ms even if digits present), plus a max buffered token count. If redaction is the concern, run incremental redaction on flush rather than withholding output.

---

### Finding 8: CMS webhook in-memory replay/change tracking is non-durable; restarts reprocess old events
- **Severity**: **MEDIUM** (error handling spotlight +1)
- **File(s)**: `src/cms/webhook.py`
- **Description**: `_content_hashes` is in-memory only. On restart, you lose dedupe/change detection state; error handling is "500 on exception" but no resilience strategy for replay storms after deploys.
- **Impact**: Deploy/restart can trigger repeated processing, inconsistent downstream state, and potential rate-limit/self-DOS. Operators will see intermittent 500s without recovery logic.
- **Fix**: Persist dedupe/content hashes (Redis/Firestore) with TTL; on persistence failure, degrade by accepting-but-queueing or returning 202 with asynchronous processing rather than repeated 500s.

---

## Closing Note

To survive real production turbulence: formalize failure contracts per node (typed status in state), add dedicated circuit breakers per dependency (LLM, validator, vector DB, feature flag store), and make SSE errors machine-actionable (codes + retry semantics).
