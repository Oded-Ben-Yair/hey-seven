# Round 10: DeepSeek Final Adversarial Review

**Date**: 2026-02-20
**Model**: DeepSeek-V3.2-Speciale (extended thinking)
**Reviewer**: DeepSeek (replaces Grok for R10)
**Mandate**: Final production sign-off. ALL findings +1 severity. Focus: async correctness, state machine bugs, mathematical bounds, algorithmic edge cases.
**Context**: R1=67.3 -> R9=67.1. 1256 tests, 90.67% coverage.

---

## Findings

### F1: Circuit Breaker Stuck in Half-Open on CancelledError (CRITICAL)

**Category**: Scalability & Production
**Severity**: CRITICAL (+1 from HIGH)
**File**: `src/agent/agents/_base.py:171-172`

**Description**: When the circuit breaker is in `half_open` state and allows a single probe request (`_half_open_in_progress=True`), if `asyncio.CancelledError` is raised during `llm.ainvoke()`, the exception is re-raised without calling `record_success()` or `record_failure()`. This leaves the circuit breaker permanently stuck: `_state="half_open"`, `_half_open_in_progress=True`. All subsequent `allow_request()` calls return `False`, blocking ALL LLM requests for the lifetime of the process.

**Evidence**:
```python
# _base.py:171-172
except asyncio.CancelledError:
    raise  # Neither record_success() nor record_failure() called

# circuit_breaker.py:160-163 — allow_request() in half_open with in_progress=True
if self._state == "half_open" and not self._half_open_in_progress:
    self._half_open_in_progress = True
    return True
return False  # <-- Stuck here forever
```

**Impact**: After a single client disconnect during a half-open probe, the entire agent becomes permanently unresponsive. Only a process restart recovers. In Cloud Run with `min-instances=1`, this means total service outage until the container restarts or scales.

**Fix**:
```python
except asyncio.CancelledError:
    # Treat cancellation as failure to unblock the circuit breaker.
    # Without this, half_open_in_progress stays True forever.
    await cb.record_failure()
    raise
```

---

### F2: PII Buffer `_PII_MAX_BUFFER` is Dead Code (MEDIUM)

**Category**: Prompts & Guardrails
**Severity**: MEDIUM (+1 from LOW)
**File**: `src/agent/graph.py:560`

**Description**: The PII streaming buffer flush condition checks `len(_pii_buffer) >= _PII_MAX_BUFFER or len(_pii_buffer) >= _PII_FLUSH_LEN`. Since `_PII_MAX_BUFFER` (500) > `_PII_FLUSH_LEN` (80), the 80-char threshold always fires first, making the 500-char hard cap unreachable dead code. The comment says "Hard cap: force-flush regardless of content (prevents unbounded growth)" but this cap can never actually be reached.

**Evidence**:
```python
# graph.py:494-495
_PII_FLUSH_LEN = 80    # Always triggers first
_PII_MAX_BUFFER = 500  # Never reached

# graph.py:560 — condition when has_digits=True:
elif len(_pii_buffer) >= _PII_MAX_BUFFER or len(_pii_buffer) >= _PII_FLUSH_LEN or ...
#    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ redundant; 80 < 500
```

**Impact**: No functional impact (buffer flushes at 80 chars which is correct), but the code misleads readers into thinking there is a safety net at 500 chars. If `_PII_FLUSH_LEN` is ever increased above `_PII_MAX_BUFFER`, the "hard cap" label would become a lie.

**Fix**: Remove `_PII_MAX_BUFFER` from the `elif` condition and use it only as the unconditional cap:
```python
# Unconditional hard cap first (defense-in-depth)
if len(_pii_buffer) >= _PII_MAX_BUFFER:
    async for tok_event in _flush_pii_buffer():
        yield tok_event
elif not has_digits:
    async for tok_event in _flush_pii_buffer():
        yield tok_event
elif len(_pii_buffer) >= _PII_FLUSH_LEN or _pii_buffer.endswith(("\n", ". ", "! ", "? ")):
    async for tok_event in _flush_pii_buffer():
        yield tok_event
```

---

### F3: `execute_specialist` Hard-Codes `retry_count=1` Instead of Incrementing (HIGH)

**Category**: Graph Architecture
**Severity**: HIGH (+1 from MEDIUM)
**File**: `src/agent/agents/_base.py:169`

**Description**: On `ValueError`/`TypeError` during LLM invocation, the error handler returns `"retry_count": 1` regardless of the current retry count. If the state already has `retry_count=1` (from a previous validation retry), this overwrites it with 1, resetting the retry budget. The validate_node then increments to 2 on the next failure, but since the graph's `GRAPH_RECURSION_LIMIT=10`, the retry loop could execute far more times than intended before the recursion limit halts it.

**Evidence**:
```python
# _base.py:166-170
return {
    "messages": [AIMessage(content=_fallback_message(...))],
    "skip_validation": False,
    "retry_count": 1,  # Should be: retry_count + 1
}
```

The local variable `retry_count = state.get("retry_count", 0)` is available but not used for the increment.

**Impact**: If a `ValueError` occurs on the second attempt (retry_count already 1), the counter resets to 1 instead of incrementing to 2. The validate_node will allow another retry (since retry_count < 1 check uses the reset value). This creates a 2-retry loop per ValueError occurrence instead of the intended single retry, doubling LLM cost per parse error.

**Fix**:
```python
"retry_count": retry_count + 1,
```

---

### F4: `_get_circuit_breaker` Uses `@lru_cache` -- No TTL for Credential Rotation (MEDIUM)

**Category**: Scalability & Production
**Severity**: MEDIUM (+1 from LOW)
**File**: `src/agent/circuit_breaker.py:214`

**Description**: The circuit breaker singleton uses `@lru_cache(maxsize=1)` which never expires. While the circuit breaker itself doesn't hold credentials, the `get_settings()` call inside the cached factory means the `CB_FAILURE_THRESHOLD` and `CB_COOLDOWN_SECONDS` values are frozen at first call. If these settings are updated via environment variables and `get_settings.cache_clear()` is called, the circuit breaker will still use the old thresholds.

**Evidence**:
```python
@lru_cache(maxsize=1)
def _get_circuit_breaker():
    settings = get_settings()  # Settings frozen at first call
    return CircuitBreaker(
        failure_threshold=settings.CB_FAILURE_THRESHOLD,
        cooldown_seconds=settings.CB_COOLDOWN_SECONDS,
    )
```

Meanwhile, the LLM singletons (`_get_llm`, `_get_validator_llm`, `_get_whisper_llm`) all use `TTLCache(maxsize=1, ttl=3600)` for credential rotation. The circuit breaker factory is inconsistent.

**Impact**: Low in practice (circuit breaker config rarely changes at runtime), but breaks the consistent TTL-cache pattern established for all other singletons. If an operator attempts to tune circuit breaker thresholds via env var hot-reload, the change is silently ignored.

**Fix**: Convert to TTLCache pattern matching LLM singletons:
```python
from cachetools import TTLCache
_cb_cache: TTLCache = TTLCache(maxsize=1, ttl=3600)

def _get_circuit_breaker():
    if "cb" not in _cb_cache:
        settings = get_settings()
        _cb_cache["cb"] = CircuitBreaker(...)
    return _cb_cache["cb"]
```

---

### F5: `failure_count` Property Mutates State Without Lock (HIGH)

**Category**: Scalability & Production
**Severity**: HIGH (+1 from MEDIUM)
**File**: `src/agent/circuit_breaker.py:77-80`

**Description**: The `failure_count` property calls `self._prune_old_failures()` which mutates `self._failure_timestamps` (a deque) by calling `popleft()`. This mutation occurs outside the `asyncio.Lock`, meaning concurrent coroutines reading `failure_count` while another is inside a locked `record_failure()` or `allow_request()` can corrupt the deque's internal state.

**Evidence**:
```python
@property
def failure_count(self) -> int:
    """Number of failures within the rolling window."""
    self._prune_old_failures()  # Mutates _failure_timestamps WITHOUT lock
    return len(self._failure_timestamps)
```

The `state` and `is_open`/`is_half_open` properties also call `_cooldown_expired()` without lock, but that method is read-only, so it's safe. The `failure_count` property is different because `_prune_old_failures()` modifies the deque.

**Impact**: Under concurrent load, if `failure_count` is accessed (e.g., for health checks or logging) while `record_failure()` holds the lock and is appending to the deque, the deque mutation from `_prune_old_failures()` races with the locked mutation. In CPython, deque operations are atomic due to the GIL, but asyncio coroutines yield at `await` points, not mid-operation. Since `_prune_old_failures()` is synchronous and CPython deque ops are GIL-protected, this is safe under CPython's GIL but is technically a concurrency bug that would manifest in a GIL-free Python runtime (PEP 703, Python 3.13+ free-threaded mode).

**Fix**: Either (a) remove the mutation from the property and make it a locked async method, or (b) document that `failure_count` is approximate and must not be used for control flow:
```python
@property
def failure_count(self) -> int:
    """Approximate failure count (no lock, no mutation). For monitoring only."""
    cutoff = time.monotonic() - self._rolling_window_seconds
    return sum(1 for t in self._failure_timestamps if t > cutoff)
```

---

### F6: PII Buffer Not Flushed on `CancelledError` in SSE Stream (MEDIUM)

**Category**: Prompts & Guardrails
**Severity**: MEDIUM (+1 from LOW)
**File**: `src/agent/graph.py:615-619, 628-631`

**Description**: When `asyncio.CancelledError` is caught in `chat_stream()`, the PII buffer (`_pii_buffer`) is NOT flushed before re-raising. Any accumulated tokens containing PII that were being buffered for redaction are silently dropped. While this prevents PII leakage (dropping is safer than emitting unredacted), it also means the client may have received partial streamed tokens from earlier flushes that form a complete PII pattern when combined with the dropped buffer contents.

**Evidence**:
```python
except asyncio.CancelledError:
    logger.info("SSE stream cancelled (client disconnect)")
    raise  # _pii_buffer contents dropped without flush

# ...
if _pii_buffer and not errored:  # Only flushes on normal completion
    async for tok_event in _flush_pii_buffer():
        yield tok_event
```

**Impact**: On client disconnect, buffered tokens (up to 80 chars) are silently dropped. Since the client disconnected, this is mostly harmless (they won't see the tokens). However, if the disconnect is a network hiccup and the client reconnects and retries, the partial response from the first attempt may contain the first half of a PII pattern that went undetected because the second half was in the buffer.

**Fix**: Acceptable as-is (fail-safe: dropping is safer than emitting). Add a comment documenting the intentional design choice:
```python
except asyncio.CancelledError:
    # Intentionally NOT flushing _pii_buffer on cancel: dropping buffered
    # tokens is safer than emitting potentially unredacted PII to a
    # disconnecting client. The partial tokens are lost (fail-safe).
    logger.info("SSE stream cancelled (client disconnect), dropping %d buffered chars", len(_pii_buffer))
    raise
```

---

### F7: Persona Envelope Issues `replace` Event After Streamed Tokens (MEDIUM)

**Category**: API Design
**Severity**: MEDIUM (+1 from LOW)
**File**: `src/agent/graph.py:568-581`

**Description**: The `persona_envelope` node is in `_NON_STREAM_NODES`, so when it completes, the `on_chain_end` handler emits a `replace` event with the full post-envelope content. However, the `generate` node already streamed tokens to the client via `token` events. The client receives a sequence: `token, token, ..., token, replace`. The `replace` event contains the same content as the streamed tokens (possibly with PII redaction applied), which overwrites the incrementally-built response. If persona_envelope modifies the content (PII redaction or SMS truncation), the replace is necessary and correct. But if no modification occurred (the common case), the replace is a no-op that causes a visible flash/re-render on the client.

**Evidence**:
```python
# persona.py:82-87 — only returns messages if content changed
if content != original:
    return {"messages": [AIMessage(content=content)]}
return {}  # No messages -> no on_chain_end with messages -> no replace event
```

Actually, upon closer inspection, persona_envelope_node returns `{}` when no modification is needed, which means `output.get("messages", [])` will be empty, and no `replace` event is emitted. This is correct. The `replace` event only fires when PII is actually redacted or SMS truncation applies.

**Impact**: Low. The behavior is correct. When persona modifies content, a `replace` event overwrites the streamed tokens, which is the intended behavior. No fix needed.

**Revised Severity**: LOW (informational, no actual bug).

---

### F8: `rolling_window_seconds` Not Configurable via Settings (LOW)

**Category**: Scalability & Production
**Severity**: LOW
**File**: `src/agent/circuit_breaker.py:230`

**Description**: The `_get_circuit_breaker()` factory passes `failure_threshold` and `cooldown_seconds` from settings but not `rolling_window_seconds`. The rolling window defaults to 300 seconds (5 minutes) and cannot be changed without code modification.

**Evidence**:
```python
def _get_circuit_breaker():
    settings = get_settings()
    return CircuitBreaker(
        failure_threshold=settings.CB_FAILURE_THRESHOLD,
        cooldown_seconds=settings.CB_COOLDOWN_SECONDS,
        # rolling_window_seconds not passed, defaults to 300.0
    )
```

**Impact**: Operators cannot tune the rolling window via environment variables. Minor operational inconvenience; the default 300s is reasonable.

**Fix**: Add `CB_ROLLING_WINDOW_SECONDS: float = 300.0` to `Settings` and pass it to the constructor.

---

### F9: `whisper_planner_node` Global Mutable State Without Lock (MEDIUM)

**Category**: Scalability & Production
**Severity**: MEDIUM (+1 from LOW)
**File**: `src/agent/whisper_planner.py:85-87, 130, 175-176`

**Description**: The whisper planner's failure counter uses module-level mutable globals (`_failure_count`, `_failure_alerted`) with `global` declaration and direct mutation. The comment says "No async lock needed: counter is only modified inside whisper_planner_node() which runs sequentially within a single graph invocation." However, if multiple concurrent graph invocations exist (e.g., two users chatting simultaneously on different thread_ids), two concurrent executions of `whisper_planner_node` can race on these globals.

**Evidence**:
```python
# whisper_planner.py:85-87
_failure_count: int = 0
_FAILURE_ALERT_THRESHOLD: int = 10
_failure_alerted: bool = False

# whisper_planner.py:130, 175-176 — used via `global` inside async function
global _failure_count, _failure_alerted
_failure_count += 1  # Read-modify-write without lock
```

**Impact**: Under concurrent load, `_failure_count` increments may be lost (two coroutines read same value, both increment to N+1 instead of N+2). This is a benign race: the counter is only used for alerting, and a missed increment delays the alert by one failure. Not a correctness issue.

**Fix**: Acceptable as-is for alerting purposes. Add a comment acknowledging the benign race:
```python
# Benign race: concurrent increments may lose counts. Acceptable for
# alerting (off-by-one delays alert by one failure, never suppresses it).
```

---

### F10: `_get_api_key()` in ApiKeyMiddleware Uses `time.monotonic()` for TTL (LOW)

**Category**: API Design
**Severity**: LOW
**File**: `src/api/middleware.py:241-248`

**Description**: The API key cache TTL in `ApiKeyMiddleware._get_api_key()` uses `time.monotonic()` for expiry tracking. The comment says "Atomic tuple: (key, timestamp)" but the read-then-write pattern (`cached = self._cached; if now - cached[1] > TTL; self._cached = (key, now)`) is not atomic under asyncio concurrency. Two concurrent coroutines can both miss the cache and both call `get_settings().API_KEY.get_secret_value()`, causing redundant (but harmless) settings reads.

**Evidence**:
```python
def _get_api_key(self) -> str:
    now = time.monotonic()
    cached = self._cached  # Atomic read
    if now - cached[1] > self._KEY_TTL:
        key = get_settings().API_KEY.get_secret_value()
        self._cached = (key, now)  # Atomic write
        return key
    return cached[0]
```

**Impact**: Harmless redundant reads on TTL expiry. The tuple assignment is atomic in CPython (GIL-protected), so no torn reads. Correct behavior.

**Revised Severity**: LOW (informational, no bug).

---

### F11: SSE `graph_node` Start Event May Fire Multiple Times Per Node (MEDIUM)

**Category**: API Design
**Severity**: MEDIUM (+1 from LOW)
**File**: `src/agent/graph.py:522-531`

**Description**: The `on_chain_start` filter uses `langgraph_node not in node_start_times` to prevent duplicate start events. However, LangGraph's `astream_events` v2 emits `on_chain_start` at multiple levels of the call stack (e.g., once for the node function, once for the LLM call inside the node). The `langgraph_node` metadata is the same for all events within that node's execution. The guard `langgraph_node not in node_start_times` prevents duplicate start events but also prevents updating `node_start_times` with the correct inner start time, which can cause inaccurate duration measurements.

**Evidence**:
```python
if (
    kind == "on_chain_start"
    and langgraph_node in _KNOWN_NODES
    and langgraph_node not in node_start_times  # First start only
):
    node_start_times[langgraph_node] = time.monotonic()
```

The duration is calculated at `on_chain_end`, but `on_chain_end` also fires multiple times per node (outer and inner). The `node_start_times.pop(langgraph_node)` on the first `on_chain_end` consumes the entry, so subsequent `on_chain_end` events for the same node are ignored. This means the reported duration covers the first start to the first end, which may not be the full node execution time if the inner chain ends before the outer one.

**Impact**: Node duration metrics in SSE `graph_node` events may underreport actual execution time. Monitoring dashboards show inaccurate latency. No functional impact on the agent's behavior.

**Fix**: Filter by a more specific event identifier if available in LangGraph v2 events, or document that durations are approximate.

---

## Dimension Scores

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | Graph Architecture | 7.5 | Clean 11-node topology, proper conditional routing, parity assertions. `retry_count` hard-coding (F3) and recursion limit interaction need attention. |
| 2 | RAG Pipeline | 7.0 | Per-item chunking, RRF reranking, SHA-256 dedup, version-stamp purging, relevance filtering. Single-retriever RRF is a no-op but harmless. |
| 3 | Data Model | 7.5 | TypedDict with reducers, `_keep_max` for escalation counters, RetrievedChunk schema. Parity assertion at module load is excellent. |
| 4 | API Design | 7.0 | Pure ASGI middleware, SSE streaming with PII buffer, structured error responses. `_PII_MAX_BUFFER` dead code (F2), duration metrics (F11). |
| 5 | Testing Strategy | 6.5 | 1256 tests, 90.67% coverage. No evidence of circuit breaker half-open + CancelledError test (F1). Integration test for full SSE stream lifecycle unclear. |
| 6 | Docker & DevOps | 6.5 | Exec-form CMD, Cloud Run config, health endpoints. Rolling window not configurable (F8). Version pinning in requirements. |
| 7 | Prompts & Guardrails | 7.5 | 5-layer deterministic guardrails, semantic injection classifier, PII redaction (fail-closed), per-stream PII buffering. Multilingual patterns (4 languages). |
| 8 | Scalability & Production | 6.0 | Circuit breaker stuck on CancelledError (F1, CRITICAL), `failure_count` property mutates without lock (F5), LRU vs TTL inconsistency (F4). Semaphore backpressure is good. |
| 9 | Trade-off Documentation | 7.5 | Extensive comments explaining design rationale, accepted trade-offs documented in docstrings, defense-in-depth reasoning articulated. |
| 10 | Domain Intelligence | 7.5 | Casino-specific guardrails (BSA/AML, responsible gaming escalation, patron privacy), comp threshold, multi-language support, regulatory awareness. |

**Total: 70.5 / 100** (simple average)

---

## Summary

| Metric | Value |
|--------|-------|
| Findings | 11 (1 CRITICAL, 2 HIGH, 5 MEDIUM, 3 LOW) |
| Score | 70.5 |
| Delta vs R9 | +3.4 (67.1 -> 70.5) |

### Critical Path

**F1 (Circuit Breaker Stuck)** is the only finding that causes total service outage in production. A single client disconnect during a half-open probe permanently blocks all LLM requests. The fix is a 2-line change (`await cb.record_failure()` before re-raise in the `CancelledError` handler).

### Consensus with R9

- Dead code removal (R9) significantly cleaned the codebase. No new dead code introduced.
- Guard patterns (parity assertions, feature flag drift checks) are production-grade.
- The DRY extraction in `_base.py` and `guardrails.py` is well-executed.

### Production Sign-Off Assessment

**CONDITIONAL PASS**: Fix F1 (CRITICAL) and F3 (HIGH) before deployment. F1 is a process-lifetime deadlock triggered by a common event (client disconnect). F3 causes doubled retry costs on parse errors. All other findings are operational improvements, not blockers.
