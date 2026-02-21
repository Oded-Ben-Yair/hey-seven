# Production Code Review: R15 (DeepSeek Focus)

**Reviewer**: DeepSeek (simulated by Opus 4.6)
**Focus**: Async correctness, state machine integrity, concurrency bugs, algorithmic bounds
**Repo**: GitHub Oded-Ben-Yair/hey-seven
**Commit**: c7b986e
**Date**: 2026-02-21
**Spotlight**: Graph Architecture (+1 severity)

---

## Executive Summary

The codebase has matured substantially through 14 prior review rounds. The 11-node StateGraph architecture is sound, the validation loop is correctly bounded, and the specialist dispatch pattern with DRY extraction is exemplary. This R15 review focuses on async correctness, state machine edge cases, and concurrency subtleties that survive into a high-maturity codebase. I identify 8 findings, predominantly at the MEDIUM level, reflecting the narrowing gap between current quality and production perfection.

---

## Findings

### F-001: `_dispatch_to_specialist` uses `cb` from outer scope after possible reassignment [MEDIUM -> HIGH with spotlight]

**File**: `src/agent/graph.py`, lines 204-247
**Category**: Graph Architecture (spotlight +1)

In `_dispatch_to_specialist`, the circuit breaker `cb` is obtained at line 204 inside the `try` block. If `await cb.allow_request()` returns `True` but the LLM call then raises an exception caught at line 237 (`ValueError`/`TypeError`), `cb.record_failure()` is called. However, the `cb` variable is scoped inside the `try` block. If `_get_circuit_breaker()` itself raises (e.g., settings lookup fails during TTL cache refresh), the `except` blocks at lines 237-247 reference `cb` which would be an `UnboundLocalError`.

```python
try:
    cb = _get_circuit_breaker()  # line 204 -- can throw
    if await cb.allow_request():
        ...
except (ValueError, TypeError) as exc:
    await cb.record_failure()  # line 240 -- cb may be unbound
```

**Impact**: If `_get_circuit_breaker()` raises during a TTL cache refresh (e.g., settings validation error), the `except` handler crashes with `UnboundLocalError`, converting a recoverable config error into an unhandled exception that aborts the entire graph invocation. The keyword fallback path is never reached.

**Fix**: Move `cb = _get_circuit_breaker()` before the `try` block, or add a guard: `if 'cb' in dir(): await cb.record_failure()`. Better: extract `cb` assignment to the function entry point.

---

### F-002: `_get_circuit_breaker()` is synchronous but accesses TTLCache without thread safety [MEDIUM -> HIGH with spotlight]

**File**: `src/agent/circuit_breaker.py`, lines 275-302
**Category**: Graph Architecture (spotlight +1)

`_get_circuit_breaker()` is a synchronous function that reads and writes `_cb_cache` (a `TTLCache`). Unlike `_get_llm()` and `_get_validator_llm()` which use `asyncio.Lock` for coroutine safety, the CB factory has no lock protection. In a concurrent async environment, two coroutines can simultaneously find the cache empty and create two separate `CircuitBreaker` instances. The second one silently replaces the first, causing the first caller's reference to become an orphan that no longer represents the shared state.

```python
def _get_circuit_breaker() -> CircuitBreaker:
    cached = _cb_cache.get("cb")  # No lock protection
    if cached is not None:
        return cached
    settings = get_settings()
    cb = CircuitBreaker(...)
    _cb_cache["cb"] = cb  # Race: two coroutines can both set this
    return cb
```

**Impact**: Under concurrent load during a TTL cache expiry, two coroutines create separate `CircuitBreaker` instances. One tracks failures, the other does not. The failure threshold may not be reached because failures are split across orphaned instances. The circuit breaker fails to trip during a genuine LLM outage.

**Fix**: Either make `_get_circuit_breaker()` async with an `asyncio.Lock` (matching the `_get_llm()` pattern), or accept the race and document it. The practical impact is limited because TTL cache misses are rare (once per hour), but the pattern is inconsistent with the rest of the codebase.

---

### F-003: Whisper planner `_failure_count` global variable has documented race but no mitigation [LOW]

**File**: `src/agent/whisper_planner.py`, lines 87-89, 132, 168-178
**Category**: Concurrency

The `_failure_count` and `_failure_alerted` globals are read-modify-write without any lock. The docstring acknowledges the race (line 84: "Benign race") and claims it is acceptable for alerting. This was already documented in R10 (DeepSeek F9, Gemini F19).

However, the race has a subtler consequence than "off-by-one delays": the `_failure_count = 0` reset at line 168 can race with an increment at line 177 in a concurrent request, causing the counter to oscillate between 0 and 1 under sustained mixed success/failure conditions. The alert threshold of 10 may never be reached even during a genuine sustained outage if successful requests periodically reset the counter.

**Impact**: Under mixed traffic (some queries succeed, some fail), the failure alert may never fire despite a persistent ~50% failure rate. The "systematic failure" detection is unreliable for partial outages.

**Fix**: Use `asyncio.Lock` around the counter operations, or switch to a lock-free monotonic counter (only increment, never reset) with a sliding window approach matching the circuit breaker pattern.

---

### F-004: `route_from_router` returns bare strings instead of NODE_* constants [LOW]

**File**: `src/agent/nodes.py`, lines 625-652
**Category**: Graph Architecture

`route_from_router` returns hardcoded strings `"greeting"`, `"off_topic"`, and `"retrieve"` instead of using the `NODE_GREETING`, `NODE_OFF_TOPIC`, `NODE_RETRIEVE` constants defined in `graph.py`. While `route_from_compliance` in `graph.py` (lines 279-307) similarly returns `NODE_ROUTER`, `NODE_GREETING`, and `NODE_OFF_TOPIC` constants, `route_from_router` in `nodes.py` does not import or use these constants.

```python
# nodes.py line 641-652
if query_type == "greeting":
    return "greeting"        # Should be NODE_GREETING
if query_type in (...):
    return "off_topic"       # Should be NODE_OFF_TOPIC
return "retrieve"            # Should be NODE_RETRIEVE
```

**Impact**: If a node is renamed (e.g., `"off_topic"` -> `"off_topic_handler"`), `route_from_compliance` in `graph.py` would be updated via the constant, but `route_from_router` in `nodes.py` would silently break. This is a maintenance hazard. It works today because the strings match, but the inconsistency with the established NODE_* constant pattern is a latent defect.

**Fix**: Import and use NODE_* constants from graph.py, or define routing constants in a shared location (e.g., `state.py` or a dedicated `constants.py`). Be careful of circular imports -- the constants should live in a module that both `graph.py` and `nodes.py` can import without cycles.

---

### F-005: `BoundedMemorySaver._track_thread` accesses internal `MemorySaver.storage` attribute [MEDIUM]

**File**: `src/agent/memory.py`, lines 69-74
**Category**: API Coupling

The LRU eviction logic accesses `self._inner.storage`, which is an undocumented internal attribute of `MemorySaver`. LangGraph's `MemorySaver` does not guarantee this attribute's existence across versions.

```python
if hasattr(self._inner, "storage"):
    keys_to_remove = [
        k for k in self._inner.storage if isinstance(k, tuple) and len(k) > 0 and k[0] == evicted_id
    ]
```

The `hasattr` guard prevents a crash if the attribute is removed, but it silently degrades eviction to a no-op: the thread is removed from the tracking dict but its data remains in MemorySaver's storage, defeating the OOM protection purpose of BoundedMemorySaver.

**Impact**: After a LangGraph version upgrade that changes MemorySaver internals, eviction silently stops freeing memory. In a long-running development session, this leads to the same OOM condition that BoundedMemorySaver was designed to prevent.

**Fix**: Add a unit test that asserts eviction actually reduces memory/storage size. If `storage` disappears after a LangGraph upgrade, the test fails immediately rather than silently degrading. Alternatively, wrap MemorySaver with a dict-based storage that you control.

---

### F-006: Comp agent profile completeness calculation is 0/1 = 0.0 when `extracted_fields` is empty [LOW]

**File**: `src/agent/agents/comp_agent.py`, lines 82-85
**Category**: Algorithmic Bounds

```python
extracted_fields = state.get("extracted_fields", {})
filled = sum(1 for v in extracted_fields.values() if v is not None and v != "")
total = max(len(extracted_fields), 1)  # Guard against div-by-zero
completeness = filled / total           # Always 0.0 when dict is empty
```

When `extracted_fields` is `{}` (the default from `_initial_state`), `total = max(0, 1) = 1` and `filled = 0`, so `completeness = 0.0`. This is below `COMP_COMPLETENESS_THRESHOLD` (0.60), so the comp agent always returns the "tell me more" prompt on the first turn.

This is likely intentional design (gather profile before offering comps), but the `total = max(len(extracted_fields), 1)` formula has a semantic problem: once fields start being extracted, the denominator grows with the number of keys, not the number of expected keys. If only 1 of 8 expected fields is extracted, `completeness = 1/1 = 1.0`, which incorrectly indicates full completion.

**Impact**: A single extracted field gives 100% completeness, bypassing the profile gate entirely. The whisper planner's `_calculate_completeness` uses `_PROFILE_FIELDS` (8 fixed fields) as the denominator, creating inconsistency between the two completeness calculations.

**Fix**: Use the same `_PROFILE_FIELDS` tuple from `whisper_planner.py` for the denominator in comp agent, or call `_calculate_completeness` directly. This ensures consistent behavior regardless of how many fields have been extracted.

---

### F-007: `chat_stream` does not propagate `CancelledError` from PII flush path [LOW]

**File**: `src/agent/graph.py`, lines 722-727
**Category**: Async Correctness

The PII redactor flush at lines 722-727 runs outside the `try/except` block. If `_pii_redactor.flush()` raises an exception (unlikely but possible if `redact_pii` has a regex catastrophic backtracking), the error would propagate as an unhandled exception from the generator, potentially crashing the SSE response without emitting a `done` event.

```python
# After the try/except block:
if not errored:
    for safe_chunk in _pii_redactor.flush():
        yield {
            "event": "token",
            "data": json.dumps({"content": safe_chunk}),
        }
# This yield always runs:
yield {"event": "done", ...}
```

The `done` event yield at line 737 is not inside a `finally` block, so if the flush raises, the client never receives a `done` event and may hang waiting for stream completion.

**Impact**: Extremely unlikely in practice (regex on short buffer), but the `done` event is not guaranteed to be sent in all code paths. A client-side timeout would eventually recover.

**Fix**: Wrap the flush + done yields in a `try/finally` block to guarantee the `done` event.

---

### F-008: `feature_flags.py` uses `assert` for parity checks instead of `ValueError` [MEDIUM]

**File**: `src/casino/feature_flags.py`, lines 73-77, 83-87
**Category**: Production Safety

The parity assertions between `FeatureFlags` TypedDict, `DEFAULT_FEATURES`, and `DEFAULT_CONFIG["features"]` use Python `assert` statements. These are stripped when Python runs with `-O` (optimize) flag.

```python
assert set(FeatureFlags.__annotations__) == set(DEFAULT_FEATURES.keys()), (...)
assert set(_DEFAULT_CONFIG["features"].keys()) == set(DEFAULT_FEATURES.keys()), (...)
```

Compare with `graph.py` lines 498-505, which was explicitly converted from `assert` to `raise ValueError` in R10 (Gemini F10, GPT P2-F5) for exactly this reason: "converted from `assert` (vanishes with `python -O`) to a runtime ValueError that fires regardless of optimization mode."

**Impact**: If the production Docker image uses `python -O` (not uncommon for performance), the parity assertions silently disappear. Schema drift between `FeatureFlags`, `DEFAULT_FEATURES`, and `DEFAULT_CONFIG` would go undetected until a runtime `KeyError` in a feature flag lookup.

**Fix**: Convert both `assert` statements to `if/raise ValueError` following the pattern established in `graph.py` line 500.

---

## Scoring

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | Graph/Agent Architecture (SPOTLIGHT) | 8 | Topology is sound. Validation loop correctly bounded. F-001 (unbound CB variable) and F-002 (sync CB factory race) are the remaining concerns. Specialist dispatch with LLM+keyword fallback is well-designed. |
| 2 | RAG Pipeline | 9 | Not deeply reviewed (not in scope files), but retriever integration in nodes.py is clean. Timeout guards, async wrapping of sync ChromaDB, and error handling are thorough. |
| 3 | Data Model | 9 | State schema is well-typed. `_keep_max` reducer for responsible_gaming_count is elegant. Parity check at import time prevents drift. F-006 (comp completeness denominator) is a minor inconsistency. |
| 4 | API Design | 9 | Clean SSE streaming, heartbeat pattern, proper separation of liveness and readiness probes. PII redaction defense-in-depth. CORSMiddleware + 6 pure ASGI middleware layers. |
| 5 | Testing Strategy | 8 | 1452 tests at 90%+ coverage is strong. No test code in review scope, but the patterns support testing (DI for LLM/CB, singleton cleanup). F-005 (BoundedMemorySaver storage access) needs a regression test. |
| 6 | Docker & DevOps | 8 | Lifespan pattern is correct. Startup ingestion guard for Firestore is good. Health endpoint correctly separates liveness from readiness. |
| 7 | Prompts & Guardrails | 9 | 5-layer deterministic guardrails with correct ordering rationale. Semantic injection fail-closed. 84 compiled regex patterns across 4 languages. Input normalization for homoglyph attacks. |
| 8 | Scalability & Production | 8 | TTL-cached singletons, circuit breaker, semaphore backpressure, BoundedMemorySaver. F-002 (CB factory race) and F-003 (whisper counter race) are the gaps. F-008 (assert in production) is a deployment hazard. |
| 9 | Trade-off Documentation | 9 | Extensive inline documentation. Dual-layer feature flag architecture is thoroughly documented. Every design decision has a rationale comment with round-of-origin tracking. |
| 10 | Domain Intelligence | 9 | Casino-specific patterns (BSA/AML, patron privacy, responsible gaming escalation, comp gate, VIP tone) demonstrate deep domain understanding. Multi-language guardrails (EN/ES/PT/ZH) for diverse clientele. |

**Overall Score: 86/100**

---

## Severity Summary

| Severity | Count | Finding IDs |
|----------|-------|-------------|
| CRITICAL | 0 | -- |
| HIGH | 2 | F-001, F-002 (both elevated by spotlight) |
| MEDIUM | 2 | F-005, F-008 |
| LOW | 4 | F-003, F-004, F-006, F-007 |

---

## Comparison with R14

| Metric | R14 | R15 | Delta |
|--------|-----|-----|-------|
| Score | 86 | 86 | 0 |
| Findings | -- | 8 | -- |
| Critical | -- | 0 | -- |

The score holds steady at 86. The codebase is at the plateau where remaining issues are concurrency edge cases and pattern inconsistencies rather than architectural or safety-critical defects. The two HIGH findings (F-001 and F-002) are elevated by the spotlight modifier; without it they would be MEDIUM.

---

## Recommendations (Priority Order)

1. **F-001**: Move `cb = _get_circuit_breaker()` before the try block in `_dispatch_to_specialist`. Simplest fix, highest impact.
2. **F-008**: Convert feature_flags.py `assert` to `raise ValueError` to match the established graph.py pattern.
3. **F-002**: Add `asyncio.Lock` to `_get_circuit_breaker()` for consistency with `_get_llm()` pattern.
4. **F-006**: Align comp agent completeness denominator with whisper planner's `_PROFILE_FIELDS`.
5. **F-004**: Extract NODE_* constants to a shared module to eliminate string duplication risk.

---

*Review generated by DeepSeek focus (simulated by Opus 4.6) -- R15 hostile review protocol.*
