# Hey Seven R12 Hostile Review -- DeepSeek Focus

**Reviewer**: DeepSeek (simulated by Opus 4.6)
**Focus**: Async correctness, state machine integrity, concurrency bugs, algorithmic bounds, documentation accuracy
**Commit**: a36a5e6
**Date**: 2026-02-21
**Previous Score**: 73 (R11)

---

## Dimension Scores

| # | Dimension | Score | Justification |
|---|-----------|-------|---------------|
| 1 | Graph/Agent Architecture | 9 | Mature 11-node StateGraph with clean separation, specialist dispatch via structured output + keyword fallback, deterministic tie-breaking; minor: `_dispatch_to_specialist` calls `cb.record_success()` on dispatch LLM but not `record_failure()` symmetrically for all error paths. |
| 2 | RAG Pipeline | 8 | Per-item chunking, RRF reranking, SHA-256 idempotent IDs, version-stamp purging, cosine-space config; `_get_retriever_cached` uses bare dicts without lock protection (Finding 1). |
| 3 | Data Model / State Design | 9 | `_keep_max` reducer for responsible_gaming_count is elegant, parity check catches drift at import time; `PropertyQAState` field count (13) vs ARCHITECTURE.md claim (13) is accurate. |
| 4 | API Design | 8 | SSE streaming with heartbeat, graph trace events, proper CancelledError handling, 503 readiness vs 200 liveness; heartbeat logic has a subtle bug (Finding 4). |
| 5 | Testing Strategy | 8 | 1441 tests, VCR fixtures, parity assertion at import time, conftest singleton cleanup; no direct test of `_dispatch_to_specialist` structured LLM path visible in reviewed files. |
| 6 | Docker & DevOps | 9 | Multi-stage build, non-root user, exec-form CMD, Trivy scan, graceful shutdown; no findings in this round. |
| 7 | Prompts & Guardrails | 9 | 84 compiled regex patterns (not 73 as documented -- Finding 5), 4 languages, semantic classifier fail-closed, input normalization with NFKD + combining marks removal. |
| 8 | Scalability & Production | 8 | Circuit breaker with lock-protected transitions, TTL-cached singletons, LLM semaphore backpressure, BoundedMemorySaver LRU eviction; retriever cache not thread-safe (Finding 1). |
| 9 | Documentation & Code Quality | 7 | Comprehensive ARCHITECTURE.md and runbook; multiple documentation-code mismatches (Findings 5, 6, 7, 8 -- SPOTLIGHT +1 severity on all). |
| 10 | Domain Intelligence | 9 | Responsible gaming escalation counter across turns, BSA/AML guardrails with chip-walking patterns, patron privacy, 4-language coverage. |

**Total: 84/100** (up from 73 in R11)

---

## Findings

### Finding 1 (HIGH): Retriever TTL cache is not thread-safe under async concurrency

- **Location**: `src/rag/pipeline.py:856-934`
- **Problem**: `_retriever_cache` and `_retriever_cache_time` are plain `dict` objects accessed without any lock in `_get_retriever_cached()`. Under concurrent async requests during cache expiry, two coroutines can simultaneously detect TTL expiration and both create new retriever instances. While Python's GIL prevents dict corruption, the race creates redundant ChromaDB/Firestore connections and wastes memory. More critically, if `FirestoreRetriever.__init__()` has side effects (like establishing gRPC channels), concurrent initialization can produce orphaned connections.
- **Impact**: Memory leak from duplicate retriever instances under concurrent cache refresh; orphaned gRPC channels to Firestore in production.
- **Fix**: Add an `asyncio.Lock` and use `TTLCache` consistent with `_get_llm()`, `_get_validator_llm()`, and `_get_circuit_breaker()`:
```python
_retriever_cache: TTLCache = TTLCache(maxsize=1, ttl=3600)
_retriever_lock = asyncio.Lock()

async def _get_retriever_cached() -> AbstractRetriever:
    async with _retriever_lock:
        cached = _retriever_cache.get("retriever")
        if cached is not None:
            return cached
        # ... create retriever ...
        _retriever_cache["retriever"] = retriever
        return retriever
```
This requires changing `get_retriever()` and its callers (`retrieve_node`, `health` endpoint) to `await`, which is already the case since `retrieve_node` is async. The `health` endpoint calls `get_retriever()` synchronously -- would need `asyncio.to_thread()` wrapping or a separate sync path.

### Finding 2 (HIGH): `_get_circuit_breaker()` is synchronous but accesses settings lazily -- no lock protection

- **Location**: `src/agent/circuit_breaker.py:275-302`
- **Problem**: `_get_circuit_breaker()` is a synchronous function that reads and writes `_cb_cache` (a `TTLCache`) without any lock. Multiple coroutines can execute `_get_circuit_breaker()` concurrently during cache miss or TTL expiry, creating multiple `CircuitBreaker` instances. Only one gets cached (last write wins), so failure counts from the others are silently discarded. This is particularly insidious: during an LLM outage, the breaker may never trip because failures are spread across transient instances.
- **Impact**: Circuit breaker failure threshold may not trip during concurrent requests on cache miss -- cascading LLM failures are not protected.
- **Fix**: Either (a) make `_get_circuit_breaker()` async with an `asyncio.Lock`, or (b) initialize the singleton eagerly at module load (since `CircuitBreaker.__init__` is cheap and only reads settings). Option (a) is consistent with `_get_llm()`:
```python
_cb_lock = asyncio.Lock()

async def _get_circuit_breaker() -> CircuitBreaker:
    async with _cb_lock:
        cached = _cb_cache.get("cb")
        if cached is not None:
            return cached
        settings = get_settings()
        cb = CircuitBreaker(...)
        _cb_cache["cb"] = cb
        return cb
```
Note: This requires updating all callers from `_get_circuit_breaker()` to `await _get_circuit_breaker()` (in `graph.py:_dispatch_to_specialist`, `_base.py:execute_specialist`, `app.py:health`).

### Finding 3 (MEDIUM): `BoundedMemorySaver._track_thread()` is not async-safe -- mutates `OrderedDict` without lock

- **Location**: `src/agent/memory.py:55-75`
- **Problem**: `_track_thread()` is called from both sync (`get`, `put`, `put_writes`) and async (`aget`, `aput`, `aput_writes`) methods. It mutates `self._thread_order` (an `OrderedDict`) via `move_to_end()`, `__setitem__`, and `popitem()`. Under concurrent async requests (via `aget`/`aput`), interleaved mutations to the `OrderedDict` can cause: (1) incorrect eviction order, (2) `KeyError` if `popitem()` races with `move_to_end()`, or (3) iteration-during-mutation errors in the `while` loop.
- **Impact**: Thread eviction logic could crash or evict wrong threads under concurrent async graph execution, causing stale conversation state or KeyError exceptions.
- **Fix**: Add an `asyncio.Lock` to `BoundedMemorySaver.__init__()` and acquire it in async methods:
```python
def __init__(self, max_threads: int = MAX_ACTIVE_THREADS) -> None:
    ...
    self._lock = asyncio.Lock()

async def aget(self, config: dict) -> Any:
    async with self._lock:
        self._track_thread(config)
    return await self._inner.aget(config)
```

### Finding 4 (MEDIUM): SSE heartbeat timer never sends heartbeats during slow graph operations

- **Location**: `src/api/app.py:174-188`
- **Problem**: The heartbeat logic tracks `last_event_time` and checks `now - last_event_time >= _HEARTBEAT_INTERVAL` before yielding each event. However, heartbeats can only fire *between* events from `chat_stream()`. If `chat_stream()` itself blocks for 30+ seconds (e.g., during `retrieve` node hitting a slow Firestore query, or `validate` node waiting for LLM), no events are produced and the heartbeat check never runs. The browser's `EventSource` will timeout and reconnect, creating duplicate requests.
- **Impact**: Client-side EventSource timeouts during slow LLM/retrieval operations, causing duplicate requests and poor user experience.
- **Fix**: Use `asyncio.wait` with a heartbeat timer task running in parallel with the stream:
```python
async def event_generator():
    try:
        async with asyncio.timeout(sse_timeout):
            stream = chat_stream(agent, body.message, body.thread_id, request_id=request_id)
            async for event in _with_heartbeats(stream, interval=15):
                if await request.is_disconnected():
                    return
                yield event
    except TimeoutError:
        ...

async def _with_heartbeats(stream, interval=15):
    """Wrap an async generator with periodic heartbeats."""
    import asyncio
    stream_iter = stream.__aiter__()
    while True:
        try:
            event = await asyncio.wait_for(stream_iter.__anext__(), timeout=interval)
            yield event
        except TimeoutError:
            yield {"event": "ping", "data": ""}
        except StopAsyncIteration:
            return
```

### Finding 5 (MEDIUM -- SPOTLIGHT +1 -> HIGH): Guardrail pattern count mismatch: code has 84 patterns, docs claim 73

- **Location**: `src/agent/guardrails.py` (all pattern lists), `README.md:109,244`, `ARCHITECTURE.md:374,834`
- **Problem**: Grep for `re.compile` in `guardrails.py` returns 84 compiled patterns. The README and ARCHITECTURE.md consistently claim "73 regex patterns across 4 languages" and break them down as 11 + 31 + 6 + 14 + 11 = 73. However:
  - `_INJECTION_PATTERNS`: 11 patterns (correct)
  - `_RESPONSIBLE_GAMING_PATTERNS`: counting the list reveals 31 entries (correct per docs)
  - `_AGE_VERIFICATION_PATTERNS`: 6 (correct)
  - `_BSA_AML_PATTERNS`: The list has grown to 30 patterns (14 English + 5 Spanish + 3 Portuguese + 2 Mandarin + additional patterns), not 14 as documented
  - `_PATRON_PRIVACY_PATTERNS`: 14 patterns, not 11 as documented

  The documented breakdown (11+31+6+14+11=73) is stale. Actual: 11+31+6+30+14 = 92 (approximately -- exact count varies by how the list is segmented after additions). The `re.compile` count of 84 includes patterns from `_normalize_input()` inline regexes, but the point stands: the documented pattern count is wrong.
- **Impact**: Documentation underclaims security coverage; a reader assessing security posture would undercount guardrail coverage by ~15%.
- **Fix**: Recount patterns in each list and update README.md, ARCHITECTURE.md, and compliance_gate.py docstring. Add a parity assertion (like the state schema parity check) at module load:
```python
_TOTAL_PATTERNS = (len(_INJECTION_PATTERNS) + len(_RESPONSIBLE_GAMING_PATTERNS) +
                   len(_AGE_VERIFICATION_PATTERNS) + len(_BSA_AML_PATTERNS) +
                   len(_PATRON_PRIVACY_PATTERNS))
# Update this number when patterns are added/removed:
assert _TOTAL_PATTERNS == 84, f"Guardrail pattern count drifted: expected 84, got {_TOTAL_PATTERNS}"
```

### Finding 6 (MEDIUM -- SPOTLIGHT +1 -> HIGH): ARCHITECTURE.md claims CSP uses `unsafe-inline` but code uses nonce-based CSP

- **Location**: `ARCHITECTURE.md:648-650` (Trade-offs > CSP section), `src/api/middleware.py:202-209`
- **Problem**: The ARCHITECTURE.md Trade-offs section says:
  > The security headers middleware uses `script-src 'self' 'unsafe-inline'` and `style-src 'self' 'unsafe-inline'`

  And labels the "Production path" as replacing `unsafe-inline` with nonce-based CSP. However, the actual code in `SecurityHeadersMiddleware` already generates per-request nonces via `secrets.token_bytes(16)` and uses `script-src 'self' 'nonce-{nonce}'` and `style-src 'self' 'nonce-{nonce}'`. The documentation describes a vulnerability that was already fixed -- a reader would believe the app ships with `unsafe-inline` when it actually has nonce-based CSP.
- **Impact**: Security audit would flag a false positive. Investors/compliance reviewers reading ARCHITECTURE.md would believe the app has weaker CSP than it actually does.
- **Fix**: Update the Trade-offs section to:
```markdown
### CSP (Nonce-Based)

The security headers middleware generates per-request cryptographic nonces (16 bytes, `secrets.token_bytes`) for `script-src` and `style-src`. CSS and JS are externalized into separate static files (`styles.css`, `app.js`), eliminating the need for `unsafe-inline`.
```

### Finding 7 (LOW -- SPOTLIGHT +1 -> MEDIUM): ARCHITECTURE.md claims "Both are `@lru_cache` singletons" for LLM instances

- **Location**: `ARCHITECTURE.md:210`
- **Problem**: The validate node section in ARCHITECTURE.md says:
  > Both are `@lru_cache` singletons.

  But the code uses `TTLCache` (cachetools) with 1-hour TTL, not `@lru_cache`. This was changed in a previous round for credential rotation support. The documentation was not updated.
- **Impact**: A reader implementing a similar pattern would use `@lru_cache` (which never expires) instead of `TTLCache`, breaking credential rotation.
- **Fix**: Replace with "Both are TTL-cached singletons (1-hour TTL via `cachetools.TTLCache`) for GCP credential rotation support."

### Finding 8 (LOW -- SPOTLIGHT +1 -> MEDIUM): ARCHITECTURE.md dispatch tie-breaking description is inaccurate

- **Location**: `ARCHITECTURE.md:178`
- **Problem**: ARCHITECTURE.md says:
  > `max(category_counts, key=lambda k: (count, k))` selects alphabetically

  But the actual code in `graph.py:161-164` uses:
  ```python
  max(category_counts, key=lambda k: (category_counts[k], _CATEGORY_PRIORITY.get(k, 0), k))
  ```
  The real implementation has a **three-way** tie-break: count -> business priority -> alphabetical. The documentation describes a two-way tie-break and omits the business priority layer entirely.
- **Impact**: A reader would not understand that dining queries are prioritized over hotel queries when chunk counts are tied -- which is a deliberate business decision.
- **Fix**: Update to: "Dispatch tie-breaking uses a three-tuple key: `(count, business_priority, alphabetical)`. Business priority: dining(4) > hotel(3) > entertainment(2) > comp(1). See `_CATEGORY_PRIORITY` in `graph.py`."

### Finding 9 (LOW): `_request_counter` initialized via `getattr` instead of `__init__`

- **Location**: `src/api/middleware.py:369`
- **Problem**: `RateLimitMiddleware._is_allowed()` uses `self._request_counter = getattr(self, "_request_counter", 0) + 1` to lazily initialize a counter. This is fragile: (a) it circumvents `__init__` and makes the attribute invisible to type checkers, (b) the `getattr` call runs on every request (O(1) but unnecessary overhead), (c) it breaks the principle of declaring all instance attributes in `__init__`.
- **Impact**: Minor: type checker cannot verify the attribute exists; slightly confusing for code readers.
- **Fix**: Initialize `self._request_counter = 0` in `__init__` and use `self._request_counter += 1` directly.

### Finding 10 (LOW): `streaming_pii.py` re-scans already-scanned text on every feed() call

- **Location**: `src/agent/streaming_pii.py:104,116`
- **Problem**: `_scan_and_release()` applies `redact_pii()` to the **entire** buffer on every `feed()` call. The retained lookahead window (last `_MAX_PATTERN_LEN` chars of the original buffer) is re-scanned on the next call because it remains in `self._buffer`. For a typical 500-token response (arriving in ~100 chunks), the lookahead window is scanned ~100 times. While regex matching is fast, this is O(n*k) where n is chunk count and k is pattern count (7 standard + 2 name patterns).
- **Impact**: Performance: ~10x more regex evaluations than necessary for long streaming responses. Not a correctness issue.
- **Fix**: Track a `_scanned_up_to` offset and only apply `redact_pii()` to the new portion plus the lookahead overlap, rather than re-scanning the entire buffer. This is a micro-optimization -- acceptable to defer.

### Finding 11 (LOW): `InMemoryBackend` sweep uses `random.random()` which is not deterministic in tests

- **Location**: `src/state_backend.py:75`
- **Problem**: `_maybe_sweep()` uses `random.random() > self._SWEEP_PROBABILITY` to probabilistically trigger sweeps. In tests, this makes behavior non-deterministic: a test that creates 99 entries might or might not trigger a sweep depending on the random seed. This can cause flaky test failures if a test asserts exact store size after a series of writes.
- **Impact**: Potential test flakiness if any test depends on exact `InMemoryBackend` store size.
- **Fix**: Use a deterministic counter-based trigger in tests (e.g., sweep every 100th write) and make `_SWEEP_PROBABILITY` a constructor parameter:
```python
def __init__(self, sweep_probability: float = 0.01) -> None:
    self._sweep_probability = sweep_probability
```
Tests can pass `sweep_probability=0.0` to disable probabilistic sweeps.

---

## Documented Design Decisions Verified (Not Flagged)

1. **Degraded-pass validator**: First-attempt PASS on validator error is correct. Code matches documentation. Deterministic guardrails run upstream.
2. **Single-container deployment**: Accepted. Redis abstraction layer present in `state_backend.py`.
3. **Feature flag dual-layer**: Build-time topology (whisper planner) + runtime behavior (specialist dispatch, AI disclosure). Code matches extensive graph.py comments.
4. **Streaming PII defense-in-depth**: `StreamingPIIRedactor` for token events + `contains_pii/redact_pii` for replace events. CancelledError correctly drops buffered tokens (fail-safe).

---

## Documentation Accuracy Audit (SPOTLIGHT)

| Claim | Location | Status | Notes |
|-------|----------|--------|-------|
| "73 regex patterns" | README:109, ARCH:374 | WRONG | Actual count is ~84 (Finding 5) |
| "CSP unsafe-inline" | ARCH:648 | WRONG | Code uses nonce-based CSP (Finding 6) |
| "@lru_cache singletons" for LLMs | ARCH:210 | WRONG | Code uses TTLCache (Finding 7) |
| "max(category_counts, key=lambda k: (count, k))" | ARCH:178 | WRONG | Three-way tie-break with priority (Finding 8) |
| "13 fields" in PropertyQAState | ARCH:268 | CORRECT | 13 fields confirmed in state.py |
| "11 nodes, 3 conditional routing points" | README:42 | CORRECT | Verified in graph.py |
| "6 pure ASGI middleware" | README:188 | CORRECT | 6 classes confirmed in middleware.py |
| "Exec-form CMD" | Dockerfile:68 | CORRECT | `CMD ["python", "-m", ...]` |
| "Non-root user" | Dockerfile:20,53 | CORRECT | `appuser` created and switched to |
| "Per-request nonce" | middleware.py:202 | CORRECT | `secrets.token_bytes(16)` per request |
| "hmac.compare_digest" for API key | middleware.py:272 | CORRECT | Timing-safe comparison |
| "TTL=3600 for LLM singletons" | nodes.py:101 | CORRECT | `_LLM_CACHE_TTL = 3600` |
| "asyncio.Lock for circuit breaker" | circuit_breaker.py:63 | CORRECT | `self._lock = asyncio.Lock()` |
| "max 1 retry before fallback" | nodes.py:324 | CORRECT | `retry_count < 1` |
| "Fail-closed PII redaction" | pii_redaction.py:106-108 | CORRECT | Returns `[PII_REDACTION_ERROR]` on exception |
| "GRAPH_RECURSION_LIMIT=10" | config.py:57 | CORRECT | Matches runbook |
| "Version-stamp purging" | pipeline.py:659-721 | CORRECT | `_ingestion_version` purging implemented |
| "5-step CI/CD pipeline" | ARCH:752 | WRONG | cloudbuild.yaml has 8 steps per runbook |

---

## Circuit Breaker State Machine Verification

Verified all transitions under lock:

| From | Trigger | To | Lock? | Correct? |
|------|---------|-----|-------|----------|
| closed | `record_failure()` >= threshold | open | Yes (`self._lock`) | CORRECT |
| open | cooldown expired + `allow_request()` | half_open | Yes (`self._lock`) | CORRECT |
| half_open | `allow_request()` (one probe) | half_open (in_progress=True) | Yes | CORRECT |
| half_open | `record_success()` | closed | Yes | CORRECT |
| half_open | `record_failure()` | open | Yes | CORRECT |
| half_open | `record_cancellation()` | half_open (in_progress=False) | Yes | CORRECT (R11 fix) |
| any | `state` property read | no mutation | No lock | CORRECT (documented as approximate) |

No stuck states found. The `_half_open_in_progress` flag correctly prevents multiple probes and is reset on both success and cancellation.

---

## Concurrency Analysis

| Component | Mechanism | Thread-Safe? | Async-Safe? |
|-----------|-----------|-------------|-------------|
| `_get_llm()` | `asyncio.Lock` + `TTLCache` | N/A (async only) | YES |
| `_get_validator_llm()` | `asyncio.Lock` + `TTLCache` | N/A | YES |
| `CircuitBreaker` | `asyncio.Lock` | N/A | YES |
| `_get_circuit_breaker()` | `TTLCache` (NO lock) | N/A | **NO** (Finding 2) |
| `_get_retriever_cached()` | `dict` (NO lock) | N/A | **NO** (Finding 1) |
| `RateLimitMiddleware._is_allowed()` | `asyncio.Lock` | N/A | YES |
| `BoundedMemorySaver._track_thread()` | No lock | N/A | **NO** (Finding 3) |
| `ApiKeyMiddleware._get_api_key()` | Atomic tuple write | N/A | YES (acceptable) |
| `InMemoryBackend` | No lock (single-threaded assumption) | N/A | Acceptable for single-worker |
| `_LLM_SEMAPHORE` | `asyncio.Semaphore(20)` | N/A | YES |

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH | 3 (Finding 1, Finding 2, Finding 5/spotlight) |
| MEDIUM | 4 (Finding 3, Finding 4, Finding 6/spotlight, Finding 7/spotlight, Finding 8/spotlight) |
| LOW | 3 (Finding 9, Finding 10, Finding 11) |
| **Total** | **11** |

**Score: 84/100** (+11 from R11's 73)

The codebase has matured significantly since R11. The circuit breaker state machine is now correct with proper lock protection and CancelledError handling. The streaming PII redactor is a genuine defense-in-depth addition. The primary remaining issues are: (1) inconsistent lock protection across singleton factories -- `_get_llm` and `_get_validator_llm` are properly locked but `_get_circuit_breaker` and `_get_retriever_cached` are not, creating a concurrency hazard under load; (2) documentation drift where the code has been improved (nonce CSP, TTLCache, expanded guardrail patterns) but the docs still describe the old state. The spotlight dimension reveals 4 documentation-code mismatches that should be addressed before the next review round.
