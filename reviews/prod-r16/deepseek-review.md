# Hey Seven Production Review - Round 16 (DeepSeek Focus)
## Reviewer: Claude Opus 4.6 (DeepSeek Persona)
## Focus: Async correctness, state machine integrity, concurrency bugs, algorithmic bounds
## Commit: a0bb225 | Date: 2026-02-21

---

## Scoring Summary

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | Graph Architecture | 9/10 | Excellent 11-node topology with validation loop. One state machine edge case found. |
| 2 | RAG Pipeline | 8/10 | Solid per-item chunking, RRF, idempotent ingestion. Retriever cache not thread-safe. |
| 3 | Data Model | 9/10 | Clean TypedDict + Pydantic models. _keep_max reducer is elegant. Parity checks strong. |
| 4 | API Design | 8/10 | Pure ASGI middleware, good SSE heartbeat. RequestBodyLimit race condition. |
| 5 | Testing Strategy | 8/10 | 1452 tests, 90%+ coverage. Cannot verify E2E coverage from source alone. |
| 6 | Docker & DevOps | 8/10 | Cloud Run config well-documented. Production guards present. |
| 7 | Prompts & Guardrails | 9/10 | 5-layer deterministic guardrails, 4-language patterns, proper priority chain. |
| 8 | Scalability & Production | 8/10 | TTL singletons, semaphore backpressure, bounded memory. Whisper global race. |
| 9 | Trade-off Documentation | 9/10 | Exceptional inline documentation with origin citations. Decision trail clear. |
| 10 | Domain Intelligence | 9/10 | BSA/AML, TCPA, patron privacy, RG escalation all present and well-reasoned. |

**Overall Score: 85/100**

Previous trajectory: 73 -> 84 -> 85 -> 86 -> 86 -> **85** (slight regression from new findings)

---

## Findings

### F-001 [HIGH] Retriever TTL cache dict is not thread-safe under concurrent async access

**File**: `src/rag/pipeline.py:894-972`

**Bug**: The retriever singleton uses plain `dict` (`_retriever_cache` and `_retriever_cache_time`) for TTL caching, with no `asyncio.Lock` protection. Every other TTL-cached singleton in this codebase (`_get_llm`, `_get_validator_llm`, `_get_whisper_llm`, `_get_circuit_breaker`, `get_checkpointer`, `get_embeddings`) uses an `asyncio.Lock` to prevent concurrent coroutines from creating duplicate instances on TTL expiry.

The retriever is accessed via `asyncio.to_thread(search_knowledge_base, query)` from `retrieve_node`, which means it runs in a thread pool. Multiple concurrent requests can hit TTL expiry simultaneously, each creating a separate `CasinoKnowledgeRetriever` or `FirestoreRetriever` instance. For ChromaDB this creates multiple SQLite connections to the same file; for Firestore this creates redundant gRPC channels.

More critically, `_get_retriever_cached()` does a non-atomic read-then-write sequence:
```python
if cache_key in _retriever_cache:
    if (now - _retriever_cache_time.get(cache_key, 0)) < _RETRIEVER_TTL_SECONDS:
        return _retriever_cache[cache_key]
# ... create new retriever ...
_retriever_cache[cache_key] = retriever  # concurrent write possible
_retriever_cache_time[cache_key] = now
```

Two threads checking simultaneously can both see the TTL as expired and both write, with one's retriever silently discarded while its Firestore client remains open (resource leak).

**Fix**: Add a `threading.Lock` (not `asyncio.Lock` -- this runs in `to_thread`) around `_get_retriever_cached()`, or use `cachetools.TTLCache` with a lock, consistent with every other singleton in the project.

**Impact**: Resource leak under concurrent load; potential SQLite corruption with ChromaDB; inconsistent Firestore client state.

---

### F-002 [HIGH] `_dispatch_to_specialist` calls `cb.allow_request()` but the keyword fallback path never calls `record_success()`, permanently starving the half-open probe

**File**: `src/agent/graph.py:184-276`

**Bug**: In `_dispatch_to_specialist()`, when the circuit breaker allows a request (`await cb.allow_request()` returns True at line 207), the half-open probe flag `_half_open_in_progress` is set to True. If the LLM dispatch call succeeds, `record_success()` is called (line 225), which resets the flag.

However, if `agent_name` is still `None` after the try/except block (lines 254-255), the function falls through to the keyword fallback path and calls `agent_fn(state)` (line 276) without ever calling `cb.record_success()` for the dispatch LLM call. Specifically:

1. CB is half-open, `allow_request()` returns True, sets `_half_open_in_progress = True`
2. The structured output LLM call succeeds but parsing fails (`ValueError`/`TypeError` at line 239)
3. Per the R15 fix, parse errors do NOT record CB failure (correct)
4. But they also don't record success
5. `agent_name` is still `None`, keyword fallback is used
6. The actual specialist agent call at line 276 may succeed or fail -- but its CB interaction is inside `execute_specialist` which uses a SEPARATE `cb.allow_request()` call
7. The dispatch LLM's half-open probe is never resolved

The `_half_open_in_progress` flag remains True. Per `allow_request()` logic (line 186-188), subsequent half-open probes are blocked: `if self._state == "half_open" and not self._half_open_in_progress` -- this is False, so it returns False. The dispatch CB is stuck until the 1-hour TTL evicts it.

**Wait** -- the dispatch LLM and the specialist agent share the same circuit breaker singleton (both call `_get_circuit_breaker()`). So the specialist's `execute_specialist` calls `cb.allow_request()` again on the SAME CB instance that already has `_half_open_in_progress = True`. This second `allow_request()` returns False (line 188), causing the specialist to immediately return a fallback message.

**Sequence under half-open + parse error**:
1. Dispatch: `allow_request()` -> True, `_half_open_in_progress = True`
2. Dispatch LLM: parse error (ValueError) -> no record_success, no record_failure
3. Keyword fallback: agent_name = "dining"
4. `execute_specialist` -> `cb.allow_request()` -> False (half_open + in_progress=True)
5. Returns fallback message: "temporary technical difficulties"
6. CB stuck in half-open with probe flag set forever

This is a production bug: after any CB trip + recovery where the dispatch LLM returns bad JSON, the system gets stuck returning fallback messages for up to 1 hour.

**Fix**: After the try/except block in `_dispatch_to_specialist`, if the CB allowed the request (was half-open or closed) but no `record_success`/`record_failure` was called, explicitly call `record_success()` -- because reaching the keyword fallback means the LLM was reachable (it returned a response, just unparseable). Or: separate the dispatch CB from the specialist CB.

**Impact**: System-wide degradation to fallback messages for up to 1 hour after a CB trip + recovery where the dispatch LLM returns unparseable JSON.

---

### F-003 [MEDIUM] `whisper_planner_node` global `_failure_count` has a benign-documented but actually harmful race

**File**: `src/agent/whisper_planner.py:87-89, 132, 167-179`

**Bug**: The module documents the race on `_failure_count` as "benign" (line 84): "Benign race: concurrent graph invocations (different thread_ids) may increment simultaneously, losing counts."

This is not fully benign. The reset on success (line 168: `_failure_count = 0`) and the alert guard (line 169: `_failure_alerted = False`) create a worse pattern:

1. Thread A enters except, reads `_failure_count` = 9
2. Thread B succeeds, sets `_failure_count = 0`, `_failure_alerted = False`
3. Thread A writes `_failure_count = 10`, checks `>= _FAILURE_ALERT_THRESHOLD` and `not _failure_alerted`
4. Alert fires -- but 9 of those 10 failures may have been interleaved with successes
5. More importantly: Thread B reset `_failure_alerted = False`, so the NEXT error sequence will re-alert, causing alert spam

The combination of `_failure_count` and `_failure_alerted` being independently set by concurrent coroutines means the "consecutive failures" semantic is completely broken. The counter can be reset by any successful request while failures are still occurring, and the alert flag can be toggled independently.

For a monitoring counter that claims to track "consecutive failures," this produces false alerts and missed alerts. Not a crash, but the observability signal is unreliable.

**Fix**: Use an `asyncio.Lock` around the success/failure counter updates, or accept that this counter is "approximate" and rename from "consecutive" to "recent" in the log message. Or use `contextvars` if per-task isolation is desired.

**Impact**: False/missed alerts on whisper planner degradation. Monitoring team acts on unreliable signals.

---

### F-004 [MEDIUM] `RequestBodyLimitMiddleware` has a response suppression race after body limit exceeded

**File**: `src/api/middleware.py:437-511`

**Bug**: When the streaming body limit is exceeded (line 477-478), `exceeded` is set to True. The `send_wrapper` (line 483-489) then suppresses ALL subsequent `send()` calls with `return`. However, the application (`self.app`) may have already started sending a response body before the body limit was exceeded -- the `receive_wrapper` only counts bytes as they are consumed by the app, not before.

Consider this sequence for a chunked POST where the app reads the body incrementally:
1. App reads first chunk (4KB) via `receive_wrapper` -- bytes_received = 4096, under limit
2. App starts sending response: `http.response.start` passes through `send_wrapper` (`exceeded` is False)
3. `response_started` would be True if ErrorHandling tracked it (it doesn't here)
4. App reads second chunk (65KB) via `receive_wrapper` -- bytes_received = 69632, exceeds 65536 limit
5. `exceeded = True`
6. App tries to send `http.response.body` -- `send_wrapper` returns silently (suppressed)
7. Client sees `http.response.start` (200 OK with headers) but never gets a body -- hangs forever

The client receives a 200 status with headers but the body is silently dropped. The connection hangs until the client-side timeout fires. This is worse than a proper 413 error.

The `sent_413` flag (line 481) attempts to handle this, but it only fires when `message.get("type") == "http.response.start"` AND `exceeded` is True -- which requires the limit to be exceeded BEFORE the app sends its response start. For most FastAPI endpoints that read the full body before responding, this works. But for streaming endpoints or endpoints that start responding before fully consuming the body, it fails.

**Fix**: Track whether `http.response.start` has been sent. If the body limit is exceeded after response start, log a warning but allow the response to complete (the body was already accepted). Or close the connection explicitly.

**Impact**: Client-side timeout hangs on edge case with large chunked POST requests to streaming endpoints. Low probability for /chat (small JSON body) but possible for /cms/webhook or /sms/webhook with large payloads.

---

### F-005 [MEDIUM] `get_feature_flags` cache read outside lock is not safe with TTLCache expiry

**File**: `src/casino/feature_flags.py:128-148`

**Bug**: The "fast path" at line 129 reads from `_flag_cache` without holding `_flag_lock`:
```python
cached = _flag_cache.get(casino_id)  # No lock
if cached is not None:
    return cached
```

`TTLCache.get()` internally calls `__getitem__` which calls `__missing__` -> `self.expire()` which MUTATES the cache dict by removing expired entries. Two concurrent coroutines hitting the fast path simultaneously can both call `_flag_cache.get()`, which both call `expire()`, modifying the internal dict concurrently.

While CPython's GIL prevents true data corruption at the dict level, the logical behavior is wrong: one coroutine's `get()` can delete the entry that another coroutine's `get()` was about to return, causing a spurious cache miss. With the lock held on the slow path, this creates a thundering herd that the lock was supposed to prevent.

The other TTL-cached singletons (`_get_llm`, `_get_circuit_breaker`, etc.) do NOT have this fast-path optimization -- they always acquire the lock first. This is the only cache in the codebase with a lock-free fast path, and it's the one most likely to see high concurrency (called on every graph invocation via `is_feature_enabled`).

**Fix**: Remove the lock-free fast path and always acquire `_flag_lock` before accessing `_flag_cache`, consistent with every other TTL-cached singleton in the project. The asyncio.Lock has near-zero overhead when uncontended.

**Impact**: Spurious cache misses causing unnecessary Firestore reads on every request during high concurrency. Performance degradation, not correctness failure.

---

### F-006 [MEDIUM] `_build_greeting_categories` uses sync file I/O in an async node

**File**: `src/agent/nodes.py:440-479`

**Bug**: `greeting_node` (line 482) is an async function that calls `_build_greeting_categories` (line 493), which performs synchronous file I/O:
```python
path = Path(settings.PROPERTY_DATA_PATH)
if not path.exists():  # sync stat()
    ...
with open(path, encoding="utf-8") as f:  # sync open + read
    data = json.load(f)
```

This blocks the event loop while reading the property JSON file. The result is cached after first call, so it only blocks once per casino_id per TTL period. However, during cold start or cache eviction, if multiple concurrent requests hit `greeting_node` simultaneously, they all block the event loop waiting for the same file read.

More importantly, `_build_greeting_categories` is NOT protected by any lock, so concurrent callers can all read the file simultaneously (wasted I/O), and concurrent writes to `_greeting_cache[cache_key]` are safe only under CPython's GIL.

The retrieve_node correctly uses `asyncio.to_thread()` for ChromaDB sync calls, but greeting_node does not follow this pattern.

**Fix**: Wrap the file I/O in `asyncio.to_thread()` or read the file during lifespan startup and cache it in `app.state`, consistent with how `property_data` is already loaded in `app.py:92-99`.

**Impact**: Event loop blocked for 1-10ms during cold start or cache miss. Low impact for single-file reads but violates the project's own async discipline.

---

### F-007 [LOW] `_initial_state` hardcodes `responsible_gaming_count: 0` which resets the `_keep_max` reducer to always return 0 on first turn

**File**: `src/agent/graph.py:495` and `src/agent/state.py:70-74`

**Analysis**: The `_keep_max` reducer preserves `max(existing, new)`. When `_initial_state()` sends `responsible_gaming_count: 0`, the reducer computes `max(existing_count, 0)` which correctly preserves the existing count. This is by design and works correctly.

However, on the FIRST turn of a conversation (no existing state), the reducer receives `max(0, 0) = 0`, which is correct. On subsequent turns, if the compliance gate increments to 1, the reducer computes `max(1, 0) = 1` -- preserved correctly.

**No bug here.** The _keep_max reducer + _initial_state(0) pattern is correct. Marking as LOW/informational to confirm analysis was done.

**Status**: Verified correct. No fix needed.

---

### F-008 [LOW] `StreamingPIIRedactor._scan_and_release` applies `redact_pii()` to already-redacted text on subsequent `feed()` calls

**File**: `src/agent/streaming_pii.py:88-121`

**Analysis**: When `force=False` (line 110-121), the lookahead buffer is set to `redacted[-_MAX_PATTERN_LEN:]` -- the last 40 chars of already-redacted text. On the next `feed()` call, this lookahead is prepended to new incoming text and `redact_pii()` is applied to the combined buffer.

This means `redact_pii()` runs on text containing previous redaction placeholders like `[PHONE]`, `[SSN]`, etc. The comment at line 117 says "re-scanning already-redacted placeholders like '[PHONE]' is a no-op" -- which is true because none of the PII regex patterns match the bracket-based placeholder format.

**Status**: Verified correct by inspection. The placeholder format `[PHONE]` does not match any PII regex pattern (phone regex requires digits, SSN requires digits with dashes, etc.). No false positives from re-scanning. Marking as LOW/confirmed-safe.

---

### F-009 [LOW] `BoundedMemorySaver._track_thread` accesses `self._inner.storage` which is an internal implementation detail of MemorySaver

**File**: `src/agent/memory.py:69-74`

**Bug**: The LRU eviction logic accesses `self._inner.storage` to remove evicted thread data:
```python
if hasattr(self._inner, "storage"):
    keys_to_remove = [
        k for k in self._inner.storage if isinstance(k, tuple) and len(k) > 0 and k[0] == evicted_id
    ]
```

`MemorySaver.storage` is not a documented public API of LangGraph's `MemorySaver`. The key format (tuples with thread_id as first element) is an implementation detail that could change between LangGraph versions. The `hasattr` guard prevents crashes but silently degrades to "track without evict" if the internal structure changes, causing the very OOM the BoundedMemorySaver was designed to prevent.

With `langgraph==0.2.60` pinned, this is stable. But on any version bump, this could silently fail.

**Fix**: Add a test that verifies `MemorySaver().storage` exists and has the expected key format. This test would fail on version bumps, providing early warning.

**Impact**: Low (pinned version). Risk on version bump: silent OOM in long-running dev sessions.

---

### F-010 [LOW] `InMemoryBackend.increment` resets TTL on every increment, extending key lifetime indefinitely

**File**: `src/state_backend.py:90-96`

**Bug**: Each `increment()` call sets `expiry = time.monotonic() + ttl`, which resets the TTL clock. A key that is incremented every 30 seconds with `ttl=60` will never expire, because each increment pushes the expiry forward.

For rate limiting counters, this means a client making exactly 1 request every 59 seconds will have their counter grow indefinitely (1, 2, 3, ...) because each increment resets the 60-second TTL. The counter never reaches 0.

This is a semantic mismatch with Redis's `INCR` + `EXPIRE` pattern (lines 145-149 in `RedisBackend`), where `EXPIRE` also resets the TTL. So the behavior is actually *consistent* between backends -- both extend the window on activity. But neither implements a true sliding window (which would require per-request timestamps, not a single counter with TTL).

For the actual rate limiter in `middleware.py`, this backend is NOT used (middleware has its own deque-based implementation). The `StateBackend` is for SMS idempotency and potentially future distributed rate limiting. So this finding is informational.

**Status**: Consistent between backends, not used for the primary rate limiter. Low impact.

---

### F-011 [HIGH] Double `allow_request()` call on the same circuit breaker creates half-open probe starvation

**File**: `src/agent/graph.py:205-207` and `src/agent/agents/_base.py:92-103`

**Bug**: This is related to F-002 but represents a distinct production bug even in the normal (non-parse-error) case.

`_dispatch_to_specialist()` calls `cb.allow_request()` (line 207) to check whether the dispatch LLM call is allowed. Then `execute_specialist()` (called at line 276 via `agent_fn(state)`) calls `cb.allow_request()` AGAIN (line 98 in `_base.py`) on the SAME circuit breaker instance.

**Normal operation (CB closed)**: Both calls return True. No issue.

**Half-open state**:
1. `_dispatch_to_specialist`: `allow_request()` -> True (sets `_half_open_in_progress = True`)
2. Dispatch LLM call succeeds and records success -> CB transitions to closed
3. `execute_specialist`: `allow_request()` -> True (CB is now closed, no issue)

This path works. But consider:

**Half-open + dispatch LLM skipped (CB not allowing)**:
1. `_dispatch_to_specialist`: `allow_request()` -> False (CB is open, not yet half-open)
2. Keyword fallback is used
3. `execute_specialist`: `allow_request()` -> False (CB is still open)
4. Returns fallback message

This also works. But:

**Half-open + dispatch allowed + specialist fails**:
1. `_dispatch_to_specialist`: `allow_request()` -> True, `_half_open_in_progress = True`
2. Dispatch LLM succeeds, `record_success()` called -> CB transitions to closed, `_half_open_in_progress = False`
3. `execute_specialist`: `allow_request()` -> True (CB is closed)
4. Specialist LLM call fails -> `record_failure()` called
5. One failure is recorded, but CB is closed so it doesn't trip

This works but reveals a design concern: the dispatch and specialist paths share one CB, so a parse error on dispatch + success on specialist (or vice versa) conflates two distinct failure domains.

**The actual bug is F-002's sequence, which I'm promoting to HIGH based on this deeper analysis.** The shared CB means any interaction between dispatch and specialist paths can create inconsistent state. F-002 documents the worst case.

**Fix**: Use two separate circuit breaker instances: one for dispatch LLM calls, one for specialist LLM calls. Or don't call `allow_request()` in `_dispatch_to_specialist` at all -- let the keyword fallback handle CB-open cases, and only use the CB in `execute_specialist`.

**Impact**: Same as F-002 -- system stuck in degraded state for up to 1 hour under specific sequences.

---

## Summary

| Severity | Count | Findings |
|----------|-------|----------|
| HIGH | 3 | F-001 (retriever cache race), F-002 (CB half-open stuck), F-011 (double allow_request) |
| MEDIUM | 3 | F-003 (whisper failure_count race), F-004 (body limit response hang), F-005 (flag cache fast-path race) |
| LOW | 4 | F-007 (verified correct), F-008 (verified correct), F-009 (MemorySaver internal API), F-010 (increment TTL semantics) |

## Assessment

The codebase shows exceptional engineering maturity for a 15-round review cycle. The documentation quality is production-grade with origin citations linking every design decision to the review round that prompted it. The architecture is sound -- the 11-node StateGraph with validation loop, 5-layer deterministic guardrails, and DRY specialist extraction are all best-in-class patterns.

The findings in this round are primarily concurrency edge cases that would only manifest under production load:
1. **F-002/F-011** are the most serious: the shared circuit breaker between dispatch and specialist paths creates a hidden coupling that can lock the system into degraded state. This is the kind of bug that passes all unit tests but fails under real traffic patterns.
2. **F-001** is a straightforward fix (add a lock to the retriever cache) that was missed because it's the only cache that runs in a thread pool rather than the event loop.
3. **F-005** is the most subtle: the TTLCache fast-path optimization in feature flags is the opposite of what every other cache in the codebase does, and it's wrong specifically because TTLCache.get() has side effects.

Score held at 85 (slight regression from R15's 86) due to the circuit breaker coupling issue, which is a real production risk. Fixing F-002/F-011 would bring this to 87-88.
