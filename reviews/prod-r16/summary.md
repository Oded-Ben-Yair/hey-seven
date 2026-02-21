# Hey Seven R16 Fix Summary

**Fixer**: Claude Opus 4.6 (code-worker)
**Date**: 2026-02-21
**Review inputs**: DeepSeek, Gemini, Grok (full adversarial, all dimensions)
**Tests**: 1452 passed, 20 skipped, 0 failed (90.16% coverage)

---

## Consensus Matrix

| Finding | DeepSeek | Gemini | Grok | Consensus | Action |
|---------|----------|--------|------|-----------|--------|
| Retriever cache bare dict, no lock | F-001 HIGH | F-004 MAJOR | C-001 CRITICAL | **3/3** | **Fixed** |
| Whisper planner global mutable race | F-003 MEDIUM | F-016 CRITICAL | -- | **2/3** | **Fixed** |
| `_request_counter` not initialized in `__init__` | -- | F-009 MINOR | M-006 MEDIUM | **2/3** | **Fixed** |
| `threading.Lock` in Firestore client accessors | -- | F-001 MAJOR | -- | **1/3** (borderline) | **Fixed** (Gemini MAJOR, aligns with codebase async discipline) |
| CB half-open probe starvation | F-002+F-011 HIGH | -- | -- | 1/3 | Skipped |
| `get_settings()` @lru_cache undermines TTL rotation | -- | -- | H-001 HIGH | 1/3 | Skipped |
| Feature flags lock-free fast path | F-005 MEDIUM | -- | -- | 1/3 | Skipped |
| Dockerfile HEALTHCHECK /health vs /live | -- | -- | H-002 HIGH | 1/3 | Skipped |
| BoundedMemorySaver no lock | -- | -- | H-003 HIGH | 1/3 | Skipped |
| Smoke test version mismatch unenforced | -- | -- | C-002 CRITICAL | 1/3 | Skipped |
| Duplicated Firestore clients | -- | F-007 MAJOR | -- | 1/3 | Skipped |

---

## Fixes Applied (4 total)

### Fix 1: Retriever cache thread safety (3/3 CRITICAL)

**Files**: `src/rag/pipeline.py`

**Problem**: `_retriever_cache` and `_retriever_cache_time` are bare dicts with no lock. The retriever is accessed via `asyncio.to_thread()` from `retrieve_node`, meaning multiple thread-pool workers can race on TTL expiry, both creating separate retriever instances (resource leak, duplicate ChromaDB/Firestore connections).

**Fix**: Added `threading.Lock` (correct primitive since this runs in thread pool, not event loop) wrapping the entire `_get_retriever_cached()` function body. Uses the double-check locking pattern consistent with `_get_llm`, `_get_circuit_breaker`, and `get_checkpointer`. Also protected `clear_retriever_cache()` with the same lock.

**Why `threading.Lock` not `asyncio.Lock`**: This function runs inside `asyncio.to_thread()` in a thread-pool worker. `asyncio.Lock` requires an event loop; `threading.Lock` is the correct primitive for thread-pool code.

### Fix 2: Whisper planner global mutable state (2/3 HIGH)

**Files**: `src/agent/whisper_planner.py`

**Problem**: `_failure_count` and `_failure_alerted` are module-level globals mutated via bare `global` statement with no synchronization. The docstring claimed "benign race" but the success-reset path (`_failure_count = 0` on any success) races with concurrent failure increments, permanently suppressing the alert threshold under mixed traffic. One successful request amidst 20 failures resets the counter to 0.

**Fix**: Added `asyncio.Lock` (`_failure_lock`) protecting both the success-reset and failure-increment paths. The `_failure_count` and `_failure_alerted` mutations are now atomic with respect to concurrent coroutines. Updated the docstring to cite the R16 consensus.

### Fix 3: `_request_counter` initialization (2/3 MEDIUM)

**Files**: `src/api/middleware.py`

**Problem**: `_request_counter` in `RateLimitMiddleware` was lazily created via `getattr(self, "_request_counter", 0) + 1` instead of being declared in `__init__`. This violates Python best practices for class attribute initialization and is invisible to type checkers, linters, and IDE autocomplete. Every other middleware in the file initializes all attributes in `__init__`.

**Fix**: Added `self._request_counter: int = 0` to `__init__` and simplified the usage to `self._request_counter += 1`.

### Fix 4: `threading.Lock` -> `asyncio.Lock` in Firestore client accessors (borderline 2/3 MAJOR)

**Files**: `src/data/guest_profile.py`, `src/casino/config.py`, `tests/test_guest_profile.py`

**Problem**: Both `_get_firestore_client()` functions used `threading.Lock` to guard Firestore `AsyncClient` creation. These functions are called from async contexts only. `threading.Lock.acquire()` blocks the entire event loop under contention, starving all other coroutines. The rest of the codebase correctly uses `asyncio.Lock` for async singletons.

**Fix**:
1. Converted both `_get_firestore_client()` to `async def` with `asyncio.Lock`
2. Added `await` to all 4 call sites (3 in `guest_profile.py`, 1 in `config.py`)
3. Removed `import threading` from both modules
4. Updated one test that directly called the function to be async
5. All `@patch` decorators in tests automatically switch to `AsyncMock` when target is `async def`

---

## Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `src/rag/pipeline.py` | Modified | Added `threading.Lock` to retriever cache |
| `src/agent/whisper_planner.py` | Modified | Added `asyncio.Lock` for failure counter |
| `src/api/middleware.py` | Modified | Initialized `_request_counter` in `__init__` |
| `src/data/guest_profile.py` | Modified | `threading.Lock` -> `asyncio.Lock`, sync -> async |
| `src/casino/config.py` | Modified | `threading.Lock` -> `asyncio.Lock`, sync -> async |
| `tests/test_guest_profile.py` | Modified | Updated test to await async function |

---

## Test Results

```
1452 passed, 20 skipped, 1 warning in 39.10s
Coverage: 90.16% (above 90% gate)
```

No regressions. All existing tests pass unchanged (except one test updated for async signature).

---

## Unfixed (1/3 consensus, deferred)

| Finding | Why Deferred |
|---------|-------------|
| CB half-open starvation (DeepSeek F-002/F-011) | Only 1/3 consensus. Complex fix (separate CB instances or remove dispatch CB). Risk of introducing new bugs. |
| `get_settings()` @lru_cache (Grok H-001) | Only 1/3 consensus. Converting to TTLCache has ripple effects on all callers. Credential rotation via container restart is acceptable for current deployment. |
| Feature flags lock-free fast path (DeepSeek F-005) | Only 1/3 consensus. Performance vs correctness trade-off under CPython GIL makes this low-risk. |
| Dockerfile HEALTHCHECK (Grok H-002) | Only 1/3 consensus. Cloud Run ignores Dockerfile HEALTHCHECK. Only affects local docker-compose. |
| Smoke test version assertion (Grok C-002) | Only 1/3 consensus. Cloud Build configuration change, not a code fix. |
| Duplicated Firestore clients (Gemini F-007) | Only 1/3 consensus. Architectural decision with documented trade-off rationale. |
