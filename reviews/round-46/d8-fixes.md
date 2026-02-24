# R46 D8 Fixes: Scalability & Production

**Date**: 2026-02-24
**Fixer**: Opus 4.6 (code-worker)

---

## Fixes Applied (6/8 findings)

### D8-C001 (CRITICAL) -- FIXED
**File**: `src/agent/circuit_breaker.py`
**Issue**: `_sync_to_backend()` and `_sync_from_backend()` called synchronous Redis methods directly in async functions, blocking the event loop.
**Fix**: Wrapped all synchronous Redis calls in `asyncio.to_thread()`. Captured state variables locally before passing to the thread function to avoid race conditions. Both `_sync_to_backend()` and `_sync_from_backend()` now execute Redis I/O off the event loop.

### D8-C002 (CRITICAL) -- FIXED
**File**: `src/api/middleware.py`
**Issue**: `_is_allowed_redis()` called synchronous Redis pipeline (zremrangebyscore, zcard, zadd, expire, execute, zrem) in async handler, blocking the event loop on every rate-limited request.
**Fix**: Extracted all Redis operations into `_do_redis_check()` inner function and wrapped in `asyncio.to_thread()`. Captured all needed values (key, now, window_start, member, redis_client, max_tokens) locally before the thread boundary.

### D8-M001 (MAJOR) -- FIXED
**File**: `src/api/middleware.py`
**Issue**: `redis.Redis.from_url()` and `ping()` called at `RateLimitMiddleware.__init__()` time with default Redis timeouts (5-30s), potentially blocking container startup if Redis is unreachable.
**Fix**: Added `socket_connect_timeout=2` and `socket_timeout=2` to `redis.Redis.from_url()` call. 2-second timeout is sufficient for VPC-internal Redis (typical RTT <1ms) while preventing startup probe timeout.

### D8-M002 (MAJOR) -- FIXED
**File**: `src/agent/agents/_base.py`
**Issue**: Semaphore timeout hardcoded to `30` at line 327 instead of reading `settings.LLM_SEMAPHORE_TIMEOUT` (config.py line 76). Config field was dead -- changing it via environment variable had no effect.
**Fix**: Replaced `timeout=30` with `timeout=settings.LLM_SEMAPHORE_TIMEOUT` (already available via the `settings` variable in scope). Updated the log message to use the configurable value.

### D8-M003 (MAJOR) -- FIXED (combined with D8-C001)
**File**: `src/agent/circuit_breaker.py`
**Issue**: `_sync_to_backend` stored `self._last_failure_time` (monotonic clock) in Redis. Monotonic clocks have different epochs per process -- stored values are meaningless cross-instance.
**Fix**: Changed the Redis-synced value from `str(self._last_failure_time)` (monotonic) to `str(time.time())` (wall clock). Local timing still uses monotonic clock for correctness. Wall clock is only written to Redis where cross-instance comparability matters.

### D8-m001 (MINOR) -- FIXED
**File**: `src/agent/circuit_breaker.py`
**Issue**: `_sync_from_backend()` called outside `async with self._lock` in `allow_request()`, causing redundant Redis reads under concurrency.
**Fix**: Moved `await self._sync_from_backend()` inside `async with self._lock:` block. This eliminates the race where two concurrent `allow_request()` calls both read from Redis before either acquires the lock.

---

## Findings Skipped (2/8)

### D8-M004 (MAJOR) -- SKIPPED (design is intentional)
**File**: `src/state_backend.py`
**Issue**: `InMemoryBackend` uses `threading.Lock` but called from async code paths.
**Rationale**: The existing code at line 64-68 already documents this design choice (R36 fix B5): "InMemoryBackend is called from async coroutines but its operations are synchronous and sub-microsecond, so threading.Lock is appropriate (no awaits inside critical sections)." The threading.Lock protects the TOCTOU race in `increment()`. Changing to lock-free removes correctness guarantees. Changing to `asyncio.Lock` would break non-async callers. The finding itself notes impact is "negligible in practice."

### D8-m002 (MINOR) -- SKIPPED (correct design for sync init)
**File**: `src/observability/langfuse_client.py`
**Issue**: `_get_langfuse_client()` uses `threading.Lock` for double-checked locking.
**Rationale**: Langfuse is a synchronous library. `threading.Lock` is correct for synchronous init. The finding rates this as "low priority since it's a one-time init cost." Changing to `asyncio.Lock` would break `is_observability_enabled()` which calls `_get_langfuse_client()` synchronously. The init happens once per TTL expiry (1 hour).

---

## Files Modified

| File | Changes |
|------|---------|
| `src/agent/circuit_breaker.py` | D8-C001 (to_thread for backend sync), D8-M003 (wall clock), D8-m001 (sync inside lock) |
| `src/api/middleware.py` | D8-C002 (to_thread for Redis rate limit), D8-M001 (socket timeout) |
| `src/agent/agents/_base.py` | D8-M002 (config-driven semaphore timeout) |

## Test Results

- `tests/test_r46_scalability.py`: **31 passed**, 0 failed
- `tests/test_middleware.py`: **34 passed**, 0 failed
- `tests/test_r5_scalability.py` + `tests/test_base_specialist.py` + `tests/test_nodes.py`: **204 passed**, 0 failed
- Total: **269 tests passed, 0 failed**
