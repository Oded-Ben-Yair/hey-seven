# Round 5 Production Review -- Summary

**Date**: 2026-02-20
**Spotlight**: SCALABILITY & ASYNC PATTERNS
**Reviewers**: Gemini 3 Pro (69), GPT-5.2 (63), DeepSeek-V3.2-Speciale (58)
**Average Score**: 63.3
**Previous**: R1=67.3, R2=61.3, R3=60.7, R4=66.7

---

## Consensus Findings (2/3+ models flagged)

| Finding | Gemini | GPT | DeepSeek | Severity | Status |
|---------|--------|-----|----------|----------|--------|
| MemorySaver unbounded growth | F1 CRIT | F10 MED | - | CRITICAL | FIXED: BoundedMemorySaver with LRU eviction (MAX_ACTIVE_THREADS=1000) |
| Rate limiter unbounded per-client deque | - | F3 HIGH | F2 CRIT | CRITICAL | FIXED: maxlen=max_tokens, rejected requests not recorded |
| Feature flag cache race condition | - | F6 HIGH | F3 HIGH | HIGH | FIXED: asyncio.Lock + TTLCache(maxsize=100) |
| Casino config cache race condition | - | F6 HIGH | F4 HIGH | HIGH | FIXED: asyncio.Lock + TTLCache(maxsize=100) |
| ApiKeyMiddleware torn pair read | - | F9 MED | F5 HIGH | HIGH | FIXED: atomic tuple (key, timestamp) |
| Firestore client creation race | - | F5 HIGH | - | HIGH | FIXED: threading.Lock + double-check |

## Single-Model CRITICAL Findings

| Finding | Model | Severity | Status |
|---------|-------|----------|--------|
| Circuit breaker deque maxlen undercounting | DeepSeek F1 | CRITICAL | FIXED: removed maxlen, rely on prune |
| Checkpointer @lru_cache no TTL for credential rotation | DeepSeek F6 | CRITICAL | FIXED: TTLCache(ttl=3600) + asyncio.Lock |
| No LLM concurrency backpressure | Gemini F3 | HIGH | FIXED: asyncio.Semaphore(20) in _base.py |
| Guest profile in-memory store unbounded | GPT F2 | CRITICAL | FIXED: _MEMORY_STORE_MAX=10K with FIFO eviction |

## Additional Fixes (MEDIUM)

| Finding | Model | Status |
|---------|-------|--------|
| PII buffer no max-size guard | Gemini F10 | FIXED: _PII_MAX_BUFFER=500 hard cap |

---

## Changes Summary

### Files Modified (11)

| File | Change | Description |
|------|--------|-------------|
| `src/agent/memory.py` | Rewritten | BoundedMemorySaver with LRU eviction, TTLCache + asyncio.Lock for credential rotation, async get_checkpointer() |
| `src/agent/circuit_breaker.py` | Modified | Removed deque maxlen to prevent failure undercounting |
| `src/api/middleware.py` | Modified | Rate limiter: deque maxlen=max_tokens; ApiKey: atomic tuple |
| `src/casino/feature_flags.py` | Modified | TTLCache(maxsize=100) + asyncio.Lock for thundering herd prevention |
| `src/casino/config.py` | Modified | TTLCache(maxsize=100) + asyncio.Lock; Firestore client with threading.Lock |
| `src/agent/agents/_base.py` | Modified | LLM concurrency semaphore (asyncio.Semaphore(20)) |
| `src/agent/graph.py` | Modified | PII buffer hard cap (_PII_MAX_BUFFER=500) |
| `src/data/guest_profile.py` | Modified | Memory store bounded to 10K entries; Firestore client with threading.Lock |
| `src/api/app.py` | Modified | await async get_checkpointer() |
| `tests/conftest.py` | Modified | Updated cache clearing for new checkpointer pattern |
| `tests/test_casino_config.py` | Modified | Adapted to TTLCache (no more tuple unpacking) |

### Files Created (1)

| File | Tests | Description |
|------|-------|-------------|
| `tests/test_r5_scalability.py` | 21 | BoundedMemorySaver, rate limiter bounds, CB maxlen, cache locks, atomic tuple, PII cap, LLM semaphore, memory store bounds |

### Tests Updated (1)

| File | Changes |
|------|---------|
| `tests/test_firestore_retriever.py` | 3 tests updated from sync to async get_checkpointer |

---

## Test Results

- **Total tests**: 1178 (was 1157, +21 new)
- **Passed**: 1178
- **Failed**: 0
- **Skipped**: 20
- **Coverage**: 90.58%

---

## Documented Trade-offs (NOT fixed -- accepted for single-container MVP)

These were flagged by reviewers but are **documented, intentional** single-container trade-offs:

1. **Rate limiter per-container**: Each Cloud Run container has independent counters (effective limit = N * limit for N containers). Mitigation: single-container deployment with `--min-instances=1 --max-instances=1`.
2. **Circuit breaker per-container**: Independent failure tracking per container. Acceptable for per-instance health detection. Fleet-wide protection requires Redis (Phase 2).
3. **Idempotency tracker per-container**: Webhook replay across replicas can bypass dedup. Acceptable for single-container; Redis SETNX for Phase 2.
4. **@lru_cache on embeddings/retriever/langfuse**: No TTL for credential rotation. Acceptable because: embeddings use API key (not Workload Identity), retriever is local ChromaDB (no credentials), Langfuse is optional monitoring. All are cleared on container restart.
5. **WEB_CONCURRENCY=1**: Single uvicorn worker. Async I/O handles concurrency for I/O-bound tasks. Multiple workers would require external state for rate limiter.

---

## Fix Approach Summary

The R5 fixes follow a consistent pattern already proven in the codebase: **TTLCache + asyncio.Lock** for credential-sensitive singletons, **bounded collections** for all in-memory state, and **atomic operations** for shared mutable state. No new dependencies were added -- all fixes use `cachetools.TTLCache` (already in requirements.txt) and stdlib `asyncio`/`threading`.
