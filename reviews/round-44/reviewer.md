# R44 Focused Review: D2 RAG + D3 Data + D5 Testing + D8 Scalability

**Reviewer**: Opus 4.6 + GPT-5.2 Codex + Gemini 3.1 Pro cross-validation
**Commit**: 04c6f70 | **Tests**: 2169 passed, 0 failures | **Coverage**: ~90%
**Target**: Lift 4 dimensions from 8.5 to 9.0

---

## D2 RAG Pipeline (8.5 -> target 9.0, weight 0.10)

### Current strengths
- Per-item chunking with category-specific formatters (unanimously praised R1-R20)
- RRF reranking with SHA-256 dedup, null-byte delimiter fix (R36)
- Version-stamp purging for stale chunks
- Retry logic for `ingest_property()` (R40) and `reingest_item()` (R40)
- Embedding health check before caching (R39)
- TTL jitter on embeddings cache (R40)

### Finding D2-M001: RRF returns raw cosine score, not fused score [MAJOR]
**File**: `src/rag/reranking.py:55`
**Issue**: `rerank_by_rrf()` returns `(doc, original_cosine_score)` but the RRF fusion score (the actual ranking metric) is discarded. Downstream consumers that filter by `RAG_MIN_RELEVANCE_SCORE` are applying thresholds to the raw cosine score, not the fused relevance. A document ranked #1 by RRF (appearing in both strategy lists) could be filtered out if its best cosine score happens to be below threshold.
**Impact**: Retrieval quality -- the score used for filtering is not the score used for ranking.
**Fix**: Return the RRF score alongside the document, or at minimum document the semantic mismatch so consumers know the `score` field is "best raw cosine", not "fused relevance."

Cross-validated: GPT-5.2 Codex confirmed as quality gap. Gemini concurs.

### Finding D2-M002: reingest_item retry is synchronous in async function [MAJOR]
**File**: `src/rag/pipeline.py:346-364`
**Issue**: `reingest_item()` is `async` but the retry loop calls `retriever.vectorstore.add_texts()` synchronously with no backoff and no `asyncio.CancelledError` propagation. ChromaDB's `add_texts` is a blocking call (writes to SQLite). Under concurrent CMS webhook calls, this blocks the event loop during retries.
**Fix**: Wrap in `asyncio.to_thread()` for the blocking vectorstore call, add exponential backoff between retries (`await asyncio.sleep(0.5 * 2**attempt)`), and ensure `CancelledError` is re-raised.

Cross-validated: GPT-5.2 Codex flagged as gap A. Gemini concurs.

### Finding D2-m001: Retriever cache key too coarse [MINOR]
**File**: `src/rag/pipeline.py:1020`
**Issue**: Cache key is `f"{settings.CASINO_ID}:default"`. If `VECTOR_DB` or `EMBEDDING_MODEL` is changed at runtime (e.g., via environment variable + cache clear), the stale retriever persists until TTL expires. Low risk in practice (these don't change at runtime), but violates the principle of cache key completeness.
**Fix**: Include `VECTOR_DB` in cache key: `f"{settings.CASINO_ID}:{settings.VECTOR_DB}:default"`.

### Lift action for D2 (8.5 -> 9.0):
1. Fix D2-M002: async-safe `reingest_item` retry with backoff (high impact, low effort)
2. Document or fix D2-M001: RRF score semantics (medium impact, clarifies contract)

---

## D3 Data Model (8.5 -> target 9.0, weight 0.10)

### Current strengths
- Custom reducers for accumulated state fields (`_merge_dicts`, `_keep_max`, `_keep_truthy`)
- `GuestContext` TypedDict for structured agent context
- ProfileField with confidence scoring and source tracking
- 90-day confidence decay with proper UTC handling
- `_initial_state()` parity check at import time (runtime ValueError, not assert)
- `filter_low_confidence()` prevents unreliable data reaching LLM context
- `update_confidence()` with confirm/contradict logic + ceiling/floor guards

### Finding D3-M001: delete_guest_profile has no batch overflow guard for Firestore [MAJOR]
**File**: `src/data/guest_profile.py:296-337`
**Issue**: The CCPA cascade delete adds ALL subcollection documents to a single Firestore batch. Firestore batch limit is 500 operations. A guest with >500 conversation messages would exceed the limit and the batch.commit() would fail, leaving a partially-visible profile (CCPA violation). The comment on line 299 acknowledges "batch limit is 500 operations; cascade unlikely to exceed this" but provides no guard.
**Fix**: Count `ops_count` and commit + start a new batch when approaching 500. This is a straightforward chunked-batch pattern:
```python
if ops_count >= 490:  # Leave margin
    await batch.commit()
    batch = db.batch()
    ops_count = 0
```

Cross-validated: This is a correctness gap, not just scalability. A power user with heavy conversation history would trigger this.

### Finding D3-m001: calculate_completeness doesn't handle concurrent field updates [MINOR]
**File**: `src/data/models.py:245-291`
**Issue**: `calculate_completeness()` is a pure function with no side effects, but it operates on a mutable dict. If called from concurrent async contexts on the same profile dict, one caller could see partially-updated weights. Low risk since `update_guest_profile` does `copy.deepcopy` in `get_agent_context`, but worth noting.
**Observation only**: No fix needed; current call pattern is safe.

### Lift action for D3 (8.5 -> 9.0):
1. Fix D3-M001: Firestore batch overflow guard in `delete_guest_profile` (high impact -- CCPA compliance, low effort)

---

## D5 Testing Strategy (8.5 -> target 9.0, weight 0.10)

### Current strengths
- 2169 tests, 0 failures -- zero-failure milestone maintained
- 58 test files covering all 10 packages
- Comprehensive conftest with 15+ singleton caches cleared
- Schema parity assertion at import time
- Regulatory invariant tests (test_regulatory_invariants.py)
- Property-based tests via hypothesis (test_property_based.py)
- E2E pipeline tests (test_e2e_pipeline.py, test_full_graph_e2e.py)
- Phase 2/3/4 integration test suites

### Finding D5-M001: No tests for RAG retry/backoff logic [MAJOR]
**File**: `src/rag/pipeline.py:763-794` (ingest retry), `src/rag/pipeline.py:346-364` (reingest retry)
**Issue**: The `_INGEST_MAX_RETRIES` and `_REINGEST_MAX_RETRIES` retry loops added in R40 have zero test coverage. No test verifies:
- Retry succeeds on 2nd attempt after 1st fails
- All retries exhausted raises the original exception
- Backoff timing (when added per D2-M002)
These are critical paths -- ingest failure means empty RAG for the container lifetime.
**Fix**: Add parametrized tests mocking `Chroma.from_texts` / `vectorstore.add_texts` to fail N times then succeed, verifying retry count and final behavior.

### Finding D5-M002: No test for Firestore batch overflow in CCPA delete [MAJOR]
**File**: `src/data/guest_profile.py:296-337`
**Issue**: `test_delete_cascades_removes_from_store` only tests the in-memory path. No test exercises the Firestore batch path with >500 operations to verify the batch doesn't exceed Firestore's limit. Related to D3-M001.
**Fix**: Add a test that mocks a guest with 600+ conversation messages and verifies the delete completes without Firestore batch limit errors.

### Finding D5-m001: No flaky test detection mechanism [MINOR]
**Issue**: With 2169 tests, flaky tests are statistically inevitable but there's no mechanism to detect them. No `pytest-rerunfailures`, no CI-level flaky detection, and no `@pytest.mark.flaky` annotations.
**Fix**: Add `pytest-rerunfailures` with `--reruns 2 --reruns-delay 1` to CI. Tests that pass on rerun are flagged as flaky for investigation.

### Finding D5-m002: Missing edge case tests for _merge_dicts reducer [MINOR]
**File**: `src/agent/state.py:25-47`
**Issue**: The `_merge_dicts` reducer filters `None` and `""` values (R37/R38 fixes), but there are no explicit tests for:
- `_merge_dicts({}, {"name": None})` -> should return `{}`
- `_merge_dicts({"name": "Alice"}, {"name": ""})` -> should preserve "Alice"
- `_merge_dicts({"name": "Alice"}, {"name": 0})` -> should 0 overwrite? (it does, since `0 is not None and 0 != ""`)
These are the exact scenarios the R37/R38 fixes addressed, but the fixes lack regression tests.

### Lift action for D5 (8.5 -> 9.0):
1. Add retry logic tests for ingest/reingest (D5-M001) -- verifies R40's most critical addition
2. Add CCPA batch overflow test (D5-M002) -- required if D3-M001 fix is implemented

---

## D8 Scalability & Production (8.5 -> target 9.0, weight 0.15)

### Current strengths
- TTL jitter on all 6+ singletons (R40) -- prevents thundering herd
- SIGTERM graceful drain with active stream tracking (R40)
- Pure ASGI middleware (no BaseHTTPMiddleware -- preserves SSE streaming)
- Sliding-window rate limiter with LRU eviction and background sweep
- Circuit breaker with rolling window, half-open probe, degraded-pass
- Settings validation (production secrets, chunk params, consent HMAC)
- Production-only guard against ChromaDB (R40)
- Exec-form CMD in Dockerfile (R42)
- SSE heartbeats during long LLM generations
- OpenAPI/Swagger disabled in production (R42)

### Finding D8-M001: InMemoryBackend sweep death spiral at capacity [MAJOR]
**File**: `src/state_backend.py:78-114`
**Issue**: When the store reaches `_MAX_STORE_SIZE` (50,000) and all entries are unexpired (plausible under sustained load with 60s TTL and 10K unique IPs per minute), `_maybe_sweep()` triggers on EVERY write (force=True), iterates up to 1000 entries, finds 0 expired, deletes 0, and returns. The store stays at 50K. The next write triggers another forced sweep. This creates an O(1000)-per-write overhead on every single request, burning CPU under the `threading.Lock` and blocking the event loop.
**Impact**: Under sustained high load, the state backend becomes a CPU bottleneck. The background sweep in RateLimitMiddleware handles rate-limit cleanup, but the state_backend is used for idempotency tracking (SMS webhooks), which could accumulate under attack.
**Fix**: When force-sweep finds 0 expired keys, evict the oldest entry (FIFO via `next(iter(self._store))`) to ensure the store drops below `_MAX_STORE_SIZE`. This converts the death spiral into bounded LRU behavior:
```python
if is_full and not expired_keys:
    oldest = next(iter(self._store))
    del self._store[oldest]
```

Cross-validated: Gemini 3.1 Pro identified this as "the ticking time bomb" preventing 9.0. GPT-5.2 Codex concurs on bounded eviction.

### Finding D8-m001: metrics endpoint walks middleware chain without await safety [MINOR]
**File**: `src/api/app.py:217-224`
**Issue**: The `/metrics` endpoint walks `app.middleware_stack` via `getattr(middleware, "app", None)`. This chain walk has no depth guard -- a misconfigured middleware stack could infinite-loop. Also, the `async with middleware._lock` accesses a private attribute.
**Fix**: Add a depth guard (`max 20 iterations`) and use the public `_requests_lock` name consistently (or expose a `get_client_count()` method on RateLimitMiddleware).

### Lift action for D8 (8.5 -> 9.0):
1. Fix D8-M001: Add FIFO eviction fallback when force-sweep finds 0 expired entries (critical for sustained load)

---

## Summary

| Dimension | Current | MAJORs | MINORs | Primary Lift Action |
|-----------|---------|--------|--------|---------------------|
| D2 RAG | 8.5 | 2 | 1 | Async-safe reingest retry with backoff |
| D3 Data | 8.5 | 1 | 1 | Firestore batch overflow guard in CCPA delete |
| D5 Testing | 8.5 | 2 | 2 | Tests for retry logic + batch overflow |
| D8 Scalability | 8.5 | 1 | 1 | Fix sweep death spiral with FIFO eviction fallback |

**Total findings**: 6 MAJORs, 5 MINORs

**Projected score impact**: Fixing all 6 MAJORs lifts each dimension to 9.0, adding approximately:
- D2: +0.5 * 0.10 = +0.05
- D3: +0.5 * 0.10 = +0.05
- D5: +0.5 * 0.10 = +0.05
- D8: +0.5 * 0.15 = +0.075
- **Total**: +0.225 points, from 94.3 to ~94.5

Note: The score ceiling is ~95.0. These are the most impactful changes remaining.
