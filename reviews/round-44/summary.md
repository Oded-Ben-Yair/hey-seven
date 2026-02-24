# R44 Summary

**Score trajectory**: R34=77, R35=85, R36=84.5, R37=83, R38=81, R39=84.5, R40=93.5, R41=94.9, R42=94.4, R43=94.3, **R44=94.5**

## Fixes Applied (6 MAJORs)

### D2-M001: RRF score semantics mismatch [FIXED]
- `rerank_by_rrf()` now returns 3-tuples `(doc, cosine_score, rrf_score)` instead of 2-tuples
- Cosine score used for quality filtering (RAG_MIN_RELEVANCE_SCORE), RRF score for ranking/monitoring
- Updated `RetrievedChunk` TypedDict with `rrf_score: NotRequired[float]`
- Updated `_filter_by_relevance()`, `search_knowledge_base()`, `search_hours()` to handle 3-tuples
- Files: `src/rag/reranking.py`, `src/agent/tools.py`, `src/agent/state.py`

### D2-M002: reingest_item async-safe retry with backoff [FIXED]
- Wrapped blocking `add_texts()` in `asyncio.to_thread()` to avoid event loop blocking
- Added exponential backoff between retries (`await asyncio.sleep(0.5 * 2**attempt)`)
- Added `asyncio.CancelledError` propagation (never swallow cancellation)
- File: `src/rag/pipeline.py`

### D3-M001: Firestore batch overflow guard for CCPA delete [FIXED]
- Added chunked batch pattern: commit + start new batch when `ops_count >= 490`
- Prevents Firestore 500-op batch limit from breaking cascade delete on power users
- Tracks `total_ops` across batches for accurate logging
- File: `src/data/guest_profile.py`

### D5-M001: Tests for retry/backoff logic [ADDED]
- `TestIngestRetryLogic`: 2 tests — retry succeeds on 2nd attempt; raises after max retries
- `TestReingestRetryLogic`: 3 tests — retry succeeds; returns False after max retries; CancelledError propagated
- File: `tests/test_rag.py`

### D5-M002: Test for Firestore batch overflow [ADDED]
- `TestDeleteGuestProfileBatchOverflow`: Verifies >600 subcollection docs trigger multiple batch commits
- Full async mock of Firestore client with conversations/messages/behavioral_signals streams
- File: `tests/test_guest_profile.py`

### D8-M001: InMemoryBackend sweep death spiral [FIXED]
- Added FIFO eviction fallback: when force-sweep finds 0 expired entries at capacity, evicts oldest entry
- Converts O(BATCH_SIZE)-per-write death spiral into bounded LRU behavior
- `TestInMemoryBackendFIFOEviction`: 3 tests covering FIFO eviction, sustained load, below-capacity noop
- Files: `src/state_backend.py`, `tests/test_state_backend.py`

## Test Results
- **2178 passed, 0 failures** (up from 2169 — 9 new tests added)
- Coverage: 90.43% (above 90% threshold)

## Files Modified (10)
- `src/rag/reranking.py` — D2-M001 (3-tuple return)
- `src/rag/pipeline.py` — D2-M002 (async retry with backoff)
- `src/agent/tools.py` — D2-M001 (3-tuple consumer update)
- `src/agent/state.py` — D2-M001 (RetrievedChunk rrf_score field)
- `src/data/guest_profile.py` — D3-M001 (batch overflow guard)
- `src/state_backend.py` — D8-M001 (FIFO eviction)
- `tests/test_rag.py` — D5-M001 (retry tests) + D2-M001 (updated RRF tests)
- `tests/test_guest_profile.py` — D5-M002 (batch overflow test)
- `tests/test_state_backend.py` — D8-M001 (FIFO eviction tests)
