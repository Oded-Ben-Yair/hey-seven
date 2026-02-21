# R14 Fix Summary

**Date**: 2026-02-21
**Commit base**: 8655074
**Reviewers**: DeepSeek, Gemini, Grok
**Scores**: DeepSeek 86/100, Gemini 82/100, Grok 72/100 (avg 80/100)

---

## Consensus Analysis

| # | Finding | DeepSeek | Gemini | Grok | Consensus | Fixed |
|---|---------|----------|--------|------|-----------|-------|
| 1 | `get_embeddings()` uses `@lru_cache` not TTLCache | F-002 (M) | F5 (M) | F-001 (C) | 3/3 | Yes |
| 2 | `reingest_item()` missing `_ingestion_version` | -- | F2 (H) | F-003 (C) | 2/3 | Yes |
| 3 | Firestore retriever property_id Python-side filter | F-003 (M) | F4 (M) | F-002 (C) | 3/3 | Yes |
| 4 | Cosine distance normalization inconsistent | -- | -- | F-007 (H) | 1/3 | Yes* |
| 5 | `property_id` derivation inconsistency in reingest | F-006 (M) | -- | -- | 1/3 | Yes* |
| 6 | CB singleton no lock protection | -- | F3 (H) | -- | 1/3 | No |
| 7 | Retriever cache race condition | F-001 (H) | -- | -- | 1/3 | No |
| 8 | CMS hash store in-memory | -- | -- | F-004 (H) | 1/3 | No |
| 9 | Graceful shutdown handler | -- | -- | F-005 (H) | 1/3 | No |

\* Fixed as part of related consensus fix (same file/function).

---

## Fixes Applied (5 logical fixes, 4 source files + 1 test file)

### Fix 1: Embeddings `@lru_cache` -> TTLCache (3/3 consensus)

**File**: `src/rag/embeddings.py`
**Severity**: CRITICAL (Grok), MEDIUM (DeepSeek, Gemini)

Replaced `@lru_cache(maxsize=4)` with `TTLCache(maxsize=4, ttl=3600)` for GCP
Workload Identity credential rotation. `@lru_cache` never expires, causing
embedding calls to fail after credential rotation (~1 hour in WIF environments).
All other credential-bearing singletons already use TTLCache.

Added `clear_embeddings_cache()` function and backward-compatible
`get_embeddings.cache_clear` attribute to avoid breaking callers.

### Fix 2: `reingest_item()` missing `_ingestion_version` + stale chunk purge (2/3 consensus)

**File**: `src/rag/pipeline.py` (lines 262-340)
**Severity**: HIGH (Gemini), CRITICAL (Grok)

- Added `_ingestion_version` metadata to CMS-updated chunks, matching `ingest_property()`.
- Added stale chunk purge after successful upsert: queries for same `property_id` + `source`
  with different `_ingestion_version` and deletes old chunks.
- Without this fix: (a) bulk `ingest_property()` purges CMS-updated chunks (missing version
  stamp matches "!= current"), and (b) repeated CMS updates accumulate ghost chunks.
- Purge failure is non-critical (logged, not raised) consistent with `ingest_property()`.

### Fix 3: `reingest_item()` property_id derivation consistency (1/3 but same function as Fix 2)

**File**: `src/rag/pipeline.py` (line 296)
**Severity**: MEDIUM (DeepSeek F-006)

Changed `property_id` derivation from `CASINO_ID` to `PROPERTY_NAME.lower().replace(" ", "_")`,
consistent with `ingest_property()`, `CasinoKnowledgeRetriever.retrieve()`, and
`FirestoreRetriever.retrieve_with_scores()`. Previously, multi-tenant deployments
where CASINO_ID differs from PROPERTY_NAME would make CMS-updated chunks invisible
to retrieval (different property_id in metadata vs filter).

### Fix 4: Firestore server-side property_id filter (3/3 consensus)

**File**: `src/rag/firestore_retriever.py`
**Severity**: CRITICAL (Grok), MEDIUM (DeepSeek, Gemini)

- Added server-side `where("metadata.property_id", "==", property_id)` pre-filter
  before `find_nearest()` to eliminate cross-tenant data reaching the application layer.
- Requires a composite index on `(metadata.property_id, embedding)` in Firestore.
- Graceful fallback: if composite index missing, logs warning once and falls back to
  Python-side 2x over-fetch filtering (previous behavior).
- Added `_use_server_filter` instance flag for test control.

### Fix 5: Cosine distance normalization alignment (part of Fix 4)

**File**: `src/rag/firestore_retriever.py` (line 117)
**Severity**: HIGH (Grok F-007), addressed as part of same file

Changed cosine distance -> similarity formula from `1 - distance` to `1 - (distance / 2)`
to match ChromaDB's LangChain normalization. Previous formula compressed the similarity
scale: distance=0.3 mapped to 0.70 (Firestore) vs 0.85 (ChromaDB), causing the same
`RAG_MIN_RELEVANCE_SCORE` threshold to filter differently across backends.

---

## Test Updates

**File**: `tests/conftest.py`
- Updated embedding cache clear from `get_embeddings.cache_clear()` to `_embeddings_cache.clear()`.

**File**: `tests/test_firestore_retriever.py`
- Updated `_make_retriever()` to accept `use_server_filter` parameter (default False for tests).
- Added `google.cloud.firestore_v1.field_path` to mock module registry.
- Updated cosine distance test expectations to match new formula (0.9 not 0.8, 0.5 not 0.0).
- Added distance=2.0 test case (opposite vectors -> similarity 0.0).
- Added `test_server_side_filter_used_when_available` test.
- Added `test_server_side_filter_fallback_on_error` test.
- Added `_server_filter_warned` reset in fixture teardown.

---

## Test Results

```
1452 passed, 20 skipped, 1 warning in 9.27s
```

No regressions. 20 skipped tests are pre-existing (environment-dependent). Warning is pre-existing (async mock in test_integration.py).

---

## Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `src/rag/embeddings.py` | Modified | `@lru_cache` -> TTLCache(maxsize=4, ttl=3600) |
| `src/rag/pipeline.py` | Modified | `reingest_item()` + `_ingestion_version` + stale purge + property_id fix |
| `src/rag/firestore_retriever.py` | Modified | Server-side property_id filter + cosine normalization |
| `tests/conftest.py` | Modified | Embedding cache clear updated for TTLCache |
| `tests/test_firestore_retriever.py` | Modified | Updated test expectations + new server filter tests |

---

## Not Fixed (1/3 consensus only)

- CB singleton TOCTOU race (Gemini F3) -- 1/3, narrow race window, build-time usage complicates async conversion
- Retriever cache race condition (DeepSeek F-001) -- 1/3, sync/async architectural tension
- CMS hash store persistence (Grok F-004) -- 1/3, operational enhancement not a bug
- Graceful shutdown handler (Grok F-005) -- 1/3, requires careful testing with real Cloud Run
- Smoke test version assertion (Grok F-006) -- 1/3, CI/CD change outside code scope
- Metrics export (Grok F-009) -- 1/3, feature request not a bug
