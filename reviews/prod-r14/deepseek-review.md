# Production Review Round 14 -- DeepSeek Focus
**Reviewer**: DeepSeek (simulated by Claude Opus 4.6)
**Commit**: 8655074
**Date**: 2026-02-21
**Focus**: Async correctness, algorithmic bounds, concurrency bugs in RAG pipeline, retrieval filtering, RRF correctness, ingestion idempotency, embedding version pinning
**Spotlight**: RAG Pipeline (Dim 2) + Domain Intelligence (Dim 10) -- +1 severity

---

## Executive Summary

The codebase shows mature production engineering after 13 review rounds. The RAG pipeline is well-architected with per-item chunking, RRF reranking, SHA-256 idempotent ingestion, and version-stamp purging. Circuit breaker concurrency is handled correctly with asyncio.Lock. However, I found several issues: a concurrency race in the retriever TTL cache, missing `property_id` filtering in the Firestore retriever, an unbounded RRF document map under adversarial input, embedding model credential rotation inconsistency between `@lru_cache` and `TTLCache`, and a subtle async/sync mixing hazard in guest profile Firestore client initialization. Most findings are MEDIUM severity with one HIGH.

**Overall Score: 86/100** (+1 from R13)

---

## Dimension Scores

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | Graph Architecture | 9/10 | Excellent 11-node topology, node constants, parity assertions, feature flag dual-layer. Minimal issues. |
| 2 | **RAG Pipeline** (SPOTLIGHT) | 7/10 | Per-item chunking, RRF, version-stamp purging all solid. Retriever cache race condition, embedding `@lru_cache` vs TTLCache inconsistency, missing category filter in Firestore `retrieve_with_scores`. |
| 3 | Data Model | 8/10 | RetrievedChunk TypedDict, PropertyQAState with `_keep_max` reducer, parity checks. Guest profile bounded store. Minor: CCPA batch size unbounded. |
| 4 | API Design | 9/10 | Pure ASGI middleware, PII redaction fail-closed, streaming PII buffer with safety. |
| 5 | Testing Strategy | 8/10 | 1450+ tests, 90%+ coverage. Cannot verify specific RAG pipeline integration tests from this review. |
| 6 | Docker & DevOps | 8/10 | Not primary focus but config validators are strong. Production secret enforcement. |
| 7 | Prompts & Guardrails | 9/10 | 5-layer deterministic guardrails, 4-language coverage, semantic injection fail-closed, compliance gate priority chain well-documented. |
| 8 | Scalability & Production | 8/10 | TTL-cached singletons, semaphore backpressure, circuit breaker. Threading.Lock in async context for guest_profile is a concern. |
| 9 | Trade-off Documentation | 9/10 | Extensive inline documentation explaining every design choice. Feature flag architecture comment is exemplary. |
| 10 | **Domain Intelligence** (SPOTLIGHT) | 8/10 | Mohegan Sun data comprehensive. Area code mapping is thorough. TCPA consent hash chain is well-engineered. Minor casino domain gap in comp system awareness. |

---

## Findings

### F-001: Retriever TTL Cache Race Condition (HIGH -- SPOTLIGHT +1)

**File**: `src/rag/pipeline.py` lines 856-934
**Category**: RAG Pipeline (Dim 2)

The retriever uses a hand-rolled dict-based TTL cache (`_retriever_cache` + `_retriever_cache_time`) with NO lock protection. Under concurrent async access, two coroutines can simultaneously:

1. Both read `cache_key not in _retriever_cache` (or TTL expired)
2. Both create separate ChromaDB/Firestore retrievers
3. Both write to the cache dict, wasting resources

More critically, `time.monotonic()` checks and dict writes are NOT atomic. In a multi-coroutine environment (FastAPI under uvicorn), a coroutine could read the time, get preempted, and then another coroutine writes a new retriever and time, causing the first coroutine to overwrite with a stale retriever and reset the TTL timer.

Compare this to the LLM caches (`_llm_cache`, `_validator_cache`) which correctly use `asyncio.Lock` for coroutine safety, and the config/flag caches which use `asyncio.Lock` with double-check pattern. The retriever cache is the only singleton cache WITHOUT lock protection.

```python
# CURRENT: No lock, race-prone
def _get_retriever_cached() -> AbstractRetriever:
    cache_key = f"{settings.CASINO_ID}:default"
    now = time.monotonic()
    if cache_key in _retriever_cache:
        if (now - _retriever_cache_time.get(cache_key, 0)) < _RETRIEVER_TTL_SECONDS:
            return _retriever_cache[cache_key]
    # ... creates retriever without any lock

# FIX: Add asyncio.Lock (but note: function is sync, not async)
```

**Additional complication**: `_get_retriever_cached()` is a sync function (no `async def`), yet it is called from the sync `get_retriever()` which is called from `search_knowledge_base()` (also sync, wrapped in `asyncio.to_thread` by `retrieve_node`). Adding `asyncio.Lock` requires making it async, which cascades. The current sync design means it runs in a thread pool where `asyncio.Lock` would not work -- `threading.Lock` would be needed for the `to_thread` path. This is an architectural tension between the sync ChromaDB API and async application layer.

**Recommendation**: Either (a) use `threading.Lock` (consistent with the `to_thread` execution path), or (b) convert `get_retriever()` and its callers to async (consistent with the rest of the codebase). Option (a) is the minimal fix.

**Severity**: HIGH (data corruption unlikely but resource waste and stale retriever on credential rotation are real risks under load)

---

### F-002: Embedding Model Uses @lru_cache Without TTL (MEDIUM -- SPOTLIGHT +1)

**File**: `src/rag/embeddings.py` lines 20-39
**Category**: RAG Pipeline (Dim 2)

`get_embeddings()` uses `@lru_cache(maxsize=4)` which **never expires**. Every other singleton in the codebase uses TTLCache with 1-hour TTL for GCP credential rotation:

- `_get_llm()` -- TTLCache(ttl=3600)
- `_get_validator_llm()` -- TTLCache(ttl=3600)
- `_get_circuit_breaker()` -- TTLCache(ttl=3600)
- `_get_retriever_cached()` -- manual TTL check (3600s)

The embedding model client holds a reference to the `GOOGLE_API_KEY` credential. Under GCP Workload Identity Federation, credentials rotate. The `@lru_cache` will hold onto the stale credential indefinitely, causing embedding calls to fail after credential rotation until the process is restarted.

This is the exact bug pattern documented in the project's own rules (`~/.claude/rules/rag-production.md`): "Replace `@lru_cache(maxsize=1)` with `cachetools.TTLCache(maxsize=1, ttl=3600)` for LLM client singletons when using GCP Workload Identity."

```python
# CURRENT: Never expires
@lru_cache(maxsize=4)
def get_embeddings(task_type: str | None = None) -> GoogleGenerativeAIEmbeddings:

# SHOULD: TTLCache with 1-hour refresh
_embeddings_cache: TTLCache = TTLCache(maxsize=4, ttl=3600)
```

**Severity**: MEDIUM (affects production credential rotation; currently mitigated by single-container deployment where process restarts are frequent enough)

---

### F-003: Firestore Retriever Missing Category Filter in retrieve_with_scores (MEDIUM -- SPOTLIGHT +1)

**File**: `src/rag/firestore_retriever.py` lines 130-158
**Category**: RAG Pipeline (Dim 2)

`FirestoreRetriever.retrieve_with_scores()` does NOT accept or apply a `filter_category` parameter, while its sibling `CasinoKnowledgeRetriever.retrieve_with_scores()` also lacks it but `CasinoKnowledgeRetriever.retrieve()` supports it via a ChromaDB `$and` filter clause (lines 801-812).

The `FirestoreRetriever.retrieve()` method does accept `filter_category` but applies it as a post-hoc Python filter (line 182), NOT as a Firestore query predicate. This means:

1. Firestore returns `top_k * 2` documents without category filtering
2. Python filters them after retrieval
3. If 80% of documents are category X and you want category Y, you might get 0 results from a `top_k=5` search even though category Y documents exist further down the ranking

The `_single_vector_query` already does post-hoc `property_id` filtering (lines 104-105), so adding `category` to the same filter would be trivial. Alternatively, Firestore's `find_nearest` supports composite filters.

**Severity**: MEDIUM (affects retrieval precision for category-specific queries in production Firestore path)

---

### F-004: RRF Document Map Unbounded Under Adversarial Input (LOW)

**File**: `src/rag/reranking.py` lines 40-53
**Category**: RAG Pipeline (Dim 2)

`rerank_by_rrf()` builds `doc_map` and `rrf_scores` dicts with no upper bound on size. While normal operation produces `2 * top_k` entries (two result lists of `top_k` each with some overlap), the function signature accepts `list[list[tuple]]` with no constraint on list count or list size.

If called with many result lists or very large `top_k` values, the SHA-256 hashing per document (line 45-47) and sorting (line 52) could become expensive. The SHA-256 computation on `page_content` is O(n * content_length) where n is total documents across all lists.

In current usage (always 2 lists of `top_k=5`), this is a non-issue. But the function's generic interface does not enforce bounds, making it a latent risk if usage expands.

**Recommendation**: Add a `max_input_docs` parameter with a reasonable default (e.g., 100) and log a warning if exceeded.

**Severity**: LOW (not exploitable in current usage; defensive hardening)

---

### F-005: threading.Lock in Async Context for Firestore Client Init (MEDIUM)

**File**: `src/data/guest_profile.py` lines 57-105 and `src/casino/config.py` lines 174-224
**Category**: Scalability & Production (Dim 8)

Both `guest_profile._get_firestore_client()` and `casino.config._get_firestore_client()` use `threading.Lock` for client singleton initialization:

```python
_firestore_client_lock = threading.Lock()

def _get_firestore_client() -> Any | None:
    cached = _firestore_client_cache.get("client")
    if cached is not None:
        return cached
    with _firestore_client_lock:  # <-- blocks the event loop thread
        ...
```

In the FastAPI async application, if `_get_firestore_client()` is called from an async context (which it is -- from `get_guest_profile()` which is `async def`), the `threading.Lock().acquire()` call will block the event loop thread. Under high concurrency during cold start (when the cache is empty and multiple async requests arrive simultaneously), this blocks ALL coroutines on that event loop, not just the one waiting for the lock.

The code uses double-check locking correctly, so the lock is only held during the first client creation. After that, the fast path (line 78-79) returns the cached client without acquiring the lock. So this is a cold-start-only issue.

However, `AsyncClient()` construction (line 95) involves network I/O (service account credential fetching), so the lock-hold duration could be significant (100ms+).

The other cache implementations in the codebase (LLM, config, feature flags) correctly use `asyncio.Lock` for the same pattern.

**Recommendation**: Convert to `asyncio.Lock` if the function is only called from async contexts, or use a non-blocking approach. Note the duplicated code pattern -- both files have near-identical `_get_firestore_client()` implementations with the same issue.

**Severity**: MEDIUM (cold-start only, but can cause request pileup during container initialization under load)

---

### F-006: property_id Derivation Inconsistency Between Ingestion and Retrieval (MEDIUM -- SPOTLIGHT +1)

**File**: `src/rag/pipeline.py` lines 429, 492, 683, 800, 844 and `src/rag/firestore_retriever.py` line 151
**Category**: RAG Pipeline (Dim 2)

`property_id` is derived from `settings.PROPERTY_NAME.lower().replace(" ", "_")` in multiple locations:

- `_load_property_json()` line 429: `settings.PROPERTY_NAME.lower().replace(" ", "_")`
- `_load_knowledge_base_markdown()` line 492: same
- `ingest_property()` line 683: same
- `CasinoKnowledgeRetriever.retrieve()` line 800: same
- `CasinoKnowledgeRetriever.retrieve_with_scores()` line 844: same
- `FirestoreRetriever.retrieve_with_scores()` line 151: same

Meanwhile, `reingest_item()` line 296 uses:
```python
property_id = (casino_id or settings.CASINO_ID).lower().replace(" ", "_")
```

This is `CASINO_ID` (default `"mohegan_sun"`), NOT `PROPERTY_NAME` (default `"Mohegan Sun"`).

If `CASINO_ID` and `PROPERTY_NAME.lower().replace(" ", "_")` produce different values (e.g., CASINO_ID = "mohegan_sun_ct", PROPERTY_NAME = "Mohegan Sun"), then:
- Bulk ingestion stamps `property_id = "mohegan_sun"` (from PROPERTY_NAME)
- CMS re-ingestion stamps `property_id = "mohegan_sun_ct"` (from CASINO_ID)
- Retrieval filters by `property_id = "mohegan_sun"` (from PROPERTY_NAME)
- CMS-updated documents become invisible to retrieval

With the current defaults (`CASINO_ID="mohegan_sun"`, `PROPERTY_NAME="Mohegan Sun"`), both produce `"mohegan_sun"`, so the bug is latent. It will manifest when a second casino is onboarded with different naming.

**Recommendation**: Use a single canonical source for `property_id` derivation. Either always use `CASINO_ID` or always use `PROPERTY_NAME.lower().replace(" ", "_")`. Extract to a helper function to enforce consistency.

**Severity**: MEDIUM (latent bug, not triggered with current single-tenant config but a multi-tenant time bomb)

---

### F-007: Ingestion Version Purge Uses Private _collection API (LOW)

**File**: `src/rag/pipeline.py` lines 700-721
**Category**: RAG Pipeline (Dim 2)

The stale chunk purge logic accesses `vectorstore._collection` (line 700), which is a private attribute of LangChain's Chroma wrapper:

```python
collection = vectorstore._collection
old_docs = collection.get(
    where={"$and": [
        {"property_id": {"$eq": property_id}},
        {"_ingestion_version": {"$ne": version_stamp}},
    ]}
)
```

This bypasses the LangChain abstraction and directly uses the chromadb Collection API. While this works reliably with the current pinned `langchain-community` version, it is fragile against LangChain updates that may rename or restructure the internal `_collection` attribute.

The code correctly wraps this in a try/except with a non-critical warning, so breakage would only disable purging (not crash ingestion). This is acceptable defensive coding.

**Severity**: LOW (non-critical path with graceful degradation)

---

### F-008: SSE Streaming Node Lifecycle Events May Fire Multiple Times (LOW)

**File**: `src/agent/graph.py` lines 619-623
**Category**: Graph Architecture (Dim 1)

The `on_chain_start` event handler uses `langgraph_node not in node_start_times` as a deduplication guard:

```python
if (
    kind == "on_chain_start"
    and langgraph_node in _KNOWN_NODES
    and langgraph_node not in node_start_times
):
    node_start_times[langgraph_node] = time.monotonic()
```

However, in the RETRY path (validate -> generate), the `generate` node executes a second time. The first execution's start time was popped from `node_start_times` on `on_chain_end` (line 689), so the second execution's `on_chain_start` will fire correctly. This is actually fine.

But if `on_chain_end` fires for `generate` (first execution) AFTER `on_chain_start` fires for `generate` (second execution) due to event ordering in `astream_events`, the second start time would be overwritten. LangGraph's `astream_events` v2 guarantees ordered delivery per node, but cross-node event interleaving is not guaranteed.

Practically, this would only cause slightly inaccurate `duration_ms` reporting for the retry case, not a functional bug.

**Severity**: LOW (observability-only impact, edge case in retry path)

---

### F-009: CCPA Batch Delete Not Bounded by Firestore 500-Op Limit (LOW)

**File**: `src/data/guest_profile.py` lines 288-329
**Category**: Data Model (Dim 3)

The CCPA cascade delete adds all subcollection documents to a single batch:

```python
batch = db.batch()
ops_count = 0
# ... iterates all conversations, messages, signals, audit entries
await batch.commit()
```

The code comments note "Firestore batch limit is 500 operations; cascade unlikely to exceed this for a single guest profile." This is a reasonable assumption for typical profiles, but a pathological case (guest with 100+ conversations, each with multiple messages) could exceed 500 operations.

Firestore batch operations that exceed 500 will raise `google.api_core.exceptions.InvalidArgument`. The outer try/except catches this and re-raises, which is correct (the CCPA delete should NOT silently succeed partially). However, the error message would be confusing ("InvalidArgument" does not clearly indicate "batch too large").

**Recommendation**: Add a counter check before commit: if `ops_count > 450`, split into multiple batches. Or add a comment documenting the 500-op limit with an explicit check.

**Severity**: LOW (unlikely for normal profiles; would fail loudly rather than silently)

---

### F-010: get_settings() Uses @lru_cache -- Stale After Environment Change (LOW)

**File**: `src/config.py` line 180-183
**Category**: Scalability & Production (Dim 8)

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

`get_settings()` uses `@lru_cache` which never expires. Since `Settings` reads from environment variables at construction time, any runtime environment variable changes (e.g., updating `MODEL_NAME` or `CASINO_ID` via Cloud Run revision) will not be picked up until the process restarts.

This is documented behavior and consistent with Pydantic Settings' design (read-once at startup). However, it means that `get_settings()` cannot be refreshed via the TTL mechanism used by other singletons.

This is a known trade-off and not a bug -- I'm noting it for completeness as it contrasts with the 1-hour TTL refresh pattern used elsewhere.

**Severity**: LOW (documented design trade-off, consistent with Pydantic Settings semantics)

---

## Positive Observations

1. **Parity assertions at import time** (graph.py lines 498-505, feature_flags.py lines 73-87): Catching state schema drift and feature flag drift before any request is served is excellent. The `ValueError` (not `assert`) survives `python -O`.

2. **Degraded-pass validation** (nodes.py lines 337-365): The first-attempt/retry-attempt asymmetry is well-reasoned and well-documented. This is exactly right for the availability/safety trade-off.

3. **Streaming PII redactor** (streaming_pii.py): The lookahead buffer design is correct. Operating on the redacted buffer for the safe/lookahead split (line 118-119) prevents misalignment from length-changing substitutions. Intentionally NOT flushing on `CancelledError` (line 704-712) is the right safety decision.

4. **Circuit breaker `record_cancellation()`** (circuit_breaker.py lines 208-229): Correctly distinguishing client disconnects from LLM failures prevents false circuit breaker trips under normal SSE traffic.

5. **Compliance gate priority chain** (compliance_gate.py): The ordering rationale (injection before content guardrails, semantic classifier after deterministic guardrails) is rigorously argued with concrete attack scenarios.

6. **DRY specialist base** (agents/_base.py): Dependency injection via `get_llm_fn` and `get_cb_fn` preserves testability without monkey-patching. The sliding window history with retry exclusion (line 144-148) prevents the LLM from parroting invalid responses.

7. **Consent hash chain** (compliance.py): HMAC-SHA256 with chained previous hashes provides both tamper-evidence and authentication. The `verify_chain()` method re-derives every hash from inputs, catching any retroactive modification.

---

## Summary

| Severity | Count | Findings |
|----------|-------|----------|
| CRITICAL | 0 | -- |
| HIGH | 1 | F-001 (retriever cache race) |
| MEDIUM | 4 | F-002 (embedding lru_cache), F-003 (Firestore category filter), F-005 (threading.Lock in async), F-006 (property_id inconsistency) |
| LOW | 5 | F-004 (RRF unbounded), F-007 (private _collection), F-008 (SSE lifecycle), F-009 (CCPA batch), F-010 (settings lru_cache) |
| **Total** | **10** | |

**Score: 86/100**

The codebase continues to improve. The RAG pipeline is architecturally sound (per-item chunking, RRF, version-stamp purging, idempotent IDs) but has several consistency gaps that would surface in multi-tenant production (F-001, F-002, F-006). The highest-priority fix is F-001 (retriever cache thread safety) as it affects the core retrieval path under concurrent load.
