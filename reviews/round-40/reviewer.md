# R40 Deep Review: D2 RAG + D5 Testing + D8 Scalability

Reviewer: reviewer (Opus 4.6)
Cross-validated: GPT-5.2 Codex (performance), Gemini 3 Pro (scalability numbers)
Commit: 914dbd7 | Date: 2026-02-23

---

## D5 Testing Strategy — Score: 6.5 -> 6.0

### Test Suite Status

- **2101 passed, 52 failed**, 64 warnings in 315.97s
- **52 test failures** in test_api.py, test_config.py, test_phase2_integration.py, test_phase4_integration.py
- All 52 failures are in API/integration tests returning HTTP 401 instead of expected status codes
- Root cause: `ApiKeyMiddleware` now protects `/property`, `/graph`, `/feedback` endpoints, but test helpers (`_make_test_app()`) don't set API key headers
- **This is a CRITICAL regression** — 52 tests have been broken by middleware changes and no one noticed

### Coverage Analysis

- **Total coverage: 89.88%** — BELOW the 90% CI gate threshold (`--cov-fail-under=90`)
- CI pipeline is CURRENTLY FAILING on every push (cloudbuild.yaml Step 1 enforces `--cov-fail-under=90`)

### Coverage Gaps by Module (lines missed)

| Module | Coverage | Uncovered Lines | Criticality |
|--------|----------|-----------------|-------------|
| `src/data/guest_profile.py` | **56%** | 71 lines (Firestore CRUD, profile merge) | HIGH — entire Firestore guest profile path untested |
| `src/observability/langfuse_client.py` | **55%** | 28 lines (Langfuse initialization, trace creation) | MEDIUM |
| `src/state_backend.py` | **76%** | 28 lines (RedisBackend entirely, sweep batch) | MEDIUM — Redis backend has zero test coverage |
| `src/rag/pipeline.py` | **88%** | 49 lines (reingest_item error paths, Firestore retriever creation fallback) | MEDIUM |
| `src/rag/embeddings.py` | **90%** | 3 lines (health check failure path, lines 82-90) | LOW-MEDIUM — R39's embedding health check is untested |

### Findings

#### D5-C001 [CRITICAL] — 52 API tests broken by middleware change
- **File**: `tests/test_api.py`, `tests/test_phase2_integration.py`, `tests/test_phase4_integration.py`
- **Symptom**: All return HTTP 401 instead of expected codes
- **Root cause**: `ApiKeyMiddleware._PROTECTED_PATHS` now includes `/property`, `/graph`, `/feedback`. Test helper `_make_test_app()` doesn't set API_KEY env var or pass `X-API-Key` header.
- **Impact**: CI pipeline is failing. The 90% coverage gate is also failing (89.88%).
- **Fix**: Either (a) set `API_KEY=""` in test env to disable auth, or (b) add `X-API-Key` header to test requests.

#### D5-C002 [CRITICAL] — CI pipeline is broken
- The coverage gate (`--cov-fail-under=90`) fails at 89.88%.
- Combined with 52 test failures, **every push to main triggers a failed pipeline**.
- This means the latest code on main has NOT been deployed — the smoke test step never runs.

#### D5-M001 [MAJOR] — guest_profile.py at 56% coverage
- **File**: `src/data/guest_profile.py` lines 87-341
- Entire Firestore CRUD path (get_guest_profile, update_guest_profile, merge logic) is untested.
- The Firestore client initialization (async lock, double-check pattern) at lines 86-112 is untested.
- **Risk**: Guest profile feature is wired but effectively unvalidated in production.

#### D5-M002 [MAJOR] — RedisBackend has zero test coverage
- **File**: `src/state_backend.py` lines 158-198
- The entire `RedisBackend` class (increment, get_count, set, get, exists, delete) has 0% coverage.
- While InMemoryBackend is the current default, the Redis path exists and could be enabled via config.

#### D5-M003 [MAJOR] — R39 embedding health check untested
- **File**: `src/rag/embeddings.py` lines 82-90
- The health check added in R39 (embed_query("health check") before caching) has no test verifying:
  - That a broken client is NOT cached
  - That the exception is re-raised
  - That subsequent calls retry creation
- This is a safety-critical path — a broken embedding client cached for 1 hour silently degrades ALL retrieval.

#### D5-M004 [MAJOR] — No test for RRF with one empty strategy
- **File**: `tests/test_rag.py` `TestReciprocalRankFusion`
- Tests cover: empty input `[]`, single list, duplicate across lists, top_k, different sources.
- Missing: `rerank_by_rrf([[doc_a], []], top_k=5)` — one strategy returns results, other returns empty.
- The code handles this correctly (verified manually), but the edge case is not regression-protected.

#### D5-M005 [MAJOR] — Property-based tests exist but don't cover state reducers deeply
- `test_state_parity.py` has property-based tests for `_merge_dicts` (identity, None-filtering, empty-string-filtering) and `_keep_max` (commutativity, idempotence).
- **Missing**: No property-based test for `_merge_dicts` associativity: `merge(merge(a, b), c) == merge(a, merge(b, c))`. This matters because extracted_fields accumulates across 3+ turns.
- **Missing**: No property-based test for `_replace_or_keep` reducer (used by guest_sentiment, whisper_plan).

#### D5-m001 [MINOR] — 2 `test_config.py` failures
- `test_production_rejects_empty_api_key` and `test_development_allows_empty_secrets` fail (likely env contamination from other tests).

#### D5-m002 [MINOR] — `isinstance` assertions in 19 tests
- Tests like `assert isinstance(result, str)` are weak — they verify type but not content. Not tautological per se, but low-value assertions when used as the primary check.

---

## D2 RAG Pipeline — Score: 7.0 -> 7.0

### Architecture Summary

- Per-item chunking with 8 category-specific formatters (restaurants, entertainment, hotel, gaming, faq, amenities, promotions + generic fallback)
- SHA-256 content hashing for idempotent ingestion (null-byte delimiter, R36 fix)
- Version-stamp purging for stale chunks
- Dual-strategy retrieval (semantic + entity-augmented) with RRF fusion (k=60)
- ChromaDB local dev / Firestore prod with AbstractRetriever interface
- Embedding model pinned (`gemini-embedding-001`) with task_type differentiation (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY)

### Findings

#### D2-M001 [MAJOR] — No retry logic for embedding API errors during ingestion
- **File**: `src/rag/pipeline.py` `ingest_property()` lines 737-748
- `get_embeddings(task_type="RETRIEVAL_DOCUMENT")` is called once. If the embedding API returns 503 mid-batch, `Chroma.from_texts()` raises and the entire ingestion fails.
- The `get_embeddings()` function has a health check, but `Chroma.from_texts()` itself calls `embed_documents()` internally with no retry wrapper.
- **Impact**: A transient embedding API outage during startup causes the container to start with empty RAG — all queries get no-context fallback responses for the container's lifetime.
- **Fix**: Wrap `Chroma.from_texts()` in a retry with exponential backoff (max 3 attempts), or pre-embed in batches with per-batch retry.

#### D2-M002 [MAJOR] — reingest_item() has no embedding retry
- **File**: `src/rag/pipeline.py` `reingest_item()` lines 282-396
- CMS webhook triggers `reingest_item()` which calls `retriever.vectorstore.add_texts()`. If the embedding API is temporarily down, the CMS update is silently lost (`return False`).
- No retry, no dead-letter queue, no webhook retry mechanism.
- **Impact**: Content updates from the CMS can be silently dropped during embedding API outages.

#### D2-M003 [MAJOR] — Firestore retriever `_use_server_filter` is instance-level state
- **File**: `src/rag/firestore_retriever.py` line 69
- `self._use_server_filter = True` starts optimistic but permanently flips to `False` on first composite index error.
- The retriever is cached for 1 hour via TTLCache. If the composite index is created 5 minutes after startup, the retriever won't use it until the TTL expires.
- Not a bug per se (it degrades gracefully), but **the permanent flip is pessimistic** — it should periodically retry server-side filtering.

#### D2-m001 [MINOR] — No max latency SLA documented for retrieval
- `RETRIEVAL_TIMEOUT` is 10s (configurable). Good.
- No observability metric tracking actual retrieval P50/P95/P99 latency. The timeout catches hung queries but doesn't measure normal-case performance.
- Suggest: Log `duration_ms` in `retrieve_node` for monitoring dashboard.

#### D2-m002 [MINOR] — Markdown chunking splits on `## ` but not `### `
- **File**: `src/rag/pipeline.py` line 563
- `re.split(r"\n(?=## )", text)` splits on `## ` headings but not `### `. A long `## ` section with many `### ` subsections becomes one large chunk that may exceed `chunk_size`.
- The fallback `RecursiveCharacterTextSplitter` handles this, but logs a warning ("structured context may be fragmented").
- Low impact: current knowledge-base markdown files are relatively short.

#### D2-m003 [MINOR] — `_FORMATTERS` dict uses string keys without validation
- Category keys from JSON data are matched against `_FORMATTERS` dict. A typo in the JSON file (e.g., "resturants") silently falls back to `_format_generic`. No validation or warning for unknown categories.

### Strengths (D2)
- Per-item chunking is excellent for structured data — avoids the text-splitter boundary destruction problem.
- RRF fusion with null-byte delimiter for document identity is production-grade.
- Version-stamp purging correctly handles the ghost data problem.
- The `AbstractRetriever` interface enables clean backend swapping.
- Cosine distance normalization (`1 - distance/2`) is correct for Firestore COSINE distance range [0,2].

---

## D8 Scalability & Production — Score: 7.0 -> 6.5

### Configuration Summary

```
Cloud Run: --memory=2Gi, --cpu=2, --concurrency=50, --max-instances=10
           --min-instances=1, --cpu-boost, --timeout=180s
LLM Semaphore: asyncio.Semaphore(20) per instance
Rate Limit: 20 req/min per IP, in-memory, per-instance
Circuit Breaker: per-process, TTLCache(maxsize=1, ttl=3600)
Checkpointer: BoundedMemorySaver(max_threads=1000) dev / FirestoreSaver prod
```

### Theoretical Capacity

- **Max concurrent SSE connections**: 50 per instance x 10 instances = 500
- **Max LLM calls**: 20 (semaphore) x 10 instances = 200 concurrent, vs Gemini Flash 300 RPM = 67% safety margin
- **Memory per SSE connection**: ~320-500 KB (state + generator + context managers)
- **Memory at max concurrency**: 50 x 500 KB = ~25 MB (well within 2Gi)
- **Rate limit actual enforcement**: 20 req/min x 10 instances = 200 req/min effective (10x leak)
- **Firestore cost at 1000 req/day**: ~$0.00 (within free tier)

### Findings

#### D8-C001 [CRITICAL] — Synchronized TTL expiry causes periodic latency spike
- **Files**: `src/agent/nodes.py` (_llm_cache, _validator_cache), `src/rag/pipeline.py` (_retriever_cache), `src/rag/embeddings.py` (_embeddings_cache), `src/agent/circuit_breaker.py` (_cb_cache), `src/config.py` (_settings_cache)
- **All 6 TTLCache instances use identical TTL=3600s**.
- On container startup, all caches are populated within ~2 seconds of each other.
- **Exactly 3600s later**, all 6 caches expire within a 2-second window.
- At that moment, 50 concurrent requests all contend for 6 different locks to recreate 6 different clients.
- GPT-5.2 Codex confirms: "Lock held during heavy construction... a classic thundering herd amplification."
- Gemini 3 Pro estimates: ~1.5s worst-case per cache reconstruction with double-checked locking (which IS implemented). But 6 sequential cache misses = up to ~9s cumulative delay for the first request that triggers all reconstructions.
- **Fix**: Add TTL jitter. Change `TTLCache(maxsize=1, ttl=3600)` to `TTLCache(maxsize=1, ttl=3600 + random.randint(-300, 300))` across all singletons. Spreads reconstruction over a 10-minute window.

#### D8-M001 [MAJOR] — Rate limiter is per-instance (10x multiplier under scaling)
- **File**: `src/api/middleware.py` `RateLimitMiddleware`
- Already documented in class docstring (ADR with 3-tier upgrade path). But:
- With `max-instances=10`, effective rate limit is 200 req/min, not 20.
- A malicious actor targeting LLM budget can burn $0.30/1M tokens x 200 req/min x 2K output tokens = significant cost.
- The class docstring says "acceptable for demo" — this is correct, but **the ADR lacks a timeline or trigger condition** for when to upgrade.
- **Fix**: Add a concrete trigger: "Upgrade to Cloud Armor when daily traffic exceeds 1000 requests OR before any paid client deployment."

#### D8-M002 [MAJOR] — No backpressure between SSE concurrency (50) and LLM semaphore (20)
- **File**: `src/agent/agents/_base.py` `_LLM_SEMAPHORE = asyncio.Semaphore(20)`
- Cloud Run allows 50 concurrent requests. LLM semaphore allows 20.
- Result: 30 users sit in memory holding SSE connections open, waiting for the semaphore.
- These 30 waiting connections consume memory, hold heartbeat timers, and appear active to the client (SSE connection is open, pings are sent).
- The heartbeat (`_HEARTBEAT_INTERVAL = 15s`) keeps them alive, but the user sees a long delay before any content appears.
- **Impact**: Under load, 60% of users experience degraded latency (waiting for semaphore) with no feedback beyond heartbeat pings.
- **Fix**: Consider returning a "high load" SSE event when semaphore acquisition takes >5s, or reduce Cloud Run concurrency to match LLM capacity (e.g., `--concurrency=25`).

#### D8-M003 [MAJOR] — _get_retriever_cached() holds threading.Lock during Firestore/ChromaDB construction
- **File**: `src/rag/pipeline.py` lines 987-1047
- Retriever construction (Firestore client init, ChromaDB loading) happens INSIDE `_retriever_lock`.
- This function is called from `asyncio.to_thread()` (thread pool). During the lock hold, ALL other thread pool workers calling `_get_retriever_cached()` are blocked.
- With `asyncio.to_thread()` default thread pool (40 workers in Python 3.12), a 2-second Firestore init blocks 39 other retrieval calls.
- The lock-free fast path (R39 fix M-002) mitigates the common case, but on TTL expiry, the slow path is triggered.
- **Fix**: Build retriever outside the lock (in-flight sentinel pattern as suggested by GPT-5.2 Codex).

#### D8-m001 [MINOR] — Memory over-provisioned
- At max concurrency (50 SSE connections), memory usage is ~275 MB (base) + 25 MB (connections) = ~300 MB.
- Cloud Run allocation is 2Gi. Utilization: ~15%.
- Could safely reduce to 1Gi or even 512Mi for cost savings.
- Counter-argument: 2Gi provides headroom for ChromaDB ingestion during startup (development mode) and for future features.

#### D8-m002 [MINOR] — No graceful degradation signal for semaphore contention
- When the LLM semaphore is full, requests silently queue. No metric tracks semaphore wait time or queue depth.
- Suggest: Log `semaphore_wait_ms` in `_base.py` for monitoring.

#### D8-m003 [MINOR] — BoundedMemorySaver eviction is O(n) on thread storage
- **File**: `src/agent/memory.py` line 87-91
- `keys_to_remove = [k for k in self._inner.storage if ...]` scans all storage keys on each eviction.
- At 1000 threads with ~10 keys each, this is a 10,000-iteration scan per eviction.
- Low impact at current scale, but O(n) per eviction is a concern at higher thread counts.

### Strengths (D8)
- Double-checked locking pattern consistently applied across all singletons.
- Lock-free fast path for retriever cache (R39) eliminates common-case contention.
- BoundedMemorySaver with LRU eviction prevents OOM from long-running dev sessions.
- CloudBuild pipeline with smoke test + rollback is production-grade.
- Proper separation of startup probe (/health) vs liveness probe (/live) prevents CB-induced instance cycling.
- Non-root Docker user, pinned base image by SHA digest, Trivy vulnerability scanning.

---

## Score Summary

| Dimension | R39 Score | R40 Score | Delta | Key Findings |
|-----------|-----------|-----------|-------|--------------|
| D5 Testing | 6.5 | **6.0** | -0.5 | 52 broken tests (CRITICAL), CI pipeline failing, 89.88% < 90% gate, guest_profile 56% coverage |
| D2 RAG | 7.0 | **7.0** | 0.0 | No embedding retry on ingestion (MAJOR), architecture is solid, version-stamp purging works |
| D8 Scalability | 7.0 | **6.5** | -0.5 | Synchronized TTL thundering herd (CRITICAL), semaphore/concurrency mismatch, per-instance rate limiting documented but unresolved |

## Finding Counts

| Severity | D2 | D5 | D8 | Total |
|----------|-----|-----|-----|-------|
| CRITICAL | 0 | 2 | 1 | **3** |
| MAJOR | 3 | 5 | 3 | **11** |
| MINOR | 3 | 2 | 3 | **8** |
| **Total** | 6 | 9 | 7 | **22** |

## Priority Fix Order

1. **D5-C001 + D5-C002**: Fix 52 broken API tests (unblock CI pipeline)
2. **D8-C001**: Add TTL jitter to all 6 singleton caches
3. **D5-M001**: Add tests for guest_profile.py Firestore CRUD
4. **D5-M003**: Add test for embedding health check failure path
5. **D8-M002**: Add semaphore wait logging / reduce concurrency to 25
6. **D2-M001**: Add retry logic for embedding API during ingestion
7. **D8-M003**: Move retriever construction outside lock (sentinel pattern)
8. **D5-M004**: Add RRF edge case test (one empty strategy)
