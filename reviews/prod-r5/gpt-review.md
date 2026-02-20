# Round 5 Production Review — GPT-5.2

**Date**: 2026-02-20
**Reviewer**: GPT-5.2 (Azure AI Foundry)
**Spotlight**: SCALABILITY & ASYNC PATTERNS
**Base Commit**: 9655fd2
**Stats**: 1157 tests, 91.27% coverage

---

## Score Table (10 Standard Dimensions)

| # | Dimension | Score | Notes |
|---|-----------|:-----:|-------|
| 1 | Graph/Agent Architecture | 7 | Custom StateGraph with specialist dispatch is well-designed; validation loop is solid |
| 2 | RAG Pipeline | 7 | Per-item chunking, SHA-256 idempotent IDs, version-stamp purging all correct |
| 3 | Data Model / State Design | 5 | Guest profile in-memory fallback is dangerous at scale; MemorySaver fallback in prod |
| 4 | API Design | 6 | Pure ASGI middleware is correct; rate limiter per-container and lock-serialized |
| 5 | Testing Strategy | 7 | 1157 tests at 91.27% is strong; concurrency tests added in R4 |
| 6 | Docker & DevOps | 6 | Cloud Run deployment ready; startup ingestion in container is risky |
| 7 | Prompts & Guardrails | 7 | 5-layer deterministic guardrails; structured output routing; degraded-pass validated |
| 8 | Scalability & Production | 4 | **SPOTLIGHT** — Multiple correctness-critical behaviors in per-container memory |
| 9 | Documentation & Code Quality | 7 | Extensive docstrings; honest vocabulary; trade-offs documented |
| 10 | Domain Intelligence | 7 | Casino domain depth; responsible gaming; multilingual patterns |
| **Total** | | **63** | |

**Previous**: R1=67.3, R2=61.3, R3=60.7, R4=66.7
**Current**: R5-GPT=63

---

## Findings

### F1: CRITICAL — Production can silently fall back to in-memory checkpointing (state loss on restart/scale-out)

**File**: `src/agent/memory.py`
**Lines**: 17-56
**Issue**: `get_checkpointer()` catches ALL exceptions from FirestoreSaver creation and falls back to `MemorySaver()`, even in production. The only signal is a `logger.warning()`. This means a transient Firestore config error silently makes ALL conversation state ephemeral.
**Impact**: Under Cloud Run scale-out, different containers maintain independent conversation histories. On container recycle (idle-to-zero, deploy, OOM), ALL conversation threads are lost. Users experience "the bot forgot everything" with no error signal. This is a correctness bug, not just a scalability issue.
**Fix**: In production (`ENVIRONMENT != "development"`), raise the exception instead of falling through to MemorySaver. Add a startup health check that validates the checkpointer backend can connect. Only allow MemorySaver in dev/test.

### F2: CRITICAL — In-memory guest profile store causes cross-instance inconsistency

**File**: `src/data/guest_profile.py`
**Lines**: 33, 107-226
**Issue**: `_memory_store: dict[str, dict] = {}` is the fallback when Firestore is unavailable. This module-level dict has no TTL, no eviction, no size bound, and no concurrency guard. Profiles written to one container are invisible to all others.
**Impact**: (1) Guest profile updates diverge across Cloud Run instances — guest preferences learned on instance A are unknown on instance B. (2) Memory grows unbounded with unique `(casino_id, phone)` keys. (3) Container restart loses all accumulated profile data. (4) CCPA delete on instance A doesn't propagate to instance B's in-memory copy.
**Fix**: In production, fail fast if Firestore is unavailable rather than silently using in-memory fallback. Add `maxsize` bound if the fallback must exist for dev. Add async lock around `_memory_store` mutations (currently unprotected).

### F3: HIGH — Rate limiting is per-container and serialized behind a single global lock

**File**: `src/api/middleware.py`
**Lines**: 284-391
**Issue**: `RateLimitMiddleware` uses a single `asyncio.Lock()` guarding the entire `_requests` OrderedDict. Every `/chat` and `/feedback` request contends on this one lock. Additionally, rate limit state is entirely in-memory — each Cloud Run container maintains independent counters.
**Impact**: (1) Throughput collapse under concurrency — all requests serialize through one lock for the rate check. (2) Effective rate limit multiplies by container count: with 5 containers, a client gets 5x the intended rate (100 req/min instead of 20). (3) Lock contention shows up as P99 latency spikes.
**Fix**: For fleet-global rate limiting, use Cloud Armor rate limiting or a shared Redis/Memorystore counter. If keeping in-process, shard locks by IP hash (e.g., 16 lock stripes) to reduce contention. Minimize time under lock to O(1) operations only.

### F4: HIGH — Circuit breaker state is per-container; cannot protect shared LLM dependency fleet-wide

**File**: `src/agent/circuit_breaker.py`
**Lines**: 40-270
**Issue**: Circuit breaker is an in-memory `lru_cache(maxsize=1)` singleton. Each Cloud Run container maintains independent failure tracking. The code acknowledges this in comments ("acceptable for per-container health detection") but the architecture doc doesn't reflect this limitation.
**Impact**: If the Gemini API is degraded, container A may trip its breaker (5 failures) while containers B-E continue hammering the failing dependency. The circuit breaker cannot protect the downstream service at fleet level. Under auto-scaling, new containers start with a fresh breaker, flooding a recovering dependency.
**Fix**: (1) Document that CB is per-container and adjust threshold accordingly (lower threshold = faster per-instance protection). (2) For fleet-wide protection, use a shared backend (Redis atomic counters) or rely on the upstream service's own throttling (429 responses). (3) Add CB state to the `/health` endpoint (already done) so Cloud Run can route away from open-breaker instances.

### F5: HIGH — Firestore AsyncClient cached without lifecycle management or creation guard

**File**: `src/data/guest_profile.py`
**Lines**: 45-84
**File**: `src/casino/config.py`
**Lines**: 166-197
**Issue**: Two separate `_get_firestore_client()` functions cache AsyncClient instances in module-level dicts without: (a) an async lock to prevent concurrent creation races, (b) shutdown/close lifecycle management, (c) health check or reconnection logic. Two concurrent requests during cold start can race and create two clients, with only the last one being cached.
**Impact**: Connection/resource leakage over long-lived containers. Under bursty cold starts, duplicate clients waste resources. No graceful teardown means pending Firestore operations may be lost on shutdown.
**Fix**: Create Firestore clients during FastAPI lifespan and store on `app.state`. Add `await client.close()` in the lifespan shutdown block. Use an `asyncio.Lock` around client creation to prevent race conditions. Consider a single shared client across both modules.

### F6: HIGH — Unbounded casino config and feature flag caches can grow without limit

**File**: `src/casino/config.py`
**Lines**: 23-24
**File**: `src/casino/feature_flags.py`
**Lines**: 85-86
**Issue**: `_config_cache` and `_flag_cache` are plain dicts with TTL semantics but no maxsize cap. The TTL check only prevents serving stale data; expired entries are never evicted (they're overwritten on next read but never cleaned up if a casino_id is never queried again). No lock protects concurrent read-modify-write operations on these dicts.
**Impact**: (1) If `casino_id` has high cardinality (future multi-tenant), memory grows linearly. (2) Concurrent requests for the same expired casino_id can cause a "thundering herd" — multiple Firestore reads for the same config. (3) In async context, dict mutation during iteration is undefined behavior.
**Fix**: Replace with `cachetools.TTLCache(maxsize=N)` for bounded TTL caching. Add an async lock (or accept eventual consistency for read paths). Validate `casino_id` against an allowlist.

### F7: MEDIUM — LLM singleton creation blocks all concurrent requests during TTL refresh

**File**: `src/agent/nodes.py`
**Lines**: 116-175
**File**: `src/agent/whisper_planner.py`
**Lines**: 47-75
**Issue**: `_get_llm()`, `_get_validator_llm()`, and `_get_whisper_llm()` all hold a global `asyncio.Lock` while constructing the LLM client on cache miss. TTLCache with `ttl=3600` means every hour ALL concurrent requests serialize behind the lock while one request constructs the client. Separate locks per client type is good (prevents cross-blocking), but each individual lock still serializes all users of that client.
**Impact**: P99 latency spike every hour at TTL boundary. On cold start (all 3 caches empty), sequential lock acquisition means 3x serial client construction. Under Cloud Run scale-to-zero, every warm-up incurs this penalty.
**Fix**: (1) Construct LLM clients during FastAPI lifespan (before requests arrive). (2) For TTL refresh, use a "singleflight" pattern: return the stale client immediately while one background task refreshes, so only the refresh coroutine takes the lock. (3) Alternatively, increase TTL significantly (24h or permanent with explicit rotation signal).

### F8: MEDIUM — Thread pool usage lacks concurrency control and cancellation semantics

**File**: `src/agent/nodes.py`
**Lines**: 267-294
**Issue**: `asyncio.to_thread(search_hours/search_knowledge_base)` uses the default thread pool executor with `wait_for` timeout. Problems: (1) No explicit thread pool size — default is `min(32, os.cpu_count() + 4)`. (2) `wait_for` cancellation only cancels the awaiting coroutine, not the underlying thread — the ChromaDB query continues running. (3) No semaphore to limit concurrent retrieval operations. (4) Cloud Run containers typically have 1-2 vCPUs, meaning the default pool is small.
**Impact**: Under concurrent requests, thread pool saturation causes retrieval latency to spike. Timed-out queries still consume thread pool slots and CPU. Cascading: if retrieval threads are exhausted, unrelated `to_thread` calls (ingestion, etc.) queue behind them.
**Fix**: (1) Use a dedicated `ThreadPoolExecutor(max_workers=N)` for retrieval with explicit sizing. (2) Add an `asyncio.Semaphore(N)` to cap concurrent retrievals. (3) Document that `wait_for` cancellation doesn't stop the thread. (4) Long-term: use async-native retrieval (Firestore is async; ChromaDB is the problem).

### F9: MEDIUM — ApiKeyMiddleware key refresh is not concurrency-safe

**File**: `src/api/middleware.py`
**Lines**: 217-244
**Issue**: `_get_api_key()` performs a time-check + settings read + cache update without any lock. Under concurrent requests at TTL boundary, multiple coroutines can race: read stale `_cached_at`, all decide to refresh, and all call `get_settings().API_KEY.get_secret_value()`. The method is called from `__call__` which is an async context.
**Impact**: Minor: redundant settings reads, possible brief window where different requests see old vs new key. Not a correctness issue since `hmac.compare_digest` is constant-time, but violates the principle of safe concurrent access to mutable state.
**Fix**: Add a simple lock or accept the race (document it as benign). Since key rotation is rare and the race window is tiny, this is LOW priority but worth noting for thoroughness.

### F10: MEDIUM — MemorySaver with no eviction stores unbounded conversation history

**File**: `src/agent/memory.py` + `src/agent/graph.py`
**Lines**: memory.py:17-56, graph.py:334-342
**Issue**: When using `MemorySaver` (dev or accidental prod fallback), ALL conversation checkpoints are stored in-memory forever. `MAX_MESSAGE_LIMIT=40` limits messages per thread, but there's no limit on the NUMBER of threads. Each `thread_id` (UUID) creates a new checkpoint entry that is never evicted.
**Impact**: Memory grows linearly with unique conversations. In a demo with moderate traffic (1000 conversations/day), memory accumulates indefinitely. Container OOM kill after extended uptime.
**Fix**: Add `MAX_ACTIVE_THREADS` configuration with LRU eviction of old threads (mentioned in CLAUDE.md as "MAX_ACTIVE_THREADS=1000 guard" but not found in the code). Implement a periodic cleanup or use TTL-based eviction for MemorySaver threads.

### F11: LOW — Module-level parity assertions run at import time in production

**File**: `src/agent/graph.py`
**Lines**: 387-394
**File**: `src/casino/feature_flags.py`
**Lines**: 62-76
**Issue**: Import-time `assert` statements validate schema parity. While guarded by `__debug__` in `graph.py`, the assertions in `feature_flags.py` are NOT guarded — they run unconditionally. If someone adds a feature flag to one dict but not the other, the import crashes the entire application.
**Impact**: A config drift bug causes a full application crash (not graceful degradation). This is aggressive fail-fast, which is debatable — good for catching bugs, bad for availability.
**Fix**: Either guard all assertions with `if __debug__:` (stripped with `python -O`) or convert to startup validation that logs ERROR and continues. Alternatively, keep them as-is and document the intention (crash-fast on config drift is intentional).

---

## Scalability Inventory Summary

### In-Memory State Lost on Restart/Scale-Out

| Component | Data | Bounded? | Impact on Loss |
|-----------|------|----------|----------------|
| MemorySaver | Conversation history | No | Users lose context |
| _memory_store | Guest profiles | No | Profile data lost |
| CircuitBreaker | Failure tracking | Yes (deque) | Resets to closed |
| RateLimitMiddleware | IP request counts | Yes (10K) | Rate limits reset |
| WebhookIdempotencyTracker | Processed IDs | Yes (10K) | Duplicate processing |
| _DELIVERY_LOG | SMS delivery statuses | Yes (10K) | Status lookup fails |
| _content_hashes | CMS content hashes | Yes (10K) | Unnecessary re-indexing |
| _config_cache | Casino configs | **No** | Re-fetched from Firestore |
| _flag_cache | Feature flags | **No** | Re-fetched from Firestore |

### Lock Contention Risk (High to Low)

| Lock | Contention | Frequency | Severity |
|------|-----------|-----------|----------|
| RateLimitMiddleware._lock | Every /chat + /feedback | Per-request | **HIGH** |
| CircuitBreaker._lock | Every LLM call (allow + record) | Per-request | MEDIUM |
| _llm_lock | On cache miss (hourly) | Rare | LOW (but P99 spike) |
| _validator_lock | On cache miss (hourly) | Rare | LOW |
| _whisper_lock | On cache miss (hourly) | Rare | LOW |
| _FailureCounter._lock | On whisper failure | Rare | LOW |
| WebhookIdempotencyTracker._lock | Per SMS webhook | Low volume | LOW |

### asyncio.to_thread Usage (Event Loop Blocking Mitigations)

| Call Site | Function | Timeout | Concern |
|-----------|----------|---------|---------|
| retrieve_node | search_hours / search_knowledge_base | 10s | Thread pool saturation under load |
| lifespan | ingest_property | None | Blocks startup; OK for single init |

---

## Recommendations (Priority Order)

1. **CRITICAL**: Make checkpointer fallback to MemorySaver a hard failure in production
2. **CRITICAL**: Make guest profile in-memory fallback a hard failure in production
3. **HIGH**: Move rate limiting to Cloud Armor or shared Redis
4. **HIGH**: Add lifecycle management to Firestore AsyncClient (lifespan-scoped)
5. **HIGH**: Bound casino config and feature flag caches with TTLCache(maxsize=N)
6. **MEDIUM**: Add dedicated thread pool executor for retrieval with explicit sizing
7. **MEDIUM**: Initialize LLM clients during lifespan to avoid cold-start lock contention
8. **MEDIUM**: Implement MAX_ACTIVE_THREADS eviction for MemorySaver
9. **LOW**: Guard all import-time assertions with `if __debug__:`
10. **LOW**: Add lock to ApiKeyMiddleware TTL refresh

---

*Review conducted with spotlight on scalability & async patterns. Scalability findings received +1 severity boost per round protocol.*
