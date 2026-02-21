# Hey Seven Production Review -- Round 14 (Grok Focus)

**Reviewer**: Grok (hostile mode)
**Commit**: 8655074
**Date**: 2026-02-21
**Focus**: Operational readiness, deployment configuration, monitoring, day-2 operations
**Spotlight**: RAG Pipeline + Domain Intelligence (+1 severity)
**Previous scores**: R11 avg 79.3, R12 avg 80.0, R13 avg 84.3

---

## Executive Summary

The codebase demonstrates significant maturity after 13 review rounds. The operational runbook, CI/CD pipeline, and probe architecture are genuinely well-designed. However, multiple day-2 operational gaps remain that would cause real production pain: no graceful shutdown handler for in-flight requests, RAG ingestion version-stamp purging is ChromaDB-only (not wired for Firestore production path), embeddings use `@lru_cache` instead of TTLCache creating an inconsistency that will bite during credential rotation, and the CMS webhook content hash store is in-memory with no persistence -- a container restart loses all change-detection state silently. The Firestore retriever performs property_id filtering in Python post-fetch rather than in the Firestore query, which is both a cost and correctness concern at scale.

---

## Dimension Scores

| # | Dimension | Score | Spotlight |
|---|-----------|-------|-----------|
| 1 | Graph Architecture | 8 | |
| 2 | RAG Pipeline | 6 | +1 severity |
| 3 | Data Model | 7 | |
| 4 | API Design | 7 | |
| 5 | Testing Strategy | 8 | |
| 6 | Docker & DevOps | 7 | |
| 7 | Prompts & Guardrails | 8 | |
| 8 | Scalability & Production | 6 | |
| 9 | Trade-off Documentation | 9 | |
| 10 | Domain Intelligence | 6 | +1 severity |

**Overall Score: 72/100**

---

## Findings

### F-001: Embeddings use @lru_cache while all other singletons use TTLCache -- credential rotation will fail silently [CRITICAL, Spotlight RAG +1 = CRITICAL]

**File**: `src/rag/embeddings.py:20-39`

The `get_embeddings()` function uses `@lru_cache(maxsize=4)` which NEVER expires. Every other credential-bearing singleton in the codebase uses TTLCache with 1-hour TTL for GCP Workload Identity credential rotation:
- `_get_llm()` -- TTLCache(maxsize=1, ttl=3600)
- `_get_validator_llm()` -- TTLCache(maxsize=1, ttl=3600)
- `_get_circuit_breaker()` -- TTLCache(maxsize=1, ttl=3600)
- `get_checkpointer()` -- TTLCache(maxsize=1, ttl=3600)

But `get_embeddings()` uses `@lru_cache(maxsize=4)` -- it will hold stale credentials indefinitely. When GCP Workload Identity Federation rotates credentials (typically every 1 hour), the embeddings client will start failing with authentication errors. The retriever and ingestion pipeline will break silently. This was explicitly called out in the project's own rules (`langgraph-patterns.md`, "TTL-Cached LLM Singletons") and yet the embeddings module violates it.

The `get_settings()` function also uses `@lru_cache(maxsize=1)` but settings are immutable for a process lifetime so this is acceptable. Embeddings clients hold credential references that rotate.

**Impact**: Retrieval and ingestion fail after credential rotation (~1 hour in WIF environments). Silent degradation -- retriever returns empty results, agent produces contextless responses.

**Fix**: Replace `@lru_cache(maxsize=4)` with TTLCache pattern matching `_get_llm()`:
```python
_embeddings_cache: TTLCache = TTLCache(maxsize=4, ttl=3600)

def get_embeddings(task_type: str | None = None) -> GoogleGenerativeAIEmbeddings:
    cache_key = task_type or "__default__"
    cached = _embeddings_cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    kwargs = {"model": settings.EMBEDDING_MODEL}
    if task_type:
        kwargs["task_type"] = task_type
    instance = GoogleGenerativeAIEmbeddings(**kwargs)
    _embeddings_cache[cache_key] = instance
    return instance
```

---

### F-002: Firestore retriever filters property_id in Python, not in the Firestore query -- cross-tenant leakage risk + cost waste [HIGH, Spotlight RAG +1 = CRITICAL]

**File**: `src/rag/firestore_retriever.py:91-106`

The `_single_vector_query()` method fetches `top_k * 2` documents from Firestore and then filters `property_id` in Python:

```python
vector_query = collection_ref.find_nearest(
    vector_field="embedding",
    query_vector=Vector(query_vector),
    distance_measure=DistanceMeasure.COSINE,
    limit=top_k * 2,  # Over-fetch because post-hoc filtering
    distance_result_field="distance",
)
# ...
for doc_snapshot in vector_query.get():
    doc_property_id = data.get("metadata", {}).get("property_id", "")
    if doc_property_id != property_id:
        continue  # Filter in Python
```

Problems:
1. **Cross-tenant data reaches the application layer**: Even though filtered out, documents from OTHER casinos are fetched and deserialized in the application process. A logging bug, exception handler, or debug statement could leak cross-tenant data.
2. **Cost**: Firestore charges per document read. Over-fetching 2x and discarding up to half is pure cost waste.
3. **Result degradation**: If the collection has many properties, the 2x over-fetch may not be enough -- you could get 0 results for the target property if other properties dominate the vector space.

The `CasinoKnowledgeRetriever` (ChromaDB path) correctly uses a `filter={"property_id": property_id}` parameter in the query itself. The Firestore path should use a Firestore `where()` pre-filter before `find_nearest()`, or use a composite index with `property_id` + vector.

**Impact**: Multi-tenant data leakage risk in the production path. Cost waste on every retrieval. Potential result set starvation.

---

### F-003: Version-stamp purging not implemented for Firestore production path [HIGH, Spotlight RAG +1 = CRITICAL]

**File**: `src/rag/pipeline.py:695-721` (ChromaDB only), `src/rag/firestore_retriever.py` (no purge logic)

The `ingest_property()` function implements `_ingestion_version` stamping and stale chunk purging, but this logic is entirely ChromaDB-specific:

```python
collection = vectorstore._collection  # ChromaDB internal API
old_docs = collection.get(
    where={
        "$and": [
            {"property_id": {"$eq": property_id}},
            {"_ingestion_version": {"$ne": version_stamp}},
        ]
    }
)
```

The Firestore production path has NO equivalent purge logic. When property data is edited and re-ingested in production:
1. New content creates new SHA-256 IDs (different content = different hash)
2. Old content's chunks remain in Firestore with stale `_ingestion_version`
3. Ghost data accumulates and pollutes retrieval results

This was explicitly called out as MANDATORY in the project's own `rag-production.md` rules: "Version-Stamp Purging for Stale RAG Chunks (MANDATORY)". The rule even has its origin listed as "Hey Seven R5-R20". Yet the production path (Firestore) does not implement it.

The `reingest_item()` function (used by CMS webhook for individual item updates) also lacks version-stamp purging -- it only does content-hash-based upsert, which does not remove old chunks when content changes.

**Impact**: Stale restaurant hours, outdated pricing, and ghost data accumulate in production vector store. Guests receive contradictory information.

---

### F-004: CMS webhook content hash store is in-memory -- container restart loses all change-detection state [MEDIUM, Spotlight Domain +1 = HIGH]

**File**: `src/cms/webhook.py:91-97`

```python
_content_hashes: TTLCache[str, str] = TTLCache(
    maxsize=_CONTENT_HASH_MAXSIZE, ttl=_CONTENT_HASH_TTL
)
```

The CMS webhook uses an in-memory TTLCache for content hash change detection. When a Cloud Run container restarts (routine in production), ALL content hashes are lost. The next CMS webhook will re-index EVERY item, even if nothing changed, because the hash comparison always fails (stored_hash is None).

This creates:
1. **Unnecessary re-indexing**: Every container restart triggers full re-indexing on the next batch of CMS webhooks
2. **Inconsistency across instances**: With `max-instances=10`, each instance has independent hash state. The same CMS update hitting different instances will be processed differently
3. **Silent regression**: No logging or alerting when the hash store is cold (all comparisons fail to "unchanged" path)

The code acknowledges this: "In production this would be backed by Firestore." But that TODO has not been implemented. For a system where content freshness is critical (restaurant hours, pricing), this is a meaningful operational gap.

**Fix**: Persist content hashes in Firestore (the production backend already exists). Or at minimum, log a warning when the hash store is cold after container start.

---

### F-005: No graceful shutdown handler -- in-flight SSE streams dropped on SIGTERM [HIGH]

**File**: `src/api/app.py:46-105`, `Dockerfile:64-70`

The Dockerfile configures `--timeout-graceful-shutdown 15` for uvicorn, and the runbook documents "15s allows in-flight SSE streams to complete." But the FastAPI lifespan handler does no graceful shutdown work:

```python
yield
app.state.ready = False
app.state.agent = None
logger.info("Application shutdown complete.")
```

On SIGTERM:
1. `app.state.ready = False` -- but nothing reads this during request processing
2. `app.state.agent = None` -- immediately nullifies the agent, so any in-flight requests that try to access `request.app.state.agent` after this line will get `None` and return 503
3. The 15-second uvicorn timeout only applies if the application cooperates -- but setting agent to None causes immediate 503 failures for in-flight requests

The correct pattern is:
```python
yield
app.state.ready = False  # Stop accepting new requests via health check
await asyncio.sleep(0)   # Let event loop drain pending responses
# Do NOT set agent=None until after shutdown timeout
logger.info("Application shutdown complete.")
# Agent cleanup happens when process exits
```

Additionally, the `/health` endpoint returns 503 when `ready=False`, but Cloud Run uses `/live` for liveness. The `/health` is used as startup probe only. So setting `ready=False` on shutdown does not actually affect traffic routing -- Cloud Run has already stopped sending new requests by the time SIGTERM fires. The `agent = None` is the real problem: it breaks in-flight requests.

**Impact**: Every Cloud Run deployment drops in-flight SSE streams with 503 errors. Users see mid-response failures during deployments.

---

### F-006: Smoke test in cloudbuild.yaml does not verify VERSION matches COMMIT_SHA [MEDIUM]

**File**: `cloudbuild.yaml:89-111`

The smoke test captures the deployed version but does NOT assert it matches:

```bash
DEPLOYED_VERSION=$(cat /tmp/health.json | python3 -c "...")
echo "Deployed version: $DEPLOYED_VERSION, Expected: $COMMIT_SHA"
break  # Breaks out of loop without comparing!
```

The version is echoed but never compared. The `break` statement exits the retry loop immediately after a 200 response, regardless of whether the version matches. This means:
1. Stale deployments pass the smoke test (old container serves 200 from previous revision)
2. The deployment proceeds to route 100% traffic to a potentially broken new revision that did not actually start

The runbook explicitly calls out "Verify version: response version field must match deployed commit SHA" as a deployment step, and the project's own `azure-deploy.md` rules mandate "Post-Deploy Version Assertion (MANDATORY)". Yet the automated pipeline does not enforce it.

**Fix**: Add version assertion after the break:
```bash
if [ "$DEPLOYED_VERSION" != "$COMMIT_SHA" ]; then
  echo "VERSION MISMATCH: deployed=$DEPLOYED_VERSION expected=$COMMIT_SHA"
  exit 1
fi
```

---

### F-007: Firestore retriever distance-to-similarity conversion is incorrect for cosine [MEDIUM, Spotlight RAG +1 = HIGH]

**File**: `src/rag/firestore_retriever.py:112-117`

```python
# Firestore COSINE distance is in [0, 2]; convert to similarity [0, 1].
distance = data.get("distance", 2.0)
similarity = max(0.0, 1.0 - distance)
```

The comment says COSINE distance is in [0, 2], which is correct. But `1.0 - distance` maps:
- distance=0 -> similarity=1.0 (correct, identical)
- distance=1 -> similarity=0.0 (orthogonal)
- distance=2 -> similarity=-1.0, clamped to 0.0 (opposite)

This means documents with distance > 1.0 (anti-correlated) all collapse to similarity=0.0. While clamping to 0 is safe (they would be filtered out by `RAG_MIN_RELEVANCE_SCORE`), the issue is that the similarity scale is compressed. A distance of 0.3 maps to similarity 0.7, while the ChromaDB path uses `similarity_search_with_relevance_scores` which uses the LangChain normalization formula: `similarity = 1 - (distance / 2)`. This means:
- Firestore: distance=0.3 -> 0.70
- ChromaDB: distance=0.3 -> 0.85

The same `RAG_MIN_RELEVANCE_SCORE=0.3` threshold filters differently across backends. Firestore is more aggressive at filtering (lower raw scores), potentially dropping relevant documents that ChromaDB would keep.

The project's own `rag-production.md` rules state: "Cosine distance range is [0, 2], not [0, 1]." The code acknowledges this but uses a different normalization formula than ChromaDB, creating an inconsistency.

**Impact**: Backend-switching between dev (ChromaDB) and prod (Firestore) produces different retrieval results for identical queries. Quality tuning in dev does not transfer to prod.

---

### F-008: InMemoryBackend probabilistic sweep uses `random.random()` -- non-deterministic in testing [LOW]

**File**: `src/state_backend.py:64-88`

The `_maybe_sweep()` method uses `random.random()` to decide whether to run a full eviction sweep:

```python
if not force and random.random() > self._SWEEP_PROBABILITY:
    return
```

This makes test behavior non-deterministic. Tests that depend on sweep behavior (e.g., verifying memory bounds) will intermittently fail because the sweep fires ~1% of the time. While this is intentional for production (amortized cost), it makes the module harder to test reliably.

A production-grade approach would accept a `sweep_probability` constructor parameter, allowing tests to pass `1.0` for deterministic behavior.

**Impact**: Intermittent test failures. Low operational impact.

---

### F-009: Rate limiter and circuit breaker have no metrics export -- monitoring is blind [MEDIUM]

**Files**: `src/api/middleware.py`, `src/agent/circuit_breaker.py`

Both the rate limiter and circuit breaker maintain internal state (request counts, failure counts, state transitions) but expose no metrics to external monitoring systems. The circuit breaker state is visible via `/health`, but:

1. **Rate limiter**: No endpoint or metric to see current rate limit utilization per IP, total tracked clients, or eviction counts. An operator cannot tell if the rate limiter is doing anything useful or if it is being bypassed.
2. **Circuit breaker**: `failure_count` is available via the property but not exposed in any structured metric. No histogram of failure rates over time. The transition logs (INFO/WARNING) are useful but not queryable as metrics.
3. **No Prometheus/OpenMetrics endpoint**: The entire application has no `/metrics` endpoint. Cloud Run supports custom metrics via Cloud Monitoring client libraries, but none are used.

The runbook defines alert thresholds (e.g., "Circuit breaker opens > 1/hour") but provides no mechanism to actually measure these metrics. The operator would have to grep Cloud Logging for "Circuit breaker OPEN" log lines and count them manually.

**Impact**: Operators cannot proactively detect degradation. Alert thresholds defined in the runbook are unenforceable without metrics infrastructure.

---

### F-010: Property data file (mohegan_sun.json) loaded at startup with no cache invalidation path [MEDIUM, Spotlight Domain +1 = HIGH]

**File**: `src/api/app.py:92-99`

```python
property_path = Path(settings.PROPERTY_DATA_PATH)
if property_path.exists():
    with open(property_path, encoding="utf-8") as f:
        app.state.property_data = json.load(f)
```

The property JSON is loaded once at startup into `app.state.property_data` and used by:
- `/property` endpoint (returns property metadata)
- `greeting_node` (builds greeting categories from the file)

There is NO mechanism to refresh this data without a container restart. The CMS webhook handles individual item updates in the vector store, but it does NOT update `app.state.property_data`. This means:

1. The `/property` endpoint serves stale metadata after CMS updates (e.g., a new restaurant category added via CMS is missing from `/property`)
2. The greeting node's category list is stale (a new category added to the JSON via CMS does not appear in the greeting)
3. The greeting cache (`_greeting_cache` in nodes.py, TTL 1 hour) reads from the file, not from `app.state.property_data`, so at least it refreshes hourly. But the `/property` endpoint reads from `app.state.property_data` which never refreshes.

For a system where "the autonomous casino host that never sleeps" is the value proposition, stale property metadata visible to end-users is a domain intelligence gap.

**Impact**: `/property` endpoint serves stale data after CMS content updates. Greeting categories lag behind actual knowledge-base content.

---

### F-011: BoundedMemorySaver eviction accesses MemorySaver internals via `self._inner.storage` [LOW]

**File**: `src/agent/memory.py:68-74`

```python
if hasattr(self._inner, "storage"):
    keys_to_remove = [
        k for k in self._inner.storage if isinstance(k, tuple) and len(k) > 0 and k[0] == evicted_id
    ]
    for k in keys_to_remove:
        del self._inner.storage[k]
```

This reaches into `MemorySaver`'s internal `storage` attribute, which is not part of any public API. LangGraph version upgrades could rename or restructure this attribute, silently breaking eviction. The `hasattr` guard prevents a crash but would cause unbounded memory growth if `storage` is removed (eviction becomes a no-op).

Additionally, the linear scan `for k in self._inner.storage` is O(n) where n is the total number of checkpoints across all threads. With `MAX_ACTIVE_THREADS=1000` and multiple checkpoints per thread, this could be slow.

**Impact**: Low -- dev/demo only. But the fragility is worth noting.

---

### F-012: SMS webhook in app.py calls different verify_webhook_signature than CMS webhook [LOW]

**File**: `src/api/app.py:405-423`

The SMS webhook imports `verify_webhook_signature` from `src.sms.webhook`:
```python
from src.sms.webhook import handle_inbound_sms, verify_webhook_signature
```

But the CMS webhook uses `verify_webhook_signature` from `src.cms.webhook`:
```python
# CMS uses its own HMAC-based verify_webhook_signature
```

These are two completely different functions with different signatures (Ed25519 vs HMAC-SHA256). This is correct behavior (different webhook providers use different signature schemes), but the shared function name is a maintenance trap. A developer refactoring "webhook signature verification" might accidentally merge them.

**Impact**: Low -- cosmetic naming concern.

---

## Detailed Dimension Analysis

### 1. Graph Architecture (8/10)

Strengths:
- 11-node StateGraph with clear topology documentation
- Validation loop with degraded-pass strategy is well-reasoned
- State parity check at import time prevents drift
- Node name constants with _KNOWN_NODES frozenset
- Feature flag dual-layer design (build-time topology vs runtime behavior) is well-documented

Weaknesses:
- No graceful shutdown coordination (F-005)
- _dispatch_to_specialist uses a broad `except Exception` that could mask unexpected errors

### 2. RAG Pipeline (6/10) [SPOTLIGHT]

Strengths:
- Per-item chunking with category-specific formatters is excellent
- SHA-256 idempotent ingestion IDs
- Version-stamp purging for ChromaDB
- RRF reranking implementation is correct
- AbstractRetriever provides clean backend abstraction

Weaknesses:
- Embeddings cache uses @lru_cache instead of TTLCache (F-001) -- CRITICAL
- Firestore retriever filters property_id in Python, not in query (F-002) -- CRITICAL
- Version-stamp purging not implemented for Firestore (F-003) -- CRITICAL
- Cosine distance normalization inconsistent between backends (F-007) -- HIGH
- `reingest_item()` has no stale chunk purge path

### 3. Data Model (7/10)

Strengths:
- PropertyQAState with Annotated reducers (add_messages, _keep_max)
- RetrievedChunk TypedDict prevents implicit dict contracts
- Pydantic structured outputs with Literal types
- _initial_state parity check

Weaknesses:
- `RetrievedChunk` type is defined but `retrieved_context` is typed as `list[RetrievedChunk]` while actual retrieval functions return `list[dict]` -- the TypedDict is aspirational, not enforced at runtime
- No data validation on the property JSON file (trusts the file structure completely)

### 4. API Design (7/10)

Strengths:
- Pure ASGI middleware stack (no BaseHTTPMiddleware)
- Separate /live and /health endpoints with correct probe assignment
- SSE heartbeat prevents client-side EventSource timeouts
- Streaming PII redaction with lookahead buffer
- Error responses include Retry-After headers

Weaknesses:
- Graceful shutdown sets agent=None while requests may be in-flight (F-005)
- /property serves stale data after CMS updates (F-010)
- No /metrics endpoint for operational visibility (F-009)

### 5. Testing Strategy (8/10)

Strengths:
- 1450+ tests across 32 files
- 90% coverage gate in CI
- Conftest singleton cleanup fixture
- Deployment tests (test_deployment.py)

Weaknesses:
- InMemoryBackend probabilistic sweep is non-deterministic in tests (F-008)
- No evidence of end-to-end tests exercising the Firestore retrieval path (all RAG tests appear to use ChromaDB)

### 6. Docker & DevOps (7/10)

Strengths:
- Multi-stage build (slim final image)
- Non-root user (appuser)
- Trivy vulnerability scan in CI
- Canary deploy with --no-traffic + smoke test + traffic routing
- Automatic rollback on smoke test failure
- CPU boost for cold start
- Exec form CMD

Weaknesses:
- Smoke test does not assert version match (F-006) -- defeats the purpose
- `knowledge-base/` directory not copied into Docker image (only `data/`, `static/`, `src/`)
- No image tagging beyond commit SHA (no `latest`, no semver tag)
- No Docker layer caching between builds (pip install re-runs on every requirements change due to COPY ordering)

### 7. Prompts & Guardrails (8/10)

Strengths:
- 5-layer deterministic guardrails before any LLM call
- Structured output routing via Pydantic Literal types
- Responsible gaming escalation counter with _keep_max reducer
- PII redaction fails closed with [PII_REDACTION_ERROR] placeholder
- Streaming PII redactor with lookahead buffer for token-spanning patterns

Weaknesses:
- No adversarial testing evidence for guardrail bypass (e.g., Unicode normalization attacks, homoglyph substitution)

### 8. Scalability & Production (6/10)

Strengths:
- Circuit breaker with rolling window, half-open probe, cancellation handling
- Rate limiter with LRU eviction, trusted proxy support, stale client sweep
- BoundedMemorySaver with LRU eviction for dev
- StateBackend abstraction with Redis option documented

Weaknesses:
- In-memory rate limiting per-instance (documented but not resolved for production)
- CMS content hash store in-memory only (F-004)
- No metrics export (F-009)
- Embeddings credential rotation gap (F-001)
- Firestore retriever operational gaps (F-002, F-003, F-007)

### 9. Trade-off Documentation (9/10)

Strengths:
- Comprehensive runbook with probe configuration, incident response, escalation matrix
- Feature flag architecture documented inline with rationale
- Known limitations called out (rate limiter per-instance, MemorySaver dev-only)
- Degraded-pass validation strategy documented with rationale
- Cloud Run probe design decision documented with anti-pattern warning

Weaknesses:
- Runbook alert thresholds are aspirational without metrics infrastructure to measure them
- The documentation says "Post-Deploy Version Assertion (MANDATORY)" but the pipeline does not enforce it

### 10. Domain Intelligence (6/10) [SPOTLIGHT]

Strengths:
- Detailed Mohegan Sun property data with 15+ restaurants, entertainment, hotel, gaming
- Category-specific formatters produce rich text for embeddings
- TCPA compliance with 280+ area code timezone mappings
- Consent hash chain with HMAC-SHA256 tamper evidence
- Quiet hours enforcement with MNP caveats documented

Weaknesses:
- Property data has no freshness indicator -- no `last_verified_date` or `data_as_of` field
- No validation that property data JSON matches expected schema (e.g., missing hours field)
- `/property` endpoint serves stale data (F-010) -- HIGH
- CMS change detection loses state on restart (F-004) -- HIGH
- No knowledge-base directory in Docker image -- markdown domain documents (regulations, comp formulas, host workflows) are not available in production unless explicitly included

---

## Summary Table

| ID | Severity | Dimension | Finding |
|----|----------|-----------|---------|
| F-001 | CRITICAL | RAG Pipeline | Embeddings @lru_cache vs TTLCache -- credential rotation failure |
| F-002 | CRITICAL | RAG Pipeline | Firestore retriever filters property_id in Python, not in query |
| F-003 | CRITICAL | RAG Pipeline | Version-stamp purging not implemented for Firestore |
| F-004 | HIGH | Domain Intelligence | CMS content hash store is in-memory only |
| F-005 | HIGH | API Design | No graceful shutdown -- agent=None breaks in-flight requests |
| F-006 | MEDIUM | Docker & DevOps | Smoke test does not assert VERSION matches COMMIT_SHA |
| F-007 | HIGH | RAG Pipeline | Cosine distance normalization inconsistent between backends |
| F-008 | LOW | Scalability | InMemoryBackend probabilistic sweep non-deterministic in tests |
| F-009 | MEDIUM | Scalability | No metrics export for rate limiter or circuit breaker |
| F-010 | HIGH | Domain Intelligence | Property data loaded once at startup, no refresh path |
| F-011 | LOW | Data Model | BoundedMemorySaver accesses MemorySaver internal storage |
| F-012 | LOW | API Design | Shared function name across different webhook verification schemes |

---

## Verdict

The codebase has reached strong architectural maturity in the graph layer, guardrails, and documentation. The operational runbook is one of the best I have seen in a seed-stage project. However, the RAG pipeline has three CRITICAL findings that all stem from the same root cause: **the production path (Firestore) has not received the same care as the development path (ChromaDB)**. Credential rotation for embeddings, multi-tenant query isolation, and stale chunk purging all work correctly in ChromaDB but are broken or missing in Firestore. This is a classic dev/prod parity gap that will cause real operational pain.

The 5-point gap from R13 (84 -> 72) may seem harsh, but I am applying the spotlight severity multiplier consistently. Three CRITICAL RAG findings in the production path warrant the score. Fix F-001 through F-003 and this codebase is solidly in the 80+ range.
