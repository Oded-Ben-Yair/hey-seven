# Hey Seven Production Review -- Round 16 (Grok Focus)

**Reviewer**: Grok (operational readiness, deployment, monitoring, day-2 operations)
**Commit**: a0bb225
**Date**: 2026-02-21
**Previous Score**: 72 (R15)

---

## Scoring Summary

| # | Dimension | Score | Key Finding |
|---|-----------|-------|-------------|
| 1 | Graph Architecture | 8 | Solid 11-node topology, parity check, structured dispatch with CB-aware fallback |
| 2 | RAG Pipeline | 7 | Per-item chunking, version-stamp purging, but retriever cache is not thread-safe |
| 3 | Data Model | 7 | TypedDict + Pydantic structured outputs, _keep_max reducer, but RetrievedChunk contract unenforced |
| 4 | API Design | 7 | Pure ASGI middleware, SSE heartbeats, but /health HEALTHCHECK uses wrong probe logic |
| 5 | Testing Strategy | 6 | 1452 tests, 90% coverage gate, but no load/soak test, no chaos/fault-injection |
| 6 | Docker & DevOps | 7 | Multi-stage build, Trivy, canary deploy, but missing image pinning digest and no `--no-log-init` |
| 7 | Prompts & Guardrails | 8 | 5-layer deterministic + semantic classifier, multilingual, but semantic classifier fail-closed on LLM outage blocks all borderline queries |
| 8 | Scalability & Production | 5 | Single-worker uvicorn, in-memory everything, no distributed rate limit, retriever race condition |
| 9 | Trade-off Documentation | 8 | Extensive comments, runbook, feature flag architecture documented inline |
| 10 | Domain Intelligence | 7 | TCPA compliance, consent hash chain, area code timezone mapping, but no DNC registry integration |

**Overall Score: 70/100**

**Delta from R15**: -2 (Grok 72 -> 70). Operational hardening has stalled while complexity has grown. Multi-container readiness is the blocking gap.

---

## Findings

### CRITICAL (Production-Breaking)

#### C-001: Retriever Cache Dict Is Not Thread-Safe Under Concurrent Async Access

**File**: `/home/odedbe/projects/hey-seven/src/rag/pipeline.py`, lines 894-972

The retriever singleton uses a bare `dict` (`_retriever_cache`, `_retriever_cache_time`) with no lock protection. Unlike the LLM cache (`_llm_cache` + `_llm_lock`) and circuit breaker cache (`_cb_cache` + `_cb_lock`), this cache has no `asyncio.Lock`. Two concurrent requests hitting an expired TTL will both enter `_get_retriever_cached()`, both see the cache miss, both construct a new retriever, and both write to the dict. While CPython's GIL prevents dict corruption, this creates duplicate Chroma/Firestore connections and wastes resources.

More critically, with `WEB_CONCURRENCY > 1` (the documented production scaling path), `dict` access across threads is NOT safe without a lock. The code documents multi-worker as the production path but the retriever singleton is not safe for it.

```python
# Current: No lock
_retriever_cache: dict[str, AbstractRetriever] = {}
_retriever_cache_time: dict[str, float] = {}

# Required: Consistent with _llm_lock, _cb_lock, _checkpointer_lock
_retriever_lock = asyncio.Lock()
```

**Impact**: Duplicate connections, potential data corruption under multi-worker deployment.
**Severity**: CRITICAL -- inconsistent with every other singleton cache in the codebase.

---

#### C-002: Smoke Test Does Not Fail on Version Mismatch

**File**: `/home/odedbe/projects/hey-seven/cloudbuild.yaml`, lines 79-111

The smoke test in Step 7 extracts `DEPLOYED_VERSION` and compares it to `$COMMIT_SHA`, but never fails the build if they mismatch. The `echo` statements print the comparison but there is no `if [ "$DEPLOYED_VERSION" != "$COMMIT_SHA" ]; then exit 1; fi` guard. The pipeline will happily route traffic to a stale revision where the old code is still running.

```yaml
# Current: prints but never fails
DEPLOYED_VERSION=$(cat /tmp/health.json | python3 -c "...")
echo "Deployed version: $DEPLOYED_VERSION, Expected: $COMMIT_SHA"
break  # <-- breaks out of retry loop regardless of version match
```

The runbook documents version assertion as mandatory ("Verify version: response `version` field must match deployed commit SHA") but the automation does not enforce it.

**Impact**: Stale code deployed to production without detection. This is the exact Azure Consumption Plan lesson (old code served 13+ hours) that the codebase's own documentation warns about.
**Severity**: CRITICAL -- the deployment pipeline has a documented-but-unenforced gate.

---

### HIGH (Significant Operational Risk)

#### H-001: `get_settings()` Uses `@lru_cache` Which Never Expires

**File**: `/home/odedbe/projects/hey-seven/src/config.py`, line 180

Every other singleton in the codebase uses TTLCache with 1-hour refresh for credential rotation (LLM, validator, circuit breaker, checkpointer, retriever). But `get_settings()` uses `@lru_cache(maxsize=1)` which caches forever until process restart. If `GOOGLE_API_KEY` is rotated in Secret Manager, the settings object still holds the old value. The LLM singletons refresh hourly, but they read credentials from `get_settings()` which is stale.

The TTL-cached LLM singletons (`_get_llm`, `_get_validator_llm`) call `get_settings()` on every cache miss to read `MODEL_NAME`, `MODEL_TEMPERATURE`, etc. But `get_settings()` returns the same stale `Settings` instance with the old `GOOGLE_API_KEY`. The TTL refresh is pointless if the settings it reads from are eternally cached.

```python
@lru_cache(maxsize=1)  # NEVER expires
def get_settings() -> Settings:
    return Settings()
```

**Impact**: Credential rotation via Secret Manager does not take effect without container restart, defeating the purpose of TTL-cached singletons.
**Severity**: HIGH -- undermines the entire credential rotation architecture.

---

#### H-002: Dockerfile HEALTHCHECK Uses `/health` Which Returns 503 on CB Open

**File**: `/home/odedbe/projects/hey-seven/Dockerfile`, line 61-62

The Dockerfile HEALTHCHECK uses `/health` which returns 503 when the circuit breaker is open. The runbook explicitly warns "Do NOT use /health as livenessProbe" and documents that `/live` should be used for liveness. While Cloud Run ignores Dockerfile HEALTHCHECK, this is actively dangerous for local `docker-compose` deployments: when the LLM API is down, Docker will mark the container as unhealthy and restart it, creating the same amplification loop the runbook warns about.

```dockerfile
# Current: /health returns 503 on CB open -> container restart loop
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Should be: /live (always 200)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/live || exit 1
```

**Impact**: Local docker-compose health monitoring creates a restart loop during LLM outages. Contradicts the runbook's own guidance.
**Severity**: HIGH -- doc says one thing, code does another.

---

#### H-003: BoundedMemorySaver Is Not Concurrency-Safe

**File**: `/home/odedbe/projects/hey-seven/src/agent/memory.py`, lines 36-120

`BoundedMemorySaver._track_thread()` mutates `self._thread_order` (an `OrderedDict`) with `move_to_end`, `popitem`, and item assignment without any lock. Under concurrent async requests (50 concurrency per Cloud Run instance), multiple coroutines can interleave dict mutations. Unlike `dict`, `OrderedDict` is NOT guaranteed to be safe under concurrent mutation in CPython.

```python
def _track_thread(self, config: dict) -> None:
    # No lock -- concurrent mutations to OrderedDict
    if thread_id in self._thread_order:
        self._thread_order.move_to_end(thread_id)  # UNSAFE
    else:
        self._thread_order[thread_id] = True
    while len(self._thread_order) > self._max_threads:
        evicted_id, _ = self._thread_order.popitem(last=False)  # UNSAFE
```

Every other mutable shared state in the codebase has an `asyncio.Lock`. This one does not.

**Impact**: OrderedDict corruption under concurrent access during development. Since BoundedMemorySaver is labeled "NOT for production," this is HIGH (not CRITICAL), but it will cause flaky behavior during demos.
**Severity**: HIGH -- missing lock in a concurrently-accessed data structure.

---

#### H-004: Single-Worker Uvicorn in Production CMD

**File**: `/home/odedbe/projects/hey-seven/Dockerfile`, line 68-70

The Dockerfile CMD runs `--workers 1` with a comment about scaling via `WEB_CONCURRENCY`. But the CMD does not read `WEB_CONCURRENCY`. It hardcodes `--workers 1`. The env var is set but never used in the actual CMD:

```dockerfile
ENV WEB_CONCURRENCY=1
# ...
CMD ["python", "-m", "uvicorn", "src.api.app:app", \
    "--host", "0.0.0.0", "--port", "8080", "--workers", "1", \
    "--timeout-graceful-shutdown", "15"]
```

The `--workers` flag is hardcoded to `"1"` as a string literal. Changing `WEB_CONCURRENCY` env var has zero effect. To actually use the env var, the CMD would need shell expansion (which exec form does not support) or a gunicorn wrapper.

The Dockerfile comments suggest `gunicorn src.api.app:app -w ${WEB_CONCURRENCY}` but this is not implemented. The comment is aspirational but the code is hardcoded.

**Impact**: Cannot horizontally scale workers within a container without rebuilding the image or overriding the CMD.
**Severity**: HIGH -- documented scaling path is non-functional.

---

#### H-005: `logging.basicConfig()` Called Inside Lifespan, Not at Module Level

**File**: `/home/odedbe/projects/hey-seven/src/api/app.py`, lines 50-53

`logging.basicConfig()` is called inside the `lifespan()` async context manager, which runs when the first request arrives or the app starts. Any log messages emitted during module import (e.g., from `src.config.get_settings()` model validators, or from the parity check in `graph.py` line 503-510) use Python's default WARNING-level logging with no format configuration. These startup messages are invisible or unformatted.

Additionally, `logging.basicConfig()` is a no-op if any handler has already been attached to the root logger. The access logger in `middleware.py` line 37-43 adds a handler at import time (before lifespan runs), potentially causing `basicConfig()` to silently do nothing.

```python
# middleware.py (import time -- before lifespan)
_access_logger = _get_access_logger()  # Adds handler to root ancestor

# app.py lifespan (runtime -- after import)
logging.basicConfig(...)  # May be a no-op
```

**Impact**: Logging configuration may be silently ignored, causing missing or unformatted logs in production.
**Severity**: HIGH -- observability gap during startup and potentially entire runtime.

---

### MEDIUM (Correctness/Quality Issues)

#### M-001: CMS Webhook `verify_webhook_signature` Is Synchronous but Called from Async Endpoint

**File**: `/home/odedbe/projects/hey-seven/src/cms/webhook.py`, line 32-81

The CMS `verify_webhook_signature()` is a synchronous function (not `async def`), but it is called from the async `handle_cms_webhook()` function on line 160. The HMAC computation itself is CPU-bound and fast, so this is not blocking. However, the SMS `verify_webhook_signature` in `sms/webhook.py` IS `async def`. The inconsistency is confusing and error-prone. More importantly, the CMS version on line 160 is called with `if not verify_webhook_signature(...)` (no `await`), while the SMS version in `app.py` line 416 is called with `if not await verify_webhook_signature(...)`. A future maintainer who copies the CMS pattern to an async context will silently get `True` (truthy coroutine object) and bypass signature verification.

**Impact**: Inconsistent async/sync contracts across webhook verification functions.
**Severity**: MEDIUM -- correctness risk from copy-paste error.

---

#### M-002: InMemoryBackend Uses `time.monotonic()` Which Resets Across Processes

**File**: `/home/odedbe/projects/hey-seven/src/state_backend.py`, lines 57-117

The InMemoryBackend uses `time.monotonic()` for TTL computation, which is correct for single-process lifetime. However, the `_sweep_counter` is a module-level instance attribute that increments on every write. Under high write rates, `_sweep_counter` will overflow Python's int (no overflow in Python, but the modulo check `% 100` relies on counter growth). This is functionally correct but at 10M writes, the sweep happens at the same 1% rate. No issue here.

The actual issue: `_cleanup_expired()` is called on every `get()`, `get_count()`, `exists()` for the specific key. But the `_maybe_sweep()` (which cleans ALL expired keys) only runs on `set()` and `increment()`. Read-heavy workloads accumulate stale keys that are never swept, growing memory until the 50K cap triggers a force-sweep. With 10K unique IPs and 60s TTL, 10K * 36 bytes = 360KB -- not dangerous, but the documented "probabilistic sweep prevents unbounded memory growth" claim is only true for write-heavy workloads.

**Impact**: Memory growth proportional to unique key count in read-heavy patterns. Bounded by `_MAX_STORE_SIZE` but the sweep semantics are misleading.
**Severity**: MEDIUM.

---

#### M-003: `_get_idempotency_tracker()` Uses Bare Global Without Lock

**File**: `/home/odedbe/projects/hey-seven/src/sms/webhook.py`, lines 22-30

The idempotency tracker singleton uses `global _idempotency_tracker` with no lock. Two concurrent webhook requests could both see `None`, both create a tracker, and the first creation is silently discarded. The `WebhookIdempotencyTracker` internally uses `asyncio.Lock`, but the singleton creation itself is unprotected.

```python
def _get_idempotency_tracker() -> WebhookIdempotencyTracker:
    global _idempotency_tracker
    if _idempotency_tracker is None:
        _idempotency_tracker = WebhookIdempotencyTracker()  # Race condition
    return _idempotency_tracker
```

In CPython with single-threaded async, this is safe because there are no yield points between the check and assignment. But this relies on CPython implementation details, not language guarantees.

**Impact**: Potential duplicate message processing if the singleton race hits during initialization.
**Severity**: MEDIUM.

---

#### M-004: No Structured Error Response for JSON Parse Failures on Webhook Bodies

**File**: `/home/odedbe/projects/hey-seven/src/api/app.py`, lines 425, 469-470

Both the SMS webhook (line 425) and CMS webhook (line 470) call `json.loads(raw_body)` without a try/except for `json.JSONDecodeError`. If the webhook receives malformed JSON (not uncommon with webhook retries), this will propagate to the outer `except Exception` handler on lines 452/484, returning a generic 500 instead of a descriptive 400 "Invalid JSON body" error.

```python
body = json.loads(raw_body)  # JSONDecodeError -> generic 500
```

Telnyx will see the 500 and retry the webhook, creating a retry storm against malformed payloads. A 400 response would stop retries immediately.

**Impact**: Malformed webhook payloads cause retry storms instead of immediate rejection.
**Severity**: MEDIUM.

---

#### M-005: No Timeout on `reingest_item()` Vector Store Operations

**File**: `/home/odedbe/projects/hey-seven/src/rag/pipeline.py`, lines 262-378

The `reingest_item()` function calls `retriever.vectorstore.add_texts()` and `collection.get()` / `collection.delete()` with no timeout protection. If the vector store hangs (SQLite lock, Firestore timeout), the CMS webhook handler will block indefinitely, holding the connection open. The retrieval path in `nodes.py` has a 10-second timeout (`_RETRIEVAL_TIMEOUT`), but the write path has none.

```python
# nodes.py (READ path): Has timeout
results = await asyncio.wait_for(
    asyncio.to_thread(search_knowledge_base, query),
    timeout=_RETRIEVAL_TIMEOUT,
)

# pipeline.py (WRITE path): No timeout
retriever.vectorstore.add_texts(texts=[text], metadatas=[metadata], ids=[doc_id])
```

**Impact**: Hung CMS webhook processing blocks a request slot indefinitely.
**Severity**: MEDIUM.

---

#### M-006: `_request_counter` Attribute Created Dynamically in RateLimitMiddleware

**File**: `/home/odedbe/projects/hey-seven/src/api/middleware.py`, line 369

The rate limiter uses `getattr(self, "_request_counter", 0) + 1` to lazily create a counter. This is not initialized in `__init__`, making the class contract unclear and the attribute invisible to type checkers, linters, and IDE autocomplete. Every other middleware in the file initializes all attributes in `__init__`.

```python
# Current: Dynamic attribute creation
self._request_counter = getattr(self, "_request_counter", 0) + 1

# Should be: Initialized in __init__
def __init__(self, app: ASGIApp) -> None:
    self._request_counter: int = 0
```

**Impact**: Code maintenance hazard, inconsistent with codebase patterns.
**Severity**: MEDIUM.

---

### LOW (Minor / Improvement)

#### L-001: Runbook Alert Thresholds Are Not Wired to Any Monitoring System

**File**: `/home/odedbe/projects/hey-seven/docs/runbook.md`, lines 255-263

The runbook defines alert thresholds (retrieval relevance < 70%, validation pass rate < 85%, etc.) but there is no code anywhere in the codebase that emits these metrics, checks these thresholds, or triggers alerts. The `observability/traces.py` module provides span recording but no metric aggregation. The `observability/evaluation.py` module exists but is test-only (`test_eval.py` is explicitly excluded from CI).

These thresholds are aspirational documentation, not implemented operational alerts.

**Impact**: Production incidents will not trigger automated alerts.
**Severity**: LOW -- documented but not implemented.

---

#### L-002: No Container Startup Duration Metric

The runbook mentions cold start latency > 10s as an incident type, but the lifespan function does not log its own startup duration. Adding a monotonic timer around the lifespan setup would provide the data needed to diagnose slow starts.

**Severity**: LOW.

---

#### L-003: `_NON_STREAM_NODES` Does Not Include `NODE_VALIDATE` or `NODE_RESPOND`

**File**: `/home/odedbe/projects/hey-seven/src/agent/graph.py`, line 76-79

The `_NON_STREAM_NODES` frozenset controls which nodes emit `replace` events in SSE. The validate and respond nodes are not in this set, which means their `on_chain_end` events are only captured for source extraction (respond) and lifecycle tracking (both). This is likely correct behavior (validate/respond do not produce user-facing content), but the naming `_NON_STREAM_NODES` is misleading -- it really means "nodes whose AIMessage output should be sent as a replace event." A name like `_REPLACE_EVENT_NODES` would be clearer.

**Severity**: LOW -- naming clarity only.

---

#### L-004: Missing `VECTOR_DB` and `FIRESTORE_PROJECT` in Production Env Var Table

**File**: `/home/odedbe/projects/hey-seven/docs/runbook.md`, lines 339-365

The environment variable reference table lists `VECTOR_DB` with default `chroma` and production `firestore`, but `FIRESTORE_PROJECT` is not listed in the production value column. Since `VECTOR_DB=firestore` requires `FIRESTORE_PROJECT` to be set, this is a runbook gap that could cause deployment failures for a new operator following the runbook.

**Severity**: LOW.

---

#### L-005: `_content_hashes` TTLCache in CMS Webhook Is Process-Scoped

**File**: `/home/odedbe/projects/hey-seven/src/cms/webhook.py`, lines 93-97

The CMS webhook uses a process-scoped TTLCache for content hash change detection. In multi-container Cloud Run (max-instances=10), each container maintains independent hash stores. A CMS update that hits container A will be indexed. The same webhook retry hitting container B will also be indexed (different hash store). With Telnyx-style retries, this means potential double-indexing on CMS updates.

The code comment says "In production this would be backed by Firestore" but this is not implemented. Since CMS updates are infrequent (human-initiated edits), the impact is low, but it should be tracked.

**Severity**: LOW.

---

## Dimension Deep-Dives

### Dimension 6: Docker & DevOps (Score: 7)

**Strengths**:
- Multi-stage build correctly separates build deps from runtime
- Non-root user (`appuser`) with explicit group creation
- Trivy vulnerability scan with pinned version (reproducible)
- Canary deploy with `--no-traffic` + smoke test + automatic rollback
- Exec-form CMD (receives SIGTERM correctly)
- `--cpu-boost` for faster cold start
- `--min-instances=1` avoids scale-to-zero latency

**Weaknesses**:
- Image not pinned by digest (`python:3.12.8-slim-bookworm` could be mutated in the registry)
- No `.dockerignore` validation (`.git`, `reviews/`, `docs/` likely included in build context)
- Smoke test version assertion is echo-only (C-002)
- HEALTHCHECK uses wrong endpoint (H-002)
- WEB_CONCURRENCY env var is unused by CMD (H-004)
- No `--no-log-init` on `useradd` (logs UID assignment noise)
- No `COPY --chown=appuser:appuser` -- files owned by root, readable by appuser but not writable

### Dimension 8: Scalability & Production (Score: 5)

This is the weakest dimension. The codebase has extensive documentation about multi-container deployment but almost no implementation:

1. **Rate limiter**: In-memory, per-container. Documented "TODO (pre-production)" for Cloud Armor.
2. **Circuit breaker**: In-memory, per-container. Documented "current single-container deployment."
3. **Checkpointer**: MemorySaver by default, FirestoreSaver not tested in CI.
4. **Retriever cache**: No lock (C-001).
5. **CMS hash store**: Process-scoped (L-005).
6. **Idempotency tracker**: Process-scoped, singleton creation unprotected (M-003).
7. **Worker count**: Hardcoded to 1 (H-004).

The architecture is designed for single-container demo deployment. The documentation claims production readiness and outlines a multi-container scaling path, but the code has not been hardened for it. With `max-instances=10` in cloudbuild.yaml, this is not hypothetical -- Cloud Run WILL scale to 10 instances under load.

### Dimension 9: Trade-off Documentation (Score: 8)

This is the strongest operational dimension. Nearly every design decision includes an inline comment explaining:
- What was decided
- Why (often citing specific review round and reviewer)
- What the alternative was
- When the decision should be revisited

The feature flag architecture comment in `graph.py` lines 385-418 is particularly good -- it explains build-time vs runtime flags, why all-runtime is impractical, and how to emergency-disable via env var. The circuit breaker docstrings explain monitoring vs control-flow usage and concurrency caveats. The TCPA compliance module explains area code limitations and mitigations with regulatory citations.

The gap: documentation accuracy. The runbook says "/live for liveness" but the Dockerfile uses "/health". The smoke test documents version assertion but does not enforce it. Documentation that overclaims is worse than documentation that underclaims.

---

## Summary of Findings

| Severity | Count | IDs |
|----------|-------|-----|
| CRITICAL | 2 | C-001, C-002 |
| HIGH | 5 | H-001, H-002, H-003, H-004, H-005 |
| MEDIUM | 6 | M-001, M-002, M-003, M-004, M-005, M-006 |
| LOW | 5 | L-001, L-002, L-003, L-004, L-005 |
| **Total** | **18** | |

---

## Top 3 Actionable Fixes

1. **C-002**: Add `if [ "$DEPLOYED_VERSION" != "$COMMIT_SHA" ]; then echo "VERSION MISMATCH"; exit 1; fi` to cloudbuild.yaml Step 7. This is a 3-line fix that closes the stale deployment gap.

2. **C-001**: Add `_retriever_lock = asyncio.Lock()` and wrap `_get_retriever_cached()` with `async with _retriever_lock:`. Convert it to `async def`. Matches the pattern already used by `_get_llm()`, `_get_circuit_breaker()`, and `get_checkpointer()`.

3. **H-001**: Replace `@lru_cache(maxsize=1)` on `get_settings()` with a TTLCache (1-hour TTL) to match every other singleton in the codebase. Or accept that credential rotation requires container restart and document that explicitly (rather than having the TTL-cached LLM singletons create a false sense of rotation support).

---

## Verdict

The codebase has matured significantly in graph architecture, guardrails, and documentation. The deployment pipeline is well-structured with canary deploys and automatic rollback. However, operational readiness for multi-container production has stalled:

- The deployment pipeline has an unenforced version assertion gate
- The retriever singleton is the only cache without a lock
- The entire credential rotation architecture is undermined by `@lru_cache` on `get_settings()`
- The documented scaling path (`WEB_CONCURRENCY`) is non-functional

Score delta of -2 reflects newly discovered inconsistencies (retriever lock, settings TTL, smoke test gap) that were present but unexamined in R15. The codebase is a strong demo platform but is not production-ready for multi-tenant, multi-container deployment without addressing the CRITICAL and HIGH findings.

**Score: 70/100**
