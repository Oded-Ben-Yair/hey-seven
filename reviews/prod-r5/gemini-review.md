# Round 5 Production Review — Gemini (Scalability Spotlight)

**Date**: 2026-02-20
**Commit**: 95d9264
**Reviewer**: Gemini 3 Pro (thinking=high)
**Spotlight**: SCALABILITY & ASYNC PATTERNS (+1 severity)
**Tests**: 1157 passed, 20 skipped, 91.27% coverage

---

## Codebase Scalability Summary

### In-Memory vs External State

| Component | Storage | Bound | TTL | Issue |
|-----------|---------|-------|-----|-------|
| Rate limiter | OrderedDict + asyncio.Lock | max_clients=10K | None (sliding window) | Process-scoped; no cross-replica enforcement |
| Circuit breaker | deque + asyncio.Lock | max(100, threshold*10) | Rolling 5-min window | Process-scoped; acceptable for per-container health |
| MemorySaver checkpointer | In-memory dict | **NONE** | **NONE** | Unbounded growth per thread_id; OOM risk |
| Idempotency tracker | dict + asyncio.Lock | _MAX_ENTRIES=10K | 3600s TTL | Process-scoped; webhook replay across replicas |
| Delivery log | TTLCache | maxsize=10K | 86400s (24h) | Process-scoped; acceptable for monitoring |
| CMS content hashes | TTLCache | maxsize=10K | 86400s (24h) | Process-scoped; acceptable for change detection |
| LLM clients | TTLCache | maxsize=1 each | 3600s (1h) | Correct — credential rotation handled |
| Settings | @lru_cache | maxsize=1 | **NONE** | Env var changes ignored until restart |
| Embeddings | @lru_cache | maxsize=4 | **NONE** | No credential rotation support |
| Retriever | @lru_cache | maxsize=1 | **NONE** | No credential rotation support |
| Langfuse client | @lru_cache | maxsize=1 | **NONE** | No credential rotation support |

### Async Patterns

| Pattern | Implementation | Verdict |
|---------|---------------|---------|
| LLM singleton locks | Separate asyncio.Lock per LLM type | Correct — prevents cross-lock contention |
| ChromaDB retrieval | asyncio.to_thread + 10s timeout | Correct — sync call offloaded |
| Rate limiter lock | Single asyncio.Lock for OrderedDict | Acceptable for single-worker |
| Idempotency lock | Single asyncio.Lock for dict | Acceptable for single-worker |
| SSE disconnect check | request.is_disconnected() in event_generator | Present in app.py:180 |
| SSE heartbeat | 15s interval to prevent client timeout | Correct |
| SSE timeout | asyncio.timeout(sse_timeout) wrapping stream | Correct |
| Event loop blocking | ChromaDB via to_thread, property JSON via sync open | Sync file I/O in lifespan (startup-only, acceptable) |

### Connection Management

| Client | Lifecycle | Issue |
|--------|-----------|-------|
| LLM (main) | TTLCache, 1h refresh | Correct |
| LLM (validator) | TTLCache, 1h refresh, separate lock | Correct |
| LLM (whisper) | TTLCache, 1h refresh, separate lock | Correct |
| ChromaDB | @lru_cache, no TTL | No rotation needed (local SQLite) |
| Firestore (checkpointer) | @lru_cache, no TTL | **Missing credential rotation** |
| Firestore (retriever) | @lru_cache, no TTL | **Missing credential rotation** |
| Langfuse | @lru_cache, no TTL | **Missing credential rotation** |

### Resource Bounds

| Resource | Bound | Eviction |
|----------|-------|----------|
| Rate limiter clients | 10,000 | LRU (OrderedDict) |
| Rate limiter window | 60s sliding | deque popleft |
| Circuit breaker failures | max(100, threshold*10) | deque maxlen |
| Idempotency IDs | 10,000 | Manual prune under lock |
| Delivery log | 10,000 | TTLCache auto-evict |
| CMS hashes | 10,000 | TTLCache auto-evict |
| Message history | MAX_HISTORY_MESSAGES=20 (to LLM) | Sliding window |
| Total messages | MAX_MESSAGE_LIMIT=40 | Forced off_topic |
| Request body | 64KB | 413 rejection |
| LLM output tokens | 2048 (main), 512 (validator/whisper) | Model config |
| Graph recursion | 10 | LangGraph built-in |

---

## Score Table

| # | Dimension | Score | Justification |
|---|-----------|:-----:|---------------|
| 1 | Graph/Agent Architecture | 8 | Clean 11-node StateGraph with validation loop, specialist dispatch via DRY base, and proper conditional edges; MemorySaver default is the only concern. |
| 2 | RAG Pipeline | 7 | Per-item chunking, SHA-256 idempotent ingestion, RRF reranking, multi-tenant property_id filtering all solid; retriever @lru_cache lacks TTL for GCP credential rotation. |
| 3 | Data Model / State Design | 8 | Per-turn reset via _initial_state(), parity assertion, Annotated[list, add_messages] reducer, proper TypedDict; no mutable defaults found. |
| 4 | API Design | 7 | Pure ASGI middleware stack, SSE streaming with PII buffer, hmac.compare_digest auth, disconnect detection present; lacks explicit SIGTERM handler for in-flight stream cleanup. |
| 5 | Testing Strategy | 7 | 1157 tests at 91.27% coverage with concurrency tests for circuit breaker and rate limiter; missing load tests and memory pressure tests for MemorySaver growth. |
| 6 | Docker & DevOps | 6 | Multi-stage build, non-root user, pinned base, healthcheck with start-period; WEB_CONCURRENCY=1 with --workers 1 prevents vertical scaling without reconfig. |
| 7 | Prompts & Guardrails | 8 | 84+ regex patterns across 4 languages, semantic injection classifier with fail-closed, 5 guardrail categories; prompt length not capped (potential latency spike from long history). |
| 8 | Scalability & Production | 3 | **SPOTLIGHT.** Process-scoped rate limiter and idempotency tracker break under multi-replica Cloud Run; MemorySaver has no eviction and will OOM; no LLM concurrency backpressure; @lru_cache singletons lack TTL for credential rotation. |
| 9 | Documentation & Code Quality | 7 | Extensive docstrings, clear naming conventions, well-structured modules; documentation does not warn operators about single-replica state limitations. |
| 10 | Domain Intelligence | 8 | TCPA keyword handling, quiet hours, BSA/AML, responsible gaming escalation, Ed25519 webhook verification, consent hash chain; solid regulatory coverage. |
| **Total** | | **69** | |

**Previous**: R1=67.3, R2=61.3, R3=60.7, R4=66.7
**Current**: R5 Gemini=69

---

## Findings

### Finding 1 (CRITICAL): MemorySaver checkpointer has no eviction — unbounded memory growth

- **Location**: `src/agent/memory.py:56`, `src/agent/graph.py:342`
- **Problem**: `MemorySaver()` stores the full state history (all messages, context, metadata) for every `thread_id` in a plain Python dict. There is no maxsize, no TTL, no eviction policy. Each conversation thread adds ~10-50KB of state that is never reclaimed.
- **Impact**: In a 24/7 casino concierge, the process heap grows monotonically. With 1000 conversations/day at ~20KB each, that is 20MB/day of leaked memory. On a Cloud Run container with 512MB RAM, OOM kill within 2 weeks. Container restarts lose ALL conversation state.
- **Fix**: Add an LRU eviction wrapper around MemorySaver for development, and document the production requirement more prominently:
```python
# memory.py — development guard
from langgraph.checkpoint.memory import MemorySaver

MAX_ACTIVE_THREADS = 1000  # Evict oldest when exceeded

class BoundedMemorySaver(MemorySaver):
    """MemorySaver with LRU eviction for development use."""

    def __init__(self, max_threads: int = MAX_ACTIVE_THREADS):
        super().__init__()
        self._max_threads = max_threads

    async def aput(self, config, checkpoint, metadata, new_versions):
        # Evict oldest if at capacity
        if hasattr(self, 'storage') and len(self.storage) >= self._max_threads:
            oldest_key = next(iter(self.storage))
            del self.storage[oldest_key]
        return await super().aput(config, checkpoint, metadata, new_versions)
```

### Finding 2 (CRITICAL → HIGH with spotlight): Process-scoped rate limiter breaks under multi-replica deployment

- **Location**: `src/api/middleware.py:284-391`
- **Problem**: Rate limiter uses an in-memory `OrderedDict` with `asyncio.Lock`. When Cloud Run scales to 2+ containers, each container maintains independent rate limit counters. A user rate-limited on Container A can make unlimited requests via Container B.
- **Impact**: Rate limiting is effectively `N * limit` where N = number of containers. An attacker can bypass rate limits entirely by forcing load balancer distribution across containers. LLM cost amplification attack becomes trivial.
- **Fix**: Document the single-replica limitation explicitly in the middleware docstring and in the deployment guide. For multi-replica, implement Redis-backed sliding window:
```python
# Future: Redis-backed rate limiter
# async def _is_allowed_redis(self, client_ip: str) -> bool:
#     key = f"rate:{client_ip}"
#     current = await redis.incr(key)
#     if current == 1:
#         await redis.expire(key, self.window_seconds)
#     return current <= self.max_tokens
```

### Finding 3 (HIGH → CRITICAL with spotlight): No LLM concurrency backpressure — 100 simultaneous requests all hit LLM API

- **Location**: `src/agent/agents/_base.py:136`, `src/agent/nodes.py:220`
- **Problem**: There is no semaphore or concurrency gate limiting how many LLM calls can be in-flight simultaneously. The circuit breaker only trips AFTER failures accumulate — it does not prevent the initial flood. If 100 requests arrive in 1 second, all 100 attempt concurrent LLM API calls.
- **Impact**: (1) LLM provider rate limiting causes cascading 429 errors. (2) httpx connection pool exhaustion. (3) Circuit breaker trips from API throttling, not actual outages, blocking ALL users for the cooldown period. (4) Google Gemini API has per-project QPS limits (typically 60-300 RPM for Flash) — exceeding this returns 429 which counts as circuit breaker failures.
- **Fix**: Add a module-level semaphore gating all LLM calls:
```python
# src/agent/nodes.py
_LLM_SEMAPHORE = asyncio.Semaphore(50)  # Max concurrent LLM calls

async def _get_llm_with_backpressure() -> ChatGoogleGenerativeAI:
    await _LLM_SEMAPHORE.acquire()
    try:
        return await _get_llm()
    except Exception:
        _LLM_SEMAPHORE.release()
        raise

# Or wrap at the call site in _base.py:
async with _LLM_SEMAPHORE:
    response = await llm.ainvoke(llm_messages)
```

### Finding 4 (HIGH): @lru_cache singletons without TTL prevent credential rotation for non-LLM clients

- **Location**: `src/rag/embeddings.py:20`, `src/rag/pipeline.py:527`, `src/observability/langfuse_client.py:26`, `src/agent/circuit_breaker.py:243`
- **Problem**: While LLM clients correctly use `TTLCache(ttl=3600)` for credential rotation, these four singletons use `@lru_cache` which never expires. If GCP Workload Identity Federation rotates credentials (every 1 hour by default), these clients hold stale credentials indefinitely.
- **Impact**: After credential rotation, embeddings calls fail with 401/403 (stale API key). Retriever fails. Langfuse calls fail. None of these recover without process restart. The LLM clients work fine (TTL cache) while all supporting services break.
- **Fix**: Replace `@lru_cache` with TTLCache pattern for credential-sensitive clients:
```python
# src/rag/embeddings.py
_embeddings_cache: TTLCache = TTLCache(maxsize=4, ttl=3600)
_embeddings_lock = asyncio.Lock()

async def get_embeddings(task_type: str | None = None) -> GoogleGenerativeAIEmbeddings:
    key = task_type or "__default__"
    async with _embeddings_lock:
        cached = _embeddings_cache.get(key)
        if cached is not None:
            return cached
        settings = get_settings()
        emb = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            **({"task_type": task_type} if task_type else {}),
        )
        _embeddings_cache[key] = emb
        return emb
```

### Finding 5 (HIGH): get_settings() @lru_cache prevents runtime configuration updates

- **Location**: `src/config.py:162`
- **Problem**: `get_settings()` is decorated with `@lru_cache(maxsize=1)`. Once called, settings are frozen for the process lifetime. Environment variable changes (including secret rotation via Cloud Run secret references) require a full container restart.
- **Impact**: (1) API key rotation requires deployment, not just secret update. (2) Feature flag changes (SEMANTIC_INJECTION_ENABLED, SMS_ENABLED) require restart. (3) Rate limit tuning requires restart. This negates Cloud Run's "rolling secret update" capability where env vars can be updated without redeployment.
- **Fix**: This is a known trade-off — settings loading is fast enough to call on every request, but the codebase relies on `get_settings()` being cached for consistency within a single request. Document the limitation explicitly. For secrets specifically, consider a separate `get_secrets()` function with TTLCache:
```python
# Option 1: Document the limitation
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings. NOTE: Requires process restart for changes."""
    return Settings()

# Option 2: TTL for secrets-sensitive settings
_settings_cache: TTLCache = TTLCache(maxsize=1, ttl=300)  # 5-min refresh
```

### Finding 6 (HIGH): Webhook idempotency tracker is process-scoped — duplicates across replicas

- **Location**: `src/sms/webhook.py:249-279`
- **Problem**: `WebhookIdempotencyTracker` stores processed message IDs in a process-local dict. When running multiple Cloud Run replicas, Telnyx webhook retries can hit different containers and bypass deduplication.
- **Impact**: A single SMS triggers agent processing twice (or more), resulting in duplicate responses to the guest. In a regulated casino environment, this could mean duplicate comp offers, duplicate responsible gaming escalations, or duplicate consent acknowledgments — all compliance-sensitive.
- **Fix**: For the current single-replica deployment, this is acceptable. Document the limitation. For multi-replica, migrate to Redis SETNX with TTL:
```python
# Future: Redis-backed idempotency
# async def is_duplicate_redis(self, message_id: str) -> bool:
#     result = await redis.set(f"idem:{message_id}", "1", nx=True, ex=self._ttl)
#     return result is None  # None means key already existed
```

### Finding 7 (MEDIUM → HIGH with spotlight): WEB_CONCURRENCY=1 with --workers 1 — no vertical scaling

- **Location**: `Dockerfile:41`, `Dockerfile:57-59`
- **Problem**: The Dockerfile hardcodes `WEB_CONCURRENCY=1` and runs uvicorn with `--workers 1`. A single asyncio event loop handles all concurrent requests. While async I/O enables concurrency for I/O-bound tasks, any CPU-bound operation (regex guardrails, PII redaction, JSON serialization) blocks ALL concurrent requests.
- **Impact**: (1) PII regex on a large LLM response blocks all SSE streams. (2) A single stuck request (e.g., ChromaDB to_thread timeout) occupies the thread pool. (3) Healthcheck can be delayed by a CPU-bound task, causing Cloud Run to mark the container unhealthy.
- **Fix**: Update Dockerfile to use the environment variable for workers and set a reasonable default for Cloud Run:
```dockerfile
# Let Cloud Run vCPU allocation determine workers
CMD ["python", "-m", "uvicorn", "src.api.app:app", \
    "--host", "0.0.0.0", "--port", "8080", \
    "--workers", "${WEB_CONCURRENCY:-2}", \
    "--timeout-graceful-shutdown", "10"]
```
Note: Multiple workers require external state for rate limiter and idempotency tracker (Finding 2, 6).

### Finding 8 (MEDIUM): Sync file I/O in lifespan startup blocks event loop

- **Location**: `src/api/app.py:93-94`
- **Problem**: Property metadata loading uses synchronous `open()` and `json.load()` in the async lifespan context manager. While this only runs once at startup, the pattern sets a bad precedent and could block the event loop if the file is on a network mount (e.g., Cloud Run volumes, NFS).
- **Impact**: On network-mounted filesystems, a slow read could delay startup beyond the Cloud Run startup probe timeout (default 240s), causing the container to be killed and restarted in a loop.
- **Fix**: Wrap in `asyncio.to_thread()` for consistency with the ChromaDB ingestion pattern on line 80:
```python
if property_path.exists():
    app.state.property_data = await asyncio.to_thread(
        lambda: json.loads(property_path.read_text(encoding="utf-8"))
    )
```

### Finding 9 (MEDIUM): No explicit graceful shutdown for in-flight SSE streams

- **Location**: `src/api/app.py:100-104`
- **Problem**: The lifespan shutdown sets `app.state.ready = False` and `app.state.agent = None`, but does not wait for in-flight SSE streams to complete. Setting `agent = None` while a stream is actively using it causes `AttributeError` or `TypeError` in the graph's `astream_events`.
- **Impact**: During deployments, active SSE streams crash mid-response with an unhandled exception instead of receiving a graceful `done` event. Clients see broken streams and may retry, creating duplicate requests.
- **Fix**: Track active SSE connections and drain during shutdown:
```python
# In lifespan:
app.state._active_streams = 0
app.state._shutdown_event = asyncio.Event()

# In shutdown:
app.state.ready = False  # Stop accepting new requests
if app.state._active_streams > 0:
    await asyncio.wait_for(app.state._shutdown_event.wait(), timeout=8)
app.state.agent = None
```

### Finding 10 (MEDIUM): PII buffer accumulation in SSE stream has no max-size guard

- **Location**: `src/agent/graph.py:498-499`
- **Problem**: The `_pii_buffer` accumulates streamed tokens and flushes at 80 chars or sentence boundaries. If the LLM generates a long sequence of digits without sentence boundaries (e.g., a table of numbers, a long formatted phone directory), the buffer grows without bound until the stream ends.
- **Impact**: Memory spike proportional to LLM output length for digit-heavy responses. While MODEL_MAX_OUTPUT_TOKENS=2048 provides an implicit cap, the buffer could reach ~8KB in the worst case (2048 tokens * ~4 chars/token). Low practical risk but violates the "no unbounded collections" principle.
- **Fix**: Add a hard cap to force-flush:
```python
_PII_FLUSH_LEN = 80
_PII_MAX_BUFFER = 500  # Hard cap: flush regardless of content

elif len(_pii_buffer) >= _PII_FLUSH_LEN or len(_pii_buffer) >= _PII_MAX_BUFFER or _pii_buffer.endswith(("\n", ". ", "! ", "? ")):
```

### Finding 11 (LOW): LangFuse callback handler created per-request without connection pooling

- **Location**: `src/observability/langfuse_client.py:115-128`
- **Problem**: `get_langfuse_handler()` creates a new `CallbackHandler` instance per sampled request. Each handler potentially opens its own HTTP connection to the LangFuse server. With 10% sampling at 100 RPM, that is 10 new HTTP connections per minute.
- **Impact**: Connection overhead and potential connection exhaustion under load. Minor because LangFuse handlers typically batch internally, but the per-request instantiation pattern is wasteful.
- **Fix**: Consider reusing a single CallbackHandler with per-trace metadata injection, or ensure the LangFuse client uses connection pooling (verify upstream library behavior).

---

## Scalability Verdict

The codebase is well-engineered for a **single-container demo deployment**. The async patterns are correct (proper asyncio.Lock usage, no threading.Lock in async code, to_thread for sync calls). The bounded data structures (TTLCache, deque maxlen, MAX_ACTIVE_THREADS concept) show awareness of memory management.

However, the architecture has **fundamental single-process assumptions** that prevent horizontal scaling:
1. Rate limiter, idempotency, and content hashes are all process-scoped
2. MemorySaver grows without bound
3. No LLM concurrency backpressure
4. @lru_cache singletons miss credential rotation for non-LLM clients
5. Single uvicorn worker prevents vertical scaling

For the Hey Seven interview context (demo/MVP), these are **documented trade-offs**. For production at casino scale (24/7, multi-region, thousands of concurrent guests), all five issues need to be addressed before launch.
