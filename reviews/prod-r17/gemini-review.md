# Hey Seven Production Review -- Round 17 (Gemini Focus)

**Reviewer**: Gemini (simulated hostile production review)
**Commit**: 3d838bf
**Date**: 2026-02-21
**Focus**: Architecture coherence, dead code, documentation honesty, design pattern consistency
**Prior Rounds**: R11-R16 reviewed, trajectory noted

---

## Score Summary

| Dimension | Score (0-10) | Notes |
|-----------|-------------|-------|
| Graph Architecture | 8 | Clean 11-node topology, well-documented dual-layer routing |
| RAG Pipeline | 8 | Per-item chunking, RRF, version-stamp purging all solid |
| Data Model | 7 | ProfileField schema mature; completeness calc weighted properly |
| API Design | 8 | Pure ASGI middleware, SSE heartbeats, structured errors |
| Testing Strategy | 7 | 1452 tests claimed but not audited in this round |
| Docker & DevOps | 7 | Cloud Run probe design is correct; known limitation documented |
| Prompts & Guardrails | 8 | 5-layer deterministic guardrails + semantic LLM classifier |
| Scalability & Production | 6 | Several in-memory singletons with known multi-instance gaps |
| Trade-off Documentation | 9 | Exceptionally honest inline documentation of decisions |
| Domain Intelligence | 8 | Casino-domain patterns (BSA/AML, TCPA, patron privacy) are strong |

**Overall Score: 76/100**

---

## Findings

### CRITICAL (0)

No critical findings in this round. Previous critical issues from R11-R16 have been addressed.

---

### HIGH (3)

#### H-001: Whisper planner failure counter uses `global` statement -- not async-safe in multi-worker scenarios

**File**: `/home/odedbe/projects/hey-seven/src/agent/whisper_planner.py`, lines 88-91, 134, 171-186

The `_failure_count` and `_failure_alerted` variables use `global` keyword with an `asyncio.Lock` for protection. The R16 fix added the lock (correctly addressing the race condition within a single process), but the `global` statement itself is a code smell that makes the intent less clear. More importantly, the pattern of `global _failure_count, _failure_alerted` inside an async function interacts poorly with code analysis tools and makes the mutation sites harder to grep.

The lock does protect correctness for a single event loop. However, the pattern would be cleaner as a module-level dataclass or a simple namespace object:

```python
class _WhisperTelemetry:
    count: int = 0
    alerted: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
```

This is a HIGH because `global` mutation in async code is a maintenance footgun -- a future developer could add a second event loop (e.g., testing) and silently corrupt the counter. The lock only protects within one loop.

**Severity**: HIGH
**Effort**: Low (30 min refactor)

---

#### H-002: `_get_retriever_cached()` uses `threading.Lock` but could be called from the event loop

**File**: `/home/odedbe/projects/hey-seven/src/rag/pipeline.py`, lines 898-991

The R16 fix correctly identified that `_get_retriever_cached()` runs inside `asyncio.to_thread()` and therefore needs `threading.Lock` (not `asyncio.Lock`). This is correct for the current call path.

However, `get_retriever()` is also called directly from:
- `app.py:260` -- `get_retriever()` called from the health endpoint (async context, NOT wrapped in `to_thread`)
- `reingest_item()` at line 285 -- also async context

If `get_retriever(persist_dir=None)` is called from the async health check (which it is -- line 260 calls `get_retriever()` with no args), it delegates to `_get_retriever_cached()`, which acquires a `threading.Lock`. A `threading.Lock` acquired from the event loop blocks the entire event loop until released. Under normal operation this is near-instant, but if the lock contends with a concurrent `to_thread` worker doing retriever initialization (which involves ChromaDB/Firestore client creation -- potentially seconds), the event loop freezes.

The health endpoint calling `get_retriever()` on every `/health` request makes this a realistic contention scenario.

**Severity**: HIGH
**Effort**: Medium -- either wrap the health check's `get_retriever()` call in `to_thread`, or use a dual-lock strategy (asyncio.Lock for coroutine callers, threading.Lock for thread callers).

---

#### H-003: `_LLM_SEMAPHORE` in `_base.py` is module-level with fixed value, not configurable

**File**: `/home/odedbe/projects/hey-seven/src/agent/agents/_base.py`, line 34

The semaphore value of 20 is hardcoded. In production, this should be tunable via settings because:

1. Different Gemini API tiers have different QPS limits (60-300 RPM depending on the plan).
2. During an incident, operators may want to reduce concurrency to avoid rate limiting without redeploying.
3. The value cannot be changed without a container restart.

This is a HIGH because a burst of concurrent requests exceeding the semaphore could cause request queuing, and tuning requires a code change rather than an env var.

**Severity**: HIGH
**Effort**: Low (add `LLM_MAX_CONCURRENT` to Settings, reference in `_base.py`)

---

### MEDIUM (5)

#### M-001: `StreamingPIIRedactor._buffer` accessed directly in `graph.py` error handler

**File**: `/home/odedbe/projects/hey-seven/src/agent/graph.py`, line 715

```python
len(_pii_redactor._buffer)
```

Direct access to a private attribute (`_buffer`) from outside the class. The `StreamingPIIRedactor` class should expose a `buffer_size` property or method for safe external inspection. This is a minor encapsulation violation but one that a future refactor of `StreamingPIIRedactor` could silently break.

**Severity**: MEDIUM
**Effort**: Trivial (add `@property def buffer_size(self) -> int`)

---

#### M-002: Greeting node calls `get_settings()` twice

**File**: `/home/odedbe/projects/hey-seven/src/agent/nodes.py`, lines 492-497

```python
settings = get_settings()
categories = _build_greeting_categories(casino_id=settings.CASINO_ID)
# ...
ai_disclosure = await is_feature_enabled(get_settings().CASINO_ID, "ai_disclosure_enabled")
```

`get_settings()` is called twice -- once stored in `settings`, then again inline for the `is_feature_enabled` call. While `get_settings()` is `@lru_cache` and returns the same instance, the duplicate call is sloppy. Similar double-call patterns exist in `off_topic_node` (line 550: `get_settings().CASINO_ID`).

**Severity**: MEDIUM
**Effort**: Trivial

---

#### M-003: `_dispatch_to_specialist` does not timeout the LLM dispatch call

**File**: `/home/odedbe/projects/hey-seven/src/agent/graph.py`, lines 206-236

The structured LLM dispatch (`dispatch_llm.ainvoke(prompt)`) has no explicit timeout. The LLM constructor specifies `timeout=settings.MODEL_TIMEOUT` (30s), which applies to the HTTP call. But if the Google GenAI SDK has internal retries (configured at `MODEL_MAX_RETRIES=2`), the effective timeout could be 30s x 3 = 90 seconds for a routing decision.

A 90-second routing delay is unacceptable -- the keyword fallback should fire much sooner (e.g., 5 seconds). The generate node already has the full LLM timeout budget; the dispatch decision should be fast or fall back.

**Severity**: MEDIUM
**Effort**: Low (wrap in `asyncio.wait_for(dispatch_llm.ainvoke(prompt), timeout=5)`)

---

#### M-004: No input validation on `ChatRequest.message` length

**File**: `/home/odedbe/projects/hey-seven/src/api/app.py`, line 148

While `RequestBodyLimitMiddleware` caps the total request body at 64KB, there is no explicit validation on the `message` field length within `ChatRequest`. A 64KB JSON body could contain a 60KB+ message string, which would be sent in its entirety through the guardrails (regex scanning) and then to the LLM. The guardrail regex patterns run in O(n) over the message length, and extremely long messages waste LLM tokens.

A `max_length=4096` constraint on the Pydantic model's `message` field would provide defense-in-depth.

**Severity**: MEDIUM
**Effort**: Trivial (one-line Pydantic field constraint)

---

#### M-005: `_load_knowledge_base_markdown` calls `rglob` twice

**File**: `/home/odedbe/projects/hey-seven/src/rag/pipeline.py`, lines 533, 568

```python
for md_file in sorted(base_path.rglob("*.md")):
    # ... processing
logger.info("... %d markdown files in %s",
            len(documents), len(list(base_path.rglob("*.md"))), base_path)
```

`rglob("*.md")` is called twice -- once for processing and once just for the log message count. The second call re-walks the filesystem. Should be `len(files)` where `files` is captured once.

**Severity**: MEDIUM
**Effort**: Trivial

---

### LOW (4)

#### L-001: `_CATEGORY_PRIORITY` does not cover all keys in `_CATEGORY_TO_AGENT`

**File**: `/home/odedbe/projects/hey-seven/src/agent/graph.py`, lines 118-138

`_CATEGORY_TO_AGENT` maps "restaurants", "entertainment", "spa", "gaming", "promotions", "hotel" to agents. `_CATEGORY_PRIORITY` covers the same keys but with implicit 0 for unlisted categories. The tie-break comment says "alphabetical for categories not in _CATEGORY_PRIORITY" which is correct behavior. However, a parity assertion (similar to the `_initial_state` parity check) would prevent silent drift if either dict is updated independently.

**Severity**: LOW
**Effort**: Trivial

---

#### L-002: `BoundedMemorySaver` does not subclass any checkpointer protocol

**File**: `/home/odedbe/projects/hey-seven/src/agent/memory.py`, lines 36-119

`BoundedMemorySaver` delegates to `MemorySaver` but does not inherit from `BaseCheckpointSaver` or any protocol class. This means LangGraph's type system cannot validate that it implements the full checkpointer contract. If a future LangGraph version adds a new required method, this will fail at runtime rather than at import/type-check time.

**Severity**: LOW
**Effort**: Low (add `BaseCheckpointSaver` as parent class)

---

#### L-003: `clear_firestore_client_cache()` is sync but `_firestore_client_lock` is asyncio.Lock

**File**: `/home/odedbe/projects/hey-seven/src/data/guest_profile.py`, lines 115-117

The `clear_firestore_client_cache()` function clears the cache dict without acquiring the `asyncio.Lock`. While `dict.clear()` is atomic under GIL, this could race with `_get_firestore_client()` which does a double-check pattern. If `clear` happens between the fast-path check and the slow-path lock acquisition, the slow path could cache a stale client reference.

In practice, this function is called from tests and shutdown -- low risk. But the asymmetry is notable.

**Severity**: LOW
**Effort**: Trivial

---

#### L-004: `off_topic_node` does not handle `"prompt_injection"` query_type explicitly

**File**: `/home/odedbe/projects/hey-seven/src/agent/nodes.py`, lines 517-617

The `compliance_gate_node` sets `query_type="off_topic"` for prompt injection detections (not a distinct type). This means prompt injection attempts get the generic "I'm your concierge" message, which is fine from a security perspective (no information leakage). However, for observability, there is no way to distinguish "genuinely off-topic" from "injection blocked" in the off_topic_node's response path. Consider adding a state field like `_guardrail_triggered` for monitoring (not routing).

**Severity**: LOW
**Effort**: Low

---

## R11-R16 Improvements Acknowledged

The codebase shows clear evidence of iterative improvement across the 16 prior review rounds:

1. **R10**: `assert` statements converted to `ValueError` (production-safe), parity checks at module level, CB `failure_count` property made read-only.
2. **R11**: CB `record_cancellation()` added for SSE disconnects, stale-client sweep in rate limiter, `clear_circuit_breaker_cache()` for incident response.
3. **R14**: `reingest_item()` fixed property_id derivation, `_ingestion_version` added for CMS updates.
4. **R15**: CB acquired before try block (UnboundLocalError fix), node name constants in routing, FeatureFlags parity check.
5. **R16**: `_failure_lock` in whisper planner, `_request_counter` initialized in `__init__`, `threading.Lock` for retriever (to_thread context), Firestore client lock converted to asyncio.Lock.

The documentation quality is exceptional -- every fix includes the review round, finding ID, and reviewer consensus that motivated it. This is the best-documented fix trail I have seen in a production codebase.

---

## Architecture Coherence Assessment

The architecture is coherent and well-structured:

- **Graph topology** is clean: 11 nodes, clearly documented edges, two-layer routing (compliance_gate -> router).
- **DRY extraction** via `_base.py` eliminates 80% duplication across 5 specialist agents. Each agent is a thin 30-50 line wrapper. This was praised in prior rounds and remains the standout structural decision.
- **Feature flag dual-layer design** (build-time topology vs runtime behavior) is correctly implemented with extensive inline documentation explaining WHY not all runtime.
- **Singleton caching** follows a consistent pattern: TTLCache(maxsize=1, ttl=3600) + asyncio.Lock across LLM, validator, whisper, circuit breaker, and checkpointer. The one exception (threading.Lock for retriever) is correctly justified.
- **Error handling** follows principled degradation: fail-open for availability (circuit breaker, degraded-pass validation) vs fail-closed for safety (PII redaction, semantic injection classifier).
- **State schema** has proper parity checks at import time, preventing drift between `PropertyQAState`, `_initial_state()`, and `FeatureFlags`/`DEFAULT_FEATURES`.

The one area of concern is the growing number of module-level singletons (LLM cache, validator cache, whisper cache, CB cache, retriever cache, checkpointer cache, greeting cache, flag cache, Firestore client cache, memory store, failure counter). Each has its own lock and TTL pattern. A future cleanup could consolidate these into a `ServiceRegistry` or similar pattern, but the current approach is functional and each cache has clear documentation.

---

## Documentation Honesty Audit

Documentation is honest and uses correct vocabulary:
- "Placeholder" is labeled as placeholder (whisper_planner `_calculate_completeness` -- line 244: "This is a placeholder").
- "Not yet implemented" features are clearly marked (Redis-backed CB, CB_BACKEND config -- circuit_breaker.py line 299).
- Known limitations are explicitly documented with TODO tickets (rate limiter Cloud Run limitation, distributed state).
- No overclaiming observed -- "scaffolded" vs "implemented" distinction is maintained.

---

## Dead Code Audit

No dead code identified in the reviewed files. All imports are used, all functions are called, all node constants map to graph nodes. The `_NON_STREAM_NODES` and `_KNOWN_NODES` frozensets correctly enumerate the actual graph nodes.

The only borderline case is `list_agents()` in `registry.py` -- it is not called from production code but is a diagnostic utility. Acceptable.

---

## Production Readiness Assessment

### VERDICT: CONDITIONAL GO

**Conditions for unconditional GO**:
1. Fix H-002 (threading.Lock contention on event loop from health endpoint) -- this is a real production risk under load.
2. Fix M-003 (dispatch LLM timeout) -- 90-second routing delays will cause visible user-facing latency.
3. Make `_LLM_SEMAPHORE` configurable (H-003) -- needed for operational flexibility.

**Rationale**: The architecture is sound, the guardrail layers are comprehensive, the error handling follows principled fail-open/fail-closed patterns, and the code quality is high. The three HIGH findings are all operational concerns rather than architectural defects -- they affect production behavior under stress but do not represent design flaws. The documentation trail is exceptional.

The codebase is production-ready for a **controlled launch** (single casino, limited traffic) with the condition fixes applied before scaling to multiple casinos or high-traffic periods.

---

## Score Trajectory Context

| Model | R11 | R12 | R13 | R14 | R15 | R16 | R17 |
|-------|-----|-----|-----|-----|-----|-----|-----|
| Gemini | 86 | 84 | 83 | 82 | 80 | 74 | 76 |

R17 score of 76 represents a stabilization and slight recovery from R16's 74. The R16 drop reflected legitimate findings around lock type correctness and initialization patterns, most of which have been fixed. The remaining findings in R17 are lower severity (no CRITICAL) and reflect operational hardening rather than fundamental defects.
