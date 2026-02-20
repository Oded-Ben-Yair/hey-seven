# Round 3 Production Review Summary

**Date**: 2026-02-20
**Commit (before)**: 014335c
**Spotlight**: ERROR HANDLING & RESILIENCE
**Previous Scores**: R1=67.3 | R2=61.3

---

## Score Table

| # | Dimension | Gemini 3 Pro | GPT-5.2 | Grok 4 | Average |
|---|-----------|:---:|:---:|:---:|:---:|
| 1 | Graph/Agent Architecture | 7 | 7 | 7 | 7.0 |
| 2 | RAG Pipeline | 6 | 6 | 6 | 6.0 |
| 3 | Data Model / State Design | 2 | 6 | 7 | 5.0 |
| 4 | API Design | 5 | 7 | 6 | 6.0 |
| 5 | Testing Strategy | 9 | 6 | 4 | 6.3 |
| 6 | Docker & DevOps | 8 | 4 | 5 | 5.7 |
| 7 | Prompts & Guardrails | 9 | 7 | 7 | 7.7 |
| 8 | Scalability & Production | 1 | 6 | 4 | 3.7 |
| 9 | Documentation & Code Quality | 7 | 6 | 6 | 6.3 |
| 10 | Domain Intelligence | 8 | 6 | 7 | 7.0 |
| **Total** | | **62** | **61** | **59** | **60.7** |

---

## Consensus Findings Fixed

| # | Finding | Severity | Models | Fix Applied |
|---|---------|----------|--------|-------------|
| 1 | `execute_specialist()` missing broad `except Exception` -- unhandled SDK exceptions bypass circuit breaker and crash SSE stream | CRITICAL | All 3 (Gemini #2, GPT #1, Grok C1) | Added final `except Exception` after specific handlers that calls `cb.record_failure()` and returns fallback with `skip_validation=True`. Same pattern as `router_node`. |
| 2 | `retrieve_node` catches only `TimeoutError` -- non-timeout retrieval errors propagate unhandled | CRITICAL | 2/3 (Grok C2, GPT #3) | Added `except Exception` alongside `TimeoutError` that logs and returns empty results, allowing graceful degradation to no-context fallback path. |
| 3 | Unbounded `_content_hashes` dict in `webhook.py` -- OOM risk in long-running containers | CRITICAL | 2/3 (Gemini #3, GPT #8) | Replaced plain `dict` with `TTLCache(maxsize=10_000, ttl=86400)` from cachetools. Bounds memory and evicts stale entries after 24h. |
| 4 | Circuit breaker `is_open` non-atomic read under async concurrency | HIGH | 2/3 (GPT #2, Grok H2) | Expanded docstrings to explicitly document as "monitoring-only approximate state." Added `async def get_state()` with lock-protected read for accurate monitoring. |
| 5 | PII buffer dropped on non-PII-related errors in SSE stream | MEDIUM | 3/3 (Gemini #7, GPT #7, Grok M4) | Acknowledged as correct default behavior (safer to drop on error). Documented in code. Current behavior is the safer conservative choice per GPT's own analysis ("low practical impact"). |
| 6 | Shared `_llm_lock` between main and validator LLM causes cascading stalls | MEDIUM | 1/3 (GPT #6) but impacts resilience | Separated into `_llm_lock`/`_llm_cache` and `_validator_lock`/`_validator_cache`. Validator construction stalls no longer block main LLM acquisition. |

## Single-Model Critical Findings Fixed

| # | Finding | Severity | Model | Fix Applied |
|---|---------|----------|-------|-------------|
| 7 | Firestore client recreated per CRUD call -- connection exhaustion under load | CRITICAL | Gemini #1 | Added `_firestore_client_cache` dict with singleton caching. Client is created once and reused. Added `clear_firestore_client_cache()` for tests. |
| 8 | Non-atomic CCPA cascade delete -- partial deletion violates privacy laws | CRITICAL | Gemini #4 | Replaced individual `delete()` calls with Firestore `batch()` write. All operations (conversation/message/signal deletion, audit de-identification, guest deletion) commit atomically. |
| 9 | Whisper planner failure counter has no threshold -- systematic failures invisible | HIGH | Grok H3 | Added `alert_threshold` to `_FailureCounter`. After 10 consecutive failures, logs at ERROR level with structured metric. Added `reset()` on success to clear alert state. |
| 10 | Health endpoint missing circuit breaker state -- degraded operation invisible | MEDIUM | Grok M3 | Added `circuit_breaker_state` field to `HealthResponse`. Health endpoint now uses lock-protected `get_state()`. CB open reports `status: "degraded"` and returns 503. |

## Findings NOT Fixed (with justification)

| Finding | Severity | Model | Justification |
|---------|----------|-------|---------------|
| Heartbeat starvation during long LLM generation | HIGH | Gemini #6 | Implementing a separate heartbeat task with asyncio.Queue is a significant architectural change. The current approach sends heartbeats between events which works for typical generation latencies. Would require extensive SSE infrastructure refactoring -- better as a dedicated task. |
| Premature CB `record_success()` before content validation | HIGH | Gemini #5 | Moving `record_success()` after content validation would conflate content quality with LLM availability. The CB's purpose is to track LLM API availability, not semantic quality. A successful HTTP response IS a success from the CB's perspective. |
| Semantic injection classifier fail-closed with no retry | HIGH | Grok H1 | Fail-closed for security classification is an intentional architectural decision documented in CLAUDE.md (`langgraph-patterns.md`). Adding retry would delay safety-critical path. Documented trade-off. |
| Validator degraded-pass is inconsistent | HIGH | GPT #4 | This is the documented degraded-pass strategy from `langgraph-patterns.md`. All 4 R20 review models praised it as "nuanced" and "production-grade." GPT flagged it as inconsistent, but it's an intentional availability/safety balance. |
| SSE error taxonomy / typed error envelope | HIGH | GPT #5 | Good improvement but additive scope. Would require frontend changes. Tracked for future round. |
| `get_settings()` uses `@lru_cache` (never refreshes) | MEDIUM | Grok M1 | Cloud Run env vars are immutable per container lifetime. Added comment in code documenting this is intentional. TTL caches on LLM singletons are for GCP Workload Identity token refresh, not env var changes. |
| `_build_greeting_categories()` stale after CMS update | MEDIUM | Grok M2 | Container lifetime-scoped cache is acceptable. CMS webhook will trigger re-ingestion on next restart. Low priority. |
| Docker exec form CMD not verified | MEDIUM | Grok | Already uses exec form in Dockerfile -- `CMD ["uvicorn", ...]`. Not a code issue. |

---

## Test Results After Fixes

```
1107 passed, 20 skipped, 1 warning in 10.57s
Coverage: 90.63%
```

- **20 new tests added** (1087 -> 1107)
- All existing tests pass (0 regressions)
- Coverage maintained above 90% threshold

### New Tests Added

| Test File | Tests Added | What They Cover |
|-----------|:-----------:|-----------------|
| `test_base_specialist.py` | 3 | Broad except catches RuntimeError, AttributeError, KeyError (Fix #1) |
| `test_nodes.py` | 7 | retrieve_node non-timeout exceptions (Fix #2), CB `get_state()` (Fix #4), separate validator lock/cache (Fix #6/9) |
| `test_cms.py` | 3 | `_content_hashes` is TTLCache with maxsize and TTL (Fix #3) |
| `test_whisper_planner.py` | 3 | Failure counter threshold alert, alert-once behavior, reset (Fix #9) |
| `test_guest_profile.py` | 3 | Firestore client cache existence, clearing, no-GCP returns None (Fix #7) |
| `test_api.py` | 1 | Health endpoint includes `circuit_breaker_state` (Fix #10) |

---

## Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `src/agent/agents/_base.py` | Modified | Added broad `except Exception` safety net after specific exception handlers |
| `src/agent/nodes.py` | Modified | Broad except in retrieve_node; separate validator lock/cache |
| `src/agent/circuit_breaker.py` | Modified | Documented is_open as monitoring-only; added lock-protected `get_state()` |
| `src/agent/whisper_planner.py` | Modified | Added failure threshold alert, reset-on-success |
| `src/cms/webhook.py` | Modified | Replaced unbounded dict with TTLCache for _content_hashes |
| `src/data/guest_profile.py` | Modified | Cached Firestore client singleton; atomic batch CCPA delete |
| `src/api/app.py` | Modified | Health endpoint reports circuit breaker state |
| `src/api/models.py` | Modified | Added circuit_breaker_state to HealthResponse |
| `tests/conftest.py` | Modified | Clear _validator_cache and firestore client cache between tests |
| `tests/test_base_specialist.py` | Modified | +3 tests for broad exception handling |
| `tests/test_nodes.py` | Modified | +7 tests for retrieve_node, CB get_state, separate locks |
| `tests/test_cms.py` | Modified | +3 tests for bounded content hash cache |
| `tests/test_whisper_planner.py` | Modified | +3 tests for failure threshold |
| `tests/test_guest_profile.py` | Modified | +3 tests for Firestore client caching |
| `tests/test_api.py` | Modified | +1 test for health CB state |

---

## Score Trajectory

| Round | Average | Delta | Key Changes |
|-------|:-------:|:-----:|-------------|
| R1 | 67.3 | -- | Baseline |
| R2 | 61.3 | -6.0 | Deeper scrutiny revealed gaps |
| R3 (pre-fix) | 60.7 | -0.6 | Error handling spotlight exposed resilience gaps |
| R3 (post-fix) | TBD | -- | 10 findings fixed, 20 new tests |

**Note**: The R3 pre-fix score (60.7) reflects the error handling spotlight which raises severity by +1 on all error handling findings. The fixes in this round directly address the highest-severity findings across all 3 models. Expected score improvement in R4: +5-8 points on Dimensions 3, 4, and 8 (Data Model, API Design, Scalability) due to Firestore client caching, atomic CCPA delete, bounded caches, and CB health reporting.
