# Round 3 Production Review — Grok 4

**Reviewer:** Grok 4 (hostile production code reviewer)
**Commit:** 014335c
**Spotlight:** ERROR HANDLING & RESILIENCE (+1 severity boost)
**Previous Scores:** R1 avg 67.3 | R2 avg 61.3

---

## Dimension Scores

| # | Dimension | Score | Notes |
|---|-----------|:-----:|-------|
| 1 | Graph/Agent Architecture | 7 | Solid 11-node topology with specialist dispatch. Business-priority tie-breaking, conditional edges, parity assertions. Specialist dispatch via registry is clean. |
| 2 | RAG Pipeline | 6 | Per-item chunking, SHA-256 idempotent IDs, version-stamp purging. No multi-strategy retrieval (RRF mentioned in rules but not implemented). Firestore retriever interface exists but untested path. |
| 3 | Data Model / State Design | 7 | TypedDict with Annotated reducers, _keep_max for responsible_gaming_count, parity assertion. RetrievedChunk schema explicit. ExtractedFields forward-compatible. |
| 4 | API Design | 6 | Pure ASGI middleware (correct), SSE with PII buffer, heartbeats, graph node lifecycle events. Missing global exception catch in retrieve_node for non-timeout errors. |
| 5 | Testing Strategy | 4 | 1070+ tests exist per R2 summary, but no test evidence provided in this review round. Test count alone doesn't validate error handling paths. No chaos/failure injection tests visible. |
| 6 | Docker & DevOps | 5 | cloudbuild.yaml has --max-instances and --set-secrets. Exec form CMD not verified. No health check liveness/readiness probe distinction in deployment config. |
| 7 | Prompts & Guardrails | 7 | 5 deterministic guardrail categories across 4 languages (84 patterns), semantic classifier fail-closed, defense-in-depth ordering. Over-blocking risk on transient LLM failures is a concern. |
| 8 | Scalability & Production | 4 | All state in-memory (circuit breaker, rate limiter, idempotency, feature flags, config cache). Redis/Memorystore documented but not wired. MemorySaver warning for production but not blocked. |
| 9 | Documentation & Code Quality | 6 | Strong inline docstrings with design rationale. Structured error taxonomy. Module-level constants with frozenset. Some comments are verbose but useful. Missing error flow diagrams. |
| 10 | Domain Intelligence | 7 | TCPA compliance (STOP/HELP/START, quiet hours, consent hash chain), BSA/AML patterns, responsible gaming escalation at 3+ triggers, patron privacy, age verification. Multilingual patterns (EN/ES/PT/ZH). |
| **Total** | | **59** | |

---

## Findings

### C1: CRITICAL (Spotlight boost: HIGH -> CRITICAL)
**Title:** `execute_specialist()` missing broad exception catch — unhandled LLM SDK exceptions propagate to graph

**Description:** In `src/agent/agents/_base.py:135-173`, the error handling catches `(ValueError, TypeError)`, `asyncio.CancelledError` (re-raise), and `(httpx.HTTPError, asyncio.TimeoutError, ConnectionError)`. However, the Google GenAI SDK (`langchain_google_genai`) can raise other exception types across versions: `google.api_core.exceptions.GoogleAPIError`, `google.api_core.exceptions.ResourceExhausted` (429 rate limit), `google.api_core.exceptions.DeadlineExceeded`, `RuntimeError`, `AttributeError` (on malformed responses). Any of these propagate unhandled through the graph, bypassing the circuit breaker (`record_failure()` never called), and crash the SSE stream.

The `router_node` in `src/agent/nodes.py:221-232` correctly has a broad `except Exception` catch as a safety net, with a documented rationale. The specialist execution path lacks this same safety net.

**Impact:** A single 429 rate limit or SDK version change could crash all concurrent requests. Circuit breaker never trips because `record_failure()` is never called for unhandled exception types.

**Recommended Fix:** Add a final `except Exception` catch after the specific handlers (before `CancelledError` re-raise) that calls `cb.record_failure()` and returns a fallback with `skip_validation=True`. Document the rationale (same pattern as `router_node`).

```python
except asyncio.CancelledError:
    raise
except (httpx.HTTPError, asyncio.TimeoutError, ConnectionError):
    await cb.record_failure()
    ...
except Exception:
    # Safety net: google-genai raises various exception types across versions.
    # KeyboardInterrupt/SystemExit propagate (not subclasses of Exception).
    await cb.record_failure()
    logger.exception("%s agent LLM call failed (unexpected)", agent_name.capitalize())
    return {
        "messages": [AIMessage(content=fallback_msg)],
        "skip_validation": True,
    }
```

---

### C2: CRITICAL (Spotlight boost: HIGH -> CRITICAL)
**Title:** `retrieve_node` catches only `TimeoutError` — non-timeout retrieval errors propagate unhandled

**Description:** In `src/agent/nodes.py:260-273`, `retrieve_node` wraps retrieval in `asyncio.wait_for(timeout=10)` and catches `TimeoutError`, but does NOT catch other exceptions. `search_knowledge_base()` and `search_hours()` call into ChromaDB (or Firestore in production), which can raise:
- `chromadb.errors.ChromaError` (corrupted collection)
- `sqlite3.OperationalError` (locked/corrupted SQLite)
- `google.api_core.exceptions.*` (Firestore errors)
- `ValueError` (embedding dimension mismatch)

An unhandled exception from retrieve propagates through the graph to `chat_stream()`'s broad `except Exception` at `graph.py:624`, which yields an error SSE event but provides no fallback response. The graph execution is aborted mid-flight — no `respond` node runs, no sources are emitted, and the user sees only "An error occurred."

This is worse than a timeout (which gracefully returns empty results and lets the pipeline proceed with the no-context fallback path).

**Impact:** Any non-timeout retrieval error crashes the entire graph execution. No fallback, no graceful degradation.

**Recommended Fix:** Broaden the except clause to catch `Exception` alongside `TimeoutError`:

```python
try:
    ...
except TimeoutError:
    logger.warning("Retrieval timed out...")
    results = []
except Exception:
    logger.exception("Retrieval failed for query: %s", query[:80])
    results = []
```

---

### H1: HIGH (Spotlight boost: MEDIUM -> HIGH)
**Title:** Semantic injection classifier fail-closed over-blocks during transient LLM outages

**Description:** In `src/agent/guardrails.py:397-426`, `classify_injection_semantic()` returns a synthetic `InjectionClassification(is_injection=True, confidence=1.0)` on ANY error. This is correct for safety, but during transient LLM outages (network blips, 429 rate limits), ALL legitimate messages that pass deterministic guardrails are blocked.

In `src/agent/compliance_gate.py:126-141`, the semantic classifier runs with a confidence threshold (`SEMANTIC_INJECTION_THRESHOLD=0.8`), and the synthetic result has `confidence=1.0`, so it always exceeds the threshold.

There is no:
1. Retry on transient errors before fail-closed
2. Circuit breaker on the semantic classifier itself (uses the main LLM singleton, which shares the circuit breaker with generate, but the CB check is only in `execute_specialist`, not in `classify_injection_semantic`)
3. Degradation metric — the `logger.error` message says "Configure alerting on this log line" but there's no structured telemetry

**Impact:** A 2-minute LLM outage blocks ALL non-deterministic messages. Users see off_topic responses for legitimate property questions.

**Recommended Fix:**
1. Add a single retry with 2s backoff before fail-closed
2. Track consecutive classifier failures; after N failures, temporarily bypass semantic classification (degrade to deterministic-only)
3. Emit a structured metric for monitoring dashboards

---

### H2: HIGH (Spotlight boost: MEDIUM -> HIGH)
**Title:** Circuit breaker `is_open` property reads state without lock — race condition in async context

**Description:** In `src/agent/circuit_breaker.py:122-133`, the `is_open` property reads `self.state` (which reads `self._state` and calls `self._cooldown_expired()`) without acquiring `self._lock`. The docstring acknowledges this: "Non-atomic read (no lock). Safe in CPython due to GIL for simple attribute reads."

However, in async code, coroutines can yield at any `await` point. While the GIL protects individual attribute reads, the sequence `read _state` → `call _cooldown_expired()` → `compare` is NOT atomic. Between reading `_state == "open"` and checking `_cooldown_expired()`, another coroutine could call `record_success()` which clears `_failure_timestamps` and sets `_state = "closed"`, leading to `state` returning `"half_open"` when the breaker is actually closed.

The code correctly notes that `allow_request()` is the authoritative check, and `execute_specialist` does use `allow_request()`. But `is_open` is part of the public API (`__all__` export, used in monitoring/health checks), and incorrect reads from monitoring could trigger false alerts.

**Impact:** Monitoring and health checks may report incorrect circuit breaker state. Not a data corruption issue, but misleading operational signals.

**Recommended Fix:** Either remove `is_open`/`is_half_open` from the public API (since `allow_request()` is authoritative), or document them as "approximate state for monitoring only, not for control flow." Add `async def get_state()` with lock for accurate reads.

---

### H3: HIGH
**Title:** Whisper planner failure counter has no threshold — degrades silently without operational signal

**Description:** In `src/agent/whisper_planner.py:82-106`, `_FailureCounter` increments on every failure and logs a warning, but there is no threshold that triggers any action. The counter grows without bound and is never reset.

The whisper planner is fail-silent by design (any failure returns `{"whisper_plan": None}`), which is correct for availability. But without a threshold, there is no way to detect systematic failures. If the whisper LLM is consistently failing (e.g., wrong model name, expired credentials), the system silently degrades forever — every response lacks planning guidance, but no alert fires.

The `_failure_counter.value` property exists for monitoring but is never checked by any code path.

**Impact:** Systematic whisper planner failures go undetected. Guest responses are consistently unguided without any operational awareness.

**Recommended Fix:** Add a threshold (e.g., 10 failures in 5 minutes) that logs at ERROR level with a structured metric. Consider integrating with the circuit breaker pattern (the whisper LLM has its own singleton but no CB).

---

### M1: MEDIUM
**Title:** `get_settings()` uses `@lru_cache` — never refreshes for config changes or credential rotation

**Description:** In `src/config.py:162-165`, `get_settings()` is cached with `@lru_cache(maxsize=1)`. Once loaded, settings never refresh for the lifetime of the process. This conflicts with:

1. The LLM singletons use `TTLCache(ttl=3600)` specifically for credential rotation, but the settings they read from are cached forever
2. The API key middleware has its own 60s TTL refresh (`_get_api_key()` in `middleware.py:238-244`), but it reads from `get_settings()` which is cached — so the API key CAN change (middleware re-reads the `SecretStr` object), but only because the middleware re-reads the `get_secret_value()` on the same Settings instance. If the env var changes, the Settings object won't pick it up.
3. Feature flags have their own 5-min TTL cache, which works because they read from Firestore, not env vars.

In a Cloud Run environment, env vars don't change during container lifetime (they're set at deploy time), so this is acceptable. But it breaks the mental model suggested by the TTL-cached LLM singletons.

**Impact:** Low for Cloud Run (env vars are immutable per container). Misleading architecture — TTL caches suggest rotation support, but the underlying settings are static.

**Recommended Fix:** Document that `get_settings()` is intentionally static per container lifetime. Add a comment to the LLM TTL cache explaining that credential rotation refers to token refresh (GCP Workload Identity), not env var changes.

---

### M2: MEDIUM
**Title:** `_build_greeting_categories()` uses `@lru_cache` — never refreshes after CMS content updates

**Description:** In `src/agent/nodes.py:437-465`, `_build_greeting_categories()` reads the property JSON file and caches the result forever via `@lru_cache(maxsize=1)`. If the CMS webhook updates the property data (via `src/cms/webhook.py`), the greeting categories remain stale until container restart.

The CMS webhook handler (`handle_cms_webhook`) validates and hashes content changes, but it does not clear this cache. The greeting node will continue showing old categories even after a CMS update takes effect.

**Impact:** Stale greeting categories after CMS content updates. Users may not see new categories (e.g., a new "Shopping" category) until the container restarts.

**Recommended Fix:** Clear `_build_greeting_categories.cache_clear()` from the CMS webhook handler when the category structure changes, or convert to TTLCache.

---

### M3: MEDIUM (Spotlight boost: LOW -> MEDIUM)
**Title:** No health check for circuit breaker state — degraded operation invisible

**Description:** The `GET /health` endpoint in `src/api/app.py:208-244` checks `agent_ready`, `property_loaded`, `rag_ready`, and `observability_enabled`, but does NOT report circuit breaker state. When the CB is open (blocking all LLM calls), the health check still reports "healthy" because the agent graph is initialized and the RAG store is accessible.

A fully open circuit breaker means every guest query returns the fallback message ("I'm experiencing temporary technical difficulties"). The system is functionally degraded, but Cloud Run continues routing traffic to this container.

**Impact:** Traffic continues flowing to a functionally degraded container. No operational visibility into CB state without manual log inspection.

**Recommended Fix:** Add `circuit_breaker_state` to `HealthResponse`. When CB is open, report `status: "degraded"` and return 503 to let Cloud Run drain traffic.

---

### M4: MEDIUM
**Title:** PII buffer in SSE streaming has edge case — digits at stream end never flushed on error

**Description:** In `src/agent/graph.py:632-635`, the PII buffer flush after the try/except only runs when `not errored`:

```python
if _pii_buffer and not errored:
    async for tok_event in _flush_pii_buffer():
        yield tok_event
```

If an error occurs mid-stream while the PII buffer contains partial PII (e.g., first 5 digits of an SSN), the buffer is silently dropped — which is correct (don't emit partial PII on error). But if the error is non-PII-related (e.g., a network glitch in the graph after the LLM finished generating), legitimate buffered text is also lost.

**Impact:** Minor text loss at the end of errored streams. The error event still fires, so the user knows something went wrong. Low practical impact.

**Recommended Fix:** Consider flushing the buffer on error if the error source is not the generate node (i.e., if the LLM finished but a downstream node like validate/respond errored). This is a nuanced fix — the current behavior (drop on error) is the safer default.

---

### L1: LOW
**Title:** `ApiKeyMiddleware._get_api_key()` not thread-safe for concurrent requests

**Description:** In `src/api/middleware.py:238-244`, `_get_api_key()` reads and writes `_cached_key` and `_cached_at` without any lock. Under concurrent async requests, two coroutines could simultaneously find the TTL expired and both call `get_settings().API_KEY.get_secret_value()`. This is a benign race (both get the same value), but it's technically a TOCTOU issue.

**Impact:** None in practice — both coroutines get the same secret value. No data corruption.

**Recommended Fix:** Document as intentional (benign race, no lock needed for read-only values).

---

### L2: LOW
**Title:** Feature flag cache (`_flag_cache`) in `feature_flags.py` is unbounded dict

**Description:** In `src/casino/feature_flags.py:85-86`, `_flag_cache` is a plain `dict[str, tuple[dict[str, bool], float]]`. For multi-tenant deployments with many casino_ids, this dict grows without bound. There's no eviction policy.

**Impact:** Negligible for the current single-tenant deployment (only "mohegan_sun"). Potential memory issue at scale.

**Recommended Fix:** Use `TTLCache(maxsize=100, ttl=300)` from cachetools (already a dependency) to bound the cache.

---

### I1: INFO
**Title:** LangFuse production sampling rate (10%) may miss resilience-related traces

**Description:** In `src/observability/langfuse_client.py:22-23`, `_PRODUCTION_SAMPLE_RATE = 0.10`. This means 90% of requests are untraced, including error paths. The comment says "Always trace error paths (caller's responsibility)" but neither `chat()` nor `chat_stream()` in `graph.py` force-sample on errors.

**Impact:** Error traces are randomly sampled at 10% in production. Debugging intermittent issues requires luck.

**Recommended Fix:** Add `should_trace_on_error=True` parameter to `get_langfuse_handler` and force-sample when the graph produces an error event.

---

## Score Summary

| Dimension | Score |
|-----------|:-----:|
| 1. Graph/Agent Architecture | 7 |
| 2. RAG Pipeline | 6 |
| 3. Data Model / State Design | 7 |
| 4. API Design | 6 |
| 5. Testing Strategy | 4 |
| 6. Docker & DevOps | 5 |
| 7. Prompts & Guardrails | 7 |
| 8. Scalability & Production | 4 |
| 9. Documentation & Code Quality | 6 |
| 10. Domain Intelligence | 7 |
| **Total** | **59** |

## Finding Count by Severity

| Severity | Count |
|----------|:-----:|
| CRITICAL | 2 |
| HIGH | 3 |
| MEDIUM | 4 |
| LOW | 2 |
| INFO | 1 |
| **Total** | **12** |

## Verdict

The codebase has strong architectural foundations — the 11-node graph with defense-in-depth guardrails, specialist dispatch with DI, and multi-language compliance patterns show real production thinking. However, the error handling and resilience layer has systematic gaps:

1. **Missing broad exception catches** in the two highest-traffic code paths (specialist execution and retrieval) mean that uncommon but real exceptions bypass the circuit breaker and crash the graph.
2. **Fail-closed semantic classifier** with no retry/degradation path will block all legitimate traffic during LLM outages.
3. **No operational visibility** into circuit breaker state or whisper planner health means degraded operation goes undetected.

The system handles the happy path and the most common failure modes well (timeout, parsing errors). It does NOT handle the long tail of failure modes (SDK version changes, rate limits, transient network errors in non-timeout categories) — and that long tail is where production systems spend 80% of their incident response time.

**Score trend:** R1 67.3 -> R2 61.3 -> R3 59. The downward trend reflects the spotlight revealing deeper structural gaps rather than actual regression — the code is getting more complex with each fix round, but the resilience layer hasn't scaled to match.

---

*Review generated by Grok 4 via mcp__grok__grok_reason, 2026-02-20*
