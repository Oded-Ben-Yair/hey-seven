# R11 Summary

## Consensus Findings (2/3+ agreement)

- **InMemoryBackend memory leak from expired-but-unreaped entries** (3/3) -- expired entries accumulate indefinitely in `_store` because cleanup only runs per-key on access. Write-once-never-read keys (transient IP rate windows) leak memory. Models: DeepSeek F-007 (LOW), Gemini F-002 (CRITICAL), GPT F-001 (HIGH)
- **Rate limiter per-instance in distributed Cloud Run** (3/3) -- each container tracks limits independently, so effective limit = RATE_LIMIT_CHAT * N instances. Bot storms can bypass by hitting different instances. Models: DeepSeek F-006 (MEDIUM), Gemini F-001 (CRITICAL), GPT F-002 (HIGH)
- **Circuit breaker TTL/config propagation** (3/3) -- TTLCache(ttl=3600) on CB singleton means config changes take up to 1 hour to propagate. During incidents, operators cannot immediately tune CB thresholds. Models: DeepSeek F-010 (LOW), Gemini F-003 (MAJOR), GPT F-004 (MEDIUM)

## Fixes Applied (7/7)

1. **InMemoryBackend probabilistic sweep** -- Added `_maybe_sweep()` that runs with ~1% probability on every `set()`/`increment()` call, evicting all expired entries. Also fires unconditionally when store exceeds `_MAX_STORE_SIZE=50000`. Prevents unbounded memory growth from write-once-never-read keys. File: `src/state_backend.py`

2. **Redis URL logging redaction** -- Replaced `redis_url.split("@")[-1]` logging with structured host/port/db/ssl fields extracted from connection pool kwargs. Prevents credential leakage via non-standard URL formats or query params. File: `src/state_backend.py`

3. **Rate limiter stale-client sweep** -- Added periodic sweep (every 100 requests) inside `_is_allowed()` that removes clients whose deques are fully expired. Prevents slow memory growth from transient IPs. Also enhanced docstring documenting the distributed limitation as a pre-production TODO with 3 specific remediation paths. File: `src/api/middleware.py`

4. **Circuit breaker explicit cache-clear for incident response** -- Added `clear_circuit_breaker_cache()` function for immediate config reload during incidents, rather than waiting for 1-hour TTL. Documented as the operational path for CB tuning. File: `src/agent/circuit_breaker.py`

5. **Circuit breaker lock-protected `get_failure_count()`** -- Added `async def get_failure_count()` method that prunes stale entries under the lock before counting, providing accurate monitoring. The existing `failure_count` property remains read-only (no mutation) for approximate checks. File: `src/agent/circuit_breaker.py`

6. **CancelledError no longer counts as CB failure** -- Added `record_cancellation()` method that resets the half_open probe flag without recording a failure. Updated `_base.py` to call `record_cancellation()` instead of `record_failure()` on CancelledError. Client disconnects (normal SSE behavior) no longer inflate the failure count or artificially trip the breaker. File: `src/agent/circuit_breaker.py`, `src/agent/agents/_base.py`

7. **Tests for all new code** -- 11 new tests: 4 for InMemoryBackend sweep, 3 for record_cancellation, 3 for get_failure_count, 1 for clear_circuit_breaker_cache. Updated 3 existing CancelledError tests to verify new behavior. Files: `tests/test_state_backend.py`, `tests/test_nodes.py`, `tests/test_base_specialist.py`

## Deferred (acknowledged, not fixed this round)

- **Distributed rate limiting via Redis/Cloud Armor** -- Architectural change requiring Redis integration or GCP Cloud Armor configuration. Documented as pre-production TODO with 3 remediation paths. Too large for a fix round; requires infrastructure planning.
- **Streaming tokens before validation** (DeepSeek only, 1/3) -- Would require either buffering all tokens (latency cost) or client-side cooperation. Architectural trade-off that needs product decision.
- **Global circuit breaker shared across specialists** (DeepSeek only, 1/3) -- Per-agent CB keying would require architecture change in registry.py and all specialist agents. Risk of breaking existing tests without full impact analysis.
- **Error handling middleware order** (DeepSeek only, 1/3) -- Current order already has ErrorHandling wrapping inner middleware; DeepSeek's analysis of "last added = outermost" is the CORRECT current behavior (ErrorHandling is 5th added = outermost of the custom ASGI middleware).
- **Semantic filter fail-closed UX** (Gemini only, 1/3) -- Changing fail-closed to fail-open for the semantic injection classifier is a security trade-off that needs product/compliance review.
- **LLM client TTL churn** (Gemini only, 1/3) -- Increasing TTL from 1h to 4h is low-risk but needs validation that GCP ADC actually handles refresh internally.

## Test Results

- Before: 1430 passed, 20 skipped
- After: 1441 passed, 20 skipped (11 new tests)
- Coverage: 90.33% (above 90.0% threshold)

## Scores

- DeepSeek: 73/100
- Gemini: 86/100
- GPT-5.2: 79/100
- Average: 79.3/100
