# GPT-5.2 Production Review — Round 4

**Date**: 2026-02-20
**Commit**: c140453
**Reviewer**: GPT-5.2 (Azure AI Foundry)
**Spotlight**: TESTING GAPS (+1 severity on testing findings)
**Previous Scores**: R1=67.3 | R2=61.3 | R3=60.7

---

## Scores (0-10)

| # | Dimension | Score | Justification |
|---|-----------|:-----:|---------------|
| 1 | Graph/Agent Architecture | 7 | Clear 11-node graph with retries/validation, but several safety invariants (parity, dispatch feature flags, failure modes) aren't exercised end-to-end under realistic conditions. |
| 2 | RAG Pipeline | 7 | Solid ingestion/idempotency and RRF, but retrieval correctness and prod backend behavior aren't validated with meaningful integration tests (mostly unit/mocked). |
| 3 | Data Model / State Design | 7 | Typed state is structured, but serialization/parity guarantees rely on debug-only checks and lack regression tests for schema drift. |
| 4 | API Design | 8 | Pure ASGI middleware and SSE are good, but critical streaming/PII/error interactions aren't covered by tests, increasing reliability risk. |
| 5 | Testing Strategy | 4 | High coverage and lots of tests, but too many "mock-shaped" tests, thin E2E coverage, skipped tests, and major async/concurrency/error paths untested. |
| 6 | Docker & DevOps | 6 | Likely workable, but test suite doesn't validate runtime behaviors that bite in deployment (timeouts, streaming proxies, env-driven config differences). |
| 7 | Prompts & Guardrails | 7 | Good breadth of patterns/languages, but tests don't convincingly demonstrate robustness against streaming tokenization, partial matches, and bypass attempts in realistic flows. |
| 8 | Scalability & Production | 6 | Circuit breaker/rate limiting/TTL caches exist, but concurrency and contention behavior is basically untested. |
| 9 | Documentation & Code Quality | 7 | Organized, but skipped tests and debug-only asserts signal "it passes locally" more than "it survives prod". |
| 10 | Domain Intelligence | 6 | Domain logic may be fine, but the test suite doesn't prove compliance behavior under edge-case user inputs and channel constraints (SMS/SSE) across the full pipeline. |
| **Total** | | **65** | |

---

## Findings

### Finding 1 (CRITICAL): SSE PII buffer can leak or be dropped on error paths — untested
- **Location**: `src/agent/graph.py:chat_stream()` (PII buffering / `errored` guard, lines 493-635) and `src/api/app.py` SSE event generator
- **Problem**: The safety-critical branch that flushes/withholds `_pii_buffer` depending on `errored` is not covered by tests, especially when exceptions occur mid-stream or client disconnects. The guard at line 633 (`if _pii_buffer and not errored`) silently drops buffered PII-containing content on error, but no test verifies this behavior. Conversely, no test verifies that partial PII patterns spanning multiple chunks are correctly detected and redacted before emission.
- **Impact**: Production can either (a) leak partially-buffered PII during a crash path where `errored` is not set correctly, or (b) silently drop buffered content and produce malformed SSE streams that clients mis-parse/retry. In a casino environment with phone numbers, SSNs, and credit card numbers, PII leakage is a compliance-level incident.
- **Fix**: Add an integration-style async test that streams tokens containing split-PII patterns (e.g., phone number `555-` in one chunk, `1234` in the next), then forces an exception mid-stream and asserts: (1) no PII is emitted before the error, (2) the buffer is dropped on error (not flushed), (3) the `error` and `done` events are correctly emitted. Use `httpx.AsyncClient` against the ASGI app, not unit-level generators.

### Finding 2 (HIGH): "Integration" suite is mostly a mock orchestra, not a pipeline test
- **Location**: `tests/test_integration.py` (notably `test_full_graph_with_real_retrieval_mocked_llm`, lines 93-169)
- **Problem**: The "full graph" test requires mocking ~7 LLM instances (router, host, dining, entertainment, comp, whisper, validator); it validates your mocks and wiring more than real behavior. The test proves that when 7 mocks return specific values, the graph produces a non-empty response. It does NOT test: SSE framing, validator retry behavior, specialist dispatch under mixed contexts, PII redaction timing, or error recovery paths.
- **Impact**: Refactors can pass tests while breaking the actual runtime contract. The 8-test integration file gives false confidence — the real integration surface (API → graph → RAG → LLM → SSE → PII → client) has zero coverage.
- **Fix**: Add at least one true end-to-end ASGI test that runs `POST /chat` using the TestClient and asserts observable outputs (SSE events, response content, headers), not internal mock calls. Inject deterministic fake LLMs once at the boundary (via DI or a single patch), not patched in 7 places. Assert the SSE event sequence: metadata → graph_node(start) → token/replace → sources → done.

### Finding 3 (HIGH): Concurrency behavior for circuit breaker is untested — race-prone
- **Location**: `src/agent/circuit_breaker.py` (all methods) and `tests/test_nodes.py` (CB tests)
- **Problem**: The circuit breaker is designed for async concurrency (rolling window, half-open probing with `asyncio.Lock`), but tests don't simulate concurrent calls, interleaved failures, or the "probe" race (multiple tasks entering half-open state simultaneously). All CB tests appear to be sequential: call method, assert state, call next method, assert state.
- **Impact**: Under load, you can get thundering-herd probes (multiple coroutines all entering half-open and starting probes before the lock serializes them), incorrect open/close transitions, or window corruption — exactly when you rely on the CB to protect upstream LLM/RAG services from cascading failures.
- **Fix**: Add `pytest.mark.asyncio` tests spawning 50-200 concurrent tasks that call `allow_request()/record_success()/record_failure()` with controlled timing, asserting: (1) exactly one probe allowed in half-open state, (2) stable state transitions under concurrent access, (3) no exceptions or data races. Use `asyncio.gather()` with randomized delays to simulate realistic contention.

### Finding 4 (MEDIUM): Rate limiter LRU eviction + concurrent access not tested — DoS footgun
- **Location**: `src/api/middleware.py:RateLimitMiddleware` (lines 284-391) and `tests/test_middleware.py`
- **Problem**: Tests don't cover the `max_clients` eviction path (line 338: `self._requests.popitem(last=False)`) or concurrent requests hitting the same/new client IPs. The OrderedDict LRU eviction correctness is easy to get wrong in async contexts, and the `_lock` protecting `_requests` mutations is critical but untested under contention.
- **Impact**: In production, you can evict hot (legitimate) clients while keeping cold (attacker) IPs, or fail open/closed unpredictably under burst traffic — causing either outages for real users or ineffective throttling for attackers.
- **Fix**: Add tests that (1) exceed `max_clients` and assert deterministic LRU eviction order, (2) concurrently hit N unique clients and assert no `KeyError`/order corruption, (3) verify that the lock actually prevents interleaving bugs (e.g., double-add of the same IP).

### Finding 5 (MEDIUM): API key middleware TTL refresh path is untested — silent auth bypass
- **Location**: `src/api/middleware.py:ApiKeyMiddleware._get_api_key()` (lines 238-244)
- **Problem**: The "refresh every 60 seconds" cache invalidation path isn't covered by tests. Tests likely only validate the happy path with a warm cache (key matches) or cold start (key missing). The TTL-based refresh path that re-reads from `get_settings()` after 60 seconds is not exercised.
- **Impact**: Key rotation breaks — old keys remain valid too long (security exposure window) or new keys are rejected (self-inflicted downtime). In a casino environment, auth bypass is a compliance issue.
- **Fix**: Add a time-controlled test (freeze `time.monotonic()` or use an injectable clock) that exercises: initial load → cached allow → time advance past `_KEY_TTL` → refresh occurs → updated key is enforced. Make the clock dependency explicit for testability.

### Finding 6 (MEDIUM): 20 skipped tests are not treated as failures — hiding test rot
- **Location**: `tests/` (20 skipped tests across multiple files), CI configuration
- **Problem**: Skipped tests aren't audited; there's no enforced policy that prevents "temporary" skips from becoming permanent blind spots. The conftest.py disables `SEMANTIC_INJECTION_ENABLED` globally, and `test_live_llm.py` (1 test) and `test_retrieval_eval.py` (1 test) are essentially placeholder files. These represent testing infrastructure that was planned but never materialized.
- **Impact**: Dead tests accumulate; critical regressions ship because the only tests that would catch them are skipped indefinitely. The 20 skips represent ~2% of the test suite — but if they cover safety-critical paths (semantic injection, live LLM behavior), the gap is disproportionate.
- **Fix**: In CI, fail the build if skip count increases beyond a baseline. Add a `pytest_sessionfinish` hook or CI step parsing `pytest -ra` output; require an explicit allowlist of known skips with issue links. Audit each of the 20 skips and either fix the underlying issue, mark as `xfail` with a reason, or delete the test.

### Finding 7 (LOW): Debug-only parity assert isn't tested and vanishes in optimized builds
- **Location**: `src/agent/graph.py` (lines 387-394) and `tests/test_state_parity.py` (6 tests)
- **Problem**: The parity check between `PropertyQAState.__annotations__` and `_initial_state()` only runs under `__debug__`. Production builds with `python -O` strip this check entirely. The test_state_parity.py file has 6 tests but none verify the check's failure mode — i.e., none intentionally introduce drift and assert the assertion fires.
- **Impact**: Schema drift between `PropertyQAState` and `_initial_state()` silently corrupts state between nodes, producing non-obvious failures (wrong routing, missing fields, stale state from previous turns leaking) that tests won't catch.
- **Fix**: Turn the parity check into a callable function that runs at startup (always, not only in debug mode) and add a negative test that intentionally introduces drift (mock `PropertyQAState.__annotations__` to include an extra field) and asserts the expected `AssertionError`. Keep `__debug__` guard only for verbose diagnostics, not correctness.

### Finding 8 (LOW): Retrieval evaluation coverage is essentially nonexistent
- **Location**: `tests/test_retrieval_eval.py` (1 test), `tests/test_eval*.py`
- **Problem**: There is no repeatable retrieval quality regression suite. One test is not a harness. The evaluation framework exists (test_evaluation_framework.py has 17 tests, test_eval.py has 14, test_eval_deterministic.py has 18) but these test the evaluation infrastructure itself, not the retrieval quality of the actual pipeline.
- **Impact**: "90% coverage" won't stop silent relevance collapse after an embedding/model/config change. You'll ship a chatbot that confidently answers wrong because no test asserts that "steakhouse" queries return steakhouse documents, not entertainment documents.
- **Fix**: Add a small, deterministic golden dataset (10-30 queries) with expected document IDs or categories per `property_id`. Assert that top-k results contain expected items. Run this against both dev (Chroma with FakeEmbeddings) and a mocked Firestore interface. Make it part of CI to prevent retrieval regressions.

---

## Testing Anti-Patterns Observed

| Anti-Pattern | Files | Severity |
|---|---|---|
| **Mock Orchestra**: Full graph test mocks 7 LLMs to prove wiring, not behavior | `test_integration.py` | HIGH |
| **Untested Concurrency**: CB and rate limiter designed for async but tested sequentially | `test_nodes.py`, `test_middleware.py` | HIGH |
| **PII Safety Without Coverage**: Streaming PII buffer error paths are compliance-critical but untested | `graph.py chat_stream()` | CRITICAL |
| **Placeholder Test Files**: `test_live_llm.py` (1 test), `test_retrieval_eval.py` (1 test) | Multiple | MEDIUM |
| **Skip Rot**: 20 skipped tests with no audit trail or CI enforcement | Global | MEDIUM |
| **Debug-Only Safety**: Parity check stripped in production builds | `graph.py` | LOW |

---

## Summary

The test suite achieves impressive **volume** (1107 tests, 90.63% coverage) but lacks **depth** in the areas that matter most for production reliability:

1. **SSE streaming + PII buffer interaction** — the most safety-critical code path has zero integration test coverage.
2. **E2E pipeline tests** — the integration suite tests mock orchestration, not observable behavior.
3. **Concurrency** — circuit breaker and rate limiter are designed for concurrent async access but tested sequentially.
4. **Auth lifecycle** — API key rotation/TTL refresh is untested.
5. **Retrieval quality** — no regression suite for the core RAG pipeline's output quality.

The gap between "coverage" and "confidence" is the central finding of this round. 90% line coverage with sequential-only tests for concurrent code creates false safety. The fixes are additive (new tests, not code changes) and should be prioritized by PII/compliance impact.
