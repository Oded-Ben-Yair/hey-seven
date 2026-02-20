# Round 4 Production Review — Summary

**Date**: 2026-02-20
**Commit**: 9655fd2 (base) -> [post-fix commit]
**Spotlight**: TESTING GAPS
**Fixer**: Claude Opus 4.6

---

## Score Table

| # | Dimension | Gemini | GPT | Grok | Average |
|---|-----------|:------:|:---:|:----:|:-------:|
| 1 | Graph/Agent Architecture | 9 | 7 | 5 | 7.0 |
| 2 | RAG Pipeline | 8 | 7 | 6 | 7.0 |
| 3 | Data Model / State Design | 9 | 7 | 4 | 6.7 |
| 4 | API Design | 7 | 8 | 7 | 7.3 |
| 5 | Testing Strategy | 5 | 4 | 3 | 4.0 |
| 6 | Docker & DevOps | 10 | 6 | 5 | 7.0 |
| 7 | Prompts & Guardrails | 9 | 7 | 6 | 7.3 |
| 8 | Scalability & Production | 9 | 6 | 4 | 6.3 |
| 9 | Documentation & Code Quality | 9 | 7 | 5 | 7.0 |
| 10 | Domain Intelligence | 9 | 6 | 6 | 7.0 |
| **Total** | | **84** | **65** | **51** | **66.7** |

**Previous**: R1=67.3, R2=61.3, R3=60.7
**Current**: R4=66.7 (upward trajectory restored)

---

## Consensus Findings Fixed (2/3+ models flagged)

### 1. CRITICAL: chat_stream PII buffer untested (Gemini F1, GPT F1)
**Fix**: Created `tests/test_chat_stream.py` with 12 tests exercising chat_stream() directly (not mocked). Tests cover:
- Phone number split across tokens -> redacted to [PHONE]
- Credit card number split across tokens -> redacted to [CARD]
- SSN split across tokens -> redacted to [SSN]
- Non-digit text flushes immediately (no unnecessary buffering)
- Buffer flush at 80-char threshold
- Buffer flush at sentence boundary
- PII buffer DROPPED on error (not flushed) — compliance safety
- Sources suppressed after error
- CancelledError re-raised (client disconnect)
- Error mid-stream yields error + done events
- metadata first / done last SSE event sequence
- Graph node lifecycle events emitted

### 2. HIGH: whisper_planner_enabled=False topology untested (Gemini F2)
**Fix**: Added `TestGraphWithWhisperDisabled` class (3 tests) to `test_graph_v2.py`:
- Graph compiles without error when flag is False
- retrieve connects directly to generate (whisper_planner skipped)
- No edges route to whisper_planner (unreachable node)

### 3. HIGH: Google Sheets client untested (Gemini F3)
**Fix**: Created `tests/test_sheets_client.py` with 16 tests covering:
- SheetsClient initialization (with/without credentials)
- Lazy service creation (missing google lib)
- read_category: valid tab, unknown category, empty sheet, API errors, short rows
- read_all_categories: iterates all CONTENT_CATEGORIES
- compute_content_hash: deterministic SHA-256, key-order independence

### 4. HIGH: HITL interrupt path untested (Gemini F4)
**Fix**: Added `TestHITLInterrupt` class (3 tests) to `test_graph_v2.py`:
- HITL disabled by default (no interrupt)
- HITL enabled compiles with interrupt_before=[generate]
- HITL interrupt pauses before generate (router ran, generate interrupted)

### 5. HIGH: Circuit breaker concurrency untested (GPT F3, Grok F2)
**Fix**: Added `TestCircuitBreakerConcurrency` class (7 tests) to `test_nodes.py`:
- 50 concurrent allow_request() all pass when closed
- 50 concurrent allow_request() all blocked when open
- Exactly one probe allowed in half-open state
- Probe success closes breaker
- Probe failure reopens breaker
- Concurrent record_failure() trips breaker correctly
- Interleaved success/failure doesn't corrupt state

### 6. MEDIUM: Specialist dispatch tie-breaking weak assertion (Gemini F6)
**Fix**: Strengthened assertion in `test_mixed_context_dispatches_to_host` from `assert mock_get.called` to `mock_get.assert_called_once_with("dining")`. Added dedicated `TestSpecialistDispatchTieBreaking` test class.

### 7. MEDIUM: Rate limiter LRU concurrency untested (GPT F4, Grok F4)
**Fix**: Added `TestRateLimitConcurrency` class (2 tests) to `test_middleware.py`:
- 50 concurrent requests from unique IPs: no KeyError or 500 errors
- max_clients eviction deterministic order

### 8. MEDIUM: conftest.py inconsistent error handling (Gemini F9, Grok F3)
**Fix**: Standardized all `except ImportError:` blocks to `except (ImportError, AttributeError):` in conftest.py singleton cleanup fixture. Consistent handling prevents misleading test failures if cache attributes are renamed.

### 9. MEDIUM: Whisper planner failure counter boundary untested (Grok F7)
**Fix**: Added 2 boundary tests to `TestFailureCounterThreshold` in `test_whisper_planner.py`:
- 10 failures (alert) -> 1 success (reset) -> 9 failures: no second alert
- After reset, reaching threshold again fires new alert

### 10. MEDIUM: Semantic injection fail-closed missing TimeoutError test (Grok F5)
**Fix**: Added 3 tests to `TestSemanticInjectionClassifier` in `test_guardrails.py`:
- TimeoutError from llm_fn fails closed
- RuntimeError from llm_fn fails closed
- TimeoutError during ainvoke fails closed with reason

### 11. MEDIUM: API key TTL refresh after expiry untested (GPT F5)
**Fix**: Added `TestApiKeyTTLRefreshExpiry` class (1 test) to `test_middleware.py`:
- Key refreshed after TTL with time mock (simulates key rotation)

---

## Findings NOT Fixed (with justification)

### GPT F2: Integration suite is "mock orchestra"
**Justification**: The existing E2E tests (`TestEndToEndGraphIntegration`) already exercise real graph execution through HTTP with real middleware stack. The "mock orchestra" criticism applies to `test_integration.py` which tests with mocked LLMs — this is necessary for deterministic CI. The new chat_stream tests add the missing depth for the most critical code path. A true zero-mock E2E test requires live LLM API keys, which is appropriate for `test_live_llm.py` (already marked as CI-skip).

### GPT F6: 20 skipped tests / skip rot
**Justification**: The 20 skips are intentional: 14 are live LLM tests (require API keys), 2 are retrieval eval (require real embeddings), and the remainder are semantic injection tests (require GOOGLE_API_KEY). These are documented skip reasons, not abandoned tests. Adding skip-count enforcement in CI is a process change, not a code fix.

### GPT F7 / F8: Debug-only parity assert, retrieval eval
**Justification**: LOW severity. Parity assertion at module level catches drift at import time (which is always during testing). Retrieval quality eval suite requires real embeddings and is tracked as a Phase 4 task.

### Grok F6: tenant_id field in state
**Justification**: This is a design opinion, not a testing gap. The state already has implicit tenant isolation via `CASINO_ID` config + `property_id` metadata filtering in RAG retrieval. Adding an explicit `tenant_id` field to the TypedDict is a Phase 2 multi-tenant feature, not an R4 fix.

### Grok F4: Unbounded rate limit cache
**Justification**: The rate limiter already has `max_clients` (default 10,000) with LRU eviction via OrderedDict. Memory is bounded. The Grok finding incorrectly states "lacks a maxsize bound" — `RATE_LIMIT_MAX_CLIENTS` is that bound.

---

## Test Results

```
1157 passed, 20 skipped, 1 warning in 10.70s
Coverage: 91.27% (up from 90.63%)
```

**New tests added**: 50
- `test_chat_stream.py`: 12 tests (NEW FILE)
- `test_sheets_client.py`: 16 tests (NEW FILE)
- `test_graph_v2.py`: 10 tests (whisper disabled, HITL, tie-breaking)
- `test_nodes.py`: 7 tests (circuit breaker concurrency)
- `test_middleware.py`: 3 tests (rate limit concurrency, API key TTL)
- `test_whisper_planner.py`: 2 tests (failure counter boundaries)
- `test_guardrails.py`: 3 tests (semantic injection fail-closed)

Existing test assertions strengthened: 1 (mixed context dispatch)

---

## Files Modified

| File | Change Type | Description |
|------|------------|-------------|
| `tests/test_chat_stream.py` | Created | 12 tests for PII buffer, SSE errors, event sequence |
| `tests/test_sheets_client.py` | Created | 16 tests for Google Sheets CMS client |
| `tests/test_graph_v2.py` | Modified | 10 new tests: whisper disabled, HITL, tie-breaking |
| `tests/test_nodes.py` | Modified | 7 new tests: circuit breaker concurrency |
| `tests/test_middleware.py` | Modified | 3 new tests: rate limit concurrency, API key TTL |
| `tests/test_whisper_planner.py` | Modified | 2 new tests: failure counter boundaries |
| `tests/test_guardrails.py` | Modified | 3 new tests: semantic injection fail-closed |
| `tests/conftest.py` | Modified | Standardized error handling (ImportError -> ImportError, AttributeError) |
