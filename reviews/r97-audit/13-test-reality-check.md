# Test Coverage Reality Check — R97 Audit

## Summary

| Metric | Value |
|--------|-------|
| Total tests collected | 3555 |
| Total test files | 104 |
| Source modules (non-init) | 56 |
| Committed line coverage | 90.62% (5465/6031 lines) |
| Tests using mocks (files) | 55 of 104 (52.9%) |
| Tests using live LLM (files) | 3 of 104 (2.9%) |
| Tests with `@pytest.mark.live` | 2 files (test_profiling_live.py, test_guardrails.py) |
| Skip/xfail markers | 19 across 7 files |
| Coverage fail_under gate | 90% (pyproject.toml) |

## The Contradiction: Rule 8 vs Reality

**CLAUDE.md Rule 8**: "NO MOCK TESTING — all tests use live real LLM API calls. No mock LLMs, no fake responses."

**Reality**: 55 of 104 test files (52.9%) use `unittest.mock`, `MagicMock`, `patch`, or `AsyncMock`. Total mock-related occurrences: **1694** across those 55 files. The project is OVERWHELMINGLY mock-based.

Only **3 files** actually import live LLM clients:
1. `test_live_llm.py` — 1 test, skipped without GOOGLE_API_KEY
2. `test_profiling_live.py` — 2 tests, `@pytest.mark.live`, skipped without API key
3. `test_llm_judge.py` — imports ChatGoogleGenerativeAI but heavily mocked

**Verdict on Rule 8 compliance**: VIOLATED. Rule 8 is aspirational, not enforced. The vast majority of tests mock all LLM calls. This is not inherently bad (see analysis below), but the documentation is dishonest.

## Coverage Matrix

| # | Component | Test Files | Test Count | Live/Mock | Coverage Assessment |
|---|-----------|-----------|-----------|-----------|---------------------|
| 1 | Graph (graph.py, state.py, constants.py) | test_agent, test_graph_v2, test_full_graph_e2e, test_graph_topology, test_graph_properties | ~180 | 100% mocked | GOOD topology coverage. All LLM calls mocked. Graph wiring verified. |
| 2 | Router/Dispatch (nodes.py, dispatch.py, registry) | test_nodes, test_dispatch, test_integration | ~200 | 100% mocked | GOOD. Router structured output mocked. Dispatch routing logic tested. |
| 3 | Specialists (_base.py, agents) | test_agents, test_base_specialist, test_r21_agent_quality | ~170 | 100% mocked | GOOD structural coverage. All LLM calls mocked via _get_llm patches. |
| 4 | Guardrails/Compliance (guardrails.py, compliance_gate.py) | test_guardrails, test_guardrail_patterns, test_guardrail_fuzz, test_guardrail_redos, test_compliance_gate | ~530 | MOSTLY deterministic (no LLM). Semantic classifier mocked. | EXCELLENT. 277 tests for guardrails alone. Deterministic regex tests need no mocks. |
| 5 | RAG (pipeline.py, embeddings, retriever, reranking) | test_rag, test_rag_quality, test_retrieval_eval, test_retrieval_resilience, test_firestore_retriever | ~115 | FakeEmbeddings (hash-based). No live embeddings. | GOOD. FakeEmbeddings is appropriate for unit tests. No live embedding API tested. |
| 6 | Profiling/Extraction (profiling.py, extraction.py, whisper_planner.py) | test_profiling, test_extraction, test_profiling_live, test_whisper_planner, test_phase5_profiling, test_extraction_llm | ~200 | 95% mocked. 2 live tests in test_profiling_live.py. | ADEQUATE. Schema validation tested live (2 tests). Extraction logic heavily mocked. |
| 7 | Incentives/Crisis/Sentiment | test_incentives, test_crisis_detection, test_crisis_progression, test_sentiment, test_semantic_sarcasm, test_slang_normalization | ~280 | NO MOCKS. All deterministic. | EXCELLENT. Pure logic tests. No LLM dependency. |
| 8 | Prompts/Persona | test_prompts, test_persona, test_few_shot_examples, test_prompt_parameterization, test_tone_calibration | ~55 | Partially mocked (persona templates). | ADEQUATE. Tests template substitution, not LLM output quality. |
| 9 | API (app.py, middleware, models, PII) | test_api, test_middleware, test_auth_e2e, test_e2e_security_enabled, test_security_headers, test_streaming_pii, test_pii_redaction, test_sse_e2e, test_chat_stream | ~190 | FastAPI TestClient with mocked agent. Auth tested with real middleware. | GOOD. Auth enabled + disabled paths both tested. SSE streaming tested. |
| 10 | CMS/SMS | test_cms, test_sms, test_sheets_client | ~180 | Mocked external services (Telnyx, Sheets). | ADEQUATE. External API mocking is appropriate. |
| 11 | Observability | test_observability, test_eval, test_eval_deterministic, test_llm_judge, test_llm_judge_live, test_evaluation_framework | ~115 | Mocked Langfuse. Eval framework uses FakeEmbeddings. | ADEQUATE for init/config. llm_judge.py (1150 LOC) has limited unit coverage. |
| 12 | Config/Flags | test_config, test_casino_config, test_state_backend, test_state_parity, test_state_serialization | ~75 | Minimal mocking. | GOOD. State serialization tested. Casino config validated. |

## Mock vs Live Detail

### Heavily Mocked (RED FLAG for behavioral quality)

The following critical components have ZERO live LLM testing:

1. **Router node** (`test_nodes.py`): All 179 tests mock `_get_llm`. The router's actual classification accuracy against real Gemini Flash is never tested in CI.

2. **Specialist agents** (`test_agents.py`, `test_base_specialist.py`): 170+ tests mock `_get_llm`. The actual quality of specialist responses is never tested automatically.

3. **Full graph E2E** (`test_full_graph_e2e.py`, `test_e2e_pipeline.py`): Schema-dispatching mock LLM (_SmartMockLLM pattern). Tests wiring, not output quality.

4. **Whisper planner** (`test_whisper_planner.py`): All 25+ mock occurrences. Planner output quality untested.

### Appropriately Deterministic (No LLM needed)

- **Guardrails** (530 tests): Regex-based, deterministic. No mocks needed.
- **Crisis detection** (72 tests): Pattern matching, no LLM.
- **Sentiment** (66 tests): VADER-based, deterministic.
- **Slang normalization** (34 tests): Dictionary lookup, no LLM.
- **Incentives** (43 tests): Rule engine, no LLM.
- **State serialization** (3 tests): JSON roundtrip, no LLM.
- **PII redaction** (tested via regex patterns): Deterministic.

### Live LLM Tests (extremely limited)

| File | Tests | What It Covers | Runs in CI? |
|------|-------|---------------|-------------|
| test_live_llm.py | 1 | Full graph smoke test | NO (skipif no API key) |
| test_profiling_live.py | 2 | Gemini schema validation | NO (skipif no API key) |
| test_llm_judge_live.py | 13 | Judge panel execution | NO (skipif no API key) |

**Total live LLM tests**: 16. All are skip-gated. None run in CI.

## Untested or Under-Tested Modules

| Module | Lines | Test References | Gap Severity |
|--------|-------|-----------------|-------------|
| `src/observability/llm_judge.py` | 1150 | 6 files reference it, but core scoring logic is tested with mocks | MEDIUM — complex scoring logic, but evaluation is a dev tool |
| `src/observability/evaluation.py` | 459 | 6 files reference | MEDIUM — evaluation framework, dev tool |
| `src/state_backend.py` | 416 | 6 files reference, tested via state tests | LOW |
| `src/observability/traces.py` | 148 | 4 files reference | LOW — thin wrapper |
| `src/agent/hours.py` | 161 | 2 files reference (test_hours.py exists) | LOW |
| `src/agent/handoff.py` | 94 | 2 files reference (test_handoff.py exists) | LOW |
| `src/data/validators.py` | 75 | 2 files reference (test_validators.py exists) | LOW |
| `src/api/errors.py` | 59 | 4 files reference, exercised via API tests | LOW |

No source module has ZERO test coverage. Every `.py` file has at least a corresponding test or is exercised through integration tests.

## Conftest Analysis

**File**: `tests/conftest.py` (237 lines)

### Critical Fixtures (all autouse=True)

1. **`_disable_semantic_injection_in_tests`**: Sets `SEMANTIC_INJECTION_ENABLED=false`. This means the semantic injection classifier (fail-closed LLM call) is OFF for 99.6% of tests. Only `test_e2e_security_enabled.py` re-enables it.

2. **`_disable_api_key_in_tests`**: Sets `API_KEY=""`. Auth middleware is bypassed for all tests except `test_auth_e2e.py` and `test_e2e_security_enabled.py`.

3. **`_clear_singleton_caches`**: Comprehensive singleton cleanup between tests — covers 17 different caches (LLM, validator, whisper, circuit breaker, embeddings, retriever, checkpointer, guest profile, casino config, feature flags, CMS, SMS, middleware, state backend, langfuse, sentiment, guardrails, semaphore, retrieval pool). This is thorough and well-maintained.

### Shared Fixtures
- `test_property_data`: Small test dataset (JSON) with restaurants, entertainment, hotel, gaming, FAQ, amenities, promotions.
- `test_property_file`: Writes test data to tmp_path.

### Assessment
The conftest is well-structured. The autouse auth/classifier disabling is documented and intentional (R40, R47 fixes). Dedicated test files re-enable these paths. Singleton cleanup is comprehensive (17 caches) — prevents cross-test pollution.

## Red Flags

### RED FLAG 1: Rule 8 ("NO MOCK TESTING") is Fiction
Rule 8 states "all tests must use live real LLM API calls." In practice, 52.9% of test files mock LLM calls, with 1694 mock occurrences. Only 16 tests (0.45% of 3555) actually call a live LLM, and all are skip-gated for CI. This rule was likely aspirational when written and never enforced.

**Impact**: LOW for code quality (mocked tests still verify wiring, logic, error handling), but HIGH for documentation honesty. Anyone reading CLAUDE.md would believe tests are live-LLM-validated.

### RED FLAG 2: Auth + Classifier Disabled by Default
The conftest globally disables two critical production security features:
- API key authentication (ApiKeyMiddleware)
- Semantic injection classification (LLM-based)

While dedicated test files re-enable them, the vast majority of tests (99%+) run with a neutered security posture. Coverage claims include these neutered paths.

**Mitigated by**: `test_auth_e2e.py` (13 tests) and `test_e2e_security_enabled.py` (15 tests) explicitly test with auth/classifier enabled. This is a known pattern (documented in R40, R47 comments).

### RED FLAG 3: No CI-Enforced Live LLM Tests
All 16 live LLM tests are `skipif not GOOGLE_API_KEY`. Without a CI secret, these never run automatically. A schema change in Gemini API (documented risk in R76) would be caught only by manual local runs.

### RED FLAG 4: Coverage Report Was Overwritten
My `--collect-only` run (which doesn't execute tests) overwrote `coverage.xml` with 24.97% because `--cov` is in pyproject.toml `addopts`. The committed coverage was 90.62% from a real test run. This means any `pytest --collect-only` or partial test run overwrites the coverage artifact. Not a quality issue but a CI artifact fragility risk.

### RED FLAG 5: RAG Tests Use FakeEmbeddings Exclusively
All RAG tests use hash-based FakeEmbeddings. This is correct for testing ingestion/retrieval logic, but means the actual Google embedding model (text-embedding-004) is never tested in the test suite. Embedding model version drift (a documented historical issue from R1) would not be caught.

### Observations (Not Red Flags)

- **Guardrails testing is excellent**: 530+ tests across 5 files, including fuzz testing (Hypothesis) and ReDoS safety. Best-tested component.
- **State management is solid**: Serialization safety, parity checks, and singleton cleanup are thoroughly tested.
- **E2E wiring tests are valuable**: Even with mocks, `test_full_graph_e2e.py` and `test_e2e_pipeline.py` catch node renaming, edge miswiring, and state key typos. The schema-dispatching mock LLM pattern is well-implemented.
- **External service mocking is appropriate**: CMS (Google Sheets), SMS (Telnyx) correctly mock external APIs — you don't test third-party services in unit tests.
- **Test count is high**: 3555 tests for 56 source modules is ~63 tests per module on average.

## Confidence: 92%

High confidence in the findings. I read conftest.py, sampled 15+ test files, cross-referenced all 56 source modules against test references, and verified coverage data from git history. The only uncertainty is whether some test files import mock for fixtures but don't actually mock LLM calls (possible minor overcount in the "55 files use mocks" figure — but the total 1694 occurrences confirms heavy mock usage).

## Verdict: ADEQUATE (with documentation dishonesty)

The test suite is **structurally sound**: high count (3555), 90.62% line coverage, comprehensive singleton cleanup, dedicated auth/security test paths, excellent deterministic guardrail testing.

However, the **documentation is misleading**:
1. Rule 8 ("NO MOCK TESTING") is violated by 55 of 104 test files
2. The test suite primarily validates **wiring, logic, and error handling** — not **LLM output quality**
3. Live LLM behavioral quality is measured by the separate evaluation framework (tests/evaluation/), not by pytest

The test suite does what a good test suite should: catches regressions in code logic, wiring, and state management. It does NOT do what Rule 8 claims: validate live LLM behavior. That function is served by the evaluation system (run_live_eval.py, judge panels) which operates outside of pytest.

**Recommendation**: Update Rule 8 to reflect reality. Something like: "Unit tests mock LLM calls for speed and CI reliability. Live LLM behavior is validated through the evaluation framework (tests/evaluation/). Schema compatibility is spot-checked via @pytest.mark.live tests."
