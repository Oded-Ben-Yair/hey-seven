# Round 9 Summary: Code Simplification

**Date**: 2026-02-20
**Scores**: Gemini=67, GPT=70.5, Grok=63.9 | **Average: 67.1**
**Previous**: R8 avg=63.7 | **Delta: +3.4**

---

## Consensus Findings Fixed (2/3+ reviewers)

| # | Finding | Severity | Reviewers | Lines Removed |
|---|---------|----------|-----------|---------------|
| 1 | `validate_state_transition()` dead code (37 lines, zero callers) | CRITICAL | GPT+Grok+Gemini | 37 |
| 2 | `ExtractedFields` unused Pydantic model (14 lines, zero references) | CRITICAL | GPT+Grok | 14 |
| 3 | `CasinoHostState` deprecated alias (persisted 9 rounds) | HIGH | GPT+Grok+Gemini | ~15 |
| 4 | `CircuitBreakerConfig` dead dataclass + `config` param path | HIGH | GPT+Gemini+Grok | 25 |
| 5 | `_FailureCounter` over-engineered class (47 lines -> 3 module-level vars) | HIGH | Gemini+GPT+Grok | 44 |
| 6 | 5 `detect_*` guardrails share identical structure (DRY violation) | CRITICAL | Grok+GPT | ~70 |
| 7 | 4 duplicate fallback Template constructions in `_base.py` | HIGH | Grok+GPT | ~30 |
| 8 | httpx + catch-all exception handlers identical in `_base.py` | HIGH | Grok+GPT | ~20 |
| 9 | Duplicate turn-limit guard in `router_node` (unreachable) | MEDIUM | Gemini+GPT | 7 |
| 10 | Duplicate `_extract_node_metadata` cases for compliance_gate/router | LOW | Gemini+GPT | 4 |
| 11 | `_failure_count` backward-compat alias in circuit_breaker | LOW | GPT | 4 |
| 12 | `_rerank_by_rrf` backward-compat alias in tools.py | LOW | Gemini | 2 |
| 13 | `get_default_features()` wrapper (only tests used it) | MEDIUM | GPT+Grok | 7 |
| 14 | `clear_circuit_breaker_cache()` zero-logic wrapper (zero callers) | MEDIUM | GPT | 7 |
| 15 | Guardrails re-exports from `nodes.py` (unnecessary indirection) | HIGH | GPT | 8 |
| 16 | Unused `DEFAULT_FEATURES` import in `_base.py` | LOW | GPT | 1 |
| 17 | Unused `httpx` import in `_base.py` (after handler consolidation) | LOW | — | 1 |

## Key Simplifications

1. **Guardrails DRY**: Extracted `_check_patterns()` helper. Each `detect_*` function is now a one-liner delegating to the shared helper. Reduced ~110 lines of near-identical code.

2. **Fallback message DRY**: Extracted `_fallback_message()` in `_base.py`. Circuit breaker, ValueError, and network error handlers all use the same function. Single source of truth for property contact info.

3. **Exception handler consolidation**: Merged `except (httpx.HTTPError, ...)` and `except Exception` in `_base.py` into one handler. Removed `httpx` import.

4. **Dead code deletion**: Removed `validate_state_transition()`, `ExtractedFields`, `CasinoHostState` alias, `CircuitBreakerConfig`, `_FailureCounter` class, `clear_circuit_breaker_cache()`, `get_default_features()`, backward-compat aliases.

5. **Import canonicalization**: Tests now import guardrails from `src.agent.guardrails` (canonical) instead of `src.agent.nodes` (re-export). Tests import `rerank_by_rrf` from `src.rag.reranking` instead of `src.agent.tools`.

## Files Modified

| File | Change |
|------|--------|
| `src/agent/state.py` | Removed `ExtractedFields`, `validate_state_transition`, `CasinoHostState` alias (-55 lines) |
| `src/agent/__init__.py` | Removed `CasinoHostState` export |
| `src/agent/circuit_breaker.py` | Removed `CircuitBreakerConfig`, `config` param, `_failure_count` alias, `clear_circuit_breaker_cache` (-45 lines) |
| `src/agent/guardrails.py` | Extracted `_check_patterns()` helper, 4 detect_* are now one-liners (-70 lines) |
| `src/agent/agents/_base.py` | Extracted `_fallback_message()`, merged exception handlers, removed unused imports (-30 lines) |
| `src/agent/whisper_planner.py` | Replaced `_FailureCounter` class with 3 module-level vars, merged except blocks (-44 lines) |
| `src/agent/nodes.py` | Removed guardrails re-exports, duplicate turn-limit guard (-15 lines) |
| `src/agent/graph.py` | Merged duplicate `_extract_node_metadata` cases (-4 lines) |
| `src/agent/tools.py` | Removed `_rerank_by_rrf` alias (-2 lines) |
| `src/casino/feature_flags.py` | Removed `get_default_features()` (-7 lines) |
| `src/casino/__init__.py` | Removed `get_default_features` export |
| `tests/conftest.py` | Added whisper failure counter reset to singleton cleanup |
| `tests/test_nodes.py` | Updated guardrail imports, turn-limit test, `_failure_count` reference |
| `tests/test_graph_v2.py` | Removed `CasinoHostState` alias tests |
| `tests/test_compliance_gate.py` | Removed `CasinoHostState` alias test |
| `tests/test_firestore_retriever.py` | Replaced `CircuitBreakerConfig` tests with direct construction tests |
| `tests/test_whisper_planner.py` | Rewrote `_FailureCounter` tests for module-level counter |
| `tests/test_rag.py` | Updated `_rerank_by_rrf` imports to canonical `rerank_by_rrf` |
| `tests/test_casino_config.py` | Replaced `get_default_features` with `dict(DEFAULT_FEATURES)` |
| `tests/test_phase3_integration.py` | Replaced `get_default_features` with `dict(DEFAULT_FEATURES)` |

## Findings NOT Fixed (intentional)

| Finding | Reason |
|---------|--------|
| PII buffer in `chat_stream()` (Gemini F6, Grok H3) | Only SSE-level PII protection; persona_envelope is post-stream. Counterpoint accepted by Gemini. |
| Feature flags system (Gemini F3) | Multi-tenant API shape is defensible per architecture doc. Only 1/3 consensus. |
| RAG formatters DRY (Gemini F2) | 8 formatters with category-specific logic. Risky to refactor without RAG regression tests on real data. |
| Specialist dispatch complexity (Gemini F13) | Architecturally sound separation of intent (router) vs domain (dispatch). Counterpoint accepted. |
| `list_agents()` in registry (Gemini F11, GPT F5, Grok M2) | Used by tests for registry verification. Kept as introspection utility. |
| `_KNOWN_NODES` / `_NON_STREAM_NODES` sets (GPT F12) | Prevents silent breakage on node renames. Legitimate safety measure. |
| Comment-to-code ratio (Grok M6) | Comments explain design decisions for future readers. Not excessive. |

## Test Results

- **Before**: 1262 passed, 20 skipped, 90.44% coverage
- **After**: 1256 passed, 20 skipped, 90.67% coverage
- **Delta**: -6 tests (removed dead code tests), +0.23% coverage (dead code removal)
- **Net production lines removed**: ~265 lines
