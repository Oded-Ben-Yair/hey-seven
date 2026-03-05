# Component 2: Router + Dispatch

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/agent/nodes.py` | 1,674 | Router node, retrieve node, validate node, respond node, fallback node, greeting node, off_topic node, `route_from_router()`, `_select_model()` (Flash->Pro routing), LLM singletons (`_get_llm`, `_get_complex_llm`, `_get_validator_llm`), `_normalize_content()`, `_enforce_tone()`, `_SLOP_PATTERNS` |
| `src/agent/dispatch.py` | 500 | Specialist dispatch orchestrator: `_dispatch_to_specialist()` (graph node), `_route_to_specialist()` (LLM + keyword fallback), `_inject_guest_context()`, `_execute_specialist()`, `_extract_node_metadata()`, `_keyword_dispatch()`, `_CATEGORY_TO_AGENT`, `_CATEGORY_PRIORITY` |
| `src/agent/agents/registry.py` | 39 | Agent registry: `get_agent(name)`, `list_agents()`, `_AGENT_REGISTRY` dict mapping 5 agent names to functions |

**Total: 2,213 lines across 3 files.**

## Wiring Verification

**Fully wired.**

- `nodes.py`: Core node functions imported by `graph.py:73-83` (`router_node`, `retrieve_node`, `validate_node`, `respond_node`, `fallback_node`, `greeting_node`, `off_topic_node`, `route_from_router`). LLM singletons (`_get_llm`, `_select_model`) imported by `dispatch.py:32`.
- `dispatch.py`: `_dispatch_to_specialist` imported by `graph.py:57` and registered as the `NODE_GENERATE` node function. Helper functions re-exported via `graph.py` for backward compat.
- `registry.py`: `_AGENT_REGISTRY`, `get_agent` imported by `dispatch.py:21`. Re-exported via `agents/__init__.py:8`.

**Entry point chain**: `app.py` -> `graph.build_graph()` -> `_dispatch_to_specialist` (generate node) -> `_route_to_specialist` -> `get_agent(name)` -> specialist function

## Architectural Strengths

1. **Three-phase dispatch**: Route -> Context inject -> Execute, each as a separate function. Clean SRP extraction from the original monolith (R52 D1).
2. **Structured LLM dispatch + keyword fallback**: `_route_to_specialist()` tries structured output (`DispatchOutput`) first, falls back to `_keyword_dispatch()` when circuit breaker is open or parsing fails. Defense-in-depth.
3. **Deterministic model routing** (`_select_model`): Flash->Pro routing based on confidence, sentiment, complexity, and crisis state. No additional LLM call required.
4. **MappingProxyType for immutable routing maps**: `_CATEGORY_TO_AGENT` and `_CATEGORY_PRIORITY` are frozen at module level. Prevents accidental mutation in concurrent context.
5. **Deterministic tie-breaking** (`_keyword_dispatch:149-152`): `max()` uses 3-key tuple `(count, priority, name)` for reproducible results on ties.
6. **Retry reuse** (`_route_to_specialist:195-201`): RETRY path reuses the same specialist. Prevents non-deterministic specialist switching on retry.
7. **Result sanitization** (`_execute_specialist:413-436`): Strips dispatch-owned keys from specialist results (R47 fix), filters unknown state keys, persists specialist name and dispatch method.
8. **Three separate TTL-cached LLM singletons** with jitter: `_get_llm` (Flash), `_get_complex_llm` (Pro), `_get_validator_llm`. Separate `asyncio.Lock` per cache prevents cascading stalls.
9. **Fail-safe routing**: Router LLM failure -> `off_topic` (safe). Parse failure -> `property_qa` at 0.5 confidence. Both avoid sending unclassified queries through the full pipeline.
10. **`string.Template.safe_substitute`**: Dispatch prompt uses `$variable` syntax. Won't crash on user input containing `{braces}`.

## Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `test_nodes.py` | 162 | Router classification, retrieve, validate, respond, greeting, off_topic, tone enforcement (MOCKED) |
| `test_dispatch.py` | 34 | `_keyword_dispatch()` category mapping, tie-breaking, `_DISPATCH_OWNED_KEYS`, `_VALID_STATE_KEYS`, `_extract_node_metadata()` (NO MOCKS - pure logic) |
| `test_graph_v2.py` | 67 | Specialist dispatch via keyword fallback (mocked CB blocks structured output) (MOCKED) |
| `test_full_graph_e2e.py` | 9 | Full pipeline dispatch (MOCKED) |
| `test_agents.py` | varies | Agent execution tests (MOCKED) |
| `test_slop_enforcement.py` | varies | `_enforce_tone()` and `_SLOP_PATTERNS` |
| `test_multilingual.py` | varies | Router language detection |

**Total: ~272+ tests covering router + dispatch.**

## Live vs Mock Assessment

**Heavily mocked.**

- `test_nodes.py` (162 tests): All mock `_get_llm` at the function level. Line 6: `from unittest.mock import AsyncMock, MagicMock, patch`.
- `test_dispatch.py` (34 tests): Pure logic tests for keyword dispatch, metadata extraction, and state key validation. No mocks needed, no LLM calls. These are genuinely live-equivalent.
- Router classification tests: All use mocked structured output. The `RouterOutput` Pydantic model is tested for literal validation in `test_state_parity.py`, but the actual `router_llm.ainvoke()` call to Gemini is never tested live except in `test_live_llm.py`.

**Rule 8 gap**: The dispatch structured output path (`_route_to_specialist` LLM call) is only tested via mocks. No live test verifies that `DispatchOutput` schema is accepted by Gemini.

## Known Gaps

1. **nodes.py is 1,674 lines**: Largest file in the codebase. Contains 8 node functions + routing + LLM management + tone enforcement. Could be further decomposed (e.g., extract LLM singletons to a `llm_factory.py`, extract `_enforce_tone` and `_SLOP_PATTERNS` to a `tone.py`).
2. **No live integration test for structured dispatch**: `_route_to_specialist` LLM -> `DispatchOutput` is only mocked. Per R76 lesson, mock tests don't validate Pydantic schema complexity against real Gemini.
3. **`ValueError` conflation** in `_route_to_specialist` (`dispatch.py:254-267`): `ValueError` catches both parse errors (prompt issue) and SDK quota/auth errors (429s). Comment documents the limitation but no resolution.
4. **CB parse error handling** (`dispatch.py:256`): Does NOT record CB failure for parse errors. This means repeated parse failures from a broken prompt will not trip the circuit breaker. Correct for LLM availability tracking, but a persistent parse bug goes undetected by CB metrics.
5. **`_select_model` does not check `MODEL_ROUTING_ENABLED` flag per-casino**: Uses `settings.MODEL_ROUTING_ENABLED` (global), not `is_feature_enabled(casino_id, ...)`. Cannot enable Pro routing for specific casinos during canary rollout.
6. **Router fallback to `off_topic` on exception** (`nodes.py:414`): A network timeout sends all queries to off_topic, which is safe but means zero functionality during outages. No degraded mode that serves cached responses.
7. **No observability for model routing decisions in SSE**: `_select_model` logs but doesn't emit routing decision via SSE `graph_node` metadata for the `generate` node. The `model_used` field is set in state but not surfaced to the client.

## Confidence: 85%

The dispatch logic is well-structured with proper separation of concerns, immutable routing maps, deterministic tie-breaking, and defense-in-depth (LLM + keyword fallback). The main gaps are (1) no live integration test for structured dispatch, (2) `nodes.py` size (1,674 lines — should be decomposed), and (3) `ValueError` conflation in CB handling.

## Verdict: production-ready

Router and dispatch logic are production-grade. The keyword fallback ensures functionality even when the dispatch LLM is down. Flash->Pro model routing is deterministic and well-documented. The main risk is the `nodes.py` monolith size, which increases the blast radius of any edit. Recommend extracting LLM factory and tone enforcement into separate modules as a follow-up.
