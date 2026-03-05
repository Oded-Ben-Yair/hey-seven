# Component 3: Specialist Agents

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/agent/agents/_base.py` | 1,379 | Shared `execute_specialist()` function: prompt assembly, behavioral signals, sarcasm detection, sentiment tone guides, proactive suggestions, few-shot examples, cross-domain hints, frustration/crisis suppression, persona reinject, booking context, profiling question injection, incentive injection, LLM backpressure (semaphore), error handling, retry feedback |
| `src/agent/agents/dining_agent.py` | 91 | Thin wrapper: `DINING_SYSTEM_PROMPT` template + `dining_agent()` calls `execute_specialist()` |
| `src/agent/agents/entertainment_agent.py` | 93 | Thin wrapper: `ENTERTAINMENT_SYSTEM_PROMPT` + `entertainment_agent()` |
| `src/agent/agents/comp_agent.py` | 121 | Thin wrapper: `COMP_SYSTEM_PROMPT` + `comp_agent()` (R77 fix: removed profile completeness gate) |
| `src/agent/agents/hotel_agent.py` | 86 | Thin wrapper: `HOTEL_SYSTEM_PROMPT` + `hotel_agent()` |
| `src/agent/agents/host_agent.py` | 37 | Thin wrapper: uses `CONCIERGE_SYSTEM_PROMPT` from prompts.py + `host_agent()` |
| `src/agent/agents/registry.py` | 39 | `_AGENT_REGISTRY` dict mapping 5 agent names to functions |
| `src/agent/agents/__init__.py` | 17 | Re-exports `get_agent` |

**Total: 1,863 lines across 8 files (1,379 in _base.py, 484 in thin wrappers + registry).**

## Wiring Verification

**Fully wired.**

- `_base.py`: `execute_specialist` imported by all 5 agent modules: `dining_agent.py:14`, `entertainment_agent.py:14`, `comp_agent.py:20`, `hotel_agent.py:14`, `host_agent.py:13`
- Each agent function imported by `registry.py:9-13` into `_AGENT_REGISTRY`
- `_AGENT_REGISTRY` / `get_agent` imported by `dispatch.py:21` for runtime dispatch
- `dispatch.py:362`: `agent_fn = get_agent(agent_name)` -> `result = await agent_fn({**state, **guest_context_update})`

**DRY pattern**: All 5 agents call `execute_specialist()` with their unique `system_prompt_template` and `no_context_fallback`. ~80% of logic lives in `_base.py`, each wrapper is 37-121 lines.

## Architectural Strengths

1. **DRY extraction** via `execute_specialist()` with dependency injection (`get_llm_fn`, `get_cb_fn`). Unanimously praised across 20+ review rounds as "the single best change."
2. **LLM backpressure** (`_base.py:1253-1270`): `asyncio.Semaphore(20)` with configurable timeout prevents request pile-up during LLM slowdowns. Returns fallback instead of queueing indefinitely.
3. **Sarcasm detection override** (`_base.py:358-371`): Context-contrast sarcasm detection overrides VADER when conversation history contradicts current sentiment.
4. **Frustration suppression** (`_base.py:1069-1092`): Hard override for comp/promotional agents when guest is frustrated. Replaces marketing tone with factual, empathy-first guidance.
5. **Crisis state suppression** (`_base.py:1094-1106`): ALL specialists suppress promotional content when `crisis_active` is True.
6. **Persona drift prevention** (`_base.py:1187-1212`): Re-injects condensed persona reminder after 5+ human turns.
7. **Flash->Pro model routing** (`_base.py:1236-1243`): Reads `model_used` from state (set by dispatch layer), selects Pro model for complex queries.
8. **CancelledError handling** (`_base.py:1295`): Records cancellation (not failure) on client disconnect, preventing SSE disconnects from inflating CB failure count.
9. **Semaphore leak protection** (`_base.py:1250-1255`): `acquired` flag with try/finally prevents semaphore count leak on CancelledError between acquire and entering try block.
10. **Booking context injection** (`_base.py:1011-1037`): Specialist-specific qualifying questions for reservations (R92).

## Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `test_agents.py` | 47 | Individual agent function tests (MOCKED) |
| `test_base_specialist.py` | 14 | `execute_specialist()` core logic: CB open fallback, empty context fallback, normal execution (MOCKED) |
| `test_r21_agent_quality.py` | 28 | Agent quality: tone, grounding, format (MOCKED) |
| `test_graph_v2.py` | 67 | Specialist dispatch integration (MOCKED) |
| `test_frustration_suppression.py` | varies | Frustration override logic |
| `test_cross_domain_hint.py` | varies | Cross-domain suggestion logic |
| `test_proactive_suggestion.py` | varies | `_should_inject_suggestion()` gating |
| `test_bridge_templates.py` | varies | Bridge template rendering |
| `test_engagement.py` | varies | Conversation dynamics detection |
| `test_behavioral_extraction.py` | varies | Behavioral signal extraction |

**Total: ~156+ tests covering specialist agents.**

## Live vs Mock Assessment

**Fully mocked.** All specialist agent tests mock `_get_llm` and `_get_circuit_breaker`. No test calls the real Gemini API through `execute_specialist()`.

- `test_base_specialist.py`: Uses `MagicMock` for both LLM and CB (line 47-59)
- `test_agents.py`: All 47 tests mock at the function level
- Pure logic helpers (`_detect_conversation_dynamics`, `_should_inject_suggestion`, `_build_behavioral_prompt_sections`) test deterministic Python logic, so no LLM needed — these are appropriately mock-free.

**Rule 8 gap**: The most critical path — `execute_specialist()` -> LLM.ainvoke() -> response generation — is never tested with a live Gemini API. The behavioral evaluation in `tests/evaluation/v2/` covers this via live agent calls, but the unit test suite is entirely mocked.

## Known Gaps

1. **`_base.py` at 1,379 lines**: Still large despite SRP extraction in R72. Contains ~15 distinct responsibilities (prompt assembly, behavioral signals, sarcasm, sentiment, dynamics, suggestions, profiling, incentives, booking, few-shot, persona reinject, error handling, LLM call, retry, response normalization). Consider further extraction into a `prompt_builder.py` module.
2. **System prompt size**: The system prompt assembled in `execute_specialist()` can be very large — base template + context + behavioral sections + agent behavior + booking context + few-shot examples + profiling + incentives + persona reinject + retry feedback. No measurement of total token count. Risk of exceeding model context window on complex turns.
3. **No prompt token counting**: No check or logging of system prompt token count before sending to LLM. A prompt that exceeds the model's context window will silently truncate or fail.
4. **Lazy imports inside function body** (`_base.py:838, 1043, 1109, 1112, 1129, 1152, 1238`): 7 lazy imports inside `execute_specialist()`. Each runs on every function call. While Python caches module objects, the overhead of 7 import lookups per request is unnecessary.
5. **No specialist-specific fallback for Pro model timeout**: When Pro model is selected but times out, the fallback is the same generic message. Could retry with Flash as a degraded response.
6. **`_LLM_SEMAPHORE` is global, not per-specialist**: A slow dining query blocks semaphore slots for all specialists. Per-specialist semaphores would provide better isolation.
7. **Comp agent's `COMP_SYSTEM_PROMPT` embeds Mohegan Sun details** (`comp_agent.py:47-55`): Momentum Rewards program details are hardcoded in the prompt template instead of coming from RAG or casino profile. Breaks multi-property portability.

## Confidence: 82%

The DRY extraction pattern is excellent, and the behavioral intelligence (sarcasm detection, frustration suppression, crisis override, proactive suggestions) is sophisticated. The main concerns are (1) `_base.py` is still too large with too many responsibilities, (2) no prompt token counting, (3) all tests are mocked, and (4) comp agent has hardcoded property details.

## Verdict: production-ready (with caveats)

The specialist agent pattern is production-grade. The thin wrapper + shared execution pattern is clean and well-tested for correctness. The behavioral intelligence layer is impressive. Key caveats: `_base.py` needs further decomposition, prompt size should be monitored, and comp agent prompt should use dynamic data for multi-property support.
