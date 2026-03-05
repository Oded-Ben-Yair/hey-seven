# Component 6: Profiling + Extraction

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/agent/profiling.py` | 594 | LLM-powered profiling enrichment node (between generate and validate), weighted completeness, golden path phases, profile confirmation injection |
| `src/agent/extraction.py` | 442 | Deterministic regex extraction (name, party_size, occasion, loyalty, urgency, fatigue, budget), LLM fallback for long messages, handoff summary formatting |
| `src/agent/whisper_planner.py` | 332 | Silent background LLM planner (WhisperPlan structured output), proactive suggestion generation, profiling question technique selection |
| `src/data/guest_profile.py` | 554 | Guest profile CRUD (Firestore + in-memory fallback), CCPA cascade delete, confidence decay, schema migration, namespaced preferences |
| **Total** | **1922** | |

## Wiring Verification

All 4 files are fully wired to the production graph:

**profiling.py:**
- `src/agent/graph.py` — `profiling_enrichment_node` added as graph node between generate and validate
- `src/agent/agents/_base.py:1129` — imports `PROFILING_TECHNIQUE_PROMPTS` for system prompt injection

**extraction.py:**
- `src/agent/profiling.py:462` — `from src.agent.extraction import get_guest_profile_summary` (profile confirmation)
- `src/agent/handoff.py:70` — `from src.agent.extraction import format_handoff_summary` (host handoff)
- `src/agent/agents/_base.py:1112` — `from src.agent.extraction import get_guest_profile_summary`

**whisper_planner.py:**
- `src/agent/agents/_base.py:33` — `from src.agent.whisper_planner import format_whisper_plan` (specialist prompt injection)
- `src/agent/profiling.py:381` — `from src.agent.whisper_planner import _get_whisper_llm` (shared LLM singleton)
- `src/agent/graph.py` — whisper_planner_node added to graph

**guest_profile.py:**
- `src/agent/dispatch.py:316` — `from src.data.guest_profile import get_agent_context` (specialist dispatch)
- `src/agent/dispatch.py:324` — `from src.data.guest_profile import namespace_preferences`
- `src/data/__init__.py:15` — re-exported in package __init__

**Verdict: All 4 files are fully wired and reachable from the production graph entry points.**

## Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `tests/test_profiling.py` | 92 | Completeness calculation, phase determination, field name mapping, weights, ProfileExtractionOutput schema |
| `tests/test_phase5_profiling.py` | 12 | Integration: profiling node with graph state, LLM extraction, question injection |
| `tests/test_profiling_instrumentation.py` | 27 | Telemetry, logging, fail-silent behavior |
| `tests/test_profiling_live.py` | 2 | Live Gemini API schema validation for ProfileExtractionOutput |
| `tests/test_extraction.py` | 27 | Regex extraction: name, party_size, occasion, date patterns |
| `tests/test_info_extraction_enhanced.py` | 22 | Enhanced extraction: loyalty signals, urgency, fatigue, budget, profile summary |
| `tests/test_extraction_llm.py` | 18 | LLM fallback extraction, merge logic (regex wins on conflicts) |
| `tests/test_behavioral_extraction.py` | 17 | Behavioral signals: loyalty, urgency, fatigue, budget from natural text |
| `tests/test_handoff.py` | 16 | format_handoff_summary: structured output, empty profiles, partial profiles |
| `tests/test_whisper_planner.py` | 30 | WhisperPlan schema, format_whisper_plan, _calculate_completeness, _format_history |
| `tests/test_guest_profile.py` | 56 | CRUD operations, in-memory store, CCPA delete, confidence tracking, migration |
| `tests/test_namespaced_prefs.py` | 6 | namespace_preferences function coverage |
| **Total** | **325** | |

## Live vs Mock Assessment

**Mixed — live LLM for profiling/whisper, deterministic for extraction:**

- `test_profiling_live.py` (2 tests): **LIVE** — calls Gemini API to validate ProfileExtractionOutput schema acceptance. Marked `@pytest.mark.live`. This is the critical test per Rule 8 (validates Gemini doesn't reject the schema).
- `test_phase5_profiling.py` (12 tests): Uses live LLM calls for profiling enrichment node integration tests.
- `test_whisper_planner.py` (30 tests): Majority test format_whisper_plan and helpers (deterministic). WhisperPlan node tests use live LLM.
- `test_extraction.py`, `test_behavioral_extraction.py`, `test_info_extraction_enhanced.py` (66 tests total): **Deterministic** — regex extraction tests. No LLM needed (pure regex).
- `test_extraction_llm.py` (18 tests): Tests LLM fallback extraction path with live calls.
- `test_guest_profile.py` (56 tests): **Deterministic** — CRUD, in-memory store, confidence math. No LLM needed.
- `test_handoff.py` (16 tests): **Deterministic** — string formatting. No LLM needed.

**Assessment: Appropriate split. Regex extraction and CRUD are deterministic — no mocks needed. LLM-dependent paths (profiling node, whisper planner) have live tests. Schema validation has dedicated `@pytest.mark.live` test.**

## Known Gaps

1. **P8 (Profile Completeness: 3.7)**: The weighted completeness calculation in `profiling.py` uses `_PROFILE_WEIGHTS` with 16 fields summing to 1.0. In 3-turn conversations, foundation fields (name, party_size, visit_purpose) cover ~37% max. The R96 strategy identifies this correctly: guests don't share enough info in 3 turns for 60% completeness. **Not a code bug — a conversation length limitation.** Phase 1 (Pro model) may improve extraction quality from the same turns.

2. **P9 (Host Handoff: 2.1)**: `format_handoff_summary()` in extraction.py exists (lines 370-442) and is wired via `handoff.py:70`. However, the R96 strategy notes this is "missing business logic, not model capability." The handoff summary is a structured markdown format but lacks: conversation history summary, recommended next actions, and risk flags. **Needs a HandoffOrchestrator tool per the R96 strategy.**

3. **Dual completeness calculations**: `whisper_planner.py:290` has `_calculate_completeness()` (simple field count against `_PROFILE_FIELDS`) while `profiling.py:188` has `_calculate_profile_completeness_weighted()` (proper weighted calculation). The whisper planner uses the simpler one. These may diverge, causing the planner to suggest profiling questions that the enrichment node considers unnecessary. **Minor inconsistency but not a blocker.**

4. **ProfileExtractionOutput schema hardcoded to 16 fields**: Adding new profile fields requires changes in 3 places: the Pydantic model, `_FIELD_NAME_MAP`, and `_PROFILE_WEIGHTS`. No parity assertion enforces this. R72 code quality rule (`assert set(TypedDict.__annotations__) == set(DEFAULT_X.keys())`) applies here.

5. **Whisper planner uses `MODEL_NAME` (Flash)**: `_get_whisper_llm()` at line 71 uses `settings.MODEL_NAME` which is Flash. For planning/strategy, Pro would likely produce better WhisperPlans. The R96 Phase 1 Pro switch should improve this path.

6. **Profile confirmation injection (R93)**: The `_CONFIRM_SIGNALS` check at profiling.py:457 uses simple substring matching. "Perfect" in "Not perfect" would trigger false positive confirmation. Low risk in practice but worth noting.

## Confidence: 75%

The profiling pipeline is architecturally sound: LLM extraction -> weighted completeness -> golden path phases -> question injection -> profile confirmation. The extraction layer (regex + LLM fallback) is well-tested. Guest profile CRUD with Firestore + in-memory fallback is solid with CCPA compliance. The main gaps are behavioral (P8 profile completeness limited by conversation length, P9 handoff needs business logic tool) rather than code quality issues.

## Verdict: needs-new-tool

The code is production-ready for what it does. P9 (Host Handoff: 2.1) requires a new HandoffOrchestrator tool that generates structured handoff summaries with conversation history, recommended actions, and risk flags. P8 improvement requires either longer conversations or smarter extraction (Pro model). Both align with the R96 Phase 2 strategy.
