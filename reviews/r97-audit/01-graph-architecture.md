# Component 1: Graph Architecture

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/agent/graph.py` | 706 | 13-node StateGraph assembly, `build_graph()`, `chat()`, `chat_stream()`, routing functions, SSE streaming, PII redaction |
| `src/agent/state.py` | 355 | `PropertyQAState` TypedDict (29 fields), 5 custom reducers (`_merge_dicts`, `_append_unique`, `_keep_latest_str`, `_keep_max`, `_keep_truthy`), Pydantic models (`RouterOutput`, `DispatchOutput`, `ValidationResult`, `RetrievedChunk`, `GuestContext`), `UNSET_SENTINEL` |
| `src/agent/constants.py` | 51 | Node name constants (13), `_KNOWN_NODES`, `_NON_STREAM_NODES` frozensets |

**Total: 1,112 lines across 3 files.**

## Wiring Verification

**Fully wired.** All 3 files are imported from production entry points:

- `graph.py`: Imported by `src/api/app.py:87` (`build_graph`), `src/api/app.py:391` (`chat_stream`), and re-exported via `src/agent/__init__.py`
- `state.py`: Imported by 16+ production modules: `dispatch.py`, `nodes.py`, `persona.py`, `compliance_gate.py`, `profiling.py`, `whisper_planner.py`, `pre_extract.py`, all 5 specialist agents, `tools.py`, `graph.py`, `__init__.py`
- `constants.py`: Imported by `graph.py`, `nodes.py`, `dispatch.py`

**Entry point chain**: `app.py` -> `graph.build_graph()` -> compiles `StateGraph(PropertyQAState)` with all 13 nodes

## Architectural Strengths

1. **Parity assertion at import time** (`graph.py:383-390`): Runtime `ValueError` (not `assert`) catches state schema drift between `PropertyQAState` and `_initial_state()` in ALL environments including `python -O`
2. **Custom reducers with algebraic properties**: `_merge_dicts` with None/empty filtering, UNSET tombstone, JSON-serializable sentinel. `_keep_max`, `_keep_truthy`, `_append_unique` have correct semantics.
3. **Feature flag dual-layer design** (`graph.py:226-260`): Build-time flags for topology, runtime flags for behavior. Well-documented rationale for why all-runtime is impractical.
4. **SSE streaming with PII redaction** (`graph.py:536-683`): Per-request `StreamingPIIRedactor` with fail-safe on `CancelledError` (drops buffer rather than emitting unredacted PII)
5. **Defensive routing** (`_route_after_validate_v2`): Handles unexpected `validation_result` by logging and routing to fallback
6. **GraphRecursionError handling**: Both `chat()` and `chat_stream()` catch recursion limit exceeded
7. **Handoff event emission** (`graph.py:633-641`): Captures handoff_request from any node output for self-harm, frustration, incentive approval

## Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `test_graph_v2.py` | 67 | Node-level + dispatch tests (MOCKED) |
| `test_graph_topology.py` | 16 | BFS reachability, stuck states, self-loops, happy path chain, conditional edges (NO MOCKS - structural) |
| `test_graph_properties.py` | 19 | Hypothesis property-based: reducer algebraic invariants (NO MOCKS - pure logic) |
| `test_state_parity.py` | 22 | Hypothesis + unit: initial state parity, Pydantic literal validation, reducer correctness, UNSET sentinel JSON roundtrip (NO MOCKS - pure logic) |
| `test_state_serialization.py` | 3 | JSON roundtrip for state and sentinel (NO MOCKS - pure logic) |
| `test_full_graph_e2e.py` | 9 | Full `build_graph()` -> `chat()` pipeline (MOCKED LLMs) |
| `test_e2e_pipeline.py` | 13 | E2E pipeline (MOCKED) |
| `test_chat_stream.py` | 13 | SSE streaming (MOCKED) |
| `test_sse_e2e.py` | 10 | SSE E2E (MOCKED) |
| `test_live_llm.py` | 1 | Live graph response via real Gemini API (LIVE, requires GOOGLE_API_KEY) |

**Total: ~173 tests covering graph architecture.**

## Live vs Mock Assessment

**Mixed — mostly mock, with important exceptions.**

- **MOCKED (majority)**: `test_graph_v2.py` (67 tests), `test_full_graph_e2e.py` (9 tests), `test_e2e_pipeline.py`, `test_chat_stream.py`, `test_sse_e2e.py` all mock LLMs at the `_get_llm` level. Line 4 of `test_graph_v2.py`: "All LLM calls are mocked."
- **STRUCTURAL (no mocks needed)**: `test_graph_topology.py` (16 tests) tests the compiled graph structure with BFS — no LLM calls involved. `test_graph_properties.py` and `test_state_parity.py` test pure Python reducer logic with Hypothesis.
- **LIVE**: `test_live_llm.py` has 1 test that calls `build_graph()` -> `chat()` with a real Gemini API. Gated by `GOOGLE_API_KEY` env var.

**Rule 8 compliance concern**: The project CLAUDE.md says "NO MOCK TESTING — all tests must use live real LLM API calls." Yet the majority of graph tests (67/67 in test_graph_v2.py, 9/9 in test_full_graph_e2e.py) are fully mocked. The structural/property tests (topology, reducers) legitimately don't need LLMs, but the E2E graph tests are mocked where they shouldn't be per Rule 8.

## Known Gaps

1. **Doc string says "12-node" but graph has 13 nodes** (`graph.py:321`: "Custom 12-node StateGraph compiled successfully" — should be 13 after profiling_enrichment was added). Minor cosmetic.
2. **SSE token streaming only from `generate` node** (`graph.py:565`): Non-streaming nodes (greeting, off_topic, fallback, persona, whisper, profiling, pre_extract) send full `replace` events. If any of these become slow, the client shows no progress.
3. **No `chat_stream()` integration with profiling/handoff metadata**: The handoff_request is captured but there's no SSE event for profiling phase transitions or model routing info. Client cannot display "thinking with Pro model" or "profiling phase: preference."
4. **MemorySaver as default checkpointer** (`graph.py:309`): Single-container dev is fine, but there's no automated smoke test verifying FirestoreSaver path actually works.
5. **Source deduplication in `_merge_sources`** uses `category:source` key which may not uniquely identify chunks from the same source with different content.
6. **`_initial_state` resets `responsible_gaming_count` to 0 each turn** — but the `_keep_max` reducer preserves the maximum. This is correct but subtle. If someone changes the reducer to `_keep_latest`, RG escalation breaks silently.

## Confidence: 88%

Strong graph architecture with formal topology verification, property-based reducer tests, and robust SSE streaming. The parity assertion is an excellent safety net. Main concerns are (1) heavy mock testing vs Rule 8 mandate, (2) no FirestoreSaver integration test, and (3) no live E2E test for the SSE streaming path.

## Verdict: production-ready

The graph assembly, state schema, and routing logic are production-grade. The 13-node topology is formally verified. Reducers have algebraic property tests. The only meaningful gap is the lack of live LLM integration tests for the full graph pipeline (beyond the single test in `test_live_llm.py`), but the structural foundation is sound.
