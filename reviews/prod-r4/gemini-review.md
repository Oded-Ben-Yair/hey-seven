# Hey Seven Production Code Review — Round 4 (Gemini 3 Pro)

**Date**: 2026-02-20
**Commit**: 9655fd2
**Spotlight**: TESTING GAPS (findings get +1 severity bump)
**Previous Scores**: R1=67.3 | R2=61.3 | R3=60.7

---

## Score Table

| # | Dimension | Score | Justification |
|---|-----------|:-----:|---------------|
| 1 | Graph/Agent Architecture | 9 | Solid 11-node StateGraph with specialist dispatch, validation loop, and defense-in-depth routing; only gap is untested whisper_planner_enabled=False topology branch. |
| 2 | RAG Pipeline | 8 | Per-item chunking, RRF reranking, SHA-256 idempotent ingestion, and version-stamp purging are all sound; Firestore retriever relies on mock-only tests. |
| 3 | Data Model / State Design | 9 | Strong TypedDict with parity assertion, CCPA batch delete, Annotated reducers; message windowing limit defined but truncation untested. |
| 4 | API Design | 7 | Pure ASGI middleware (no BaseHTTPMiddleware), SSE streaming with PII buffer, hmac.compare_digest for auth; however chat_stream's PII buffer logic is entirely mocked out in tests. |
| 5 | Testing Strategy | 5 | 1107 tests at 90.63% coverage, but coverage is inflated by mocking the most complex components (SSE streaming, Sheets client, Firestore, HITL). Critical production paths are effectively untested. **(Spotlight penalty applied)** |
| 6 | Docker & DevOps | 10 | Multi-stage build, non-root user, pinned base images, exec-form CMD, health checks, Cloud Build pipeline. No issues found. |
| 7 | Prompts & Guardrails | 9 | 73+ deterministic patterns across English/Spanish/Mandarin, structured output routing with Literal types, defense-in-depth (compliance_gate + router); minor gap in multi-byte SMS truncation. |
| 8 | Scalability & Production | 9 | Async-safe circuit breaker with get_state(), TTL-cached LLM singletons, separate validator lock/cache, bounded TTLCache for webhooks; no blocking calls found. |
| 9 | Documentation & Code Quality | 9 | Clean module docstrings, Google-style docstrings, node name constants, parity assertions, consistent patterns; no dead code or misleading comments found. |
| 10 | Domain Intelligence | 9 | Ed25519 webhook verification, TCPA quiet hours, BSA/AML guardrails, responsible gaming helplines (CT-specific 1-800-MY-RESET), age verification with configurable state, comp system with profile completeness threshold. |
| **Total** | | **84** | |

---

## Findings

### Finding 1 (CRITICAL): chat_stream() PII Redaction Buffer Has Zero Direct Test Coverage

- **Location**: `src/agent/graph.py:449-620` (chat_stream function)
- **Problem**: The `chat_stream()` function implements ~150 lines of critical PII redaction logic: a streaming buffer that accumulates tokens, detects digits (potential phone/SSN/card patterns), and flushes at sentence boundaries or after 80 chars. `test_api.py` mocks `chat_stream` entirely via `_mock_chat_stream()`, meaning the actual buffering, digit detection, flush threshold, and sentence-boundary flush logic is **never executed in any test**. This is the single most complex function in the codebase and it handles PII security.
- **Impact**: If the PII buffer has an off-by-one error, fails to flush on stream end, or mishandles the digit regex, PII (credit card numbers, SSNs, phone numbers) will leak to the client in production. This is a compliance/security risk in a regulated casino environment.
- **Fix**: Create `TestChatStreamPiiBuffer` in `test_api.py` (or a new `test_chat_stream.py`). Instantiate `chat_stream()` directly with a mock compiled graph that yields `astream_events` containing split PII tokens (e.g., `["My card is ", "4242", " ", "4242", " ", "4242", " ", "4242"]`). Assert the yielded SSE events contain `[CARD]` instead of the raw number. Also test: (a) buffer flush at sentence boundary, (b) buffer flush at 80-char threshold, (c) non-digit text flushes immediately, (d) buffer NOT flushed on error (dropped for safety).

### Finding 2 (HIGH): whisper_planner_enabled=False Graph Topology Branch Untested

- **Location**: `src/agent/graph.py:310-314`
- **Problem**: The graph compilation has a conditional branch: when `whisper_planner_enabled=False`, the topology changes from `retrieve → whisper_planner → generate` to `retrieve → generate` (skipping the whisper node entirely). There is **zero test coverage** for this configuration path. No test in `test_graph_v2.py` or any other file compiles the graph with this flag disabled.
- **Impact**: If `NODE_GENERATE` (specialist dispatch) depends on state keys normally populated by `whisper_planner_node` (e.g., `whisper_plan`), the graph will crash with unexpected state when this flag is disabled. Since the whisper planner sets `whisper_plan` and `extracted_fields`, specialists that check these fields could behave incorrectly.
- **Fix**: Add to `test_graph_v2.py`:
```python
class TestGraphWithWhisperDisabled:
    def test_compiles_with_whisper_disabled(self):
        with patch.dict("src.casino.feature_flags.DEFAULT_FEATURES", {"whisper_planner_enabled": False}):
            graph = build_graph()
            drawable = graph.get_graph()
            user_nodes = set(drawable.nodes) - {"__start__", "__end__"}
            assert "whisper_planner" not in user_nodes
            # retrieve connects directly to generate
            retrieve_edges = [e for e in drawable.edges if e.source == "retrieve"]
            assert any(e.target == "generate" for e in retrieve_edges)
```

### Finding 3 (HIGH): Google Sheets Client (`cms/sheets_client.py`) Completely Untested

- **Location**: `src/cms/sheets_client.py`
- **Problem**: `test_cms.py` covers webhook and validation logic (53 tests), but there is **no test file** for `sheets_client.py`. The Google Sheets integration (authentication, cell parsing, error handling) has zero test coverage.
- **Impact**: Authentication failures, API changes, empty row handling, and header mismatch errors will break the CMS sync process silently in production. Since the CMS feeds the RAG pipeline, this is a single point of failure for knowledge base freshness.
- **Fix**: Create `test_sheets_client.py` with mocked `gspread` (or equivalent) client. Test: (a) successful sheet fetch returns expected format, (b) authentication failure returns meaningful error, (c) empty sheet returns empty list (not crash), (d) missing expected columns raises validation error.

### Finding 4 (HIGH): HITL Interrupt Path Defined but Untested

- **Location**: `src/agent/graph.py:347-348`
- **Problem**: The code supports `ENABLE_HITL_INTERRUPT=True` which compiles the graph with `interrupt_before=[NODE_GENERATE]`. No test verifies that: (a) the graph actually pauses before generate, (b) the graph can be resumed after an interrupt, (c) state is preserved across the pause/resume boundary.
- **Impact**: If operations attempts to intervene in a live conversation (HITL review), the system may fail to pause (generating a response anyway, violating the review gate) or fail to resume (losing conversation context), leading to incoherent or non-compliant guest interactions.
- **Fix**: Add an integration test:
```python
@pytest.mark.asyncio
async def test_hitl_interrupt_pauses_before_generate(self):
    with patch.dict(os.environ, {"ENABLE_HITL_INTERRUPT": "true"}):
        graph = build_graph()
        config = {"configurable": {"thread_id": "test-hitl"}}
        # Should pause before generate
        result = await graph.ainvoke(_initial_state("What restaurants?"), config=config)
        # Verify state has retrieved_context but no AI response yet
        # Then resume and verify completion
```

### Finding 5 (HIGH): No Test Coverage for SSE Stream Error Paths

- **Location**: `src/agent/graph.py:516-620` (try/except in chat_stream)
- **Problem**: There are no tests covering exceptions raised *during* the SSE stream generation. Specifically: (a) `asyncio.CancelledError` from client disconnect (should be logged at INFO, not ERROR), (b) graph errors mid-stream (should yield a proper `error` SSE event), (c) the PII buffer behavior on error (documented in R3 as "safer to drop on error" but never verified by a test).
- **Impact**: If an error occurs mid-stream, the PII buffer might be flushed unredacted (if the finally block is flawed), or the client receives a broken SSE stream with no error event, causing the frontend to hang indefinitely.
- **Fix**: Create `TestChatStreamErrorHandling` that:
```python
async def test_error_mid_stream_yields_error_event():
    # Mock graph.astream_events to raise RuntimeError after 3 events
    # Assert chat_stream yields an "error" SSE event
    # Assert PII buffer is dropped (not flushed)

async def test_cancelled_error_during_stream():
    # Mock graph.astream_events to raise CancelledError
    # Assert clean termination without error event
```

### Finding 6 (MEDIUM): Specialist Dispatch Tie-Breaking Test Has Weak Assertion

- **Location**: `tests/test_graph_v2.py:455-477` (test_mixed_context_dispatches_to_host)
- **Problem**: The test for mixed-category dispatch with tie only asserts `mock_get.called` (line 477) — it does NOT verify which agent name was passed. The comment explains the tie-breaking logic but the assertion doesn't enforce it. The integration test (`test_mixed_categories_with_tie_dispatches_deterministically`) patches `dining_agent` specifically, which implicitly tests the routing, but the unit test is a rubber stamp.
- **Impact**: If the tie-breaking logic changes (e.g., someone edits `_CATEGORY_PRIORITY`), the unit test will still pass because it only checks that *some* agent was called, not the *correct* agent. The codebase has deterministic tie-breaking (good), but the test doesn't enforce it (bad).
- **Fix**: Change the assertion to:
```python
mock_get.assert_called_once_with("dining")  # restaurants wins tie via _CATEGORY_PRIORITY
```

### Finding 7 (MEDIUM): Unsafe Multi-byte Character Truncation in Persona Envelope

- **Location**: `src/agent/persona.py` (persona_envelope_node)
- **Problem**: SMS truncation at `PERSONA_MAX_CHARS=160` uses Python string slicing. For strings containing multi-byte characters (emojis, CJK characters), slicing at byte index 160 can split a Unicode surrogate pair, producing invalid characters. Tests only cover ASCII truncation (`"A" * 300`).
- **Impact**: Casino guests receiving SMS with emojis or Chinese characters (Mohegan Sun has significant Asian clientele per CT casino demographics) will see garbled text or delivery failures at the carrier level.
- **Fix**: Use grapheme-aware truncation:
```python
import grapheme
truncated = grapheme.slice(content, 0, max_chars - 3) + "..."
```
Add test:
```python
async def test_sms_mode_truncates_emoji_safely(self):
    content = "Hello! " + "\U0001f3b0" * 80  # slot machine emoji
    # Should not split emoji in half
```

### Finding 8 (MEDIUM): Observability Trace Recording During Graph Execution Untested

- **Location**: `src/observability/traces.py`, `src/agent/graph.py:526-535`
- **Problem**: `test_observability.py` tests `TraceContext.to_dict()` and `NodeSpan` dataclass serialization in isolation, but no test verifies that trace spans are actually recorded during graph execution. The `node_start_times` tracking in `chat_stream()` populates timing data, but no test asserts that timing data is correct or that all 11 nodes emit start/end spans.
- **Impact**: Observability blind spots in production — if trace recording silently breaks, operators lose visibility into per-node latency and error rates without any alert.
- **Fix**: Add to `test_observability.py`:
```python
async def test_chat_stream_emits_graph_node_events():
    # Mock graph, collect all yielded SSE events
    # Assert graph_node events with status=start appear for expected nodes
```

### Finding 9 (LOW): conftest.py Singleton Cleanup Uses try/except ImportError Pattern

- **Location**: `tests/conftest.py:33-131`
- **Problem**: The singleton cleanup fixture wraps each cache clear in a separate `try/except ImportError` block (12 separate blocks). If a module import succeeds but the cache attribute was renamed, the `except ImportError` won't catch the `AttributeError`, and the test will fail with a misleading error message. One block already handles this with `except (ImportError, AttributeError)` (line 73), but the rest don't.
- **Impact**: If a developer renames a cache variable (e.g., `_llm_cache` to `_main_llm_cache`), 11 of 12 cleanup blocks will raise `AttributeError` instead of silently handling the mismatch, causing confusing test failures.
- **Fix**: Change all `except ImportError` to `except (ImportError, AttributeError)` for consistency, or better, use a loop:
```python
_CACHE_CLEANUPS = [
    ("src.config", "get_settings", "cache_clear"),
    ("src.agent.nodes", "_llm_cache", "clear"),
    # ...
]
for mod, attr, method in _CACHE_CLEANUPS:
    try:
        m = importlib.import_module(mod)
        getattr(getattr(m, attr), method)()
    except (ImportError, AttributeError):
        pass
```

---

## Summary

**Total Score: 84/100**

The codebase has matured significantly from R3 (60.7). The 10 fixes from R3 (Firestore client caching, atomic CCPA delete, bounded TTLCache, CB health reporting, separate validator locks) are all verified by new tests. Architecture, guardrails, domain intelligence, and production resilience are strong.

The primary gap remains **testing quality vs. testing quantity**. 1107 tests at 90.63% coverage looks impressive, but the most complex and security-critical function in the codebase (`chat_stream` with its PII buffer) is entirely mocked out. The Google Sheets client and HITL interrupt path are untested. Several test assertions are weaker than they should be (checking `called` instead of `called_with`).

### Priority Fix Order
1. **CRITICAL**: Add direct tests for chat_stream PII buffer (Finding 1)
2. **HIGH**: Test whisper_planner_enabled=False topology (Finding 2)
3. **HIGH**: Add sheets_client.py tests (Finding 3)
4. **HIGH**: Test HITL interrupt path (Finding 4)
5. **HIGH**: Test SSE stream error paths (Finding 5)
6. **MEDIUM**: Strengthen tie-breaking test assertion (Finding 6)
7. **MEDIUM**: Fix multi-byte truncation (Finding 7)
8. **MEDIUM**: Test trace recording during graph execution (Finding 8)
9. **LOW**: Standardize conftest.py error handling (Finding 9)
