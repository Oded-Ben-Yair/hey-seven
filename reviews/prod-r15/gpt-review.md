# Hey Seven Production Review -- R15 (GPT Focus)

**Reviewer**: GPT-calibrated Senior Engineer (simulated by Opus 4.6)
**Commit**: c7b986e
**Focus**: Graph topology correctness, state machine guarantees, integration test quality
**Spotlight**: Graph Architecture (Dimension 1, +1 severity)
**Date**: 2026-02-21

---

## Dimension Scores

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | **Graph Architecture** (SPOTLIGHT) | 9 | Exemplary 11-node StateGraph with formal BFS verification. Two substantive findings below. |
| 2 | RAG Pipeline | 8 | Per-item chunking, RRF reranking, SHA-256 idempotent IDs -- strong. Not the focus of this review. |
| 3 | Data Model | 9 | PropertyQAState with `_keep_max` reducer, parity assertion at import time, `RetrievedChunk` typed. |
| 4 | API Design | 8 | Pure ASGI middleware, SSE heartbeats, per-request PII redactor. One finding on /sms/webhook. |
| 5 | Testing Strategy | 8 | BFS topology tests, E2E wiring tests, 1452 tests. One gap identified. |
| 6 | Docker & DevOps | 8 | Cloud Run probes correctly separated (/live vs /health). Not primary focus. |
| 7 | Prompts & Guardrails | 9 | 5-layer deterministic guardrails + semantic LLM classifier. 84 compiled patterns across 4 languages. |
| 8 | Scalability & Production | 8 | TTL-cached singletons, BoundedMemorySaver, LLM semaphore. One finding on distributed state. |
| 9 | Trade-off Documentation | 9 | Dual-layer feature flag architecture documented inline. Degraded-pass rationale explicit. |
| 10 | Domain Intelligence | 9 | Responsible gaming escalation counter, BSA/AML refusal, patron privacy, 4-language coverage. |

**Overall Score: 85/100**

---

## Findings

### Finding 1 -- MAJOR (Spotlight +1 = CRITICAL)

**Dimension**: 1 (Graph Architecture)
**Title**: `_dispatch_to_specialist` uses a separate circuit breaker instance from the specialist agent it dispatches to
**Severity**: CRITICAL (elevated from MAJOR due to spotlight)

**Location**: `src/agent/graph.py:204-271`

**Issue**: `_dispatch_to_specialist` acquires the circuit breaker via `_get_circuit_breaker()` (line 204) and calls `allow_request()` on it (line 205) to decide whether to attempt LLM-based structured dispatch. If the LLM call fails, it calls `record_failure()` on this same CB (lines 241, 247). However, the specialist agent it then dispatches to (line 271) has its *own* call to `get_cb_fn()` inside `execute_specialist` (line 92 of `_base.py`), which returns the same singleton CB instance.

The problem is the double `allow_request()` call. The dispatch function calls `cb.allow_request()` at line 205 for structured routing. Then `execute_specialist` calls `cb.allow_request()` again at line 97 of `_base.py`. In half-open state, the first call consumes the single probe slot (`_half_open_in_progress = True`), and the second call returns `False`, causing the specialist to return fallback even though the dispatch succeeded.

**Scenario**:
1. CB enters half-open state (cooldown expired after being open)
2. `_dispatch_to_specialist` calls `allow_request()` -- returns `True`, sets `_half_open_in_progress = True`
3. Structured dispatch LLM call succeeds, calls `record_success()` -- resets to `closed`
4. `execute_specialist` calls `allow_request()` -- returns `True` (now closed)

Actually, upon careful re-examination: step 3's `record_success()` resets the state to `closed`, so step 4 will succeed. The double-call is wasteful but not buggy in the success path. However, there is still a timing issue:

**Revised scenario (failure path)**:
1. CB in closed state, 4 prior failures
2. `_dispatch_to_specialist` calls `allow_request()` -- True (closed)
3. Structured dispatch LLM call fails, calls `record_failure()` -- 5th failure, CB transitions to OPEN
4. Keyword fallback selects "dining", dispatches to dining_agent
5. `execute_specialist` calls `cb.allow_request()` -- returns False (CB just opened)
6. Guest gets generic fallback instead of a real response

This means a structured dispatch failure causes the specialist agent to also fail, even though the specialist uses a separate LLM call path that might succeed. The dispatch and generation are logically independent LLM calls, but share a single circuit breaker.

**Fix**: Either (a) do not record dispatch LLM failures on the shared CB (use a separate dispatch-only counter), or (b) skip the CB check in `execute_specialist` when the dispatch already consumed a probe. The cleanest fix is option (a): create a separate dispatch CB or do not count dispatch failures toward the generation CB threshold.

---

### Finding 2 -- MAJOR (Spotlight +1 = HIGH)

**Dimension**: 1 (Graph Architecture)
**Title**: Validate-generate retry loop does not re-run the whisper planner
**Severity**: HIGH (elevated from MEDIUM due to spotlight)

**Location**: `src/agent/graph.py:426-433`

**Issue**: The edge map defines `validate -> generate (RETRY)` as a direct edge back to the generate node, skipping `whisper_planner`. On the initial path, the chain is `retrieve -> whisper_planner -> generate -> validate`. On retry, the chain is `validate -> generate -> validate` (skipping whisper_planner entirely).

This is likely intentional (the whisper plan does not change between retries since the conversation history has not changed), but the whisper_plan state field persists from the first pass, so the specialist agent on retry still receives the original whisper guidance. This is acceptable behavior.

However, the retry path also means the retrieval context is not refreshed. If validation fails because the response was not grounded in the retrieved context, the retry uses the same context and the same whisper plan, making it likely to produce a similar response. The max-1-retry design limits the damage, but the retry is unlikely to produce a meaningfully different response without fresh retrieval.

**Impact**: The retry loop may be wasted computation -- the same inputs (context + whisper + conversation) produce a similar output. The only difference is the `retry_feedback` injection, which may or may not cause the LLM to correct the specific issue.

**Fix**: This is a documented trade-off (max 1 retry limits cost). No code change required, but document the known limitation that retries are feedback-guided only (same context, same plan). Consider logging retry success rate in production to measure actual retry effectiveness.

---

### Finding 3 -- MEDIUM

**Dimension**: 1 (Graph Architecture)
**Title**: `_dispatch_to_specialist` does not validate structured output `specialist` field against registry before use
**Severity**: MEDIUM

**Location**: `src/agent/graph.py:226`

**Issue**: Line 226 checks `result.specialist in {"dining", "entertainment", "comp", "hotel", "host"}`, which is a hardcoded set that must stay in sync with `_AGENT_REGISTRY` in `registry.py`. If a new specialist is added to the registry but not to this inline set, structured dispatch will silently fall back to keyword counting for that specialist.

The Pydantic `DispatchOutput.specialist` Literal type (state.py:99) also has this same list. That makes three places where the specialist names are defined:
1. `DispatchOutput.specialist` Literal type (state.py:99)
2. Inline validation set (graph.py:226)
3. `_AGENT_REGISTRY` dict (registry.py:15-21)

**Fix**: Replace the hardcoded set at graph.py:226 with a reference to the registry:

```python
from .agents.registry import _AGENT_REGISTRY
# ...
if result.specialist in _AGENT_REGISTRY:
```

Or better, add a `list_agents()` call and compare against it. The Pydantic Literal type is harder to parameterize but could be generated from the registry at module level.

---

### Finding 4 -- MEDIUM

**Dimension**: 5 (Testing Strategy)
**Title**: No test for the SSE `chat_stream` path through the full graph
**Severity**: MEDIUM

**Location**: `tests/test_e2e_pipeline.py`

**Issue**: The E2E pipeline tests (`test_e2e_pipeline.py`) exercise the `graph.ainvoke()` path (via `chat()` helper) but do not test the `chat_stream()` path. `chat_stream()` has significantly different logic: SSE event generation, PII streaming redaction, node lifecycle tracking, and error handling. The streaming path is the primary production path (the `/chat` endpoint uses it exclusively).

The test file tests `chat()` and direct `graph.ainvoke()`, but never calls `chat_stream()`. Key streaming-specific logic that is untested in E2E:
- `StreamingPIIRedactor` integration with the graph event stream
- `graph_node` lifecycle events (start/complete with duration_ms)
- `_pii_redactor.flush()` at end of stream
- CancelledError handling (dropping buffered PII)
- `replace` event generation for non-streaming nodes (greeting, off_topic)

**Fix**: Add at least one E2E test that calls `chat_stream()`, collects all yielded events, and asserts:
1. `metadata` event is first
2. `token` or `replace` events contain expected content
3. `done` event is last
4. No `error` event on happy path

---

### Finding 5 -- MEDIUM

**Dimension**: 4 (API Design)
**Title**: `/sms/webhook` does not route agent responses back through SMS
**Severity**: MEDIUM

**Location**: `src/api/app.py:447-451`

**Issue**: The SMS webhook handler processes inbound messages and returns a JSON response to the webhook caller, but for non-keyword messages (the "regular message" path, lines 447-451), it only returns `{"status": "received"}` without routing the message through the agent graph or sending an SMS reply. The comment says "Phase 2.4 will handle full routing" but there is no tracking ticket or TODO comment with a ticket ID.

Per the project's own Placeholder Response Tracking rule: "Never ship placeholder API responses without a tracking ticket with a due date."

**Fix**: Add a TODO comment with a tracking ticket ID, or return HTTP 501 (Not Implemented) for the unimplemented path. The current `{"status": "received"}` silently drops guest messages in production.

---

### Finding 6 -- LOW

**Dimension**: 8 (Scalability & Production)
**Title**: `RateLimitMiddleware._request_counter` initialized via `getattr` on first use
**Severity**: LOW

**Location**: `src/api/middleware.py:369`

**Issue**: The `_request_counter` attribute is initialized lazily via `getattr(self, "_request_counter", 0)` inside the async lock. While this works, it is a code smell -- the attribute should be initialized in `__init__` alongside other instance attributes. The current pattern means a reader of `__init__` will not see all instance state.

**Fix**: Add `self._request_counter = 0` to `__init__` (line 322).

---

### Finding 7 -- LOW

**Dimension**: 3 (Data Model)
**Title**: `responsible_gaming_count` missing from `_state()` helper in test_graph_v2.py
**Severity**: LOW

**Location**: `tests/test_graph_v2.py:36-51`

**Issue**: The `_state()` helper in `test_graph_v2.py` builds a minimal `PropertyQAState` dict but omits the `responsible_gaming_count` field. While this does not cause test failures (TypedDict does not enforce at runtime, and the graph uses `.get()` with defaults), it means test states are structurally incomplete. The production `_initial_state()` in `graph.py` includes it (line 490), and the parity check (lines 498-505) would catch drift there -- but test helpers are not covered by that check.

**Fix**: Add `"responsible_gaming_count": 0` to the `_state()` helper's base dict to maintain parity with the production initial state.

---

### Finding 8 -- LOW

**Dimension**: 7 (Prompts & Guardrails)
**Title**: Guardrail normalization strips combining marks, potentially mangling non-Latin scripts
**Severity**: LOW

**Location**: `src/agent/guardrails.py:201-206`

**Issue**: The `_normalize_input()` function applies NFKD decomposition and then removes all combining marks (diacritics). While this is useful for catching homoglyph attacks in Latin scripts, it destructively mangles text in scripts that rely on combining marks (e.g., some Arabic diacritics, Korean jamo, Hebrew niqqud). Since the codebase already has responsible gaming patterns in Mandarin and Portuguese, and BSA/AML patterns in multiple languages, the normalization step could reduce pattern matching accuracy for those languages.

The mitigating factor is that the raw text is checked first (line 257-258), and normalization is only applied as a second pass if the text differs after normalization (lines 261-263). So non-Latin scripts that do not use combining marks are unaffected. But scripts with optional diacritics (Arabic, Hebrew) may have reduced pattern matching after normalization.

**Fix**: This is an accepted risk given the dual-pass approach. Consider documenting it as a known limitation for Arabic/Hebrew patterns that include diacritical marks.

---

## Summary

The codebase is production-grade with strong architectural patterns. The 11-node StateGraph is formally verified (BFS reachability, stuck-state detection, cycle enumeration), which is uncommon and praiseworthy. The dual-layer feature flag architecture (build-time topology vs runtime behavior) is well-documented and correctly implemented.

**Strengths**:
- Formal BFS topology verification catches wiring bugs at test time
- Import-time parity assertion between `PropertyQAState` and `_initial_state()` prevents state schema drift
- Degraded-pass validation strategy is correctly implemented with principled rationale
- 15+ singleton caches cleaned in conftest.py with consistent `(ImportError, AttributeError)` handling
- Streaming PII redaction with lookahead buffer is a thoughtful production detail
- Pure ASGI middleware preserves SSE streaming (BaseHTTPMiddleware lesson learned)

**Primary concern**: The shared circuit breaker between structured dispatch and specialist generation (Finding 1) creates a coupling where a dispatch LLM failure can cascade to block the specialist agent, even though they are logically independent operations. This is the highest-priority fix.

**Score trajectory**: R11(82) -> R12(84) -> R13(85) -> R14(86) -> R15(85). The score plateau at 85-86 suggests the codebase has reached diminishing returns on review-driven improvements. The remaining findings are increasingly marginal.
