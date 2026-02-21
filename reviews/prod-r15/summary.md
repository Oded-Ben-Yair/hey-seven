# R15 Fix Summary

**Date**: 2026-02-21
**Reviews**: DeepSeek (86/100), Gemini (80/100), GPT (85/100)
**Consensus threshold**: 2/3+ reviewers

---

## Consensus Findings Identified

| # | Finding | Reviewers | Severity |
|---|---------|-----------|----------|
| 1 | `_get_circuit_breaker()` sync without asyncio.Lock | DeepSeek F-002, Gemini M2, GPT F1 (3/3) | HIGH |
| 2 | `cb` variable potentially unbound in except blocks | DeepSeek F-001, GPT F1 (2/3) | HIGH |
| 3 | Dispatch CB recording parse failures as LLM failures | GPT F1, Gemini F7 (2/3) | MEDIUM |
| 4 | Comp agent completeness wrong denominator | DeepSeek F-006, Gemini F2 (2/3) | MEDIUM |
| 5 | `route_from_router` uses bare strings instead of NODE_* constants | DeepSeek F-004, GPT F3 (2/3) | LOW |
| 6 | Specialist name triple-definition drift risk | GPT F3, Gemini F1/F7 (2/3) | MEDIUM |
| 7 | `assert` in feature_flags.py instead of `raise ValueError` | DeepSeek F-008 + graph.py precedent (1.5/3) | MEDIUM |

---

## Fixes Applied (7 total)

### Fix 1: `_get_circuit_breaker()` async with `asyncio.Lock` (3/3 consensus)

**File**: `src/agent/circuit_breaker.py`

Converted `_get_circuit_breaker()` from sync to `async def` with `asyncio.Lock` protection, matching the established pattern used by `_get_llm()`, `_get_validator_llm()`, and `_get_whisper_llm()`. Added `_cb_lock = asyncio.Lock()` module-level variable.

While the function body is fully synchronous (no yield points, so the race cannot happen in a single-threaded asyncio event loop), the lock ensures pattern consistency and future-proofs against async operations being added to the factory.

**Downstream updates**:
- `src/agent/agents/_base.py`: `cb = get_cb_fn()` -> `cb = await get_cb_fn()`
- `src/api/app.py`: `cb = _get_circuit_breaker()` -> `cb = await _get_circuit_breaker()`
- `tests/test_base_specialist.py`: 13 instances of `MagicMock(return_value=mock_cb)` -> `AsyncMock(return_value=mock_cb)` for `get_cb_fn`
- `tests/test_r5_scalability.py`: 3 sync test functions -> async with `pytest.mark.asyncio`
- `tests/test_nodes.py`: 1 sync test function -> async with `pytest.mark.asyncio`

### Fix 2: `cb` variable moved before try block (2/3 consensus)

**File**: `src/agent/graph.py`

Moved `cb = await _get_circuit_breaker()` before the `try` block in `_dispatch_to_specialist()`. Previously, if `_get_circuit_breaker()` raised during TTL cache refresh, the `except` handlers would crash with `UnboundLocalError` when referencing `cb`, converting a recoverable error into an unhandled exception.

### Fix 3: Dispatch parse failures no longer record CB failure (2/3 consensus)

**File**: `src/agent/graph.py`

In the `(ValueError, TypeError)` handler of `_dispatch_to_specialist()`, removed `await cb.record_failure()`. Parse failures mean the LLM IS reachable but returned bad JSON -- this is a prompt engineering issue, not an LLM availability problem. Recording parse failures inflated the CB failure count, potentially tripping the breaker on prompt issues rather than actual outages.

Network/API failures (the broad `except Exception` handler) continue to record CB failures correctly.

### Fix 4: Comp agent completeness denominator aligned with whisper planner (2/3 consensus)

**File**: `src/agent/agents/comp_agent.py`

Changed the profile completeness calculation from `filled / max(len(extracted_fields), 1)` to `filled / len(_PROFILE_FIELDS)`, importing `_PROFILE_FIELDS` from `whisper_planner.py`. The old formula had a semantic bug: with 1 extracted field, completeness was `1/1 = 100%`, bypassing the profile gate immediately.

Now both `comp_agent` and `whisper_planner._calculate_completeness()` use the same 8-field denominator for consistent behavior.

**Test updates**: `tests/test_phase2_integration.py` -- 2 tests updated to use flat `_PROFILE_FIELDS` keys instead of nested GuestProfile structure, matching the field format the completeness gate actually checks.

### Fix 5: `route_from_router` uses NODE_* constants (2/3 consensus)

**File**: `src/agent/nodes.py`

Replaced bare string returns (`"greeting"`, `"off_topic"`, `"retrieve"`) with local constants (`_NODE_GREETING`, `_NODE_OFF_TOPIC`, `_NODE_RETRIEVE`). Defined locally to avoid circular imports (graph.py imports from nodes.py). Cross-referenced with graph.py NODE_* constants in a comment.

### Fix 6: Specialist dispatch validation uses `_AGENT_REGISTRY` (2/3 consensus)

**File**: `src/agent/graph.py`

Replaced hardcoded specialist name set `{"dining", "entertainment", "comp", "hotel", "host"}` at the structured output validation point with a reference to `_AGENT_REGISTRY` from `registry.py`. This eliminates one of the three places where specialist names were defined, reducing the drift risk when new specialists are added.

Remaining definitions:
1. `DispatchOutput.specialist` Literal type (state.py) -- Pydantic constraint, harder to parameterize
2. `_AGENT_REGISTRY` (registry.py) -- single source of truth, now used by dispatch validation

### Fix 7: `assert` -> `raise ValueError` in feature_flags.py (DeepSeek F-008 + precedent)

**File**: `src/casino/feature_flags.py`

Converted both `assert` statements (FeatureFlags TypedDict parity and DEFAULT_CONFIG parity) to `if/raise ValueError`, matching the pattern established in `graph.py:498-505` during R10 (which was the same fix for the same reason). Asserts vanish with `python -O`, silently removing schema drift detection.

---

## Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `src/agent/circuit_breaker.py` | Modified | Added `_cb_lock`, made `_get_circuit_breaker` async |
| `src/agent/graph.py` | Modified | `cb` before try, removed parse-failure CB recording, imported `_AGENT_REGISTRY` |
| `src/agent/agents/_base.py` | Modified | `await get_cb_fn()` |
| `src/agent/agents/comp_agent.py` | Modified | Completeness uses `_PROFILE_FIELDS` denominator |
| `src/agent/nodes.py` | Modified | Local NODE_* constants for `route_from_router` |
| `src/api/app.py` | Modified | `await _get_circuit_breaker()` |
| `src/casino/feature_flags.py` | Modified | `assert` -> `raise ValueError` |
| `tests/test_base_specialist.py` | Modified | `MagicMock` -> `AsyncMock` for `get_cb_fn` (13 instances) |
| `tests/test_r5_scalability.py` | Modified | 3 sync CB factory tests -> async |
| `tests/test_nodes.py` | Modified | 1 sync CB clear test -> async |
| `tests/test_phase2_integration.py` | Modified | 2 comp agent tests: nested -> flat extracted_fields |

---

## Test Results

```
1452 passed, 20 skipped, 1 warning in 38.72s
Coverage: 90.14% (required: 90.0%)
```

All 1452 tests pass. No regressions.

The 1 warning (`coroutine 'AsyncMockMixin._execute_mock_call' was never awaited`) is pre-existing from `test_integration.py` fixture teardown, not caused by R15 changes.

---

## Non-Consensus Findings (Not Fixed)

| Finding | Reviewer(s) | Reason Not Fixed |
|---------|-------------|------------------|
| Dispatch LLM shares main LLM (temp/tokens mismatch) | Gemini M1 only (1/3) | Below consensus threshold |
| `BoundedMemorySaver` protocol compliance | DeepSeek F-005, Gemini F3 (2/3 but LOW/MOD) | Acknowledged but BoundedMemorySaver is dev-only ("NOT for production" per memory.py:44). Lower priority. |
| Validate-generate retry loop does not re-run whisper planner | GPT F2 only (1/3) | Documented trade-off, no code change needed |
| Whisper planner `_failure_count` race | DeepSeek F-003, Gemini M3 (2/3) | Conftest already clears these globals (confirmed at conftest.py:57-63). The race is documented as "benign" and accepted for alerting purposes. |
| Missing SSE `chat_stream` E2E test | GPT F4 only (1/3) | Below consensus threshold |
