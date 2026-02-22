# R31 Review: Group A (Graph Architecture, RAG, Data Model, API Design, Testing Strategy)

## Scores

| Dimension | Score | Evidence |
|-----------|-------|----------|
| Graph Architecture | 9/10 | Excellent 11-node StateGraph with proper validation loop, dual-layer routing, node constants, defense-in-depth. New E2E tests verify wiring but lack RETRY/FAIL path coverage. |
| RAG Pipeline | 9/10 | Per-item chunking, SHA-256 dedup, version-stamp purging, property_id isolation all solid. New quality tests cover purging and isolation. Entertainment guide is well-structured. Minor: `os.environ.setdefault` anti-pattern in test_rag_quality.py. |
| Data Model | 9/10 | TypedDict with proper reducers (_merge_dicts, _keep_max), parity check at import time, RetrievedChunk schema. No drift detected between state and _initial_state. |
| API Design | 9/10 | Pure ASGI middleware, proper SSE streaming with PII redaction, heartbeat pattern, separation of liveness/readiness probes. No regressions from Phase 5. |
| Testing Strategy | 8/10 | New E2E tests add real value (verify graph wiring, not just node logic). But 5 E2E tests cover only 3 paths (happy path, greeting, off-topic, guardrail, multi-turn). Missing: RETRY loop, FAIL-to-fallback, circuit breaker open path, whisper planner failure. |
| **Subtotal** | **44/50** | |

## CRITICAL Findings

### C-001: `get_responsible_gaming_helplines()` called without `casino_id` in `_base.py:151`

**File**: `/home/odedbe/projects/hey-seven/src/agent/agents/_base.py:151`
**Severity**: CRITICAL (regulatory compliance)

The `execute_specialist()` function calls `get_responsible_gaming_helplines()` with no arguments:

```python
system_prompt = system_prompt_template.safe_substitute(
    property_name=settings.PROPERTY_NAME,
    current_time=current_time,
    responsible_gaming_helplines=get_responsible_gaming_helplines(),  # <-- no casino_id
    property_description=_property_description,
)
```

This always returns Connecticut helplines regardless of property. A New Jersey property (Hard Rock AC) would show CT helplines instead of the NJ-mandated 1-800-GAMBLER number. This is a direct continuation of the R25-R29 persona hardcoding issue that was supposedly fixed.

**Fix**: Pass `settings.CASINO_ID` to `get_responsible_gaming_helplines()`:
```python
responsible_gaming_helplines=get_responsible_gaming_helplines(settings.CASINO_ID),
```

### C-002: `DEFAULT_CONFIG` import in `persona.py` for fallback branding

**File**: `/home/odedbe/projects/hey-seven/src/agent/persona.py:19` and `:183`
**Severity**: CRITICAL (multi-tenant correctness)

`persona.py` imports `DEFAULT_CONFIG` directly and uses it as a fallback:

```python
from src.casino.config import DEFAULT_CONFIG  # line 19
...
except Exception:
    branding = DEFAULT_CONFIG.get("branding", {})  # line 183
```

Per the project's own rules (CLAUDE.md / langgraph-patterns.md): "Every import of `DEFAULT_CONFIG` for runtime data is a multi-tenant bug." The R29 fix comment on line 177 says "use property-specific profile instead of DEFAULT_CONFIG" but the fallback path still reads DEFAULT_CONFIG directly. If `get_casino_profile()` throws (import error, Firestore timeout), the persona envelope falls back to Mohegan Sun branding for all properties.

**Fix**: The fallback should use a safe empty dict `{}` rather than `DEFAULT_CONFIG`:
```python
except Exception:
    branding = {}  # Safe empty: _enforce_branding defaults handle missing keys
```

## HIGH Findings

### H-001: E2E tests do not cover the RETRY validation loop path

**File**: `/home/odedbe/projects/hey-seven/tests/test_full_graph_e2e.py`
**Severity**: HIGH (test gap)

All 5 E2E tests use `ValidationResult(status="PASS")`. None test the RETRY path (validate -> generate -> validate -> persona_envelope) or the FAIL path (validate -> fallback -> END). These are the two most important graph wiring paths to verify because they involve conditional edges and state mutation (retry_count increment).

The RETRY loop is arguably the single most important architectural feature (all reviewers across 20 rounds praised it). Yet the new E2E tests -- which are explicitly designed to verify "node wiring, conditional edges, and state transitions" -- never exercise it.

**Fix**: Add at least two more E2E tests:
1. `test_retry_then_pass_full_pipeline` -- first validation returns RETRY, second returns PASS
2. `test_fail_to_fallback_full_pipeline` -- validation returns FAIL, verify fallback message

### H-002: E2E tests do not verify circuit breaker open path through full graph

**File**: `/home/odedbe/projects/hey-seven/tests/test_full_graph_e2e.py`
**Severity**: HIGH (test gap)

When the circuit breaker is open, `_dispatch_to_specialist` should return a fallback message with `skip_validation=True`, and the graph should route through validate (auto-PASS) -> persona_envelope -> respond. This path exists but is never E2E tested. All tests use `_make_permissive_cb()` which always allows requests.

### H-003: `os.environ.setdefault("RAG_MIN_RELEVANCE_SCORE", "-100")` at module level in test_rag_quality.py

**File**: `/home/odedbe/projects/hey-seven/tests/test_rag_quality.py:17`
**Severity**: HIGH (test reliability)

Per the project's own rules (langgraph-patterns.md): "`os.environ.setdefault()` at module import time does NOT work reliably in test suites." The file has a proper monkeypatch fixture on line 38-46, but the module-level `setdefault` on line 17 is redundant and unreliable. If another test module that runs first sets `RAG_MIN_RELEVANCE_SCORE` to a different value, `setdefault` is a no-op. The monkeypatch fixture is the correct approach and makes the module-level call unnecessary.

**Fix**: Remove line 17 (`os.environ.setdefault("RAG_MIN_RELEVANCE_SCORE", "-100")`). The `_rag_quality_env` fixture handles this correctly.

## MEDIUM Findings

### M-001: E2E test `test_responsible_gaming_trigger` asserts wrong helpline number

**File**: `/home/odedbe/projects/hey-seven/tests/test_full_graph_e2e.py:259`
**Severity**: MEDIUM (test correctness)

The test asserts:
```python
assert "1-800-522-4700" in response_text or "helpline" in response_text.lower() or "problem gambling" in response_text.lower()
```

But the actual helpline number in `prompts.py:23` is `1-800-MY-RESET (1-800-699-7378)`, not `1-800-522-4700`. The assertion passes because of the `or "problem gambling"` fallback, which matches the off_topic_node's gambling_advice response. This means the assertion is not actually verifying the specific helpline content -- it's succeeding on a generic string match.

**Fix**: Assert for `1-800-699-7378` or `1-800-MY-RESET` instead of `1-800-522-4700`.

### M-002: E2E `test_multi_turn_conversation` does not verify message accumulation

**File**: `/home/odedbe/projects/hey-seven/tests/test_full_graph_e2e.py:264-313`
**Severity**: MEDIUM (test depth)

The multi-turn test verifies that 3 turns return non-empty responses with matching thread_ids, but it does not verify that conversation state actually accumulated. There is no assertion that the graph state contains messages from prior turns. This means the test would pass even if the checkpointer was not working (each turn treated independently).

**Fix**: After turn 3, retrieve the checkpoint state and assert `len(messages) >= 6` (3 human + 3 AI messages minimum).

### M-003: `_base.py` calls `get_casino_profile` via redundant import twice

**File**: `/home/odedbe/projects/hey-seven/src/agent/agents/_base.py:144` and `:194`
**Severity**: MEDIUM (code quality)

`get_casino_profile` is imported inside the function body at line 144 and again at line 194 within the same function. The import at line 144 is already in scope for line 194 since Python caches module-level imports after the first call, but the duplicate `from src.casino.config import get_casino_profile` is confusing. Additionally, both calls construct `settings.CASINO_ID if hasattr(settings, "CASINO_ID") else ""` -- the `hasattr` check is unnecessary since `CASINO_ID` has a default value in Settings.

### M-004: No E2E test for whisper_planner_node failure (fail-silent path)

**File**: `/home/odedbe/projects/hey-seven/tests/test_full_graph_e2e.py`
**Severity**: MEDIUM (test gap)

The whisper planner's fail-silent contract (return `{"whisper_plan": None}` on any error) is never tested end-to-end. If the whisper planner raised and the graph did not handle it, the entire pipeline would crash. A test that mocks `_get_whisper_llm` to raise an exception and verifies the pipeline still returns a valid response would catch wiring bugs.

### M-005: Entertainment guide is Mohegan-Sun-specific with no multi-property annotation

**File**: `/home/odedbe/projects/hey-seven/knowledge-base/casino-operations/entertainment-guide.md`
**Severity**: MEDIUM (multi-tenant)

The entertainment guide references "Mohegan Sun Arena", "Wolf Den", "Comix at Mohegan Sun", and "mohegansun.com/entertainment" without any multi-property annotation or property_id metadata. When this is ingested for other properties (Foxwoods, Hard Rock AC), it will create misleading context chunks. The property_id metadata filter during retrieval prevents cross-property leakage, but only if the ingestion assigns the correct property_id. If a property accidentally ingests this file, the Mohegan-specific content would appear as their own.

## Phase 5 Impact Assessment

### Step 1: Full-graph E2E tests (`test_full_graph_e2e.py`)

**Impact**: +2 points. These tests genuinely improve confidence in graph wiring. They compile the real graph via `build_graph()` and send messages through `chat()`, catching wiring issues that unit tests miss. The greeting test correctly asserts `mock_search.assert_not_called()` to verify the greeting path bypasses RAG. However, the tests only cover 3 of ~6 possible graph paths (missing RETRY, FAIL, CB-open), limiting the wiring coverage to the happy path and two bypass paths.

### Step 2: System prompt parameterization (`test_prompt_parameterization.py`)

**Impact**: +1 point. The parameterization tests are well-designed. They verify that each property profile renders correctly and that no Mohegan-specific content leaks into other property prompts. The test at line 87-93 (`test_prompt_template_no_hardcoded_description`) is particularly valuable -- it inspects the raw template to ensure no hardcoded content exists. However, C-001 (helplines not parameterized) partially negates this improvement.

### Step 3: LLM-as-judge with G-Eval (not in my review group, but noted)

N/A for Group A scoring.

### Step 4: RAG quality tests (`test_rag_quality.py`)

**Impact**: +1 point. The stale chunk purging test (`test_reingest_purges_old_version_chunks`) directly validates a past review finding (ghost data accumulation). The property isolation tests (`test_mohegan_query_excludes_foxwoods`, `test_foxwoods_query_excludes_mohegan`) verify multi-tenant safety end-to-end with real ChromaDB operations. SHA-256 idempotency tests confirm deterministic IDs. The entertainment guide adds useful structured content for entertainment queries.

### Summary: Phase 5 adds approximately +4 points to the Group A dimensions over the R30 baseline.

The main areas holding back a higher score:
- C-001 and C-002 are regressions of previously-fixed multi-tenant issues
- E2E test coverage is good for the happy path but misses the most architecturally significant paths (RETRY loop, FAIL path)
- Test assertions sometimes match too broadly (M-001 helpline number, M-002 message accumulation)
