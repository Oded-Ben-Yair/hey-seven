# R43 Post-Refactor Review + Conservative Calibration

**Date**: 2026-02-23
**Reviewer**: Post-refactor hostile reviewer (Opus 4.6)
**Cross-validator**: GPT-5.2 Codex (azure_code_review)
**Prior calibration**: R42 conservative baseline = 93.6/100
**Focus**: Validate dispatch SRP refactor, final 10-dimension calibration

---

## Part 1: SRP Refactor Validation

### Structure Assessment

| Function | LOC (AST) | Responsibility | SRP? |
|----------|-----------|---------------|------|
| `_route_to_specialist()` | 113 | LLM dispatch + keyword fallback + retry reuse + feature flag | **Partial** |
| `_inject_guest_context()` | 32 | Guest profile lookup (fail-silent) | **Yes** |
| `_execute_specialist()` | 74 | Agent execution + timeout + result sanitization | **Yes** |
| `_dispatch_to_specialist()` | 37 | Thin orchestrator | **Yes** |
| **Total refactored area** | 256 | | |

**Before**: 1 function, ~195 LOC, 5 responsibilities mixed together.
**After**: 4 functions, clear separation with well-defined interfaces.

The orchestrator (`_dispatch_to_specialist`) is now 37 lines and reads as a clean 3-step pipeline: route, inject context, execute. This is a significant improvement.

### ADR Quality

`docs/adr/ADR-0001-dispatch-srp-refactor.md` follows a reasonable ADR format (Problem, Decision, Constraints, Consequences, Alternatives Considered). The LOC estimates in the ADR are slightly off vs actual AST measurement (`_route_to_specialist` is 113 LOC, not ~65; `_execute_specialist` is 74 LOC, not ~55), but this is cosmetic.

### Behavioral Parity

**FINDING 1 (MINOR): Feature flag bypass on retry path**

The refactor introduced an early `return` in `_route_to_specialist()` at line 227:

```python
existing_specialist = state.get("specialist_name")
if existing_specialist and existing_specialist in _AGENT_REGISTRY:
    return existing_specialist, "retry_reuse"
```

In the **original** code, the retry path set `agent_name` but did NOT return early — the `specialist_agents_enabled` feature flag check on line 305-308 still ran afterward. In the **refactored** code, the early return bypasses that check.

**Impact**: Low in practice. The retry path only activates when `specialist_name` is already set (from a previous dispatch in the same turn). If `specialist_agents_enabled` was True during the first dispatch (which set `specialist_name`), it's extremely unlikely to flip to False during the 50ms retry window. However, it IS a behavioral change, violating the ADR's "behavioral parity required" constraint.

**Verdict**: MINOR — not a production risk, but the ADR's claim of "no functional changes" is technically incorrect.

**FINDING 2 (NOT A BUG): dispatch_method "unused"**

GPT-5.2 flagged `dispatch_method` as unused. This is **incorrect** — the orchestrator logs it at line 454-458:
```python
logger.info("Dispatching to %s agent (method=%s)", agent_name, dispatch_method)
```
The value flows correctly: `_route_to_specialist` returns it, `_dispatch_to_specialist` destructures it, and the log statement uses it. No issue.

### Test Coverage of New Helpers

**FINDING 3 (MINOR): No dedicated unit tests for extracted helpers**

The 3 new functions (`_route_to_specialist`, `_inject_guest_context`, `_execute_specialist`) are tested **indirectly** through ~25 existing `_dispatch_to_specialist` tests in `test_graph_v2.py`. No git diff in tests/ — zero new test functions added.

This is acceptable for a pure refactor (behavior unchanged, same code paths exercised), but the purpose of SRP extraction is to enable **independent** testing. Future rounds should add targeted tests for:
- `_route_to_specialist` with mocked CB states (open, half-open, closed)
- `_inject_guest_context` with various extracted_fields shapes
- `_execute_specialist` with timeout edge cases

### Error Handling Preservation

All error handling paths verified preserved:
- CB acquisition before try block (R15 fix) — preserved in `_route_to_specialist` L234
- ValueError/TypeError non-CB-failure (R15/Gemini F7) — preserved in L279-286
- Broad Exception for network errors + CB failure recording — preserved in L287-292
- Guest profile fail-silent — preserved in `_inject_guest_context` L342-344
- Specialist timeout fallback (R34 fix A1) — preserved in `_execute_specialist` L374-389
- Dispatch-owned key collision logging (R33 fix) — preserved in L394-400
- Unknown state key filtering (R37 fix M-001) — preserved in L402-412
- Specialist name persistence for retry (R37 fix C-001) — preserved in L415

### Refactor Quality Verdict

| Criterion | Pass? | Notes |
|-----------|:-----:|-------|
| SRP applied | **YES** | 3 helpers + 1 orchestrator, each with clear responsibility |
| Behavioral parity | **PARTIAL** | Feature flag bypass on retry (Finding 1) |
| Error handling preserved | **YES** | All 8 error paths verified |
| Tests pass | **YES** | 2169 passed, 0 failures |
| New tests added | **NO** | Acceptable for refactor, but missed opportunity |
| ADR written | **YES** | `docs/adr/ADR-0001-dispatch-srp-refactor.md` |
| Documentation updated | **YES** | Module docstring updated with 3-helper description |

**Overall refactor grade: B+** — Well-executed structurally, but the behavioral regression (Finding 1) and missing dedicated tests prevent an A.

---

## Part 2: GPT-5.2 Codex Cross-Validation Summary

10 findings were raised. Assessment:

| # | Finding | Severity | Verdict |
|---|---------|----------|---------|
| 1 | Feature flag bypass on retry | MINOR | **Valid** — behavioral regression (see Finding 1 above) |
| 2 | dispatch_method unused | — | **Invalid** — used in orchestrator logger.info |
| 3 | CB doesn't record bad output | INFO | **Pre-existing** design decision (R15 fix Gemini F7) — parse errors != availability |
| 4 | CB open state silent | INFO | **Pre-existing** — line 278 has logger.info for this case |
| 5 | Exception swallowing | INFO | **Partial** — _route_to_specialist has logging (L286, L292); _inject_guest_context has L343. GPT reviewed simplified version, not actual code |
| 6 | get_agent throws on unknown | INFO | **Pre-existing** — `get_agent` has its own KeyError handling in registry.py |
| 7 | Result filtering drops keys | INFO | **Pre-existing** design decision (R37 fix M-001) — intentional |
| 8 | Collision handling only logs | INFO | **Pre-existing** — dispatch-owned keys are overwritten by L415-419 |
| 9 | asyncio.timeout 3.11+ | — | **Non-issue** — project targets Python 3.12+ (Dockerfile base image) |
| 10 | Prompt category duplicates | INFO | **Pre-existing** — duplicates in prompt don't meaningfully affect dispatch |

**Only Finding 1 is a genuine regression from the refactor.** All other findings are either pre-existing design decisions or incorrect.

---

## Part 3: Conservative 10-Dimension Calibration

### Methodology
- Start from R42 conservative calibration (93.6/100)
- Apply R43 refactor impact to D1 only (only graph.py changed)
- Apply R43 ADR to D9 (new ADR directory)
- All other dimensions UNCHANGED from R42

### D1: Graph Architecture (Weight: 0.20) — R42 conservative: 8.7

**R43 changes:**
- (+) SRP refactor resolves the #1 most-flagged issue carried since R34
- (+) `_dispatch_to_specialist` is now 37 LOC (was ~195 LOC)
- (+) Each helper has clear single responsibility with typed interfaces
- (+) Docstrings with Args/Returns documentation on all 3 helpers
- (-) Feature flag bypass on retry (Finding 1) — minor regression
- (-) `_route_to_specialist` is still 113 LOC — large for a "routing" function
- (-) No dedicated unit tests for extracted helpers

The SRP refactor was the single biggest D1 gap identified in R42 calibration ("+0.3 to +0.5 weighted"). The refactor addresses it substantially but not perfectly:
- The orchestrator is exemplary (37 LOC, clean pipeline)
- `_inject_guest_context` is clean (32 LOC, single responsibility)
- `_execute_specialist` is solid (74 LOC, clear boundaries)
- `_route_to_specialist` at 113 LOC still mixes LLM dispatch + keyword fallback + retry reuse + feature flag — it's better than before (all 5 concerns in one function) but could benefit from further extraction of the LLM dispatch try/except block into its own helper

**D1 post-refactor: 9.0** (was 8.7). The SRP fix earns +0.3, not the full +0.5 ceiling, because of Finding 1 and the 113 LOC routing function.

### D9: Trade-off Documentation (Weight: 0.05) — R42 conservative: 8.3

**R43 changes:**
- (+) Created `docs/adr/` directory — resolves the #1 D9 gap from R42
- (+) ADR-0001 follows proper format with Problem/Decision/Constraints/Consequences/Alternatives

**D9 post-refactor: 8.5** (was 8.3). Formal ADR directory earns +0.2. Still missing: placeholder URL fix, capacity planning doc, SLA/SLO definitions.

### All Other Dimensions — UNCHANGED

No code changes outside graph.py. All R42 calibrated scores stand:

| Dimension | Score | Reason unchanged |
|-----------|:-----:|-----------------|
| D2: RAG Pipeline | 8.5 | No RAG changes |
| D3: Data Model | 8.5 | state.py unchanged |
| D4: API Design | 8.5 | app.py unchanged |
| D5: Testing Strategy | 8.3 | No test changes (tests/ unchanged) |
| D6: Docker & DevOps | 8.3 | Dockerfile/cloudbuild unchanged |
| D7: Prompts & Guardrails | 9.0 | guardrails.py unchanged |
| D8: Scalability & Prod | 8.3 | No scalability changes |
| D10: Domain Intelligence | 8.5 | casino config unchanged |

---

## Final Score Card

| Dimension | Weight | R42 Score | R43 Score | Delta | Reasoning |
|-----------|--------|:---------:|:---------:|:-----:|-----------|
| D1: Graph Architecture | 0.20 | 8.7 | **9.0** | +0.3 | SRP refactor resolves 9-round carried issue |
| D2: RAG Pipeline | 0.10 | 8.5 | **8.5** | 0.0 | Unchanged |
| D3: Data Model | 0.10 | 8.5 | **8.5** | 0.0 | Unchanged |
| D4: API Design | 0.10 | 8.5 | **8.5** | 0.0 | Unchanged |
| D5: Testing Strategy | 0.10 | 8.3 | **8.3** | 0.0 | No new tests added |
| D6: Docker & DevOps | 0.10 | 8.3 | **8.3** | 0.0 | Unchanged |
| D7: Prompts & Guardrails | 0.10 | 9.0 | **9.0** | 0.0 | Unchanged |
| D8: Scalability & Prod | 0.15 | 8.3 | **8.3** | 0.0 | Unchanged |
| D9: Trade-off Docs | 0.05 | 8.3 | **8.5** | +0.2 | Formal ADR directory |
| D10: Domain Intelligence | 0.10 | 8.5 | **8.5** | 0.0 | Unchanged |

### Weighted Total

| Dimension | Weight | Score | Weighted |
|-----------|--------|------:|--------:|
| D1 | 0.20 | 9.0 | 1.800 |
| D2 | 0.10 | 8.5 | 0.850 |
| D3 | 0.10 | 8.5 | 0.850 |
| D4 | 0.10 | 8.5 | 0.850 |
| D5 | 0.10 | 8.3 | 0.830 |
| D6 | 0.10 | 8.3 | 0.830 |
| D7 | 0.10 | 9.0 | 0.900 |
| D8 | 0.15 | 8.3 | 1.245 |
| D9 | 0.05 | 8.5 | 0.425 |
| D10 | 0.10 | 8.5 | 0.850 |
| **Total** | **1.00** | | **9.430 -> 94.3/100** |

---

## Remaining Findings (For Future Rounds)

| # | Finding | Dimension | Severity | Fix Effort |
|---|---------|-----------|----------|-----------|
| 1 | Feature flag bypass on retry path in `_route_to_specialist` | D1 | MINOR | LOW — move specialist_agents_enabled check to `_dispatch_to_specialist` after routing |
| 2 | No dedicated unit tests for 3 extracted helpers | D5 | MINOR | MEDIUM — add ~10 focused tests |
| 3 | `_route_to_specialist` at 113 LOC could be further decomposed | D1 | INFO | MEDIUM — extract LLM dispatch try/except into `_try_structured_dispatch()` |
| 4 | ADR-0001 LOC estimates slightly off vs actual AST | D9 | TRIVIAL | LOW — update numbers |

---

## Score Trajectory (Updated)

| Round | Claimed | Calibrated | Delta | Notes |
|-------|:-------:|:----------:|:-----:|-------|
| R34 | 77.0 | 77.0 | — | Baseline |
| R40 | 93.5 | 92.25 | -1.25 | D5/D8 inflation corrected |
| R41 | 94.9 | 93.6 | -1.3 | D1/D6/D9 fix deltas too aggressive |
| R42 | — | 93.6 | — | Calibration-only round (no code changes) |
| **R43** | — | **94.3** | — | **SRP refactor D1 +0.3, ADR D9 +0.2** |

**Net R43 improvement: +0.7 points** (D1 +0.6 weighted, D9 +0.01 weighted).
