# R76 Learning Log — 30-Dimension Perfection Sprint

## Round 1: Baseline + Bug Discovery

### Date: 2026-03-01

### Critical Bugs Found (from live eval logs)

1. **CRITICAL: ProfileExtractionOutput schema overflow** (P0)
   - Symptom: 400 INVALID_ARGUMENT on every profiling extraction attempt
   - Root cause: 19 nested ConfidenceField objects (each with value:Any + confidence:float[0,1] + source:Literal) produced "too many schema states" for Gemini Flash
   - Fix: Flattened to 16 `str | None` fields. Confidence gating moved to extraction prompt.
   - Impact: P1-P10 all zero → all should now work

2. **MAJOR: DispatchOutput.reasoning max_length=200** (P1)
   - Symptom: Structured dispatch parsing failed every turn, fell back to keyword dispatch
   - Root cause: Gemini reasoning consistently exceeds 200 chars
   - Fix: Increased max_length from 200 → 500
   - Impact: Specialist dispatch quality degraded → now uses LLM classification

3. **MAJOR: Router overwrites grief sentiment** (P1)
   - Symptom: Grief messages ("My dad passed") get generic "explore our rewards!" responses
   - Root cause: Compliance gate sets guest_sentiment="grief" at position 7.6, but router_node runs VADER and overwrites it with "neutral"
   - Fix: Added _PRIORITY_SENTIMENTS guard — don't overwrite grief with VADER
   - Impact: B5 emotional handling completely broken for grief → should now work

4. **MINOR: ValidationResult.reason required** (P2)
   - Symptom: Gemini omits reason field for PASS → validation parsing fails → degraded-pass
   - Root cause: reason field was required, Gemini occasionally omits it
   - Fix: Added default="" to ValidationResult.reason
   - Impact: Unnecessary degraded-pass bypassing validation

### Patterns Identified

- **Gemini Flash has strict schema complexity limits**: Any nested Pydantic model with Literal + float constraints produces "too many states". Must use flat schemas.
- **Gemini Flash omits optional-feeling fields**: When it thinks a field is unnecessary (reason for PASS), it omits it. Always add defaults.
- **LangGraph state overwrites without reducers**: Nodes that set the same field as earlier nodes will overwrite. Use priority guards or reducers for cross-node state.
- **Keyword dispatch is worse than structured dispatch**: Keyword fallback misroutes entertainment→dining when categories overlap. Structured dispatch with LLM is much more accurate.

### Files Modified
- `src/agent/profiling.py` — Flattened ProfileExtractionOutput, removed 3 low-value fields
- `src/agent/state.py` — DispatchOutput.reasoning max_length 200→500, ValidationResult.reason default=""
- `src/agent/nodes.py` — Grief sentiment priority guard in router_node
- `tests/test_profiling.py` — Updated 90 tests for flat schema
- `tests/test_phase5_profiling.py` — Updated 12 tests for flat schema

### Test Count: 3236 passed, 1 skipped, 0 failures (maintained)
