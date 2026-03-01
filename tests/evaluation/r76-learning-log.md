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

## Round 2: WhisperPlan + Validator Leniency

### Fixes Applied
1. **WhisperPlan schema simplified** — 10 fields with 3 Literal types + 4 bounded floats → 6 flat str fields. Was 100% dead (systematic failure after 10 consecutive errors). Now should work.
2. **Validation prompt made lenient for cross-domain suggestions** — "After dinner, check out the show" no longer triggers FAIL for grounding violation. This was causing excessive fallback responses.
3. **suggestion_confidence type conversion** — float→str with try/except fallback in _should_inject_suggestion()

### Pattern: Gemini Flash rejects ALL schemas with >5 bounded/Literal fields
- ProfileExtractionOutput: 19 nested ConfidenceField → 100% failure
- WhisperPlan: 10 fields with 3 Literal + 4 bounded float → 100% failure
- DispatchOutput: 3 fields (1 Literal, 1 bounded float) → works fine
- RouterOutput: 3 fields (1 Literal, 1 bounded float) → works fine

**Rule**: Keep Gemini Flash structured output schemas under 5 constrained fields.

## Round 3: Celebration Detection + Router Guidance

### Fixes Applied
1. **Celebration detection** in compliance_gate (position 7.65) — "I just won $5K!" now sets guest_sentiment="celebration" instead of relying on VADER "positive". Added to _PRIORITY_SENTIMENTS so router can't overwrite.
2. **Celebration tone guide** added to SENTIMENT_TONE_GUIDES — "Match their energy authentically, suggest elevated experiences."
3. **Router prompt enhanced** — emotional statements about property experience now classify as property_qa (not ambiguous). Grief + property context also property_qa. This means emotional inputs get RAG retrieval + specialist dispatch (better context for LLM).

### Key Insight: Sentiment Detection Hierarchy
```
compliance_gate (position 7.6-7.65) → sets priority sentiment (grief, celebration)
       ↓ query_type=None → goes to router
router_node → VADER sentiment detection (skipped for priority sentiments)
       ↓ routes to retrieve → specialist
execute_specialist → reads guest_sentiment → injects tone guide
```
Without the priority guard, VADER overwrites grief→neutral and celebration→positive, losing the specific context.

### Test Count: 3236 passed, 1 skipped, 0 failures (maintained)
