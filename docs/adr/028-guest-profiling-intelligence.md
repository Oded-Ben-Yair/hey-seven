# ADR 028: Guest Profiling Intelligence System

## Status
Accepted (2026-03-01)

## Context
Hey Seven's AI casino host needs deep guest intelligence to deliver personalized service. Current guest data extraction is limited to basic fields (name, occasion) set during specialist dispatch. A systematic profiling approach is needed to:
- Passively extract guest preferences from natural conversation
- Track profile completeness for comp agent readiness
- Inject natural profiling questions without disrupting conversation flow
- Trigger profile-driven incentives (dining credits, free play)

Casino hosts in practice build mental models of guests over multiple interactions. The AI agent needs a structured equivalent that accumulates across turns.

## Decision
Add a `profiling_enrichment_node` between the `generate` and `validate` nodes in the StateGraph, creating a 12-node topology. The node uses LLM structured output to extract 19 profile fields with confidence scoring.

### Architecture

1. **profiling_enrichment_node** (new graph node):
   - Sits between generate and validate (feature-flag gated via `profiling_enabled`)
   - Reuses the whisper planner's LLM singleton (no additional cold-start cost)
   - Extracts fields via `ProfileExtractionOutput` Pydantic model with `ConfidenceField` wrappers
   - Confidence threshold (`PROFILING_MIN_CONFIDENCE=0.7`) gates field acceptance
   - Fail-silent: any error returns empty state update, never crashes the pipeline

2. **Weighted completeness scoring**:
   - 19 fields with differentiated weights summing to 1.0
   - High-impact fields (guest_name=0.15, party_size=0.10) weighted higher
   - Low-impact fields (communication_preference=0.01) weighted lower
   - Score drives comp agent readiness gate (60% threshold)

3. **Golden path phases**: foundation -> preference -> relationship -> behavioral
   - Foundation: name, party_size, visit_purpose (need 2 of 3)
   - Preference: dining, entertainment, gaming, spa (need 1 of 4)
   - Relationship: occasion, companions, visit_frequency (need 1 of 4)
   - Behavioral: all prior phases satisfied

4. **Profiling question injection**:
   - Whisper planner sets `next_profiling_question` and `question_technique`
   - 7 techniques: give_to_get, assumptive_bridge, contextual_inference, need_payoff, incentive_frame, reflective_confirm, none
   - Question appended to AI response naturally (not a separate message)
   - Technique "none" suppresses injection (guest rushed/annoyed)

5. **Incentive engine** (incentives.py):
   - Pure business logic, no I/O, no LLM calls
   - Per-casino rules (5 casinos configured, immutable MappingProxyType)
   - Trigger conditions: birthday, anniversary, gaming_preference, profile_completeness_75
   - Tiered autonomy: auto-approve below threshold, host approval above
   - `string.Template.safe_substitute()` for framing (prevents KeyError on user data)

6. **Field name mapping**:
   - Profiling model uses descriptive names (guest_name, dining_preferences)
   - Extracted fields use compact keys (name, preferences) matching existing state schema
   - Explicit `_FIELD_NAME_MAP` dict prevents silent key drift

### State Schema Changes
- `profiling_phase: Annotated[str | None, _keep_latest_str]` -- persists across turns
- `profile_completeness_score: float` -- recomputed each turn
- `profiling_question_injected: bool` -- ephemeral per-turn flag

## Consequences

### Positive
- Passive intelligence gathering without explicit interrogation
- Weighted completeness prevents false readiness (high-value fields > low-value)
- Confidence gating prevents hallucinated profile data (threshold 0.7)
- Fail-silent design: profiling failure never degrades core agent functionality
- Per-casino incentive rules support multi-tenant deployment
- Reuses whisper LLM singleton: +1 LLM call per turn, no additional cold-start
- Feature-flag gated at build time (graph topology): clean disable without code changes

### Negative
- +1 LLM call per turn increases latency (~200-400ms for structured extraction)
- 19-field extraction prompt is verbose (increases input token cost)
- Field name mapping adds indirection between model and state

## Alternatives Considered

### Inline extraction in generate node
Rejected: SRP violation. Generate node already handles specialist dispatch, context injection, and response generation. Adding extraction would exceed 200 LOC.

### Separate extraction graph
Rejected: Unnecessary complexity. The extraction is a single LLM call with structured output -- it does not need its own graph topology with routing and validation.

### Regex-only extraction
Rejected: Cannot handle paraphrases ("the wife and I" -> party_composition="couple"), conversational context ("for our big day" -> occasion), or corrective statements ("Actually, it's Sarah with an H"). LLM structured output handles all three.

### Dedicated extraction LLM
Rejected: Additional cold-start cost and cache management. The whisper planner's LLM singleton (Gemini Flash, temperature 0.2) is suitable for extraction -- deterministic classification benefits from low temperature.
