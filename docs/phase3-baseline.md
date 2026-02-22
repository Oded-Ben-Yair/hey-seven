# Phase 3 Evaluation Baseline

**Date**: 2026-02-21
**Commit**: aa0a758
**Branch**: main

## Test Counts

| Category | Test Count | File |
|----------|-----------|------|
| Existing deterministic eval | 17 | `tests/test_evaluation_framework.py` |
| Deterministic VCR eval | 18 | `tests/test_eval_deterministic.py` |
| LLM-as-judge metrics | 36 | `tests/test_llm_judge.py` |
| Conversation scenarios | 63 (55 scenarios + 8 validation) | `tests/test_conversation_scenarios.py` |
| **New Phase 3 total** | **99** | `make test-eval-quality` |
| Full suite (pre-existing) | 1406 passing, 52 pre-existing failures | `make test-ci` |

## Deterministic Evaluation Baseline (20 golden cases)

Scored against keyword-stuffed responses to establish ceiling.

| Dimension | Score | Weight |
|-----------|-------|--------|
| Groundedness | 0.9833 | 0.25 |
| Helpfulness | 1.0000 | 0.25 |
| Safety | 1.0000 | 0.35 |
| Persona Adherence | 1.0000 | 0.15 |
| **Overall** | **0.9958** | — |
| Pass Rate | 100% (20/20) | — |

## LLM-as-Judge Baseline (55 scenarios, 10 categories)

Scored using offline deterministic heuristics against representative casino host responses (no API keys).

| Dimension | Avg Score | Description |
|-----------|-----------|-------------|
| Empathy | 0.5000 | Emotional attunement, guest care |
| Cultural Sensitivity | 0.7982 | Respect for diverse backgrounds |
| Conversation Flow | 0.5909 | Natural multi-turn progression |
| Persona Consistency | 0.8019 | Seven persona adherence |
| **Guest Experience** | **0.6516** | Overall quality (weighted composite) |
| **Overall Average** | **0.6685** | Mean of all 5 dimensions |

### Per-Category Breakdown

| Category | Count | Empathy | Cultural | Flow | Persona | Experience | Avg |
|----------|-------|---------|----------|------|---------|------------|-----|
| comp_eligibility | 5 | 0.500 | 0.700 | 0.600 | 0.783 | 0.672 | 0.651 |
| cultural_sensitivity | 5 | 0.500 | 1.000 | 0.600 | 0.883 | 0.692 | 0.735 |
| dining_journey | 6 | 0.600 | 0.800 | 0.600 | 0.786 | 0.667 | 0.691 |
| edge_cases | 6 | 0.600 | 0.800 | 0.600 | 0.780 | 0.681 | 0.692 |
| entertainment | 6 | 0.300 | 0.700 | 0.600 | 0.788 | 0.578 | 0.593 |
| escalation_paths | 6 | 0.500 | 0.700 | 0.600 | 0.780 | 0.626 | 0.641 |
| greeting_to_deep | 5 | 0.400 | 0.800 | 0.500 | 0.883 | 0.642 | 0.645 |
| hotel_planning | 5 | 0.400 | 0.800 | 0.600 | 0.888 | 0.638 | 0.665 |
| profile_building | 6 | 0.500 | 0.900 | 0.600 | 0.783 | 0.672 | 0.691 |
| sentiment_shifts | 5 | 0.700 | 0.800 | 0.600 | 0.683 | 0.657 | 0.688 |

### Key Observations

- **Strongest**: Persona consistency (0.80) and cultural sensitivity (0.80) — deterministic patterns detect these well
- **Weakest**: Empathy (0.50) and conversation flow (0.59) — keyword matching misses nuanced emotional attunement
- **Entertainment gap**: Lowest overall (0.59) due to low empathy signal in informational responses
- **Sentiment shifts strongest empathy**: 0.70 — empathetic mock responses score well against emotional scenarios

## Conversation Scenario Coverage (55 scenarios, 10 categories)

| Category | Count | Description |
|----------|-------|-------------|
| dining_journey | 6 | Restaurant discovery, budget, fine dining, price comparison, location, late arrival |
| hotel_planning | 5 | Room types, tower comparison, upgrades, amenities |
| entertainment | 6 | Shows, spa, pool, events, package deals |
| sentiment_shifts | 5 | Positive-to-frustrated, neutral-to-enthusiastic, pricing concerns |
| profile_building | 6 | Name/party size/birthday/anniversary extraction |
| comp_eligibility | 5 | Rewards programs, tier levels, benefits |
| edge_cases | 6 | Empty messages, long messages, special chars, topic switching |
| greeting_to_deep | 5 | Greeting to progressively deeper questions |
| cultural_sensitivity | 5 | Multilingual greetings, dietary accommodations |
| escalation_paths | 6 | Info to gambling advice to responsible gaming |

## Makefile Targets

```bash
make test-eval-quality   # Run all Phase 3 eval tests (99 tests)
make test-ci             # Run full CI suite (1505+ tests)
make test-eval           # Run live LLM eval (requires GOOGLE_API_KEY)
```

## Notes

- Offline heuristic scores are intentionally conservative (empathy baseline 0.50)
- These scores will improve when LLM-as-judge mode is enabled (`EVAL_LLM_ENABLED=true`)
- Persona consistency and cultural sensitivity score highest because deterministic patterns detect these well
- Empathy is lowest because keyword matching cannot fully capture emotional attunement
- Conversation flow is limited by static mock responses (no real multi-turn adaptation)
- This baseline establishes the "before" measurement for tracking Phase 2-5 improvements

---

## Phase 3 Completion Summary (2026-02-21)

### Test Count Progression

| Milestone | Tests | Delta |
|-----------|-------|-------|
| Pre-Phase 3 (v1.0.0) | 1452 | — |
| Phase 1 (eval infra) | 1551 | +99 |
| Phase 2-5 (all phases) | **1580** | **+128 total** |

### What Was Implemented

**Phase 2: Guest Profile Wiring + Sentiment Detection**
- 3 new state fields: `guest_sentiment`, `guest_context`, `guest_name`
- VADER sentiment detection in router (sub-1ms, casino-aware overrides)
- Guest profile wiring via `get_agent_context()` (425 LOC scaffolded code now live)
- Whisper planner enabled for all 5 specialist agents (was only host_agent)
- 3 new feature flags: `sentiment_detection_enabled`, `guest_profile_enabled`, `field_extraction_enabled`

**Phase 3: Persona Depth + Tone Calibration**
- BrandingConfig wired into system prompts via `get_persona_style()`
- 5 unique specialist personas (Foodie Insider, Excitement Builder, Comfort Expert, Strategic Advisor, Master Host)
- Sentiment-adaptive tone guidance (frustrated, negative, positive, neutral)
- `PERSONA_STYLE_TEMPLATE` + `SENTIMENT_TONE_GUIDES` in prompts.py

**Phase 4: extracted_fields + Persona Envelope**
- Deterministic regex field extraction: name, party_size, visit_date, preferences, occasion
- Fields accumulate across conversation turns
- Persona envelope expanded: BrandingConfig enforcement (exclamation limit, emoji check)
- Guest name injection in responses

**Phase 5: Infrastructure Polish**
- BoundedMemorySaver audit: legitimately needed (MemorySaver 0.2.60 has no bounds), `hasattr` guard sufficient
- ChatRequest max_length=4096 already in place (Phase 5.2 was pre-done)
- SSE E2E test: 9 tests covering lifecycle, error handling, event ordering
- Requirements.txt pinned (deepeval, vaderSentiment, pyyaml)

### Files Modified/Created

| File | Action | Phase |
|------|--------|-------|
| `src/agent/state.py` | Modified (3 new fields) | P2 |
| `src/agent/graph.py` | Modified (initial_state, guest profile wiring) | P2 |
| `src/agent/sentiment.py` | Created (VADER detection) | P2 |
| `src/agent/extraction.py` | Created (regex field extraction) | P4 |
| `src/agent/nodes.py` | Modified (sentiment + extraction in router) | P2, P4 |
| `src/agent/agents/_base.py` | Modified (guest context, persona style, sentiment tone) | P2, P3 |
| `src/agent/agents/dining_agent.py` | Modified (persona + whisper) | P2, P3 |
| `src/agent/agents/entertainment_agent.py` | Modified (persona + whisper) | P2, P3 |
| `src/agent/agents/hotel_agent.py` | Modified (persona + whisper) | P2, P3 |
| `src/agent/agents/comp_agent.py` | Modified (persona + whisper) | P2, P3 |
| `src/agent/prompts.py` | Modified (host persona, persona style template, tone guides) | P3 |
| `src/agent/persona.py` | Modified (branding enforcement, guest name injection) | P4 |
| `src/casino/config.py` | Modified (3 new feature flags) | P2 |
| `src/casino/feature_flags.py` | Modified (3 new feature flags, parity checks) | P2 |
| `requirements.txt` | Modified (pinned versions) | P5 |
| `tests/test_sse_e2e.py` | Created (9 SSE E2E tests) | P5 |
| `tests/test_doc_accuracy.py` | Modified (state field count 13→16) | P5 |

### Architectural Decisions

1. **BrandingConfig from DEFAULT_CONFIG (not async Firestore)**: Phase 3 persona style reads from `DEFAULT_CONFIG` synchronously. This is intentional — persona style is build-time config (like graph topology), not per-request. When multi-tenant Firestore config is needed, the `get_casino_config()` async path is already wired.

2. **Extraction merges across turns**: `extracted_fields` accumulate — if a guest says their name in turn 1 and party size in turn 3, both are preserved. This mirrors natural conversation profiling.

3. **Guest name injection is conservative**: Only injects when name is NOT already in the response and response is > 50 chars. Avoids double-naming and doesn't inject into fallback messages.

4. **Persona enforcement runs AFTER PII redaction**: Processing order in persona_envelope_node is PII-first (fail-closed), then branding (best-effort), then name injection, then SMS truncation.
