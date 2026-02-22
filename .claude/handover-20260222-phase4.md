# Phase 4 Handover: Hey Seven → Phase 5 (88/100 → 95/100)

**Session**: hey-seven-session-20260222-06d4f8
**Date**: 2026-02-22
**Score**: 88/100 (Code 34/40, Agent 53/60)
**Memory MCP**: `hey-seven-phase4-learnings`, `hey-seven-95-roadmap`

---

## What Was Done (Phase 4: R21-R30)

9 commits, 189 new tests (1839 total), 11 critical bugs fixed across 4 review rounds.

| Round | Type | Key Deliverable |
|-------|------|----------------|
| R21 | Implement | Frustration escalation (HEART framework), proactive suggestions (WhisperPlan), persona drift prevention |
| R22 | Implement | Multi-turn golden conversations (6 scenarios), regression detection CI gate, quality baseline |
| R23 | Review | 6 CRITICAL: duplicate suggestion injection, sentiment gate accepted None, suggestion_offered unenforced, NaN passed CI, empathy baseline non-functional, invalid baseline keys silent |
| R24 | Implement | 3 knowledge-base files (loyalty programs, dining, hotel ops), CASINO_PROFILES (3 properties), HEART escalation language |
| R25 | Review | 3 CRITICAL: HEART was dead code, helplines hardcoded to CT, The Mirage listed as active |
| R26 | Implement | 69 tests: topic switching (34), E2E Phase 4 integration (35) |
| R27 | Review | 2 CRITICAL: suggestion gate allowed "neutral" (tightened to positive-only), persona style read DEFAULT_CONFIG |
| R28 | Review | Final 10-dimension scoring: 87/100. Ship GO with conditions. |
| R29 | Fix | Persona hardcoding eliminated in all 3 locations (persona.py, _base.py x2) |
| R30 | Validate | 1839 tests passing, zero regressions |

---

## Honest Gap Analysis: Why We're at 88, Not 95

### Dimensions That Improved (good)
- Prompts & Guardrails: 8.5 → 9.5 (HEART framework, property-aware helplines)
- EQ: 8.0 → 9.0 (frustration escalation with graduated HEART response)
- Domain: 8.0 → 9.0 (3 knowledge-base files, 6 loyalty programs)
- Testing: 7.0 → 8.5 (golden conversations, regression detection, 189 new tests)

### Dimensions That Barely Moved (problem)
- **RAG Pipeline: 8.0 → 8.0** — Zero changes. No new ingestion tests, no multi-property isolation, no retrieval benchmarks.
- **API Design: 8.5 → 8.5** — Zero changes. No request validation improvements, no SSE reconnection tests.
- **Guest Experience: 8.0 → 8.5** — Only +0.5. Proactive suggestions exist but gate is so restrictive (positive-only) that they rarely fire.
- **Persona: 8.0 → 8.5** — Only +0.5. Multi-property config exists but CONCIERGE_SYSTEM_PROMPT still hardcodes Mohegan Sun description.
- **Eval: 8.0 → 8.5** — Only +0.5. Golden conversations + regression detection added, but still keyword-based scoring. No real LLM-as-judge.

### Why Score Plateaued
Phase 4 focused exclusively on agent quality dimensions (EQ, Guest, Eval) while leaving code dimensions untouched. You can't reach 95 overall by perfecting 6 dimensions and ignoring 4.

---

## The 4 Concrete Deliverables to Reach 93+ (Prioritized)

### 1. Full-Graph E2E Tests (Testing 8.5→9.5, Architecture 9.0→9.5)
**Why it matters**: Every single reviewer (R27, R28) flagged this as the #1 gap. All 186 Phase 4 tests call `execute_specialist()` in isolation. Zero tests compile the graph via `build_graph()` and invoke `chat()`. Wiring bugs between nodes are invisible to CI.

**What to build**:
```python
# tests/test_full_graph_e2e.py
async def test_full_pipeline_dining_query():
    """E2E: user message → compliance_gate → router → retrieve → whisper → generate → validate → persona → respond"""
    graph = build_graph()  # Real graph, mocked LLMs
    config = {"configurable": {"thread_id": "test-e2e-1"}}
    result = await graph.chat("What restaurants do you have?", config=config)
    # Verify: response mentions restaurants, persona applied, no gambling advice
```

**Files**: `src/agent/graph.py` (read build_graph), new `tests/test_full_graph_e2e.py`
**Mocking**: Mock `_get_llm`, `_get_validator_llm`, `_get_whisper_llm` to return deterministic responses. Use the patterns in `tests/test_graph_v2.py` for reference.

### 2. System Prompt Parameterization (Persona 8.5→9.5, Guest 8.5→9.0)
**Why it matters**: CONCIERGE_SYSTEM_PROMPT in `prompts.py` describes Mohegan Sun specifics ("casino resort in Uncasville, Connecticut"). Multi-property deployment would serve CT knowledge to NJ guests.

**What to build**: Read `CONCIERGE_SYSTEM_PROMPT` and replace Mohegan Sun specifics with `$property_description` variable. Add `property_description` to `CASINO_PROFILES` for each property. Update `execute_specialist()` to substitute from the active profile.

**Files**: `src/agent/prompts.py` (CONCIERGE_SYSTEM_PROMPT), `src/casino/config.py` (CASINO_PROFILES), `src/agent/agents/_base.py` (substitution)

### 3. Real LLM-as-Judge (Eval 8.5→9.5)
**Why it matters**: `llm_judge.py` has `EVAL_LLM_ENABLED=true` path that falls back to offline. The actual scoring is VADER + regex patterns which can't assess subjective quality (empathy depth, persona drift, response coherence).

**What to build**: Implement the LLM judge path using G-Eval pattern:
1. Define evaluation rubric as a structured prompt
2. Use `with_structured_output()` on Gemini Flash for fast scoring
3. Score 5 dimensions: groundedness, persona_fidelity, safety, contextual_relevance, proactive_value
4. Run as opt-in CI stage (requires GOOGLE_API_KEY)

**Files**: `src/observability/llm_judge.py` (implement the `llm_enabled` branch), new `tests/test_llm_judge_live.py` (marked `@pytest.mark.skipif` without API key)

### 4. RAG Pipeline Improvements (RAG 8.0→9.0)
**Why it matters**: RAG hasn't been touched since R10. No version-stamp purging tests, no multi-property collection isolation verification, no retrieval quality benchmarks.

**What to build**:
- Test `_purge_stale_chunks()` with mock ChromaDB
- Test property_id metadata filtering (multi-tenant isolation)
- Add retrieval quality golden tests (known queries → expected chunks)
- Add `knowledge-base/casino-operations/entertainment-guide.md` (flagged missing by R25 reviewer)

**Files**: `src/rag/pipeline.py`, new `tests/test_rag_quality.py`, new `knowledge-base/casino-operations/entertainment-guide.md`

---

## Key Rules Learned (Already Added to Rule Files)

These were added to `~/.claude/rules/langgraph-patterns.md` during this session:

1. **Proactive suggestion gate: positive-only** — Never allow "neutral" sentiment. Neutral is absence of evidence, not positive evidence. (R23-R27)
2. **Frustration escalation from message history** — Count consecutive frustrated HumanMessages from history instead of adding a state field. No reducer complexity. (R21)
3. **Always use get_casino_profile(), never DEFAULT_CONFIG** — Every import of DEFAULT_CONFIG for runtime data is a multi-tenant bug. (R25-R29)
4. **Research agents can't write files** — Use code-worker or general-purpose for tasks that need file output. (R21)

---

## Files Modified in Phase 4

### Source Code (6 files)
- `src/agent/agents/_base.py` — frustration escalation, proactive suggestions, persona drift, HEART framework
- `src/agent/whisper_planner.py` — proactive_suggestion + suggestion_confidence fields
- `src/agent/prompts.py` — whisper planner prompt, HEART_ESCALATION_LANGUAGE, property-aware helplines
- `src/agent/state.py` — suggestion_offered field with _keep_max reducer
- `src/agent/graph.py` — _initial_state() updated for suggestion_offered
- `src/agent/persona.py` — property-specific branding via get_casino_profile()
- `src/casino/config.py` — CASINO_PROFILES, get_casino_profile()
- `src/observability/llm_judge.py` — golden conversations, detect_regression(), NaN guard, baseline calibration

### Knowledge Base (3 files)
- `knowledge-base/casino-operations/loyalty-programs.md` — 6 real casino programs
- `knowledge-base/casino-operations/dining-guide.md` — Mohegan Sun + Foxwoods
- `knowledge-base/casino-operations/hotel-operations.md` — Room types, VIP tiers

### Research (4 files)
- `research/phase4-casino-host-profiles.md` (40.6KB)
- `research/phase4-ai-agent-practices.md` (12.8KB)
- `research/phase4-casino-regulations.md` (33.3KB)
- `research/phase4-social-intelligence.md` (10.9KB)

### Tests (5 files, 189 tests)
- `tests/test_r21_agent_quality.py` (28 tests)
- `tests/test_r22_llm_judge.py` (35 tests)
- `tests/test_r24_domain.py` (54 tests)
- `tests/test_r26_conversation.py` (34 tests)
- `tests/test_r26_e2e_phase4.py` (35 tests + 3 from R23 fixes)

### Review Reports (4 directories)
- `reviews/r23/` (reviewer-alpha.md, reviewer-beta.md)
- `reviews/r25/` (review.md)
- `reviews/r27/` (review.md)
- `reviews/r28/` (final-review.md)

---

## Anti-Patterns to Avoid in Phase 5

1. **Don't just chase agent quality points** — Phase 4 improved agent dims but left code dims flat. Attack ALL dimensions.
2. **Don't test execute_specialist() and claim E2E** — Must test full graph (build_graph + chat). Reviewers will flag this every time.
3. **Don't use research-specialist for file-writing tasks** — It lacks Write tool. Use code-worker.
4. **Don't add golden conversations without calibrating baseline** — Each new conversation changes avg scores. Recalibrate QUALITY_BASELINE.
5. **Score plateau (3+ rounds within +/-2) = change strategy** — Don't repeat the same review+fix pattern. Shift to untouched dimensions.

---

## Next Session Prompt

```
Read the handover at .claude/handover-20260222-phase4.md and status.json.
Read Memory MCP entities: hey-seven-phase4-learnings, hey-seven-95-roadmap.

Phase 4 is complete (88/100). We need Phase 5 to reach 95/100.

The 4 highest-impact deliverables (in order):
1. Full-graph E2E tests (build_graph + chat with mocked LLMs)
2. System prompt parameterization from CASINO_PROFILES
3. Real LLM-as-judge via EVAL_LLM_ENABLED
4. RAG pipeline tests + entertainment guide knowledge base

Start with #1 (full-graph E2E) — it's the #1 gap flagged by every reviewer.
```

---

*Handover generated 2026-02-22 by Claude Opus 4.6*
