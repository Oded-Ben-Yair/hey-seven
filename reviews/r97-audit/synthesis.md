# R97 Architecture Audit — Synthesis

**Date**: 2026-03-05
**Auditors**: auditor-core, auditor-pipeline, auditor-api, auditor-tests
**Scope**: All 12 components + test reality check (66 source files, ~24K LOC, 3555 tests)

---

## 1. Confidence Matrix (sorted lowest-first)

| # | Component | Confidence | Verdict | LOC | Tests |
|---|-----------|-----------|---------|-----|-------|
| 11 | Observability | 72% | needs-fixes | 770 | ~15 |
| 6 | Profiling + Extraction | 75% | needs-new-tool | 1,922 | 325 |
| 8 | Prompts + Persona | 78% | needs-fixes | 1,385 | 69 |
| 5 | RAG Pipeline | 82% | production-ready | 1,686 | 142 |
| 3 | Specialist Agents | 82% | production-ready* | 1,863 | 156 |
| 10 | CMS + SMS | 82% | production-ready* | 1,796 | 155 |
| 2 | Router + Dispatch | 85% | production-ready | 2,213 | 272 |
| 7 | Incentives + Crisis + Sentiment | 85% | needs-new-tool | 1,446 | 205 |
| 1 | Graph Architecture | 88% | production-ready | 1,112 | 173 |
| 9 | API Layer | 88% | production-ready | 2,294 | 127 |
| 12 | Config + Feature Flags | 90% | production-ready | 2,201 | 108 |
| 4 | Guardrails + Compliance | 92% | production-ready | 2,059 | 372 |

**Weighted average confidence: 83.6%** (weighted by LOC)

\* = production-ready with documented caveats (see reports)

### Key Finding: 0 scaffolded modules

All 66 source files are wired to production entry points. Every `from src.X import Y` chain traces back to `app.py` or `graph.py`. This is a genuine production codebase, not a prototype.

---

## 2. Gap Priority List (mapped to eval dimensions)

### CRITICAL — Missing Business Logic (not model capability)

| Priority | Dimension | Score | Gap | Required Tool | Impact |
|----------|-----------|-------|-----|---------------|--------|
| P0 | H9 Comp Strategy | 1.9 | No deterministic comp policy. Incentive engine has per-casino rules but cannot calculate comps from player worth/ADT/trip history. | **CompStrategy** LangGraph tool | +3-4 pts |
| P0 | P9 Host Handoff | 2.1 | `format_handoff_summary()` exists but lacks conversation history, recommended actions, risk flags. | **HandoffOrchestrator** LangGraph tool | +3-4 pts |
| P1 | H10 Lifetime Value | 3.5 | No return-visit seeding in any module. Incentive engine handles one-time offers only. | **LTV Nudge Engine** LangGraph tool | +2-3 pts |
| P1 | P8 Profile Completeness | 3.7 | Weighted completeness maxes ~37% in 3 turns. Not a code bug — conversation length limitation. Pro model may extract more from same turns. | Phase 1 Pro switch (extraction quality) | +1-2 pts |
| P2 | H6 Rapport Depth | 4.0 | System prompt says "be warm" but no micro-pattern retrieval for guest types. | **Rapport Ladder** retrieval tool | +1-2 pts |

### HIGH — Structural Issues

| Priority | Component | Issue | Fix Effort |
|----------|-----------|-------|------------|
| P2 | nodes.py | 1,674 LOC monolith (extract LLM factory + tone enforcement) | 1 day |
| P2 | _base.py | 1,379 LOC with ~15 responsibilities (extract prompt_builder.py) | 1 day |
| P2 | prompts.py | 1,052 LOC (split into system/routing/helpline/few-shot) | 0.5 day |
| P2 | pipeline.py | 1,203 LOC (split ingestion/retrieval/caching) | 0.5 day |
| P3 | Rule 8 | Documentation claims "NO MOCK TESTING" — 55/104 test files use mocks (1694 occurrences). Update Rule 8 to reflect reality. | 10 min |

### MEDIUM — Production Deployment Gaps

| Priority | Component | Issue | Fix Effort |
|----------|-----------|-------|------------|
| P2 | Observability | Trace spans local-only (no Cloud Trace export), no LangFuse failure alerting, no request-ID correlation | 1-2 days |
| P2 | SMS | Consent store in-memory (lost on restart = TCPA liability). Must persist to Firestore before enabling SMS. | 0.5 day |
| P3 | Config | No live Redis/Firestore integration tests. Lua scripts and hot-reload untested against real backends. | 1 day |
| P3 | RAG | No live Vertex AI Vector Search test. ChromaDB tested but prod uses Firestore backend. | 0.5 day |
| P3 | Router | `_select_model()` uses global flag, not per-casino. Blocks canary rollout of Pro by casino. | 0.5 day |
| P3 | Agents | Comp agent hardcodes Mohegan Sun details in prompt. Breaks multi-property. | 0.5 day |
| P3 | Prompts | Spanish prompt parity has no automated check. | 0.5 day |
| P4 | Agents | No prompt token counting. Risk of exceeding context window on complex turns. | 0.5 day |

---

## 3. Phase 1 Readiness — Pro Switch Assessment

### Safe to proceed? YES, with one fix.

**What works:**
- `_select_model()` in `nodes.py` already routes Flash->Pro based on confidence, sentiment, complexity, crisis state
- All 5 specialist agents read `model_used` from state and select the correct LLM singleton
- TTL-cached LLM singletons with jitter handle credential rotation for both Flash and Pro
- Circuit breaker tracks failures per-model (separate `_get_llm` and `_get_complex_llm` caches)
- Graph topology doesn't change — Pro is a drop-in model replacement

**One blocker:**
- `_select_model()` uses `settings.MODEL_ROUTING_ENABLED` (global boolean), not `is_feature_enabled(casino_id, ...)` (per-casino). This means Pro routing is all-or-nothing — no canary by casino.

**Fix**: Change `_select_model()` to read from `is_feature_enabled(property_id, "model_routing_enabled")`. Already used by 5+ other feature checks in dispatch.py/nodes.py. ~10 LOC change.

**Canary plan (after fix):**
1. Enable `model_routing_enabled: true` for Mohegan Sun only (Firestore feature flag)
2. Monitor: latency (P50/P95 from `/metrics`), cost (LangFuse traces), behavioral quality (judge panel)
3. If B-avg >= 7.0 for Mohegan Sun after 100 conversations, roll out to remaining 4 casinos
4. Cost estimate: $280/mo at 10K conversations (4x Flash cost)

---

## 4. Phase 2 Tool Specs (Draft)

### Tool 1: CompStrategy

**Purpose**: Deterministic comp policy engine. Replaces the vague "explore rewards" responses that kill H9.

```python
class CompStrategyInput(BaseModel):
    guest_tier: Literal["new", "regular", "vip", "high_roller"]
    estimated_adt: float | None  # Average Daily Theoretical
    visit_frequency: Literal["first", "occasional", "regular", "weekly"]
    current_occasion: str | None  # birthday, anniversary, loss_recovery
    property_id: str

class CompStrategyOutput(BaseModel):
    eligible_comps: list[CompOffer]  # Ranked by appropriateness
    approval_required: bool  # True if above auto-approve threshold
    talking_points: list[str]  # Natural language framing
    restrictions: list[str]  # "Not available on holiday weekends"
```

**Architecture**: LangGraph tool (not node). Called by comp_agent when guest asks about comps, loyalty, or rewards. Reads per-casino comp policies from `casino/config.py` or Firestore.

**Key rules**: ADT-based comp tiers, occasion multipliers, loss-recovery escalation, auto-approve below $50, human handoff above $200.

### Tool 2: HandoffOrchestrator

**Purpose**: Structured host handoff summary. Fixes P9 (2.1) by providing actionable intelligence for human hosts.

```python
class HandoffSummary(BaseModel):
    guest_name: str | None
    profile_completeness: float
    conversation_summary: str  # 3-5 sentence narrative
    key_preferences: list[str]  # Extracted from profiling
    risk_flags: list[str]  # Crisis history, RG triggers, complaints
    recommended_actions: list[str]  # Specific next steps
    urgency: Literal["routine", "priority", "urgent"]
    handoff_reason: str  # Why AI is handing off
```

**Architecture**: LangGraph tool called when `handoff_request` is emitted. Aggregates data from `guest_profile`, conversation `messages`, `crisis_active` state, and extraction fields.

### Tool 3: LTV Nudge Engine

**Purpose**: Return-visit seeding. Fixes H10 (3.5) by planting forward-looking hooks.

```python
class LTVNudge(BaseModel):
    nudge_type: Literal["upcoming_event", "seasonal_offer", "loyalty_milestone", "personal_callback"]
    message_fragment: str  # Natural language to inject into response
    timing: str  # "next visit", "this weekend", "next month"
    confidence: float  # How relevant this nudge is to current context
```

**Architecture**: Called by `execute_specialist()` in `_base.py` at response assembly time. Reads upcoming events from knowledge-base, guest visit patterns from profile, and casino calendar. Injects a brief "By the way, [nudge]" into the specialist response.

### Tool 4: Rapport Ladder

**Purpose**: Micro-pattern retrieval for rapport building. Fixes H6 (4.0) by providing context-specific conversation techniques.

```python
class RapportPattern(BaseModel):
    guest_type: Literal["first_timer", "regular", "vip", "family", "couple", "solo", "grieving", "celebrating"]
    technique: str  # "callback to earlier mention", "specific knowledge flex", "anticipatory service"
    example: str  # Natural language example
    when_to_use: str  # Contextual trigger
```

**Architecture**: RAG retrieval tool. Rapport patterns stored in knowledge-base as structured JSON. Retrieved by guest_type + conversation context. Injected into specialist system prompt as "Rapport Technique" section.

---

## 5. Summary Statistics

| Metric | Value |
|--------|-------|
| Total source files audited | 66 |
| Total LOC | ~20,733 |
| Scaffolded files | **0** |
| Production-ready components | 8 of 12 |
| Components needing fixes | 2 (Observability, Prompts) |
| Components needing new tools | 2 (Profiling, Incentives) |
| Total tests | 3,555 |
| Live LLM tests | 16 (0.45%) |
| Mock-using test files | 55 of 104 (52.9%) |
| Deterministic test files | 49 of 104 (47.1%) |
| Line coverage | 90.62% |
| Phase 1 blocker | 1 (per-casino model routing flag) |
| Estimated fix effort | ~10 LOC |

---

## 6. Recommended Action Sequence

### Immediate (before Phase 1)
1. Fix `_select_model()` to use per-casino feature flag (~10 LOC)
2. Update Rule 8 documentation to reflect mock testing reality

### Phase 1 (Days 1-3): Pro Switch
3. Enable `model_routing_enabled` for Mohegan Sun via Firestore
4. Run 20-scenario canary eval with GPT-5.2 judge
5. If B-avg >= 7.0, roll out to all casinos

### Phase 2 (Weeks 1-3): Structured Tools
6. Build CompStrategy tool (targets H9: 1.9 -> 5.0+)
7. Build HandoffOrchestrator tool (targets P9: 2.1 -> 5.0+)
8. Build LTV Nudge Engine (targets H10: 3.5 -> 5.0+)
9. Build Rapport Ladder retrieval (targets H6: 4.0 -> 5.0+)

### Phase 2b (parallel): Structural cleanup
10. Extract LLM factory from nodes.py
11. Extract prompt_builder from _base.py
12. Persist SMS consent store to Firestore
13. Add Cloud Trace span export
14. Add request-ID correlation

### Phase 3 (Weeks 4-8): Distillation
15. Collect 2K+ graded Pro conversations
16. Fine-tune Flash on Vertex AI
17. Bandit routing (Pro for complex, fine-tuned Flash for routine)
