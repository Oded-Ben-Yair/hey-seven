# Hey Seven: Roadmap to 9.5+/10 Across All 30 Dimensions

**Created**: 2026-03-02
**Method**: Honest Answer analysis + Reasoning Specialist blind-spot validation
**Status**: Plan — awaiting approval

---

## Current Reality (Verified Scores)

| Domain | Score | Source | Date |
|--------|-------|--------|------|
| **Technical (D1-D10)** | **9.63/10** | R75 4-model panel | 2026-03-01 |
| **Behavioral (B1-B10)** | **3.2/10** | R81 3-model panel (5 scenarios) | 2026-03-02 |
| **Profiling (P1-P10)** | **Never measured** | No live eval + judge run exists | — |

### Why Behavioral Is 3.2, Not Higher

The R81 audit identified 5 systemic failures. These are NOT bugs — they are architectural gaps:

1. **Model noncompliance** — prompts say "NEVER start with Oh!" but Gemini Flash does it anyway. The prompt already contains the right instructions. Adding more won't help.
2. **27% fallback rate** — confirmations trigger RAG → validator → reject → fallback cascade. (Fix 1 from R81 code changes addresses this partially.)
3. **Identical crisis responses** — 3x verbatim repeat regardless of guest's evolving disclosure. (Fix 2 from R81 addresses this.)
4. **Proactivity gates never open** — `_should_inject_suggestion()` has 5 gates that rarely ALL pass. Zero cross-domain proactivity observed.
5. **Comp agent upsells to angry guests** — frustrated guest → dispatched to comp → promotional prompt overrides tone guidance.

### Critical Insight: This Is NOT a Prompt Problem

The validator's analysis confirmed: **the system prompt already says the right things and the LLM ignores them.** Phase 3B of the original plan (behavioral pre-reasoning step) was a prompt approach to a compliance problem. The corrected plan uses **system-level enforcement** — post-generation controls that mechanically reject what the LLM produces when it violates constraints.

---

## Corrected Plan (3 Tracks, 10-12 Days)

### Track 1: System Controls (Days 1-4) — Highest Leverage

These are code changes to enforce behavioral quality that the LLM's prompt alone cannot guarantee.

#### 1A: Unblock Evaluation (Day 1, 1 hour)
- Get fresh Gemini API key (current leaked, 403 on all live calls)
- Without this, nothing can be measured
- **Deliverable**: Working API key, successful 5-scenario smoke test

#### 1B: Full Baseline (Day 1, 3 hours)
- Run full 220-scenario eval (behavioral + profiling)
- Run 3-model judge panel (Gemini Pro, GPT-5.2, Grok 4)
- First-ever P1-P10 scores established
- **Deliverable**: `r82-responses.json`, `r82-judge-scores.json`, `r82-icc-report.md`
- **Cost**: ~$20 API

#### 1C: Post-Generation Slop Detector (Day 2, 4 hours)
**File**: New node or hook in `persona_envelope_node` (src/agent/nodes.py)

The LLM ignores "NEVER say X" instructions. Instead, enforce mechanically:

```python
_SLOP_PATTERNS = [
    (r"^(?:Oh[,!]?\s)", ""),  # Strip "Oh, " opener
    (r"(?:I'd be (?:absolutely |truly )?delighted)", "I can help with that"),
    (r"(?:What a (?:wonderful|great|fantastic) question)", ""),
    (r"(?:Ah[,!]\s)", ""),
    # ... 15-20 patterns from R75 data
]

def _enforce_tone(response: str, sentiment: str) -> str:
    """Post-generation tone enforcement. Returns cleaned response."""
    for pattern, replacement in _SLOP_PATTERNS:
        response = re.sub(pattern, replacement, response, count=1)
    # Cap exclamation marks
    if response.count("!") > 1:
        # Keep only the last one
        parts = response.split("!")
        response = ".".join(parts[:-1]) + "!" + parts[-1] if parts[-1] else ".".join(parts[:-1]) + "."
    return response
```

- Runs AFTER LLM generation, BEFORE streaming to client
- Zero latency impact (regex, <1ms)
- **Directly attacks B6 (tone calibration) which scored 3-4/10**
- **Deliverable**: `_enforce_tone()` function + 15 slop patterns from R75 data

#### 1D: Intent-Aware Validation (Day 2-3, 4 hours)
**File**: `src/agent/prompts.py` (VALIDATION_PROMPT) + `src/agent/nodes.py` (validate_node)

Current validator applies the same 6 criteria to ALL responses. This means:
- Acknowledgment turns ("sounds good, what else?") get rejected for lacking grounded facts
- Emotional turns get rejected for not having RAG sources
- Short responses get rejected for "insufficient detail"

Fix: Pass `query_type` into the validation prompt and adjust criteria:

```
## Validation Criteria (adapted to intent: $query_type)

IF query_type == "greeting" or acknowledgment:
  - Must be on-topic (about the property). ✓
  - Must NOT fabricate facts. ✓
  - NO length or detail requirement. ← new
  - NO grounding requirement. ← new

IF query_type == "self_harm" or crisis:
  - Must include crisis resources. ✓
  - Must NOT upsell or redirect to property services. ← new
  - Must NOT repeat verbatim from previous turn. ← new

IF query_type == "property_qa" or "hours_schedule":
  - Must be grounded in retrieved context. ✓
  - Specific facts must match context. ✓
  - Category-level mentions OK without specific facts. ✓ (R80 fix)
```

- **Directly attacks the 27% fallback rate**
- **Deliverable**: Updated VALIDATION_PROMPT with $query_type variable, validate_node passes query_type

#### 1E: Frustration/Crisis Suppression of Promotional Content (Day 3, 2 hours)
**File**: `src/agent/agents/_base.py` (_build_behavioral_prompt_sections)

When `effective_sentiment` is "frustrated" or "negative" AND the specialist is "comp":

```python
if effective_sentiment in ("frustrated", "negative"):
    sections.append(
        "## OVERRIDE: Guest Is Frustrated\n"
        "The guest is upset. Your ENTIRE response must follow these rules:\n"
        "1. NO promotional language. No 'explore rewards', 'benefits shine', 'exciting perks'.\n"
        "2. Be factual and direct. State what they get, not how great it is.\n"
        "3. Acknowledge their frustration FIRST before any information.\n"
        "4. Offer to connect with a human host.\n"
        "5. Keep response under 3 sentences."
    )
```

- **Directly attacks the "I'd love to help explore rewards!" to angry guest** (R81 CRITICAL)
- **Deliverable**: Frustration override in _build_behavioral_prompt_sections

#### 1F: Proactivity Gate Instrumentation + Threshold Tuning (Day 3-4, 3 hours)
**File**: `src/agent/agents/_base.py` (_should_inject_suggestion)

Step 1: Add logging to each gate to measure pass rates.
Step 2: Based on data, likely changes:
- Lower `suggestion_confidence` threshold from 0.8 → 0.6
- Allow proactivity on "neutral" sentiment (currently blocked)
- Allow proactivity even when `retrieved_context` is limited (use general domain knowledge)

- **Directly attacks B4 (agentic proactivity: 2.6/10)**

#### 1G: Response Length/Format Budgets (Day 4, 2 hours)
**File**: `src/agent/agents/_base.py` or `persona_envelope_node`

Add per-intent response constraints:
- Greeting/acknowledgment: max 2 sentences
- Information response: max 150 words, max 3 bullet points
- Crisis: exact template structure (no free-form)
- Terse guest (detected via dynamics): max 60 words

Enforce post-generation (count words, trim or request shorter regeneration).

---

### Track 2: Content Fixes (Days 4-7) — Medium Leverage

These are prompt content changes that are meaningful ONLY after Track 1's enforcement layer exists.

#### 2A: Few-Shot Example Library (Day 4-5, 4 hours)
- Write 5 few-shot examples per specialist (25 total)
- Each covers: sarcasm response, grief response, implicit signal response, celebration response, VIP expectation
- Format: `Guest: [message]\nIdeal: [response with behavioral reasoning]\n`
- Inject via `$few_shot_examples` template variable in each specialist prompt
- These now work because Track 1C enforces the patterns mechanically

#### 2B: Comp Agent Prompt Rewrite (Day 5, 2 hours)
**File**: `src/agent/agents/comp_agent.py`

Current: "the guest's trusted rewards insider," "benefits really start to shine"
After: Factual, no-hype comp specialist. Leads with "Here's what your tier includes:" not "You're going to love this!"

#### 2C: Cross-Domain Bridge Templates (Day 5-6, 3 hours)
Instead of hoping the LLM generates cross-domain suggestions, provide 15 bridge templates:

```python
BRIDGE_TEMPLATES = {
    ("dining", "entertainment"): "Since you're having dinner at {venue}, the {show} starts at {time} — could be a great way to round out the evening.",
    ("dining", "spa"): "After dinner, our {spa} is open until {time} — a nice way to wind down.",
    # ... 13 more combinations
}
```

Inject the appropriate bridge into the system prompt based on `domains_discussed` and available context.

#### 2D: Crisis Response Progression Testing (Day 6, 2 hours)
The R81 code changes (Fix 2) added `crisis_turn_count` and `_build_crisis_followup()`. Verify these work correctly across all 6 crisis scenarios with 3-turn conversations. Add 3 new crisis scenarios that specifically test:
- Turn 2 adaptation (different from turn 1)
- On-site help request detection
- Spanish crisis followup

#### 2E: Profiling Instrumentation (Day 6-7, 3 hours)
Before fixing profiling, measure it:
- Log what `ProfileExtractionOutput` actually returns per turn
- Log whether `profiling_question_injected` reaches the specialist
- Log whether the specialist's response includes the injected question
- If extraction returns all-None or questions are ignored, fix those pipes before tuning

---

### Track 3: Model Selection + Final Validation (Days 7-10) — Calibration

#### 3A: Model Ceiling Test (Day 7, 4 hours)
- Run 50 representative scenarios through Gemini 2.5 Pro
- Same scenarios as Flash baseline from Phase 1B
- 3-model judge panel on both
- Compare dimension-by-dimension
- **Decision gate**: If Pro scores 2+ points higher → behavioral turns use Pro. If within 1 → Flash is sufficient.
- **Cost**: ~$15 API

#### 3B: Post-Fix Full Eval (Day 8, 4 hours)
- Run full 220-scenario eval (all Track 1 + Track 2 fixes applied)
- 3-model judge panel
- Compare against Phase 1B baseline
- **Expected**: B-overall 6.5-7.5 (from 3.2), P-overall first measurement
- **Deliverable**: `r83-judge-scores.json`, `r83-icc-report.md`

#### 3C: Technical Regression Check (Day 8, 1 hour)
- Run full 3305-test suite
- Spot-check 10 property_qa scenarios for fabricated venue names
- Verify crisis hotline numbers unchanged
- Verify gambling advice still blocked
- **Gate**: If any technical regression detected, roll back Track 2 changes and investigate

#### 3D: Human Casino Host Validation (Day 9, 4 hours)
- Send 20 scenario conversations (10 behavioral, 10 profiling) to 3 actual casino hosts
- Simple rubric: "Would you be comfortable if this AI was answering your guests?" + dimension-specific questions
- This is the REAL validation. LLM judges measure consistency. Humans measure quality.
- **Decision gate**: If humans score 7+ on 80%+ of scenarios → product is viable. Below that → structural rework needed.

#### 3E: Targeted Ceiling Push (Day 10, 4 hours)
- Based on 3B scores, identify dimensions still below 8.0
- Write 3 additional few-shot examples per weak dimension
- If model ceiling was hit, implement model routing (Flash for simple, Pro for emotional/crisis)
- Final eval run
- **Target**: B-overall ≥ 8.0, P-overall ≥ 7.0

---

## Honest Targets (Not Aspirational)

| Dimension Group | Current | After Track 1 | After Track 2 | After Track 3 | Ceiling |
|----------------|---------|---------------|---------------|---------------|---------|
| **D1-D10 (Technical)** | 9.63 | 9.63 | 9.63 | 9.63 | 9.8 |
| **B1-B10 (Behavioral)** | 3.2 | 5.5-6.5 | 6.5-7.5 | 7.5-8.5 | 8.5-9.0* |
| **P1-P10 (Profiling)** | Unknown | Unknown (measured) | 4.0-5.0 | 6.0-7.0 | 7.0-8.0* |

*\* 9.5/10 across ALL 30 dimensions is likely unreachable with Gemini Flash alone. The realistic ceiling with Flash + system controls + optimal prompting is ~8.5 behavioral, ~8.0 profiling. Reaching 9.5+ requires either Gemini Pro for behavioral turns (4x cost) or a two-pass system (Flash draft + Pro rewrite for emotional/crisis turns).*

### Why 9.5 Is Honest But Hard

- **D1-D10**: Already there (9.63). Maintenance only.
- **B1-B10 at 9.5**: Requires the agent to read sarcasm, match energy, adapt tone, synthesize multi-domain plans, and handle crisis with human-like empathy — all from a single LLM call. Even Gemini Pro may not reliably do this. The best human casino hosts take years of training.
- **P1-P10 at 9.5**: Requires extracting guest data naturally while appearing conversational, offering incentives at exactly the right moment, and respecting privacy — these are Sales 201 skills. An LLM can be coached to do this at 7-8/10 with good prompting but 9.5 requires fine-tuning or a dedicated profiling model.

### The Honest Answer About "Perfection"

9.5+ on technical: **Already achieved.** Score: 9.63.

9.5+ on behavioral: **Achievable in 2-3 weeks with Flash+Pro routing and post-generation controls.** Score ceiling: ~9.0 with optimal execution. 9.5 requires either model fine-tuning or a breakthrough in prompt compliance.

9.5+ on profiling: **Unknown until measured.** Score ceiling: ~8.0 with current architecture. 9.5 requires dedicated profiling R&D.

**Composite 9.5+ (all 30)**: Requires (a) maintaining technical 9.63, (b) behavioral reaching 9.0+, (c) profiling reaching 9.0+. This is a **4-6 week effort**, not 8 days. The 10-day plan targets 8.0+ behavioral and 7.0+ profiling, which is the honest first milestone.

---

## What This Plan Does NOT Do (Concealment Disclosure)

Following the Heideggerian analysis framework:

1. **Does not address fine-tuning.** Fine-tuning Gemini Flash on Hey Seven's specific voice would likely jump behavioral scores 2-3 points. This is excluded because it requires GCP Vertex AI fine-tuning infrastructure that isn't set up yet. But it's the highest-ceiling approach.

2. **Does not address competitive benchmarking.** We don't know if 3.2 behavioral is terrible or average for AI concierge products. Without competitor baselines, "9.5" is an arbitrary target. The human host validation in 3D partially addresses this.

3. **Does not address voice/multimodal.** Hey Seven's website mentions "autonomous casino host" which implies phone/voice interactions. This plan is text-chat only. Voice adds latency constraints, interruption handling, and tone-of-voice that text evaluation cannot measure.

4. **Assumes the evaluation rubric is correct.** The R72 ICC showed B1 (sarcasm) at 0.348 (poor agreement). If judges can't agree on what good sarcasm handling looks like, optimizing B1 is optimizing noise. The plan should include rubric revision for low-ICC dimensions.

5. **Does not account for Gemini API instability.** The current API key is leaked (403). API keys can be revoked, rate limits can change, model versions can be deprecated. Any plan that depends on continuous API access has execution risk.

---

## Cost Estimate

| Item | Cost |
|------|------|
| Gemini Flash eval (220 scenarios × 3 runs) | ~$15 |
| Gemini Pro eval (50 scenarios × 2 runs) | ~$10 |
| Judge panel (3 models × 3 runs × 220 scenarios) | ~$30 |
| Human casino host review (3 hosts × 20 scenarios) | $0-300* |
| Total API costs | ~$55 |
| Total time | 10-12 days of focused work |

*\* If hosts are Hey Seven team members: $0. If external consultants: ~$100/hour × 3 hours.*

---

## Decision Points (Human Gates)

| After Phase | Decision | Options |
|-------------|----------|---------|
| 1B (Baseline) | Are P-scores below 3.0? | If yes → reprioritize profiling before behavioral |
| 3A (Model test) | Is Pro 2+ points better? | If yes → implement model routing. If no → stay Flash. |
| 3B (Post-fix eval) | Is B-overall ≥ 7.0? | If yes → proceed to polish. If no → investigate which Track 1 control isn't working. |
| 3D (Human review) | Do hosts score 7+? | If yes → product viable. If no → structural rework needed. |
