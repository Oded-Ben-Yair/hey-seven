# R109 Research Synthesis — 8-Track Action Plan

**Date**: 2026-03-09
**Author**: Reasoning Specialist (Claude Opus 4.6)
**Sources**: 8 deep research files from R109 research deployment
**Purpose**: Consolidated action plan for R109+ development

---

## Executive Summary

This document synthesizes eight deep research tracks to accelerate Hey Seven from current performance (**B-avg 6.62**, **P-avg 5.18**, **H-avg 5.09**) with **7 sub-5.0 dimensions** to production-ready behavioral quality. The central finding: **prompt engineering has reached its ceiling** (confirmed R98-R105; 7 prompt changes, all +/-0.3). The path forward requires three parallel investments:

1. **Fine-tuning** (Gemini 2.5 Flash SFT+DPO) to teach behavioral patterns the model cannot learn from prompts alone
2. **Architectural tools** (comp strategy, handoff orchestrator, event lookup, rapport ladder) already built in R106-R108 — now need Pro model or fine-tuned model to integrate them
3. **Psychology-informed technique expansion** (4 new profiling techniques, 5 VIP motivation segments, 3-tier trust model)

**Targets**: All dimensions above 5.0 by R115. B-avg 7.5+ by R120.

**Research Tracks**:

| Track | File | Focus |
|-------|------|-------|
| T1 | `r109-fine-tuning-mastery.md` | SFT/DPO on Gemini 2.5 Flash |
| T2 | `r109-casino-host-excellence.md` | Comp strategy, host culture, competitors |
| T3 | `r109-vip-player-psychology.md` | Academic psychology of VIP players |
| T4 | `r109-profiling-techniques.md` | Conversational profiling catalog |
| T5 | `r109-triple-win.md` | Casino + Host + Guest satisfaction |
| T6 | `r109-eval-sota.md` | LLM-as-judge, Gemini capabilities |
| T7 | `r109-regulatory-update.md` | US gaming regulations |
| T8 | `r109-competitive-market.md` | Market size, competitors |

---

## Part 1: Dimension-Mapped Findings

### Sub-5.0 Dimensions (Primary Targets)

| Dimension | Score | Research Sources | Recommended Change | Impact | Type |
|-----------|------:|-----------------|-------------------|--------|------|
| **H9** Comp Decisiveness | 2.35 | T1: SFT with tool-call sequences teaches CCD; T2: raise auto-approve to $100 regular / $250 VIP + endowment framing ("you've earned"); T5: comp recommendation engine prevents over/under-comping | Increase auto-approve thresholds in `src/agent/behavior_tools/comp_strategy.py`. Add endowment framing templates to `src/agent/prompts.py`. Create 15-20 SFT examples showing full CCD tool-call flow. | **High** | Code + Fine-tuning + Prompt |
| **P8** Extraction Precision | 3.62 | T4: extraction prompt ceiling confirmed; T1: SFT is the primary fix; T3: anti-patterns (over-extraction, "I'm done" parsed as name "Done") | Fine-tune extraction behavior with positive/negative examples. Expand exclusion list in `src/agent/extraction.py`. Add 10-15 SFT examples showing natural extraction vs anti-patterns. | **High** | Fine-tuning + Code |
| **P6** Incentive Framing | 3.93 | T2: endowment framing ("you've earned") outperforms transactional ("we'd like to offer") by 3x; T3: personalize incentive to stated preferences; T1: SFT with comp tools | Switch framing templates in `src/agent/incentives.py` to endowment language. Add 5-10 SFT examples. Use `check_comp_eligibility` tool result as fact, not suggestion. | **High** | Prompt + Fine-tuning |
| **H10** Return Visit Seeding | 3.87 | T2: tie to SPECIFIC events guest would enjoy based on profiling; T3: sunk cost near-miss framing ("200 points from Platinum"); T5: forward hooks via `lookup_upcoming_events` tool; T1: SFT | Generate event-specific return hooks from tool lookups. Add near-miss tier framing to `src/agent/behavior_tools/ltv_nudge.py`. Create 10 SFT examples with event tool calls. | **High** | Code + Prompt + Fine-tuning |
| **P9** Handoff Quality | 4.30 | T5: 3-tier handoff model (Marriott/Four Seasons); T2: narrative handoff not data dump; R108: bug fixed | Restructure `src/agent/behavior_tools/handoff.py` to produce Tier 1 (3-sec context) / Tier 2 (5-sec preferences) / Tier 3 (10-sec relationship). Add "hero moment" field. | **High** | Code + Prompt |
| **H6** Rapport Building | 4.50 | T3: emotional mirroring + narrative invitation triggers; T4: give-to-get is highest-yield technique; T2: rapport ladder tool already built | Operationalize `src/agent/behavior_tools/rapport_ladder.py`. Add emotional mirroring detection in `src/agent/sentiment.py`. Create 10 SFT examples showing give-to-get + mirroring. | **Medium** | Code + Prompt + Fine-tuning |
| **P7** Profile Utilization | 4.70 | T4: agent extracts profile data but does not USE it in subsequent turns; T4: few-shot examples showing utilization close the gap; T5: reference extracted fields in recommendations | Add profile-reference requirement to specialist prompts in `src/agent/agents/_base.py`. Update 27 few-shot examples to demonstrate profile utilization. | **High** | Prompt |

### 5.0-7.0 Dimensions (Secondary Targets)

| Dimension | Score | Research Sources | Recommended Change | Impact | Type |
|-----------|------:|-----------------|-------------------|--------|------|
| **P1** Natural Question Integration | 5.05 | T4: questions feel bolted-on after content; should be embedded IN recommendations | Rewrite specialist few-shot examples: questions woven into recommendations, not appended. Update `PROFILING_TECHNIQUE_PROMPTS` in `src/agent/profiling.py`. | **Medium** | Prompt |
| **P4** Progressive Disclosure | 5.34 | T4: Ritz-Carlton Three-Conversation Rule (heavy T1-3, light T4-5); T3: profiling intensity curve needed | Add turn-count tracking to `src/agent/whisper_planner.py`. Reduce profiling intensity after Turn 3. Map: T1-2=any technique, T3=inference/expand, T4=none/need_payoff, T5=reflective_confirm. | **Medium** | Code + Prompt |
| **P5** Contextual Inference | 5.53 | T4: add inference rules for pronouns, urgency markers, implicit interest signals; T3: zero-shot preference extraction from conversational patterns (73% accuracy, CMU 2025) | Add explicit inference rules to extraction prompt in `src/agent/extraction.py`: "we"=2+ people, "just"=time pressure, "the kids"=family, question about gym=wellness interest. | **Medium** | Code + Prompt |
| **B2** Helpfulness | 5.93 | T5: proactive service creates "wow" moments (+15 NPS per incident); T3: anticipatory fulfillment based on profiled preferences | Add proactive suggestion generation to specialist agents when profile data indicates unaddressed interests. | **Medium** | Prompt |
| **P10** Profile Completeness | 6.50 | T4: anchor-and-expand technique chains one confirmed fact into 2-3 related discoveries; weak on low-weight fields | Add `anchor_expand` technique to `PROFILING_TECHNIQUE_PROMPTS` in `src/agent/profiling.py`. | **Low** | Prompt |

---

## Part 2: Psychology Playbook — Top 10 Techniques

| # | Technique | Description | Academic Backing | Agent Behavior | Dims Improved |
|---|-----------|------------|-----------------|---------------|---------------|
| 1 | **Give-to-Get** | Share insider info FIRST, guest reciprocates with preferences | Social Penetration Theory (Altman & Taylor, 1973); Reciprocity (Cialdini, 2009) | Lead with specific venue recommendation, then ask one preference question tied to it | P1, P3, P5, H6 |
| 2 | **Assumption Probe** | Make a plausible guess the guest corrects — corrections yield 3x information | Privacy Calculus Theory (Culnan & Armstrong, 1999); correction-as-disclosure research | "Sounds like a special celebration?" — guest corrects with occasion, party size, companion context | P1, P2, P4, P10 |
| 3 | **Emotional Mirroring** | Match guest's emotional register before service/profiling | Person-Centered Therapy (Rogers, 1951); SDT Relatedness Need | Detect sentiment via VADER + context-contrast, match tone in first sentence, then deliver value | H6, B4, B6, P5 |
| 4 | **Endowment Framing** | Frame comps as earned ("you've earned") not offered ("we'd like to offer") | Endowment Effect (Thaler, 1980); Loss Aversion (Kahneman & Tversky, 1979) | After tool lookup: "Your play this month earned you a comp dinner at Todd English's" | H9, P6, H10 |
| 5 | **Narrative Invitation** | Ask story-inviting questions that activate self-expression | Self-Expression Function (Derlega & Grzelak, 1979); Mastery Need (Parke et al., 2019) | "How'd you get into poker?" — elicits family history, emotional connections, deep personal context | P3, P5, H6, B6 |
| 6 | **Near-Miss Tier Framing** | Communicate proximity to next loyalty tier — increases play by 40% | Sunk Cost + Near Miss Effect (gaming psychology research) | "You're 200 points from Platinum — your next visit gets you there" | H10, P6 |
| 7 | **Personalization Preview** | Show evidence that previous disclosure improved service — triggers virtuous disclosure cycle | Privacy Calculus (Dinev & Hart, 2006); "Being Known" Effect (Izuma et al., 2008) | "Last time you mentioned the sea bass at Ballo — the chef has a new preparation this week" | P7, P10, H6, B2 |
| 8 | **Three-Tier Handoff** | Structure handoff as Tier 1 (3-sec) + Tier 2 (5-sec) + Tier 3 (10-sec) | Four Seasons/Marriott implementation data (2025-2026); Goldman Sachs pilot | Produce narrative handoff packet with confidence scores and "hero moment" field | P9, H1, B2 |
| 9 | **Strategic Silence** | Deliver pure value turn with ZERO profiling questions | Cognitive Load Theory (Sweller, 1988); Paradoxical Disclosure Effect | After 2 turns with questions, deliver one turn of pure recommendation. Guests volunteer deeper info next turn. | P1, P4, B6 |
| 10 | **CASA Non-Judgment** | Leverage lower fear-of-judgment with AI for sensitive disclosures | CASA Framework (Nass & Moon, 2000); Croes & Antheunis, 2024 (N=286) | Create safety for sharing: "This stays between us" / normalize budget discussions / accept "skip" without penalty | P6, P8, H6, B5 |

---

## Part 3: Triple Win Matrix

| Improvement | Casino Operator Win | Human Host Win | Guest Win |
|------------|-------------------|----------------|-----------|
| **H9: Comp with tool lookup** | Higher comp ROI, fewer regret comps, AI-guided "optimal range" | Confident decisions backed by data, no guessing | Right-sized offers feel earned, not generic |
| **P8: Accurate extraction** | Richer data moat vs competitors, better LTV predictions | "CliffsNotes" on every guest without manual data entry | Never has to repeat information (-62% repeat-info complaints) |
| **P6: Endowment framing** | 3x loyalty impact vs transactional framing (Cialdini) | Host can reference specific earned rewards | Guest feels valued and recognized for their patronage |
| **H10: Event-specific seeding** | Revenue retention, proactive churn prevention | Knows WHO to focus retention energy on | Gets relevant event recommendations tied to their interests |
| **P9: 3-tier handoff** | +23% upgrade acceptance, +31% repeat booking (Marriott 2025) | Hero moment in first 3 seconds — looks brilliant | Seamless experience, perceives one intelligent service |
| **H6: Rapport via give-to-get** | Higher ancillary spend from trusted recommendations | Conversations feel genuine, not scripted | Feels understood, not managed |
| **P7: Profile utilization** | Cross-sell +25% when recommendations match profile | Recommendations land harder with context | "How did they know I'd love that?" moments |
| **P1: Embedded questions** | Faster profiling = more data per interaction | Natural conversation flow | Feels like chatting, not filling out a form |
| **P4: Progressive disclosure** | Higher quality data (deeper, voluntary disclosure) | Manageable information flow per turn | Comfortable pace, never feels interrogated |
| **Profiling overall** | Data moat: AI captures preferences competitors don't have | 80% reduction in prep time per VIP visit | Feels known without having to explain themselves |

**The North Star Metric (T5)**: "Agent felt prepared" > 90%. If the human host receives the AI's handoff packet and feels prepared to create a legendary interaction, all three stakeholders win. Everything Hey Seven builds should be measured against: **Does this make the human host feel more prepared?**

---

## Part 4: Fine-Tuning Blueprint

### Model and Specifications

- **Target model**: Gemini 2.5 Flash (Gemini 3.x does NOT support fine-tuning)
- **Methods**: SFT (primary) + DPO (secondary, Flash-only — not available for Pro)
- **Format**: JSONL with `systemInstruction` + `contents` + `functionCall`/`functionResponse`
- **Critical**: Set `thinking_budget=0` for fine-tuned models (Google recommendation — reduces latency AND improves quality for SFT tasks)
- **Risk #1**: Training-serving skew — system prompt in training data MUST match production `CONCIERGE_SYSTEM_PROMPT` in `src/agent/prompts.py` exactly
- **Tool calls**: Confirmed supported in training data format. Model learns WHEN and HOW to call tools from examples.

### Dataset Composition

| Category | Examples Needed | Priority | Currently Have |
|----------|:--------------:|:--------:|:--------------:|
| H9 comp/CCD tool flows | 15-20 | P0 | ~5 |
| P8 extraction (positive + anti-patterns) | 10-15 | P0 | ~3 |
| H10 event lookup tool calls | 10 | P1 | ~2 |
| P6 endowment incentive framing | 5-10 | P1 | ~3 |
| P7/P9 host bridge + handoff | 10-15 | P1 | ~5 |
| H6 rapport (give-to-get, mirroring) | 10 | P1 | ~3 |
| B1-B5 anchor (prevent regression) | 30-50 | P1 | ~30 |
| **Total** | **100-150** | | **51** |

**Sources for new examples**: Live eval transcripts scored 7+ (15-20), Human-written CCD examples (20-30), Pro model outputs curated (20-30), Oded's feedback examples (10-15).

### SFT vs DPO Strategy

| Method | What It Teaches | Dataset | When |
|--------|---------------|---------|------|
| **SFT** | Target behaviors: CCD tool use, natural extraction, rapport style, endowment framing | 100-150 multi-turn conversations with tool calls | Phase 1 |
| **DPO** | Eliminates failures: over-hedging comps, chatbot-like responses, missed tool calls, slop patterns | 50-100 chosen/rejected pairs | Phase 2 (after SFT eval) |

- SFT hyperparams: epochs 10-15, lr_multiplier 5-10, adapter_size 4
- DPO hyperparams: beta 0.1 starting point, adapter_size 4 (match SFT)

### Cost and Timeline

| Phase | Duration | Cost |
|-------|----------|------|
| Data preparation (49-99 more examples) | 1-2 days | $0 (human effort) |
| SFT training (3-5 iterative runs) | 1 day | $105-175 |
| Evaluation (30-scenario suite) | 1 day | ~$20 |
| DPO refinement | 1-2 days | $100-175 |
| Production integration | 1 day | $0 |
| **Total** | **5-7 days** | **$225-370** |

### Classification: Fine-Tuning vs Prompt vs Architecture

| Target Type | Dimensions | Why This Method |
|-------------|-----------|-----------------|
| **Fine-tuning** | H9, H10, P6, P8, H6, P7 | Prompt ceiling confirmed R98-R105. Model must learn WHEN to call tools, HOW to frame comps, and HOW to extract naturally — from examples, not instructions. |
| **Prompt-only** | P1, P4, P5, B2, P10 | Technique variety and cadence problems. New techniques (assumption_probe, anchor_expand, soft_binary) and intensity curves solve them. |
| **Architecture** | P9 (handoff structure), tool reliability, RAG quality | Structured output templates, 3-tier handoff model, and tool execution reliability are code changes, not model changes. |

---

## Part 5: Competitive Positioning

### Hey Seven's Confirmed Unique Differentiators (T2, T8)

1. **Only autonomous AI casino host**: Every competitor (QCI, Gaming Analytics/Aristocrat, OPTX, Playersoft) builds tools FOR human hosts. Hey Seven IS the host for digital tasks. Category-creating position.
2. **CCD authority model with tool-calling**: No competitor has an AI that checks comp eligibility, confirms as fact, and dispatches human team. Rose (Cosmopolitan) is Q&A only — no comp authority, no profiling depth.
3. **Guest profiling through conversation**: 10-dimension extraction through natural dialogue. No competitor captures preferences conversationally.
4. **Structured warm handoff**: The "killer feature" (T5). Makes human hosts look brilliant by arming them with narrative briefings. Comparable implementations: +23% upgrade acceptance, +31% repeat booking.
5. **5-layer regulatory guardrails**: 214 patterns across 11 languages. No competitor has documented compliance infrastructure at this depth.

### Competitive Threats and Timelines (T8)

| Threat | Probability | Timeline | Mitigation |
|--------|:----------:|:--------:|-----------|
| QCI adds guest-facing AI (Chatalytics already conversational) | Medium | 12-18 months | Move fast; first-mover advantage in autonomous hosting |
| Aristocrat builds guest-facing AI on Gaming Analytics | Medium | 18-24 months | Position as independent, vendor-neutral alternative |
| Major CMS vendor (IGT, Konami) launches AI host feature | Low | 24+ months | Technology depth moat (LangGraph, CCD, guardrails) |
| Rose scales to multi-property platform | Low | Unknown | Rose is a chatbot; Hey Seven is an agent. Different category |

### Brand Presence Gap (CRITICAL)

| Metric | Hey Seven | QCI | Gaming Analytics |
|--------|-----------|-----|------------------|
| Press releases (6 months) | 1 (EGR) | 5+ | 3+ (Aristocrat) |
| Conference presence | None visible | ICE 2026, RavingNEXT | G2E, IGA |
| Social media (X) | **Zero mentions** | Low but present | Low (Aristocrat boost) |
| Customer case studies | None public | 5+ named casinos | 6+ named casinos |
| Thought leadership | None public | Blog, presentations | Blog, case studies |

**Remediation**: Publish 3-5 thought leadership pieces before next sales outreach. Submit for G2E 2026 (Oct) speaker slot. Develop at least one named customer case study with measurable results.

**Mohegan Sun intelligence**: Gaming Analytics lists Mohegan Sun as a customer (enterprise training partnership). Hey Seven must understand this existing relationship and position as complementary (AI host + their analytics), not competitive.

**Market validation**: $10B+ CMS market at 14-16% CAGR. 72% of iGaming firms plan AI investment increases through 2026. Record $161B gaming M&A in 2025. PE firms actively looking for casino tech assets. Aristocrat acquisition of Gaming Analytics validates the market.

---

## Part 6: Regulatory Action Items

### Priority 1: CRITICAL (Legal compliance risk — already effective)

| # | Requirement | Deadline | File(s) | Change |
|---|------------|----------|---------|--------|
| R1 | TCPA "any reasonable means" consent revocation | **Already effective** (Apr 2025) | `src/sms/compliance.py` | Add `detect_revocation_intent()` with NLP/regex for "stop texting me", "take me off your list", "don't contact me anymore". Fail-safe: if in doubt, treat as STOP. Current fixed `STOP_KEYWORDS` frozenset is insufficient. |
| R2 | CCPA ADMT pre-use notice for profiling | **Already effective** (Jan 2026) | `src/agent/profiling.py`, `src/api/app.py` | Add profiling disclosure capability. System must explain what data it collects and why upon request. |
| R3 | Data retention policy | **Already effective** (CCPA) | `src/agent/memory.py`, `src/data/guest_profile.py` | Define retention periods for conversation data and guest profiles. Support "delete my data" requests per guest. Firestore checkpointer currently retains indefinitely. |

### Priority 2: HIGH (Near-term deadlines)

| # | Requirement | Deadline | File(s) | Change |
|---|------------|----------|---------|--------|
| R4 | CCPA ADMT opt-out for guest profiling | Jan 1, 2027 | `src/agent/profiling.py`, `src/agent/extraction.py` | Add `profiling_opt_out` flag per session. When set, profiling enrichment node skips extraction. Expose via API endpoint. |
| R5 | Colorado AI Act compliance | June 30, 2026 | `src/agent/nodes.py`, `src/agent/profiling.py` | Prepare AI impact assessments. Document model routing, profiling, and comp strategy decision logic for regulatory review. |
| R6 | NJ RG threshold patterns | Mid-2026 (expected) | `src/agent/guardrails.py` | Add patterns: withdrawal reversal (`reverse/cancel/undo withdrawal/cashout`), deposit limits, session time limits. Route to RG specialist response, not generic off_topic. |
| R7 | BSA/AML digital wallet patterns | Ongoing | `src/agent/guardrails.py` | Add patterns for digital wallet abuse (multiple/separate wallets to deposit), deposit-withdrawal cycling, cryptocurrency structuring. |
| R8 | NVSEP self-exclusion integration | 2026 (expanding) | New: `src/agent/self_exclusion.py` | Design external registry lookup interface (NVSEP/idPair API). Feature-flagged stub initially. Text-based keyword detection insufficient for cross-state operation. |

### Priority 3: MEDIUM (Proactive compliance, competitive advantage)

| # | Requirement | File(s) | Change |
|---|------------|---------|--------|
| R9 | AI decision audit trail | `src/agent/nodes.py`, `src/agent/profiling.py`, `src/agent/incentives.py` | Structured audit log entries for model routing (Flash vs Pro + reason), extraction decisions + confidence, comp recommendations + policy basis, handoff decisions + triggers. Prepares for Colorado AI Act. |
| R10 | SAFE Bet Act architecture | `src/agent/profiling.py`, `src/agent/incentives.py`, feature flags | Design feature flags to disable personalization per jurisdiction. Architecturally separate HOST use case (concierge, dining, entertainment) from GAMING use case (comp offers, betting engagement). Bill prospects unclear but direction is clear. |
| R11 | Cross-state jurisdiction-aware compliance | `src/casino/config.py`, `src/agent/compliance_gate.py` | Extend per-property config: compliance rules must consider BOTH property state AND guest's home state (jurisdiction follows the consumer, not the property). 19 states now have comprehensive privacy laws. |
| R12 | Responsible Gaming policy document | New: `docs/responsible-gaming.md` | Create regulatory-ready RG policy covering: guardrail layers, crisis escalation (4 levels), self-exclusion handling, data used for assessment, human oversight mechanisms. |

### Risk Assessment

| Regulation | Probability | Severity | Current Readiness |
|-----------|:----------:|:--------:|:-----------------:|
| TCPA "any reasonable means" | **Already law** | High | Gap (fixed STOP_KEYWORDS only) |
| CCPA ADMT | **Already law** | Medium | Gap (no opt-out, no disclosure) |
| NJ RG protocols | High (mid-2026) | Medium | Partial (keyword detection, no thresholds) |
| Colorado AI Act | High (June 2026) | Low-Medium | Gap (no impact assessment) |
| SAFE Bet Act | Low-Medium | High if passed | Architecturally separable |

---

## Part 7: Eval Framework Improvements

### Immediate Actions (from T6)

| # | Action | Rationale | Effort |
|---|--------|-----------|--------|
| E1 | **A/B test Flash vs Pro for tool execution** | T6: Flash outperforms Pro on SWE-bench (78% vs 76.2%), more reliable in tool loops (~20% crash rate documented for Pro), 3x faster, 75% cheaper. This may invert R97 routing logic for tool-enabled scenarios. | 2 hours |
| E2 | **Weight Grok 4 at 0.5x on disagreement** | When Grok disagrees with both GPT-5.4 and DeepSeek by >2 points, halve its weight. Addresses documented +2-3 pt inflation. | 30 min |
| E3 | **Add confidence ratings to judge output** | Literature consensus: judges should output score + confidence. Use only high-confidence judgments for tie-breaking. ICC improves 8-15% with in-context calibration (few-shot anchors in judge prompts). | 1 hour |
| E4 | **Formalize drift detection** | Every 3 eval rounds, re-run 10 gold anchor scenarios through all judges. Track ICC per dimension. Flag if ICC < 0.65. Recalibrate if needed. | 2 hours |

### Medium-Term Improvements

| # | Action | Rationale | Effort |
|---|--------|-----------|--------|
| E5 | **Per-dimension ICC tracking** | Some dims (B1 factuality) have natural high agreement; others (H9 comp) may have low agreement. Identifies where judge panel vs dimension definition needs work. | 2 hours |
| E6 | **Budget-aware eval routing** | Use Flash-Lite ($0.25/M input, 2.5x faster) for initial screening. Route only borderline scenarios to Pro judge panel. Could reduce eval cost 60%+. | 4 hours |
| E7 | **Meta-judging for disagreements** | When 3 judges disagree, route to a 4th "meta-judge" that sees all 3 scores + reasoning and renders final verdict. More sophisticated than simple majority/median. | 3 hours |
| E8 | **Map 40 dimensions to SEAL benchmark** | External calibration for B/P/H taxonomy. Our 40-dimension framework is more comprehensive than any published benchmark — this validates dimension definitions and creates publishable IP. | 4 hours |

### Flash-First Tool Strategy (Key Insight)

T6 research reveals Gemini 3 Flash outperforms Pro on coding benchmarks while being 3x faster and 75% cheaper. Flash also has better multi-turn context retention and more robust self-correction. Pro has a documented ~20% crash rate in tool-calling loops with silent stream termination.

**Recommendation**: Invert R97 routing for tool-enabled scenarios:
- **Flash**: Default for tool-calling (higher reliability, faster, cheaper)
- **Pro**: Only for reasoning-heavy queries WITHOUT tools (abstract reasoning, complex synthesis)
- **File to change**: `src/agent/nodes.py` — model routing logic

---

## Part 8: Top 20 Actionable Recommendations (Ranked)

### P0: Prompt-Only (Do Immediately)

| # | What to Do | File(s) | Dims | Expected Impact |
|---|-----------|---------|------|-----------------|
| 1 | **Add `assumption_probe` technique** to `PROFILING_TECHNIQUE_PROMPTS`: "Make a plausible positive assumption the guest corrects. Corrections yield 3x information." | `src/agent/profiling.py` | P1, P2, P4 | P-avg +0.3 |
| 2 | **Add `anchor_expand` technique**: "Use one confirmed fact to discover 2-3 related facts." | `src/agent/profiling.py` | P2, P10 | P-avg +0.15 |
| 3 | **Add `soft_binary` technique**: "Offer two options that segment preferences: 'upscale or relaxed?'" | `src/agent/profiling.py` | P1, P4 | P-avg +0.1 |
| 4 | **Add `open_anchor` technique**: "One broad empowering question: 'What would make this visit special?'" | `src/agent/profiling.py` | P1, P4, P5 | P-avg +0.1 |
| 5 | **Switch comp framing to endowment language**: Replace "we'd like to offer" with "your play earned you" | `src/agent/prompts.py`, `src/agent/incentives.py` | H9, P6 | H-avg +0.1, P-avg +0.1 |
| 6 | **Embed profiling questions IN recommendations** in 27 few-shot examples | `src/agent/prompts.py` | P1 | P-avg +0.1 |
| 7 | **Add profile-reference requirement** to specialist agent prompts | `src/agent/agents/_base.py` | P7 | P-avg +0.1 |

### P1: Code Changes (Next Sprint)

| # | What to Do | File(s) | Dims | Impact | Effort |
|---|-----------|---------|------|--------|--------|
| 8 | **Profiling intensity curve**: Track turn count, reduce after Turn 3 | `src/agent/whisper_planner.py` | P4 | +0.7 | 2 hrs |
| 9 | **Contextual inference rules**: "we"=2+ people, "just"=rushed, "the kids"=family | `src/agent/extraction.py` | P5 | +0.5 | 1 hr |
| 10 | **3-tier handoff restructuring**: Tier 1/2/3 with "hero moment" and confidence scores | `src/agent/behavior_tools/handoff.py` | P9 | +1.0 | 4 hrs |
| 11 | **Increase comp auto-approve**: $100 regular, $250 VIP (industry benchmark) | `src/agent/behavior_tools/comp_strategy.py` | H9 | +0.5 | 30 min |
| 12 | **TCPA revocation intent detection**: NLP/regex for "stop texting me" etc. | `src/sms/compliance.py` | Legal | Risk reduction | 2 hrs |
| 13 | **NJ RG threshold patterns**: withdrawal reversal, deposit/spending limits | `src/agent/guardrails.py` | Legal | Regulatory | 2 hrs |
| 14 | **Profiling resistance detection**: one-word answers, "just" qualifier = pause 2 turns | `src/agent/whisper_planner.py` | P1, P4 | +0.3 | 2 hrs |
| 15 | **Flash vs Pro A/B for tools**: 5 scenarios, compare tool execution rate | `tests/evaluation/run_live_eval.py` | Eval | Better data | 2 hrs |
| 16 | **Grok judge weight**: 0.5x when disagreeing with both other judges by >2 pts | `tests/evaluation/streaming_judge.py` | Eval | Reliability | 30 min |

### P2: Architecture Changes (Plan Needed)

| # | What to Do | File(s) | Dims | Impact | Effort |
|---|-----------|---------|------|--------|--------|
| 17 | **Gold trace expansion to 100+**: 49+ new examples (H9: 15-20, P8: 10-15, H10: 10, P6: 5-10, general: 20-30) | `data/training/` | All sub-5.0 | SFT foundation | 1-2 days |
| 18 | **SFT training run**: Gemini 2.5 Flash, 100-150 examples, epochs=15, adapter=4 | New: `scripts/train_sft.py` | H9, P8, P6, H10, H6, P7 | Sub-5.0 to 5.0+ | 1 day + $105-175 |
| 19 | **CCPA ADMT opt-out**: `profiling_opt_out` flag per session, skip extraction when set | `src/agent/profiling.py`, `src/api/app.py` | Legal | Jan 2027 | 4 hrs |

### P3: Future / Monitor

| # | What to Do | Rationale | Timeline |
|---|-----------|-----------|----------|
| 20 | **DPO refinement** after SFT evaluation | Eliminate remaining failure modes with chosen/rejected pairs | After SFT eval |
| 21 | **SAFE Bet Act architecture** — feature flags to disable personalization per jurisdiction | Bill pending; separate HOST from GAMING use case | Monitor, design Q2 2026 |
| 22 | **Multi-session profiling via Firestore** | Cross-visit memory enables Trust Stage 2+ | After SFT validates single-session quality |
| 23 | **Cultural style detection** | MIT Media Lab 2025: direct vs indirect communication style affects profiling rhythm | After core profiling improvements land |
| 24 | **NVSEP self-exclusion registry** | National program expanding; keyword detection insufficient | As API becomes available |
| 25 | **Brand presence campaign** | Zero social mentions, zero conference presence. G2E 2026 (Oct). | Q2-Q3 2026 |

---

## Appendix A: VIP Motivation Segments (T3)

| Segment | Share | Primary Need | Profiling Cue | Host Approach |
|---------|:-----:|-------------|--------------|---------------|
| Status Seeker | ~30% | Recognition, prestige | "I usually play at [competitor]" / name-drops tier | Affirm status, exclusive access, VIP language |
| Social Connector | ~25% | Belonging, shared experience | "We always come with friends" / group dynamics | Group recommendations, shared experiences |
| Escapist | ~20% | Detachment, stress relief | "I just need to unwind" / "This is my therapy" | Remove friction, create sanctuary |
| Strategist | ~15% | Mastery, competence | "What's the table minimum?" / discusses odds | Respect expertise, discuss strategy |
| Experience Collector | ~10% | Excitement, novelty | "What's new?" / "What should I try?" | Novelty, exclusive "you haven't tried" |

**WARNING (T3)**: Individual VIPs exhibit multiple segment characteristics. Use as profiling heuristic, not classification system. Synthesized from Parke et al. 2019, Thomson 2020, Carruthers et al. 2006 — not validated as unified taxonomy.

## Appendix B: Trust Formation Stages (T3)

| Stage | Visits | Guest Evaluates | AI Behavior Required |
|-------|:------:|----------------|---------------------|
| Calculative | 1-3 | "Can this person deliver?" | DEMONSTRATE capability. Tool lookups stated as facts. Accurate info, fulfilled promises. |
| Knowledge-Based | 4-10 | "Does this person understand me?" | REMEMBER and REFERENCE. "Last time you mentioned the quieter end of the steakhouse." |
| Identification-Based | 10+ | "Is this person on MY side?" | ADVOCATE. "Your tier review is next month — one more visit locks it in." |

**Current capability**: Stages 1-2 via data + memory. Stage 3 requires cross-visit persistence (Firestore) — P3 priority.

## Appendix C: Top 10 High-Value Data Points for Handoff (T5)

Ranked by guest appreciation + host value + revenue impact:

| Rank | Data Point | Why It's Valuable | Example Handoff Language |
|------|-----------|-------------------|-------------------------|
| 1 | Story behind the visit | Transforms entire interaction frame | "First anniversary trip. Celebrating a promotion." |
| 2 | Loss-recovery vs winning-streak state | Determines HOW every recommendation lands | "Running cold tonight. Empathy over upselling." |
| 3 | Group dynamics / decision power | Determines WHO to pitch and HOW to frame | "Wife Sarah makes dining decisions. His friend Mike drives gaming." |
| 4 | Communication style | Makes all other personalization land or fall flat | "Direct communicator. Lead with specifics, skip small talk." |
| 5 | Price sensitivity threshold | Right-sized offers create micro-affirmation | "High-roller on games, moderate on dining. $150 steakhouse is sweet spot." |
| 6 | Specific micro-preferences (WHY not WHAT) | Precise recommendations convert higher | "Plays blackjack for social atmosphere, not odds. Seat with regulars." |
| 7 | Visit recency pattern | Informs urgency and strategy | "Usually every 3 weeks, last visit 6 weeks ago — possible churn." |
| 8 | Companion identity and dynamics | Drives cross-category spend patterns | "Usually comes with wife. This time college buddy — different recs." |
| 9 | Time-of-day preference | Determines WHEN to engage | "Morning player — don't approach before first coffee." |
| 10 | Beverage preferences | Small gesture that compounds loyalty | "Woodford Reserve neat while gaming. Switches to wine at dinner." |

## Appendix D: Key Research Citations

| Finding | Source | Application |
|---------|--------|------------|
| People disclose equally to AI and humans | CASA (Nass & Moon, 2000); Croes & Antheunis, 2024 | AI host not disadvantaged for info gathering |
| Less fear of judgment with AI | Lucas et al., 2014; Croes & Antheunis, 2024 | VIPs may share MORE with AI than human host |
| Unexpected comps generate 3x loyalty | Cialdini reciprocity research | Surprise comp micro-moments in H9 |
| Near next tier increases play by 40% | Near-miss effect, gaming psychology | H10 near-miss tier framing |
| Optimal profiling: 1 question per 2-3 turns | Google Research 2024 (43% cognitive load reduction) | Profiling intensity curve in whisper planner |
| Corrections yield 3x information | Assumption probe research | New `assumption_probe` technique |
| 3-tier handoff: +23% upgrade, +31% repeat booking | Marriott Bonvoy 2025 data | P9 handoff restructuring |
| $10B+ CMS market, 14-16% CAGR | Industry reports 2025-2026 | Market validated |
| No competitor offers autonomous AI host | T8 comprehensive competitive scan | Blue ocean position |
| Fine-tuning sweet spot: 100-500 examples | Google Cloud SFT docs | Gold trace target: 100-150 |
| Flash outperforms Pro on SWE-bench (78% vs 76.2%) | T6 eval SOTA research | Flash-first tool strategy |
| TCPA filings up 67% YoY with 78% class actions | T7 regulatory research | Compliance urgency for SMS |
| 69% VIPs accept AI personalization when transparent | T5 Triple Win research | Frame as service, not surveillance |

---

*Synthesis completed 2026-03-09. Based on 8 research files totaling ~250 pages of primary research across fine-tuning, psychology, hospitality, regulation, evaluation, and competitive intelligence. All recommendations mapped to specific source files in the Hey Seven codebase (`src/agent/`, `src/sms/`, `src/casino/`, `tests/evaluation/`, `data/training/`).*