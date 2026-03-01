# 30-Dimension Perfection Sprint — Session Handover

**Date**: 2026-03-01
**Commit**: `88f4924` (v1.4.0)
**Goal**: Push ALL 30 dimensions (D1-D10, B1-B10, P1-P10) toward 9.0+/10 through iterative live evaluation with 4-model review panel.

---

## The 30 Dimensions

### Technical (D1-D10) — Last scored R75: weighted 9.63/10
| Dim | Name | Last Score | Gap to 10 | Fix Strategy |
|-----|------|-----------|-----------|--------------|
| D1 | Graph Architecture | 9.5 | SRP completion of execute_specialist (was 195 LOC, refactored to ~120). profiling_enrichment node is SRP-clean. | Verify SRP <100 LOC per function in hot path |
| D2 | RAG Pipeline | 9.5 | Semantic reranker after RRF. Embedding version enforcement at ingestion time. | Low ROI — skip unless reviewer flags |
| D3 | Data Model | 9.5 | UNSET_SENTINEL string-based (UUID-prefixed, collision astronomical). guest_name has _keep_latest_str reducer now. | Likely at ceiling |
| D4 | API Design | 10.0 | AT CEILING | No action needed |
| D5 | Testing | 10.0 | AT CEILING (3236 tests, 0 failures) | Just verify count holds after fixes |
| D6 | Docker & DevOps | 9.5 | AT CEILING | No action needed |
| D7 | Guardrails | 9.5 | 1 pattern uses lookahead (intentional, documented). VADER limitations. | Consider adding sentiment_llm_augmented note to ADR |
| D8 | Scalability | 9.5 | No load testing evidence. Fixed semaphore. | Add ADR noting Cloud Run --concurrency=50 as capacity gate |
| D9 | Documentation | 10.0 | AT CEILING (28 ADRs) | Keep parity after fixes |
| D10 | Domain | 9.5 | Hours as strings. Only Mohegan Sun data. | Parse hours for "open now" — already implemented in hours.py |

### Behavioral (B1-B10) — Last scored R75: 5.8/10 (LIVE)
**THIS IS WHERE THE BIGGEST GAINS ARE. Live behavioral is the bottleneck.**

| Dim | Name | Last Live | Gap to 9 | Fix Strategy |
|-----|------|----------|----------|--------------|
| B1 | Sarcasm Detection | 8.0 | VADER + context-contrast at ceiling. | LLM-augmented sentiment already wired (sentiment_llm_augmented=True). Verify it fires in live. |
| B2 | Implicit Signal | 5.0 | Regex extraction misses paraphrases. | LLM extraction augmented wired. Verify profiling node captures what regex misses. |
| B3 | Engagement | 5.0 | Generic responses. Cross-domain hints exist but may not fire. | Verify _build_cross_domain_hint() fires in live. Add more varied conversation steering in prompt. |
| B4 | Proactive Suggestion | 5.0 | Suggestion infrastructure wired but may not fire in live. | Verify _should_inject_suggestion() gates are not too strict. Test with positive-sentiment scenarios. |
| B5 | Escalation | 4.0 | Frustration escalation exists but handoff may be too late. | Verify HEART framework fires at frustrated_count >= 2. Lower threshold if needed. |
| B6 | Persona Consistency | 6.0 | Persona reinject at 5 turns. Exclamation clamping exists. | Prompt tone is the #1 lever. "Warmth from SUBSTANCE" already in prompt. Verify no enthusiasm drift. |
| B7 | Memory & Continuity | 9.0 | AT CEILING for single-session. | Verify profiling state persists across turns via checkpointer. |
| B8 | Cultural/Multilingual | 7.0 | EN/ES prompts exist. Spanish support enabled. | Verify Spanish scenarios route correctly through live agent. |
| B9 | Safety & Compliance | 3.0 | Crisis detection works but live agent may not route correctly. | Verify compliance_gate → off_topic → self_harm path works in live. Check grief detection. |
| B10 | Overall Quality | 6.0 | Composite of all above. | Rises naturally as individual dims improve. |

### Profiling (P1-P10) — NEVER SCORED (new system, needs baseline)
| Dim | Name | Expected Baseline | Key Risk |
|-----|------|------------------|----------|
| P1 | Natural Extraction | 6-7 | Regex + LLM augmented extraction. Should capture name, party, occasion. |
| P2 | Active Probing | 4-5 | Depends on whisper planner generating next_profiling_question. New — untested live. |
| P3 | Give-to-Get | 5-6 | Prompt instructs "give before get" but LLM compliance is unpredictable. |
| P4 | Assumptive Bridge | 3-4 | Technique exists in prompt but may not fire naturally. |
| P5 | Progressive Sequence | 5-6 | Golden path logic in _determine_profiling_phase(). Whisper planner should respect. |
| P6 | Incentive Framing | 3-4 | Incentive engine exists but not wired into live prompt path yet (get_incentive_prompt_section in _base.py needs verification). |
| P7 | Privacy Respect | 6-7 | Prompt says "explain WHY" but LLM may not consistently. |
| P8 | Profile Completeness | 4-5 | Depends on extraction + probing working together. |
| P9 | Host Handoff | 5-6 | get_guest_profile_summary() exists. HandoffRequest model exists. |
| P10 | Cross-Turn Memory | 6-7 | extracted_fields persists via _merge_dicts reducer. Should work if extraction fires. |

---

## Iteration Strategy: The 5-Round Protocol

### Round 1: BASELINE (evaluate only, no fixes)
**Goal**: Get honest scores across all 30 dimensions from 4 models.

**Steps**:
1. Run the live agent against ALL scenario files:
   ```bash
   # Behavioral (existing 74 scenarios)
   python3 tests/evaluation/run_eval.py --scenarios tests/scenarios/behavioral_*.yaml
   # Profiling (new 56 scenarios)
   python3 tests/evaluation/run_eval.py --scenarios tests/scenarios/profiling_*.yaml
   ```
   If no eval runner exists, build one: send each scenario turn to the live agent via `/chat` API, collect responses.

2. Judge with 4 models in parallel (use MCP tools directly, NOT subagents):
   - `gemini-query` (thinking=high) — D1/D2/D3, P1/P5/P8
   - `azure_chat` (GPT-5.2) — D4/D5/D6, B1/B2/B3
   - `grok_reason` — D7/D8/D9, B4/B5/B6
   - `azure_deepseek_reason` — D10, B7/B8/B9/B10, P2/P3/P4/P6/P7/P9/P10

3. Synthesize: consensus findings (2+ models agree) are REAL. Record all 30 scores.

4. Identify the 5 lowest-scoring dimensions — these are Round 2 targets.

### Round 2: FIX LOWEST 5 (code changes + prompt tuning)
**Goal**: Fix the 5 worst dimensions. Expected yield: +2-4 points per dimension.

**Steps**:
1. For each low dimension, diagnose root cause:
   - If score < 5: likely a WIRING issue (code exists but doesn't fire in live)
   - If score 5-7: likely a PROMPT issue (code fires but LLM doesn't comply)
   - If score 7-8: likely an EDGE CASE issue (works for happy path, fails on tricky scenarios)

2. Apply fixes (use subagents for parallel code changes):
   - Wiring fixes: trace the code path, add logging, verify it fires
   - Prompt fixes: adjust CONCIERGE_SYSTEM_PROMPT or WHISPER_PLANNER_PROMPT
   - Edge case fixes: add new regex patterns, expand extraction, add scenarios

3. Run ONLY the fixed dimensions' scenarios through live agent.
4. Judge ONLY the fixed dimensions with 2 models (quick validation).

### Round 3: FIX NEXT 5 (repeat for dimensions 6-10)
Same protocol as Round 2 but for the next tier of low-scoring dimensions.

### Round 4: POLISH (prompt tuning for 8+ dimensions)
**Goal**: Push dimensions from 8 to 9+. This is prompt engineering territory.

**Steps**:
1. For each 8-scoring dimension, analyze the live responses that scored 7-8.
2. Identify specific prompt language that would improve responses.
3. Apply prompt changes (small, surgical edits to system prompts).
4. Re-evaluate affected scenarios only.

### Round 5: FINAL CONSENSUS (full 4-model panel, all 30 dims)
**Goal**: Final scores. All 30 dimensions. 4 models. ICC calculation.

---

## Agent Architecture for Each Round

### Evaluation Agent (for Rounds 1 and 5)
```
Team: "perfection-eval-r{N}"
  - eval-runner: Runs scenarios through live agent, collects responses
  - judge-alpha: Scores D1-D5 + P1-P5 (Gemini + GPT-5.2)
  - judge-beta: Scores D6-D10 + P6-P10 (Grok + DeepSeek)
  - judge-gamma: Scores B1-B10 (all 4 models for behavioral — highest variance)
  - synthesizer: Merges scores, calculates ICC, identifies lowest 5
```

### Fix Agent (for Rounds 2-4)
```
Team: "perfection-fix-r{N}"
  - diagnostician: Reads eval results, traces code paths, identifies root causes
  - prompt-tuner: Modifies CONCIERGE_SYSTEM_PROMPT, WHISPER_PLANNER_PROMPT (owns prompts.py)
  - code-fixer: Fixes wiring issues in graph.py, _base.py, profiling.py (owns src/agent/)
  - validator: Re-runs affected scenarios, verifies fixes improved scores
```

---

## Critical Lessons (from 75 prior rounds)

### DO
- Run live agent (Gemini Flash) — never trust mock scores
- Use MCP tools directly from main session for judging (subagents exhaust context)
- Batch 20 scenarios per judge call (efficient, proven in R72-R75)
- Validate reviewer findings against actual code before accepting
- Fix prompts before code — prompt changes are cheaper and often higher impact
- Check `_should_inject_suggestion()` gates — they may be too strict for the live agent
- Verify feature flags are True in DEFAULT_FEATURES for all new features

### DON'T
- Don't trust internal scoring at ceiling (R47: internal 96.7, external 65)
- Don't use mock LLM for behavioral scoring (43% overestimate proven)
- Don't use forced-finding quotas ("minimum 5 findings" — retired per ADR-023)
- Don't use hostile framing in eval prompts
- Don't attempt all 30 dimensions in one fix round (max 5 per round)
- Don't use `Explore` subagent type (hardcoded to haiku)

---

## Eval Prompts

### Technical (D1-D10): `docs/eval-prompt-v2.0.md`
- Frozen, version-controlled
- All 4 models score ALL 10 dimensions
- No spotlight, no severity bumps

### Behavioral (B1-B10): Use live agent responses + judge prompt
- Scenarios in `tests/scenarios/behavioral_*.yaml`
- Judge with batch prompt (20 scenarios per call)
- Score each response on relevant B-dimensions

### Profiling (P1-P10): `tests/evaluation/profiling-eval-prompt.md`
- New, frozen at v1.0
- 56 scenarios across 7 files
- Calibration anchors for 3/6/9 per dimension

---

## File Map for Fixes

| To improve | Edit these files | Key functions |
|------------|-----------------|---------------|
| B1 Sarcasm | prompts.py, sentiment.py | detect_sarcasm_context(), emotional context guides |
| B2 Signals | extraction.py, profiling.py | extract_fields(), profiling_enrichment_node() |
| B3 Engagement | prompts.py, _base.py | _build_cross_domain_hint(), conversation dynamics |
| B4 Proactive | _base.py | _should_inject_suggestion() — check 5 gates |
| B5 Escalation | _base.py, crisis.py | _count_consecutive_frustrated(), HEART framework |
| B6 Persona | prompts.py | CONCIERGE_SYSTEM_PROMPT tone section |
| B7 Memory | state.py, graph.py | extracted_fields reducer, _merge_dicts |
| B8 Cultural | prompts.py, nodes.py | CONCIERGE_SYSTEM_PROMPT_ES, greeting_node Spanish |
| B9 Safety | compliance_gate.py, nodes.py | Crisis routing, self_harm handling |
| B10 Overall | All of the above | Composite — rises with individual dims |
| P1-P10 | profiling.py, whisper_planner.py, prompts.py, _base.py | profiling_enrichment_node, WhisperPlan, system prompt |
| D1-D10 | Already at 9.5+ — mostly docs/ADR fixes | Parity tests catch drift |

---

## Success Criteria

| Tier | Criteria | Action |
|------|----------|--------|
| **Gold** | All 30 dims >= 9.0 | Declare perfection. Ship it. |
| **Silver** | 25+ dims >= 9.0, none below 7.0 | Document remaining gaps in ADRs |
| **Bronze** | 20+ dims >= 8.0, none below 5.0 | Focus next session on remaining dims |
| **Needs work** | Any dim below 5.0 | Critical wiring issue — must fix before next eval |

---

## Quick Start for Next Session

```
1. Read this document
2. Read MEMORY.md for full context
3. Run: python3 -m pytest tests/ -q --no-cov  (verify 3236 pass)
4. Start Round 1 baseline evaluation
5. Follow the 5-round protocol above
```
