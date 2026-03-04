# R85 Handover — Next Session Guide

## Current State (2026-03-03)
- **Commit**: `0fb1e74` (pushed to GitHub main)
- **Tests**: 3503 passed, 0 failed, 1 skipped
- **R85 scores**: B10=6.2, P-avg=5.0, Safety=95%
- **R84 baseline**: B10=6.0, P-avg=2.0, Safety=42%
- **Target**: B10>=8.0, P-avg>=7.0, Safety>=95%

## Score Breakdown by Dimension

### Already at Target (>=8.0)
| Dim | Score | Notes |
|-----|-------|-------|
| B9 safety | 8.6 | Crisis, BSA, RG all working |
| P7 privacy | 8.8 | Strong baseline |

### Close to Target (6.0-7.9) — Quick Fix Territory
| Dim | Score | Gap | Fix Strategy |
|-----|-------|-----|-------------|
| B7 coherence | 7.0 | 1.0 | Multi-turn state already works; may need persona reinforcement at turn 5+ |
| B6 tone | 6.8 | 1.2 | Slop patterns working; check if few-shot examples need tuning |
| B10 overall | 6.2 | 1.8 | Composite — rises when others rise |
| P10 cross-turn | 6.1 | 1.9 | Verify extracted_fields actually persists across turns |
| B2 implicit | 6.0 | 2.0 | R85 signal reading works; need more signal types |
| B3 engagement | 5.9 | 2.1 | Format adaptation works; test with longer conversations |
| B5 emotional | 5.9 | 2.1 | Grief/celebration guides exist; may need more emotional context types |

### Needs Research (4.0-5.9) — Don't Jump to Code
| Dim | Score | Gap | Research Needed |
|-----|-------|-----|----------------|
| B8 cultural | 5.7 | 2.3 | Is the Spanish prompt template working? Are multilingual scenarios getting real responses? |
| B4 agentic | 5.6 | 2.4 | **Deep think**: Stay planner concept — should the agent maintain a "guest evening plan"? Research how Ritz-Carlton/Four Seasons concierges do this |
| P5 sequencing | 5.2 | 2.8 | Is the golden path (foundation→preference→relationship) actually being followed? Log profiling_phase per turn |
| P9 handoff | 4.9 | 3.1 | Are handoff summaries actually being generated? When do they trigger? |
| P4 bridging | 4.6 | 3.4 | Research: what does "assumptive bridging" look like in hospitality? Need worked examples |
| P1 extraction | 4.4 | 3.6 | **Verify**: Is profiling_enrichment_node actually extracting in live eval? Add logging |
| P2 probing | 4.2 | 3.8 | Unconditional follow-up helps; but questions need to be MORE SPECIFIC to context |

### Needs Architecture (below 4.0) — Don't Quick-Fix
| Dim | Score | Gap | Architecture Needed |
|-----|-------|-----|-------------------|
| P3 give-to-get | 3.8 | 4.2 | Agent needs to deliver value FIRST, then ask. Currently asks without earning the right |
| P6 incentive | 3.8 | 4.2 | Incentive engine wired but completeness gate (50%) too high. Lower to 25% AND add incentive scenarios to eval |
| P8 completeness | 3.6 | 4.4 | Downstream of P1-P6 — fix inputs, completeness follows |
| B1 sarcasm | 3.4 | 4.6 | **Measurement problem**: GPT-5.2 marks -1 (not testable), Grok gives ~6. Fix judge rubric first |

## Critical Insights for Next Session

### 1. We've Hit the Prompt Engineering Ceiling
The system prompt now has 15+ injected sections. The LLM can't attend to all simultaneously. Evidence: follow-up questions appear ~60%, not 100%, despite "MUST include" language.

**Next level requires architectural changes:**
- Post-generation enforcement node that ADDS cross-domain suggestions if LLM omitted them
- Profiling question injection node (don't rely on LLM compliance)
- Response quality gate checking for follow-up presence

### 2. Research BEFORE Planning
For each gap dimension, the next session should run a micro-research task:
- `perplexity_research`: "How do luxury hotel concierges handle [dimension X]?"
- `grok_reason`: "Analyze this response — why does it score [Y] on [dimension Z]?"
- Read 5 actual R85 responses per gap dimension and diagnose the SPECIFIC failure

### 3. Profiling Pipeline Observability
We don't know if `profiling_enrichment_node` actually works in live eval. Before fixing P1-P10:
1. Add `logger.info` output to eval script showing extracted_fields per turn
2. Run 5 profiling-specific scenarios (5+ turns, guest volunteers info gradually)
3. Confirm extraction is happening before optimizing what to extract

### 4. Measurement Fixes Before Score Optimization
- B1 sarcasm: Fix judge rubric (force GPT-5.2 to score, or use Grok-only for B1)
- GPT-5.2/Grok calibration: GPT scores 1-3 points lower than Grok consistently. Consider anchoring both to the same calibration examples
- Dedicated profiling scenarios: Current 20-scenario subset tests behavioral. Need 10+ dedicated P1-P10 scenarios with 5+ turns

## Recommended Session Plan

### Phase 0: Micro-Research (30 min, parallel subagents)
- **Research agent 1**: Perplexity — "luxury casino host proactive service patterns"
- **Research agent 2**: Perplexity — "give-to-get profiling techniques in hospitality"
- **Research agent 3**: Grok — analyze 5 lowest-scoring R85 responses, diagnose specific failures
- **Research agent 4**: Read `profiling.py` + `_base.py` — trace profiling data flow end-to-end

### Phase 1: Quick Fixes (20 min)
- Lower incentive completeness gate 50% → 25% (1 line)
- Fix fallback response deflection in specialist agents (3 files)
- Fix B1 sarcasm judge rubric (force scoring)

### Phase 2: Architecture (1-2 hours, team with 3 workers)
Based on research findings:
- **Worker 1**: Post-generation enforcement node (cross-domain + follow-up guarantee)
- **Worker 2**: Profiling pipeline observability + dedicated profiling scenarios
- **Worker 3**: Stay planner concept for B4 agentic (if research supports it)

### Phase 3: Eval + Judge (30 min)
- Run expanded eval (20 behavioral + 10 profiling scenarios = 30 total)
- 2-model judge panel (GPT-5.2 + Grok 4)
- Compare to R85 baseline

## Files to Read First
1. `src/agent/agents/_base.py` — the 15+ injection points (understand the ceiling)
2. `src/agent/profiling.py` — extraction prompt + enrichment node
3. `src/agent/incentives.py` — get_incentive_prompt_section (gate at line 502)
4. `tests/evaluation/r85-subset-responses.json` — actual R85 responses
5. `tests/evaluation/r85-icc-report.md` — full score breakdown
6. `docs/r85-handover.md` — THIS document

## Available Tools for Next Session
| Task | Tool |
|------|------|
| Domain research | `perplexity_research` |
| Response analysis | `grok_reason` / `azure_chat` |
| Code review | `azure_code_review` (GPT-5.3 Codex) |
| Deep reasoning | `azure_deepseek_reason` |
| Multi-model debate | `/multi-model-debate` skill |
| Brainstorming | `/brainstorming` superskill |
| Architecture planning | `architect-planner` agent |
| Parallel implementation | Agent Teams (3 workers max) |
| Visual validation | `gemini-analyze-image` |
| Library docs | `context7` (LangGraph docs) |
