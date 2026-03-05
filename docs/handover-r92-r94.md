# R92-R94 Handover — Booking Pipeline + Profile + Emotional Intelligence

## Session Summary
**Date**: 2026-03-05
**Commit**: 6ad57db
**Rounds**: R92 (structural fix) + R93 (profiling) + R94 (emotional intelligence)
**Tests**: 3551 passed, 3 pre-existing failures, 90%+ coverage

## What Changed

### R92 — The Structural Fix (Highest Leverage)
**Root cause fixed**: `route_from_router()` was sending ALL `action_request` queries to `off_topic_node` where they got canned templates. Now they flow through the full specialist pipeline (retrieve → whisper → pre_extract → generate → profiling → validate → persona → respond).

Key changes:
- `action_request` removed from off_topic routing tuple in `route_from_router()`
- Repeat detection preserved: if previous AI response contained "host team"/"reservations team", still routes to off_topic for handoff
- `booking_intent` state field (29 fields total) signals specialists to inject qualifying questions
- `MODEL_MAX_OUTPUT_TOKENS` 2048→4096 eliminates truncation
- Closed-conversation detection in greeting_node
- Anti-deflection: "NEVER say 'I can't make reservations'"
- Slop detector message ID fix for proper replacement

### R93 — Profile Confirmation + Active Probing
- Profile confirmation in `profiling_enrichment_node`: when guest confirms and profile ≥30% complete, prepends "So I've got: [name, occasion, party]"
- Recommendation→question micro-flow in booking context
- Profile-aware farewell in greeting_node

### R94 — Emotional Intelligence + VIP
- Loss recovery: empathy-first, no immediate upsell after "$5K loss"
- Disappointment: subtle detection (not frustration), acknowledge then offer alternative
- VIP recognition: specific comp mechanics, tier checks, host introduction (not generic "valued guest")
- 2 new few-shot examples (27 total)
- "disappointed" added to Pro model routing

## Files Modified (14 files, +420/-86 lines)

| File | Changes |
|------|---------|
| `src/agent/nodes.py` | Route action_request to retrieve, greeting closer, model routing |
| `src/agent/state.py` | Add `booking_intent: str \| None` (29 fields) |
| `src/agent/graph.py` | `_initial_state()` parity |
| `src/agent/agents/_base.py` | Booking context, anti-deflection, loss/VIP/disappointment signals |
| `src/agent/profiling.py` | Profile confirmation injection |
| `src/agent/prompts.py` | 2 VIP/loss few-shot examples |
| `src/config.py` | MODEL_MAX_OUTPUT_TOKENS 2048→4096 |
| `tests/test_nodes.py` | Updated routing + closed conversation tests |
| `tests/test_r26_conversation.py` | Updated routing test |
| `tests/test_eval_deterministic.py` | booking_refusal→booking_request |
| `tests/test_eval.py` | Updated booking test |
| `tests/test_doc_accuracy.py` | 28→29 state fields |
| `tests/test_few_shot_examples.py` | 25→27 examples |
| `CLAUDE.md` | Updated architecture overview |

## Eval Results

### R92-pre (before R93-R94 anti-deflection)
- 65 scenarios, 199 turns, 0 errors, 100% response rate
- 7 deflections (3.5%) — "While I can't make reservations" still in some responses

### R93-R94 fresh eval (in progress)
- 7+ scenarios completed, 0 deflections (0%) — anti-deflection working
- Response quality: venue-specific, hours, cross-domain suggestions
- Eval still running (RPM 15 = ~2 min/scenario)

### GPT-5.2 Judge Panel (R92-pre, 7 full-text scenarios)
| Dimension | Score | vs R91 |
|-----------|-------|--------|
| B1 Knowledge | 6.7 | = |
| B2 Implicit | 6.1 | -0.6 |
| B3 Engagement | 6.3 | -0.4 |
| B4 Agentic | 5.9 | -2.4 |
| B5 Safety | 7.3 | +0.6 |
| P2 Probing | 3.7 | -0.3 |
| P8 Extraction | 2.4 | -1.6 |
| P9 Confirmation | 3.0 | +1.0 |
| H7 VIP | 1.7 | -4.0 |
| H10 Booking | 3.0 | -1.7 |

**Note**: These scores are from R92-pre eval BEFORE R93-R94 changes. The R93 profile confirmation and R94 VIP mechanics are NOT reflected in these scores. Fresh eval with all changes is running and showing 0% deflection — scores should improve on P9, H7, H10.

## R95 Next Steps
1. Wait for fresh eval to complete (~250 scenarios)
2. Run 3-model judge panel on fresh eval results
3. Max 2-3 regression fixes if safety drops
4. Final documentation + handover
