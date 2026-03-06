# Phase 1+2 Design: Pro-First, Distill-Down

**Date**: 2026-03-05
**Status**: Approved
**Goal**: B-avg >= 8.0, all P-dims >= 5.0, all H-dims >= 5.0

## Wave 1: Pro Switch Canary
- Enable model_routing_enabled per-casino (code done in R97)
- Run 20 behavioral + 20 profiling scenarios with Pro
- Judge with GPT-5.2, compare to Flash baseline
- Gate: B-avg >= 7.0

## Wave 2: 4 Structured Tools
1. CompStrategy (H9:1.9→5.0) — deterministic comp policy engine
2. HandoffOrchestrator (P9:2.1→5.0) — structured host handoff summary
3. LTV Nudge (H10:3.5→5.0) — return-visit seeding
4. Rapport Ladder (H6:4.0→5.0) — micro-pattern retrieval

## Wave 3: Integration Eval
- Full 165-scenario eval across all 3 categories
- All weak dims >= 5.0
- Max 2 iteration rounds
