# Eval v2: Real-Time Streaming Evaluation System

> **Status**: Designed (2026-03-03). Ready to implement.
> **Motivation**: Current eval takes 2+ hours, writes nothing until done, can't resume on crash, and blocks the judge panel.

## Problem

| Issue | Impact |
|-------|--------|
| Atomic write at end | Zero visibility for 2+ hours |
| No resume | Crash at scenario 80 = start over |
| Sequential judge | Must wait for ALL responses before ANY judging |
| No re-run subset | Fix 1 bug → re-run all 109 scenarios |
| Preview model rate limits | 10 RPM free tier → 2h for 327 LLM calls |

## Solution: Directory-per-scenario + Watch-mode Judge

### Directory Layout

```
tests/evaluation/runs/
  CURRENT                                    # Active run ID
  r83-20260303-141500-7a97e03/
    meta.json                                # Run metadata
    summary.json                             # Live dashboard stats
    scenarios/
      sarcasm-01.json                        # Written atomically on completion
      crisis-01.json
      ...
    judge/
      gpt52/sarcasm-01.json                  # Per-model, per-scenario
      grok4/sarcasm-01.json
      consensus/sarcasm-01.json              # Multi-judge merge
```

### Key Decisions

1. **Directory-per-scenario** over JSONL/SQLite — filesystem IS the state machine, no locking needed
2. **Atomic writes** via tmp+rename — readers never see partial JSON
3. **Resume = set difference** — completed scenarios = `ls scenarios/*.json`
4. **Concurrent eval + judge** — eval writes to `scenarios/`, judge reads `scenarios/` and writes to `judge/` (zero contention)
5. **Rate-limit aware** — use `gemini-2.5-flash` (300 RPM) for bulk, 3.x for targeted re-eval
6. **Concurrency 4** — bounded by asyncio.Semaphore, conservative for in-memory checkpointer

### CLI

```bash
eval run   --round r83 --pattern "behavioral_*.yaml"   # Stream results
eval judge --watch                                       # Judge in parallel
eval dashboard                                           # Live stats
eval rerun --bottom 20                                   # Fix loop
eval compare --base <run1> --target <run2>               # Before/after
eval export --format legacy                              # Backward compat
```

### Time Savings

| Scenario | Current | v2 |
|----------|---------|-----|
| Full 109-scenario eval (2.5-flash) | 20 min (no visibility) | 4 min (live dashboard) |
| Full 109-scenario eval (3-flash-preview) | 2+ hours | Same, but with live results |
| Judge panel start | After eval finishes | 30s after first scenario completes |
| Fix loop (re-run 20 worst) | 2+ hours (full re-run) | 4 min (subset only) |
| Crash recovery | Start over | Resume from last completed |

### Files to Create (~930 LOC)

| File | Purpose |
|------|---------|
| `cli.py` | Click CLI with subcommands |
| `runner.py` | Async streaming runner (replaces `run_live_eval.py`) |
| `judge.py` | Watch-mode judge (replaces `run_judge_panel.py`) |
| `storage.py` | Atomic writes, run management |
| `dashboard.py` | Live stats to stdout |
| `compare.py` | Before/after score deltas |
| `rate_limiter.py` | Async token-bucket |
| `schemas.py` | Shared constants |
| `export.py` | Legacy format export |
