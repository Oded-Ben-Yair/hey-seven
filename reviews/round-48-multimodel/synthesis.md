# R48 Multi-Model Hostile Review Synthesis

**Date**: 2026-02-24
**Consensus Score**: 71.4/100 (median-based)
**Models**: Gemini 82.0 | DeepSeek 72.0 | GPT-5.2 61.0 | Grok 61.0 (corrected)

## 4 Blind Spot Patterns
1. **Concurrency primitives in async context** — threading.Lock, unguarded task creation, set mutation
2. **Security testing vs implementation gap** — ALL 4 models: auth+classifier disabled in tests
3. **Fix-introduces-new-finding cycle** — R47 classifier degradation became R48 attack surface
4. **Single-instance assumptions in distributed design** — global counters, monotonic time in Redis

## 8 Real Bugs Found
1. Middleware order enables API key brute-force (DeepSeek C1)
2. URL decode limited to 3 iterations (DeepSeek C3)
3. _keep_truthy returns None on None inputs (GPT M1)
4. _ensure_sweep_task race: multiple tasks spawn (DeepSeek M5)
5. Rate limiter bucket orphaned deque (DeepSeek M4)
6. dispatch_method not set in feature-flag path (GPT m2)
7. _active_streams not copied in drain (GPT M7)
8. Greeting cache not invalidated by CMS webhook (DeepSeek M3)

## Top 5 Fixes for Score Impact
1. Fix middleware order — +0.15 weighted
2. URL decode to stable (max 10 iter) — +0.05 weighted
3. E2E tests with auth+classifier — +0.20 weighted
4. Fix sweep race, _active_streams, _keep_truthy — +0.20 weighted
5. Restricted-mode classifier degradation — +0.175 weighted

## Target: ~79/100 after Phase 1 fixes
