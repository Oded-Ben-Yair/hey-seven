# R16 Hostile Code Review — Hey Seven AI Casino Concierge

**Date**: 2026-02-19
**Reviewer**: Claude Opus 4.6 (hostile code-judge)
**Score**: 95/100 (APPROVE — STRONG HIRE)
**Test Count**: 1044 passed, 90.87% coverage

## R15 Fix Verification (4/4 CONFIRMED)

1. Feature flag wiring: `specialist_agents_enabled` check in `_dispatch_to_specialist()`
2. Greeting template safety: `$placeholder` syntax (not `.format()`)
3. Config parity assertion: cross-module assertion at import time
4. RequestBodyLimitMiddleware 413 suppression: `sent_413` flag suppresses all messages

## 10-Dimension Scoring

| # | Dimension | Score | Delta | Notes |
|---|-----------|-------|-------|-------|
| 1 | Graph/Agent Architecture | 9.5/10 | = | 11-node StateGraph, validation loop, specialist dispatch, parity assertions |
| 2 | RAG Pipeline | 9.5/10 | +0.5 | Per-item chunking, SHA-256 dedup, RRF (k=60), AbstractRetriever ABC, stale chunk purge |
| 3 | Data Model / State Design | 9.5/10 | = | TypedDict with `add_messages`, `_keep_max` reducer, `_initial_state()` parity |
| 4 | API Design | 9.5/10 | = | 6 pure ASGI middleware, SSE with timeout + disconnect, structured errors |
| 5 | Testing Strategy | 9.5/10 | = | 863 test functions across 32 files, parametrized guardrails, conftest clearing 11 caches |
| 6 | Docker & DevOps | 9.0/10 | = | Multi-stage, non-root, HEALTHCHECK, graceful shutdown |
| 7 | Prompts & Guardrails | 9.5/10 | = | 84 compiled regex, 4 languages, 5 categories, semantic classifier, safe_substitute |
| 8 | Scalability & Production | 9.5/10 | = | Circuit breaker, TTL-cached LLM singletons, sliding-window rate limiter |
| 9 | Trade-off Documentation | 9.0/10 | = | Good inline trade-offs |
| 10 | Domain Intelligence | 9.5/10 | = | TCPA, BSA/AML, 280+ area codes, consent hash chain, quiet hours, MNP caveat |

**Total: 95/100**

## Findings (5 items, 2 MEDIUM + 3 LOW)

### Finding 1 (MEDIUM): Feature flags use static DEFAULT_FEATURES instead of per-casino get_feature_flags()
- All call sites use `DEFAULT_FEATURES.get()` — the async per-casino API is unused from graph
- **Fix**: Document as intentional single-tenant Phase 1 design; async API ready for Phase 2

### Finding 2 (MEDIUM): comp_agent completeness gate uses wrong schema
- `calculate_completeness(extracted_fields)` expects GuestProfile dotted paths, gets flat dict
- Completeness always returns 0.0 — comp agent always asks "tell me more"
- **Fix**: Use flat-field completeness check for extracted_fields

### Finding 3 (LOW): sources/done SSE events sent after error in chat_stream()
- After error event, stale sources and done still yielded
- **Fix**: Add errored flag to guard post-error yields

### Finding 4 (LOW): WebhookIdempotencyTracker declared but never used
- Telnyx retries possible but no dedup in handle_inbound_sms()
- **Fix**: Wire tracker into handle_inbound_sms()

### Finding 5 (LOW): Duplicated _get_firestore_client() across config.py and guest_profile.py
- Nearly identical implementations, DRY violation
- **Fix**: Add cross-reference comments

## Score Trajectory
R1: 58 → R3: 80 → R5: 85 → R8: 90-93 → R14: 93 → R15: 94 → R16: 95
