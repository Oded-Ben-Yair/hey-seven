# Round 8 Batch 2: FIX-8 through FIX-18 Applied

**Date**: 2026-02-13
**Target**: assignment/architecture.md
**Fixes applied**: 11 (FIX-8 through FIX-18)
**Priority**: 2 (HIGH)

## Fix Summary

| Fix | Description | Lines Affected | Status |
|-----|-------------|----------------|--------|
| FIX-8 | Added SecurityHeadersMiddleware (X-Content-Type-Options: nosniff, X-Frame-Options: DENY, HSTS note) | After CORS section (~1962) | APPLIED |
| FIX-9 | Propagated request_id through LangGraph config["configurable"] and extracted in node logging | ~1753 (config), ~2806 (monitoring) | APPLIED |
| FIX-10 | Added structlog.configure() with JSONRenderer, TimeStamper, log level filtering | ~2779 | APPLIED |
| FIX-11 | respond and fallback nodes now return retry_feedback: None to clear stale feedback across turns; updated state comment | ~629, ~654, ~794 | APPLIED |
| FIX-12 | Router confidence now used: stored in state as router_confidence, low confidence (<0.6) reroutes to safe RAG path | ~251, ~286, ~664, ~793 | APPLIED |
| FIX-13 | Expanded validate guard to short-circuit on retry_count >= 99 (covers both empty-context PASS and LLM-exception FAIL) | ~479 | APPLIED |
| FIX-14 | test_router_gambling expected category changed from off_topic to gambling_advice (with routing note) | ~2112 | APPLIED |
| FIX-15 | SAFE Bet Act: "online gambling" corrected to "sports betting" | ~3310 | APPLIED |
| FIX-16 | "Source citations" changed to "source tracking" in Key Differentiators and respond node heading; clarified as metadata, not inline citations | ~48, ~186, ~614 | APPLIED |
| FIX-17 | Rate limiter rewritten as pure ASGI middleware (not @app.middleware("http") which buffers SSE) | ~2880 | APPLIED |
| FIX-18 | Added agent None guard before streaming (returns 503 JSONResponse if agent not initialized) | ~1740 | APPLIED |

## Notes

- Worked bottom-up to minimize line-shift conflicts with the parallel batch1 agent (FIX-1 through FIX-7).
- FIX-1/FIX-2 were already applied by the other agent when this batch started (confirmed on re-read).
- FIX-12 implementation chose the "use it" path: low confidence routes to safe RAG path rather than removing the field.
- FIX-16 preserved internal code comments referencing "source citations" in metadata context (lines 885, 967) since those describe the mechanism accurately.
- CORS expose_headers updated to include X-Request-ID as part of FIX-8 security headers work.
