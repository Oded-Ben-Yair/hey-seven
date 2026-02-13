# Round 6 Hostile Review Summary

**Date**: 2026-02-12
**Reviewers**: 5 parallel code-judge agents (2 dimensions each)
**Architecture Version**: 6.0 → 7.0 (all fixes applied)

## Scores (Pre-Fix)

| Dimension | Score | Key Issues |
|-----------|-------|------------|
| Graph Architecture | 8.0/10 | Diagram showed greeting→respond (wrong), retry polluted messages, no empty-context guard |
| RAG Pipeline | 8.5/10 | Cosine distance range wrong [0,1] should be [0,2], validate_items not wired |
| Data Model | 7.5/10 | last_updated as string, no source validation, no schema_version, price_range untyped |
| API Design | 8.0/10 | Rate limiter not async-safe, circuit breaker not integrated, no response model for /property/info |
| Testing Strategy | 7.5/10 | No API auth tests, no startup failure tests, no compound adversarial tests, lettered test numbers |
| Docker & DevOps | 8.0/10 | nginx.conf in .dockerignore missing, no .env.example content, port 8080 undocumented |
| Prompts & Guardrails | 7.5/10 | No player psychology in prompt, keyword gambling detection fragile, no "other properties" rule |
| Scalability & Production | 8.0/10 | LangSmith section thin, no connection pooling, rate limiter scaling not called out |
| Trade-off Documentation | 8.0/10 | Decisions 5&6 are strawmen, no evidence qualifiers, no self-doubt |
| Domain Intelligence | 8.5/10 | No Hey Seven Pulse connection, Appendix C too thin, prompt lacks hospitality personality |

## Fixes Applied (40+ edits)

### Batch 1: Core Architecture
- Fixed flow diagram: greeting/off_topic/fallback → END directly (not through respond)
- Fixed cosine distance range: [0,2] not [0,1]
- Retry feedback via state field (not messages list) to prevent history pollution
- Added `retry_feedback: str | None` to state schema
- Added deterministic empty-context guard in generate node
- Added `event: replace` SSE handler for non-streaming nodes
- Replaced keyword gambling detection with router sub-category `gambling_advice`
- Added `turn_limit_exceeded` response template

### Batch 2: Data Model
- Changed `last_updated: str` → `last_updated: date`
- Added `source: str = Field(..., min_length=1)`
- Added `schema_version: int = 1`
- Added `HoursEntry` model with time-awareness documentation
- Changed `price_range: str` → `Literal["", "$", "$$", "$$$", "$$$$"]`
- Removed `property_id` from state (config-driven, not state)

### Batch 3: API & Safety
- Added `asyncio.Lock` to RateLimiter for async safety
- Added scaling limitation callout for rate limiter
- Clarified circuit breaker integration into generate node (3 lines, not wrapper)
- Added `PropertyInfoResponse` Pydantic model
- Fixed `nginx.conf` in `.dockerignore`
- Made CORS origins configurable via `CORS_ORIGINS` env var
- Added `request_id` correlation ID to chat endpoint

### Batch 4: Testing
- Renumbered all lettered tests (20a-20q) to sequential (21-37)
- Added 7 integration tests: API auth (49-52), startup failures (53-54), property info (55)
- Added 3 eval tests: chained injection (67), unicode adversarial (68), max-length (69)
- Updated test pyramid counts: 37 unit / 18 integration / 14 eval = 69 total

### Batch 5: Domain Intelligence
- Rewrote concierge prompt with GUEST INTERACTION STYLE section (player psychology)
- Added Rule 10: "NEVER discuss other casino properties"
- Expanded Appendix C with Hey Seven Pulse connection, product evolution path, Rafi Ashkenazi context
- Added "How This Assignment Connects to Hey Seven Pulse" subsection

### Batch 6: Production Path
- Expanded LangSmith section with production workflow (dev/staging/prod/alerting)
- Added data residency note for CCPA
- Added `.env.example` with inline comments
- Added connection pooling requirement for PostgresSaver scaling
- Added `min-instances=1` recommendation for cold start
- Documented port 8080 as intentional in Docker decisions table

### Other Fixes
- Consolidated strawman decisions 5 & 6 into single Decision 5 (SSE + Vanilla Frontend)
- Added empirical evidence qualifiers to cost estimates and latency claims
- Added self-critique item 5: "I might be over-indexing on the 3-LLM-call pattern"
- Updated version to 7.0
