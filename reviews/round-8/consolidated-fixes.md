# Round 8 Consolidated Fix List

**Total findings**: 73 across 5 reviewers, 10 dimensions
**Target file**: assignment/architecture.md (3310 lines)

## PRIORITY 1: CRITICAL (factual errors, show-stoppers)

### FIX-1: Rafi Ashkenazi deal values WRONG [~line 3291]
- "$4.7B PokerStars acquisition" → was Sky Betting & Gaming, not PokerStars
- "$12B Flutter merger" → was ~$6B, not $12B
- Fix: "His The Stars Group tenure ($4.7B Sky Betting & Gaming acquisition, ~$6B Flutter Entertainment merger creating the world's largest online gambling company)"

### FIX-2: "NV NGC" should be "NV NGCB" [~line 3293]
- NGC = numismatic grading. NGCB = Nevada Gaming Control Board.

### FIX-3: .format() crash vector [~lines 272, 492]
- ROUTER_PROMPT.format() and VALIDATION_PROMPT.format() inject user input
- User sending {curly_braces} causes KeyError → DoS
- Fix: Document using str.replace() or Template.safe_substitute() instead

### FIX-4: state["property_id"] KeyError [~line 2768]
- property_id was removed from state in R6. Monitoring code still references it.
- Fix: Change to get_property_config()["id"]

### FIX-5: Monitoring metric inverts cosine distance [~line 2777]
- "top result score > 0.8" with cosine DISTANCE means similarity=0.2 (terrible)
- Fix: "top result distance < 0.4" (similarity > 0.6)

### FIX-6: Stale validation state across turns [~lines 278-281]
- retry_count, validation_result, retry_feedback never reset between turns
- Fix: Router must return reset values: validation_result: None, retry_count: 0, retry_feedback: None

### FIX-7: Streaming corruption on RETRY path [~lines 1764-1771, 655-661]
- Tokens from failed generate already streamed. Second generate appends more.
- Frontend shows [failed][corrected] concatenated.
- Fix: Document emitting event:replace at start of retry generate, or suppress streaming on retry

## PRIORITY 2: HIGH (CTO would notice)

### FIX-8: Missing security headers [~lines 1660-1938]
- Flagged in R1+R2, never fixed. No X-Content-Type-Options, X-Frame-Options.
- Fix: Add SecurityHeadersMiddleware section with nosniff + DENY + HSTS note

### FIX-9: request_id not propagated to graph nodes [~lines 1734-1740, 2764-2770]
- Generated at API level, never passed to LangGraph config. Breaks trace correlation.
- Fix: Pass through config["configurable"]["request_id"], extract in nodes

### FIX-10: No structlog configuration [~lines 2758-2761]
- import structlog shown but zero config. No JSON output, no log level, no timestamp.
- Fix: Add structlog.configure() block with JSONRenderer, TimeStamper, filtering

### FIX-11: retry_feedback never cleared [~line 775, 616, 638]
- Stale feedback leaks from dining turn to spa turn in multi-turn.
- Overlaps FIX-6. Fix: respond/fallback nodes return retry_feedback: None

### FIX-12: Router confidence unused [~lines 251, 644-653]
- confidence: float computed but never read in routing.
- Fix: Either use it (low confidence → safe RAG path) or remove with note

### FIX-13: Empty-context guard wastes LLM call [~lines 387-401, 469-473, 701]
- generate sets validation_result:"PASS" + retry_count:99, but validate still runs
- Fix: Expand validate guard to short-circuit on retry_count >= 99

### FIX-14: test_router_gambling asserts wrong category [~line 2091]
- "best slot odds" → gambling_advice, not off_topic
- Fix: Change expected to gambling_advice with note about routing to off_topic node

### FIX-15: SAFE Bet Act characterization [~line 3297]
- Targets sports betting specifically, not general AI. Fix wording.

### FIX-16: "Source citations" overpromises [~line 48, 599-617]
- Claims "Every answer references which data category" but it's metadata only, no inline citations
- Fix: Change "citations" to "source tracking" / "source metadata"

### FIX-17: Rate limiter breaks SSE [~lines 2862-2868]
- @app.middleware("http") uses BaseHTTPMiddleware which buffers responses
- Fix: Document as pure ASGI middleware

### FIX-18: No agent None check before streaming [~line 1730]
- agent = app.state.agent without None check → AttributeError → 200 + SSE error
- Fix: Add guard returning 503 JSONResponse

## PRIORITY 3: MEDIUM (polish for 9.5)

### FIX-19: InMemorySaver no max_threads [~lines 718, 3065]
- OOM via thread creation. Add MAX_ACTIVE_THREADS guard or document attack vector.

### FIX-20: Health endpoint ignores circuit breaker [~lines 1836-1857]
- Returns 200 when LLM circuit breaker open. Add cb state to health.

### FIX-21: No liveness vs readiness probe distinction [~line 1831]
- Single /health conflates both. Add doc note about Kubernetes split.

### FIX-22: Base images not pinned [~lines 2395, 2405, 2489]
- python:3.12-slim → python:3.12.8-slim, nginx:alpine → nginx:1.27-alpine

### FIX-23: CI/CD no test step [cloudbuild.yaml reference ~line 2338]
- Build→push→deploy with zero test gate. Add test step.

### FIX-24: No coverage reporting [~lines 2626-2633]
- No pytest-cov, no --cov in Makefile. Add coverage config.

### FIX-25: No requirements-dev.txt content [~line 3164]
- Listed in project structure, contents never shown. Add section.

### FIX-26: PropertyDataFile.category unvalidated str [~line 1072]
- Should be Literal type. Fix: Literal["dining","entertainment",...]

### FIX-27: Flow diagram missing gambling_advice [~lines 157-159]
- State comment missing gambling_advice + turn_limit_exceeded [~lines 763-764]

### FIX-28: 3 different error response shapes [~lines 1890-1917, 2867]
- Rate limiter, HTTPException, health all return different shapes. Unify.

### FIX-29: CORS missing X-Request-ID [~line 1934]

### FIX-30: thread_id regex rejects underscores [~line 1718]

### FIX-31: Multilingual injection defense note [~lines 1542-1548]

### FIX-32: Role-play attack defense [~lines 1542-1548]

### FIX-33: Validation criterion 6 references unreachable system prompt [~line 1622]

### FIX-34: .dockerignore missing tests/ [~lines 2468-2484]

### FIX-35: No rollback strategy documented [~line 2338]

### FIX-36: PropertyRetriever null guard [~lines 999-1023]

### FIX-37: Burst rate limiting for Gemini [~lines 2935-2942]

### FIX-38: LangSmith sampling unimplementable [~lines 2966-2968]

### FIX-39: Decision 9 cost calculation [~line 3082]

### FIX-40: Self-critique #5 reframe [~line 3252]

### FIX-41: Host salary unsourced [~line 3295]

### FIX-42: Competitive landscape shallow [~line 3295]

### FIX-43: top-k=5 trade-off undocumented

### FIX-44: temperature=0 trade-off undocumented

### FIX-45: Failed AI response pollutes history on retry [~lines 420,431]

### FIX-46: lru_cache breaks test mocking [~lines 253-256, 2175-2193]

### FIX-47: Chunk text header inconsistency [~line 923 vs 948]

### FIX-48: Smoke test Makefile target [~lines 2553-2597]

### FIX-49: CI staging note [cloudbuild.yaml]

### FIX-50: Version bump to 9.0
