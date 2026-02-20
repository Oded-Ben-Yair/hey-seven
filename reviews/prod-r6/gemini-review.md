# Round 6 Production Review -- Gemini 3 Pro

**Date**: 2026-02-20
**Spotlight**: API CONTRACT & DOCUMENTATION (+1 severity on all API/docs findings)
**Reviewer**: Gemini 3 Pro (thinking=high)
**Previous Scores**: R1=67.3, R2=61.3, R3=60.7, R4=66.7, R5=63.3

---

## Verdict: REJECTED

The documentation drift is severe enough that README.md and ARCHITECTURE.md describe a different application than the codebase. API consumers are given false information about authentication requirements, SSE event schemas, agent capabilities, and configuration surface area.

---

## Dimensional Scores

| # | Dimension | Score | Key Observations |
|---|-----------|-------|------------------|
| 1 | Graph Architecture | 6 | Solid 11-node design with compliance gate + validation loop. Undermined by undocumented hotel agent creating a phantom routing path. State schema count disagrees across 3 documents. |
| 2 | RAG Pipeline | 7 | RRF reranking, per-item chunking, idempotent ingestion, property_id isolation are all strong. Missing VECTOR_DB and SEMANTIC_INJECTION_* in .env.example hurts configurability. |
| 3 | Data Model | 5 | SSE ping event emitted in production but has no Pydantic model in models.py. State field count: README says 12+2=14, ARCHITECTURE.md says 13 (line 267) and 15 (line 836), actual code has 13. |
| 4 | API Design | 4 | **Spotlight dimension.** False auth documentation on /feedback (says "None", actually requires API key). /sms/webhook documented as "None" auth but performs Ed25519 signature verification. Rate limiter scope undocumented (/feedback is rate-limited but docs only mention /chat). |
| 5 | Testing Strategy | 6 | Strong test infrastructure (~989 test functions across 36 files). README claims "1090 tests, 34 files" -- stale. R5 summary says 1178. Three conflicting numbers. |
| 6 | Docker & DevOps | 7 | Dockerfile is well-structured (multi-stage, non-root, exec-form CMD, healthcheck). .env.example missing ~43% of config.py settings (24 out of 56) undermines operability. |
| 7 | Prompts & Guardrails | 7 | 73 regex patterns across 4 languages is impressive. Hotel agent's guardrail coverage is undocumented and therefore unverifiable. |
| 8 | Scalability & Production | 7 | R5 fixes (BoundedMemorySaver, TTLCache singletons, LLM semaphore, atomic tuple) are solid. Rate limit coupling /chat + /feedback is a noisy-neighbor risk. |
| 9 | Trade-off Documentation | 5 | Hardcoded line numbers in ARCHITECTURE.md are already stale (RouterOutput documented at line 79, actual line 82; ValidationResult documented at line 93, actual line 96). Documentation accuracy is decaying. |
| 10 | Domain Intelligence | 7 | Casino domain modeling is strong (responsible gaming, BSA/AML, patron privacy, TCPA compliance). Hotel as a domain category exists in routing but is absent from all documentation. |

**Overall Score: 61/100**

---

## Findings

### CRITICAL

**F1: False authentication contract on /feedback and /sms/webhook**
- **File**: `README.md` line 151, `src/api/middleware.py` line 230
- **Issue**: README API table says `/feedback` Auth is "None" and `/sms/webhook` Auth is "None". In reality:
  - `/feedback` is in `ApiKeyMiddleware._PROTECTED_PATHS` = `{"/chat", "/graph", "/property", "/feedback"}` -- requires X-API-Key header when API_KEY is set
  - `/sms/webhook` performs Ed25519 signature verification via `verify_webhook_signature()` when TELNYX_PUBLIC_KEY is configured
- **Impact**: Frontend developers integrating /feedback will get 401 errors in production. Telnyx integration teams won't know they need to configure webhook signing.
- **Fix**: Update README API table:
  - `/feedback`: Auth = "API key (when configured)"
  - `/sms/webhook`: Auth = "Ed25519 signature (TELNYX_PUBLIC_KEY)"
  - `/graph` and `/property`: Auth = "API key (when configured)" (also in _PROTECTED_PATHS but documented as "None")
- **Severity boost**: +1 for spotlight (API CONTRACT)

**F2: Undocumented hotel agent ("Ghost Agent")**
- **File**: `src/agent/agents/hotel_agent.py`, `src/agent/agents/registry.py` line 20, `src/agent/graph.py` line 123
- **Issue**: hotel_agent.py exists and is registered in the agent registry. `_CATEGORY_TO_AGENT` maps `"hotel": "hotel"`. But README says "4 specialist agents (host, dining, entertainment, comp)" in 3 separate places. README project structure doesn't list hotel_agent.py. ARCHITECTURE.md specialist dispatch table doesn't mention hotel routing.
- **Impact**: The system silently routes hotel category queries to an agent that no documentation acknowledges. Reviewers, testers, and new developers cannot verify its behavior.
- **Fix**: Either (a) add hotel to all documentation (README specialist list, project structure, ARCHITECTURE.md dispatch table) making it "5 specialist agents", or (b) remove it if experimental.

### HIGH

**F3: SSE ping event undocumented and unmodeled**
- **File**: `src/api/app.py` line 185, `src/api/models.py`
- **Issue**: app.py emits `{"event": "ping", "data": ""}` as an SSE heartbeat during long generations. But:
  - models.py has no `SSEPingEvent` class (all other SSE events have models)
  - README SSE Events section lists 7 event types but omits `ping`
- **Impact**: Frontend clients using strict TypeScript/Zod schemas for SSE events will fail validation on the undocumented `ping` event.
- **Fix**: Add `SSEPingEvent` to models.py. Add `event: ping -- heartbeat (every 15s during generation)` to README SSE section.
- **Severity boost**: +1 for spotlight

**F4: .env.example missing 24 settings (43% of config surface)**
- **File**: `.env.example`, `src/config.py`
- **Issue**: config.py defines 56 settings. .env.example documents only 32. Missing settings include:
  - Multi-tenant: CASINO_ID
  - CMS: CMS_WEBHOOK_SECRET, GOOGLE_SHEETS_ID
  - SMS: SMS_ENABLED, SMS_FROM_NUMBER, CONSENT_HMAC_SECRET, TELNYX_API_KEY, TELNYX_MESSAGING_PROFILE_ID, TELNYX_PUBLIC_KEY, QUIET_HOURS_START, QUIET_HOURS_END, PERSONA_MAX_CHARS
  - Observability: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
  - Vector DB: VECTOR_DB, FIRESTORE_PROJECT, FIRESTORE_COLLECTION
  - Agent: SEMANTIC_INJECTION_ENABLED, SEMANTIC_INJECTION_THRESHOLD, SEMANTIC_INJECTION_MODEL, WHISPER_LLM_TEMPERATURE, COMP_COMPLETENESS_THRESHOLD
  - API: TRUSTED_PROXIES
  - Property: PROPERTY_STATE
- **Impact**: A fresh `git clone && cp .env.example .env` won't expose these settings. New developers must reverse-engineer config.py. Production deployments risk missing required secrets (CMS_WEBHOOK_SECRET validator will hard-fail in production).
- **Fix**: Add all 56 settings to .env.example with descriptive comments, grouped by category matching config.py sections.

**F5: State field count inconsistent across 3 documents**
- **File**: `README.md` line 64, `ARCHITECTURE.md` lines 267 and 836, `src/agent/state.py`
- **Issue**: Four different counts for the same type:
  - README: "12 fields, 2 v2 additions" = 14
  - ARCHITECTURE.md line 267: "13 fields"
  - ARCHITECTURE.md line 836: "15 fields"
  - Actual code (state.py): 13 fields in PropertyQAState
- **Fix**: Remove hardcoded field counts from documentation. Reference the TypedDict directly: "See `PropertyQAState` in `src/agent/state.py`".

**F6: /graph and /property auth undocumented**
- **File**: `README.md` line 148-149, `src/api/middleware.py` line 230
- **Issue**: README says `/graph` and `/property` both have Auth = "None". But both are in `ApiKeyMiddleware._PROTECTED_PATHS` and require X-API-Key when API_KEY is set.
- **Impact**: Frontend visualization and property info endpoints will return 401 in production if API_KEY is configured.
- **Fix**: Update README to show "API key (when configured)" for these endpoints.
- **Severity boost**: +1 for spotlight

### MEDIUM

**F7: Settings count claim stale**
- **File**: `README.md` lines 96, 182, 220
- **Issue**: README says "48 env-overridable settings" in 3 places. config.py has 56. Undercounts by 8.
- **Fix**: Update all 3 occurrences to "56" or better yet, use a count that auto-updates (or remove the count entirely).
- **Severity boost**: +1 for spotlight

**F8: Test count stale and inconsistent**
- **File**: `README.md` lines 126-131, R5 summary
- **Issue**: README claims "1090 tests passed, 14 skipped" across "34 test files". R5 summary says 1178 tests. Actual grep of `def test_` functions yields ~989 across 36 files. Three conflicting numbers, none current.
- **Fix**: Run `pytest --co -q | tail -1` to get actual count. Update README. Consider CI badge or Makefile target that auto-generates test count.

**F9: Rate limiter scope undocumented**
- **File**: `README.md` line 177, `src/api/middleware.py` line 379
- **Issue**: README says "RateLimit | Token-bucket per IP, 20 req/min on `/chat`". Middleware actually rate-limits both `/chat` AND `/feedback` (line 379: `if path not in ("/chat", "/feedback")`). Shared bucket means a chatty user who hits the rate limit on /chat cannot submit feedback.
- **Fix**: Either (a) document the shared scope, or (b) separate rate limits per endpoint if the coupling is unintentional.

**F10: ARCHITECTURE.md line number references already stale**
- **File**: `ARCHITECTURE.md` lines 267, 287-289
- **Issue**:
  - "state.py:79" for RouterOutput -- actual line is 82
  - "state.py:93" for ValidationResult -- actual line is 96
  - "state.py:107" for WhisperPlan reference -- WhisperPlan is actually in whisper_planner.py:107 (this one is correct but fragile)
- **Fix**: Remove line numbers from ARCHITECTURE.md. Reference by class/function name only: "See `RouterOutput` in `src/agent/state.py`".

### LOW

**F11: .env.example shows LangSmith but not LangFuse**
- **File**: `.env.example` lines 56-59
- **Issue**: .env.example has commented-out `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` (LangSmith), but config.py defines `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (LangFuse). The observability stack appears to have migrated but .env.example wasn't updated.
- **Fix**: Add LangFuse variables to .env.example. Keep LangSmith as optional/commented if still supported.

**F12: README middleware name inconsistency**
- **File**: `README.md` line 175
- **Issue**: README calls it "ApiKeyAuth" but the actual class name is `ApiKeyMiddleware` (middleware.py line 217).
- **Fix**: Update README to match class name: "ApiKeyMiddleware".
- **Severity boost**: +1 for spotlight

---

## Score Summary

| Dimension | Score |
|-----------|-------|
| 1. Graph Architecture | 6 |
| 2. RAG Pipeline | 7 |
| 3. Data Model | 5 |
| 4. API Design | 4 |
| 5. Testing Strategy | 6 |
| 6. Docker & DevOps | 7 |
| 7. Prompts & Guardrails | 7 |
| 8. Scalability & Production | 7 |
| 9. Trade-off Documentation | 5 |
| 10. Domain Intelligence | 7 |
| **TOTAL** | **61** |

---

## Finding Summary

| Severity | Count | Findings |
|----------|-------|----------|
| CRITICAL | 2 | F1 (false auth contract), F2 (ghost hotel agent) |
| HIGH | 4 | F3 (SSE ping undocumented), F4 (.env.example gaps), F5 (state count drift), F6 (/graph /property auth) |
| MEDIUM | 4 | F7 (settings count), F8 (test count), F9 (rate limit scope), F10 (stale line numbers) |
| LOW | 2 | F11 (LangSmith vs LangFuse), F12 (middleware name) |
| **TOTAL** | **12** | |

---

## Fix Priority

1. **F1 + F6**: Fix the API auth table in README (all 4 protected paths documented correctly)
2. **F2**: Document or remove hotel agent -- add to README, ARCHITECTURE.md, project structure
3. **F3**: Add SSEPingEvent to models.py, add ping to README SSE events section
4. **F4**: Sync .env.example with all 56 config.py settings
5. **F5 + F7 + F8**: Remove hardcoded counts from documentation (field count, settings count, test count)
6. **F10**: Replace line numbers with class/function references in ARCHITECTURE.md
7. **F9**: Document rate limit scope or decouple /chat and /feedback limits
8. **F11 + F12**: Minor doc fixes
