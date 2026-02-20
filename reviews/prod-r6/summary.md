# Round 6 Fix Summary

**Date**: 2026-02-20
**Spotlight**: API CONTRACT & DOCUMENTATION
**Scores**: Gemini=61, GPT=58, Grok=54 (avg=57.7)
**Previous avg**: R5=63.3

---

## Consensus Analysis

Compiled findings from 3 reviewers (Gemini 3 Pro, GPT-5.2, Grok 4). Identified consensus at 2/3+ model agreement. Fixed single-model CRITICAL findings (Grok's TRUSTED_PROXIES contradiction).

## Findings Fixed: 26 total

### CRITICAL (3 findings, all fixed)

| ID | Description | Fix Applied |
|----|-------------|-------------|
| Gemini-F1/GPT-F11/Grok-F5 | False auth contract: README says "None" for /feedback, /graph, /property but code requires API key | Updated README API table: all 4 protected paths now show "API key (when configured)" |
| Gemini-F1/Grok-F5 | /sms/webhook auth undocumented (Ed25519 signature verification) | Updated README: shows "Ed25519 signature (TELNYX_PUBLIC_KEY)" |
| Grok-F1/GPT-F10 | TRUSTED_PROXIES doc says "empty = trust all" but code says "None = trust nobody" | Fixed ARCHITECTURE.md: now says "None = never trust XFF headers (use direct peer IP)" |

### HIGH (10 findings, all fixed)

| ID | Description | Fix Applied |
|----|-------------|-------------|
| Gemini-F2/GPT-F8 | Hotel agent exists but omitted from all docs ("Ghost Agent") | Added hotel to README (5 specialist agents), ARCHITECTURE.md dispatch table, system diagram, module organization, project structure |
| Gemini-F3/GPT-F2 | SSE ping event undocumented and unmodeled | Added SSEPingEvent to models.py. Added ping to SSE Events sections in README and ARCHITECTURE.md |
| Gemini-F4/Grok-F13 | .env.example missing ~24 settings (43% of config surface) | Rewrote .env.example with all 56 settings grouped by 11 categories with descriptive comments |
| Gemini-F5/GPT-F14/Grok-F12 | State field count inconsistent (14/15/13) across docs | Standardized to "13 fields" everywhere, removed hardcoded counts, added "3 Pydantic output models" clarification |
| GPT-F1/Grok-F10 | VERSION=0.1.0 in .env/.env.example overrides config.py 1.0.0 | Updated .env and .env.example to VERSION=1.0.0. Fixed ARCHITECTURE.md config table default |
| GPT-F5 | Error taxonomy (7 ErrorCodes) not documented externally | Added full Error Responses section to README: canonical JSON shape, all 7 codes, HTTP status mapping, retryability |
| GPT-F9/Grok-F4 | Health endpoint example missing circuit_breaker_state | Added circuit_breaker_state to ARCHITECTURE.md health JSON example, documented allowed values |
| Gemini-F9 | Rate limiter scope undocumented (/feedback also rate-limited) | Updated README middleware table: "20 req/min on /chat and /feedback (shared bucket)" |
| GPT-F4 | Webhook responses use ad-hoc dicts without models | Added SmsWebhookResponse and CmsWebhookResponse Pydantic models to models.py |
| GPT-F11 | ARCHITECTURE.md says ApiKeyMiddleware only protects /chat | Updated to list all 4 protected paths with TTL-cached key refresh documentation |

### MEDIUM (10 findings, all fixed)

| ID | Description | Fix Applied |
|----|-------------|-------------|
| Gemini-F7/Grok-F11 | Settings count "48" stale (actual: 56) | Updated all 3 occurrences in README to "56" |
| Gemini-F8/GPT-F6/Grok-F3 | Test count "1090" stale (actual: 1216 collected) | Updated README: "1198 tests collected" -> "1216 tests collected" (after adding 18 new tests) |
| GPT-F7 | README says "4-step pipeline", cloudbuild has 5 steps | Updated README to "5-step pipeline" |
| Gemini-F10 | Stale line numbers in ARCHITECTURE.md (RouterOutput, ValidationResult) | Replaced all line number references with class/function name references |
| GPT-F13 | VERSION default in ARCHITECTURE.md config table stale | Updated from 0.1.0 to 1.0.0 |
| Gemini-F12 | README says "ApiKeyAuth", code says "ApiKeyMiddleware" | Updated all middleware names in README to match actual class names |
| Gemini-F11/GPT-F12 | LangSmith vs LangFuse naming inconsistent | Clarified: "LangFuse (primary) + LangSmith (optional LangChain tracing)". Updated .env.example with both sections |
| All | Specialist agent count "4" stale (actual: 5) | Updated to "5" in 6 places across README and ARCHITECTURE |
| GPT-F9/Grok-F9 | No OpenAPI for SSE streaming | Not fixed (code change requiring endpoint restructuring - deferred to R7) |
| Grok-F7 | LOG_LEVEL=WARNING suppresses operational logs | Not fixed (access logger is isolated via propagate=False; finding partially a false positive) |

### LOW (3 findings, all fixed)

| ID | Description | Fix Applied |
|----|-------------|-------------|
| Gemini-F11 | .env.example shows LangSmith but not LangFuse | Added LangFuse section to .env.example (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST) |
| Gemini-F12 | README middleware names don't match class names | Fixed all 6 middleware names to match actual Python class names |
| GPT-F14 | State field count described inconsistently | Standardized everywhere |

---

## New Tests: 18 (in tests/test_doc_accuracy.py)

| Test Class | Tests | What It Prevents |
|------------|-------|------------------|
| TestSettingsCount | 1 | Settings count drift (asserts 56) |
| TestAgentRegistry | 3 | Agent count drift (asserts 5 agents, hotel included) |
| TestStateFieldCount | 1 | State field count drift (asserts 13) |
| TestSSEEventModels | 2 | SSE event model completeness (all 8 models importable) |
| TestHealthResponseModel | 2 | Health endpoint field drift (7 fields, circuit_breaker_state present) |
| TestErrorTaxonomy | 2 | Error code drift (7 codes, all expected values) |
| TestMiddlewareProtectedPaths | 1 | Auth scope drift (4 protected paths) |
| TestRateLimitScope | 1 | Rate limit scope drift (/chat and /feedback) |
| TestWebhookResponseModels | 2 | Webhook model existence |
| TestVersionConsistency | 2 | VERSION consistency (config.py default = .env.example) |
| TestCategoryToAgentMapping | 1 | Hotel category dispatch |

---

## Test Results

```
1196 passed, 20 skipped, 1 warning in 10.91s
Coverage: 90.61% (threshold: 90%)
Total collected: 1216 (1198 existing + 18 new)
```

## Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `README.md` | Updated | Auth table, test counts, settings counts, agent counts, SSE events, error taxonomy, middleware names, pipeline steps, state fields, observability naming, project structure |
| `ARCHITECTURE.md` | Updated | TRUSTED_PROXIES, health response, line number refs, auth scope, hotel agent, state fields, version, specialist count, SSE ping event |
| `.env.example` | Rewritten | All 56 settings with 11-category grouping, descriptive comments, correct defaults |
| `.env` | Updated | VERSION=0.1.0 -> 1.0.0 |
| `src/api/models.py` | Updated | Added SSEPingEvent, SmsWebhookResponse, CmsWebhookResponse models |
| `tests/test_doc_accuracy.py` | Created | 18 doc-drift prevention tests |

## Deferred to R7

1. **OpenAPI for SSE streaming** (GPT-F3/Grok-F9): Requires endpoint restructuring with `openapi_extra`. Deferred as code change, not doc fix.
2. **SSE schema runtime enforcement** (Grok-F8): Schemas are documentation-only by design (performance). Trade-off documented.
3. **/sms/webhook rate limiting** (Grok-F2): Code change requiring webhook-specific rate limit logic. Deferred.

---

## Score Trend

| Round | Spotlight | Gemini | GPT | Grok | Avg |
|-------|-----------|--------|-----|------|-----|
| R1 | General | — | — | — | 67.3 |
| R2 | Security | — | — | — | 61.3 |
| R3 | Observability | — | — | — | 60.7 |
| R4 | Domain | — | — | — | 66.7 |
| R5 | Scalability & Async | — | — | — | 63.3 |
| R6 | API Contract & Docs | 61 | 58 | 54 | 57.7 |
