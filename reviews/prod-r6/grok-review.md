# Round 6 Production Review -- Grok 4

**Date**: 2026-02-20
**Reviewer**: Grok 4 (via grok_reason, high effort)
**Spotlight**: API CONTRACT & DOCUMENTATION (+1 severity for API/doc findings)
**Previous Scores**: R1=67.3, R2=61.3, R3=60.7, R4=66.7, R5=63.3

---

## Findings

### CRITICAL

**F1. TRUSTED_PROXIES documentation directly contradicts code behavior** (Fact 5)
- ARCHITECTURE.md Configuration table says: `TRUSTED_PROXIES` default is `[]` and "empty = trust all, for Cloud Run"
- config.py actual default: `TRUSTED_PROXIES: list[str] | None = None`
- middleware.py code comment: "None = trust nobody's XFF (default -- prevents IP spoofing)"
- **The documentation says the OPPOSITE of what the code does.** Doc says "trust all", code says "trust nobody."
- A developer reading the ARCHITECTURE.md who wants to restrict XFF trust will think the default already trusts all and may not configure it. A developer reading the code who wants Cloud Run XFF trust will configure `[]` thinking it means "trust all" per the doc, but the middleware would treat empty list differently than None.
- **Impact**: Security misconfiguration leading to IP spoofing or broken rate limiting behind Cloud Run LB.
- Severity: CRITICAL (base HIGH + spotlight)

**F2. /sms/webhook has no rate limiting and minimal auth** (Fact 9)
- The endpoint only validates Telnyx signature when `TELNYX_PUBLIC_KEY` is configured (optional).
- RateLimitMiddleware only applies to `/chat` and `/feedback` -- webhooks are exempt.
- An attacker who discovers the webhook URL can flood it with POST requests, triggering `handle_inbound_sms()` processing for each.
- ARCHITECTURE.md does not document this security gap or provide mitigation guidance.
- **Impact**: DoS vector on an unprotected endpoint that processes inbound SMS events.
- Severity: CRITICAL (base HIGH + spotlight for missing documentation)

### HIGH

**F3. Stale test counts across README and ARCHITECTURE.md** (Fact 1)
- README.md says: "1090 tests passed, 14 skipped" and "1090 tests across 34 files"
- R5 summary shows: 1178 tests
- Actual collection: 1198 tests
- Test count is stale by 108+ tests across two documentation files.
- README `make test-ci` comment says "1090 tests, no API key needed" -- also stale.
- Severity: HIGH (base MEDIUM + spotlight)

**F4. Health endpoint documentation missing `circuit_breaker_state` field** (Fact 3)
- ARCHITECTURE.md `/health` response example shows 6 fields (status, version, agent_ready, property_loaded, rag_ready, observability_enabled)
- models.py `HealthResponse` has 7 fields (includes `circuit_breaker_state: str = "unknown"`)
- Added in R5, never reflected in documentation.
- Client integrations built from ARCHITECTURE.md will not expect or handle this field.
- Severity: HIGH (base MEDIUM + spotlight)

**F5. API auth scope understated in README** (Fact 8)
- README API table says `/chat` requires "API key + rate limit"
- Other endpoints (`/graph`, `/property`, `/feedback`) listed as "None" for auth
- Code: `ApiKeyMiddleware._PROTECTED_PATHS = {"/chat", "/graph", "/property", "/feedback"}`
- All four endpoints are API-key protected when `API_KEY` is set. README is wrong.
- Severity: HIGH (base MEDIUM + spotlight)

**F6. Dockerfile HEALTHCHECK only checks HTTP liveness, not readiness** (Fact 10)
- HEALTHCHECK runs `urllib.request.urlopen('http://localhost:8080/health')` -- checks HTTP 200 only
- The `/health` endpoint returns 503 for degraded state, but urllib treats any HTTP response as success (non-exception)
- Does not verify `agent_ready`, `rag_ready`, or `circuit_breaker_state`
- Container could be marked healthy while agent is uninitialized or circuit breaker is open.
- Severity: HIGH (base MEDIUM + spotlight)

**F7. Production LOG_LEVEL=WARNING suppresses access logs** (Fact 11)
- cloudbuild.yaml: `--set-env-vars=ENVIRONMENT=production,LOG_LEVEL=WARNING`
- Access logs in `RequestLoggingMiddleware` emit at `_access_logger.info(...)` -- INFO level
- WARNING level suppresses ALL access logs in production
- This contradicts "Cloud Logging compatible" and "structured JSON access logs" documentation
- The access logger has `log.setLevel(logging.INFO)` hardcoded, but the `logging.basicConfig(level=...)` in lifespan sets the root logger to WARNING, which may propagate.
- **Wait** -- the access logger sets `log.propagate = False` and has its own handler at INFO level. So the access logger should still work even with root at WARNING. **This is actually a false positive** if propagate=False truly isolates it. But the root logger's level could still affect child loggers. Let me re-examine: `logging.getLogger("hey_seven.access")` with `propagate=False` and explicit INFO handler -- this IS isolated. So access logs ARE preserved.
- **Revised**: The access logger is isolated. But other `logger.info()` calls throughout the codebase (e.g., in app.py, graph.py) WILL be suppressed at WARNING level. This means startup logs ("Agent initialized successfully", "Property metadata loaded") and operational logs are lost.
- Severity: HIGH (operational visibility loss in production, +spotlight)

**F8. SSE event contract documentation inconsistencies** (Fact 13)
- `SSEGraphNodeEvent` model: `metadata: dict | None = None`
- README SSE examples show: `{"node": "router", "status": "complete", "duration_ms": 450, "metadata": {"query_type": "property_qa", "confidence": 0.95}}`
- This is consistent -- but the models.py SSE event schemas are documentation-only (comment says "Events are serialized as dicts in graph.py for streaming performance; these schemas serve as the canonical API reference").
- The schemas are NOT enforced at runtime (no `.model_validate()` on emit). Wire format could drift from schema without detection.
- Severity: HIGH (base MEDIUM + spotlight for unenforced API contract)

**F9. Missing OpenAPI/Swagger for SSE streaming endpoint** (Fact 14)
- `/chat` endpoint returns `EventSourceResponse` which FastAPI cannot represent in OpenAPI spec
- The auto-generated `/docs` page will show `/chat` but with no response schema
- For an API-first product, this is a significant documentation gap
- No custom OpenAPI schema override or separate SSE documentation beyond README examples
- Severity: HIGH (base MEDIUM + spotlight)

### MEDIUM

**F10. Version string mismatches across documentation** (Fact 2)
- ARCHITECTURE.md health example: `"version": "0.1.0"`
- config.py: `VERSION: str = "1.0.0"`
- .env.example: `VERSION=0.1.0`
- ARCHITECTURE.md configuration table (line 724): `VERSION` default `0.1.0`
- Three different version references, two different values. config.py says 1.0.0, everything else says 0.1.0.
- Severity: MEDIUM (base LOW + spotlight)

**F11. Configuration count claim "48 settings" is stale** (Fact 4)
- README.md says "48 env-overridable settings"
- ARCHITECTURE.md design philosophy section says "48 env-overridable settings"
- ARCHITECTURE.md configuration table lists ~57 settings (including PROPERTY_STATE, WHISPER_LLM_TEMPERATURE, COMP_COMPLETENESS_THRESHOLD, SEMANTIC_INJECTION_ENABLED/THRESHOLD/MODEL, CASINO_ID, TELNYX_API_KEY/MESSAGING_PROFILE_ID/PUBLIC_KEY, QUIET_HOURS_START/END, SMS_FROM_NUMBER, LANGFUSE_PUBLIC_KEY/SECRET_KEY/LANGFUSE_HOST, SMS_ENABLED, CONSENT_HMAC_SECRET, PERSONA_MAX_CHARS)
- Severity: MEDIUM (base LOW + spotlight)

**F12. PropertyQAState field count contradictions** (Fact 6)
- ARCHITECTURE.md State Schema section: "PropertyQAState is a TypedDict with 13 fields (src/agent/state.py:39)"
- ARCHITECTURE.md State Schema table: lists 13 fields
- README Project Structure: "PropertyQAState (15 fields)"
- Self-contradicting within the same document and across documents.
- Severity: MEDIUM (base LOW + spotlight)

**F13. .env.example missing many production-relevant settings** (Fact 12)
- .env.example covers ~25 settings (core Google, property, LLM, RAG, API, agent, observability)
- config.py defines ~57 settings
- Missing from .env.example: PROPERTY_STATE, SEMANTIC_INJECTION_*, CASINO_ID, TELNYX_API_KEY, TELNYX_MESSAGING_PROFILE_ID, SMS_FROM_NUMBER, SMS_ENABLED, LANGFUSE_*, TRUSTED_PROXIES, PERSONA_MAX_CHARS, FIRESTORE_*, COMP_COMPLETENESS_THRESHOLD, WHISPER_LLM_TEMPERATURE, CONSENT_HMAC_SECRET, QUIET_HOURS_*
- No indication which settings are required for production vs optional
- Severity: MEDIUM

---

## Dimension Scores

| # | Dimension | R5 Avg | R6 Score | Delta | Notes |
|---|-----------|--------|----------|-------|-------|
| 1 | Graph Architecture | 7.0 | 7 | 0 | Graph topology solid. Doc inconsistencies on state fields (F12) hurt but don't break architecture. |
| 2 | RAG Pipeline | 7.0 | 7 | 0 | Untouched by spotlight findings. Pipeline design remains solid. |
| 3 | Data Model & State | 6.0 | 5 | -1 | HealthResponse doc omits field (F4), state field counts contradictory (F12), SSE schemas unenforced (F8). |
| 4 | API Design | 6.0 | 4 | -2 | Spotlight round devastates: auth scope understated (F5), webhook unprotected (F2), no OpenAPI for SSE (F9), SSE contracts unenforced (F8). |
| 5 | Testing Strategy | 7.0 | 6 | -1 | Stale test counts (F3). Tests exist and work but docs are unreliable. |
| 6 | Docker & DevOps | 6.0 | 5 | -1 | HEALTHCHECK liveness-only (F6), log level suppression (F7), .env.example incomplete (F13). |
| 7 | Prompts & Guardrails | 7.0 | 7 | 0 | Not directly impacted this round. Guardrails remain solid. |
| 8 | Scalability & Production | 5.0 | 4 | -1 | Webhook flood risk (F2), TRUSTED_PROXIES contradiction (F1), log suppression (F7). |
| 9 | Trade-off Documentation | 6.0 | 3 | -3 | Spotlight-round massacre. Version mismatches (F10), config count stale (F11), field count contradictions (F12), TRUSTED_PROXIES contradiction (F1), auth scope wrong (F5). Docs cannot be trusted. |
| 10 | Domain Intelligence | 7.0 | 6 | -1 | Casino domain settings exist but undocumented in .env.example (F13). |

**Total: 54/100**

---

## Summary

| Metric | Value |
|--------|-------|
| Total Score | 54/100 |
| Score Trajectory | R1=67.3, R2=61.3, R3=60.7, R4=66.7, R5=63.3, **R6=54** |
| Finding Count | 13 |
| CRITICAL | 2 (F1: TRUSTED_PROXIES contradiction, F2: webhook no rate limit) |
| HIGH | 7 (F3-F9) |
| MEDIUM | 4 (F10-F13) |
| LOW | 0 |

### Top 3 Fixes (highest impact)

1. **Fix TRUSTED_PROXIES documentation to match code** (F1) -- ARCHITECTURE.md says "trust all" but code says "trust nobody". This is a security-critical doc lie. Update ARCHITECTURE.md to say `None` (default) = trust nobody, and document how to configure for Cloud Run.

2. **Add rate limiting to /sms/webhook** (F2) -- Either add `/sms/webhook` to RateLimitMiddleware paths or implement webhook-specific rate limiting. Document the security model for all webhook endpoints.

3. **Update all stale documentation** (F3, F4, F10, F11, F12) -- Single sweep to fix test counts (1198), health response schema (add circuit_breaker_state), version strings (pick one: 0.1.0 or 1.0.0), config count (~57), state field count, and auth scope description.

### Verdict

The documentation rot is severe. Five rounds of code fixes have outpaced documentation updates, creating a growing gap between what the docs promise and what the code delivers. The TRUSTED_PROXIES contradiction (F1) is the most dangerous -- it's not just stale, it's actively misleading about security behavior. The webhook security gap (F2) is a real production vulnerability. The remaining findings are doc-staleness that erodes developer trust. The code itself is solid, but the contract between code and documentation is broken.
