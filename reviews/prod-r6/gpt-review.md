# Round 6 Production Review — GPT-5.2

**Date**: 2026-02-20
**Spotlight**: API CONTRACT & DOCUMENTATION (+1 severity boost on doc/API findings)
**Model**: GPT-5.2 (Azure AI Foundry)
**Previous Scores**: R1=67.3, R2=61.3, R3=60.7, R4=66.7, R5=63.3

---

## Dimension Scores

| # | Dimension | Score | Justification |
|---|-----------|------:|---------------|
| 1 | Graph Architecture | 7 | 11-node custom StateGraph with specialist dispatch is well-structured. Hotel agent exists in code but omitted from all public documentation, creating a hidden capability gap in architecture understanding. |
| 2 | RAG Pipeline | 6 | RRF reranking, idempotent ingestion, and relevance filtering are solid. Documentation describes pipeline steps inconsistently (README: "4-step", ARCHITECTURE: "5-step", actual: 5 steps in cloudbuild.yaml). |
| 3 | Data Model | 6 | PropertyQAState TypedDict is clean with 13 fields. Documentation repeatedly misstates "15 fields" (README:224, ARCHITECTURE:836), conflating state schema with Pydantic output models. |
| 4 | API Design | 4 | SSE endpoints lack OpenAPI representation. `ping` heartbeat event undocumented. Webhook responses use ad-hoc dicts without Pydantic models. Error taxonomy (7 ErrorCodes) well-implemented but completely absent from external docs. Protected endpoint list misdocumented. |
| 5 | Testing Strategy | 6 | 1198 tests collected (verified via pytest --co). README claims 1090 in 3 places — 108 tests behind reality. R5 summary says 1178. No automated check prevents test count drift. |
| 6 | Docker & DevOps | 6 | 5-step Cloud Build pipeline with Trivy scan is production-grade. README says "4-step pipeline" (line 274). VERSION mismatch between .env (0.1.0) and config.py (1.0.0) causes /health to report incorrect version. |
| 7 | Prompts & Guardrails | 7 | 73 regex patterns across 4 languages, 5 guardrail categories, compliance gate before LLM — all well-implemented. No documentation drift detected in guardrail descriptions specifically. |
| 8 | Scalability & Production | 5 | Health endpoint contract documented incompletely (circuit_breaker_state missing from example JSON). VERSION env mismatch means /health can report 0.1.0 in production — breaks rollback verification and monitoring. TRUSTED_PROXIES default mismatch between docs and code persists from R2. |
| 9 | Trade-off Documentation | 4 | Degraded-pass strategy and single-worker trade-offs are well-documented. However, core operational contracts (error format, auth scope, SSE event taxonomy, version semantics) are missing or inaccurate — integrators cannot build reliable clients from docs alone. |
| 10 | Domain Intelligence | 7 | 5 specialist agents (host, dining, entertainment, comp, hotel) with category-based dispatch and feature flag control. Hotel agent exists and is tested but entirely absent from README and ARCHITECTURE — domain capability is better than documentation suggests. |

**Total: 58.0 / 100** (5.8 avg x 10)

---

## Findings

### CRITICAL

*None*

### HIGH (5 findings)

| ID | Title | File:Line | Description | Fix |
|---:|-------|-----------|-------------|-----|
| F1 | `.env` VERSION=0.1.0 overrides config.py 1.0.0 — /health reports wrong version | `.env:51`, `.env.example:54`, `src/config.py:91` | When running with `.env` file (which Docker Compose uses), `VERSION` is set to `0.1.0`, overriding the config.py default of `1.0.0`. The `/health` endpoint reports this value. In production, operators verifying deployment via `/health` version will see `0.1.0` while expecting `1.0.0`. This breaks release verification, rollback detection, and version-keyed metrics. **Spotlight boost**: operational contract violation. | Update `.env` and `.env.example` to `VERSION=1.0.0`. Add startup warning log if VERSION contains "0." prefix in non-development environment. Update ARCHITECTURE.md:724 to match. |
| F2 | SSE `ping` heartbeat event undocumented | `src/api/app.py:185-186`, `README.md:156-165`, `ARCHITECTURE.md:459-469` | Server emits `event: ping` with `data: ""` every 15 seconds during SSE streams, but this event type is NOT listed in the SSE Event Types documentation in README or ARCHITECTURE. Clients that don't expect `ping` events may log errors, treat them as malformed data, or fail to implement keepalive handling. Browser EventSource ignores unknown events by default, but custom SSE parsers (mobile, server-to-server) may not. **Spotlight boost**: API contract gap. | Add `ping` to SSE Event Types documentation in both README and ARCHITECTURE. Document cadence (15s), payload (empty), and client handling guidance. Consider using standard SSE comment heartbeats (`: keepalive\n\n`) instead of named events for transparent client compatibility. |
| F3 | No OpenAPI representation for SSE streaming or event schemas | `src/api/app.py:146-203`, `src/api/models.py:26-60` | FastAPI's auto-generated OpenAPI spec cannot describe SSE stream semantics. The SSE event schema classes in models.py (SSEMetadataEvent, SSETokenEvent, etc.) are documentation-only — they are never used for serialization or validation. No `operationId`, no tags, no response examples. Integrators using code-generated clients from `/openapi.json` get no useful contract for `/chat`. **Spotlight boost**: API contract gap. | Add `responses` parameter to `/chat` endpoint with `text/event-stream` media type. Define event schemas in OpenAPI via `openapi_extra` or custom schema. Add `operation_id` and `tags` to all endpoints. Reference SSE models in endpoint docstrings. |
| F4 | Webhook responses use ad-hoc dicts without Pydantic models | `src/api/app.py:381-406` (sms), `src/api/app.py:411-438` (cms) | `/sms/webhook` returns `{"status": "ignored"}`, `{"status": "keyword_handled", "response": ...}`, `{"status": "received", "from": ...}` — none tied to Pydantic models. `/cms/webhook` returns `result` dict from handler without model validation. These ad-hoc responses have no type safety, no OpenAPI documentation, and can silently break on refactoring. **Spotlight boost**: API contract gap. | Define `SmsWebhookResponse` and `CmsWebhookResponse` Pydantic models. Add `response_model` to webhook endpoints. Define error response variants. |
| F5 | Error taxonomy (7 ErrorCodes) not documented externally | `src/api/errors.py:1-46`, `README.md`, `ARCHITECTURE.md` | Well-implemented `ErrorCode` enum with 7 codes (unauthorized, rate_limit_exceeded, payload_too_large, agent_unavailable, internal_error, validation_error, service_degraded) and `error_response()` helper producing `{"error": {"code": "...", "message": "..."}}`. None of this is documented in README or ARCHITECTURE API sections. Integrators cannot implement stable error handling. **Spotlight boost**: documentation gap. | Add "Error Responses" section to README and ARCHITECTURE: canonical JSON shape, list all 7 codes, HTTP status mapping, retryability guidance, and examples per endpoint. |

### MEDIUM (8 findings)

| ID | Title | File:Line | Description | Fix |
|---:|-------|-----------|-------------|-----|
| F6 | README test count stale: claims 1090, actual 1198 | `README.md:126,131,271` | Three places in README hardcode "1090 tests". R5 summary reported 1178. pytest --co now collects 1198. README is 108 tests behind reality. Published quality signal is inaccurate. | Update all 3 occurrences. Better: remove hardcoded count and reference CI badge or dynamic count. |
| F7 | README claims "4-step pipeline", actual is 5 steps | `README.md:274`, `cloudbuild.yaml` | README project structure comment says "4-step pipeline". ARCHITECTURE.md correctly says 5-step. cloudbuild.yaml has 5 steps. README is wrong. | Update README:274 to "5-step pipeline". |
| F8 | Hotel agent exists but omitted from all documentation | `src/agent/agents/hotel_agent.py`, `src/agent/agents/registry.py:20`, `src/agent/graph.py:123`, `README.md`, `ARCHITECTURE.md:166-175` | hotel_agent.py (82 lines) is registered in the agent registry, mapped in `_CATEGORY_TO_AGENT`, and has 9 tests in test_agents.py. README and ARCHITECTURE both say "4 specialist agents" and list only host/dining/entertainment/comp. ARCHITECTURE dispatch mapping table (lines 166-175) omits `"hotel": "hotel"`. | Update README and ARCHITECTURE to document 5 specialist agents. Add hotel to dispatch mapping table. Update all "4 specialist agents" references. |
| F9 | ARCHITECTURE /health JSON example missing circuit_breaker_state | `ARCHITECTURE.md:499-508`, `src/api/models.py:62-69` | Health endpoint JSON example in docs shows 6 fields. HealthResponse model has 7 fields (circuit_breaker_state added in R5). Example is stale. | Add `"circuit_breaker_state": "closed"` to the example JSON. Document allowed values (closed, open, half_open, unknown). |
| F10 | TRUSTED_PROXIES default mismatch: docs say `[]`, code has `None` | `ARCHITECTURE.md:711`, `src/config.py:49` | Fixed in R2 (code changed from `[]` to `None`), but ARCHITECTURE config table still shows default as `[]`. None vs [] have critically different semantics: None=never trust XFF, []=also never trust (empty frozenset). But the description "empty = trust all, for Cloud Run" is actively misleading and dangerous. | Update ARCHITECTURE:711 to `None` with description: "None = never trust XFF headers (use direct peer IP). Set explicitly to Cloud Run LB IPs to trust XFF." |
| F11 | Docs say ApiKeyMiddleware only protects /chat, code protects 4 paths | `ARCHITECTURE.md:566`, `src/api/middleware.py:230` | ARCHITECTURE middleware table says ApiKeyMiddleware validates on `/chat`. Code protects `/chat`, `/graph`, `/property`, `/feedback`. Integrators hitting /graph or /property without API key will get unexpected 401s. | Update ARCHITECTURE to list all 4 protected paths. Ideally generate from `_PROTECTED_PATHS` constant. |
| F12 | Observability naming inconsistent: LangSmith vs LangFuse | `README.md:105,183`, `.env.example:57-59`, `src/config.py:92-94`, `ARCHITECTURE.md` | README:105 says "LangFuse + evaluation framework", README:183 says "LangSmith", .env.example references LANGCHAIN_TRACING_V2 (LangSmith), config.py has LangFuse env vars. ARCHITECTURE says LangFuse. Mixed naming confuses operators on which system is active. | Clarify: LangFuse is primary (config.py has explicit vars). LangSmith is LangChain ecosystem pass-through (LANGCHAIN_TRACING_V2). Document both explicitly. Update README:183 to "LangSmith" → "LangFuse (primary) + LangSmith (optional LangChain tracing)". |
| F13 | VERSION default in ARCHITECTURE.md stale (0.1.0 vs 1.0.0) | `ARCHITECTURE.md:724`, `src/config.py:91` | ARCHITECTURE config table says VERSION defaults to 0.1.0. Code default is 1.0.0 (changed in R1). | Update ARCHITECTURE:724 to `1.0.0`. |

### LOW (1 finding)

| ID | Title | File:Line | Description | Fix |
|---:|-------|-----------|-------------|-----|
| F14 | State field count inconsistently described (13 vs 15) | `README.md:224`, `ARCHITECTURE.md:267,836` | ARCHITECTURE:267 correctly says "13 fields". ARCHITECTURE:836 and README:224 say "15 fields". The 15 count appears to include Pydantic output models (RouterOutput, ValidationResult, ExtractedFields) as "fields", which conflates state schema with derived models. Minor but persistent confusion. | Standardize: "PropertyQAState has 13 fields. 3 Pydantic models (RouterOutput, ValidationResult, ExtractedFields) define structured LLM outputs." |

---

## Summary

| Severity | Count |
|----------|------:|
| CRITICAL | 0 |
| HIGH | 5 |
| MEDIUM | 8 |
| LOW | 1 |
| **Total** | **14** |

### Key Theme: Documentation is a Liability, Not an Asset

The codebase is production-quality. The documentation is not. Across 14 findings, the pattern is consistent: **code evolves, docs don't**. This is not cosmetic — it has concrete operational impact:

1. **VERSION mismatch** (F1): /health reports wrong version, breaking deployment verification
2. **Undocumented events** (F2): clients can't implement complete SSE handling
3. **Missing error taxonomy** (F5): integrators can't build stable error handling
4. **Security scope misdocumented** (F11): unexpected 401s on /graph and /property
5. **Hotel agent invisible** (F8): domain capability undiscoverable from docs

### Actionable Priority

1. **Immediate**: Fix F1 (VERSION in .env files) — this is a runtime bug, not just docs
2. **Before next review**: F2, F5, F8, F9, F10, F11 — all are doc updates, low risk
3. **Before production**: F3, F4 — OpenAPI and webhook models require code changes
4. **Nice-to-have**: F6, F7, F12, F13, F14 — accuracy polish

### What's Working Well

- Error taxonomy implementation (errors.py) is clean and consistent — just needs documentation
- SSE heartbeat pattern (ping) is the right solution for keepalive — just needs documentation
- Hotel agent is well-tested (9 tests in test_agents.py) — just needs documentation
- Middleware stack is production-grade with correct ASGI patterns
- Specialist agent dispatch with priority tie-breaking is deterministic and testable
- Config validation (production secrets, consent HMAC, RAG consistency) is thorough

---

## Score Trend

| Round | Spotlight | Score |
|-------|-----------|------:|
| R1 | General | 67.3 |
| R2 | Security | 61.3 |
| R3 | Observability | 60.7 |
| R4 | Domain | 66.7 |
| R5 | Scalability & Async | 63.3 |
| R6 | API Contract & Documentation | **58.0** |

R6 drop reflects spotlight-weighted assessment of documentation drift. The code quality has improved steadily (R5 fixes were substantial), but documentation has not kept pace. This is a common failure mode: teams optimize for code quality and treat docs as second-class. In a regulated casino environment, documentation IS the contract.
