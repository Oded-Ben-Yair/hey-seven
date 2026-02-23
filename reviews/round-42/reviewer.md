# R42 Deep Review: D1 Graph Architecture + D4 API Design + D10 Domain Intelligence

**Reviewer**: Claude Opus 4.6 (reviewer teammate)
**Cross-validated with**: GPT-5.2 Codex (azure_code_review)
**Commit**: a9b5fc1
**Date**: 2026-02-23
**Prior scores**: D1=8.5, D4=8.5, D10=8.5

---

## D1 Graph Architecture (weight 0.20, prior 8.5)

### Strengths (maintaining 8.5)
- 11-node StateGraph v2.2 topology is clean, well-documented, and production-grade
- Dual-layer compliance gate + router provides defense-in-depth
- Validation loop (generate -> validate -> retry max 1 -> fallback) is textbook
- `_route_after_validate_v2` has a defensive default for unknown validation_result values (line 424-429)
- State parity check at import time catches drift in ALL environments (ValueError, not assert)
- `_DISPATCH_OWNED_KEYS` at module level for zero per-call allocation
- Feature flag architecture (build-time vs runtime) is well-documented with emergency disable path
- Specialist state key filtering (line 358-365) prevents unknown keys from poisoning state
- Per-request `_initial_state()` reset with explicit field list prevents cross-turn leakage
- `_extract_node_metadata()` provides rich SSE observability metadata per node

### Findings

#### D1-M001 (MAJOR): No GraphRecursionError handling
`GRAPH_RECURSION_LIMIT=10` (configurable 2-50) is set on the compiled graph (line 565), but there is zero handling for `GraphRecursionError` anywhere in the codebase. If the validate->generate retry loop somehow exceeds the recursion limit (e.g., due to a bug in retry_count tracking), LangGraph raises `GraphRecursionError` which:
- In `chat()`: propagates as unhandled exception, returning no response
- In `chat_stream()`: caught by the broad `except Exception` (line 825), returning a generic error SSE event

The `retry_count` logic SHOULD prevent this (max 1 retry = max 2 generate calls), but there's no explicit test proving the recursion limit is never hit, and no specific error handling for it.

**Fix**: Add explicit `GraphRecursionError` handling in `chat()` that returns a structured fallback response. Add a test that verifies `retry_count` correctly bounds execution below `recursion_limit`.

**Impact**: +0.1 — defensive gap, not a runtime bug due to retry_count bounds, but a production hardening miss.

#### D1-M002 (MAJOR): No concurrency model documentation in graph.py
The graph module handles concurrent SSE streams (via `_active_streams` tracking in app.py), uses `asyncio.Semaphore(20)` for LLM backpressure in `_base.py`, and has TTL-cached singletons with separate locks. But `graph.py` itself has zero documentation of the concurrency model:
- How many concurrent `chat_stream()` calls can run?
- What happens when MemorySaver (local dev) is used with concurrent threads? (MemorySaver is thread-safe but sequential)
- What's the expected behavior under concurrent writes to the same `thread_id`?

**Fix**: Add a concurrency model docstring in `graph.py` (or a linked ADR) documenting: max concurrent streams, checkpointer thread safety, and `thread_id` conflict behavior.

**Impact**: +0.1 — documentation gap that matters for production operations and incident response.

#### D1-m001 (minor): `route_from_compliance` silently routes unknown query_type values to off_topic
Line 408-409: any non-None, non-greeting query_type falls through to `NODE_OFF_TOPIC` without logging. If a new guardrail type is added to `compliance_gate_node` but not to `route_from_compliance`, it silently misroutes without any warning.

**Fix**: Add `logger.debug("Compliance gate triggered: query_type=%s -> off_topic", query_type)` before the return.

**Impact**: +0.05 — debuggability improvement, not a correctness issue since off_topic is the safe default.

#### D1-m002 (minor): `_keyword_dispatch` does not normalize category casing
Categories from retrieved context metadata are used as-is (line 147-149). If the RAG pipeline ever produces a chunk with `category: "Restaurants"` (capitalized) instead of `"restaurants"`, it would create a separate count bucket and potentially misroute.

**Fix**: Add `.lower()` normalization: `cat = chunk.get("metadata", {}).get("category", "").lower()`.

**Impact**: +0.05 — defensive, depends on RAG ingestion consistency which currently normalizes correctly.

### D1 Score: 8.5 -> **8.8** (with fixes: +0.1 recursion handling, +0.1 concurrency docs, +0.1 minor fixes)

---

## D4 API Design (weight 0.10, prior 8.5)

### Strengths (maintaining 8.5)
- Pure ASGI middleware preserves SSE streaming (no BaseHTTPMiddleware)
- Middleware execution order is documented and correct (line 162-173 in app.py)
- SIGTERM graceful drain with `_active_streams` tracking (R40 fix)
- SSE heartbeat prevents client-side EventSource timeouts
- `aclosing()` wrapper ensures async generator cleanup on timeout/disconnect
- Request body limit with both Content-Length and streaming enforcement
- API key with `hmac.compare_digest()` (timing-attack safe)
- Structured error taxonomy with `ErrorCode` enum
- Request ID sanitization prevents log injection (regex + 64 char limit)
- SSE event schemas documented as Pydantic models (wire format reference)
- Security headers include HSTS, X-Frame-Options, Referrer-Policy, X-Content-Type-Options
- CSP split: strict for API paths, none for static frontend (correct trade-off)

### Findings

#### D4-M001 (MAJOR): No SSE reconnection protocol (Last-Event-ID)
The SSE implementation does not support `Last-Event-ID` for reconnection. When a client disconnects mid-stream (network blip, mobile app backgrounding), the browser's EventSource auto-reconnects but restarts the entire conversation turn from scratch — sending a duplicate message to the LLM. This causes:
- Duplicate LLM API costs
- Duplicate messages in the conversation thread (checkpointer stores both)
- Confusing UX with repeated/partial responses

The SSE events have no `id` field, and the server does not read `Last-Event-ID` from reconnection requests.

**Fix**:
1. Add sequential `id` to each SSE event: `yield {"event": "token", "data": ..., "id": str(seq)}`
2. Read `Last-Event-ID` header on reconnection
3. If reconnection detected, return cached response from checkpointer instead of re-invoking graph

This is a known limitation of EventSource-based SSE. For MVP, documenting it as a known limitation with a `retry: 0` SSE field to disable auto-reconnect is acceptable.

**Impact**: +0.15 — significant production gap for mobile/unreliable network clients.

#### D4-M002 (MAJOR): Missing Permissions-Policy header
SecurityHeadersMiddleware includes HSTS, X-Frame-Options, Referrer-Policy, and X-Content-Type-Options, but is missing `Permissions-Policy` (formerly Feature-Policy). This header restricts browser APIs (camera, microphone, geolocation, payment) that should never be needed by a text-based chat API.

**Fix**: Add `(b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=()")` to `_STATIC_HEADERS`.

**Impact**: +0.05 — defense-in-depth hardening.

#### D4-M003 (MAJOR): OpenAPI/Swagger docs exposed in production
FastAPI auto-generates `/docs` (Swagger UI) and `/openapi.json` by default. The SecurityHeadersMiddleware `_API_PATHS` includes `/docs` and `/openapi.json` (line 208-209), showing awareness of these paths. But there's no mechanism to disable them in production. These endpoints expose:
- Full API schema (endpoint names, parameters, types)
- Internal model names (e.g., `ChatRequest`, `HealthResponse`)
- Attack surface reconnaissance data

**Fix**: In `create_app()`, disable docs in production:
```python
docs_url = "/docs" if settings.ENVIRONMENT == "development" else None
redoc_url = "/redoc" if settings.ENVIRONMENT == "development" else None
app = FastAPI(..., docs_url=docs_url, redoc_url=redoc_url)
```

**Impact**: +0.1 — security hardening for production deployment.

#### D4-m001 (minor): SSE responses lack Cache-Control: no-store
SSE streaming responses from `/chat` do not include `Cache-Control: no-store`. While most proxies don't cache SSE, explicit headers prevent edge cases where CDN/proxy caching causes response replay.

**Fix**: Add `Cache-Control: no-store` header to EventSourceResponse.

**Impact**: +0.05 — defense-in-depth for CDN-fronted deployments.

#### D4-m002 (minor): `/sms/webhook` and `/cms/webhook` missing from SecurityHeaders `_API_PATHS`
Webhook endpoints `/sms/webhook` and `/cms/webhook` are not in `_API_PATHS` (line 208-209), so they don't get CSP headers. While webhooks are server-to-server (no browser), consistency in security headers is a best practice.

**Fix**: Add `/sms/webhook` and `/cms/webhook` to `_API_PATHS`.

**Impact**: +0.05 — consistency, minimal real security impact.

### D4 Score: 8.5 -> **8.9** (with fixes: +0.15 SSE reconnection, +0.1 docs protection, +0.15 headers/minor fixes)

---

## D10 Domain Intelligence (weight 0.10, prior 8.5)

### Strengths (maintaining 8.5)
- 5 casino profiles with real operational data (Mohegan Sun, Foxwoods, Parx, Wynn, Hard Rock AC)
- 185+ guardrail regex patterns across 11 languages (EN, ES, PT, ZH, FR, VI, AR, JP, KO, Hindi, Tagalog)
- State-specific regulatory data: self-exclusion authorities, helplines, options per state (CT, PA, NV, NJ)
- Tribal vs commercial casino distinction (property_type field)
- Per-property branding: persona_name, tone, formality_level, exclamation_limit
- Real self-exclusion phone numbers and URLs for all 4 states
- BSA/AML pattern coverage in 11 languages matching injection + RG coverage
- Age verification patterns in English, Hindi, and Tagalog
- Patron privacy patterns in English, Spanish, and Tagalog
- Knowledge base covers regulations, casino operations, player psychology, and company context
- `get_casino_profile()` returns deepcopy to prevent mutation of global state

### Findings

#### D10-M001 (MAJOR): No casino onboarding checklist or process documentation
Adding a new casino property requires changes across multiple files:
1. `src/casino/config.py` — add CASINO_PROFILES entry
2. `src/agent/prompts.py` — potentially add property-specific helplines
3. `knowledge-base/` — add property-specific data files
4. `data/` — add property JSON file for RAG ingestion
5. `src/config.py` — update default PROPERTY_* settings
6. Environment variables — CASINO_ID, PROPERTY_NAME, etc.

There is no onboarding checklist, validation script, or documentation of the required steps. A missed step (e.g., forgetting to add the casino to CASINO_PROFILES but updating env vars) would result in silent fallback to DEFAULT_CONFIG (Mohegan Sun defaults for a non-Mohegan property).

**Fix**: Create `docs/casino-onboarding.md` with a step-by-step checklist. Add a startup validation that checks `CASINO_ID` exists in `CASINO_PROFILES` (or Firestore) and logs a CRITICAL warning if falling back to defaults.

**Impact**: +0.15 — operational readiness gap for the primary product use case (onboarding new casinos).

#### D10-M002 (MAJOR): No regulatory update process documented
State gaming regulations change frequently (e.g., NJ SB 3401 push notification ban, CT SB 2 AI disclosure). The knowledge base at `knowledge-base/regulations/state-requirements.md` contains current regulatory data, but there is no documented process for:
- Who monitors regulatory changes?
- How are changes propagated to guardrails, config, and knowledge base?
- What's the testing/validation process after a regulatory update?
- What's the SLA for regulatory updates (e.g., within 30 days of effective date)?

**Fix**: Add `docs/regulatory-update-process.md` documenting the monitoring cadence, update workflow, and testing requirements. At minimum, add a `_regulatory_version` field to each casino profile with a date stamp.

**Impact**: +0.1 — operational process gap in a regulated industry.

#### D10-M003 (MAJOR): Cross-state patron handling is undefined
When a guest who has a profile at a CT casino (Mohegan Sun) visits a NJ casino (Hard Rock AC):
- Which state's helplines should be displayed? (NJ, since that's the current property)
- Does the CT self-exclusion apply? (No — self-exclusion is state-specific)
- Should the guest profile persist across properties? (Privacy concern)

The current implementation isolates by `CASINO_ID` which is correct for single-property deployment. But the multi-property architecture (5 profiles) implies cross-property guest movement is a real scenario, and the system has no documentation or handling for it.

**Fix**: Add a "Cross-Property Guest Movement" section to domain documentation. At minimum, document that:
1. Self-exclusion is property/state-specific — no cross-state checking
2. Guest profiles are per-casino-id (no sharing by design)
3. Helplines always reflect the CURRENT property's state

**Impact**: +0.1 — domain completeness for the multi-property architecture.

#### D10-m001 (minor): Knowledge base missing per-casino data files
`knowledge-base/` contains generic casino operations, regulations, player psychology, and company context. But the 5 configured casinos (Mohegan Sun, Foxwoods, Parx, Wynn, Hard Rock AC) only have property data for the default (`data/mohegan_sun.json`). The RAG pipeline ingests from `PROPERTY_DATA_PATH` which is a single file. There are no casino-specific knowledge base files for the other 4 casinos.

**Fix**: Either create property-specific data files (even minimal) for all 5 configured casinos, or document that only Mohegan Sun has RAG data and the other profiles are for multi-property config demonstration only.

**Impact**: +0.05 — documentation of intentional scope limitation.

#### D10-m002 (minor): CT DCP vs Tribal Gaming Commission inconsistency in state-requirements.md
Line 48 of `state-requirements.md` states: "CT Department of Consumer Protection handles patron self-exclusion (applies to both tribal casinos)". But the casino profiles in `config.py` correctly use tribal-specific authorities:
- Mohegan Sun: "Mohegan Tribal Gaming Commission" (line 221)
- Foxwoods: "Mashantucket Pequot Tribal Nation Gaming Commission" (line 297)

The R39 fix comments acknowledge this distinction, but the knowledge base file was not updated to match. The knowledge base says DCP; the config says tribal commissions.

**Fix**: Update `state-requirements.md` line 48 to clarify: "CT DCP handles the shared self-exclusion DATABASE, but each tribal casino's gaming commission is the self-exclusion AUTHORITY for their property."

**Impact**: +0.05 — correctness of knowledge base content.

### D10 Score: 8.5 -> **8.9** (with fixes: +0.15 onboarding, +0.1 regulatory process, +0.1 cross-state docs, +0.1 minor fixes)

---

## Summary

| Dimension | Prior | Proposed | Delta | Key Finding |
|-----------|-------|----------|-------|-------------|
| D1 Graph Architecture | 8.5 | 8.8 | +0.3 | GraphRecursionError handling, concurrency docs |
| D4 API Design | 8.5 | 8.9 | +0.4 | SSE reconnection, OpenAPI production disable, Permissions-Policy |
| D10 Domain Intelligence | 8.5 | 8.9 | +0.4 | Casino onboarding checklist, regulatory update process, cross-state docs |

### Finding Counts
- **CRITICAL**: 0
- **MAJOR**: 7 (D1: 2, D4: 3, D10: 3 — note: all are "near-ceiling" improvement opportunities, none are production-blocking bugs)
- **minor**: 5 (D1: 2, D4: 2, D10: 2)

### Cross-Validation Notes
GPT-5.2 Codex flagged:
1. Middleware execution order concern — CONFIRMED as correctly documented (Starlette reverse order). BodyLimit is added LAST so it executes FIRST (outermost). The code comments at lines 162-173 of app.py explain this correctly.
2. Rate limiting before auth — ACCEPTED trade-off for per-IP limiting (documented in ADR).
3. Reducer type safety — CONFIRMED as adequately handled by `_keep_max`'s None guards and `_merge_dicts`'s None/empty filtering.
4. OpenAPI docs exposure — CONFIRMED and included as D4-M003.
5. Security headers incomplete — PARTIALLY confirmed (Permissions-Policy missing, but HSTS/X-Frame/Referrer already present).
