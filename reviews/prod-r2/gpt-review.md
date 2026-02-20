# Round 2 Review — Security Hardening Spotlight

**Reviewer:** GPT-5.2 (Azure AI Foundry)
**Commit:** d3c1a89
**Date:** 2026-02-20
**Focus:** Security hardening (all security findings receive +1 severity bump)

---

## Summary

R1 fixes were solid on their own terms (CONSENT secret hard-fail, PII stream buffering, SSE heartbeat, middleware ordering, retrieval timeout, CB sizing). Good.

But under a regulated casino threat model, this build still has **multiple "ship-stopper" security gaps**: **auth can be silently disabled**, **CORS is dev-only but not enforced for prod**, **CSP allows unsafe-inline**, **webhooks can be accepted unsigned depending on config**, and **prompt-injection defenses fail open in the semantic layer**.

Security findings get +1 severity bump as instructed.

---

## Dimension Scores

| Dimension | R1 | R2 | Delta | Notes |
|---|---:|---:|:--:|---|
| 1. Graph/Agent Architecture | 7.0 | 7.0 | = | Structure ok; security invariants not enforced end-to-end |
| 2. RAG Pipeline | 7.0 | 6.0 | -1 | Ingestion path/updates are a security surface; insufficient validation |
| 3. Data Model / State Design | 7.3 | 6.8 | -0.5 | Better limits, but state still can carry unsafe user-controlled fields |
| 4. API Design | 6.7 | 5.5 | -1.2 | Authn/authz + webhook + CORS/CSP issues |
| 5. Testing Strategy | 6.7 | 6.0 | -0.7 | Not seeing targeted security tests (auth bypass, CORS, webhook replay) |
| 6. Docker & DevOps | 6.3 | 5.8 | -0.5 | Cloud Run deployed --allow-unauthenticated while API key may be empty |
| 7. Prompts & Guardrails | 8.0 | 7.5 | -0.5 | Regex layer ok; semantic layer fails open; output constraints unclear |
| 8. Scalability & Production | 4.7 | 5.2 | +0.5 | Max instances + better timeouts help; still no real per-tenant controls |
| 9. Documentation & Code Quality | 6.0 | 6.2 | +0.2 | Reasonable; still missing security runbooks & env hardening guidance |
| 10. Domain Intelligence | 7.7 | 7.7 | = | Compliance categories exist; enforcement consistency is the issue |
| **Total** | **67.3** | **64.7** | **-2.6** | Security spotlight exposed regressions/holes |

---

## Detailed Findings

### 1) Graph/Agent Architecture (R2: 7.0, Delta =)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| G1 | **HIGH** | Fail-open behavior for semantic injection classification | `src/agent/guardrails.py` `classify_injection_semantic()` — "FAILS OPEN" | Fail **closed** above a risk threshold: if semantic classifier errors, treat as suspicious or require deterministic-only path with stricter routing. |
| G2 | **MEDIUM->HIGH** | Compliance gate priority order can be gamed by injection + domain triggers | `src/agent/compliance_gate.py` priority list: turn-limit > empty > injection > gaming > age > BSA/AML > privacy | Add "hard blocks" for certain topics (AML evasion, underage gambling) regardless of injection classification outcome. |
| G3 | **MEDIUM->HIGH** | Whisper planner "fail silent" can suppress safety signals | `src/agent/whisper_planner.py` returns None on ANY failure | Log + metric + fallback to deterministic checks; don't silently remove controls. |

**R1 fixes acknowledged:** Streaming PII buffering + heartbeats reduced accidental leakage/timeouts.

---

### 2) RAG Pipeline (R2: 6.0, Delta -1)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| R1 | **HIGH** | CMS-driven content updates are a prompt-injection supply chain unless aggressively sanitized | `src/cms/webhook.py` (updates + re-indexing) + `src/api/app.py:/cms/webhook` | Add content sanitization + allowlists per field; strip instructions ("ignore previous", "system:", tool directives), store provenance, gate publication with review/audit trail. |
| R2 | **MEDIUM->HIGH** | Property ingestion on startup based on filesystem presence can be manipulated | `src/api/app.py` lifespan: checks `data/chroma/chroma.sqlite3` then ingests | Make ingestion an explicit deploy step; verify hashes/signatures of the property data file; run as read-only FS. |
| R3 | **MEDIUM->HIGH** | Low relevance threshold allows low-signal retrieval in adversarial queries | `src/config.py` `RAG_MIN_RELEVANCE_SCORE=0.3` | Raise threshold; add retrieval "refusal" when relevance is poor; add citation-based answering with "no answer in KB" hard rule. |

**R1 fix acknowledged:** Retrieval timeout (10s) is good.

---

### 3) Data Model / State Design (R2: 6.8, Delta -0.5)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| D1 | **MEDIUM->HIGH** | User-controlled fields may flow into logs/state without strict schema limits | `src/api/middleware.py` sanitizes `x-request-id`; `src/api/app.py` logs feedback comment (redacted) | Add strict pydantic constraints on all user fields (thread_id length/charset, message max length, forbid control chars). |
| D2 | **LOW** | `PERSONA_MAX_CHARS=0` meaning unclear (disables safety truncation?) | `src/config.py` | Make 0 invalid or interpret explicitly with docs + tests. |

---

### 4) API Design (R2: 5.5, Delta -1.2)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| A1 | **CRITICAL** | API key auth can be disabled by configuration (empty API_KEY => pass-through) | `src/api/middleware.py` `ApiKeyMiddleware`: "Empty API_KEY = auth disabled" | Hard-fail startup in non-dev if `API_KEY` empty. Make `/chat` always require auth. Consider OAuth/JWT with audience. |
| A2 | **HIGH** | Cloud Run deploy explicitly allows unauthenticated access | `cloudbuild.yaml` `--allow-unauthenticated` | Combined with A1, this is an internet-facing, unauthenticated LLM endpoint. Remove `--allow-unauthenticated`; require IAM or an API gateway. |
| A3 | **HIGH** | Telnyx webhook signature verification is conditional on `TELNYX_PUBLIC_KEY` being set | `src/api/app.py:/sms/webhook`: `if telnyx_public_key:` | If env mis-set, forged inbound SMS events are accepted. Hard-fail when `SMS_ENABLED` and no key; always verify when endpoint enabled. |
| A4 | **HIGH** | CMS webhook signature accepts empty secret (no hard-fail) | `src/config.py` `CMS_WEBHOOK_SECRET` defaults to `""`; signature header optional | Empty secret makes HMAC verification meaningless. Startup validation: require non-empty secret; reject if secret missing. |
| A5 | **MEDIUM->HIGH** | CORS is dev-only allowlist with no prod enforcement | `src/config.py` `ALLOWED_ORIGINS=["http://localhost:8080"]` | In prod, either forgotten (breaking clients) or widened to `*` (leaking credentials). Validate: in production, forbid `*`, forbid localhost, require explicit HTTPS origins. |
| A6 | **MEDIUM->HIGH** | SSE endpoint may leak events cross-origin; no cache-control headers | `src/api/app.py:/chat` returns `EventSourceResponse` | SSE is a data exfil channel. Add `Cache-Control: no-store`, ensure correct CORS, consider per-request authz scopes. |
| A7 | **LOW** | Missing explicit JSONResponse import (stability issue) | `src/api/app.py` uses `JSONResponse` but import is inside handlers | Sloppy; may crash and trigger error middleware. Fix import; add tests. |

---

### 5) Testing Strategy (R2: 6.0, Delta -0.7)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| T1 | **MEDIUM->HIGH** | No explicit security regression tests cited | No auth bypass, webhook signature, replay, CORS/CSP tests referenced | Add tests: (1) API_KEY empty in production => startup failure; (2) Telnyx unsigned => 401; (3) timestamp outside tolerance => 401; (4) CMS missing sig => 403; (5) CSP validation. |

---

### 6) Docker & DevOps (R2: 5.8, Delta -0.5)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| K1 | **HIGH** | Only GOOGLE_API_KEY uses Cloud Run secrets; other secrets risk being env vars or defaults | `cloudbuild.yaml` only: `--set-secrets=GOOGLE_API_KEY=...` | Move **all** secrets to Secret Manager and bind them; add startup validation for required secrets based on enabled features. |
| K2 | **MEDIUM->HIGH** | No mention of runtime egress restrictions; SSRF not discussed | No egress controls visible | LLM/RAG systems often become SSRF pivots. Enforce egress allowlists at infra level; ensure no URL-fetch tools exist. |

---

### 7) Prompts & Guardrails (R2: 7.5, Delta -0.5)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| P1 | **HIGH** | Injection regex patterns decent but incomplete; bypass possible beyond NFKD normalization | `src/agent/guardrails.py` `_normalize_input` and `_INJECTION_PATTERNS` | Make semantic classifier fail-closed; add structured policy checks; incorporate "retrieved content is untrusted" instruction + post-answer policy validation. |
| P2 | **MEDIUM->HIGH** | Output PII guardrail (`_validate_output`) coverage unclear; may not cover all regulated data | `src/agent/persona.py` | Expand PII patterns, add negative tests, enforce "no account-specific actions" policy. |

**R1 fix acknowledged:** PII streaming buffer improvement reduces mid-stream leaks.

---

### 8) Scalability & Production (R2: 5.2, Delta +0.5)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| S1 | **MEDIUM->HIGH** | Rate limiting only on `/chat`; other endpoints can be abused | `src/api/middleware.py` RateLimitMiddleware: "Only rate-limit /chat" | Apply rate limits per-route class; separate webhook limits; add global concurrent connection caps (esp SSE). |
| S2 | **MEDIUM->HIGH** | Rate limit bypass vectors via proxy headers if TRUSTED_PROXIES misconfigured | `src/config.py` TRUSTED_PROXIES default `[]` | Hard-require `TRUSTED_PROXIES` in production when behind LB; log effective client IP source. |

---

### 9) Documentation & Code Quality (R2: 6.2, Delta +0.2)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| Q1 | **LOW->MEDIUM** | Missing security runbook: required envs per feature, expected headers, rotation steps | README summary doesn't cover | Add "Production Security Checklist" section; document required secrets + validation behavior. |

---

### 10) Domain Intelligence (R2: 7.7, Delta =)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| I1 | **MEDIUM->HIGH** | Responsible gaming/AML/privacy detection exists but enforcement consistency unclear | `src/agent/guardrails.py` + `compliance_gate.py` | Make compliance gate return explicit refusal templates and block downstream nodes deterministically for regulated topics. |

---

## R1 Fix Assessment

### Well Fixed
- CONSENT_HMAC_SECRET hard-fail when SMS enabled (`src/config.py` validator) -- genuine improvement
- PII streaming buffer in SSE path with digit-detection flushing -- creative solution
- SSE heartbeat (15s ping) -- prevents client-side EventSource timeouts
- Retrieval timeout (10s via `asyncio.wait_for`) -- prevents hung ChromaDB blocking thread pool
- Middleware ordering awareness (RequestBodyLimitMiddleware outermost) -- correct ASGI execution order
- `--max-instances=10` cap on Cloud Run -- prevents cost runaway
- Circuit breaker deque sizing (`max(100, threshold*10)`) -- prevents silent undercount

### Not Good Enough
- Addressed symptoms (timeouts/leaks) but left **core access control and webhook verification** dependent on optional config
- That's how incidents happen in regulated environments

---

## Top 5 Remaining Risks

1. **CRITICAL:** Auth can be disabled (empty API_KEY) + **unauthenticated Cloud Run** deploy = public LLM endpoint (`src/api/middleware.py`, `cloudbuild.yaml`)
2. **HIGH:** Webhook signature verification is optional/misconfig-prone (Telnyx conditional; CMS secret default empty) (`src/api/app.py`, `src/config.py`)
3. **HIGH:** Semantic injection detection **fails open**, defeating "Layer 2" when it matters most (`src/agent/guardrails.py`)
4. **HIGH:** CSP includes `'unsafe-inline'`, widening XSS blast radius if any HTML ever becomes attacker-influenced (`src/api/middleware.py`)
5. **HIGH:** RAG/CMS content pipeline is a durable prompt-injection supply chain without strong sanitization/provenance (`src/cms/webhook.py` + ingestion)

---

## Recommendations (do these before calling it "production")

1. **Make auth non-optional in prod:** startup validation `ENVIRONMENT != development => API_KEY must be set` and remove `--allow-unauthenticated`
2. **Webhook hardening:** if endpoint enabled, require signature headers + non-empty secrets/keys; reject otherwise. Add replay/nonce logging
3. **Fail closed on semantic classifier errors** (or at minimum elevate to "suspicious" and route to refusal)
4. **Fix CSP:** remove `'unsafe-inline'` by using hashed/nonced scripts/styles; add `connect-src` restrictions
5. **Add security tests** that specifically prevent regressions on the above (config validation, auth required, webhook signature required, replay window, CORS/CSP)

---

## Finding Count by Severity

| Severity | Count |
|---|---:|
| CRITICAL | 1 |
| HIGH | 10 |
| MEDIUM->HIGH | 11 |
| LOW->MEDIUM | 1 |
| LOW | 2 |
| **Total** | **25** |
