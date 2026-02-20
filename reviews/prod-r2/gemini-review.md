# Round 2 Review — Security Hardening Spotlight

**Reviewer:** Gemini 3 Pro
**Commit:** d3c1a89
**Date:** 2026-02-20
**Focus:** Security hardening (all security findings receive +1 severity bump)

---

## Summary

You have fixed the functional bugs from Round 1 but introduced (or revealed) catastrophic configuration vulnerabilities. You are relying on "Empty String means Bypass" logic for critical security controls, and your deployment pipeline *guarantees* those strings will be empty. You have effectively deployed a vault with the door unlocked by default.

**Verdict**: REJECTED. The R1 fixes addressed functional issues, but the security spotlight exposes systemic configuration-as-security-bypass patterns.

---

## Dimension Scores

| Dimension | R1 Avg | R2 | Delta | Notes |
|---|---:|---:|:--:|---|
| 1. Graph/Agent Architecture | 7.0 | 8 | +1.0 | retrieve_node timeout and CB sizing good; PII leak via non-streaming nodes |
| 2. RAG Pipeline | 7.0 | 6 | -1.0 | CMS webhook auth bypass due to empty secret default |
| 3. Data Model / State Design | 7.3 | 6 | -1.3 | Ephemeral security state (idempotency, rate limits) in serverless |
| 4. API Design | 6.7 | 4 | -2.7 | IP spoofing via trust-all proxies, unprotected /feedback |
| 5. Testing Strategy | 6.7 | 9 | +2.3 | 1073 tests >90% coverage strong; missing config safety tests |
| 6. Docker & DevOps | 6.3 | 3 | -3.3 | cloudbuild.yaml missing critical secret injection |
| 7. Prompts & Guardrails | 8.0 | 4 | -4.0 | Semantic guardrail fails OPEN — negligent in regulated domain |
| 8. Scalability & Production | 4.7 | 6 | +1.3 | max-instances good; distributed rate limiting broken across instances |
| 9. Documentation & Code Quality | 6.0 | 7 | +1.0 | Readable code, SecretStr good, but dangerous defaults logic |
| 10. Domain Intelligence | 7.7 | 8 | +0.3 | Pattern library is the strongest part of this codebase |
| **Total** | **67.3** | **61** | **-6.3** | Security spotlight exposed systemic configuration bypass |

---

## Detailed Findings

### 1) Graph/Agent Architecture (R2: 8, Delta +1.0)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| G-H1 | **HIGH** | PII leak via non-streaming nodes — `replace` events bypass PII buffer | `graph.py` emits `replace` events for greeting/off_topic/fallback nodes. These bypass the `contains_pii` buffer logic and are emitted directly without redaction. | All node outputs, streaming or static, must pass through PII redaction gate before hitting the wire. |

**R1 fixes acknowledged:** retrieve_node timeout and circuit breaker deque sizing are good improvements.

---

### 2) RAG Pipeline (R2: 6, Delta -1.0)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| R-C1 | **CRITICAL** | CMS webhook authentication bypass — empty secret disables verification | `cms/webhook.py`: `if not signature or not secret: return False`. But `cloudbuild.yaml` does NOT inject `CMS_WEBHOOK_SECRET`. Production defaults to empty string = no signature verification possible. Attackers can inject fake casino content/rules. | Change default to fail-closed (raise Error if secret missing in production). Add secret to `cloudbuild.yaml`. |

---

### 3) Data Model / State Design (R2: 6, Delta -1.3)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| D-H1 | **HIGH** | Ephemeral security state in serverless environment | `sms/webhook.py` uses in-memory `idempotency_tracker`. `middleware.py` uses in-memory rate limiting. Cloud Run scales to zero or restarts. On restart, replay protection and rate limits vanish. Attackers can trigger race conditions during cold starts. | Move idempotency and rate-limit counters to Redis (Memorystore). |

---

### 4) API Design (R2: 4, Delta -2.7)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| A-C1 | **CRITICAL** | IP spoofing via trust-all proxies | `config.py`: `TRUSTED_PROXIES: []`. `middleware.py`: XFF trusted when TRUSTED_PROXIES empty. Any attacker can send `X-Forwarded-For: <Arbitrary_IP>` and rate limiter trusts it. Trivial bypass of all rate limits. | Default `TRUSTED_PROXIES` to `None` (trust no one) or strictly the Google Cloud Load Balancer IP ranges. |
| A-H1 | **HIGH** | Unprotected /feedback endpoint — no auth, no rate limiting | `/feedback` not in `_PROTECTED_PATHS`, rate limiter only checks `/chat`. Trivial DoS vector. Can fill disk/logs with junk data. | Add `/feedback` to protected paths or apply strict rate limiting. |

---

### 5) Testing Strategy (R2: 9, Delta +2.3)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| T-I1 | **INFO** | Missing configuration safety tests | Tests likely mock secrets. No tests verify app refuses to start if `API_KEY` or `CMS_WEBHOOK_SECRET` are unset/empty in production. | Add tests that verify the application crashes on startup if critical secrets are missing. |

---

### 6) Docker & DevOps (R2: 3, Delta -3.3)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| X-C1 | **CRITICAL** | cloudbuild.yaml missing critical secret injection | Step 5 only injects `GOOGLE_API_KEY`. Fails to inject `API_KEY`, `CMS_WEBHOOK_SECRET`, and `TELNYX_PUBLIC_KEY`. Combined with "empty string = disable auth" logic, production deploys with NO AUTHENTICATION on the API and NO SIGNATURE VERIFICATION on webhooks. | Inject all secrets in Cloud Build step 5 via `--set-secrets`. |
| X-H1 | **HIGH** | CSP unsafe-inline negates XSS protection | `middleware.py`: CSP includes `'unsafe-inline'` for script-src and style-src. Negates primary benefit of CSP. | Use nonces or hashes. Remove `unsafe-inline`. |

---

### 7) Prompts & Guardrails (R2: 4, Delta -4.0)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| P-C1 | **CRITICAL** | Semantic guardrail fails OPEN — negligent in regulated domain | `classify_injection_semantic()` returns `None` on ANY error. `compliance_gate.py` proceeds if result is not explicitly injection. If LLM service hangs/errors (common under load), prompt injections pass through directly to agent. | Fail closed: if semantic classification fails, block the request or require deterministic-only path with stricter routing. |

---

### 8) Scalability & Production (R2: 6, Delta +1.3)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| S-H1 | **HIGH** | Distributed rate limiting failure across instances | `middleware.py` uses in-memory limits. `cloudbuild.yaml` sets `--max-instances=10`. A user can hit instance A then B. Effective rate limit is 10x higher than configured. | Redis-backed rate limiting. |

---

### 9) Documentation & Code Quality (R2: 7, Delta +1.0)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| Q-M1 | **MEDIUM** | Dangerous default configurations — empty string = bypass security | `config.py` defaults critical secrets to empty strings, triggering logic to bypass security controls silently. No "fail loud" behavior. | Defaults should be `None` or raise `ValueError` on startup if missing. Force explicit "I want to be insecure" flags for dev rather than implicit defaults. |

---

### 10) Domain Intelligence (R2: 8, Delta +0.3)

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| I-L1 | **LOW** | Static pattern updates require redeployment | Patterns hardcoded in `guardrails.py`. No hot-patching without redeploying code. | Move patterns to external config or database for operational flexibility. |

---

## Finding Summary

| Severity | Count | Findings |
|---|---|---|
| CRITICAL | 4 | CMS auth bypass (R-C1), IP spoofing (A-C1), Missing secrets in deploy (X-C1), Semantic fail-open (P-C1) |
| HIGH | 5 | PII leak non-streaming (G-H1), Ephemeral state (D-H1), Unprotected /feedback (A-H1), CSP unsafe-inline (X-H1), Distributed limits (S-H1) |
| MEDIUM | 1 | Dangerous defaults (Q-M1) |
| LOW | 1 | Static patterns (I-L1) |
| INFO | 1 | Missing config tests (T-I1) |

## Top 3 Critical Fixes Required

1. **Ops/Auth Integration**: Update `cloudbuild.yaml` to inject `API_KEY`, `CMS_WEBHOOK_SECRET`, and `TELNYX_PUBLIC_KEY`. Currently deploying a public API.
2. **Fail-Closed Logic**: Change `config.py` — if a secret is empty in production, crash on startup. Don't disable the security control.
3. **Semantic Guardrail**: Rewrite `compliance_gate.py` to block requests if `classify_injection_semantic` returns `None` or errors out.
