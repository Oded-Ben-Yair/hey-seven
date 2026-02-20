# Round 2 Review -- Grok 4 (Security Hardening Spotlight)

**Commit**: d3c1a89
**Reviewer**: Grok 4 (hostile production review)
**Spotlight**: Security Hardening (+1 severity bump on all security findings)

---

## Executive Summary

The 14 fixes from Round 1 show some minimal effort to plug obvious holes -- the CONSENT_HMAC_SECRET hard-fail (preventing a brain-dead default in production), the PII streaming buffer (which at least attempts to mitigate real-time leaks, even if it's clunky), middleware ordering fix, and the Trivy scan in cloudbuild.yaml are all steps up from the prior state. But this is still a leaky ship masquerading as a secure AI casino concierge. The security posture is amateur-hour at best -- full of half-baked defenses, permissive configs, and bypassed controls that invite exploitation. With the +1 severity bump for all security findings this round, several issues escalate to CRITICAL.

---

## Findings

### CRITICAL

**[CRITICAL] SEC-INJ-01: Prompt Injection Defenses Lack Depth and Are Bypassable**
_(Bumped from HIGH)_

The regex-based injection detection in `guardrails.py` has only 11 patterns that miss advanced techniques like multi-language injections, obfuscated payloads (e.g., via Unicode homoglyphs not covered in the weak NFKD normalization), or chained multi-turn prompts. The semantic classification fails open on errors (returns `None`, allowing malicious input through), and while enabled by default (`SEMANTIC_INJECTION_ENABLED=True`), the threshold of 0.8 is arbitrary.

Evidence:
- `audit_input()` normalizes but re-checks the same weak patterns -- no expansion of detection surface after normalization
- `compliance_gate_node` routes to "off_topic" only if patterns match OR semantic confidence >= 0.8 -- no fallback to deny on classifier failure
- `classify_injection_semantic()` explicitly returns `None` on any exception (fail-open)
- No protection against multi-turn prompt manipulation (building toward injection across messages)

In a casino context with regulated data, this invites prompt exfiltration or data extraction attacks.

**Fix**: Layer in more robust defenses like input tokenization entropy checks, mandatory multi-model semantic verification (no fail-open), anomaly detection on prompt patterns. Block on any suspicion, not "fails open." Add multi-turn injection tracking across conversation history.

---

**[CRITICAL] SEC-RATE-01: Rate Limiting Easily Bypassed via Proxy Spoofing**
_(Bumped from HIGH)_

`RateLimitMiddleware` trusts `X-Forwarded-For` blindly when `TRUSTED_PROXIES=[]` (comment says "trust all for Cloud Run"), allowing IP spoofing to bypass rate limits entirely. Only applies to `/chat` (20 req/min), ignoring all other endpoints including webhooks and feedback. No burst handling, no global limits, no distributed state.

Evidence:
- `_get_client_ip` logic: `if not trusted or peer_ip in trusted` evaluates to `True` when `trusted=frozenset()`, meaning any XFF header is trusted from any peer
- Only `/chat` is rate-limited -- `/sms/webhook`, `/cms/webhook`, `/feedback` are completely unprotected
- In-memory only -- resets on container restart, no cross-instance sharing

**Fix**: Set `TRUSTED_PROXIES` to actual Cloud Run load balancer IPs (e.g., `["169.254.1.1"]` or use Cloud Run's built-in metadata). Implement token-bucket algorithm with Redis backing. Extend rate limiting to all mutable endpoints.

---

**[CRITICAL] SEC-SECRET-01: Secret Management Lifecycle Non-Existent Beyond Basics**
_(Bumped from HIGH)_

Secrets use `SecretStr` in `config.py`, and Cloud Run injects via `--set-secrets` (credit for R1 fix on CONSENT_HMAC_SECRET hard-fail), but there's zero lifecycle management: no rotation enforcement, auditing, or expiry checks. Multiple secrets default to empty string with no startup validation. No versioning or KMS integration.

Evidence:
- `GOOGLE_API_KEY: SecretStr = SecretStr("")` -- empty default, no startup validation
- `CMS_WEBHOOK_SECRET: SecretStr = SecretStr("")` -- empty default means CMS webhook accepts unsigned payloads
- `TELNYX_API_KEY: SecretStr = SecretStr("")` -- no validation when `SMS_ENABLED=True`
- No code for periodic rotation or secret version pinning
- `LANGFUSE_SECRET_KEY` passed directly to LangFuse client constructor -- no rotation path

**Fix**: Integrate Google Secret Manager fully with auto-rotation via lifecycle policies. Add startup validation: hard-fail for all critical secrets when `ENVIRONMENT=production`. Pin secret versions in deploy config.

---

### HIGH

**[HIGH] SEC-AUTH-01: Authentication Incomplete and Easily Disabled**
_(Bumped from MEDIUM)_

`ApiKeyMiddleware` protects only three paths (`/chat`, `/graph`, `/property`) with a static `X-API-Key`. If `API_KEY` is empty (the default in config.py), auth is fully disabled, turning the API into a public playground. Unprotected endpoints like `/feedback` accept arbitrary input without auth, enabling spam and abuse.

Evidence:
- `_get_api_key()` returns cached value; if empty, `__call__` proceeds without auth
- Cloud Run deploy uses `--allow-unauthenticated`, exposing everything publicly
- `/feedback`, `/sms/webhook`, `/cms/webhook` have no API key requirement
- No token rotation, revocation, or multi-factor support
- API key comparison is constant-time (`hmac.compare_digest`) -- good, but the key itself is static

**Fix**: Mandate `API_KEY` in production (hard-fail if empty when `ENVIRONMENT=production`). Implement JWT with expiry and claims for production use. Require auth on all mutable endpoints. Add IP allowlisting for webhook endpoints.

---

**[HIGH] SEC-PII-01: PII Redaction Coverage Gaps and Inconsistent Application**
_(Bumped from MEDIUM)_

`pii_redaction.py` patterns are decent for US basics (phone, email, cards, SSN, names), but miss edge cases: international phone formats, physical addresses, dates of birth, or compound PII ("John Doe at 123 Main St"). The streaming PII buffer in `graph.py` (R1 fix) is flawed: buffers only on digits (flushes immediately if no digits, potentially leaking names/emails in real-time); `_PII_FLUSH_LEN=80` is arbitrary. Not applied universally (missing in webhook handlers).

Evidence:
- `_NAME_PATTERNS` are regex-only and require specific prefixes (`my name is`, `Mr.`) -- miss contextual names like "Tell John I'm here"
- `_PII_DIGIT_RE = re.compile(r"\d")` -- non-digit PII (emails, names) bypasses buffering entirely
- `feedback_endpoint` redacts `comment` but not `thread_id` (though validated as UUID, it's still logged)
- SMS `handle_inbound_sms` logs `from_number[-4:]` -- good truncation, but `text` field is not redacted in logs
- No PII redaction in `handle_cms_webhook` logs or CMS webhook processing

**Fix**: Expand patterns to cover international formats (use libraries like presidio). Apply redaction to ALL inputs/outputs including webhook payloads and log messages. Make streaming buffer digit-agnostic -- buffer ALL tokens until sentence boundary for PII check.

---

**[HIGH] SEC-CSP-01: CSP Headers Inadequate and Permit XSS**
_(Bumped from MEDIUM)_

`SecurityHeadersMiddleware` CSP allows `'unsafe-inline'` for both scripts and styles (enabling inline XSS attacks) and `data:` for images (potential data URI injection). No `frame-ancestors` directive to fully block clickjacking. The code comments acknowledge this is a known trade-off for the single-file demo HTML, but production deployment should not ship with these weaknesses.

Evidence:
- CSP: `script-src 'self' 'unsafe-inline'` -- any injected `<script>` tag executes
- CSP: `style-src 'self' 'unsafe-inline'` -- CSS injection possible
- CSP: `img-src 'self' data:` -- data URI attacks possible
- Static files mounted at `/` via `StaticFiles` could reflect user input
- `x-frame-options: DENY` is present but `frame-ancestors` CSP directive is missing (redundancy)

**Fix**: Remove `'unsafe-inline'` and use nonces or hashes for scripts/styles. Add `frame-ancestors 'none'` to CSP. Remove `data:` from `img-src` unless demonstrably needed. Externalize CSS/JS from the single-file HTML.

---

**[HIGH] SEC-WEBHOOK-01: CMS Webhook Verification Lacks Replay Protection**
_(Bumped from MEDIUM)_

SMS webhook uses Ed25519 with 5-minute replay protection (good, with timestamp check in `verify_webhook_signature`). But CMS webhook uses plain HMAC-SHA256 without any timestamp or replay protection, allowing replay attacks. No nonce or IP restrictions on either webhook.

Evidence:
- SMS `verify_webhook_signature`: checks `abs(int(time.time()) - ts_int) > tolerance` (300s) -- good
- CMS `verify_webhook_signature`: only checks `hmac.compare_digest(expected, signature)` -- no timestamp, no replay window
- CMS `handle_cms_webhook`: signature verification only runs when both `raw_body` and `signature` are provided -- could be bypassed if signature header is omitted
- No webhook IP allowlisting for either Telnyx or Google Apps Script

**Fix**: Add timestamp + replay protection to CMS webhook (mirror the SMS pattern). Make signature verification mandatory (not conditional on header presence). Add IP allowlisting for webhook sources. Consider nonce-based deduplication.

---

**[HIGH] SEC-XSS-SSRF-01: Potential XSS/SSRF Vectors in Static Serving and Agent Outputs**
_(Bumped from MEDIUM)_

`StaticFiles` at `/` could serve user-reflected content combined with weak CSP (`'unsafe-inline'` enables XSS). Agent outputs go through `persona_envelope_node` which does PII redaction but no HTML escaping -- if responses are rendered as HTML in the frontend, XSS is possible. No SSRF guards on agent LLM calls or embedding requests.

Evidence:
- `app.mount("/", StaticFiles(..., html=True))` serves static HTML with `html=True` flag
- `_validate_output()` in `persona.py` only does PII redaction via `redact_pii()`, not output encoding
- No Content-Type enforcement on SSE responses (should be `text/event-stream`)
- LLM calls to Gemini API could theoretically be SSRF-exploited if model name is user-controllable (it's not currently, but no validation)
- Retrieved context from RAG is passed directly to LLM without sanitization

**Fix**: Sanitize all LLM outputs with HTML escaping before sending to clients. Block external URL fetches in agent processing. Validate model names against an allowlist. Add Content-Type headers to SSE responses.

---

**[HIGH] SEC-INPUT-01: Input Validation Incomplete and Permissive**
_(Bumped from MEDIUM)_

`ChatRequest` validates message length (1-4096) and `thread_id` UUID format, but no sanitization for injections, scripts, or control characters (relying entirely on the weak guardrails). `FeedbackRequest` validates `thread_id` UUID and rating range but `comment` is unbounded (no `max_length`). Webhook payloads have no schema validation at the API layer.

Evidence:
- `ChatRequest.message`: min_length=1, max_length=4096, but no stripping of control characters (null bytes, ANSI escapes)
- `FeedbackRequest.comment: str | None = None` -- no max_length, could be used for log injection
- SMS webhook `payload` is parsed from raw JSON with no schema validation
- CMS webhook `payload` is parsed with no Pydantic model validation at the API layer (only downstream `validate_item`)

**Fix**: Add strict input sanitization (strip non-printable characters, validate UTF-8). Use Pydantic strict mode. Add `max_length` to `FeedbackRequest.comment`. Add schema validation models for webhook payloads at the API layer.

---

**[HIGH] GEN-02: Cloud Run Deployment Exposes Unauthenticated Access**

`--allow-unauthenticated` in `cloudbuild.yaml` makes the Cloud Run service publicly accessible. The comment says "defense-in-depth: app layer enforces API key auth" but API key is empty by default. Combined, this means the service is wide open.

Evidence:
- `cloudbuild.yaml`: `--allow-unauthenticated` flag
- `config.py`: `API_KEY: SecretStr = SecretStr("")` -- empty default
- No IAM invoke restrictions or VPC connector

**Fix**: Remove `--allow-unauthenticated` for production. Use Cloud Run's built-in IAM for service-level auth. If public access is needed for the demo, enforce API_KEY non-empty in production via startup validation.

---

### MEDIUM

**[MEDIUM] SEC-CORS-01: CORS Policy Incomplete for Production**
_(Bumped from LOW)_

CORS allows only `["http://localhost:8080"]` (good for dev), but in production this blocks legitimate cross-origin requests while providing no configurable production allowlist. No `allow_credentials` control. Missing `Vary: Origin` header for caching correctness.

Evidence:
- `ALLOWED_ORIGINS: list[str] = ["http://localhost:8080"]` -- hardcoded dev default
- No production override mechanism documented
- `CORSMiddleware` configured without `allow_credentials` parameter

**Fix**: Use a configurable allow-list with regex matching (e.g., `*.mohegansun.com`). Set `allow_credentials=False` explicitly. Document production origin configuration.

---

**[MEDIUM] GEN-01: Dockerfile Lacks Security Hardening Beyond Non-Root**

Runs as `appuser` (good), but uses `python:3.12.8-slim-bookworm` instead of distroless. No `--security-opt=no-new-privileges`. HEALTHCHECK is basic (no auth). Single uvicorn worker is fine for demo but lacks process isolation.

Evidence:
- `FROM python:3.12.8-slim-bookworm` -- contains unnecessary packages vs distroless
- No `--security-opt` in docker run config
- HEALTHCHECK hits unauthenticated `/health` endpoint

**Fix**: Consider distroless base for production. Add `no-new-privileges` security option. Consider gunicorn for better process isolation.

---

**[MEDIUM] SEC-LOG-01: Logging May Leak Sensitive Data**

Multiple log statements include partial user data that could leak PII. The logging middleware uses structured JSON which is good, but there's no systematic log redaction pipeline.

Evidence:
- `handle_inbound_sms` logs `from_number[-4:]` (good truncation) but `text` length only
- `compliance_gate_node` logs `semantic_result.reason[:100]` which could contain user message fragments
- `guardrails.py` logs `pattern.pattern[:60]` -- safe (pattern, not input)
- `classify_injection_semantic` logs `len(message)` -- safe
- No centralized log sanitization middleware

**Fix**: Implement a centralized log sanitization layer. Ensure no user input is ever logged in raw form. Audit all `logger.warning` and `logger.info` calls for PII leakage.

---

**[MEDIUM] SEC-SEMANTIC-01: Semantic Injection Classifier Uses Same LLM as Main Agent**

The semantic injection classifier (`classify_injection_semantic`) uses the same LLM instance as the main agent (`_get_llm`). An attacker who can manipulate the LLM (via adversarial inputs or model poisoning) compromises both the classifier and the agent simultaneously.

Evidence:
- `classify_injection_semantic(message, llm_fn=None)` defaults to importing `_get_llm` from `nodes`
- `SEMANTIC_INJECTION_MODEL: str = ""` config exists but is unused -- no code reads it
- Both use the same `ChatGoogleGenerativeAI` with same model name

**Fix**: Use a dedicated, hardened model for the semantic classifier (separate from the conversational model). Actually wire `SEMANTIC_INJECTION_MODEL` to override the classifier's model. Consider using a fine-tuned classifier model.

---

**[LOW] SEC-TIMING-01: API Key Cache Creates Window for Stale Keys**

`ApiKeyMiddleware._get_api_key()` caches the API key for 60 seconds. During key rotation, old keys remain valid for up to 60 seconds after being changed. While the code comment acknowledges this as intentional for rotation support, it creates a window where revoked keys still work.

Evidence:
- `_KEY_TTL = 60` seconds
- No mechanism to force-invalidate the cache on key change

**Fix**: Accept this as a known trade-off for demo deployments. For production, consider a webhook-triggered cache invalidation from Secret Manager.

---

**[LOW] SEC-ENUM-01: Error Responses May Enable Endpoint Enumeration**

Error responses use structured `ErrorCode` enum values that could help attackers enumerate valid paths and states. The 401 response for invalid API keys doesn't distinguish between "no key" and "wrong key" (good), but 429 responses expose the rate limit window.

Evidence:
- `error_response(ErrorCode.RATE_LIMITED, "Too many requests.")` with `Retry-After: 60`
- Health endpoint returns detailed status including `rag_ready`, `observability_enabled`

**Fix**: Minor concern. Consider reducing detail in health responses for production. Accept as low risk.

---

## R1 Fix Acknowledgments

| Fix | Assessment |
|-----|------------|
| CONSENT_HMAC_SECRET hard-fail | Good fix, proper severity |
| PII streaming buffer | Attempt noted but flawed (digit-only buffering) |
| SSE heartbeat pings | Good fix for UX |
| Middleware ordering | Good fix, necessary |
| --max-instances=10 | Good cost guard |
| Circuit breaker deque maxlen | Good fix |
| Retrieval timeout guard | Good fix |
| VERSION bump to 1.0.0 | Cosmetic but correct |
| Business-priority tie-breaking | Good domain fix |
| RAG ingestion guard | Good safety fix |

---

## Scores

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| 1. Graph/Agent Architecture | 7 | R1 fixes like recursion limit and PII buffer help, but injection defense depth is shallow; PII buffer has design flaws (digit-only). |
| 2. RAG Pipeline | 6 | Top-K=5 and relevance=0.3 are reasonable. R1 ingestion guard is good. No security guards on retrieval (SSRF in embeddings untested). |
| 3. Data Model / State Design | 7 | Firestore/Chroma configs exist. Guest profile has CCPA delete. But PII gaps persist across data paths. |
| 4. API Design | 5 | Middleware ordering fixed (R1 credit). Auth/rate-limiting are fundamentally weak with multiple bypasses. Webhook auth inconsistent. |
| 5. Testing Strategy | 4 | cloudbuild.yaml has pytest cov=90 and Trivy scan. But no security-specific tests (injection fuzzing, rate limit bypass, webhook replay). |
| 6. Docker & DevOps | 6 | Non-root user and Trivy scan (good). Lacks distroless base, seccomp profiles, no-new-privileges. |
| 7. Prompts & Guardrails | 6 | 73 regex patterns across 4 languages is good breadth. But injection defense depth is shallow; semantic classifier fails open using same LLM. |
| 8. Scalability & Production | 5 | --max-instances=10 (R1 fix). In-memory rate limiter and circuit breaker are single-container only. No Redis path implemented. |
| 9. Documentation & Code Quality | 5 | Config comments are decent. No security architecture document, no threat model, no security runbook. |
| 10. Domain Intelligence | 7 | Casino-specific guardrails (responsible gaming, BSA/AML, patron privacy) are domain-appropriate. PII/compliance gaps undermine regulatory trust. |
| **Total** | **58** | Up from 52 (R1 fixes helped). Security spotlight reveals how many fundamental gaps remain -- still a vulnerability magnet. |

---

## Finding Summary

| Severity | Count | IDs |
|----------|:-----:|-----|
| CRITICAL | 3 | SEC-INJ-01, SEC-RATE-01, SEC-SECRET-01 |
| HIGH | 7 | SEC-AUTH-01, SEC-PII-01, SEC-CSP-01, SEC-WEBHOOK-01, SEC-XSS-SSRF-01, SEC-INPUT-01, GEN-02 |
| MEDIUM | 4 | SEC-CORS-01, GEN-01, SEC-LOG-01, SEC-SEMANTIC-01 |
| LOW | 2 | SEC-TIMING-01, SEC-ENUM-01 |
| **Total** | **16** | |

---

## Top 3 Actions for Round 3

1. **Fix rate limiting bypass** (SEC-RATE-01): Set `TRUSTED_PROXIES` correctly for Cloud Run or use Cloud Armor. Extend rate limiting to all mutable endpoints.
2. **Harden authentication** (SEC-AUTH-01 + GEN-02): Hard-fail on empty `API_KEY` in production. Remove `--allow-unauthenticated` or enforce IAM. Add auth to `/feedback`.
3. **Fix CMS webhook replay** (SEC-WEBHOOK-01): Add timestamp + replay protection to CMS webhook signature verification, mirroring the SMS pattern. Make signature verification mandatory.
