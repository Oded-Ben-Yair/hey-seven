# Round 2 Summary -- Security Hardening

## Scores

| Dimension | Gemini | GPT-5.2 | Grok | Average |
|---|---:|---:|---:|---:|
| 1. Graph/Agent Architecture | 8 | 7.0 | 7 | 7.3 |
| 2. RAG Pipeline | 6 | 6.0 | 6 | 6.0 |
| 3. Data Model / State Design | 6 | 6.8 | 7 | 6.6 |
| 4. API Design | 4 | 5.5 | 5 | 4.8 |
| 5. Testing Strategy | 9 | 6.0 | 4 | 6.3 |
| 6. Docker & DevOps | 3 | 5.8 | 6 | 4.9 |
| 7. Prompts & Guardrails | 4 | 7.5 | 6 | 5.8 |
| 8. Scalability & Production | 6 | 5.2 | 5 | 5.4 |
| 9. Documentation & Code Quality | 7 | 6.2 | 5 | 6.1 |
| 10. Domain Intelligence | 8 | 7.7 | 7 | 7.6 |
| **Total** | **61** | **64.7** | **58** | **61.2** |

## Consensus Findings Fixed

| # | Finding | Severity | Models | Fix Applied |
|---|---|---|---|---|
| 1 | cloudbuild.yaml missing critical secret injection | CRITICAL | 3/3 | Added `--set-secrets` for `API_KEY`, `CMS_WEBHOOK_SECRET`, `TELNYX_PUBLIC_KEY` in Cloud Run deploy step |
| 2 | Empty string = auth/webhook bypass | CRITICAL | 3/3 | Added `validate_production_secrets()` model validator: hard-fail on startup if `API_KEY` or `CMS_WEBHOOK_SECRET` empty when `ENVIRONMENT != development`. Telnyx key validated when `SMS_ENABLED=True` |
| 3 | Semantic injection classifier fails OPEN | CRITICAL | 3/3 | Changed to fail-CLOSED: on error, returns synthetic `InjectionClassification(is_injection=True, confidence=1.0)`. Moved semantic check AFTER all deterministic guardrails so safety responses (helplines, age info, BSA/AML) are never blocked by classifier failure |
| 4 | IP spoofing via TRUSTED_PROXIES=[] (trust all) | CRITICAL | 3/3 | Changed `TRUSTED_PROXIES` default from `[]` to `None`. `None` = never trust XFF headers (use direct peer IP). Explicit list required to trust specific proxy IPs |
| 5 | Unprotected /feedback endpoint | HIGH | 3/3 | Added `/feedback` to both `_PROTECTED_PATHS` (API key auth) and rate-limited paths |
| 6 | CMS webhook replay attacks | HIGH | 2/3 | Added timestamp-based replay protection to CMS webhook (`X-Webhook-Timestamp` header, 5-minute tolerance window), mirroring SMS webhook's Ed25519 pattern |
| 7 | PII leak via non-streaming replace events | HIGH | 2/3 | Added PII redaction gate on all `replace` events from non-streaming nodes (greeting, off_topic, fallback). Also fixed PII name detection false positives by adding proper-case validation (`_is_proper_name()`) |
| 8 | FeedbackRequest.comment unbounded | HIGH | 2/3 | Added `max_length=2000` to `FeedbackRequest.comment` field |

## Single-Model Critical Findings Fixed

None required -- all CRITICALs were consensus.

## Findings Not Fixed (Accepted Trade-offs / LOW / INFO)

| # | Finding | Severity | Reason Not Fixed |
|---|---|---|---|
| 1 | CSP `unsafe-inline` | HIGH (all 3) | Documented trade-off for single-file demo HTML. No user-generated content rendered as HTML. Production path documented: externalize CSS/JS + nonce-based CSP |
| 2 | In-memory rate limiting / idempotency | HIGH (Gemini, Grok) | Accepted for demo deployment. Production requires Redis/Memorystore. Cloud Run `max-instances=10` limits blast radius |
| 3 | Distributed rate limits across instances | HIGH (Gemini) | Same as above -- in-memory is sufficient for demo, Redis for production |
| 4 | Same LLM for classifier and agent | MEDIUM (Grok) | Config field `SEMANTIC_INJECTION_MODEL` exists but not wired. Acceptable for demo; production should use dedicated classifier model |
| 5 | CORS localhost-only for dev | MEDIUM (Grok, GPT) | Correct for dev. Production override via `ALLOWED_ORIGINS` env var |
| 6 | Static guardrail patterns require redeploy | LOW (Gemini) | Acceptable for demo. CMS webhook enables hot-patching content without redeploy |
| 7 | API key cache 60s stale window | LOW (Grok) | Accepted trade-off for rotation support. Cache invalidation webhook would add complexity |
| 8 | Error response detail level | LOW (Grok) | Minimal risk. Health endpoint details help ops debugging |
| 9 | Missing security regression tests | MEDIUM (GPT) | Partially addressed: added 14 new tests for production validation, replay protection, fail-closed classifier, TRUSTED_PROXIES=None. Full security fuzzing suite is a Phase 5 item |

## Tests After Fixes

- **Pass count**: 1070 (was 1056, +14 new security tests)
- **Coverage**: 90.53%
- **New tests added**:
  - `test_config.py::TestProductionSecretValidation` (6 tests): production startup validation for API_KEY, CMS_WEBHOOK_SECRET, TELNYX_PUBLIC_KEY
  - `test_cms.py::TestWebhookReplayProtection` (5 tests): timestamp validation, stale/future/invalid timestamps, backward compatibility
  - `test_guardrails.py::TestSemanticInjectionClassifier` (2 tests): fail-closed on error, success path
  - `test_middleware.py::TestTrustedProxiesNone` (1 test): XFF ignored when TRUSTED_PROXIES=None

## Files Modified

| File | Changes |
|---|---|
| `src/config.py` | Added `validate_production_secrets()` validator; changed `TRUSTED_PROXIES` default to `None` |
| `src/agent/guardrails.py` | Changed `classify_injection_semantic()` to fail-CLOSED (returns synthetic injection on error) |
| `src/agent/compliance_gate.py` | Moved semantic classifier AFTER all deterministic guardrails (safety responses never blocked by classifier failure) |
| `src/api/middleware.py` | `TRUSTED_PROXIES=None` handling (never trust XFF); added `/feedback` to auth + rate-limited paths |
| `src/api/models.py` | Added `max_length=2000` to `FeedbackRequest.comment` |
| `src/api/app.py` | Pass `X-Webhook-Timestamp` header to CMS webhook handler |
| `src/api/pii_redaction.py` | Added `_is_proper_name()` proper-case validation to prevent false positives on bot text |
| `src/agent/graph.py` | Added PII redaction gate on non-streaming `replace` events |
| `src/cms/webhook.py` | Added timestamp-based replay protection (5-minute tolerance window) |
| `cloudbuild.yaml` | Added `--set-secrets` for `API_KEY`, `CMS_WEBHOOK_SECRET`, `TELNYX_PUBLIC_KEY` |
| `tests/conftest.py` | Added `_disable_semantic_injection_in_tests` autouse fixture |
| `tests/test_config.py` | Added `TestProductionSecretValidation` (6 tests) |
| `tests/test_cms.py` | Added `TestWebhookReplayProtection` (5 tests) |
| `tests/test_guardrails.py` | Added `TestSemanticInjectionClassifier` (2 tests) |
| `tests/test_middleware.py` | Added `TestTrustedProxiesNone`; updated XFF tests for `TRUSTED_PROXIES` changes |
| `tests/test_compliance_gate.py` | Updated tests for semantic injection being disabled by default in tests |
