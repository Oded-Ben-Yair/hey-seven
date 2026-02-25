# R50 Hostile Review — Gemini 3.1 Pro (thinking=high)

**Reviewer**: Gemini 3.1 Pro via `mcp__gemini__gemini-query` (thinking=high), 3 calls
**Date**: 2026-02-24
**Codebase**: 23K+ LOC, 51 modules, 2229 tests, 90.53% coverage
**Last Commit**: `57cfad5` (R49 consensus fixes)

## Score Table

| Dim | Name | Weight | Score | Weighted | Rationale |
|-----|------|--------|-------|----------|-----------|
| D1 | Graph/Agent Architecture | 0.20 | 7.0 | 1.40 | Elite DRY extraction, validation loops, structured routing. CRITICAL: self_harm falls through to generic concierge response. |
| D2 | RAG Pipeline | 0.10 | 9.5 | 0.95 | Per-item chunking, SHA-256 idempotent ingestion, RRF reranking, version-stamp purging. Enterprise-grade. |
| D3 | Data Model | 0.10 | 9.5 | 0.95 | 3 custom reducers, UNSET_SENTINEL tombstone, import-time parity check. Exemplary. |
| D4 | API Design | 0.10 | 9.5 | 0.95 | Pure ASGI middleware, StreamingPIIRedactor with lookahead, RateLimit-before-ApiKey brute-force prevention. |
| D5 | Testing Strategy | 0.10 | 6.5 | 0.65 | 2229 tests, Hypothesis, schema-dispatching mocks. CRITICAL GAP: Zero self_harm tests. |
| D6 | Docker & DevOps | 0.10 | 9.5 | 0.95 | Digest-pinned, --require-hashes, non-root, exec-form CMD, 10 ADRs. Textbook. |
| D7 | Prompts & Guardrails | 0.10 | 7.5 | 0.75 | 185 patterns, 11 languages, semantic classifier with degradation. Detection works but response handler missing. |
| D8 | Scalability & Production | 0.15 | 8.5 | 1.275 | CB Redis L1/L2, TTL jitter, semaphore backpressure, graceful drain. Minor: CB TTL state loss. |
| D9 | Trade-off Documentation | 0.05 | 9.5 | 0.475 | 10 ADRs, inline rationale, feature flag upgrade path. Exemplary engineering hygiene. |
| D10 | Domain Intelligence | 0.10 | 6.0 | 0.60 | Multi-property config, RG escalation. CRITICAL: Casino host that ignores suicidal guest = industry negligence. |

**Weighted Total: 8.95 / 11.0 = 81.4 / 100 (normalized to 1.0 weight sum: 8.14 / 10.0)**

> Note: Weights sum to 1.10 per rubric. Normalized score = 8.14/10.0 = 81.4/100.

---

## Findings

### CRITICAL-D1-001: self_harm query_type has no response handler

**Severity**: CRITICAL
**Dimensions**: D1, D5, D7, D10
**Verified**: Yes (3 Gemini calls + independent code trace)

**Evidence**:
- `compliance_gate.py:148-150` sets `query_type="self_harm"` when `detect_self_harm()` triggers
- `graph.py:520-521` `route_from_compliance()` routes ALL non-greeting/non-None guardrail types to `NODE_OFF_TOPIC`
- `nodes.py:592-696` `off_topic_node()` handles: `bsa_aml`, `patron_privacy`, `gambling_advice`, `age_verification`, `action_request`
- `nodes.py:681-688` The `else` branch returns generic: _"I'm your concierge for {settings.PROPERTY_NAME}"_
- `grep -rn "self_harm" tests/` returns ZERO results
- `grep -rn "988\|crisis\|lifeline\|suicide" src/agent/nodes.py` returns ZERO results

**Impact**: A suicidal guest interacting with the casino host receives a cheerful concierge redirect instead of the 988 Suicide & Crisis Lifeline and crisis intervention resources. This is:
1. A life-safety failure in a regulated gaming environment
2. A massive legal/PR liability for any casino deploying this agent
3. An ethical violation — the detection infrastructure (R49) works but the response layer was never wired

**Root Cause**: R49 added `detect_self_harm()` patterns and compliance gate routing but did NOT add the corresponding response handler in `off_topic_node`. The `route_from_compliance` catch-all at line 521 sends it to `off_topic_node` where it falls through to the generic `else` branch.

**Fix**:
1. Add `elif query_type == "self_harm":` handler in `off_topic_node` with 988 Lifeline information
2. Add at least 3 tests: detection triggers, response contains crisis resources, no false positives
3. Consider: should self_harm log an audit event for compliance reporting?

---

### MAJOR-D8-001: Circuit breaker TTLCache causes hourly state amnesia (without Redis)

**Severity**: MAJOR (without Redis), MINOR (with Redis)
**Dimension**: D8
**Verified**: Yes (code trace)

**Evidence**:
- `circuit_breaker.py:487` — `_cb_cache: TTLCache = TTLCache(maxsize=1, ttl=3600 + _random.randint(0, 300))`
- When TTL expires, `_get_circuit_breaker()` creates a fresh CB starting in `closed` state
- With `STATE_BACKEND=redis`, the new CB syncs from Redis immediately (mitigated)
- Without Redis (local dev, single-instance demo), state is lost hourly

**Impact**: In local/demo mode, a CB that has detected a down LLM backend and opened will silently reset to closed after ~1 hour. The next N requests will hit the dead backend until the new CB trips again.

**Mitigation**: Documented as accepted trade-off for config refresh. `clear_circuit_breaker_cache()` exists for manual reset. Redis sync mitigates in production.

**Recommendation**: Consider separating config refresh TTL from state lifecycle. State should persist independently of config changes.

---

### MINOR-D1-002: Turn-limit guard runs before safety guardrails

**Severity**: MINOR
**Dimension**: D1
**Verified**: Yes (code trace)

**Evidence**:
- `compliance_gate.py:98-104` — Turn limit check at position 1, before all safety guards
- `compliance_gate.py:148-150` — Self-harm check at position 7.5
- If `len(messages) > MAX_MESSAGE_LIMIT` (40), self_harm is never checked

**Impact**: A guest who sends a self-harm message after 40+ turns will get the turn-limit off_topic response instead of crisis resources. This is an extremely unlikely edge case (requires 20+ back-and-forth turns before the crisis message), but in a regulated gaming environment, even edge cases involving life-safety deserve attention.

**Recommendation**: Move safety-critical checks (self-harm, responsible gaming) before the turn-limit guard, or add a separate always-on crisis detection bypass that runs regardless of structural limits.

---

## Invalidated Findings (from initial review — rejected after code verification)

These findings were proposed in the initial review but are INVALID based on actual code evidence:

### REJECTED: "Streaming PII leak"
**Why Invalid**: `streaming_pii.py` implements a proper `StreamingPIIRedactor` with a 120-char lookahead buffer. The `feed()` method buffers text, applies `redact_pii()` to the full buffer, and retains trailing lookahead. `flush()` at end-of-stream redacts remaining buffer. Same `redact_pii()` function used for both streaming and non-streaming paths. No vulnerability.

### REJECTED: "Middleware ordering wrong (ApiKey should precede RateLimit)"
**Why Invalid**: The ordering is an INTENTIONAL R48 security fix. `app.py:184-187` documents: "wrong-key attempts were rejected (401) before rate limiting ran — enabling unlimited API key brute-force." RateLimit before ApiKey ensures brute-force attempts are rate-limited.

### REJECTED: "Webhook auth collision"
**Why Invalid**: `ApiKeyMiddleware._PROTECTED_PATHS = {"/chat", "/graph", "/property", "/feedback"}`. Webhook paths (`/sms/webhook`, `/cms/webhook`) are NOT protected by API key auth. They use their own signature verification (Telnyx HMAC, CMS_WEBHOOK_SECRET). No collision.

### REJECTED: "threading.Lock in Settings blocks event loop"
**Why Invalid**: Lock hold time is bounded to Settings() construction (~1-5ms), executes once per TTL hour. Academic concern with zero measurable impact in an LLM application where network calls take seconds.

### REJECTED: "RouterOutput Literal types missing guardrail query_types"
**Why Invalid**: Correct architectural separation. Compliance gate types (self_harm, bsa_aml, etc.) bypass the router entirely. Router only classifies messages that pass ALL guardrails. Adding guardrail types to RouterOutput would waste tokens and invite hallucinations.

### REJECTED: "Fail-open semantic injection classifier"
**Why Invalid**: `guardrails.py:720-796` shows `classify_injection_semantic` catches ALL exceptions via `_handle_classifier_failure`, which returns `is_injection=True, confidence=1.0` (fail-closed). The function never returns None on error.

### REJECTED: "Unhandled TimeoutError in specialist execution"
**Why Invalid**: `graph.py:396-411` explicitly wraps agent execution in `asyncio.timeout()` with `except TimeoutError` returning a fallback message with `skip_validation: True`.

### REJECTED: "Global lock serialization bottleneck in RateLimitMiddleware"
**Why Invalid**: `middleware.py:400-409` and `508-559` show the lock is only held briefly for structural dict mutations (bucket creation/LRU eviction). Deque operations happen OUTSIDE the lock with zero await points (inherently atomic in asyncio). R39 CRITICAL fix explicitly documents this.

---

## Summary

The codebase demonstrates elite engineering across infrastructure, API design, data modeling, and security hardening. The RAG pipeline, Docker setup, and middleware architecture are genuinely excellent (9.5/10). The circuit breaker with Redis L1/L2 sync and the streaming PII redactor with lookahead buffer show deep distributed systems and security expertise.

However, a single CRITICAL finding — the self_harm response handler gap — severely impacts the score across 4 dimensions. In a regulated gaming environment, a casino host agent that detects suicidal intent but responds with a cheerful concierge redirect is a life-safety failure that must be fixed before any production deployment. The detection infrastructure (R49) works perfectly; only the response layer is missing.

**Top 3 Actions**:
1. **IMMEDIATE**: Add `self_harm` handler in `off_topic_node` with 988 Lifeline + tests
2. **BEFORE GA**: Separate CB config refresh TTL from state lifecycle
3. **CONSIDER**: Move safety-critical checks before turn-limit guard

---

## Methodology

- **Gemini Call 1**: Full hostile review with code excerpts from all 17 files. Initial findings generated.
- **Gemini Call 2**: Focused validation of 6 specific findings against actual code. 1 confirmed CRITICAL, 5 invalidated.
- **Gemini Call 3**: Dimension scoring with verified evidence. Independent findings evaluated (2 of 5 were also invalid based on actual code).
- **Independent verification**: All findings cross-checked against actual code via grep, read, and code trace. 8 false positives rejected.
