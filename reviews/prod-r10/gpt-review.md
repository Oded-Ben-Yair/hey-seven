# GPT-5.2 Final Adversarial Review — Round 10 (Production Sign-off Gate)

**Date**: 2026-02-20
**Model**: GPT-5.2 (Azure AI Foundry)
**Reviewer Mode**: MAXIMALLY HOSTILE — all findings +1 severity
**Previous Score**: R9 avg=67.1 (Gemini=67, GPT=70.5, Grok=63.9)

---

## Scores (0-10 each)

| Dimension | Score | Notes |
|-----------|-------|-------|
| 1. Graph Architecture | 4/10 | Static feature flag at build time, routing tie-breaker gameable |
| 2. RAG Pipeline | 5/10 | Cross-tenant retriever caching risk, content inference via deterministic hashes |
| 3. Data Model & State | 4/10 | __debug__ parity check vanishes with -O, state validation gaps |
| 4. API Design | 4/10 | Health endpoint leaks security posture, /graph exposes guardrail structure |
| 5. Testing Strategy | 6/10 | Coverage high but no regulatory invariant tests |
| 6. Docker & DevOps | 5/10 | HEALTHCHECK local-only, no prod health strategy documented |
| 7. Prompts & Guardrails | 4/10 | PII buffer is a privacy placebo, semantic injection runs too late |
| 8. Scalability & Production Readiness | 3/10 | Streaming leaks, validator bypass, key rotation window |
| 9. Trade-off Documentation | 3/10 | Missing DPA/data residency docs for LangFuse, quiet-hours risk undocumented |
| 10. Domain Intelligence | 5/10 | Timezone mapping unreliable for TCPA, consent chain in-memory only |

**Total: 43/100**

---

## P0 — Critical (crash, data leak, regulatory breach)

### P0-F1: Static feature flag captured at build time causes unsafe behavior drift
- **File**: `src/agent/graph.py: build_graph` (line ~299-309)
- **Description**: `whisper_planner_enabled` is decided from `DEFAULT_FEATURES` at graph build time, not per-request/runtime. In production, toggling flags won't take effect without restart. Worse: a rollout intended to disable planner during a privacy incident won't stop it.
- **Fix**: Evaluate whisper flag at runtime inside the node before using planner output; or rebuild graph per request (expensive but safest). At minimum, re-check flag inside the whisper_planner_node before executing.
- **Note**: R9 intentionally kept this as "topology-level flag". GPT-5.2 escalates: in a regulated environment, inability to disable a component without restart is a P0.

### P0-F2: SSE stream can leak sensitive content before redaction/validation
- **File**: `src/api/app.py:/chat` + `src/agent/graph.py: chat_stream` (line ~537-563)
- **Description**: Model output chunks are streamed per token. PII/prohibited content emitted before persona/output redaction reaches the client and logs/observability. The "PII buffer" digit detection is not actual PII detection.
- **Fix**: Enforce a streaming gate: buffer model tokens until output guardrails + PII redaction pass, then release. Or implement incremental PII filter (regex+ML) with rollback suppression.

### P0-F3: PII "digit detection" buffering is a privacy placebo
- **File**: `src/agent/graph.py: chat_stream` (line ~493-563)
- **Description**: Triggering buffering on digits + fixed flush sizes misses emails, names, IDs without digits. Also flushes partial sensitive sequences. Demonstrably non-compliant for patron privacy.
- **Fix**: Replace with real incremental redaction strategy: streaming-safe regex automaton. Do not emit until validated. Treat any PII detection as block+fallback.

### P0-F4: Degraded-pass validator explicitly allows unsafe responses
- **File**: `src/agent/nodes.py: validate_node` (line ~341-366)
- **Description**: "First attempt validator fail = PASS" is a direct safety bypass. Network flakiness becomes a free pass to emit potentially disallowed/regulated content.
- **Fix**: Fail-closed on validator failure. If you must degrade, degrade to safe fallback response (contact info / human handoff), not PASS.

### P0-F5: Compliance gate priority ordering allows semantic injection after patron privacy / AML checks
- **File**: `src/agent/compliance_gate.py` (line ~46-144)
- **Description**: Semantic injection classifier runs after patron privacy / AML checks (position 8 in priority chain). Prompt injection should be earliest because it can subvert downstream logic, including privacy and AML behaviors.
- **Fix**: Move injection detection (pattern + semantic) to the earliest stage after empty/turn-limit. If injection suspected: fail-closed + safe response.

### P0-F6: Health/readiness endpoint leaks internal security posture
- **File**: `src/api/app.py:/health` (line ~223-275)
- **Description**: Exposing `circuit_breaker_state`, `rag_ready`, `property_loaded` publicly enables targeted attacks (time outages, degrade validation, prompt injection timing). Operational security leakage in regulated environment.
- **Fix**: Restrict `/health` to internal network or require API key. Publish only boolean "ok" externally.

### P0-F7: API key middleware refresh TTL allows key rotation window abuse
- **File**: `src/api/middleware.py: ApiKeyMiddleware` (line ~240-248)
- **Description**: 60s caching implies a period where revoked keys may still pass. In incident response, you need immediate revocation.
- **Fix**: Support push-based cache invalidation or shorter TTL in prod (<=5s) + denylist; or validate against KMS/secret store each request.

### P0-F8: LangFuse callback may export regulated conversation data offsite
- **File**: `src/agent/graph.py: chat` + `src/observability/langfuse_client.py`
- **Description**: No mention of content minimization, PII scrubbing before sending traces, data residency, DPA, retention controls. In casinos, this is a compliance violation. Sampling "10% prod" is still data leakage.
- **Fix**: Default to metadata-only tracing; redact content fields; make tracing opt-in per property; document retention and region; add hard kill-switch.

### P0-F9: SMS webhook idempotency cap weaponizable for TCPA liability
- **File**: `src/sms/webhook.py: WebhookIdempotencyTracker` (line ~262-291)
- **Description**: Hard cap 10K + TTL cleanup means an attacker can flood unique IDs to evict real ones, causing duplicate consent/STOP handling or repeated messages (TCPA liability).
- **Fix**: Use per-sender bucketing + persistent store (Redis). Rate-limit by sender and signature-valid requests. Treat eviction as "deny" not "allow".

### P0-F10: Quiet hours timezone mapping is a compliance trap
- **File**: `src/sms/compliance.py` (area code mapping)
- **Description**: Area code → timezone mapping is unreliable (number portability, VoIP, overlays). Quiet-hours enforcement based on this can violate TCPA by texting during prohibited hours.
- **Fix**: Use carrier data / telco-provided timezone, or obtain explicit patron timezone. If unknown, default to strictest quiet hours.

### P0-F11: `/graph` introspection endpoint leaks prompt/guardrail structure
- **File**: `src/api/app.py:/graph` (line ~330-361)
- **Description**: Graph structure and node names materially aid prompt injection and bypass strategies, especially when coupled with known node semantics.
- **Fix**: Remove in prod or require admin auth; sanitize output to minimal.

---

## P1 — High (serious risk, likely incident under load/attack)

### P1-F1: Conditional routing tie-breaker can deterministically bias to wrong specialist
- **File**: `src/agent/graph.py:_dispatch_to_specialist` (line ~160-167)
- **Description**: Category counts + `_CATEGORY_PRIORITY` + alphabetical tie-break makes routing predictable and gameable by crafted docs/prompt injection.
- **Fix**: Add confidence thresholds; use calibrated classifier + allowlist by intent.

### P1-F2: get_agent raises KeyError → 500 crash path
- **File**: `src/agent/agents/registry.py` (line ~30-33)
- **Description**: If a routed name is unexpected (LLM structured output drift, new label), KeyError bubbles into unhandled 500.
- **Fix**: Convert to controlled fallback: unknown agent → fallback node + logging.

### P1-F3: Circuit breaker deque without maxlen can balloon memory under edge conditions
- **File**: `src/agent/circuit_breaker.py` (line ~58)
- **Description**: "Bounded by prune" is not a bound if prune conditions fail (clock skew, missing prune call, exception path). Under sustained errors, memory leaks.
- **Fix**: Use `deque(maxlen=N)` plus prune; enforce hard cap always.

### P1-F4: `@lru_cache` on greeting category builder risks stale data + cross-tenant bleed
- **File**: `src/agent/nodes.py: _build_greeting_categories` (line ~437-465)
- **Description**: Cached without keying on property_id. Multi-tenant deployment shows wrong categories across properties.
- **Fix**: Key cache by `property_id` + property version hash; or remove caching for tenant-specific data.

### P1-F5: Retriever singleton dict TTL cache risks cross-tenant retrieval mixing
- **File**: `src/rag/pipeline.py: _get_retriever_cached` (line ~759-832)
- **Description**: Singleton keyed "default" without property_id can reuse retriever with wrong property filter. Catastrophic data leak between casino properties.
- **Fix**: Ensure cache key includes property_id + environment + index version; add tenant isolation tests.

### P1-F6: Ingestion ID hashing enables content inference attacks
- **File**: `src/rag/pipeline.py` (line ~570-575)
- **Description**: Deterministic SHA-256 of normalized content allows attacker with partial corpus guesses to confirm presence (membership inference).
- **Fix**: Salt hashes with per-property secret; avoid exposing IDs.

### P1-F7: `retrieve_node` thread can outlive request
- **File**: `src/agent/nodes.py: retrieve_node` (line ~252-265)
- **Description**: `asyncio.wait_for` cancels the coroutine, not the underlying thread. Threads accumulate under load and consume memory/CPU.
- **Fix**: Use async-native client or bounded threadpool with cancellation-aware design; add backpressure.

### P1-F8: Rate limit per IP is unsafe behind proxies; TRUSTED_PROXIES is a footgun
- **File**: `src/api/middleware.py: RateLimitMiddleware` (line ~316-337)
- **Description**: Misconfig = rate-limit all users as one IP (DoS) or trust spoofed headers (bypass). Common in regulated deployments.
- **Fix**: Fail-closed if behind proxy and not configured correctly; use mTLS ingress identity; add startup validation.

### P1-F9: CSP includes `unsafe-inline`
- **File**: `src/api/middleware.py: SecurityHeaders` (line ~185-197)
- **Description**: Direct XSS enabler if any HTML is served (static files are mounted). Not acceptable.
- **Fix**: Remove `unsafe-inline`; use nonces/hashes; separate static domain.

### P1-F10: `/live` always 200 can mask dead dependencies
- **File**: `src/api/app.py:/live` (line ~215-217)
- **Description**: Kubernetes keeps routing traffic while app can't serve (LLM down, RAG down), increasing user harm and noisy failures.
- **Fix**: Make liveness meaningful or rely solely on readiness; detect event loop stall.

---

## P2 — Medium (reliability, compliance edge cases)

### P2-F1: Output redaction only in persona node; other nodes can emit unredacted
- **File**: `src/agent/persona.py` + streaming
- **Description**: Off-topic, fallback, greeting outputs may bypass the persona envelope path depending on routing.
- **Fix**: Centralize output redaction at the final emission layer (API boundary), not inside one node.

### P2-F2: Unicode normalization can alter meaning and break legal keywords
- **File**: `src/agent/guardrails.py`
- **Description**: NFKD + combining mark removal can change Spanish words/STOP equivalents. Can be abused to evade keyword detection or cause false positives.
- **Fix**: Maintain raw text for compliance keywords; use normalization only for injection patterns with careful equivalence classes.

### P2-F3: STOP/HELP handling ordering not structurally guaranteed
- **File**: `src/sms/webhook.py`
- **Description**: "Mandatory keyword handling before LLM" is claimed but regressions here are catastrophic TCPA liability. Needs hard guarantees.
- **Fix**: Add invariant tests + code structure that cannot call LLM before keyword gate (separate endpoint path).

### P2-F4: RequestBodyLimitMiddleware chunked edge cases
- **File**: `src/api/middleware.py`
- **Description**: If server/proxy passes chunked without Content-Length, streaming cap may not enforce correctly.
- **Fix**: Add tests for chunked uploads; ensure early termination and connection close.

### P2-F5: __debug__ parity check not a safety mechanism
- **File**: `src/agent/graph.py:_initial_state` (line ~382-389)
- **Description**: Assertions disappear with `python -O`, so parity check vanishes in prod. Invalid state causes runtime KeyErrors deeper in graph.
- **Fix**: Replace with runtime validation (Pydantic model or explicit checks) always-on.

### P2-F6: Specialist execution sets skip_validation=True on many errors
- **File**: `src/agent/agents/_base.py` (line ~173-184)
- **Description**: Network errors leading to skipping validation can emit unsafe content if partial outputs leak.
- **Fix**: Never skip validation for generated content; degrade to fallback.

### P2-F7: Replay protection windows without clock sync enforcement
- **File**: `src/sms/webhook.py`, `src/cms/webhook.py`
- **Description**: If system clock drifts, valid requests fail or attackers succeed.
- **Fix**: Enforce NTP/chrony; reject if drift detected; include monotonic counters where possible.

---

## P3 — Low (hardening, maintainability, audit gaps)

### P3-F1: Tests don't prove regulatory invariants
- **Description**: 90%+ coverage means nothing without invariant tests: "no SMS outside quiet hours", "STOP always blocks", "no PII in traces", "no cross-tenant retrieval".
- **Fix**: Add property-based tests and end-to-end contract tests asserting those invariants.

### P3-F2: Docker HEALTHCHECK "for local docker" suggests prod blind spot
- **File**: `Dockerfile`
- **Fix**: Provide production health strategy docs + enforce mandatory readiness.

### P3-F3: Static files mounted last can create cache + CSP mismatch issues
- **File**: `src/api/app.py`
- **Fix**: Serve static via CDN; set immutable cache; lock down MIME types.

---

## Summary

**Total: 43/100**

| Category | Count |
|----------|-------|
| P0 (Critical) | 11 |
| P1 (High) | 10 |
| P2 (Medium) | 7 |
| P3 (Low) | 3 |
| **Total findings** | **31** |

### Not Shippable

This is NOT shippable for a regulated casino environment primarily due to:

1. **Streaming leaks before redaction/validation** (P0-F2, P0-F3)
2. **Validator "degraded-pass" safety bypass** (P0-F4)
3. **Runtime feature flag immutability for privacy-critical behavior** (P0-F1)
4. **Observability export risk (LangFuse) without proven redaction/data controls** (P0-F8)
5. **TCPA quiet-hours timezone inference risk** (P0-F10)
6. **Cross-tenant caching/retriever isolation risks** (P1-F4, P1-F5)

### Must-Fix Before Production (Top 6)

1. Real incremental PII redaction on SSE stream (not digit buffering)
2. Fail-closed on validator failure (no degraded-pass)
3. Runtime feature flag check for whisper_planner inside node
4. LangFuse content redaction/minimization + DPA documentation
5. Health endpoint restricted to internal network
6. Cross-tenant cache isolation (greeting categories + retriever keyed by property_id)
