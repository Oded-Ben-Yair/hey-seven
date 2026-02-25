# R49 Hostile Code Review — DeepSeek R1 (Extended Reasoning)

**Reviewer**: DeepSeek R1 via `mcp__azure-ai-foundry__azure_deepseek_reason` (thinking_budget=extended), 2 calls
**Date**: 2026-02-24
**Codebase Version**: v1.1.0, commit `2a2dd99` (main + uncommitted R49 fixes)
**Protocol**: 2x DeepSeek extended reasoning calls + manual code verification of all findings
**Orchestrator**: Claude Opus 4.6

---

## Methodology

1. **Call 1**: Full 10-dimension review with extended thinking. Identified 6 findings across all dimensions.
2. **Verification**: Manually verified all Call 1 findings against actual source code. Identified 1 false positive (semaphore count inflation in `_base.py` — the acquire and release are in separate try blocks; timeout returns before reaching the finally block).
3. **Call 2**: Deep-dive on 5 nuanced findings with specific line references and code snippets. All 5 verified as real bugs.
4. **Cross-check**: All findings verified against the R48 "already fixed" list to avoid re-flagging.

### False Positive Caught and Excluded

DeepSeek Call 1 flagged `_LLM_SEMAPHORE` in `_base.py` as a CRITICAL — claiming that `finally: _LLM_SEMAPHORE.release()` would execute even on `asyncio.wait_for()` timeout, inflating the semaphore count. Manual verification disproved this:

- Lines 328-339: `try: await asyncio.wait_for(sem.acquire(), timeout=...) except TimeoutError: return fallback`
- Lines 341-381: `try: ... finally: sem.release()` — a SEPARATE try block
- On timeout, the function returns at line 337, **before ever entering** the second try block. The finally clause only executes for code that successfully acquired the semaphore.

This finding was excluded from the final review. DeepSeek's extended reasoning conflated the two try blocks into one.

---

## Scores Table

| Dim | Name | Weight | Score | Weighted | Findings |
|-----|------|--------|-------|----------|----------|
| D1 | Graph/Agent Architecture | 0.20 | 7.5 | 1.500 | 0C 1M 0m |
| D2 | RAG Pipeline | 0.10 | 8.0 | 0.800 | 0C 0M 0m |
| D3 | Data Model | 0.10 | 6.5 | 0.650 | 1C 0M 0m |
| D4 | API Design | 0.10 | 7.0 | 0.700 | 0C 1M 0m |
| D5 | Testing Strategy | 0.10 | 6.5 | 0.650 | 0C 1M 0m |
| D6 | Docker & DevOps | 0.10 | 8.0 | 0.800 | 0C 0M 0m |
| D7 | Prompts & Guardrails | 0.10 | 6.0 | 0.600 | 1C 0M 0m |
| D8 | Scalability & Production | 0.15 | 6.5 | 0.975 | 1C 1M 0m |
| D9 | Trade-off Docs | 0.05 | 7.5 | 0.375 | 0C 0M 0m |
| D10 | Domain Intelligence | 0.10 | 7.5 | 0.750 | 0C 0M 0m |
| **TOTAL** | | **1.00** | | **7.80** | **3C 3M 0m** |

**Weighted Score: 7.80 / 10.0 (78.0 / 100)**

Calculation: (7.5*0.20) + (8.0*0.10) + (6.5*0.10) + (7.0*0.10) + (6.5*0.10) + (8.0*0.10) + (6.0*0.10) + (6.5*0.15) + (7.5*0.05) + (7.5*0.10)
= 1.500 + 0.800 + 0.650 + 0.700 + 0.650 + 0.800 + 0.600 + 0.975 + 0.375 + 0.750 = 7.800

---

## Detailed Findings

### CRITICAL-D7-001: Restricted-mode classifier returns confidence=0.5 which is below the compliance gate threshold (0.8) — effectively FAIL-OPEN

**File**: `src/agent/guardrails.py` (restricted mode return) + `src/agent/compliance_gate.py:159-163` (threshold check)
**Severity**: CRITICAL (security bypass)
**Dimensions**: D7 (Guardrails), D8 (Scalability)

**The Bug**: When the semantic injection classifier enters restricted mode (after `_CLASSIFIER_DEGRADATION_THRESHOLD` consecutive failures), it returns:

```python
# guardrails.py restricted mode:
return InjectionClassification(
    is_injection=True,
    confidence=0.5,
    reason=f"Classifier degraded after {failures} failures (restricted mode)",
)
```

But compliance_gate.py checks:

```python
# compliance_gate.py:159-163
if (
    semantic_result
    and semantic_result.is_injection
    and semantic_result.confidence >= settings.SEMANTIC_INJECTION_THRESHOLD  # 0.8
):
```

Since `0.5 < 0.8`, the threshold check **silently drops the restricted-mode block**. The message passes through as if no injection was detected.

**Attack Vector**: An attacker sends 3+ rapid requests designed to timeout the LLM classifier (e.g., extremely long inputs). After 3 consecutive timeouts, the classifier enters restricted mode. Subsequent injection attempts return `confidence=0.5` which is below the `0.8` threshold, effectively bypassing semantic injection detection entirely.

**The R48 "fix" made this worse**: R48 changed restricted mode from `is_injection=False` (documented fail-open) to `is_injection=True, confidence=0.5` (appearing fail-closed but actually fail-open due to the threshold gap). The code now *claims* to block but doesn't. This is worse than an honest fail-open because it creates a false sense of security.

**Fix**: Either (a) set restricted-mode confidence to 1.0 (same as normal fail-closed), or (b) add a separate check in compliance_gate for restricted mode that bypasses the threshold.

**Impact**: Complete bypass of semantic injection classifier via forced degradation. Regex guardrails still operate, but sophisticated injections that evade regex patterns will pass through undetected.

---

### CRITICAL-D8-001: CircuitBreaker._sync_from_backend() mutates state outside asyncio.Lock — TOCTOU race under concurrent SSE streams

**File**: `src/agent/circuit_breaker.py:143-204`
**Severity**: CRITICAL (distributed state corruption)
**Dimension**: D8 (Scalability)

`_sync_from_backend()` reads from Redis and directly mutates `self._state` and `self._half_open_in_progress` at lines 181 and 195-196 without holding `self._lock`. Meanwhile, `allow_request()` calls `_sync_from_backend()` at line 350 BEFORE acquiring the lock at line 352.

Under 50+ concurrent SSE streams on Cloud Run, two coroutines interleave:

```
Coroutine A: reads Redis state="closed" at await async_get (line 173)
  -- context switch at await --
Coroutine B: reads Redis state="open" at await async_get (line 173)
  -- context switch at await --
Coroutine A: self._state = "closed" (line 195)
  -- context switch --
Coroutine B: self._state = "open" (line 181)
```

Final state depends on coroutine scheduling, not actual Redis state. The race window exists at every `await self._backend.async_get()` call in `_sync_from_backend()` — there are two await points (lines 173-174).

**Why R47 fix C15 created this bug**: R47 intentionally moved `_sync_from_backend()` outside the lock to prevent head-of-line blocking (Redis I/O inside lock serialized all callers). The fix was correct for I/O performance but introduced the TOCTOU race. The proper pattern is: read outside lock, then apply inside lock.

**Fix**: Make `_sync_from_backend()` return the remote state tuple `(remote_state, remote_count)` without mutating self. Apply the state transition inside `allow_request()`'s `async with self._lock:` block:

```python
async def _sync_from_backend(self) -> tuple[str | None, int | None]:
    """Read remote state without mutating local state."""
    remote_state = await self._backend.async_get(state_key)
    remote_count_str = await self._backend.async_get(count_key)
    return (remote_state, int(remote_count_str) if remote_count_str else None)

async def allow_request(self) -> bool:
    remote = await self._sync_from_backend()  # I/O outside lock
    async with self._lock:
        self._apply_remote_state(remote)  # Fast mutation inside lock
        ...
```

**Impact**: Circuit breaker can get stuck in wrong state. An open CB appearing closed allows requests to a failed LLM (cascading failures). A closed CB appearing open blocks all legitimate traffic (self-DoS).

---

### CRITICAL-D3-001: UNSET_SENTINEL = object() will crash JSON serialization in FirestoreSaver checkpointer

**File**: `src/agent/state.py:37`
**Severity**: CRITICAL (production crash in Firestore-backed deployment)
**Dimension**: D3 (Data Model)

`UNSET_SENTINEL: object = object()` is used in `_merge_dicts` for tombstone deletion of accumulated state fields. The docstring at lines 34-36 explicitly acknowledges the problem:

> "For JSON serialization across checkpointer boundaries, the LLM extraction layer must map the string `'__UNSET__'` to this sentinel object. Direct JSON roundtrip will NOT preserve object() identity."

However, **no such mapping code exists anywhere in the codebase**. `grep -r "__UNSET__"` across all Python files returns only `state.py` itself and the old string sentinel references in docstrings/comments.

- **MemorySaver** (local dev): Works because Python objects are stored in-memory; identity comparison (`v is UNSET_SENTINEL`) preserves.
- **FirestoreSaver** (production): Serializes state to Firestore documents. `object()` is not JSON-serializable — calling `json.dumps()` on state containing `UNSET_SENTINEL` raises `TypeError: Object of type object is not JSON serializable`.

**Scenarios where this crashes**:
1. Guest says "remove my peanut allergy" → LLM extraction returns `{"dietary": UNSET_SENTINEL}` → state update triggers checkpointer serialize → `TypeError`
2. Any accumulated field deletion attempt in Firestore-backed deployment

**Fix**: Replace `object()` with a unique string sentinel that survives JSON roundtrip, and add serialization-layer mapping. Or use a custom JSON encoder/decoder pair registered with the checkpointer.

**Impact**: Tombstone deletion feature is completely broken in production (Firestore). Works only in local dev (MemorySaver). Any guest attempting to remove an accumulated field crashes the graph.

---

### MAJOR-D4-001: Webhook endpoints (/sms/webhook, /cms/webhook) have no rate limiting — DoS vector

**File**: `src/api/middleware.py:598-603` (rate limit path filter) + `src/api/app.py:548-646` (webhook endpoints)
**Severity**: MAJOR
**Dimension**: D4 (API Design)

`RateLimitMiddleware.__call__` at line 601 only rate-limits `/chat` and `/feedback`:

```python
if path not in ("/chat", "/feedback"):
    await self.app(scope, receive, send)
    return
```

`/sms/webhook` and `/cms/webhook` bypass rate limiting entirely. Both endpoints are also excluded from API key auth (`_PROTECTED_PATHS = {"/chat", "/graph", "/property", "/feedback"}` at line 249).

While `/sms/webhook` has Telnyx signature verification and `/cms/webhook` has webhook secret verification, neither has rate limiting. An attacker with no valid signature can still flood the endpoint with invalid requests. Each request:
- Parses the full JSON body (`request.json()`)
- Attempts signature verification (crypto operations)
- Logs the failure

At sustained 10K req/s, this exhausts CPU on signature verification, blocks the event loop, and prevents legitimate `/chat` requests from being served.

**Fix**: Add `/sms/webhook` and `/cms/webhook` to the rate-limited paths, with a higher threshold (e.g., 100/min instead of the /chat limit) to accommodate legitimate webhook bursts.

**Impact**: Unauthenticated DoS vector. Attacker can degrade service without needing any credentials.

---

### MAJOR-D8-002: Streaming PII redactor re-scans the entire retained buffer on every feed() call — O(n^2) performance

**File**: `src/agent/streaming_pii.py:65-134`
**Severity**: MAJOR
**Dimension**: D8 (Scalability)

`feed()` appends the new chunk to `self._buffer` (line 78), then `_scan_and_release()` calls `redact_pii(self._buffer)` on the FULL buffer (line 117). The trailing `_MAX_PATTERN_LEN` (120) chars are retained as lookahead and re-scanned on the next `feed()` call. This means:

- Feed 1: scan 120 chars (retained lookahead)
- Feed 2: scan 120 + chunk_size chars (re-scan lookahead + new)
- Feed 3: scan 120 + chunk_size chars (re-scan again + new)

The docstring at line 129 acknowledges this: "The lookahead will be re-scanned on next feed() or flush(), which is safe because re-scanning already-redacted placeholders like `[PHONE]` is a no-op."

While correctness is maintained (re-scanning redacted placeholders is indeed a no-op for the regex engine), the performance cost is real: `redact_pii()` applies ~15 compiled regex patterns to the full buffer on every chunk. For a 2000-token response (~8000 chars) arriving in 50-char chunks:
- 160 feed() calls, each scanning 120+50=170 chars
- Total chars scanned: 160 * 170 = 27,200 (vs. 8,000 if each chunk scanned once)
- **3.4x overhead** for PII redaction under streaming

Under 50 concurrent SSE streams, this overhead is multiplied.

**Mitigation**: The overhead is bounded (3.4x, not truly O(n^2) since the retained window is capped at 120 chars). But the re-scanning of already-redacted text is wasted CPU.

**Fix**: Track the boundary between already-scanned and new text. Only apply regex patterns to the `_MAX_PATTERN_LEN` overlap zone + new chunk, not the full buffer.

**Impact**: 3.4x PII redaction CPU overhead per SSE stream. Under 50 concurrent streams, this is measurable on 1-vCPU Cloud Run instances.

---

### MAJOR-D5-001: Test suite does not exercise the restricted-mode → compliance-gate threshold interaction

**File**: `tests/test_e2e_security_enabled.py:397-416`
**Severity**: MAJOR
**Dimension**: D5 (Testing)

`test_classifier_restricted_mode_distinct_from_normal_block` (line 397) tests that restricted mode returns `confidence=0.5` and normal fail-closed returns `confidence=1.0`. But it **never passes this result through `compliance_gate_node()`** to verify the end-to-end behavior.

The test validates the classifier's return value in isolation, but the actual security bug (CRITICAL-D7-001 above) only manifests when the return value flows through compliance_gate's threshold check. This is a classic unit-test-passes / integration-test-would-fail gap.

Additionally, there are no integration tests that:
1. Set the classifier to restricted mode (`_classifier_consecutive_failures >= threshold`)
2. Send an injection attempt through `compliance_gate_node()`
3. Assert that the injection is still blocked despite `confidence=0.5`

**Fix**: Add an integration test that calls `compliance_gate_node()` with a mocked classifier in restricted mode and verifies that injections are blocked (or that the test fails, surfacing CRITICAL-D7-001).

**Impact**: The CRITICAL security bypass in D7 has been present since R48 and was not caught by the test suite.

---

## Dimension Analysis

### D1 Graph Architecture — 7.5/10

Strong 11-node StateGraph with well-designed validation loops, specialist DRY extraction via `_base.py` with dependency injection, structured output routing with Pydantic `Literal` types. The `_route_to_specialist` / `_inject_guest_context` / `_execute_specialist` decomposition shows good SRP adherence.

However, `graph.py` at ~996 LOC still mixes orchestration, streaming, SSE event generation, and dispatch concerns. The `_dispatch_to_specialist` function, while split into 3 helpers per R43, still has the parent orchestrator doing too much. The process-local `_LLM_SEMAPHORE` is acknowledged in ADR comments but ineffective at multi-instance scale.

### D2 RAG Pipeline — 8.0/10

Per-item chunking with category-specific formatters is the right approach for structured casino data. SHA-256 content hashing for idempotent ingestion, version-stamp purging for stale chunks, and RRF reranking with `k=60` per the original RRF paper. The pipeline is well-structured. No new findings beyond what previous rounds have addressed.

### D3 Data Model — 6.5/10

TypedDict state with custom reducers (`_merge_dicts`, `_keep_max`, `_keep_truthy`) shows thoughtful design. Import-time parity check catches drift. However, the UNSET_SENTINEL object() serialization crash (CRITICAL-D3-001) is a production blocker for any Firestore-backed deployment. The feature was designed, documented, and tested — but only works in the dev environment.

### D4 API Design — 7.0/10

Pure ASGI middleware stack is correct for SSE streaming. Rate limiting with Redis sorted set sliding window via atomic Lua script is production-grade. IP normalization handles IPv4-mapped IPv6 and XFF trust. However, webhook endpoints completely bypass both rate limiting and API key auth (MAJOR-D4-001), creating an unauthenticated DoS vector.

### D5 Testing Strategy — 6.5/10

2229 tests with 90.53% coverage and 0 failures is impressive quantity. The `test_e2e_security_enabled.py` with 15+ tests exercising auth and classifier-enabled paths addresses R47 feedback. However, the tests verify components in isolation without testing the integration between classifier restricted mode and compliance gate threshold (MAJOR-D5-001). This gap allowed CRITICAL-D7-001 to ship.

### D6 Docker & DevOps — 8.0/10

Multi-stage build, `--require-hashes`, non-root user, exec-form CMD, comprehensive `.dockerignore`. The Dockerfile follows security best practices. No new findings.

### D7 Prompts & Guardrails — 6.0/10

185 compiled regex patterns across 11 languages, multi-layer input normalization (10-iteration URL decode, Cf strip, NFKD, confusable table), fail-closed semantic classifier. The guardrail suite is comprehensive. However, the restricted-mode threshold mismatch (CRITICAL-D7-001) creates an attacker-exploitable bypass path. The *intent* is fail-closed, but the *implementation* is fail-open. This is scored harshly because it's a security feature that advertises protection it doesn't deliver.

### D8 Scalability & Production — 6.5/10

TTL jitter on singleton caches, circuit breaker with Redis L1/L2 sync, SIGTERM graceful drain, per-client rate limiting with LRU eviction. The scalability infrastructure is well-designed. However, the CB race condition (CRITICAL-D8-001) and streaming PII overhead (MAJOR-D8-002) are both production-relevant under Cloud Run's concurrent request model.

### D9 Trade-off Docs — 7.5/10

ADRs are embedded in source code docstrings with round/fix references (e.g., "R47 fix C15", "R39 CRITICAL fix D8-C001"). This provides excellent traceability but lacks a centralized ADR directory. Operational runbooks are absent. The threading.Lock-in-async rationale (state_backend.py lines 100-112) is a model of clear trade-off documentation.

### D10 Domain Intelligence — 7.5/10

Multi-property config via `get_casino_profile()`, state-by-state regulatory configuration, responsible gaming escalation with session-level tracking. The R49 self-harm detection addition addresses the Gemini CRITICAL finding. Casino domain expertise is evident in the guardrail patterns and specialist agent design.

---

## Summary

| Category | Count |
|----------|-------|
| CRITICAL | 3 |
| MAJOR | 3 |
| MINOR | 0 |
| False Positives Caught | 1 (semaphore count inflation) |

**Top 3 findings by impact**:

1. **CRITICAL-D7-001**: Restricted-mode classifier confidence=0.5 is below compliance gate threshold=0.8, creating an attacker-exploitable bypass of semantic injection detection via forced degradation.
2. **CRITICAL-D8-001**: `_sync_from_backend()` mutates circuit breaker state outside the asyncio lock, creating TOCTOU races under concurrent SSE streams.
3. **CRITICAL-D3-001**: `UNSET_SENTINEL = object()` is not JSON-serializable, causing production crashes in Firestore-backed deployments when guests attempt to remove accumulated fields.

**Strengths**:
- 11-node StateGraph with validation loops and bounded retries is architecturally sound
- Multi-layer guardrails (5 deterministic + 1 semantic) with correct priority ordering
- Redis-backed distributed rate limiting with atomic Lua script
- Per-item RAG chunking with version-stamp purging for stale data
- Comprehensive test suite with auth-enabled and classifier-enabled paths

**Weaknesses**:
- Security features that appear robust but have implementation gaps (restricted mode bypass)
- Distributed state patterns (CB sync, sentinel serialization) that work locally but break in production topology
- Test suite validates components in isolation but misses critical integration paths
