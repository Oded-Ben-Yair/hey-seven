# R68 Wave 6 — Code Review Summary

**Reviewer**: Claude Opus 4.6 (manual review, GPT-5.3 Codex methodology)
**Date**: 2026-02-26
**Files Reviewed**: 10 production source files
**Findings**: 1 MAJOR, 2 MINOR (0 CRITICAL)

---

## Findings by File

### 1. `src/agent/guardrails.py` (944 LOC)

**Verdict**: CLEAN

Multi-layer normalization (URL decode 10x -> HTML unescape 2-pass -> NFKD -> Cf+Cc strip -> confusable replace -> punctuation strip) is thorough. 185+ compiled regex patterns across 11 languages. Semantic injection classifier with fail-closed + degradation mode (3 consecutive failures -> bypass to deterministic-only). `audit_input()` correctly normalizes before all detection passes. No findings.

---

### 2. `src/agent/compliance_gate.py` (268 LOC)

**Verdict**: CLEAN

Priority chain is correctly ordered: turn-limit -> empty -> injection (MUST run before content guardrails) -> responsible gaming -> age -> BSA/AML -> patron privacy -> self-harm -> semantic injection -> pass. Structured JSON audit logging on every trigger. Session-level escalation counter for responsible gaming. No findings.

---

### 3. `src/agent/circuit_breaker.py` (605 LOC)

**Verdict**: CLEAN

Redis L1/L2 sync with pipelining (1 RTT), bidirectional recovery propagation (R47 fix), half-open decay (halves failure count instead of clearing). Structured JSON transition alerts with severity=ALERT. Rolling window failure tracking with bounded deque. `_sync_from_backend()` runs outside lock, fast mutation inside lock (R47 fix — never hold async lock across I/O). No findings.

---

### 4. `src/api/app.py` (810 LOC) **MODIFIED**

**Finding M1 — MAJOR: 304 Response with Body (RFC 7232 violation)**
- **Location**: Lines 562-575 (before fix)
- **Issue**: `JSONResponse(content=None, status_code=304)` serializes `None` as `"null"` JSON body. HTTP 304 MUST NOT contain a message body per RFC 7232 Section 4.1. Some proxies (nginx, Cloudflare) reject or strip the body; some clients may interpret the `"null"` body as actual content.
- **Fix**: Replaced with bare `starlette.responses.Response(status_code=304, headers=...)` which sends no body. Added inline RFC reference comment.
- **Status**: FIXED

**Finding m1 — MINOR: Percentile Index Guard Inconsistency**
- **Location**: Lines 271-277 (before fix)
- **Issue**: Only p99 had `min(int(n * 0.99), n - 1)` guard, but p50 and p95 did not. While mathematically safe for current percentile values (p50 of n=1 gives index 0, p95 of n=2 gives index 1 which equals n-1), inconsistent guarding is fragile if percentile values change.
- **Fix**: Added `min(..., n - 1)` to all three percentile calculations for consistency.
- **Status**: FIXED

**Finding m2 — MINOR: Dead `_DEPRECATED_ENDPOINTS` Dict**
- **Location**: Inside `create_app()` function
- **Issue**: `_DEPRECATED_ENDPOINTS` dict is defined but never referenced. Appears to be scaffolding for a deprecation framework that was started but not wired.
- **Recommendation**: Either wire to the deprecation header logic or remove. Not fixing (no runtime impact).
- **Status**: DEFERRED (no runtime impact)

---

### 5. `src/api/middleware.py` (826 LOC)

**Verdict**: CLEAN

Pure ASGI middleware (no BaseHTTPMiddleware — preserves SSE streaming). XFF IP validation with IPv4-mapped IPv6 normalization. Background sweep task for stale rate limit entries with proper error boundary. Module-level deque `_latency_samples` with maxlen=1000. Security headers (CSP, X-Frame-Options, HSTS, Referrer-Policy, Permissions-Policy) applied consistently. No findings.

---

### 6. `src/agent/graph.py` (633 LOC)

**Verdict**: CLEAN

11-node StateGraph assembly with source provenance dedup via `_source_key()` and `_merge_sources()`. Feature flag dual-layer design (build-time topology + runtime behavior). Parity check at import time (`_EXPECTED_FIELDS` vs `_INITIAL_FIELDS`) prevents state field drift. StreamingPIIRedactor for token-level redaction. `recursion_limit` set from config. No findings.

---

### 7. `src/agent/nodes.py` (752 LOC)

**Verdict**: CLEAN

Router, retrieve, validate, respond, fallback, greeting, off_topic nodes. Rich provenance in respond_node (category, source, score dicts). Degraded-pass validation strategy (first attempt + validator failure = PASS; retry + failure = FAIL). TTL-cached LLM singletons with separate locks per client type. `_get_last_human_message()` correctly handles both string and HumanMessage types. No findings.

---

### 8. `src/agent/state.py` (263 LOC)

**Verdict**: CLEAN

PropertyQAState TypedDict with custom reducers (`_merge_dicts`, `_keep_max`, `_keep_truthy`). UNSET_SENTINEL with UUID-namespaced prefix for JSON serialization safety. `_merge_dicts` supports tombstone deletion via `__UNSET__` pattern. `sources_used` supports both old str and new dict format for backward compatibility. No findings.

---

### 9. `src/rag/pipeline.py` (1187 LOC)

**Verdict**: CLEAN

ChromaDB production guard (raises RuntimeError if VECTOR_DB=chroma in production). Per-item chunking with 8 category-specific formatters. SHA-256 content hashing with null byte delimiter for idempotent ingestion. Version-stamp purging for stale RAG chunks. TTL-cached retriever singleton with threading.Lock (runs in thread pool — correct for sync ChromaDB). Module-level ThreadPoolExecutor(50) for concurrent retrieval. No findings.

---

### 10. `src/casino/config.py` (839 LOC)

**Verdict**: CLEAN

5 casino profiles with enforcement_context (strict_liability, recent_enforcement, commission_url). Escalation thresholds per jurisdiction (2 for NJ/PA, 3 for CT/NV). Profile completeness validation at import time. `get_casino_profile()` returns `copy.deepcopy()` to prevent mutation of global state — both happy path (known ID) and fallback path (unknown ID -> default). `_deep_merge()` correctly handles nested dicts. No findings.

---

## Summary

| Severity | Count | Fixed | Deferred |
|----------|-------|-------|----------|
| CRITICAL | 0 | 0 | 0 |
| MAJOR | 1 | 1 | 0 |
| MINOR | 2 | 1 | 1 |

**Files Modified**: `src/api/app.py` (2 edits: 304 body fix + percentile guard consistency)

**Overall Assessment**: Codebase is production-grade. The only MAJOR finding was an HTTP protocol compliance issue in the conditional request handling (304 with body). All 9 other files passed review with no findings. Security patterns (multi-layer normalization, fail-closed PII redaction, ASGI middleware), correctness patterns (custom reducers with tombstone deletion, degraded-pass validation), and scalability patterns (TTL jitter, circuit breaker with Redis sync, asyncio.Semaphore backpressure) are all solid.
