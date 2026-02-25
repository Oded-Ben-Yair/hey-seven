# R50 Hostile Code Review — Grok 4 (reasoning_effort=high)

**Date**: 2026-02-24
**Reviewer**: Grok 4 via `mcp__grok__grok_reason` (2 calls, reasoning_effort=high)
**Verification**: Claude Opus 4.6 cross-referenced all findings against actual code with Grep/Read
**Files Reviewed**: 16 (all specified files read in full)
**R49 Fixes Skipped**: classifier confidence=1.0, CB TOCTOU read+apply, UNSET_SENTINEL UUID-string, self-harm patterns, Mandarin injection, webhook CSP, 10 ADRs, middleware order, URL decode 10, _keep_truthy bool(), _active_streams copy, sweep task lock

---

## Dimension Scores

| Dim | Name | Weight | Score | Justification |
|-----|------|--------|-------|---------------|
| D1 | Graph/Agent Architecture | 0.20 | 7.0 | **CRITICAL**: self_harm query_type unhandled in off_topic_node. 11-node graph well-structured, specialist DRY extraction excellent, but missing handler for a safety-critical path is disqualifying for 8+. |
| D2 | RAG Pipeline | 0.10 | 8.0 | Per-item chunking, RRF reranking, idempotent SHA-256 ingestion, version-stamp purging. Solid. Minor: no explicit embedding model fallback test. |
| D3 | Data Model | 0.10 | 7.5 | Custom reducers (_merge_dicts with tombstone, _keep_max, _keep_truthy), parity check at import time. UNSET_SENTINEL UUID-string survives JSON roundtrip. Minor: _CATEGORY_TO_AGENT is mutable dict, not MappingProxyType. |
| D4 | API Design | 0.10 | 8.0 | Pure ASGI middleware (correct for SSE), streaming PII redaction with lookahead buffer, SIGTERM graceful drain. Rate limiter uses Lua script (atomic). |
| D5 | Testing Strategy | 0.10 | 7.0 | 2229 tests, ~90% coverage, 0 failures. Hypothesis property-based tests exist (17 @given across test_property_based.py + test_state_parity.py). E2E tests with auth+classifier enabled exist. But: no test covers the self_harm → off_topic path. Coverage gap on the most critical safety path. |
| D6 | Docker & DevOps | 0.10 | 9.0 | SHA-256 digest pinning, --require-hashes, multi-stage build, non-root user, exec-form CMD, .dockerignore excludes test/review artifacts. Best dimension. |
| D7 | Prompts & Guardrails | 0.10 | 7.5 | 185+ regex patterns, 11 languages, multi-layer normalization (URL decode x10, HTML unescape, NFKD, Cf strip, confusables), semantic classifier with degradation. Self-harm detection WORKS correctly in compliance_gate. Problem is downstream handling, not detection. |
| D8 | Scalability & Prod | 0.15 | 7.5 | Circuit breaker with Redis L1/L2 sync, TOCTOU-safe read+apply, TTL jitter on caches, LLM semaphore backpressure. **MAJOR**: RedisBackend never closes async client — connection leak on TTL expiry. Lua rate limiter is correct. |
| D9 | Trade-off Docs | 0.05 | 8.0 | 10 ADRs indexed, docstrings reference review round and fix ID, known limitations documented in CLAUDE.md. |
| D10 | Domain Intelligence | 0.10 | 8.0 | Multi-property config via get_casino_profile(), state-specific helplines, responsible gaming escalation (3+ triggers), age verification with minor-friendly areas listed. |

### Weighted Score Calculation

```
D1:  7.0 x 0.20 = 1.400
D2:  8.0 x 0.10 = 0.800
D3:  7.5 x 0.10 = 0.750
D4:  8.0 x 0.10 = 0.800
D5:  7.0 x 0.10 = 0.700
D6:  9.0 x 0.10 = 0.900
D7:  7.5 x 0.10 = 0.750
D8:  7.5 x 0.15 = 1.125
D9:  8.0 x 0.05 = 0.400
D10: 8.0 x 0.10 = 0.800
─────────────────────────
TOTAL:           7.425 x 10 = 74.25
```

**Weighted Score: 74.3 / 100**

---

## Findings

### CRITICAL-D1-001: self_harm query_type unhandled in off_topic_node

**Severity**: CRITICAL
**Dimension**: D1 (Graph Architecture) + D5 (Testing)
**Impact**: Suicidal guest receives generic concierge response instead of 988 Suicide & Crisis Lifeline

**Evidence chain**:
1. `compliance_gate.py:148-150` — `detect_self_harm()` correctly triggers, returns `{"query_type": "self_harm", "router_confidence": 1.0}`
2. `graph.py:521` — `route_from_compliance()` catch-all sends all non-None/non-greeting query_types to `NODE_OFF_TOPIC`
3. `nodes.py:592-696` — `off_topic_node()` handles: `bsa_aml`, `patron_privacy`, `gambling_advice`, `age_verification`, `action_request` — but has NO `self_harm` case
4. `nodes.py:681-688` — Falls through to `else` branch: "I'm your concierge for {property_name}" — a generic redirect with zero crisis resources

**Grep verification**: `self_harm` appears in `compliance_gate.py` (detection) and `guardrails.py` (regex patterns) only. Zero occurrences in `nodes.py` or `graph.py`.

**Fix**: Add `elif query_type == "self_harm":` branch in `off_topic_node()` with 988 Lifeline, Crisis Text Line, and property-specific crisis contact. This is the highest-priority fix.

```python
elif query_type == "self_harm":
    content = (
        "I hear you, and I want you to know that help is available right now.\n\n"
        "**988 Suicide & Crisis Lifeline**: Call or text **988** (24/7, free, confidential)\n"
        "**Crisis Text Line**: Text **HOME** to **741741**\n\n"
        f"You can also reach a live team member at {settings.PROPERTY_NAME} "
        f"by calling {settings.PROPERTY_PHONE} — they can connect you with "
        "on-property support.\n\n"
        "You are not alone."
    )
```

---

### MAJOR-D8-001: RedisBackend never closes async client (connection leak)

**Severity**: MAJOR
**Dimension**: D8 (Scalability & Production)
**Impact**: Connection pool leak when TTLCache expires and recreates RedisBackend

**Evidence**:
1. `state_backend.py:250-283` — `RedisBackend.__init__()` creates both `self._client` (sync) and `self._async_client` (async via `redis.asyncio`)
2. `state_backend.py:224-333` — RedisBackend class has NO `close()`, `aclose()`, or `__del__()` method
3. `state_backend.py:338` — `_state_backend_cache` is `TTLCache(maxsize=1, ttl=3600 + jitter)` — every ~1 hour the old RedisBackend is garbage-collected WITHOUT closing its connection pool
4. Grep for `aclose|async_close|close\(\)|shutdown` in state_backend.py: **zero matches**

**Fix**: Add `async def aclose(self)` that calls `await self._async_client.aclose()` and `self._client.close()`. Wire it into the FastAPI lifespan shutdown sequence.

---

### MAJOR-D1-002: _CATEGORY_TO_AGENT and _CATEGORY_PRIORITY are mutable module-level dicts

**Severity**: MAJOR (downgraded from initial assessment due to low mutation risk in practice)
**Dimension**: D1 (Graph Architecture) / D3 (Data Model)
**Impact**: Accidental mutation by any caller corrupts dispatch routing for all concurrent requests

**Evidence**:
1. `graph.py:138-145` — `_CATEGORY_TO_AGENT: dict[str, str] = {...}` — plain mutable dict
2. `graph.py:151-158` — `_CATEGORY_PRIORITY: dict[str, int] = {...}` — plain mutable dict
3. Neither is wrapped in `types.MappingProxyType`
4. `graph.py:185` — `_CATEGORY_TO_AGENT.get(dominant, "host")` — read-only access currently, but no protection against future mutation

**Fix**: Wrap both in `MappingProxyType`:
```python
from types import MappingProxyType
_CATEGORY_TO_AGENT: MappingProxyType = MappingProxyType({...})
_CATEGORY_PRIORITY: MappingProxyType = MappingProxyType({...})
```

---

### MINOR-D5-001: No test for self_harm routing path

**Severity**: MINOR (consequence of CRITICAL-D1-001)
**Dimension**: D5 (Testing)
**Impact**: The untested path is also the broken path

**Evidence**: Grep for `self_harm` in all test files returns zero matches for off_topic_node or graph routing. The detection tests exist (guardrails), but the end-to-end response test does not.

**Fix**: After fixing CRITICAL-D1-001, add integration test that sends a self-harm message through the full graph and asserts the response contains "988".

---

### MINOR-D3-001: UNSET_SENTINEL collision risk is theoretical but documented

**Severity**: MINOR (informational)
**Dimension**: D3 (Data Model)

The R49 fix changed UNSET_SENTINEL from `object()` to a UUID-namespaced string (`$$UNSET:7a3f9c2e-...$$`). This correctly survives JSON serialization through FirestoreSaver. The collision risk with natural language input is astronomically low. Well-documented decision with clear rationale in the comment block. No action needed.

---

### MINOR-D8-002: InMemoryBackend uses threading.Lock in async context

**Severity**: MINOR (documented, intentional)
**Dimension**: D8 (Scalability)

`state_backend.py` InMemoryBackend uses `threading.Lock` for sub-microsecond TOCTOU protection. This is documented as intentional (R36 fix B5) in CLAUDE.md known limitations. The lock hold time is sub-microsecond (dict mutation only), so event loop blocking is negligible. Accepted trade-off.

---

## Verification Summary

| Finding | Grok Assessment | Code Verification | Final Status |
|---------|----------------|-------------------|--------------|
| self_harm unhandled | CRITICAL | **CONFIRMED** — zero occurrences in nodes.py | CRITICAL-D1-001 |
| RedisBackend no close | MAJOR | **CONFIRMED** — no close/aclose/shutdown method | MAJOR-D8-001 |
| Mutable dispatch dicts | MAJOR | **CONFIRMED** — plain dict, not MappingProxyType | MAJOR-D1-002 |
| No Hypothesis tests | MAJOR | **REJECTED** — 17 @given decorators found | Finding withdrawn |
| PII redaction order | MINOR | **REJECTED** — redact_pii() called before logger.info | Finding withdrawn |
| Normalization leaks to state | MINOR | **REJECTED** — _normalize_input used only in detect_* functions | Finding withdrawn |

3 findings rejected after code verification (false positives from Grok's initial assessment).

---

## Strengths (What Earns 7+ Base)

1. **Specialist DRY extraction** (`_base.py`): 390 LOC shared base with dependency injection. Each specialist is a thin wrapper. Universally praised pattern.
2. **Pre-LLM deterministic guardrails**: 185+ regex, 11 languages, multi-layer normalization. Detection is excellent — the gap is handling, not detection.
3. **Validation loop**: generate -> validate -> retry(max 1) -> fallback with degraded-pass strategy. Principled availability/safety trade-off.
4. **State parity check**: Import-time `ValueError` if `_initial_state()` drifts from `PropertyQAState` annotations. Catches schema drift in all environments.
5. **Docker security**: SHA-256 digest pinning + --require-hashes + non-root + exec-form CMD. Best-in-class.
6. **Circuit breaker**: Redis L1/L2 with TOCTOU-safe read+apply pattern, bidirectional sync, Lua atomic rate limiting.

## What Prevents 8+

1. **The self_harm gap is disqualifying** for any score above 7 on D1. A compliance gate correctly detects a suicidal guest, but the graph silently drops the classification and returns a concierge greeting. This is the worst possible failure mode for a regulated hospitality product.
2. **Resource lifecycle management**: RedisBackend creates connections but never closes them. In a TTL-cached singleton pattern, this means connection leaks every TTL cycle.
3. **Immutability discipline**: Module-level dispatch tables should be frozen. The codebase already uses `MappingProxyType` for other constants but missed these.

---

## Recommendations (Priority Order)

1. **P0**: Fix CRITICAL-D1-001 — add self_harm handler in off_topic_node with 988 Lifeline
2. **P0**: Add E2E test for self_harm routing (assert "988" in response)
3. **P1**: Fix MAJOR-D8-001 — add RedisBackend.aclose(), wire into lifespan shutdown
4. **P2**: Fix MAJOR-D1-002 — wrap _CATEGORY_TO_AGENT and _CATEGORY_PRIORITY in MappingProxyType
5. **P3**: Consider adding self_harm as explicit case in route_from_compliance docstring for documentation clarity

---

*Review generated by Grok 4 (reasoning_effort=high) with Claude Opus 4.6 cross-verification. 2 Grok reasoning calls used. 3/6 initial findings rejected after code verification.*
