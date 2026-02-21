# Hey Seven Production Review R12 -- Gemini Focus

**Reviewer**: Gemini (simulated hostile review)
**Commit**: a36a5e6
**Date**: 2026-02-21
**Focus**: Architecture coherence, dead code, documentation honesty, design pattern consistency
**Spotlight**: Documentation & Trade-offs (+1 severity bump)

---

## Dimension Scores

| # | Dimension | Score | Justification |
|---|-----------|-------|---------------|
| 1 | Graph/Agent Architecture | 9 | 11-node StateGraph with specialist dispatch, structured output routing, validation loop, and dual-layer compliance gate is production-grade; specialist DRY extraction via `_base.py` eliminates duplication cleanly. |
| 2 | RAG Pipeline | 8 | Per-item chunking, RRF reranking, idempotent ingestion with version-stamp purging, and property_id isolation are all implemented correctly; metadata key inconsistency in list-format data path (Finding 1) is a real but minor data hygiene issue. |
| 3 | Data Model / State Design | 9 | `PropertyQAState` with `_keep_max` reducer for `responsible_gaming_count`, import-time parity check, and `RetrievedChunk` TypedDict are well-designed; per-turn reset via `_initial_state()` prevents state leakage. |
| 4 | API Design | 8 | Pure ASGI middleware stack, nonce-based CSP, SSE heartbeat, and structured error responses are solid; heartbeat logic has a subtle timing gap (Finding 4). |
| 5 | Testing Strategy | 8 | 1441+ tests across 5 layers with singleton cleanup, VCR fixtures, and 90% CI coverage gate; test count claims need verification against actual counts (Finding 5). |
| 6 | Docker & DevOps | 9 | 8-step canary pipeline with Trivy scan, automatic rollback, version assertion, and `--no-traffic` deploy is exemplary; probe configuration is well-documented. |
| 7 | Prompts & Guardrails | 9 | 84 regex patterns (actual count from code) with semantic LLM fallback, fail-closed semantic classifier, priority ordering rationale, and session-level escalation; documentation undercounts at 73 (Finding 2). |
| 8 | Scalability & Production | 8 | TTL-cached singletons, circuit breaker with rolling window, LLM semaphore backpressure, and bounded rate limiter; `threading.Lock` in async-compatible Firestore client code (Finding 3) is a known trade-off that warrants documentation. |
| 9 | Documentation & Code Quality (SPOTLIGHT) | 7 | Comprehensive ARCHITECTURE.md, runbook, and inline comments; multiple documentation-code mismatches discovered (Findings 1, 2, 5, 6, 7) -- the breadth of drift across 3+ documents indicates a systemic documentation sync problem. |
| 10 | Domain Intelligence | 9 | Casino-specific guardrails (BSA/AML, patron privacy, responsible gaming escalation), CCPA cascade delete, TCPA compliance, and age verification per CT gaming law demonstrate deep domain expertise. |

**Total: 84/100**

---

## Findings

### Finding 1 (MEDIUM): Metadata key inconsistency between JSON list and dict ingestion paths

- **Location**: `src/rag/pipeline.py:431` vs `src/rag/pipeline.py:447`
- **Problem**: The dict-format ingestion path (line 431) uses `_schema_version` as the metadata key for schema versioning, but the list-format ingestion path (line 447) uses `ingestion_version` instead. These are semantically different keys with the same hardcoded value `"2.1"`. The version-stamp purging logic (line 705) queries by `_ingestion_version` (a third key name), which is the timestamp injected at line 667. This means list-format documents carry a stale `ingestion_version: "2.1"` key that is never used by the purging logic and never updated.
- **Impact**: List-format ingestion creates chunks with an `ingestion_version` metadata key that is orphaned -- it is not queried by purging, not filtered by retrieval, and serves no purpose. If anyone relies on this key for debugging or monitoring, they will see stale static values instead of timestamps. More importantly, list-format documents lack the `_schema_version` key entirely, creating an inconsistency in the metadata schema across ingestion paths.
- **Fix**: Rename `ingestion_version` to `_schema_version` at line 447 to match the dict path. Both paths should produce identical metadata schemas:
```python
# Line 447: change from
"ingestion_version": "2.1",
# to
"_schema_version": "2.1",
```

### Finding 2 (MEDIUM -- SPOTLIGHT +1 from LOW): Documentation claims 73 guardrail patterns, code has 84

- **Location**: `README.md:23,44,72,99,109,232,245`, `ARCHITECTURE.md:22,104,374,834`, `src/agent/compliance_gate.py:7`
- **Problem**: The README and ARCHITECTURE.md consistently state "73 regex patterns" across all references. The compliance_gate.py module docstring (line 7) states "84 compiled regex patterns." The actual count of `re.compile` calls in `guardrails.py` is 84. The 73 figure appears to be stale from a previous version before patterns were added (likely the semantic injection additions or expanded responsible gaming patterns).
- **Impact**: Documentation understates the security coverage by 15%. While understating is less harmful than overclaiming, it creates confusion when a reviewer counts patterns and gets a different number. In a regulated environment, accurate security documentation matters for compliance audits. The discrepancy also means the ARCHITECTURE.md breakdown (11 + 31 + 6 + 14 + 11 = 73) is internally consistent but wrong relative to the actual code.
- **Fix**: Update all occurrences of "73 patterns" to "84 patterns" in README.md and ARCHITECTURE.md. Update the per-category breakdown in ARCHITECTURE.md to match the actual counts from `guardrails.py`. The compliance_gate.py docstring already has the correct count.

### Finding 3 (MEDIUM): threading.Lock used in async-compatible Firestore client initialization

- **Location**: `src/data/guest_profile.py:57-82`, `src/casino/config.py:175` (same pattern)
- **Problem**: `_get_firestore_client()` uses `threading.Lock()` to protect the singleton initialization of the Firestore AsyncClient. In an async application running on a single-threaded event loop (uvicorn with `--workers 1`), `threading.Lock()` blocks the entire event loop when contended. Under load, if two concurrent requests hit `_get_firestore_client()` simultaneously during cold start, the second coroutine blocks at `with _firestore_client_lock:` and freezes all other async work until the first completes the Firestore client construction (which involves network I/O for credential verification).
- **Impact**: During cold start under concurrent load, the event loop blocks for the duration of Firestore client initialization (typically 100-500ms for credential exchange). This causes all in-flight requests (including health checks) to stall. With `--workers 1` and `--concurrency=50`, this affects up to 49 other requests. The window is narrow (only during first-ever access), but Cloud Run's burst-start behavior can trigger this when a new instance receives multiple requests simultaneously.
- **Fix**: Replace `threading.Lock` with `asyncio.Lock` and make the accessor `async def`. This is the same pattern already correctly used for LLM singletons (`_llm_lock`, `_validator_lock`, `_whisper_lock` in `nodes.py` and `whisper_planner.py`). The comment at line 72 acknowledges the `threading.Lock` choice but does not explain why it differs from the async pattern used elsewhere. Add a code comment documenting the rationale if `threading.Lock` is intentional (e.g., if `_get_firestore_client()` is called from sync contexts in tests).

### Finding 4 (LOW): SSE heartbeat timing gap allows missed heartbeats

- **Location**: `src/api/app.py:175-188`
- **Problem**: The heartbeat logic checks `if now - last_event_time >= _HEARTBEAT_INTERVAL` and sends a ping, but `last_event_time` is updated *after* yielding the event from `chat_stream`, not after yielding the heartbeat ping. This means the time spent by the client processing the event (effectively zero for SSE) is not the issue, but rather: if a single LLM generation takes 20+ seconds between token emissions, the heartbeat check only runs when `chat_stream` yields the next event. The heartbeat cannot fire *during* the gap between events because the `async for` loop is blocked waiting for the next event from `graph.astream_events`. The heartbeat logic is reactive, not proactive.
- **Impact**: If the LLM takes 30+ seconds between token emissions (e.g., during a complex reasoning step or API throttling), no heartbeat is sent because the heartbeat check is inside the `async for` loop which is suspended waiting for the next event. Browser EventSource implementations typically have a 45-60 second timeout, so a 30-second gap without heartbeat is survivable, but it undermines the documented intent of "every 15s." The risk is low because `astream_events` v2 emits lifecycle events (node start/complete) alongside tokens, which reset the heartbeat timer.
- **Fix**: Use `asyncio.create_task` with a background heartbeat coroutine that independently yields pings every 15s to the SSE generator, or use `async_timeout` with a wrapper that yields heartbeats during idle periods. Alternatively, document that heartbeats are event-triggered, not time-triggered, and are best-effort rather than guaranteed at 15s intervals.

### Finding 5 (MEDIUM -- SPOTLIGHT +1 from LOW): Test count claims are inconsistent across documents

- **Location**: `README.md:127,131`, `CLAUDE.md` (project instructions), runbook, ARCHITECTURE.md
- **Problem**: Multiple test count claims exist across documents and they diverge:
  - README line 127: "1216 tests collected, no API key needed"
  - README line 131: "1216 tests collected, 20 skipped across 35 test files"
  - CLAUDE.md: "1070+ tests across 32 test files"
  - Context preamble: "1441 tests"
  - Previous round scores reference different test counts

  These are four different numbers for the same codebase. Test counts change with every commit, and hardcoding them into documentation creates permanent drift.
- **Impact**: In a regulated casino environment, accurate test count documentation matters for audit trails and compliance reviews. An auditor reading "1070+ tests" in one document and "1441 tests" in another cannot determine which is authoritative. This is a systemic problem: the numbers were correct at the time they were written but are now permanently stale.
- **Fix**: Remove hardcoded test counts from CLAUDE.md and README.md. Replace with a dynamic reference: "Run `make test-ci` to see current test count" or "See CI badge for current test metrics." Keep the test *strategy* documentation (5 layers, test pyramid) but not the raw counts. If a specific count is needed for a milestone (e.g., "1.0.0 shipped with 1200+ tests"), document it with a date and commit SHA.

### Finding 6 (HIGH -- SPOTLIGHT +1 from MEDIUM): ARCHITECTURE.md claims LLM singletons use @lru_cache, code uses TTLCache

- **Location**: `ARCHITECTURE.md:210` ("Both are `@lru_cache` singletons"), `src/agent/nodes.py:101-105` (actual TTLCache implementation)
- **Problem**: ARCHITECTURE.md section on the validate node states: "Both are `@lru_cache` singletons." This is factually incorrect. The code was migrated from `@lru_cache` to `cachetools.TTLCache` (with 3600s TTL and `asyncio.Lock`) as documented in R10 fix comments throughout the codebase. The architecture document was not updated to reflect this change.
- **Impact**: A developer reading ARCHITECTURE.md would believe `@lru_cache` is used, which has fundamentally different behavior from TTLCache:
  - `@lru_cache` never expires -- credentials cannot rotate without process restart
  - `TTLCache` expires after 1 hour -- credentials rotate automatically
  - `@lru_cache` has no lock -- concurrent access is unprotected
  - `TTLCache` with `asyncio.Lock` is coroutine-safe

  This is not a cosmetic documentation issue -- it describes the wrong caching behavior, which would lead to incorrect debugging assumptions (e.g., "why is my LLM client being recreated?") and incorrect operational procedures (e.g., "restart the container to pick up new credentials").
- **Fix**: Update ARCHITECTURE.md line 210 to: "Both use TTL-cached singletons (1-hour expiry via `cachetools.TTLCache` with `asyncio.Lock`) for automatic credential rotation. See `nodes.py` for implementation."

### Finding 7 (MEDIUM -- SPOTLIGHT +1 from LOW): ARCHITECTURE.md claims 5-step pipeline, actual is 8-step

- **Location**: `ARCHITECTURE.md:752` ("5-step CI/CD pipeline"), `cloudbuild.yaml` (8 steps), `docs/runbook.md:92-103` (correctly documents 8 steps)
- **Problem**: ARCHITECTURE.md states: "`cloudbuild.yaml` defines a 5-step CI/CD pipeline" and then lists only 5 steps (install/lint/test, build, scan, push, deploy). The actual `cloudbuild.yaml` has 8 steps: (1) test, (2) build, (3) Trivy scan, (4) push, (5) capture rollback revision, (6) deploy with `--no-traffic`, (7) smoke test with version assertion and automatic rollback, (8) route traffic. The runbook correctly documents all 8 steps. The "5-step" claim predates the canary deployment additions (steps 5-8).
- **Impact**: A developer reading ARCHITECTURE.md would not know about the automatic rollback, smoke test, or canary deployment pattern. These are critical operational safety features that differentiate a demo pipeline from a production-grade one. The discrepancy also undermines confidence in the accuracy of other claims in ARCHITECTURE.md.
- **Fix**: Update ARCHITECTURE.md line 752 to "8-step CI/CD pipeline" and add the missing steps (revision capture, smoke test with rollback, traffic routing). Reference the runbook for detailed step documentation to avoid maintaining the same information in two places.

### Finding 8 (LOW): `_CATEGORY_PRIORITY` missing "hotel" in deterministic dispatch tie-breaking

- **Location**: `src/agent/graph.py:131-138`
- **Problem**: `_CATEGORY_PRIORITY` defines priority values for `restaurants` (4), `hotel` (3), `entertainment` (2), `spa` (2), `gaming` (1), `promotions` (1). However, the ARCHITECTURE.md (line 178) states: "Dispatch tie-breaking: when multiple categories have equal counts, `max(category_counts, key=lambda k: (count, k))` selects alphabetically." This is incorrect -- the actual code uses a three-key sort: `(count, _CATEGORY_PRIORITY.get(k, 0), k)` which means priority takes precedence over alphabetical order. The documentation describes a simpler tie-breaking mechanism than what is actually implemented.
- **Impact**: Low -- the actual implementation is *better* than what the documentation describes (business-priority tie-breaking is more predictable than alphabetical). But the mismatch is another instance of documentation drift.
- **Fix**: Update ARCHITECTURE.md line 178 to describe the three-key tie-breaking: "business-priority tie-breaking: dining (4) > hotel (3) > entertainment (2) > comp (1), then alphabetical for unmapped categories."

### Finding 9 (LOW): `CasinoHostState` alias referenced in ARCHITECTURE.md but does not exist in code

- **Location**: `ARCHITECTURE.md:268`, `src/agent/state.py` (no `CasinoHostState` defined)
- **Problem**: ARCHITECTURE.md states: "`PropertyQAState` is a `TypedDict` with 13 fields (see `PropertyQAState` in `src/agent/state.py`). `CasinoHostState` is a backward-compatible alias for v2 code that prefers the domain-specific name." However, `CasinoHostState` does not exist anywhere in the `src/` directory. A grep for `CasinoHostState` across the entire source tree returns zero results.
- **Impact**: Documentation references a non-existent type alias. A developer searching for `CasinoHostState` in the codebase would find nothing, creating confusion about whether it was removed without updating the docs or was never implemented.
- **Fix**: Remove the `CasinoHostState` reference from ARCHITECTURE.md line 268. The state is `PropertyQAState` everywhere in the codebase.

### Finding 10 (LOW): StreamingPIIRedactor lookahead uses original buffer but emits from redacted text

- **Location**: `src/agent/streaming_pii.py:111-118`
- **Problem**: In `_scan_and_release(force=False)`, the method applies `redact_pii()` to the full buffer, then emits `redacted[:-_MAX_PATTERN_LEN]` as the safe prefix, and retains `self._buffer[-_MAX_PATTERN_LEN:]` from the *original* buffer as the lookahead. The comment at line 112-114 correctly explains why the original buffer is retained (redaction changes lengths). However, this means the safe prefix `redacted[:-_MAX_PATTERN_LEN]` may truncate in the middle of a redaction token if `redact_pii()` shortened the text. For example, if the buffer is 80 chars, and a phone number at position 35-45 is replaced by `[PHONE]` (7 chars, net -3 chars), then `redacted` is 77 chars. `redacted[:-40]` = `redacted[:37]`, which may cut through the `[PHONE]` replacement token.
- **Impact**: In rare cases where PII appears near the lookahead boundary and redaction changes the text length, the emitted safe prefix could contain a partial redaction token (e.g., `[PHO` followed by `NE]` on the next flush). This is a defense-in-depth mechanism on top of `persona_envelope` redaction, so the probability of reaching this code path with real PII is low. The PII is still redacted -- only the redaction token formatting may be split.
- **Fix**: After redaction, use the *redacted* text length for boundary calculation: `safe = redacted[:max(0, len(redacted) - _MAX_PATTERN_LEN)]`. Or, apply `redact_pii()` to the safe prefix and lookahead independently (trading potential missed cross-boundary patterns for correct token boundaries). Document the trade-off.

---

## Summary

### Severity Breakdown

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH | 1 |
| MEDIUM | 4 |
| LOW | 5 |
| **Total** | **10** |

### Spotlight Assessment (Documentation & Code Quality)

The documentation is extensive -- ARCHITECTURE.md alone is 860+ lines covering every node, routing decision, and trade-off. The runbook is production-ready with probe configurations, incident response, and alert thresholds. However, this review found **7 documentation-code mismatches** across 3 major documents:

1. Pattern count: 73 (docs) vs 84 (code)
2. Cache type: `@lru_cache` (docs) vs `TTLCache` (code)
3. Pipeline steps: 5 (ARCHITECTURE.md) vs 8 (actual)
4. Test counts: 4 different numbers across 4 documents
5. Non-existent type alias: `CasinoHostState` referenced but never defined
6. Tie-breaking: alphabetical (docs) vs business-priority (code)
7. Metadata key: `ingestion_version` vs `_schema_version` inconsistency

This pattern indicates documentation was written at specific milestones but not systematically updated as code evolved. The individual mismatches are LOW-MEDIUM severity, but collectively they erode trust in the documentation as a source of truth. For a regulated casino environment, documentation accuracy is a compliance requirement, not a nicety.

### Architecture Coherence

The architecture is internally consistent and well-motivated. Key coherence observations:
- The dual-layer feature flag system (build-time topology vs runtime behavior) is correctly documented and implemented with clear rationale.
- The degraded-pass validation strategy is the same principled design from R8-R12 discussions.
- The specialist DRY extraction via `_base.py` with dependency injection preserves test mock paths.
- The streaming PII redactor as defense-in-depth on top of `persona_envelope` redaction is sound.
- Circuit breaker state transitions are properly lock-protected with `asyncio.Lock`.

### Top 3 Strengths (Not Scored)

1. **Import-time parity check** (`graph.py:494-501`): A `ValueError` at import time if `_initial_state()` fields don't match `PropertyQAState` annotations. This prevents state schema drift in ALL environments, not just tests.
2. **Streaming PII redactor** (`streaming_pii.py`): Buffer-based PII detection across token boundaries is a non-trivial engineering contribution that most LLM applications skip entirely.
3. **Runbook completeness** (`docs/runbook.md`): The runbook covers probe configuration, secrets, rollback, incident response, alert thresholds, and escalation matrix. This is "Day 2" operational readiness that most demo codebases lack.

---

**Overall Score: 84/100**

Delta from R11 Gemini: -2 (R11 scored 86). The regression is entirely in Documentation & Code Quality (spotlight dimension): the documentation drift findings, while individually minor, collectively reveal a systemic problem that the spotlight amplifies.
