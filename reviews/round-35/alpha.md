# R35 Review — Dimensions 1-5 (reviewer-alpha)

**Date**: 2026-02-23
**Reviewer**: reviewer-alpha
**Cross-validated with**: GPT-5.2 Codex (azure_code_review), Gemini 3 Pro (thinking=high)
**Codebase snapshot**: ~2107 tests, 51 source modules, 23K LOC

---

## Dimension 1: Graph/Agent Architecture (weight 0.20)

**Score: 8.5 / 10** (R34: 8.0, delta +0.5)

### Strengths
- 11-node custom StateGraph with validation loop is architecturally sound — generate -> validate -> retry(max 1) -> fallback
- Specialist DRY extraction via `_base.py` with dependency injection (get_llm_fn, get_cb_fn) eliminates ~600 LOC duplication
- Shared constants module (`constants.py`) with `_KNOWN_NODES` frozenset prevents string typo bugs
- Degraded-pass validation strategy is nuanced and production-grade
- Parity check at import time (ValueError if _initial_state and PropertyQAState annotations diverge)
- `_LLM_SEMAPHORE = asyncio.Semaphore(20)` for concurrency backpressure in specialist execution
- **R34 A1 FIX APPLIED**: `asyncio.timeout(settings.MODEL_TIMEOUT)` now wraps `agent_fn(state)` execution (graph.py:~289-304) — both models praised this as highest-priority fix from R34
- Feature flag ADR distinguishes build-time (graph topology) vs runtime (per-request) flags — well-documented

### Findings

**MAJOR-A1: _dispatch_to_specialist SRP violation — routing + injection + execution** (graph.py:~250-340)
- Single function handles: (1) structured LLM dispatch, (2) keyword fallback routing, (3) guest profile injection, (4) agent function execution, (5) dispatch-owned key collision logging, (6) error handling with fallback
- ~90 lines mixing concerns that should be separated
- **Carried from R34-A2** — deferred to post-MVP. Both models re-flagged.
- **GPT-5.2 consensus**: "The function mixes routing decisions with execution and state injection."
- **Gemini consensus**: "SRP violation — dispatch does routing, profile injection, and execution in one function."
- **Fix**: Extract guest profile injection and agent execution into separate helper functions

**MAJOR-A2: Three separate get_settings() calls in _dispatch_to_specialist** (graph.py:~216, ~259, ~267)
- `get_settings()` is called 3 times within the same function invocation
- Settings are TTL-cached (1 hour) so the values are almost certainly identical, but it's a TOCTOU violation in principle
- If settings TTL expires between call 1 and call 3, the function operates with mixed configuration
- **GPT-5.2 consensus**: Flagged as code smell. "Multiple settings reads in same function should be hoisted."
- **Gemini consensus**: Not flagged (low severity).
- **Fix**: Hoist `settings = get_settings()` once at function start, use the single reference throughout

**MAJOR-A3: record_success() clears ALL failure timestamps — flapping risk** (circuit_breaker.py:245)
- `record_success()` calls `self._failure_timestamps.clear()` wiping the entire failure history
- A single successful half-open probe clears 4 prior failures, resetting the breaker to a clean slate
- Under intermittent failure conditions (e.g., LLM returning 50% errors), the breaker flaps between closed->open->half_open->closed->open indefinitely
- Industry-standard pattern: half-open success should reduce failure count (e.g., halve it), not zero it
- **GPT-5.2 consensus**: Flagged. "Full history clear on single success enables rapid flapping."
- **Gemini consensus**: Flagged. "record_success should decay failures, not clear them."
- **Fix**: On half-open success, halve the failure count instead of clearing. On closed success, clear is fine.

**MINOR-A1: No validation that agent_fn returns dict** (graph.py:~305)
- `result = await agent_fn(state)` assumes dict return type
- If a specialist returns None or wrong type, downstream code fails with unclear error
- Carried from R34 — low risk due to contract tests now verifying return shapes
- **Fix**: Add `if not isinstance(result, dict): result = {"messages": [...fallback...]}`

**MINOR-A2: _count_consecutive_frustrated iterates ALL messages without cap** (_base.py)
- `_count_consecutive_frustrated()` iterates `reversed(messages)` with no upper bound
- For conversations with 40+ messages (at MAX_MESSAGES window), this scans up to 40 messages per turn
- VADER is sub-1ms per message so 40 * 1ms = ~40ms — acceptable but unguarded
- **Fix**: Add `max_scan=10` parameter to cap iteration depth

---

## Dimension 2: RAG Pipeline (weight 0.10)

**Score: 8.0 / 10** (R34: 8.0, delta 0.0)

### Strengths
- Per-item chunking with 8 category-specific formatters (_format_restaurant, _format_entertainment, etc.) — praised by all reviewers across 20+ rounds
- RRF reranking with k=60 per original paper, SHA-256 document identity preserving highest cosine score
- `_compute_chunk_id()` shared between bulk and single ingest — R33 fix for SHA-256 consistency
- Query-type-aware augmentation (`_get_augmentation_terms`) with _TIME_WORDS and _PRICE_WORDS frozensets
- Version-stamp purging eliminates ghost data from edited content
- Post-fusion filtering uses original cosine score (not RRF rank) as quality gate
- Property ID metadata isolation prevents cross-tenant leakage
- Embedding model version pinned (`gemini-embedding-001`)

### Findings

**MAJOR-A4: Inconsistent purge scopes between bulk and single ingest** (pipeline.py)
- `ingest_property()` purges by `property_id` + `_ingestion_version` (all sources for that property)
- `reingest_item()` purges by `property_id` + `source` + `_ingestion_version` (single source only)
- If an item is moved between sources (e.g., restaurant reclassified), the old-source chunk is never purged
- **Carried from R34-A4** — design decision needed. Gemini re-flagged.
- **Gemini consensus**: "Inconsistent purge filter predicates — bulk purges broadly, single purges narrowly."
- **Fix**: Document the intentional difference OR unify purge scope with source parameter

**MAJOR-A5: No embedding dimension validation on model change** (embeddings.py, pipeline.py)
- If `EMBEDDING_MODEL` is changed, existing vectors in ChromaDB have incompatible dimensions
- No validation checks that the embedding dimension of new vectors matches existing collection
- ChromaDB will silently accept mismatched dimensions and return garbage similarity scores
- **Carried from R34-A5** — both models re-flagged.
- **Gemini consensus**: "No dimension check — model swap corrupts the entire vector store."
- **Fix**: Query collection metadata for dimension count on retriever init, assert match with embedding model output

**MINOR-A3: Purge logic uses vectorstore._collection private API** (pipeline.py)
- `self.vectorstore._collection.get(where=...)` and `.delete(ids=...)` access ChromaDB internal API
- If chromadb upgrades change the private `_collection` interface, purge breaks silently
- **GPT-5.2 consensus**: Flagged. "Private API access is a maintenance risk."
- **Fix**: Use `self.vectorstore.get()` public API if available, or pin chromadb version and document the dependency

**MINOR-A4: _get_augmentation_terms uses word-level split, not stemming** (pipeline.py)
- `words = set(query_lower.split())` means "pricing" won't match _PRICE_WORDS frozenset containing "price"
- Morphological variants missed. Low severity — augmentation is a recall-boosting heuristic.
- Carried from R34.

---

## Dimension 3: Data Model (weight 0.10)

**Score: 8.5 / 10** (R34: 8.5, delta 0.0)

### Strengths
- PropertyQAState TypedDict with 17 fields and 4 custom reducers — well-documented
- `_merge_dicts` reducer with clear docstring explaining latest-wins semantics and design rationale
- `_keep_max` and `_keep_truthy` reducers for responsible_gaming_count and suggestion_offered — correct patterns
- `__all__` exports list for clean public API
- **R34 A6 FIX APPLIED**: `GuestContext` TypedDict (total=False) added at state.py:80-96 — structured type replaces `dict[str, Any]`
- Pydantic models with Literal types: RouterOutput (7 types), DispatchOutput (5 specialists), ValidationResult (3 states)
- RetrievedChunk TypedDict provides pipeline contract
- Import-time parity check catches schema drift between state and _initial_state

### Findings

**MAJOR-A6: guest_context has no reducer — fragile under feature flag toggle** (state.py:133)
- `guest_context: GuestContext` has NO Annotated reducer
- The field is re-derived each turn from `extracted_fields` (which has `_merge_dicts` reducer), so data is NOT lost across turns
- However, if `guest_profile_enabled` is toggled from True to False mid-session, `_dispatch_to_specialist` stops populating guest_context, and `_initial_state()` resets it to `{}`
- The next turn sees empty guest_context even though extracted_fields still has the data
- This is by design (guest_context is a denormalized view), but fragile
- **Gemini consensus**: Flagged as fragile. "Feature flag toggle mid-session causes context loss."
- **GPT-5.2 consensus**: Flagged indirectly. "guest_context has no accumulation guarantee."
- **Fix**: Either add a `_merge_dicts` reducer to guest_context, or document explicitly that it's a derived field that resets per-turn by design

**MINOR-A5: No TypedDict for specialist agent return shape** (state.py, _base.py)
- Specialist agents return `dict` with keys like "messages", "skip_validation", "guest_context"
- No TypedDict defines the expected return shape — validated only by contract tests at runtime
- Carried from R34-A7. Contract tests now exist (R34 A11 fix), reducing severity from MAJOR to MINOR.
- **Fix**: Define a SpecialistResult TypedDict for compile-time enforcement

**MINOR-A6: _merge_dicts has no cycle detection for nested dicts** (state.py:25-43)
- `{**a, **b}` performs shallow merge only — nested dicts are overwritten, not deep-merged
- If `extracted_fields` ever contains nested structures (e.g., `{"preferences": {"dietary": "vegan"}}`), a new extraction of `{"preferences": {"seating": "window"}}` overwrites the entire preferences dict
- Currently safe because all extracted fields are flat (string/int/list values)
- **Fix**: Document the shallow-merge contract, or add depth guard if nested structures are planned

---

## Dimension 4: API Design (weight 0.10)

**Score: 8.0 / 10** (R34: 8.0, delta 0.0)

### Strengths
- 6 pure ASGI middleware classes — no BaseHTTPMiddleware (correct pattern for SSE streaming)
- Per-request nonce-based CSP in SecurityHeadersMiddleware
- `hmac.compare_digest()` for ALL secret comparisons in ApiKeyMiddleware
- TTL-cached API key refresh every 60s with atomic tuple caching
- Two-layer body limit (Content-Length header + streaming byte counting)
- IPv6 normalization in rate limiter (bracket stripping, IPv4-mapped normalization) — R33 fix
- CancelledError at INFO level (client disconnect is normal for SSE)
- OrderedDict for LRU eviction semantics in rate limiter
- Structured JSON error responses via ErrorHandlingMiddleware
- SSE heartbeat + timeout pattern in chat_stream

### Findings

**MAJOR-A7: CSP nonce generated but not passed to template rendering** (middleware.py:~SecurityHeadersMiddleware)
- `SecurityHeadersMiddleware` generates a per-request nonce for Content-Security-Policy
- The nonce is embedded in the CSP header but never passed to template rendering context
- Any inline scripts in HTML templates would be blocked by CSP unless they have the matching nonce
- Currently mitigated by API-only usage (no server-rendered HTML with inline scripts), but a latent bug
- **Carried from R34-A7** — still applies. Gemini re-flagged.
- **Gemini consensus**: "Nonce generated but never passed downstream — useless for inline scripts."
- **Fix**: Either pass nonce via ASGI scope for template access, or document that nonce-based CSP is for API-only mode

**MAJOR-A8: Rate limiter stale sweep on request path** (middleware.py:~RateLimitMiddleware)
- Stale client cleanup runs every 100 requests on the request path
- Under high load, the sweep adds latency to every 100th request
- Under low load, stale entries accumulate
- **Carried from R34-A8** — still applies. GPT-5.2 re-flagged.
- **GPT-5.2 consensus**: "Sweep on request path is a latency concern under high load."
- **Fix**: Move sweep to a background asyncio task on a timer (e.g., every 60 seconds)

**MINOR-A7: System prompt in _base.py has no length bounds** (_base.py)
- `_build_system_prompt()` concatenates property context, regulations, persona, frustration guidance, and proactive suggestion instructions without any total length check
- A property with extensive regulations and context could produce a system prompt exceeding model context limits
- Currently safe because property configs are hand-curated, but no guard against future growth
- **GPT-5.2 consensus**: Flagged. "No bounds on concatenated system prompt length."
- **Fix**: Add a `MAX_SYSTEM_PROMPT_CHARS` constant and truncate with warning log if exceeded

**MINOR-A8: Streaming body limit potential race in byte counting** (middleware.py)
- The streaming byte counter increments per chunk in the receive wrapper
- In theory, concurrent receive() calls are not atomic. Very low risk — ASGI receive is sequential per request.
- Carried from R34.

---

## Dimension 5: Testing Strategy (weight 0.10)

**Score: 7.5 / 10** (R34: 7.5, delta 0.0)

### Strengths
- ~2107 tests collected across 58+ test files — significant test volume
- E2E graph tests (`test_graph_v2.py`) test full pipeline with mocked LLMs
- Schema-dispatching mock LLM pattern for multi-node E2E tests
- Golden conversation tests in LLM judge (7 test cases)
- Property-based hypothesis tests for PII redactor, guardrails, streaming redactor, router types, IPv6 normalization
- Conftest with 18+ singleton cache clears (autouse, function scope)
- `_disable_semantic_injection_in_tests` fixture prevents fail-closed blocking
- Load test and doc accuracy enforcement tests exist
- Deterministic offline scoring functions for CI (no LLM calls needed)
- **R34 A11 FIX APPLIED**: Contract tests added for 5 specialist agents (test_agents.py:~714, TestSpecialistContract)
- Multi-turn conversation tests exist (test_r26_e2e_phase4.py)
- Reducer edge case tests exist (TestMergeDictsReducer, suggestion_offered persistence)

### Findings

**MAJOR-A9: ~25% code coverage despite ~2107 tests** (pytest output)
- `pytest --cov` shows ~24.97% coverage — unchanged from R34
- 2107 tests but only 25% line coverage suggests heavy path duplication
- Critical paths like error handling branches, fallback paths, and edge cases likely untested
- **Carried from R34-A9** — no improvement. Both models re-flagged as highest testing gap.
- **Gemini consensus**: "25% coverage with 2107 tests is alarming — test quality over quantity."
- **Fix**: Run `pytest --cov --cov-report=term-missing` and identify uncovered critical paths in src/agent/ and src/api/. Target 60%+ for src/agent/.

**MAJOR-A10: Only 5 property-based hypothesis tests** (test_property_based.py)
- Property-based testing limited to: PII redactor, guardrails, streaming redactor chunks, router types exhaustive, IPv6 normalization
- Missing hypothesis tests for: RRF reranking edge cases, state reducer interactions, circuit breaker state transitions, middleware header injection
- **Carried from R34-A10** (was reported as 3, actual count is 5) — both models re-flagged.
- **Gemini consensus**: "5 hypothesis tests for a 23K LOC codebase is inadequate."
- **Fix**: Add hypothesis tests for circuit breaker state machine, RRF with duplicate documents, reducer merge conflicts

**MINOR-A9: Conftest clears 18+ caches but no meta-test verifies completeness** (conftest.py)
- If a new singleton cache is added, conftest may miss it
- No import-time or test-time check that all `@lru_cache` or TTLCache instances are in the cleanup list
- The try/except (ImportError, AttributeError) pattern silently swallows failures — a renamed module is silently not cleared
- **Carried from R34-A9** — still applies.
- **Fix**: Add a test that introspects all modules for `cache_clear` methods and asserts they're in conftest

**MINOR-A10: LLM judge golden conversations hardcoded in source** (llm_judge.py)
- 7 golden conversation test cases are hardcoded in the module (~100 lines)
- Adding new golden conversations requires code changes, not just data file updates
- Carried from R34.

---

## Cross-Model Consensus Summary

| Finding | GPT-5.2 Codex | Gemini 3 Pro | Severity |
|---------|---------------|--------------|----------|
| A1: Dispatch SRP violation | Flagged | Flagged strongly | MAJOR (deferred) |
| A2: 3x get_settings() TOCTOU | Flagged as smell | Not flagged | MAJOR |
| A3: CB record_success clears all | Flagged | Flagged | MAJOR |
| A4: Inconsistent purge scopes | Not flagged | Flagged | MAJOR (carried) |
| A5: No embedding dimension validation | Flagged | Flagged | MAJOR (carried) |
| A6: guest_context no reducer | Flagged indirectly | Flagged as fragile | MAJOR |
| A7: CSP nonce unused | Not flagged | Flagged | MAJOR (carried) |
| A8: Rate limit sweep latency | Flagged | Not flagged | MAJOR (carried) |
| A9: 25% coverage | Flagged | Flagged strongly | MAJOR (carried) |
| A10: Only 5 hypothesis tests | Flagged | Flagged | MAJOR (carried) |

**Both models agree on**: CB flapping risk (A3), coverage gap (A9), hypothesis test gap (A10)
**GPT-5.2 unique**: TOCTOU get_settings (A2), system prompt length (MINOR-A7)
**Gemini unique**: guest_context fragility (A6), purge scopes (A4)

### R34 Fixes Verified

| R34 Finding | Status | Evidence |
|-------------|--------|----------|
| R34-A1: No agent_fn timeout | **FIXED** | asyncio.timeout wraps agent_fn (graph.py:~289-304) |
| R34-A6: guest_context untyped dict | **FIXED** | GuestContext TypedDict added (state.py:80-96) |
| R34-A11: No contract tests | **FIXED** | TestSpecialistContract added (test_agents.py:~714) |
| R34-A2: Dispatch SRP | **DEFERRED** | Intentional post-MVP deferral |
| R34-A3: CB outside try | **RESOLVED** | Intentional per R15 fix comment — CB creation is synchronous, cannot raise |
| R34-A4: Inconsistent purge | **OPEN** | Needs design decision |
| R34-A5: Embedding validation | **OPEN** | Still no dimension check |
| R34-A7: CSP nonce | **OPEN** | Still not passed to templates |
| R34-A8: Rate limit sweep | **OPEN** | Still on request path |
| R34-A9: 25% coverage | **OPEN** | Still at ~25% |
| R34-A10: Hypothesis tests | **OPEN** | Now 5 tests (was reported as 3), still insufficient |

---

## Dimension Score Summary

| Dimension | Weight | R34 Score | R35 Score | Delta | Weighted |
|-----------|--------|-----------|-----------|-------|----------|
| 1. Graph Architecture | 0.20 | 8.0 | 8.5 | +0.5 | 1.70 |
| 2. RAG Pipeline | 0.10 | 8.0 | 8.0 | 0.0 | 0.80 |
| 3. Data Model | 0.10 | 8.5 | 8.5 | 0.0 | 0.85 |
| 4. API Design | 0.10 | 8.0 | 8.0 | 0.0 | 0.80 |
| 5. Testing Strategy | 0.10 | 7.5 | 7.5 | 0.0 | 0.75 |
| **Subtotal (dims 1-5)** | **0.60** | — | — | — | **4.90** |

**Net change from R34**: +0.10 weighted (Graph Architecture improved due to agent_fn timeout fix and GuestContext TypedDict)

**Projected full score** (if dims 6-10 maintain R34 levels ~8.0 avg): 4.90 + 3.20 = **8.10 / 10 (81/100)**

---

## Top 5 Findings for Fixer (Priority Order)

1. **MAJOR-A9**: Investigate and improve code coverage from ~25%. Run `pytest --cov --cov-report=term-missing` to identify critical uncovered paths in src/agent/ and src/api/. This is the single largest drag on Testing Strategy score and has been carried since R34.

2. **MAJOR-A3**: Fix circuit breaker flapping — `record_success()` should halve failure count on half-open recovery instead of clearing all timestamps (circuit_breaker.py:245). Both models flagged this as a production reliability risk.

3. **MAJOR-A2**: Hoist `settings = get_settings()` once at the top of `_dispatch_to_specialist` instead of calling it 3 times (graph.py:~216, ~259, ~267). Quick fix, eliminates TOCTOU, reduces function calls.

4. **MAJOR-A5**: Add embedding dimension validation in retriever init — query collection metadata and assert dimension match with embedding model output (embeddings.py). Carried from R34, both models re-flagged.

5. **MAJOR-A10**: Add 3-5 more hypothesis tests: circuit breaker state machine transitions, RRF with duplicate/degenerate documents, reducer merge edge cases (test_property_based.py). Incremental improvement toward testing score.

---

## Deferred (Not Blocking, Track for Future)

- MAJOR-A1: Dispatch SRP refactor — significant change, defer to post-MVP (carried from R34-A2)
- MAJOR-A4: Inconsistent purge scopes — design decision needed (carried from R34-A4)
- MAJOR-A6: guest_context reducer — design decision (intentionally derived, fragile under flag toggle)
- MAJOR-A7: CSP nonce passthrough — only relevant when server-rendering HTML (carried from R34-A7)
- MAJOR-A8: Background sweep for rate limiter — optimization, not correctness (carried from R34-A8)
- All MINOR findings — track but don't block
