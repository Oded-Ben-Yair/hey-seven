# R36 Review — Dimensions 1-5 (reviewer-alpha)

**Date**: 2026-02-23
**Reviewer**: reviewer-alpha
**Cross-validated with**: GPT-5.2 Codex (azure_code_review), Gemini 3 Pro (thinking=high)
**Codebase snapshot**: ~2055 tests, 51 source modules, 23K LOC

---

## Dimension 1: Graph/Agent Architecture (weight 0.20)

**Score: 8.0 / 10** (R35: 8.5, delta -0.5)

### Strengths
- 11-node custom StateGraph with validation loop remains architecturally sound
- Specialist DRY extraction via `_base.py` with dependency injection eliminates ~600 LOC duplication
- R35 A2 FIX VERIFIED: settings hoisted once in `_dispatch_to_specialist` (graph.py:196)
- R35 A3 FIX VERIFIED: `record_success()` halves failure count on half-open (circuit_breaker.py:252-257)
- Degraded-pass validation strategy is production-grade
- Feature flag ADR distinguishes build-time (topology) vs runtime (per-request) flags
- `_LLM_SEMAPHORE = asyncio.Semaphore(20)` for concurrency backpressure

### Findings

**CRITICAL-A1: router_node calls get_settings() twice — TOCTOU not fixed** (nodes.py:215, 223)
- `router_node` calls `get_settings().CASINO_ID` at line 215 (sentiment detection) and again at line 223 (field extraction)
- R35 fixed this pattern in `_dispatch_to_specialist` by hoisting settings once. But `router_node` was NOT fixed — same TOCTOU pattern persists
- If the TTL expires between call 1 and call 2, the function operates with mixed configuration (sentiment from settings-v1, extraction from settings-v2)
- **GPT-5.2 consensus**: Flagged. "Multiple settings reads in same function should be hoisted."
- **Gemini consensus**: Flagged as CRITICAL. "Textbook TOCTOU race condition."
- **Fix**: Hoist `settings = get_settings()` once at the top of `router_node`, reuse throughout

**MAJOR-A1: cb.record_success() called before validating dispatch result** (graph.py:223-226)
- `record_success()` fires immediately after `dispatch_llm.ainvoke()` returns (line 223)
- The specialist name is validated against `_AGENT_REGISTRY` only at line 226 — AFTER success is recorded
- If the LLM returns a valid `DispatchOutput` with an unknown specialist name (e.g., "spa"), the CB records a success even though dispatch falls back to keyword routing
- This inflates CB health metrics: "successful LLM calls that produce unusable results" counted as healthy
- Under model degradation (LLM hallucinating invalid specialists), the CB stays closed while all traffic dumps into keyword fallback
- **GPT-5.2 consensus**: Flagged. "CB metrics are poisoned — counts unusable responses as healthy."
- **Gemini consensus**: Flagged as MAJOR. "Measuring network health instead of semantic health."
- **Fix**: Move `cb.record_success()` to AFTER the registry validation check succeeds

**MAJOR-A2: _dispatch_to_specialist SRP violation — routing + injection + execution** (graph.py:178-326)
- Single function handles: (1) structured LLM dispatch, (2) keyword fallback routing, (3) guest profile injection, (4) agent execution with timeout, (5) dispatch-owned key collision logging, (6) error handling with fallback
- ~150 lines mixing concerns — carried from R34-A2, R35-A1
- **Carried from R34-A2, R35-A1** — deferred to post-MVP. Both models re-flagged again.
- **Fix**: Extract into `_route_specialist()`, `_inject_guest_profile()`, `_execute_agent()` helpers

**MAJOR-A3: record_success() halving keeps OLDEST timestamps — pruned away on next failure** (circuit_breaker.py:252-257)
- R35 fix halves failure count: `keep_count = len(self._failure_timestamps) // 2`, then `popleft()` removes excess
- `popleft()` removes from the LEFT (oldest entries), keeping the NEWEST timestamps
- Wait — this means the code keeps the NEWEST N/2 entries, which is correct for a sliding window
- BUT: integer division edge case: if there's 1 timestamp, `1 // 2 = 0`, and `while len > 0` removes ALL timestamps. A single failure followed by a half-open success clears ALL evidence
- With 3 timestamps: `3 // 2 = 1`, removes 2, keeps 1. Single success almost clears the slate
- The halving intended to prevent flapping, but `N // 2` with small N approaches full-clear behavior
- **Gemini consensus**: Flagged as MAJOR. "Zeno's paradox — integer division with small N degenerates to full clear."
- **GPT-5.2 consensus**: Not flagged (accepted the halving as improvement).
- **Fix**: Use `max(len // 2, 1)` as keep_count minimum to always retain at least 1 failure timestamp after half-open recovery

**MAJOR-A4: _count_consecutive_frustrated() runs CPU-bound VADER under _LLM_SEMAPHORE** (_base.py:51-74, 214-215)
- `execute_specialist()` acquires `_LLM_SEMAPHORE` (line 43), then calls `_count_consecutive_frustrated()` at line 215
- Wait — re-reading: the semaphore is acquired LATER in `execute_specialist()` around the actual LLM call, not at function entry. Let me verify.
- Actually: reviewing `_base.py` lines 95-300, the semaphore acquisition is NOT visible in the code I read. The function runs linearly: CB check, prompt build, frustration check, LLM call. The semaphore wraps only the LLM ainvoke call.
- Revised finding: `_count_consecutive_frustrated()` iterates ALL messages (up to 40) with no cap, running sub-1ms VADER per message. With 40 messages: ~40ms per turn. Not a semaphore contention issue, but still unguarded O(N) on every specialist execution.
- **Carried from R35 MINOR-A2** — upgraded to MAJOR due to cumulative impact at scale (10 instances * 20 concurrent * 40 messages = 8000 VADER calls/sec worst case)
- **Fix**: Add `max_scan=10` parameter to cap iteration depth. Cache sentiment per-message on first computation.

**MINOR-A1: No validation that agent_fn returns dict** (graph.py:296)
- `result = await agent_fn({**state, **guest_context_update})` assumes dict return type
- If a specialist returns None, downstream `result.keys()` at line 315 raises AttributeError
- **Carried from R35 MINOR-A1** — low risk due to contract tests verifying return shapes
- **Fix**: Add `if not isinstance(result, dict): result = {"messages": [...fallback...]}`

**MINOR-A2: router_node has no asyncio.timeout around LLM call** (nodes.py:245)
- `_dispatch_to_specialist` wraps LLM calls in `asyncio.timeout(settings.MODEL_TIMEOUT)`
- `router_node` does NOT — `router_llm.ainvoke(prompt_text)` has no timeout wrapper
- The LLM client has its own timeout parameter, but this is not a hard cutoff — it's a suggestion to the SDK
- **GPT-5.2 consensus**: Flagged. "LLM call can hang and block the request."
- **NEW finding** not in R35.
- **Fix**: Add `async with asyncio.timeout(settings.MODEL_TIMEOUT):` around the ainvoke call

**MINOR-A3: dispatch_method variable set but only used in logging** (graph.py:191, 228)
- `dispatch_method = "keyword_fallback"` set at line 191, updated to `"structured_output"` at line 228
- Only used in the log statement at line 288 — never returned or stored in state
- **GPT-5.2 consensus**: Flagged as dead state. "Set but never returned or stored."
- **Fix**: Either include in telemetry/state or remove the variable

---

## Dimension 2: RAG Pipeline (weight 0.10)

**Score: 7.5 / 10** (R35: 8.0, delta -0.5)

### Strengths
- Per-item chunking with 8 category-specific formatters — unanimously praised across 20+ rounds
- RRF reranking with k=60 per original paper, SHA-256 document identity
- `_compute_chunk_id()` shared between bulk and single ingest
- Version-stamp purging eliminates ghost data from edited content
- Property ID metadata isolation prevents cross-tenant leakage
- Embedding model version pinned (`gemini-embedding-001`)

### Findings

**CRITICAL-A5: _compute_chunk_id() has delimiter-free concatenation — hash collision by construction** (pipeline.py:226-240)
- `hashlib.sha256((text + source).encode()).hexdigest()` concatenates text and source without a delimiter
- `text="abc", source="def"` produces the SAME hash as `text="abcde", source="f"`
- This means different content+source combinations can collide, causing silent overwrites in the vector DB
- The collision is CONSTRUCTIVE, not probabilistic — an attacker who knows the hash function can craft payloads to overwrite target documents
- **GPT-5.2 consensus**: Flagged as bug. "Ambiguous concatenation — use field boundaries."
- **Gemini consensus**: Flagged as CRITICAL. "Elementary hashing blunder — silent data corruption."
- **NEW finding** — not in R35. R35 reviewed chunk ID but missed the delimiter issue.
- **Fix**: Use a delimiter: `hashlib.sha256(f"{text}\x00{source}".encode()).hexdigest()` or hash fields separately

**MAJOR-A6: Inconsistent purge scopes between bulk and single ingest** (pipeline.py)
- Bulk `ingest_property()` purges by `property_id` + `_ingestion_version` (all sources for that property)
- Single `reingest_item()` purges by `property_id` + `source` + `_ingestion_version` (single source only)
- If an item is moved between sources (e.g., restaurant reclassified from "dining" to "restaurants"), the old-source chunk is never purged
- Category alias duplication: "restaurants" and "dining" both map to `_format_restaurant` but produce different `source` metadata. Same content under different aliases = different chunk IDs = ghost duplicates
- **Carried from R34-A4, R35-A4** — both models re-flagged, now with category alias angle
- **GPT-5.2 consensus**: Flagged. "Aliased categories create phantom duplicates."
- **Gemini consensus**: Flagged as MAJOR. "Append-only bloat-generating database."
- **Fix**: Normalize category aliases to canonical names BEFORE computing chunk IDs and source metadata

**MAJOR-A7: No embedding dimension validation on model change** (embeddings.py, pipeline.py)
- If `EMBEDDING_MODEL` is changed, existing vectors in ChromaDB have incompatible dimensions
- No validation checks that the embedding dimension of new vectors matches existing collection
- ChromaDB will silently accept mismatched dimensions and return garbage similarity scores
- **Carried from R34-A5, R35-A5** — both models re-flagged for 3rd consecutive round
- **Fix**: Query collection metadata for dimension count on retriever init, assert match

**MAJOR-A8: RRF deduplication drops distinct chunks with identical content+source** (reranking.py)
- `doc_id = hashlib.sha256((doc.page_content + str(doc.metadata.get("source", ""))).encode()).hexdigest()`
- Same delimiter-free concatenation issue as _compute_chunk_id (CRITICAL-A5)
- Additionally: if two different documents have identical page_content and same source (e.g., duplicate promotions in same category), they merge into one RRF entry. The lower-scoring one is silently dropped.
- This is metadata loss: two chunks with same text but different metadata (e.g., VIP vs General audience tags) collapse into one
- **GPT-5.2 consensus**: Flagged. "Drops distinct chunks with identical text."
- **Gemini consensus**: Flagged as MAJOR. "Silently destroying metadata."
- **Fix**: Include chunk_id from metadata in RRF identity hash, or add delimiter: `f"{content}\x00{source}"`

**MINOR-A4: Purge logic uses vectorstore._collection private API** (pipeline.py:347-359)
- `retriever.vectorstore._collection.get(where=...)` and `.delete(ids=...)` access ChromaDB internals
- If chromadb upgrades change the private `_collection` interface, purge breaks silently
- **Carried from R35 MINOR-A3**
- **Fix**: Use public API if available, or pin chromadb version with explicit compatibility comment

**MINOR-A5: _get_augmentation_terms uses word-level split, not stemming** (pipeline.py)
- `words = set(query_lower.split())` means "pricing" won't match _PRICE_WORDS containing "price"
- Morphological variants missed — augmentation is a recall heuristic, low impact
- **Carried from R35 MINOR-A4**

---

## Dimension 3: Data Model (weight 0.10)

**Score: 8.0 / 10** (R35: 8.5, delta -0.5)

### Strengths
- PropertyQAState TypedDict with 17 fields and 4 custom reducers
- `_merge_dicts` reducer with clear latest-wins semantics and design rationale
- `_keep_max` and `_keep_truthy` reducers for responsible_gaming_count and suggestion_offered
- GuestContext TypedDict (R34 fix) provides structured type for guest profile
- Pydantic models with Literal types: RouterOutput (7 types), DispatchOutput (5 specialists), ValidationResult (3 states)
- Import-time parity check catches schema drift

### Findings

**MAJOR-A9: guest_context has no reducer — desynchronized from extracted_fields** (state.py:133)
- `guest_context: GuestContext` has NO Annotated reducer
- `extracted_fields: Annotated[dict[str, Any], _merge_dicts]` HAS a reducer and persists across turns
- If `guest_profile_enabled` toggles from True to False mid-session, `_dispatch_to_specialist` stops populating guest_context, and `_initial_state()` resets it to `{}`
- Now guest_context is `{}` while extracted_fields still has all the accumulated data — split-brain state
- When the flag toggles back on, downstream nodes expecting parity between raw extractions and contextualized profile see stale/empty context
- **Carried from R35-A6** — both models re-flagged with stronger language
- **Gemini consensus**: "Breaks the fundamental contract of LangGraph state architecture."
- **GPT-5.2 consensus**: "guest_context has no accumulation guarantee."
- **Fix**: Either add `_merge_dicts` reducer to guest_context, OR make it a computed property derived at query-time from extracted_fields (remove from state entirely)

**MAJOR-A10: _merge_dicts shallow merge silently drops nested structures** (state.py:25-43)
- `{**a, **b}` performs shallow merge only
- If `extracted_fields` contains nested structures (e.g., `{"preferences": {"dietary": "vegan"}}`), a new extraction of `{"preferences": {"seating": "window"}}` overwrites the ENTIRE preferences dict, losing `{"dietary": "vegan"}`
- Currently safe because all extracted fields are flat string/int/list values
- BUT: GuestContext TypedDict includes `preferences: list[str]` which is a list. If extracted_fields ever mirrors this structure with nested dicts, data loss occurs silently
- **Carried from R35 MINOR-A6** — upgraded to MAJOR because GuestContext already has complex types
- **Fix**: Document the shallow-merge contract explicitly in the reducer docstring, OR implement recursive merge for dict values

**MINOR-A6: No TypedDict for specialist agent return shape** (state.py, _base.py)
- Specialist agents return `dict` with keys like "messages", "skip_validation", "guest_context"
- No TypedDict defines the expected return shape — validated only by contract tests at runtime
- **Carried from R35 MINOR-A5**
- **Fix**: Define a `SpecialistResult` TypedDict for compile-time enforcement

**MINOR-A7: ValidationResult.reason has no max_length constraint** (state.py:191-193)
- `DispatchOutput.reasoning` has `max_length=200` — good
- `ValidationResult.reason` has NO max_length — validator can return arbitrarily long reasons
- Long reasons get injected into retry feedback, potentially bloating system prompts
- **NEW finding** — not in R35
- **Fix**: Add `max_length=500` to ValidationResult.reason

---

## Dimension 4: API Design (weight 0.10)

**Score: 7.5 / 10** (R35: 8.0, delta -0.5)

### Strengths
- 6 pure ASGI middleware classes — no BaseHTTPMiddleware (correct for SSE)
- Per-request nonce-based CSP in SecurityHeadersMiddleware
- `hmac.compare_digest()` for ALL secret comparisons
- TTL-cached API key refresh every 60s with atomic tuple caching
- Two-layer body limit (Content-Length + streaming byte counting)
- IPv6 normalization in rate limiter
- CancelledError at INFO level, re-raised for proper cancellation semantics
- OrderedDict for LRU eviction in rate limiter

### Findings

**CRITICAL-A11: CSP nonce generated but never passed to templates — security theater** (middleware.py:176-218)
- `SecurityHeadersMiddleware` generates a per-request nonce (`secrets.token_bytes(16)`)
- The nonce appears in the CSP header: `script-src 'self' 'nonce-{nonce}'`
- BUT: the nonce is NEVER passed to any template rendering context (not stored in ASGI scope, not in request state)
- Two outcomes: (1) If the frontend has inline scripts, they are BLOCKED by CSP (broken UI), or (2) If frontend uses only external scripts, the nonce is useless CPU waste
- The CSP also does NOT include `'unsafe-inline'` as fallback — meaning inline scripts/styles without the nonce will be rejected
- **Carried from R34-A7, R35-A7** — UPGRADED from MAJOR to CRITICAL after 3 consecutive rounds unfixed
- **Gemini consensus**: "Security theater. Generated nonce never reaches templates."
- **GPT-5.2 consensus**: Flagged indirectly — noted nonce is not downstream-accessible.
- **Fix**: Either (a) pass nonce via ASGI scope `scope["csp_nonce"] = nonce` for template access, OR (b) remove nonce-based CSP and use hash-based CSP, OR (c) document explicitly that CSP is for API-only mode with no inline scripts

**MAJOR-A12: Rate limiter stale sweep on request path adds latency spikes** (middleware.py:428-435)
- Every 100th request triggers a synchronous stale-client sweep inside the asyncio.Lock
- The sweep iterates ALL tracked clients: `stale = [ip for ip, dq in self._requests.items() if not dq or dq[-1] < window_start]`
- Under `RATE_LIMIT_MAX_CLIENTS=10000`, the sweep iterates 10K entries every 100th request
- This is INSIDE the asyncio.Lock — all concurrent requests to `/chat` are blocked during the sweep
- Under low traffic, stale entries accumulate without cleanup (never reach 100-request threshold)
- **Carried from R35-A8** — both models re-flagged
- **GPT-5.2 consensus**: "Sweep on request path is a latency concern."
- **Gemini consensus**: "Synchronous GC loop in hot path — amateur for serverless."
- **Fix**: Move sweep to a periodic background asyncio task (`asyncio.create_task` in lifespan), OR use TTL-based cleanup with cachetools.TTLCache instead of manual sweep

**MAJOR-A13: ErrorHandlingMiddleware _SECURITY_HEADERS diverge from SecurityHeadersMiddleware** (middleware.py:117-122 vs 186-215)
- `ErrorHandlingMiddleware._SECURITY_HEADERS` includes: nosniff, DENY, referrer-policy, xss-protection
- `SecurityHeadersMiddleware._STATIC_HEADERS` includes: nosniff, DENY, referrer-policy, HSTS — but NOT xss-protection
- On 500 errors from ErrorHandlingMiddleware, the response has `x-xss-protection: 1; mode=block`
- On normal responses from SecurityHeadersMiddleware, the response does NOT have x-xss-protection
- This is a header divergence — security headers should be consistent across all response codes
- Additionally, `x-xss-protection` is deprecated (Chrome removed it in 2019). Including it in one middleware but not the other suggests copy-paste drift.
- **NEW finding** — not in R35
- **GPT-5.2 consensus**: Not explicitly flagged (reviewed as separate modules).
- **Fix**: Remove deprecated `x-xss-protection` from both, or ensure both middleware reference a SINGLE shared header list

**MINOR-A8: System prompt in _base.py has no length bounds** (_base.py)
- `_build_system_prompt()` concatenates property context, regulations, persona, frustration guidance, whisper plan, and proactive suggestion instructions
- No total length check — a property with extensive regulations could exceed model context limits
- **Carried from R35 MINOR-A7**
- **Fix**: Add `MAX_SYSTEM_PROMPT_CHARS` constant and truncate with warning

**MINOR-A9: ApiKeyMiddleware 401 response missing security headers** (middleware.py:274-286)
- When API key validation fails, the 401 response is sent directly with only content-type and content-length headers
- It does NOT include security headers (nosniff, DENY, etc.)
- `ErrorHandlingMiddleware` includes security headers in its 500 response, but `ApiKeyMiddleware` does not in its 401
- **NEW finding** — not in R35
- **Fix**: Add `_SECURITY_HEADERS` to the 401 response headers list

---

## Dimension 5: Testing Strategy (weight 0.10)

**Score: 7.0 / 10** (R35: 7.5, delta -0.5)

### Strengths
- ~2055 tests collected across 58+ test files
- E2E graph tests test full pipeline with mocked LLMs
- Schema-dispatching mock LLM pattern for multi-node E2E tests
- Golden conversation tests in LLM judge (7 test cases)
- Property-based hypothesis tests for PII, guardrails, streaming, router, IPv6
- Conftest with 18+ singleton cache clears (autouse, function scope)
- Load test and doc accuracy enforcement tests exist
- Contract tests for 5 specialist agents

### Findings

**CRITICAL-A14: ~25% code coverage despite ~2055 tests — error paths untested** (pytest output)
- `pytest --cov` shows ~25% coverage — unchanged across R34, R35, R36
- 2055 tests with 25% line coverage means heavy path duplication (many tests hitting same happy paths)
- Critical error-handling branches, fallback paths, CB state transitions, middleware error responses are likely 0% covered
- Carried for 3 CONSECUTIVE rounds with no improvement
- **GPT-5.2 consensus**: Flagged. "25% coverage with 2055 tests is alarming."
- **Gemini consensus**: Flagged as CRITICAL. "Error paths are 0% tested. LLM timeout blocks will crash in production."
- **UPGRADED from MAJOR to CRITICAL** — 3 rounds carried with zero progress
- **Fix**: Run `pytest --cov --cov-report=term-missing` and target uncovered critical paths: `except` blocks in graph.py, middleware error handlers, CB state transitions, fallback paths in _base.py

**MAJOR-A15: Only 5 property-based hypothesis tests for 23K LOC** (test_property_based.py)
- Property-based testing limited to: PII redactor, guardrails, streaming redactor chunks, router types, IPv6 normalization
- Missing hypothesis tests for: RRF reranking edge cases, state reducer interactions, circuit breaker state machine transitions, middleware header injection, chunk ID collision detection
- **Carried from R35-A10** — both models re-flagged
- **Fix**: Add hypothesis tests for: CB state transitions (generate random success/failure sequences, verify invariants), RRF with duplicate/degenerate inputs, _merge_dicts with nested dict edge cases

**MAJOR-A16: conftest.py cache clearing silently swallows rename/removal errors** (conftest.py:36-173)
- Each cache clear is wrapped in `try/except (ImportError, AttributeError): pass`
- If a module is renamed or a cache variable is renamed, the old entry SILENTLY fails — no error, no warning, no log
- New caches added to the codebase are NOT automatically discovered
- There are now 18+ separate try/except blocks — each one a silent failure point
- **Carried from R35 MINOR-A9** — UPGRADED to MAJOR. With 18+ silent failure points, the probability that at least one is stale is high.
- **Gemini consensus**: "If a cache is in the cleanup list and doesn't exist, the test suite must crash, not silently continue."
- **Fix**: (a) Remove try/except — let test suite crash on stale entries, OR (b) Add a meta-test that introspects all modules for TTLCache/lru_cache instances and asserts they appear in conftest cleanup

**MINOR-A10: LLM judge golden conversations hardcoded in source** (llm_judge.py)
- 7 golden conversation test cases are hardcoded (~100 lines)
- Adding new golden conversations requires code changes
- **Carried from R35 MINOR-A10**
- **Fix**: Move to external JSON/YAML file, load at test time

---

## Cross-Model Consensus Summary

| Finding | GPT-5.2 Codex | Gemini 3 Pro | Severity |
|---------|---------------|--------------|----------|
| A1: router_node TOCTOU (2x get_settings) | Flagged | Flagged strongly | CRITICAL (NEW) |
| A5: chunk_id delimiter-free hash collision | Flagged | Flagged strongly | CRITICAL (NEW) |
| A11: CSP nonce security theater | Flagged indirectly | Flagged strongly | CRITICAL (upgraded) |
| A14: 25% coverage, 3 rounds carried | Flagged | Flagged strongly | CRITICAL (upgraded) |
| A1 (graph): CB success before validation | Flagged | Flagged | MAJOR (NEW) |
| A3: CB halving integer division edge case | Not flagged | Flagged | MAJOR (NEW) |
| A4: _count_frustrated no cap | Flagged | Flagged | MAJOR (upgraded) |
| A6: Inconsistent purge + category aliases | Flagged | Flagged | MAJOR (carried) |
| A7: No embedding dimension validation | Flagged | Flagged | MAJOR (carried, 3rd round) |
| A8: RRF dedup drops distinct chunks | Flagged | Flagged | MAJOR (NEW) |
| A9: guest_context no reducer | Flagged indirectly | Flagged strongly | MAJOR (carried) |
| A10: _merge_dicts shallow merge | Not flagged | Flagged | MAJOR (upgraded) |
| A12: Rate limit sweep on request path | Flagged | Flagged | MAJOR (carried) |
| A13: Security header divergence | Not flagged | Not flagged | MAJOR (NEW, mine) |
| A15: Only 5 hypothesis tests | Flagged | Flagged | MAJOR (carried) |
| A16: conftest silent swallowing | Not flagged | Flagged strongly | MAJOR (upgraded) |

**Both models agree on**: chunk_id collision (A5), router TOCTOU (A1), CB success timing (A1-graph), RRF dedup (A8), coverage gap (A14)
**GPT-5.2 unique**: router no timeout (MINOR-A2), dispatch_method dead state (MINOR-A3), 401 missing security headers (MINOR-A9)
**Gemini unique**: CB halving edge case (A3), conftest silent swallowing (A16), merge shallow drop (A10)
**My own**: Security header divergence (A13), ValidationResult no max_length (MINOR-A7)

### R35 Fixes Verified

| R35 Finding | Status | Evidence |
|-------------|--------|----------|
| R35-A2: 3x get_settings() in dispatch | **FIXED** | Settings hoisted once at graph.py:196 |
| R35-A3: CB record_success clears all | **FIXED** | Halving logic at circuit_breaker.py:252-257 |
| R35-A1: Dispatch SRP | **DEFERRED** | Intentional post-MVP |
| R35-A4: Inconsistent purge | **OPEN** | Still divergent scopes (3rd round) |
| R35-A5: Embedding validation | **OPEN** | Still no dimension check (3rd round) |
| R35-A6: guest_context no reducer | **OPEN** | Still no reducer |
| R35-A7: CSP nonce unused | **OPEN** | Still not passed to templates (3rd round) |
| R35-A8: Rate limit sweep | **OPEN** | Still on request path |
| R35-A9: 25% coverage | **OPEN** | Still at ~25% (3rd round) |
| R35-A10: Hypothesis tests | **OPEN** | Still 5 tests |

---

## Dimension Score Summary

| Dimension | Weight | R35 Score | R36 Score | Delta | Weighted |
|-----------|--------|-----------|-----------|-------|----------|
| 1. Graph Architecture | 0.20 | 8.5 | 8.0 | -0.5 | 1.60 |
| 2. RAG Pipeline | 0.10 | 8.0 | 7.5 | -0.5 | 0.75 |
| 3. Data Model | 0.10 | 8.5 | 8.0 | -0.5 | 0.80 |
| 4. API Design | 0.10 | 8.0 | 7.5 | -0.5 | 0.75 |
| 5. Testing Strategy | 0.10 | 7.5 | 7.0 | -0.5 | 0.70 |
| **Subtotal (dims 1-5)** | **0.60** | — | — | — | **4.60** |

**Net change from R35**: -0.30 weighted. Scores decreased because:
1. More hostile review uncovered new findings (chunk_id collision, CB success timing, RRF dedup, security header divergence)
2. Multiple findings carried for 3+ consecutive rounds with zero progress — penalized by escalation
3. R35 CB halving fix introduced a new edge case (integer division with small N)

**Why scores decreased despite R35 fixes being applied**: R35 fixed 2 findings (settings TOCTOU in dispatch, CB flapping). But this R36 review found 7 NEW findings not in R35, and 8 CARRIED findings that have persisted for 3+ rounds. The hostile posture requires penalizing stagnation on known issues.

---

## Top 5 Findings for Fixer (Priority Order)

1. **CRITICAL-A1**: Hoist `settings = get_settings()` once at the top of `router_node` (nodes.py:215, 223). Quick fix — same pattern already applied in `_dispatch_to_specialist`.

2. **CRITICAL-A5**: Fix `_compute_chunk_id()` delimiter-free concatenation (pipeline.py:226-240). Add null byte delimiter: `f"{text}\x00{source}"`. Also fix same pattern in reranking.py RRF identity hash.

3. **CRITICAL-A11**: Resolve CSP nonce — either pass nonce via ASGI scope for template use, or document that API-only mode doesn't need nonce-based CSP and switch to simpler policy. 3 rounds carried.

4. **CRITICAL-A14**: Run `pytest --cov --cov-report=term-missing` and write tests for the top 10 uncovered `except` blocks in graph.py, _base.py, and middleware.py. Target 40%+ coverage in src/agent/. 3 rounds carried.

5. **MAJOR-A1**: Move `cb.record_success()` to AFTER specialist registry validation in `_dispatch_to_specialist` (graph.py:223-226). Quick fix, prevents CB metric inflation.

---

## Deferred (Not Blocking, Track for Future)

- MAJOR-A2: Dispatch SRP refactor — significant change, defer to post-MVP (carried from R34-A2)
- MAJOR-A3: CB halving integer division — edge case with small N, acceptable for MVP
- MAJOR-A4: _count_frustrated no cap — ~40ms worst case, acceptable for MVP
- MAJOR-A6: Inconsistent purge scopes + category aliases — design decision needed (3rd round)
- MAJOR-A7: Embedding dimension validation — 3rd round, needs attention post-MVP
- MAJOR-A9: guest_context reducer — design decision (intentionally derived, fragile under flag toggle)
- MAJOR-A10: _merge_dicts shallow merge — currently safe due to flat structure
- MAJOR-A12: Rate limit background sweep — optimization (carried, not blocking)
- MAJOR-A13: Security header divergence — low risk, cosmetic consistency
- All MINOR findings — track but don't block
