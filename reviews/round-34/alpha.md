# R34 Review — Dimensions 1-5 (reviewer-alpha)

**Date**: 2026-02-23
**Reviewer**: reviewer-alpha
**Cross-validated with**: GPT-5.2 Codex (azure_code_review), Gemini 3 Pro (thinking=high)
**Codebase snapshot**: 2019 tests, 51 source modules, 23K LOC

---

## Dimension 1: Graph/Agent Architecture (weight 0.20)

**Score: 8.0 / 10**

### Strengths
- 11-node custom StateGraph with validation loop is architecturally sound — generate -> validate -> retry(max 1) -> fallback
- Specialist DRY extraction via `_base.py` with dependency injection (get_llm_fn, get_cb_fn) eliminates ~600 LOC duplication
- Shared constants module (`constants.py`) with `_KNOWN_NODES` frozenset prevents string typo bugs
- Degraded-pass validation strategy is nuanced and production-grade (first attempt + validator failure = PASS; retry + failure = FAIL)
- Parity check at import time (ValueError if _initial_state and PropertyQAState annotations diverge)
- `_DISPATCH_OWNED_KEYS` properly at module level (R33 fix applied)
- `_LLM_SEMAPHORE = asyncio.Semaphore(20)` for concurrency backpressure in specialist execution
- Feature flag ADR distinguishes build-time (graph topology) vs runtime (per-request) flags — well-documented trade-off

### Findings

**MAJOR-A1: No timeout on agent_fn execution in _dispatch_to_specialist** (graph.py:~390)
- `asyncio.timeout(settings.MODEL_TIMEOUT)` wraps the dispatch LLM call but NOT the `agent_fn(state)` call
- A hung specialist agent (e.g., waiting on external API) blocks the entire graph indefinitely
- The circuit breaker only tracks failures, not latency — a slow-but-not-failing call bypasses CB
- **GPT-5.2 consensus**: Flagged as risk. "No timeout on agent execution means a single slow specialist blocks the thread."
- **Gemini consensus**: Flagged as critical. "agent_fn has no timeout wrapper — unbounded execution."
- **Fix**: Wrap `result = await agent_fn(state)` in `async with asyncio.timeout(settings.MODEL_TIMEOUT * 2):`

**MAJOR-A2: _dispatch_to_specialist SRP violation — routing + injection + execution** (graph.py:~350-430)
- Single function handles: (1) structured LLM dispatch, (2) keyword fallback routing, (3) guest profile injection, (4) agent function execution, (5) dispatch-owned key collision logging, (6) error handling with fallback
- ~80 lines mixing concerns that should be separated
- **GPT-5.2 consensus**: "The function mixes routing decisions with execution and state injection."
- **Gemini consensus**: Flagged strongly. "SRP violation — dispatch does routing, profile injection, and execution in one function."
- **Fix**: Extract guest profile injection and agent execution into separate helper functions

**MAJOR-A3: Circuit breaker created outside try block in _dispatch_to_specialist** (graph.py:~370)
- `cb = get_circuit_breaker()` call happens before the try/except block
- If `get_circuit_breaker()` raises (e.g., asyncio.Lock initialization race), the exception is unhandled
- **GPT-5.2 consensus**: Flagged. "CB creation should be inside the try block."
- **Fix**: Move CB acquisition inside the try block

**MINOR-A1: No validation that agent_fn returns dict** (graph.py:~395)
- `result = await agent_fn(state)` assumes dict return type
- If a specialist returns None or wrong type, downstream code fails with unclear error
- **GPT-5.2 consensus**: Flagged. "No type check on agent_fn return value."
- **Fix**: Add `if not isinstance(result, dict): result = {"messages": [...fallback...]}`

**MINOR-A2: Feature flag function called twice in build_graph** (graph.py)
- `get_feature_flags()` called at graph build time for topology AND at runtime for per-request behavior
- Build-time call is correct (topology is static), but the double-call pattern is confusing
- **GPT-5.2 consensus**: Noted as style issue

**MINOR-A3: asyncio.timeout requires Python 3.11+** (graph.py, _base.py)
- Multiple uses of `asyncio.timeout()` which was added in Python 3.11
- No minimum Python version declared in pyproject.toml or equivalent
- Low risk since Docker image likely pins Python version, but should be documented

---

## Dimension 2: RAG Pipeline (weight 0.10)

**Score: 8.0 / 10**

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
- **Gemini consensus**: Flagged. "Inconsistent purge filter predicates — bulk purges broadly, single purges narrowly."
- **Fix**: Document the intentional difference OR unify purge scope with source parameter

**MAJOR-A5: No embedding dimension validation on model change** (embeddings.py, pipeline.py)
- If `EMBEDDING_MODEL` is changed (e.g., from `gemini-embedding-001` to a new version with different dimensions), existing vectors in ChromaDB have incompatible dimensions
- No validation checks that the embedding dimension of new vectors matches existing collection
- ChromaDB will silently accept mismatched dimensions and return garbage similarity scores
- **Gemini consensus**: Flagged. "No dimension check — model swap corrupts the entire vector store."
- **Fix**: Query collection metadata for dimension count on retriever init, assert match with embedding model output

**MINOR-A4: _get_augmentation_terms uses word-level split, not stemming** (tools.py:58-79)
- `words = set(query_lower.split())` means "pricing" won't match _PRICE_WORDS frozenset containing "price"
- Morphological variants like "scheduling", "priced", "hourly" are missed
- Low severity because the augmentation is a recall-boosting heuristic, not a hard router
- **Fix**: Consider using substring matching or a small stemmer for augmentation term selection

**MINOR-A5: Retriever singleton uses threading.Lock in async context** (pipeline.py)
- `_retriever_lock = threading.Lock()` used in `get_retriever()` which may be called from async code
- Currently safe because retriever init runs in `asyncio.to_thread()`, but the lock type is misleading
- **Fix**: Document that threading.Lock is intentional because retriever runs in thread pool

---

## Dimension 3: Data Model (weight 0.10)

**Score: 8.5 / 10**

### Strengths
- PropertyQAState TypedDict with 17 fields and 4 custom reducers — well-documented
- `_merge_dicts` reducer with clear docstring explaining latest-wins semantics
- `_keep_max` and `_keep_truthy` reducers for retry_count and skip_validation — correct patterns
- `__all__` exports list for clean public API (R33 fix)
- Pydantic models with Literal types: RouterOutput (7 types), DispatchOutput (5 specialists), ValidationResult (3 states)
- RetrievedChunk TypedDict provides pipeline contract
- Import-time parity check catches schema drift between state and _initial_state

### Findings

**MAJOR-A6: guest_context is dict[str, Any] — untyped internal structure** (state.py)
- `guest_context: dict[str, Any]` is the only untyped complex field in the state
- Multiple producers (guest_profile.py, _base.py) write different keys without a shared schema
- A typo in key name (e.g., "dinning_preferences" vs "dining_preferences") is silently accepted
- **Gemini consensus**: Flagged. "guest_context has no schema — producers and consumers can silently diverge."
- **Fix**: Define a GuestContext TypedDict with explicit fields and use it as the type annotation

**MINOR-A6: retry_count has _keep_max reducer but no upper bound in type** (state.py)
- `retry_count: Annotated[int, _keep_max]` allows unbounded values in theory
- The graph's recursion_limit (10) and validate_node logic (max 1 retry) provide runtime bounds
- But the type system doesn't enforce this — a bug in a new node could set retry_count=100
- **Fix**: Add a validator or use `Annotated[int, _keep_max, Field(ge=0, le=5)]` if Pydantic validation is desired

**MINOR-A7: No TypedDict for specialist agent return shape** (state.py, _base.py)
- Specialist agents return `dict` with keys like "messages", "skip_validation", "guest_context"
- No TypedDict defines the expected return shape — each specialist's return is validated only by runtime usage
- **Fix**: Define a SpecialistResult TypedDict for the return contract

---

## Dimension 4: API Design (weight 0.10)

**Score: 8.0 / 10**

### Strengths
- 6 pure ASGI middleware classes — no BaseHTTPMiddleware (correct pattern for SSE)
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
- **Gemini consensus**: Flagged. "Nonce generated but never passed downstream — useless for inline scripts."
- **Fix**: Either pass nonce via ASGI scope for template access, or document that nonce-based CSP is for API-only mode

**MAJOR-A8: Rate limiter stale sweep on request path** (middleware.py:~RateLimitMiddleware)
- Stale client cleanup runs every 100 requests on the request path: `self._request_count += 1; if self._request_count % 100 == 0: self._sweep_stale()`
- Under high load, the sweep (iterating all tracked clients) adds latency to every 100th request
- Under low load, stale entries accumulate until the next 100th request
- **GPT-5.2 consensus**: "Sweep on request path is a latency concern under high load."
- **Fix**: Move sweep to a background asyncio task on a timer (e.g., every 60 seconds)

**MINOR-A8: Streaming body limit potential race in byte counting** (middleware.py:~RequestBodyLimitMiddleware)
- The streaming byte counter increments `bytes_received` per chunk in the receive wrapper
- In theory, if ASGI server calls receive() concurrently (unlikely but spec-allowed), the counter is not atomic
- Very low risk in practice since ASGI receive is typically sequential per request
- **Fix**: Document assumption that receive() is called sequentially

---

## Dimension 5: Testing Strategy (weight 0.10)

**Score: 7.5 / 10**

### Strengths
- 2019 tests collected across 61 test files — significant test volume
- E2E graph tests (`test_graph_v2.py`) test full pipeline with mocked LLMs
- Schema-dispatching mock LLM pattern for multi-node E2E tests
- Golden conversation tests in LLM judge (7 test cases)
- Property-based hypothesis tests for PII redactor, guardrails, streaming redactor
- Conftest with 18 singleton cache clears (autouse, function scope)
- `_disable_semantic_injection_in_tests` fixture prevents fail-closed blocking
- Load test and doc accuracy enforcement tests exist
- Deterministic offline scoring functions for CI (no LLM calls needed)

### Findings

**MAJOR-A9: 25% code coverage despite 2019 tests** (pytest output)
- `pytest --cov` shows ~25% coverage
- 2019 tests but only 25% line coverage suggests many tests exercise the same paths
- Critical paths like error handling branches, fallback paths, and edge cases may be untested
- **Gemini consensus**: Flagged strongly. "25% coverage with 2019 tests is alarming — test quality over quantity."
- **Fix**: Run `pytest --cov --cov-report=html` and identify uncovered critical paths. Target 60%+ for src/agent/ and src/api/

**MAJOR-A10: Only 3 hypothesis fuzz tests** (test_property_based.py)
- Property-based testing is limited to: PII redactor, guardrails, streaming redactor chunks
- Missing hypothesis tests for: RRF reranking edge cases, state reducer interactions, middleware header injection, circuit breaker state transitions, IPv6 normalization edge cases
- **Gemini consensus**: "3 hypothesis tests for a 23K LOC codebase is inadequate."
- **Fix**: Add hypothesis tests for circuit breaker state machine, RRF with duplicate documents, middleware with malformed headers

**MAJOR-A11: No contract tests for specialist agent return shapes** (tests/)
- No test verifies that all 6 specialist agents return dicts with the expected keys
- A specialist returning `{"message": ...}` instead of `{"messages": ...}` would fail at runtime only
- **Gemini consensus**: Flagged. "No parametrized contract tests verifying specialist return schema."
- **Fix**: Add parametrized test: `@pytest.mark.parametrize("agent_fn", ALL_AGENTS)` asserting return keys include "messages"

**MINOR-A9: Conftest clears 18 caches but no test verifies the list is complete** (conftest.py)
- If a new singleton cache is added (e.g., for a new specialist), conftest may miss it
- No import-time or test-time check that all `@lru_cache` or TTLCache instances are in the cleanup list
- **Fix**: Add a test that introspects all modules for `cache_clear` methods and asserts they're in conftest

**MINOR-A10: LLM judge golden conversations hardcoded in source** (llm_judge.py)
- 7 golden conversation test cases are hardcoded in the module (~100 lines)
- Adding new golden conversations requires code changes, not just data file updates
- **Fix**: Move golden conversations to a JSON/YAML fixture file

---

## Cross-Model Consensus Summary

| Finding | GPT-5.2 Codex | Gemini 3 Pro | Severity |
|---------|---------------|--------------|----------|
| A1: No agent_fn timeout | Flagged as risk | Flagged as critical | MAJOR |
| A2: Dispatch SRP violation | Flagged | Flagged strongly | MAJOR |
| A3: CB outside try block | Flagged | Not mentioned | MAJOR |
| A4: Inconsistent purge scopes | Not mentioned | Flagged | MAJOR |
| A5: No embedding dimension validation | Not mentioned | Flagged | MAJOR |
| A6: guest_context untyped | Flagged indirectly | Flagged | MAJOR |
| A7: CSP nonce unused | Not mentioned | Flagged | MAJOR |
| A8: Rate limit sweep latency | Flagged | Not mentioned | MAJOR |
| A9: 25% coverage | Not mentioned | Flagged strongly | MAJOR |
| A10: Only 3 hypothesis tests | Not mentioned | Flagged | MAJOR |
| A11: No contract tests | Flagged indirectly | Flagged | MAJOR |

**Both models agree on**: No agent_fn timeout (A1), dispatch SRP violation (A2), guest_context untyped (A6)
**GPT-5.2 unique**: CB outside try block (A3), rate limit sweep (A8)
**Gemini unique**: Purge scopes (A4), embedding validation (A5), CSP nonce (A7), coverage (A9), hypothesis tests (A10)

---

## Dimension Score Summary

| Dimension | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| 1. Graph Architecture | 0.20 | 8.0 | 1.60 |
| 2. RAG Pipeline | 0.10 | 8.0 | 0.80 |
| 3. Data Model | 0.10 | 8.5 | 0.85 |
| 4. API Design | 0.10 | 8.0 | 0.80 |
| 5. Testing Strategy | 0.10 | 7.5 | 0.75 |
| **Subtotal (dims 1-5)** | **0.60** | — | **4.80** |

**Projected full score** (if dims 6-10 maintain R33 levels ~8.0 avg): 4.80 + 3.20 = **8.00 / 10 (80/100)**

---

## Top 5 Findings for Fixer (Priority Order)

1. **MAJOR-A1**: Add `asyncio.timeout()` around `agent_fn(state)` call in `_dispatch_to_specialist` (graph.py). Both models flagged this as the highest-risk gap.

2. **MAJOR-A9**: Investigate and improve code coverage from 25%. Run `pytest --cov --cov-report=term-missing` to identify critical uncovered paths in src/agent/ and src/api/.

3. **MAJOR-A6**: Define a `GuestContext` TypedDict in state.py to replace `dict[str, Any]` for guest_context. Update producers in guest_profile.py and _base.py.

4. **MAJOR-A11**: Add parametrized contract tests for all 6 specialist agents verifying return dict shape (must include "messages" key, optional "skip_validation", "guest_context").

5. **MAJOR-A5**: Add embedding dimension validation in retriever init — query collection metadata and assert dimension match with embedding model output.

---

## Deferred (Not Blocking, Track for Future)

- MAJOR-A2: Dispatch SRP refactor — significant change, defer to post-MVP
- MAJOR-A3: Move CB acquisition inside try block — low risk, quick fix but not blocking
- MAJOR-A4: Document or unify purge scopes — design decision needed
- MAJOR-A7: CSP nonce passthrough — only relevant when server-rendering HTML
- MAJOR-A8: Background sweep for rate limiter — optimization, not correctness
- MAJOR-A10: More hypothesis tests — incremental improvement
- All MINOR findings — track but don't block
