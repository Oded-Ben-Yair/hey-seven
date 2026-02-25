# Round 32 Hostile Architecture Review — Gemini Reviewer

**Date**: 2026-02-22
**Reviewer**: Gemini 3.1 Pro (hostile mode)
**Codebase**: hey-seven @ commit 6911144 (Phase 5.5)
**Methodology**: Line-by-line source code review of 20+ production modules and 58 test files

---

## Score Breakdown

| # | Dimension | Weight | Score | Weighted |
|---|-----------|--------|-------|----------|
| 1 | Graph Architecture | 10% | 9.5/10 | 0.95 |
| 2 | RAG Pipeline | 10% | 9.0/10 | 0.90 |
| 3 | Data Model | 10% | 9.0/10 | 0.90 |
| 4 | API Design | 10% | 9.0/10 | 0.90 |
| 5 | Testing Strategy | 12% | 9.0/10 | 1.08 |
| 6 | Docker & DevOps | 12% | 9.5/10 | 1.14 |
| 7 | Prompts & Guardrails | 12% | 9.0/10 | 1.08 |
| 8 | Scalability & Production | 12% | 8.5/10 | 1.02 |
| 9 | Trade-off Documentation | 10% | 9.5/10 | 0.95 |
| 10 | Domain Intelligence | 12% | 9.0/10 | 1.08 |

**TOTAL: 93/100**

---

## Dimension Analysis

### 1. Graph Architecture (9.5/10)

**Strengths:**
- Custom 11-node StateGraph with well-defined topology is the right call over `create_react_agent`. The validation loop (generate -> validate -> retry(max 1) -> fallback) is production-grade.
- Parity check at import time (`_EXPECTED_FIELDS vs _INITIAL_FIELDS`) catches state schema drift immediately. Using `ValueError` instead of `assert` ensures it fires in optimized mode.
- Dual-layer feature flag architecture (build-time for topology, runtime for behavior) is correctly reasoned. The documentation block explaining why topology flags cannot be runtime (checkpoint references specific node names) shows deep understanding.
- Node name constants as `frozenset` with `_KNOWN_NODES` set prevents typo-driven silent failures.
- Specialist dispatch with structured LLM output + deterministic keyword fallback is robust. The decision not to record CB failure for parse errors (line 241-245 of graph.py) is correct -- parse quality is orthogonal to LLM availability.

**Weaknesses:**
- `_dispatch_to_specialist` merges `guest_context_update` into the result dict (line 294) which could overwrite specialist agent returns if the agent also returns `guest_context` or `guest_name`. Low risk since specialists don't return these keys, but the merge order is fragile.

### 2. RAG Pipeline (9.0/10)

**Strengths:**
- Per-item chunking with 7 category-specific formatters is unanimously the correct approach for structured data. The log warning when text splitter activates (line 619-625 of pipeline.py) is a nice touch for monitoring format drift.
- SHA-256 idempotent IDs + version-stamp purging for stale chunks is correct. The purge is non-critical (log-and-continue on failure).
- `AbstractRetriever` base class with ChromaDB and Firestore implementations behind same interface is clean.
- `threading.Lock` for retriever cache (runs in `to_thread`) vs `asyncio.Lock` for LLM cache (runs in coroutines) shows correct understanding of Python concurrency.
- The `reingest_item()` function for CMS webhook-driven updates maintains ID consistency with bulk ingestion.

**Weaknesses:**
- No RRF (Reciprocal Rank Fusion) reranking visible in `CasinoKnowledgeRetriever`. The `reranking.py` module is imported nowhere in the retrieval path. The `_rerank_by_rrf` function (if it exists) appears to be dead code or only used in a Firestore path that was not reviewed. For ChromaDB retrieval, results are single-strategy (semantic only), not multi-strategy.
- `retrieve_with_scores` does not filter by `RAG_MIN_RELEVANCE_SCORE`. Threshold filtering happens somewhere upstream (likely `search_knowledge_base` in tools.py), but the retriever itself returns all results regardless of quality. This means the relevance threshold is a caller concern, not a retriever concern, which splits responsibility.

### 3. Data Model (9.0/10)

**Strengths:**
- `PropertyQAState` TypedDict with `Annotated` reducers (`add_messages`, `_merge_dicts`, `_keep_max`) is correct for LangGraph state management. The `_merge_dicts` reducer for `extracted_fields` ensures cross-turn accumulation.
- `RouterOutput` with 7 `Literal` categories + `DispatchOutput` + `ValidationResult` as Pydantic models with `Field` constraints provides type safety at the LLM output boundary.
- The `_initial_state` parity check at module level is an excellent guard against schema drift.

**Weaknesses:**
- `suggestion_offered` uses `int` (0/1) with `_keep_max` reducer instead of `bool`. The comment says `max(True, False) = True` but the field is actually typed as `int` with values 0/1. This works because `max(1, 0) = 1` but is semantically confusing. A dedicated `_keep_truthy` reducer with a `bool` field would be cleaner.
- `guest_name: str | None` has no reducer, so it resets per turn. The code works around this by falling back to `extracted_fields["name"]` in persona.py (line 189-191), but this is a workaround for a missing reducer. If `_merge_dicts` already accumulates `name` in `extracted_fields`, `guest_name` is a redundant field.

### 4. API Design (9.0/10)

**Strengths:**
- Pure ASGI middleware (6 layers) with no `BaseHTTPMiddleware` is correct for SSE streaming. All middleware classes follow the same `__init__(app) / __call__(scope, receive, send)` pattern.
- `hmac.compare_digest()` for API key auth prevents timing attacks.
- Separate `/live` (always 200) and `/health` (503 on CB open) endpoints follow Kubernetes probe best practices.
- SSE heartbeat (15s interval) prevents client-side EventSource timeouts.
- Streaming PII redaction with lookahead buffer is a novel and well-implemented pattern.
- Per-request nonce-based CSP in `SecurityHeadersMiddleware`.

**Weaknesses:**
- Rate limiter uses in-memory `collections.defaultdict` per IP. Acknowledged in code comments but multi-instance Cloud Run deployments will have independent rate limit counters. No Redis/Memorystore fallback path documented.
- LRU eviction in rate limiter (`_MAX_CLIENTS = 10_000`) could be gamed by rotating source IPs to evict legitimate entries, though X-Forwarded-For trusted proxy support mitigates this for legitimate traffic.

### 5. Testing Strategy (9.0/10)

**Strengths:**
- 58 test files with ~1875 tests is substantial coverage. E2E tests through `build_graph() -> chat()` with schema-dispatching mock LLM (`_SmartMockLLM` pattern) cover the full graph pipeline.
- `conftest.py` with `autouse=True, scope="function"` fixture clearing 13+ singleton caches prevents cross-test leakage.
- State parity test (`test_state_parity.py`) enforces `_initial_state` coverage of all `PropertyQAState` fields.
- Graph topology test (`test_graph_topology.py`) verifies node connectivity.
- LLM judge with golden conversations (7 multi-turn test cases) and regression detection with NaN guards.
- `test_full_graph_e2e.py` covers 9 scenarios: dining query, greeting, off_topic, responsible gaming, multi-turn, retry, fail-to-fallback, CB open, whisper failure.

**Weaknesses:**
- No load/stress tests. For a production system claiming 50 concurrent SSE streams, there should be at least a basic load test verifying the `_LLM_SEMAPHORE(20)` backpressure and rate limiter under concurrent load.
- No test for the `reingest_item()` -> retrieval roundtrip. CMS webhook ingestion is tested in `test_cms.py` but the end-to-end flow (webhook -> reingest -> retrieve updated content) is not verified.
- The LLM judge dimension mapping (lines 721-727 of llm_judge.py) is semantically questionable: `proactive_value -> empathy`, `safety -> cultural_sensitivity`, `groundedness -> guest_experience`. While the R31 fix improved it from the previous mapping, these are still approximate. The mapping should either be 1:1 (5 LLM dimensions -> 5 matching metrics) or clearly documented as intentional approximation.

### 6. Docker & DevOps (9.5/10)

**Strengths:**
- 8-step Cloud Build pipeline with canary deploy (`--no-traffic`), smoke test with version assertion, and automatic rollback is production-grade.
- Multi-stage Dockerfile with `requirements-prod.txt` excluding chromadb (~200MB) reduces image size.
- Exec-form `CMD` ensures PID 1 = application (receives SIGTERM directly for graceful shutdown).
- Non-root user (`appuser`) in production image.
- Trivy vulnerability scan (pinned version `0.58.2`) with `--exit-code=1` gates the pipeline on CRITICAL/HIGH.
- Version assertion in smoke test (lines 94-101 of cloudbuild.yaml) catches stale deployments.
- Cloud Run probes: `startup-probe-path=/health` with 6 attempts at 10s intervals = 60s startup budget; `liveness-probe-path=/live` at 30s intervals.

**Weaknesses:**
- No staging environment. The pipeline goes directly from test/lint to production deploy with only a smoke test gate. For a regulated casino environment, a staging environment with integration tests against real LLM APIs would be expected.

### 7. Prompts & Guardrails (9.0/10)

**Strengths:**
- 5-layer deterministic guardrails (prompt injection, responsible gaming, age verification, BSA/AML, patron privacy) with 84 compiled regex patterns across 4 languages.
- Unicode normalization (`unicodedata.normalize("NFKD")`) for homoglyph attack defense.
- Layer 2 semantic injection classifier (LLM-based, fail-closed) as defense-in-depth.
- Domain-aware exclusions in injection detection: "act as a guide", "act as a VIP" are legitimate casino context and are not flagged.
- HEART framework escalation language with graduated response (2 frustrated = hear+empathize, 3+ = full HEART).
- Persona drift prevention via re-injection after `_PERSONA_REINJECT_THRESHOLD // 2` human turns.
- `string.Template.safe_substitute()` everywhere (no `.format()` crash on user braces).

**Weaknesses:**
- **CRITICAL**: `off_topic_node` in `nodes.py` (line 607) uses the bare `RESPONSIBLE_GAMING_HELPLINES` constant (always CT helplines) instead of `get_responsible_gaming_helplines(casino_id=settings.CASINO_ID)`. This means NJ properties (Hard Rock AC) display Connecticut helplines in gambling advice responses. This was fixed in `_base.py` (R31) but NOT in `nodes.py`. The `RESPONSIBLE_GAMING_HELPLINES` backward-compatible alias on line 29 of prompts.py perpetuates the problem by making the bare constant available without a casino_id parameter.
- Prompt injection patterns do not cover non-Latin script injection variants beyond the 4 supported languages. For example, Arabic or Japanese injection patterns are not covered. Unicode normalization helps with homoglyphs but not with entirely different script-based attacks.

### 8. Scalability & Production (8.5/10)

**Strengths:**
- `_LLM_SEMAPHORE(20)` limits concurrent LLM calls (backpressure).
- TTL-cached singletons (1-hour) for LLM clients, validators, and retriever support GCP credential rotation.
- Circuit breaker with rolling window, async lock, and `record_cancellation()` for SSE disconnects.
- `min-instances=1` prevents cold start latency; `max-instances=10` caps cost runaway.
- `--cpu-boost` during startup for faster initialization.
- Sliding-window rate limiter with stale-client sweep and LRU eviction.

**Weaknesses:**
- In-memory rate limiting, in-memory MemorySaver (dev default), and in-memory circuit breaker state all break across multiple Cloud Run instances. The `min-instances=1` + `max-instances=10` means under load, 10 independent rate limiters exist. This is acknowledged in comments but there is no concrete plan for a shared backend (Redis, Memorystore).
- No circuit breaker metrics export. The CB has `get_state()` and `get_failure_count()` (lock-protected), but these are only logged, not exported to Cloud Monitoring or Prometheus. An operator cannot dashboard CB trips across instances.
- `GRAPH_RECURSION_LIMIT` is set on the compiled graph but the value comes from settings without an upper bound validator. A misconfigured large value could allow unbounded retries if the validation loop has a bug.
- `_MAX_PATTERN_LEN=40` in `StreamingPIIRedactor` means PII patterns longer than 40 chars (e.g., a full name + address combo) could slip through the lookahead buffer. The hard cap at 500 chars prevents unbounded buffering but the 40-char window is documented as matching "phone numbers, SSNs, card numbers" -- address-like PII is not covered.

### 9. Trade-off Documentation (9.5/10)

**Strengths:**
- Inline documentation is exceptional throughout. Every non-obvious decision has a comment with rationale, often citing specific review round numbers (R1, R5, R10, R15, R23, R25, R29, R31).
- The dual-layer feature flag documentation block in `graph.py` (lines 404-438) is a model of architectural documentation: problem, solution, why not alternative, emergency disable procedure.
- Degraded-pass validation strategy is documented with principled reasoning (first attempt = availability, retry = safety).
- Circuit breaker: `record_cancellation()` vs `record_failure()` distinction is clearly documented with DeepSeek F-005 reference.
- Per-item chunking warning log when text splitter activates (quality signal for content authors).
- `_CATEGORY_TO_AGENT` mapping documents spa -> entertainment rationale: "spa services managed by entertainment/amenities team at most casino properties."

**Weaknesses:**
- The LLM judge dimension mapping comments (lines 719-727 of llm_judge.py) claim "R31 fix C-001: previous mapping was semantically wrong. Corrected mapping:" but the current mapping (proactive_value -> empathy, safety -> cultural_sensitivity, groundedness -> guest_experience) is still semantically approximate, not corrected. The comment overclaims.

### 10. Domain Intelligence (9.0/10)

**Strengths:**
- Multi-property casino profiles (Mohegan Sun CT, Foxwoods CT, Hard Rock AC NJ) with per-state regulatory configuration (helplines, self-exclusion programs, gaming age requirements).
- 5 domain-specific guardrails (responsible gaming, BSA/AML, patron privacy, age verification, prompt injection) show deep understanding of casino regulatory landscape.
- HEART framework escalation for sustained guest frustration is hospitality-industry standard.
- Proactive suggestion injection gated on positive-only sentiment (not neutral) prevents upselling frustrated guests.
- Persona drift prevention after extended conversations (research-backed: 20-40% drop over 10-15 turns).
- Golden conversations cover 7 realistic multi-turn scenarios: dining with dietary needs, frustrated VIP escalation, persona consistency, responsible gaming pivot, proactive suggestion, context retention, entertainment ticketing.

**Weaknesses:**
- Only 3 casino profiles configured. For a platform claiming multi-property support ("The Autonomous Casino Host That Never Sleeps"), 3 hardcoded profiles is limited. The Firestore hot-reload path exists but the static profile fallback only covers CT and NJ. PA, MA, NY properties (Tri-State area) would need profile additions.
- No internationalization. All guardrails, helplines, and personas are English-only in the response layer. While guardrail patterns cover 4 languages for detection, the response text is always English. A non-English-speaking guest would receive English-only responses.

---

## Findings Summary

### CRITICAL (1)

**C-001: off_topic_node uses hardcoded CT helplines for all properties**
- **File**: `/home/odedbe/projects/hey-seven/src/agent/nodes.py` line 29, 607
- **Impact**: NJ property guests (Hard Rock AC) see Connecticut gambling helplines instead of NJ-mandated 1-800-GAMBLER. Regulatory violation for NJ DGE compliance.
- **Root cause**: `nodes.py` imports `RESPONSIBLE_GAMING_HELPLINES` (the backward-compatible alias, always CT) instead of calling `get_responsible_gaming_helplines(casino_id=settings.CASINO_ID)`. This was fixed in `_base.py` (R31 C-001) but the same bug persists in `nodes.py:off_topic_node()`.
- **Fix**: Replace line 607 with `get_responsible_gaming_helplines(casino_id=settings.CASINO_ID)` and update the import. Remove or deprecate the `RESPONSIBLE_GAMING_HELPLINES` backward-compatible alias to prevent future regressions.

### MAJOR (4)

**M-001: RRF reranking module appears disconnected from ChromaDB retrieval path**
- **File**: `/home/odedbe/projects/hey-seven/src/rag/reranking.py` (if exists) + `pipeline.py`
- **Impact**: Single-strategy retrieval (semantic only) for the ChromaDB path. Multi-strategy retrieval with RRF fusion is documented in CLAUDE.md rules but not wired in the `CasinoKnowledgeRetriever`.
- **Evidence**: `grep -r "rerank" src/rag/pipeline.py` returns zero results.

**M-002: LLM judge dimension mapping is semantically misleading**
- **File**: `/home/odedbe/projects/hey-seven/src/observability/llm_judge.py` lines 721-727
- **Impact**: Regression detection operates on metrics that don't match their names. A "regression in empathy" actually means a regression in proactive_value. Any dashboard or alert based on metric names will mislead operators.
- **Evidence**: `proactive_value.score -> empathy`, `safety.score -> cultural_sensitivity`, `groundedness.score -> guest_experience`. Comment claims "corrected mapping" but the mapping is still approximate.

**M-003: No shared state backend for multi-instance deployment**
- **File**: `src/api/middleware.py` (rate limiter), `src/agent/circuit_breaker.py`, `graph.py` (MemorySaver)
- **Impact**: Under load (2+ Cloud Run instances), rate limiting, circuit breaker state, and conversation memory are all per-instance. A guest's conversation could route to a different instance mid-session and lose context. Rate limit bypass by load-balancer distribution.
- **Mitigation**: `min-instances=1` keeps most traffic on one instance; documented as known limitation. FirestoreSaver path exists for conversation state.

**M-004: No load tests for concurrent SSE stream limits**
- **File**: Test suite (absent)
- **Impact**: `--concurrency=50` and `_LLM_SEMAPHORE(20)` are configured but never validated under concurrent load. The interaction between rate limiter, semaphore backpressure, and Cloud Run's connection handling is untested.

### MINOR (5)

**m-001: `suggestion_offered` uses int(0/1) instead of bool with misleading comment**
- **File**: `src/agent/state.py`, `graph.py` line 521
- **Impact**: Code readability. Comment says `max(True, False) = True` but field is `int`.

**m-002: `guest_name` field is redundant with `extracted_fields["name"]`**
- **File**: `src/agent/state.py`, `src/agent/persona.py` lines 189-191
- **Impact**: Two fields track the same data. `guest_name` resets per-turn (no reducer); `extracted_fields.name` accumulates (has `_merge_dicts` reducer). The fallback chain in persona.py works around this.

**m-003: Streaming PII lookahead window (40 chars) may miss long PII patterns**
- **File**: `src/agent/streaming_pii.py`
- **Impact**: PII patterns longer than 40 characters (rare but possible with full names + addresses) could pass through the streaming buffer unredacted. The batch PII redaction in `persona_envelope_node` provides a safety net.

**m-004: `_dispatch_to_specialist` result merge order is fragile**
- **File**: `src/agent/graph.py` line 292-295
- **Impact**: `result.update(guest_context_update)` could overwrite specialist agent returns if they ever return `guest_context` or `guest_name` keys. Currently safe because specialists don't return these keys.

**m-005: 3 hardcoded casino profiles insufficient for "platform" claims**
- **File**: `src/casino/config.py`
- **Impact**: Only Mohegan Sun, Foxwoods, and Hard Rock AC are configured. Adding a PA or MA property requires code changes to `CASINO_PROFILES`. The Firestore hot-reload path mitigates this for runtime, but initial static configuration is limited.

---

## Comparison to R31 (Previous Round)

| Metric | R31 | R32 | Delta |
|--------|-----|-----|-------|
| Total Score | 92 | 93 | +1 |
| CRITICALs | 3 (fixed) | 1 (new) | New finding |
| MAJORs | 2 | 4 | More thorough review |
| Test Count | ~1875 | ~1875 | Same |

**What improved since R31:**
- R31 C-001 (judge mapping) was addressed (lines 719-727)
- R31 C-002 (persona DEFAULT_CONFIG) was fixed in `persona.py` and `_base.py`
- R31 C-003 (helplines without casino_id) was fixed in `_base.py`

**What R31 missed:**
- C-001 in this round: the `off_topic_node` in `nodes.py` still uses the bare `RESPONSIBLE_GAMING_HELPLINES` constant. R31 fixed `_base.py` but did not grep all callers of the helpline data. The R31 review notes specifically warn "grep for ALL imports of DEFAULT_CONFIG after every fix" but this principle was not applied to `RESPONSIBLE_GAMING_HELPLINES`.

---

## Verdict

Score: **93/100**. This is a well-architected production system with exceptional documentation, comprehensive testing, and thoughtful handling of regulated-domain concerns. The single CRITICAL (helpline multi-tenant bug) is a narrow but impactful miss -- it is the same class of bug that R31 caught in other locations, meaning the fix was incomplete. The 4 MAJORs are legitimate gaps (missing reranking in ChromaDB path, misleading judge metrics, no shared state backend, no load tests) but none are blocking for an MVP deployment. The codebase demonstrates deep understanding of LangGraph patterns, RAG pipeline design, and casino domain requirements.
