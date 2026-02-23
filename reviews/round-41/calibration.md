# R41 Calibration: Score Validation + Ceiling Analysis

**Date**: 2026-02-23
**Purpose**: Validate R40 calibrated scores against actual code state. Identify inflated scores. Determine realistic ceiling per dimension.

---

## Methodology

For each of the 10 dimensions:
1. Read the actual source code (not just summaries)
2. Compare R40 calibrated score against observable code quality
3. Determine realistic ceiling given current architecture constraints
4. Identify specific 0.5-1.0 point improvements

---

## Per-Dimension Validation

### D1: Graph Architecture (Weight: 0.20)

**R40 Calibrated**: 8.5

**Code Evidence**:
- `graph.py` (850 LOC): 11-node StateGraph with 3 conditional routing points -- well-architected
- `_dispatch_to_specialist()`: Structured LLM dispatch with keyword fallback, TOCTOU hoist, CB-guarded, timeout-bounded, unknown-key filtering, retry-reuse. This is genuinely production-grade.
- `_route_after_validate_v2()`: Clean routing with defensive default.
- State parity check at import time (line 601-612): catches schema drift immediately.
- `_initial_state()` DRY helper with per-turn reset.
- Feature flag dual-layer architecture (build-time topology vs runtime behavior) is well-documented and correct.
- Specialist DRY extraction in `_base.py` with dependency injection -- reduces 600 LOC duplication.

**Concerns**:
- `_dispatch_to_specialist` is ~195 lines -- does dispatch + guest profile + specialist execution + result filtering. This mixes dispatch, profile lookup, and execution in one function. SRP refactor is the correct remaining finding (carried since R34).
- The `graph.py` module-level `_DISPATCH_OWNED_KEYS` is a code smell (data coupling between graph.py and specialist agents).

**R41 Score: 8.5** -- Fair. The architecture is genuinely strong. SRP refactor would bring it to 9.0 but it's explicitly deferred. No inflation.

**Ceiling: 9.0-9.5** -- SRP refactor of `_dispatch_to_specialist` (+0.5), extract dispatch as its own graph node (+0.5 if warranted).

---

### D2: RAG Pipeline (Weight: 0.10)

**R40 Calibrated**: 8.5 (was 8.0 pre-R40 calibrated, bumped +0.5 by R40 embedding retry fixes)

**Code Evidence**:
- `pipeline.py`: Per-item chunking with category-specific formatters (`_format_restaurant`, `_format_entertainment`, `_format_hotel`) -- correct pattern for structured data.
- SHA-256 content hashing for idempotent ingestion.
- Version-stamp purging for stale chunks.
- `reranking.py`: RRF implementation with k=60 per original paper.
- `embeddings.py`: Pinned embedding model version (`gemini-embedding-001`).
- R40 fix: Embedding retry with exponential backoff (max 3) during ingestion.

**Concerns**:
- `requirements-prod.txt` does NOT contain hashes -- the `--require-hashes` comment in Dockerfile is documentation only (TODO: HEYSEVEN-58). This is a D6 issue, not D2, but worth noting.
- Firestore `_use_server_filter` permanent flip is deferred (design decision).
- No embedding dimension validation at ingestion time (deferred since R36).

**R41 Score: 8.5** -- Fair. The R40 embedding retry is a genuine improvement. RAG pipeline is solid for an MVP.

**Ceiling: 9.0** -- Embedding dimension validation (+0.25), Firestore filter retry (+0.25), production Vertex AI integration tests (+0.5). True 10/10 requires cross-encoder reranking which is out of scope.

---

### D3: Data Model (Weight: 0.10)

**R40 Calibrated**: 8.5

**Code Evidence**:
- `state.py` (210 LOC): Clean TypedDict with 3 custom reducers (`_merge_dicts`, `_keep_max`, `_keep_truthy`).
- `_merge_dicts`: Filters None AND empty strings (R38 fix) -- correct.
- `_keep_max`: None guard with explicit check instead of `or 0` (R39 fix) -- correct.
- `_keep_truthy`: Sticky boolean for suggestion_offered.
- `RetrievedChunk` and `GuestContext` typed dicts for explicit contracts.
- Pydantic models for all structured outputs (`RouterOutput`, `DispatchOutput`, `ValidationResult`) with `Literal` types.
- `_initial_state` parity check at import time.

**Concerns**:
- `guest_context` has no reducer (by design -- it's derived data, not accumulated). This is a documented design decision, not a defect.
- No `Annotated` reducer on `responsible_gaming_count` -- wait, it DOES have `_keep_max` reducer. Good.

**R41 Score: 8.5** -- Fair. Data model is clean and well-guarded. No inflation.

**Ceiling: 9.0** -- Only improvement would be stronger type annotations on reducer inputs, or a formal state migration pattern for schema changes. Minor.

---

### D4: API Design (Weight: 0.10)

**R40 Calibrated**: 8.5

**Code Evidence**:
- `app.py` (664 LOC): FastAPI with lifespan, pure ASGI middleware, SSE streaming via `astream_events` v2.
- R40 SIGTERM graceful drain: `_active_streams` set + `_shutting_down` event + `_DRAIN_TIMEOUT_S=30`. New requests get 503 during shutdown. Solid.
- Middleware stack: 6 pure ASGI middleware (BodyLimit outermost, RateLimit innermost) -- correct execution order documented.
- SSE heartbeat with `asyncio.wait_for()` per-token (not elapsed time check).
- `aclosing()` for async generator cleanup on timeout/disconnect.
- Streaming PII redaction via `StreamingPIIRedactor`.
- Pydantic v2 models for all request/response types.
- Version assertion in smoke test (cloudbuild.yaml Step 7).
- Production secrets validator on Settings (API_KEY, CMS_WEBHOOK_SECRET, TELNYX_PUBLIC_KEY).

**Concerns**:
- Request ID sanitization is good but uses inline `import re` inside the endpoint function (line 257). Trivial.
- `/feedback` endpoint logs but doesn't forward to LangFuse (TODO: HEYSEVEN-42). Documented, not a defect.
- Rate limiter is in-memory only (documented ADR with upgrade path).

**R41 Score: 8.5** -- Fair. The R40 SIGTERM drain is a genuine improvement. API design is strong for an MVP.

**Ceiling: 9.5** -- Redis rate limiting (+0.5), LangFuse feedback forwarding (+0.25), structured request/response logging (+0.25). 10/10 would need distributed tracing correlation end-to-end.

---

### D5: Testing Strategy (Weight: 0.10)

**R40 Calibrated**: 9.0 (was 8.0 pre-R40, bumped +1.0 by 22 new tests + CI fix)

**Code Evidence**:
- 59 test files, ~1699 `def test_` functions (summary claims ~2168 passing -- includes parametrized).
- 28,327 total lines in test files.
- Property-based tests (Hypothesis) in `test_property_based.py` and `test_state_parity.py`.
- E2E pipeline tests in `test_full_graph_e2e.py`, `test_sse_e2e.py`.
- Phase integration tests: `test_phase2_integration.py`, `test_phase3_integration.py`, `test_phase4_integration.py`.
- Domain-specific: `test_regulatory_invariants.py`, `test_r24_domain.py`, `test_tenant_isolation.py`.
- Schema-dispatching mock LLM pattern in E2E tests.
- R40 fix: Autouse fixture disabling API_KEY in tests (fixed 52 broken tests).
- Coverage gate: `--cov-fail-under=90` in cloudbuild.yaml.
- Coverage at 90.11% (post R40 fix).

**Concerns**:
- `guest_profile.py` at 56% coverage (deferred -- Firestore CRUD needs integration infra).
- `RedisBackend` at 0% coverage (Redis not in scope for MVP).
- `test_live_llm.py` and `test_retrieval_eval.py` have only 1 test function each -- likely stubs.
- `test_load.py` has only 2 test functions -- minimal load testing.

**POSSIBLE INFLATION**: R40 jumped D5 from 8.0 to 9.0 (+1.0). The 22 new tests and CI fix are genuine, but 9.0 implies "near-excellent" testing. The guest_profile.py and RedisBackend coverage gaps are real (even if deferred). 2168 passing tests is impressive, but some test files are thin stubs. I'd put this at **8.5** rather than 9.0.

**R41 Score: 8.5** -- Slight downward correction from 9.0. The coverage gaps (guest_profile 56%, Redis 0%) and thin stub files prevent a true 9.0.

**Ceiling: 9.5** -- Guest profile integration tests (+0.25), Redis backend tests (+0.25), mutation testing (+0.5). True 10/10 requires chaos engineering / fault injection tests.

---

### D6: Docker & DevOps (Weight: 0.10)

**R40 Calibrated**: 7.0

**Code Evidence**:
- `Dockerfile` (80 LOC): Multi-stage build, SHA-256 digest pinning, non-root user, HEALTHCHECK, exec-form CMD, graceful shutdown timeout.
- `.dockerignore`: Comprehensive exclusion list.
- `docker-compose.yml`: Clean dev setup with volumes and health checks.
- `cloudbuild.yaml` (174 LOC): 8-step pipeline with test+lint, Trivy vulnerability scan, smoke test, version assertion, automatic rollback on failure, `--no-traffic` canary deploy.
- Artifact Registry (not deprecated gcr.io).
- Per-step timeouts in Cloud Build.
- Rollback verification (health check after rollback).

**Concerns**:
- `--require-hashes` is still TODO (HEYSEVEN-58). `pip install --no-cache-dir` without hash verification.
- No SBOM generation (v2 scope).
- No image signing (v2 scope).
- No build failure notifications (v2 scope).
- `requirements-prod.txt` has no hashes. The Dockerfile comment is aspirational documentation.
- No separate staging environment (documented as planned in cloudbuild.yaml ADR).

**R41 Score: 7.0** -- Fair. The fundamentals are solid (digest pin, Trivy, non-root, exec-form CMD, canary deploy with rollback). But supply chain hardening (hashes, SBOM, signing) is entirely absent. For a seed-stage MVP, 7.0 is accurate.

**Ceiling: 8.5** -- `--require-hashes` with pip-compile (+0.5), SBOM generation (+0.5), staging environment (+0.5). True 10/10 requires image signing, policy enforcement (OPA/Kyverno), and k8s-level supply chain (cosign, in-toto attestations) -- none of which are realistic for Cloud Run single-container MVP.

---

### D7: Prompts & Guardrails (Weight: 0.10)

**R40 Calibrated**: 9.0

**Code Evidence**:
- `guardrails.py` (673 LOC): 5 guardrail layers (prompt injection, responsible gaming, age verification, BSA/AML, patron privacy).
- 13 Latin injection patterns + 24 non-Latin patterns (Arabic, Japanese, Korean, French, Vietnamese, Hindi, Tagalog/Taglish).
- Responsible gaming: 43 patterns across 10 languages.
- BSA/AML: 32 patterns across 8 languages.
- Age verification: 12 patterns with Hindi/Tagalog coverage.
- Patron privacy: 12 patterns with Spanish/Tagalog.
- Input normalization: iterative URL decode (max 3), HTML unescape, Unicode Cf stripping, NFKD normalization, confusable table (68 chars), delimiter stripping, whitespace collapse.
- Length guard: 8192 chars pre- and post-normalization (DoS prevention).
- Semantic injection classifier (LLM Layer 2) with 5s timeout and fail-closed behavior.
- Confusables translation table via `str.maketrans()` for O(n) single-pass.

**Assessment**: This is genuinely the strongest dimension. 6 rounds of sustained security hardening. 3 CRITs + 13 MAJORs fixed. The multilingual coverage across 10 languages is exceptional for a casino AI agent. The normalization pipeline (URL decode -> Cf strip -> NFKD -> confusables -> delimiters) is defense-in-depth done correctly.

**R41 Score: 9.0** -- Fair. Possibly even slightly conservative given the breadth and depth.

**Ceiling: 9.5** -- Unicode Script detection (block mixed-script messages) (+0.25), automated fuzz testing of guardrails (+0.25). True 10/10 would need formal adversarial red-teaming with documented test results.

---

### D8: Scalability & Production (Weight: 0.15)

**R40 Calibrated**: 9.0 (was 8.0 pre-R40, bumped +1.0 by thundering herd fix + SIGTERM drain)

**Code Evidence**:
- TTL jitter on all 8 singleton caches (R40 fix -- `ttl=3600 + random.randint(0, 300)`).
- SIGTERM graceful drain with active stream tracking (R40 fix).
- Circuit breaker with rolling window, async lock, half-open probe.
- `asyncio.Semaphore(20)` for LLM concurrency backpressure.
- `--concurrency=50` with `--max-instances=10` Cloud Run config.
- `--min-instances=1` for cold-start avoidance.
- `--cpu-boost` for startup CPU allocation.
- Message windowing (`MAX_HISTORY_MESSAGES=20`).
- InMemoryBackend with `MAX_ACTIVE_THREADS=1000` guard.
- 180s request timeout with documented rationale.

**Concerns**:
- Retriever lock during construction (deferred -- mitigated by R39 lock-free fast path).
- Cloud Run memory at 2Gi -- the runbook says this is for "LangGraph + embeddings + Firestore client overhead". Possibly over-provisioned for a single-worker instance.
- CB state is per-process (documented known limitation with upgrade path).

**POSSIBLE INFLATION**: R40 jumped D8 from 8.0 to 9.0 (+1.0). The thundering herd fix and SIGTERM drain are genuine improvements, but 9.0 implies "near-excellent" scalability. Per-process CB, retriever lock during construction, and no distributed state are real limitations. I'd put this at **8.5** rather than 9.0.

**R41 Score: 8.5** -- Slight downward correction from 9.0. The per-process CB and retriever construction lock prevent a true 9.0 for a "Scalability & Production" dimension.

**Ceiling: 9.5** -- Redis CB state (+0.5), retriever lock refactor (+0.25), load test results documenting actual throughput (+0.25). True 10/10 requires autoscaling validation, multi-region, and distributed tracing correlation.

---

### D9: Trade-off Documentation (Weight: 0.05)

**R40 Calibrated**: 8.0

**Code Evidence**:
- `ARCHITECTURE.md` (863 LOC): Comprehensive system overview with Mermaid diagrams, node descriptions, middleware stack, feature flags, deployment config.
- `docs/runbook.md` (468 LOC): Service config, probe config, incident playbooks, environment config.
- Inline ADRs in 4 source files: LLM concurrency limits, checkpointer choice, rate limiter strategy, i18n decision.
- `cloudbuild.yaml` staging strategy ADR at file top.
- Circuit breaker has a 3-phase upgrade path documented inline.
- Feature flag dual-layer architecture documented in `graph.py` (lines 479-512).
- Trade-off comments throughout: "Why not all runtime?", "Known limitation (multi-instance)", "PLANNED (production)", etc.

**Concerns**:
- No dedicated `docs/adr/` directory with formal ADR format (context, decision, consequences). All ADRs are inline comments.
- Runbook uses placeholder URL (`hey-seven-XXXXX.run.app`).
- No capacity planning document.
- No SLA/SLO definitions documented.

**R41 Score: 8.0** -- Fair. Documentation is embedded inline (good for discoverability) but lacks formal ADR structure. Appropriate for a seed-stage product.

**Ceiling: 8.5-9.0** -- Formal ADR directory (+0.5), capacity planning doc (+0.25), SLA/SLO definitions (+0.25). Low weight (0.05) means ROI is minimal.

---

### D10: Domain Intelligence (Weight: 0.10)

**R40 Calibrated**: 8.5

**Code Evidence**:
- 5 casino profiles (Mohegan Sun, Parx, Wynn, Hard Rock, Borgata) with state-specific regulations.
- Self-exclusion procedures for all 5 properties.
- Multilingual guardrails: 10 languages across 5 guardrail categories.
- Responsible gaming helplines per state (CT, PA, NV, NJ).
- Tribal authority distinction (Mohegan Tribal Gaming Commission vs PGCB vs NGC).
- Per-casino feature flags.
- Casino host persona with property-specific branding.
- `get_casino_profile(casino_id)` pattern for multi-tenant data access.

**Concerns**:
- Knowledge base data is static JSON files -- no live data feeds.
- Only Mohegan Sun has detailed property data (`data/mohegan_sun.json`); other 4 are config-only.
- No patron tier/segment modeling.

**R41 Score: 8.5** -- Fair. Domain coverage is comprehensive for an MVP with 5 casinos. Static data is appropriate for seed stage.

**Ceiling: 9.0** -- Live data feeds (+0.5), patron tier modeling (+0.25), dynamic hours/events (+0.25). True 10/10 requires PMS/CRM integration which is product roadmap, not code quality.

---

## Calibrated Score Card

| Dimension | Weight | R40 Post-Fix | R41 Calibrated | Delta | Reasoning |
|-----------|--------|:------------:|:--------------:|:-----:|-----------|
| D1: Graph Architecture | 0.20 | 8.5 | **8.5** | 0.0 | Fair -- genuinely strong architecture |
| D2: RAG Pipeline | 0.10 | 8.5 | **8.5** | 0.0 | Fair -- embedding retry is genuine improvement |
| D3: Data Model | 0.10 | 8.5 | **8.5** | 0.0 | Fair -- clean reducers, typed contracts |
| D4: API Design | 0.10 | 8.5 | **8.5** | 0.0 | Fair -- SIGTERM drain, SSE streaming, pure ASGI |
| D5: Testing Strategy | 0.10 | 9.0 | **8.5** | -0.5 | Slightly inflated -- coverage gaps (guest_profile 56%, Redis 0%), stub test files |
| D6: Docker & DevOps | 0.10 | 7.0 | **7.0** | 0.0 | Fair -- fundamentals solid, supply chain hardening absent |
| D7: Prompts & Guardrails | 0.10 | 9.0 | **9.0** | 0.0 | Fair -- strongest dimension, 10-language coverage |
| D8: Scalability & Prod | 0.15 | 9.0 | **8.5** | -0.5 | Slightly inflated -- per-process CB, retriever lock, no distributed state |
| D9: Trade-off Docs | 0.05 | 8.0 | **8.0** | 0.0 | Fair -- good inline docs, no formal ADRs |
| D10: Domain Intelligence | 0.10 | 8.5 | **8.5** | 0.0 | Fair -- 5 casinos, 10 languages |

### Weighted Total

| Dimension | Weight | Score | Weighted |
|-----------|--------|------:|--------:|
| D1 | 0.20 | 8.5 | 1.700 |
| D2 | 0.10 | 8.5 | 0.850 |
| D3 | 0.10 | 8.5 | 0.850 |
| D4 | 0.10 | 8.5 | 0.850 |
| D5 | 0.10 | 8.5 | 0.850 |
| D6 | 0.10 | 7.0 | 0.700 |
| D7 | 0.10 | 9.0 | 0.900 |
| D8 | 0.15 | 8.5 | 1.275 |
| D9 | 0.05 | 8.0 | 0.400 |
| D10 | 0.10 | 8.5 | 0.850 |
| **Total** | **1.00** | | **9.225 -> 92.25/100** |

**R40 Post-Fix Total**: 93.5/100
**R41 Calibrated Total**: 92.25/100
**Delta**: -1.25 points (corrections on D5 and D8 inflation)

---

## Ceiling Analysis

| Dimension | Current | Ceiling | Gap | What Gets You There |
|-----------|:-------:|:-------:|:---:|---------------------|
| D1: Graph Architecture | 8.5 | 9.5 | 1.0 | SRP refactor of `_dispatch_to_specialist` (+0.5), extract dispatch node (+0.5) |
| D2: RAG Pipeline | 8.5 | 9.0 | 0.5 | Embedding dim validation (+0.25), Vertex AI integration tests (+0.25) |
| D3: Data Model | 8.5 | 9.0 | 0.5 | Formal state migration pattern (+0.25), stronger reducer type annotations (+0.25) |
| D4: API Design | 8.5 | 9.5 | 1.0 | Redis rate limiting (+0.5), LangFuse feedback (+0.25), structured logging (+0.25) |
| D5: Testing Strategy | 8.5 | 9.5 | 1.0 | Guest profile integration tests (+0.25), Redis tests (+0.25), mutation testing (+0.5) |
| D6: Docker & DevOps | 7.0 | 8.5 | 1.5 | `--require-hashes` (+0.5), SBOM (+0.5), staging env (+0.5) |
| D7: Prompts & Guardrails | 9.0 | 9.5 | 0.5 | Unicode script detection (+0.25), adversarial fuzz testing (+0.25) |
| D8: Scalability & Prod | 8.5 | 9.5 | 1.0 | Redis CB state (+0.5), retriever lock refactor (+0.25), load test results (+0.25) |
| D9: Trade-off Docs | 8.0 | 9.0 | 1.0 | Formal ADR directory (+0.5), capacity planning (+0.25), SLA/SLO doc (+0.25) |
| D10: Domain Intelligence | 8.5 | 9.0 | 0.5 | Live data feeds (+0.5) |

### Weighted Ceiling

| Dimension | Weight | Ceiling | Weighted |
|-----------|--------|--------:|--------:|
| D1 | 0.20 | 9.5 | 1.900 |
| D2 | 0.10 | 9.0 | 0.900 |
| D3 | 0.10 | 9.0 | 0.900 |
| D4 | 0.10 | 9.5 | 0.950 |
| D5 | 0.10 | 9.5 | 0.950 |
| D6 | 0.10 | 8.5 | 0.850 |
| D7 | 0.10 | 9.5 | 0.950 |
| D8 | 0.15 | 9.5 | 1.425 |
| D9 | 0.05 | 9.0 | 0.450 |
| D10 | 0.10 | 9.0 | 0.900 |
| **Ceiling Total** | **1.00** | | **10.175 -> 96.2/100** |

**Realistic ceiling**: ~96/100 with all achievable improvements (no k8s, no multi-region, no formal red-teaming).

---

## Inflation Detection Summary

| Dimension | R40 Score | R41 Score | Inflated? | Evidence |
|-----------|:---------:|:---------:|:---------:|----------|
| D1 | 8.5 | 8.5 | No | Code validates the score |
| D2 | 8.5 | 8.5 | No | Embedding retry is genuine |
| D3 | 8.5 | 8.5 | No | Clean reducers verified |
| D4 | 8.5 | 8.5 | No | SIGTERM drain verified in source |
| D5 | **9.0** | **8.5** | **Yes** (-0.5) | Coverage gaps exist (guest_profile 56%, Redis 0%), stub test files |
| D6 | 7.0 | 7.0 | No | Supply chain gaps acknowledged |
| D7 | 9.0 | 9.0 | No | Code matches -- 10-language, 5-layer guardrails |
| D8 | **9.0** | **8.5** | **Yes** (-0.5) | Per-process CB, retriever lock, no load test results |
| D9 | 8.0 | 8.0 | No | Inline ADRs verified |
| D10 | 8.5 | 8.5 | No | 5 casinos, 10 languages verified |

**Total inflation corrected**: -1.0 raw points (-1.25 weighted)

---

## Highest ROI Improvements (Effort-Ranked)

### 1. D6: `--require-hashes` in Dockerfile (LOW effort, +0.5 on D6)
```bash
pip-compile --generate-hashes --output-file=requirements-prod.txt requirements-prod.in
# Then: pip install --require-hashes --no-cache-dir -r requirements-prod.txt
```
Closes the most frequently cited D6 finding. Carried since R38.

### 2. D8: Retriever lock refactor (LOW effort, +0.25 on D8)
Move ChromaDB client construction outside the lock. Lock-free fast path already exists (R39), but the construction lock remains.

### 3. D9: Create `docs/adr/` directory with 4-5 existing inline ADRs (LOW effort, +0.5 on D9)
Extract existing inline ADRs (rate limiter, checkpointer, concurrency, staging) into formal ADR format.

### 4. D5: Guest profile integration tests (MEDIUM effort, +0.25 on D5)
Add Firestore-mocked integration tests for guest_profile.py CRUD paths.

### 5. D1: SRP refactor of `_dispatch_to_specialist` (MEDIUM effort, +0.5 on D1)
Split into dispatch + execution + profile enrichment functions. Carries since R34.

**If all 5 completed**: +1.25 weighted points -> ~93.5/100
