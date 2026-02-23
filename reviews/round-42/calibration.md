# R42 Calibration: Score Validation + Conservative Estimate

**Date**: 2026-02-23
**Purpose**: Validate R41 post-fix score of 94.9/100. Provide conservative estimate a fresh hostile reviewer would give. Identify remaining gaps per dimension.

---

## Methodology

1. Read actual source code for all 10 dimensions (not summaries)
2. Verify every R41 fix claim against the codebase
3. Apply "fresh hostile reviewer" lens: what would someone score this without knowing prior rounds?
4. Conservative bias: round down on ambiguous evidence, round up only on exceptional code

---

## R41 Fix Verification (Code-Confirmed)

| R41 Claim | Verified? | Evidence |
|-----------|:---------:|----------|
| D6-C001: --require-hashes in Dockerfile | YES | `Dockerfile:19` — `pip install --no-cache-dir --require-hashes --target=/build/deps` |
| D6-M001: SBOM generation | YES | `cloudbuild.yaml:46-54` — Trivy CycloneDX SBOM step |
| D6-M003: curl removed, Python healthcheck | YES | `Dockerfile:68` — `python -c "import urllib.request; ..."` |
| D6-M004: .dockerignore .claude/ | YES | `.dockerignore:34` — `.claude/` and `.hypothesis/` added |
| D1-M001: retrieved_context cleanup | YES | `nodes.py:456,485,583,695` — `"retrieved_context": []` in 4+ nodes |
| D1-M002: specialist metadata in SSE | YES | `graph.py:93-94` — `"specialist": output.get("specialist_name")` |
| D9-M001/M002: pattern count 185 | YES | `ARCHITECTURE.md:22,104,374` — consistent "185" across all occurrences |
| D9-M003: 3 new runbook sections | YES | `runbook.md:251,266,275` — Graceful Shutdown, TTL Jitter, URL Encoding |
| requirements-prod.txt has hashes | YES | `requirements-prod.txt:1-9` — pip-compile generated with `--generate-hashes` |

**All 10 R41 fix claims are code-verified.** No phantom fixes.

---

## Per-Dimension Calibration (Fresh Hostile Reviewer Lens)

### D1: Graph Architecture (Weight: 0.20) — R41 claims 8.9

**Strengths a fresh reviewer would see:**
- 11-node StateGraph with well-defined topology (graph.py:432-567)
- Structured output routing via Pydantic Literal types — no substring matching
- Validation loop with bounded retry (max 1) + fallback
- Specialist DRY extraction in `_base.py` — reduces 600 LOC duplication
- State parity check at import time (graph.py:604-615)
- Feature flag dual-layer architecture (build-time topology vs runtime behavior) — well-documented
- Deterministic keyword fallback when LLM unavailable
- CB-guarded dispatch with TOCTOU hoist, timeout-bounded, unknown-key filtering

**What a hostile reviewer would flag:**
- `_dispatch_to_specialist` is ~195 lines (179-373) — mixes dispatch logic, guest profile lookup, specialist execution, and result filtering. This is the single most-flagged SRP violation across all review rounds. Still unfixed.
- `_DISPATCH_OWNED_KEYS` at module level is a data coupling smell between graph.py and specialist agents
- graph.py is 852 LOC total — the dispatch function alone is 23% of the file

**Conservative score: 8.7** (not 8.9). The SRP issue has been carried since R34 (9 rounds). A fresh reviewer would dock ~0.2 more than the incumbent review which has normalized it. The R41 fixes (retrieved_context cleanup, specialist metadata) are genuine but incremental — worth +0.2 from 8.5, not +0.4.

**Gap to 9.5**: SRP refactor of `_dispatch_to_specialist` into 3 focused functions (+0.5), extract dispatch as its own graph node (+0.3)

---

### D2: RAG Pipeline (Weight: 0.10) — R41 claims 8.5

**Strengths:**
- Per-item chunking with category-specific formatters — correct for structured data
- SHA-256 content hashing for idempotent ingestion
- Version-stamp purging for stale chunks
- RRF reranking with k=60 per original paper
- Pinned embedding model version (`gemini-embedding-001`)
- Embedding retry with exponential backoff

**What a hostile reviewer would flag:**
- No embedding dimension validation at ingestion time (deferred since R36)
- Firestore `_use_server_filter` permanent flip is deferred
- No cross-encoder reranking (acceptable for MVP)

**Conservative score: 8.5** — Fair. No inflation detected. Solid MVP-grade RAG.

**Gap to 9.0**: Embedding dimension validation (+0.25), Vertex AI integration tests (+0.25)

---

### D3: Data Model (Weight: 0.10) — R41 claims 8.5

**Strengths:**
- Clean TypedDict with 3 custom reducers (`_merge_dicts`, `_keep_max`, `_keep_truthy`) — each with documented rationale
- `_merge_dicts` filters None AND empty strings (R38 fix)
- `_keep_max` has explicit None guard (R39 fix)
- `RetrievedChunk` and `GuestContext` typed dicts for explicit contracts
- Pydantic models for all structured outputs with `Literal` types and `max_length` constraints
- Import-time parity check prevents schema drift

**What a hostile reviewer would flag:**
- `guest_context` has no reducer (documented design decision, not a defect)
- No formal state migration pattern for schema changes

**Conservative score: 8.5** — Fair. Clean, well-guarded data model. No inflation.

**Gap to 9.0**: Formal state migration pattern (+0.25), stronger reducer type annotations (+0.25)

---

### D4: API Design (Weight: 0.10) — R41 claims 8.5

**Strengths:**
- FastAPI with lifespan, pure ASGI middleware (not BaseHTTPMiddleware — correct for SSE)
- SIGTERM graceful drain with `_active_streams` set + `_shutting_down` event + 30s timeout
- 6 pure ASGI middleware in correct execution order (documented in runbook)
- SSE heartbeat with `asyncio.wait_for()` per-token
- `aclosing()` for async generator cleanup
- Streaming PII redaction via `StreamingPIIRedactor`
- Version assertion in smoke test (cloudbuild.yaml Step 7)
- Production secrets validator on Settings

**What a hostile reviewer would flag:**
- Rate limiter is in-memory only (documented ADR with upgrade path)
- `/feedback` endpoint logs but doesn't forward to LangFuse (TODO: HEYSEVEN-42)
- `import re` inline inside endpoint function (trivial)
- Runbook still has placeholder URL (`hey-seven-XXXXX.run.app`)

**Conservative score: 8.5** — Fair. Solid API design for MVP. The in-memory rate limiter is documented, not hidden.

**Gap to 9.5**: Redis rate limiting (+0.5), LangFuse feedback forwarding (+0.25), structured request/response logging (+0.25)

---

### D5: Testing Strategy (Weight: 0.10) — R41 claims 8.5

**Strengths:**
- 1700 `def test_` functions across 60 test files (with parametrize, actual passing tests claimed ~2152)
- Property-based tests (Hypothesis)
- E2E pipeline tests, SSE E2E tests
- Schema-dispatching mock LLM pattern
- Coverage gate at 90% in CI
- Domain-specific regulatory invariants tests
- Autouse fixture clearing 13+ singleton caches

**What a hostile reviewer would flag:**
- `test_live_llm.py` — 1 test function, always skipped in CI (no GOOGLE_API_KEY)
- `test_retrieval_eval.py` — 1 parametrized test function, always skips (no populated ChromaDB)
- `test_load.py` — 2 test functions, minimal load testing
- `guest_profile.py` at 56% coverage (Firestore CRUD paths untested)
- `RedisBackend` at 0% coverage
- No mutation testing
- Some test files appear thin — e.g., `test_property_based.py` (5 tests), `test_conversation_scenarios.py` (9 tests)

**Conservative score: 8.3** — Slight downward correction from 8.5. The 1700 test functions are impressive in quantity, but the thin stub files and always-skipped tests inflate the perception. A fresh hostile reviewer would dock for the stub files and the guest_profile coverage gap. The 90% coverage gate is strong but the specific gaps (guest_profile 56%, Redis 0%) are real.

**Gap to 9.0**: Guest profile integration tests (+0.25), Redis backend tests (+0.15), fill out stub test files (+0.1), mutation testing (+0.5 but low ROI)

---

### D6: Docker & DevOps (Weight: 0.10) — R41 claims 8.5

**Strengths (post-R41 fixes):**
- Multi-stage Docker build with SHA-256 digest pinning
- `--require-hashes` enforced with pip-compile generated hashes (genuine supply chain hardening)
- Non-root user, exec-form CMD, graceful shutdown timeout
- Trivy vulnerability scan + SBOM generation (CycloneDX)
- 8-step CI pipeline with test+lint, build, scan, SBOM, deploy, smoke test, version assertion, traffic routing
- `--no-traffic` canary deploy with automatic rollback on failure
- Rollback verification (health check after rollback)
- Artifact Registry (not deprecated gcr.io)
- Per-step timeouts

**What a hostile reviewer would flag:**
- No image signing (cosign) — ADR only, no implementation
- No staging environment (documented as planned)
- No build failure notifications
- No policy enforcement (OPA/Kyverno) — out of scope for Cloud Run
- Runbook has placeholder URL

**Conservative score: 8.3** — Slightly below R41's claim of 8.5. The R41 `--require-hashes` fix is the single biggest D6 improvement (carried since R37). SBOM is genuine. But a fresh hostile reviewer would note: no image signing, no staging, no notifications. The jump from 7.0 to 8.5 in one round (+1.5) is aggressive. I'd put it at +1.3 -> 8.3.

**Gap to 9.0**: Cosign image signing (+0.3), staging environment (+0.3), build failure notifications (+0.1)

---

### D7: Prompts & Guardrails (Weight: 0.10) — R41 claims 9.0

**Strengths:**
- 185 compiled regex patterns across 11 languages
- 5 guardrail layers (injection, responsible gaming, age verification, BSA/AML, patron privacy)
- Input normalization pipeline: URL decode (iterative, max 3) -> HTML unescape -> Unicode Cf strip -> NFKD -> confusables (68 chars) -> delimiters -> whitespace collapse
- Semantic injection classifier (LLM Layer 2) with 5s timeout, fail-closed
- Tagalog/Taglish injection patterns (demographically justified)
- Length guard pre- and post-normalization (8192 chars)
- Domain-aware exclusions ("act as a guide" OK in casino context)

**Assessment:** This is genuinely the strongest dimension. A fresh hostile reviewer would also recognize this — the multilingual coverage and normalization pipeline depth are exceptional.

**Conservative score: 9.0** — Fair. No inflation. Possibly even conservative.

**Gap to 9.5**: Unicode script detection (+0.25), automated adversarial fuzz testing (+0.25)

---

### D8: Scalability & Production (Weight: 0.15) — R41 claims 8.5

**Strengths:**
- TTL jitter on all singleton caches (staggered expiry preventing thundering herd)
- SIGTERM graceful drain with active stream tracking
- Circuit breaker with rolling window, async lock, half-open probe
- `asyncio.Semaphore(20)` for LLM concurrency backpressure
- Cloud Run config: `--concurrency=50`, `--max-instances=10`, `--min-instances=1`, `--cpu-boost`
- Message windowing (MAX_HISTORY_MESSAGES=20)
- InMemoryBackend with `MAX_ACTIVE_THREADS=1000` guard
- 180s request timeout with documented rationale
- Separate locks per client type (main LLM vs validator LLM) — prevents cascading stalls

**What a hostile reviewer would flag:**
- CB state is per-process (documented, but still a real limitation for multi-instance)
- In-memory rate limiter (per-instance, not distributed)
- No load test results documenting actual throughput
- No distributed tracing correlation

**Conservative score: 8.3** — Slightly below R41's 8.5. The per-process CB is a genuine limitation for a "Scalability & Production" dimension. The R41 calibrator already corrected D8 from 9.0 to 8.5, which was right direction. A fresh hostile reviewer focused specifically on scalability would note the per-process limitations more harshly. But the TTL jitter, SIGTERM drain, and semaphore backpressure are genuinely production-grade patterns. Splitting the difference: 8.3.

**Gap to 9.0**: Redis CB state (+0.3), load test results (+0.2), distributed rate limiting (+0.2)

---

### D9: Trade-off Documentation (Weight: 0.05) — R41 claims 8.7

**Strengths (post-R41 fixes):**
- ARCHITECTURE.md (863 LOC) comprehensive with Mermaid diagrams
- Runbook (now ~500 LOC) with 3 new incident response sections
- Inline ADRs in 4+ source files
- Feature flag dual-layer architecture documented
- Pattern counts now accurate (185)
- Staging strategy ADR at top of cloudbuild.yaml

**What a hostile reviewer would flag:**
- No formal `docs/adr/` directory (confirmed: does not exist)
- Runbook has placeholder URL (`hey-seven-XXXXX.run.app`)
- No capacity planning document
- No SLA/SLO definitions
- All ADRs are inline comments, not formal ADR format (context/decision/consequences)

**Conservative score: 8.3** — Below R41's 8.7. The R41 fixes (pattern count correction, runbook expansion) are genuine but incremental. A fresh hostile reviewer would flag the missing formal ADR directory and placeholder URL as unprofessional for a production codebase. The jump from 8.0 to 8.7 (+0.7) is too aggressive for what was primarily documentation cleanup.

**Gap to 9.0**: Formal ADR directory (+0.3), fix placeholder URL (+0.1), capacity planning doc (+0.15), SLA/SLO definitions (+0.15)

---

### D10: Domain Intelligence (Weight: 0.10) — R41 claims 8.5

**Strengths:**
- 5 casino profiles with state-specific regulations
- Self-exclusion procedures for all 5 properties
- 10-language guardrail coverage
- Per-state helplines (CT, PA, NV, NJ)
- Tribal authority distinction
- Per-casino feature flags
- `get_casino_profile(casino_id)` pattern for multi-tenant access

**What a hostile reviewer would flag:**
- Static JSON files only — no live data feeds
- Only Mohegan Sun has detailed property data; other 4 are config-only
- No patron tier/segment modeling
- Knowledge base data is manually curated

**Conservative score: 8.5** — Fair. Domain coverage is comprehensive for MVP. No inflation.

**Gap to 9.0**: Detailed property data for all 5 casinos (+0.25), patron tier modeling (+0.25)

---

## Conservative Score Card

| Dimension | Weight | R41 Claim | Conservative | Delta | Reasoning |
|-----------|--------|:---------:|:------------:|:-----:|-----------|
| D1: Graph Architecture | 0.20 | 8.9 | **8.7** | -0.2 | SRP violation normalized by incumbent reviews; fresh reviewer docks harder |
| D2: RAG Pipeline | 0.10 | 8.5 | **8.5** | 0.0 | Fair — solid MVP RAG |
| D3: Data Model | 0.10 | 8.5 | **8.5** | 0.0 | Fair — clean, well-guarded |
| D4: API Design | 0.10 | 8.5 | **8.5** | 0.0 | Fair — strong ASGI patterns |
| D5: Testing Strategy | 0.10 | 8.5 | **8.3** | -0.2 | Stub files, always-skipped tests, guest_profile 56% |
| D6: Docker & DevOps | 0.10 | 8.5 | **8.3** | -0.2 | +1.5 in one round is aggressive; no signing, no staging |
| D7: Prompts & Guardrails | 0.10 | 9.0 | **9.0** | 0.0 | Genuinely the strongest dimension |
| D8: Scalability & Prod | 0.15 | 8.5 | **8.3** | -0.2 | Per-process CB, no load test results |
| D9: Trade-off Docs | 0.05 | 8.7 | **8.3** | -0.4 | No formal ADR dir, placeholder URL, +0.7 too aggressive |
| D10: Domain Intelligence | 0.10 | 8.5 | **8.5** | 0.0 | Fair — good MVP coverage |

### Weighted Total

| Dimension | Weight | Score | Weighted |
|-----------|--------|------:|--------:|
| D1 | 0.20 | 8.7 | 1.740 |
| D2 | 0.10 | 8.5 | 0.850 |
| D3 | 0.10 | 8.5 | 0.850 |
| D4 | 0.10 | 8.5 | 0.850 |
| D5 | 0.10 | 8.3 | 0.830 |
| D6 | 0.10 | 8.3 | 0.830 |
| D7 | 0.10 | 9.0 | 0.900 |
| D8 | 0.15 | 8.3 | 1.245 |
| D9 | 0.05 | 8.3 | 0.415 |
| D10 | 0.10 | 8.5 | 0.850 |
| **Total** | **1.00** | | **9.360 -> 93.6/100** |

---

## Verdict: Is 94.9 Honest?

**No. 94.9 is inflated by ~1.3 points.**

The R41 fixes are all real and code-verified. The inflation comes from:

1. **D1 +0.4 claim is too generous** — retrieved_context cleanup and specialist metadata are incremental fixes worth +0.2, not +0.4. The SRP issue (carried 9 rounds) should weigh against claiming 8.9.

2. **D6 +1.5 claim is aggressive** — `--require-hashes` is the genuine big fix (+0.8), SBOM (+0.2), curl removal (+0.1), .dockerignore (+0.1) = +1.2 more realistic than +1.5. Missing signing and staging prevent 8.5.

3. **D9 +0.7 claim is the most inflated** — Pattern count corrections and runbook additions are documentation maintenance, not architecture improvements. +0.3 is more honest than +0.7.

4. **D5 and D8 were already corrected by R41 calibrator** (9.0 -> 8.5 each). But 8.5 is still slightly high for both given the specific gaps.

**Conservative estimate for a fresh hostile reviewer: 93.6/100**

This is 1.3 points below R41's claim (94.9) and 1.35 points above R41's pre-fix calibrated baseline (92.25).

---

## Top 5 Remaining Gaps (Highest ROI)

### 1. D1: SRP refactor of `_dispatch_to_specialist` (MEDIUM effort, +0.3-0.5 weighted)
- 195 lines mixing dispatch, profile, execution, filtering
- Carried since R34 (9 rounds)
- Highest weight dimension (0.20) — biggest ROI per point gained
- Split into: `_select_specialist()`, `_enrich_with_profile()`, `_execute_specialist()`

### 2. D6: Cosign image signing (MEDIUM effort, +0.03 weighted)
- ADR exists (cloudbuild.yaml:62-73), needs implementation
- Requires KMS key provisioning
- Low weighted impact but closes a supply chain gap that every hostile reviewer flags

### 3. D8: Redis circuit breaker state (MEDIUM effort, +0.10 weighted)
- Per-process CB documented as limitation
- Second-highest weight dimension (0.15)
- Blocks true multi-instance scalability claims

### 4. D5: Guest profile integration tests (MEDIUM effort, +0.02 weighted)
- guest_profile.py at 56% coverage
- Needs Firestore-mocked integration tests
- Low weighted impact but frequently flagged

### 5. D9: Formal `docs/adr/` directory (LOW effort, +0.01 weighted)
- Extract existing inline ADRs into formal format
- Low weight (0.05) but easy win
- Also fix runbook placeholder URL

---

## Ceiling Analysis (Updated)

| Dimension | Conservative | Ceiling | Gap | Weighted Gap |
|-----------|:------------:|:-------:|:---:|:------------:|
| D1 | 8.7 | 9.3 | 0.6 | 0.120 |
| D2 | 8.5 | 9.0 | 0.5 | 0.050 |
| D3 | 8.5 | 9.0 | 0.5 | 0.050 |
| D4 | 8.5 | 9.3 | 0.8 | 0.080 |
| D5 | 8.3 | 9.0 | 0.7 | 0.070 |
| D6 | 8.3 | 9.0 | 0.7 | 0.070 |
| D7 | 9.0 | 9.5 | 0.5 | 0.050 |
| D8 | 8.3 | 9.3 | 1.0 | 0.150 |
| D9 | 8.3 | 9.0 | 0.7 | 0.035 |
| D10 | 8.5 | 9.0 | 0.5 | 0.050 |
| **Total** | **93.6** | **95.0** | | **+0.725** |

**Realistic ceiling with all achievable improvements: ~95.0/100**

The ceiling is lower than R41's estimate (96.2) because:
- D1 ceiling reduced from 9.5 to 9.3 (full 9.5 would require the dispatch function to become its own graph node — architectural change, not refactor)
- D8 ceiling reduced from 9.5 to 9.3 (true 9.5 requires Redis + load test results + distributed tracing — significant infra investment)
- D4 ceiling reduced from 9.5 to 9.3 (Redis rate limiting alone doesn't get to 9.5)

---

## Score Trajectory (Corrected)

| Round | Claimed | Calibrated | Delta | Notes |
|-------|:-------:|:----------:|:-----:|-------|
| R34 | 77.0 | 77.0 | — | Baseline |
| R40 | 93.5 | 92.25 | -1.25 | D5/D8 inflation corrected |
| R41 | 94.9 | **93.6** | -1.3 | D1/D6/D9 fix deltas too aggressive |

**Net R41 improvement over R40 calibrated: +1.35 points** (real, code-verified improvements).
