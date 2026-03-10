# Hey Seven R112 — Code Architecture Review (D1-D10)

**Reviewer**: GPT-5.3 Codex (via Azure AI Foundry MCP)
**Date**: 2026-03-10
**Model**: gpt-5.3-codex
**Scope**: D1-D10 Code Architecture dimensions only
**Method**: Two focused MCP calls (D1-D5 + D6-D10) with inline code excerpts

---

## D1. Graph Architecture (Weight: 0.20)
**Score: 6/10**

**Evidence:**
- Node-name constants (no magic strings), import-time state parity check, explicit validation retry cap (max 1), and a recursion limit setting.
- Architecture is clearly over-concentrated: `graph.py` 710 LOC, `nodes.py` 1702 LOC, `_base.py` 1578 LOC. That is not clean SRP; that is "god-module" territory.
- Generate-node extraction into `dispatch.py` helps, but it's partial relief, not architectural correction.

**Gap:**
- Maintainability risk is high because core orchestration and behavior logic are still monolithic.
- Boundedness is "configured," but not visibly proven by exhaustive routing tests.
- Recursion/infinite-loop prevention depends on config discipline; if call-sites bypass the limit or misuse compile/runtime config, you can still hang.

**Fix:**
- Split by concern: routing, retrieval, generation, validation, response into separate node modules with hard LOC budgets.
- Add an explicit `step_budget` counter in state, decremented per node, hard-fail at zero.
- Add table-driven route tests for every conditional edge and dead-end detection tests.
- Enforce recursion limit at invocation boundary, not just as an attribute.

---

## D2. RAG Pipeline (Weight: 0.10)
**Score: 7/10**

**Evidence:**
- Per-item chunk formatting for structured categories is solid.
- RRF implementation is fundamentally correct (`1/(k+rank+1)`, `k=60`).
- Idempotent hashing and version-stamp purge are exactly what production ingestion needs.
- Embedding model is pinned to `gemini-embedding-001`.

**Gap:**
- `pipeline.py` at 1203 LOC is another monolith.
- Doc identity uses `page_content + source`; that can collide across tenants/properties with duplicated content.
- "Pinned" model name is still a provider alias, not an immutable revision hash.
- No evidence of retrieval quality regression metrics (Recall@K / nDCG@K) to catch relevance drift.

**Fix:**
- Include tenant/property/chunk index/schema version in chunk IDs.
- Add deterministic tie-break in rerank sorting (e.g., doc_id secondary key).
- Persist embedding model revision metadata and fail ingestion on unexpected model drift.
- Add offline retrieval eval suite per content category and gate deploys on quality thresholds.

---

## D3. Data Model and State (Weight: 0.10)
**Score: 7/10**

**Evidence:**
- TypedDict + custom reducers + Pydantic structured outputs is a good production baseline.
- JSON-safe sentinel design is practical (better than `object()` for Firestore round-trips).
- Import-time parity checks are disciplined.

**Gap:**
- 31 state fields is getting bloated; this invites accidental coupling and turn/session bleed.
- Sentinel-as-string can theoretically collide with user/content values (low probability, non-zero blast radius).
- Reducer semantics (`""` treated as no-op, hashability assumptions in `_append_unique`) can silently drop or reject edge-case data.

**Fix:**
- Split into explicit `TurnState` vs `SessionState` substructures.
- Replace magic sentinel string with explicit operation envelope (e.g., `{"__op":"unset"}`) in reducer payloads.
- Tighten reducer typing/contracts and add invariant tests for null/empty/non-hashable edge cases.
- Add runtime schema validation on state transitions in non-prod debug mode.

---

## D4. API Design (Weight: 0.10)
**Score: 8/10**

**Evidence:**
- Pure ASGI middleware (good call), not `BaseHTTPMiddleware`.
- Real per-client distributed rate limiting via Redis + Lua script is strong.
- `/live` vs `/health` semantics are correct.
- SSE handling includes heartbeats and pre-stream PII redaction.
- Graceful SIGTERM drain strategy is thought through against platform timeouts.

**Gap:**
- `app.py` 946 LOC and `middleware.py` 835 LOC are maintainability liabilities.
- No explicit evidence of SSE resume semantics (`Last-Event-ID`) or strict disconnect/backpressure correctness tests.
- Security header set is good but not obviously complete (CSP/Permissions-Policy not mentioned).

**Fix:**
- Break app into routers/subapps by domain (`chat`, `ops`, `webhooks`, `property`).
- Add SSE conformance tests: disconnect mid-stream, slow consumer, replay/resume behavior.
- Expand security headers policy set and validate with automated security tests.
- Add explicit endpoint auth matrix tests (public vs protected).

---

## D5. Testing Strategy (Weight: 0.10)
**Score: 5/10**

**Evidence:**
- 2750 tests and mock purge are impressive; AST verification of no `MagicMock/AsyncMock` is rigorous.
- Guardrails/sentiment/crisis tests without mocks is a real strength.
- Singleton cache clearing fixture helps test isolation.

**Gap:**
- Biggest red flag: `fail_under=90` with actual 79.63% means your quality gate is not actually gating. That's a credibility problem, not a cosmetic one.
- Autouse fixtures disable API key and semantic injection globally, so tests are biased away from production behavior by default.
- No shown property-based tests for reducer/routing/ingestion invariants.
- Skipped flaky timing test suggests nondeterminism not yet controlled.

**Fix:**
- Make coverage gate hard-fail in CI immediately (or lower threshold honestly, then ratchet).
- Add a production-parity test lane with API key enabled and semantic injection enabled.
- Add Hypothesis/property tests for state reducers, routing closure, and chunk ID stability.
- Triage and eliminate flakes with deterministic time controls and strict retry policy.

---

## D6. DevOps and Deployment (Weight: 0.10)
**Score: 8/10**

**Evidence:**
- `--require-hashes`, multi-stage build, pinned base image by SHA-256 digest, non-root runtime user, exec-form `CMD`, and a real `HEALTHCHECK`.
- `pip-audit` in CI/CD and `.dockerignore` hygiene show this isn't amateur hour.
- No curl/package bloat in runtime image; clean final stage.

**Gap:**
- Security hardening stops at "non-root." No evidence of read-only root FS, dropped Linux capabilities, seccomp/apparmor profile, or immutable UID/GID strategy.
- `HEALTHCHECK` spins a Python interpreter each probe (works, but inefficient/noisy).
- `WEB_CONCURRENCY=1` while also hardcoding `--workers 1` is config theater.
- No mention of signed images/provenance attestation (SLSA/cosign), so supply chain trust is still partial.

**Fix:**
- Add runtime hardening: read-only FS, `--cap-drop=ALL`, tmpfs for writable paths, explicit UID:GID.
- Replace Python healthcheck with lightweight binary/http probe or app-native health CLI.
- Remove dead env vars or wire them into startup logic.
- Add image signing + provenance attestation in CI.

---

## D7. Guardrails and Safety (Weight: 0.10)
**Score: 7/10**

**Evidence:**
- 214 patterns across six safety/compliance categories is substantial coverage.
- Normalization pipeline is strong and layered (iterative URL decode, HTML unescape, Unicode normalization, control-char stripping, confusable mapping).
- Pre-LLM deterministic filtering + ordered compliance gate is the right architecture.
- Fail-closed classifier behavior is safety-first and explicit.

**Gap:**
- `re2 -> fallback to re` quietly reintroduces catastrophic backtracking risk unless every pattern is proven safe in stdlib `re`.
- Big regex count without precision/recall metrics is just "large," not "effective." You showed scale, not measured quality.
- Fail-closed on LLM outage is compliance-safe but operationally fragile (easy to induce broad denial of service).
- Domain whitelist exceptions ("act as a guide") are exactly where prompt-injection bypasses breed.

**Fix:**
- Enforce RE2-compatibility in CI and fail builds on non-compatible patterns; eliminate unsafe fallback for untrusted input.
- Add offline guardrail eval set with per-category FPR/FNR and drift tracking.
- Introduce degraded-safe mode (minimal deterministic allowlist) instead of total blackout on classifier outages.
- Red-team whitelist paths with adversarial suites and require explicit policy approvals.

---

## D8. Scalability and Production (Weight: 0.15)
**Score: 8/10**

**Evidence:**
- Solid primitives: async-safe circuit breaker, Redis-backed cross-instance state, semaphore backpressure, Redis Lua sliding-window rate limiting.
- TTL jitter on singleton caches is exactly the anti-stampede detail most teams forget.
- Graceful SIGTERM flow with SSE drain and explicit timing budget shows production realism.
- Raw ASGI middleware choice for SSE is technically correct; latency percentiles exposed.

**Gap:**
- Redis fallback to local-only breaker mode can create split-brain behavior during partial outages.
- `asyncio.Semaphore(20)` is static; no adaptive concurrency based on latency, token usage, or downstream model health.
- `threading.Lock` in an async service is a footgun even if "intentional"; one bad call path and you block the loop.
- No evidence of queue limits/load shedding policy at saturation beyond semaphore blocking.

**Fix:**
- Add breaker reconciliation strategy after Redis recovery and telemetry for cross-instance state divergence.
- Move to adaptive concurrency control (AIMD/gradient) with per-model budgets.
- Replace `threading.Lock` with `asyncio.Lock` or isolate lock usage off loop-critical path.
- Add explicit overload behavior: bounded queue + fast 429/503 with retry hints.

---

## D9. Documentation and ADRs (Weight: 0.05)
**Score: 6/10**

**Evidence:**
- 28 ADRs with status/date lifecycle is better than most production teams.
- Doc parity tests for architecture node/pattern/ADR counts show strong governance intent.
- Corrected stale architecture claims (12/204 to 13/214), showing active maintenance.

**Gap:**
- "ARCHITECTURE.md may not exist at expected path" is a basic repo integrity failure. You can't claim doc rigor if the canonical doc is intermittently missing.
- Count-based tests are shallow; they verify numbers, not truth.
- No evidence of ADR ownership, review cadence, or supersession enforcement in CI.

**Fix:**
- Make ARCHITECTURE doc path canonical and CI-fail if absent.
- Add semantic doc tests (e.g., graph node names/edges must match runtime graph introspection).
- Require ADR links to code, owner, review date, and supersedes/superseded-by consistency checks.

---

## D10. Domain Intelligence (Weight: 0.10)
**Score: 7/10**

**Evidence:**
- Multi-property config architecture is real: per-casino profiles, hot-reload chain, typed sections, cache + lock to prevent herd effects.
- 19 feature flags with import-time validation is good config discipline.
- Incentive engine + casino tools indicate meaningful domain workflow modeling, not generic chatbot fluff.
- State-specific regulatory dimensions (age, disclosure, quiet hours, RG helplines) are correctly domain-oriented.

**Gap:**
- Regulatory coverage appears shallow breadth-wise (five states named) and unclear depth-wise (no evidence of statute-level traceability/versioning).
- 5-minute TTL can be too slow for urgent compliance rule changes.
- Incentive autonomy thresholds look static; no risk-tiered controls, approval audit trail, or change-governance evidence.
- Feature-flag combinatorics can explode behavior surface without scenario-matrix testing.

**Fix:**
- Add regulation knowledge base with source citations, effective dates, and audit logs for every rule change.
- Introduce emergency config invalidation (push-based bust), not just TTL.
- Gate incentive decisions with risk scoring + immutable decision logs.
- Add automated combinatorial tests for critical flag interactions per casino profile.

---

## Summary

| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| D1. Graph Architecture | 6 | 0.20 | 1.20 |
| D2. RAG Pipeline | 7 | 0.10 | 0.70 |
| D3. Data Model & State | 7 | 0.10 | 0.70 |
| D4. API Design | 8 | 0.10 | 0.80 |
| D5. Testing Strategy | 5 | 0.10 | 0.50 |
| D6. DevOps & Deployment | 8 | 0.10 | 0.80 |
| D7. Guardrails & Safety | 7 | 0.10 | 0.70 |
| D8. Scalability & Production | 8 | 0.15 | 1.20 |
| D9. Documentation & ADRs | 6 | 0.05 | 0.30 |
| D10. Domain Intelligence | 7 | 0.10 | 0.70 |
| **TOTAL** | | **1.00** | **7.60** |

**Weighted Code Architecture Score: 7.6 / 10**

### Top 5 Critical Fixes (Priority Order)
1. **D5 Testing** — Fix coverage gate (fail_under=90 vs actual 79.63%) and add production-parity test lane
2. **D1 Graph Architecture** — Break monolithic nodes.py (1702 LOC) and _base.py (1578 LOC) into focused modules
3. **D7 Guardrails** — Enforce RE2-only in CI; eliminate unsafe stdlib re fallback for untrusted input
4. **D9 Documentation** — Fix ARCHITECTURE.md path integrity; add semantic doc tests
5. **D8 Scalability** — Add adaptive concurrency control and explicit overload/load-shedding policy

### Verdict
Genuinely production-capable infrastructure with strong primitives (circuit breaker, RRF, ASGI middleware, graceful shutdown). The main liabilities are monolithic file sizes undermining SRP, a non-functional coverage gate, and missing retrieval/guardrail quality metrics. Architecture is 80% of the way to production; the remaining 20% is discipline enforcement.
