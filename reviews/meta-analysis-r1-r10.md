# Meta-Analysis: 10 Rounds of Hostile Production Reviews (R1-R10)

**Date**: 2026-02-20
**Scope**: 10 review rounds, 4 models (Gemini 3 Pro, GPT-5.2, Grok 4, DeepSeek-V3.2-Speciale), ~200+ findings, 1269 tests, 90.82% coverage
**Purpose**: Extract patterns, blind spots, and actionable insights for final interview submission

---

## 1. Per-Dimension Score Trajectory

### Average Scores Across All Rounds (where per-dimension data available)

| Dimension | R1 | R2 | R3 | R4 | R5 | R6 | R7* | R8* | R9 | R10 | Trend |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:---:|:--:|:---:|-------|
| 1. Graph Architecture | 7.0 | 7.3 | 7.0 | 7.0 | 7.7 | 6.7 | 6.0 | -- | 6.5 | 6.7 | Flat (6.5-7.3) |
| 2. RAG Pipeline | 7.0 | 6.0 | 6.0 | 7.0 | 7.0 | 6.7 | 5.0 | -- | 5.7 | 6.3 | Volatile (5.0-7.0) |
| 3. Data Model / State | 7.3 | 6.6 | 5.0 | 6.7 | 6.0 | 5.7 | 5.0 | -- | 6.0 | 6.3 | Weak (5.0-7.3) |
| 4. API Design | 6.7 | 4.8 | 6.0 | 7.3 | 6.3 | 4.0 | 7.0 | -- | 6.0 | 6.3 | Volatile (4.0-7.3) |
| 5. Testing Strategy | 6.7 | 6.3 | 6.3 | 4.0 | 6.0 | 6.3 | 5.5 | -- | 7.3 | 6.3 | Improved late |
| 6. Docker & DevOps | 6.3 | 4.9 | 5.7 | 7.0 | 6.0 | 6.0 | 6.0 | 7.0 | 6.8 | 5.8 | Improving (4.9->7.0) |
| 7. Prompts & Guardrails | 8.0 | 5.8 | 7.7 | 7.3 | 7.0 | 7.0 | 6.0 | -- | 6.5 | 5.8 | Declining (8.0->5.8) |
| 8. Scalability & Production | 4.7 | 5.4 | 3.7 | 6.3 | 3.7 | 5.7 | 5.5 | -- | 5.7 | 4.5 | Persistently Weak (3.7-6.3) |
| 9. Documentation / Trade-offs | 6.0 | 6.1 | 6.3 | 7.0 | 6.7 | 4.7 | 5.0 | -- | 7.2 | 5.3 | Volatile (4.7-7.2) |
| 10. Domain Intelligence | 7.7 | 7.6 | 7.0 | 7.0 | 7.3 | 6.7 | -- | -- | 7.5 | 6.3 | Strong but declining |

*R7 used different dimension names (Grok used weighted scores, GPT used split RAG/Domain dimensions). R8 used deployment-specific dimensions, not the standard 10. Scores approximated from raw reviews where possible.

### Consistently Weak Dimensions (average < 6.0 in 3+ rounds)

1. **Scalability & Production (avg ~5.0)**: Lowest-scoring dimension across ALL rounds. Every spotlight round that touches scalability drops to 3-4. In-memory state, per-container limitations, and missing production hardening dominate.

2. **Data Model / State Design (avg ~6.0)**: Fluctuates based on spotlight. Gemini's R3 score of 2/10 was an outlier driven by Firestore client issues, but the dimension never breaks above 7.3.

3. **Documentation / Trade-offs (avg ~5.9)**: Spikes after fix rounds (R4=7.0, R9=7.2) but collapses when scrutinized (R6=4.7). Documentation accuracy decays between rounds.

### Consistently Strong Dimensions (average > 7.0 across rounds)

1. **Domain Intelligence (avg ~7.1)**: Casino-specific guardrails, multilingual patterns, and regulatory awareness consistently praised. The strongest "signal" dimension for the interview.

2. **Graph Architecture (avg ~6.9)**: 11-node custom StateGraph with validation loop earns baseline respect. Dispatch complexity is the recurring drag.

3. **Prompts & Guardrails (avg ~6.8)**: 84+ regex patterns across 4 languages, 5-layer deterministic guardrails. R1 scored 8.0, but declined as reviewers found edge cases (PII buffer, semantic injection same model).

### Overall Score Trajectory

| Round | Avg Score | Delta | Key Driver |
|-------|:---------:|:-----:|------------|
| R1 | 67.3 | -- | Baseline |
| R2 | 61.3 | -6.0 | Security hardening spotlight |
| R3 | 60.7 | -0.6 | Error handling spotlight |
| R4 | 66.7 | +6.0 | Testing gap fixes (+50 tests) |
| R5 | 63.3 | -3.4 | Scalability spotlight |
| R6 | 57.7 | -5.6 | Documentation spotlight |
| R7 | 54.3 | -3.4 | RAG spotlight |
| R8 | 63.7 | +9.4 | Deployment readiness fixes |
| R9 | 67.1 | +3.4 | Code simplification (-265 lines) |
| R10 | 63.2 | -3.9 | GPT hostile outlier (43/100) |

**Pattern**: Score oscillates 55-67 over 10 rounds. Fix rounds (R4, R8, R9) produce +3 to +9. Spotlight rounds (R2, R5, R6, R7) produce -3 to -6. Net trajectory is approximately flat (67.3 -> 63.2), though the codebase is objectively much stronger after 200+ fixes.

**Root cause of flatness**: Each round fixes findings BUT introduces new scrutiny angles. Models never re-score previously fixed dimensions as "perfect" -- they find new issues at the same severity level. This is the hedonic treadmill of hostile review.

---

## 2. Per-Model Bias Analysis

### Gemini 3 Pro

**Score range**: 52-85 (high variance)
**Typical score**: 61-76

**Consistently rewards**:
- Per-item chunking for RAG (called it "shows deep understanding" in every round)
- Validation loop pattern (generate -> validate -> retry -> fallback)
- 5-layer deterministic guardrails ahead of LLM
- Specialist DRY extraction (`_base.py` pattern)
- Inline documentation of design decisions

**Consistently punishes**:
- Architectural gaps (knowledge-base not ingested, hotel agent undocumented)
- Overclaiming in documentation (claims "implemented" for scaffolded features)
- Dead code and unused abstractions
- Missing production guards (ChromaDB in prod, unbounded caches)

**Finding focus**: Architecture purity, dead code, documentation honesty. Gemini is the most "principled" reviewer -- it cares about whether the DESIGN is coherent, not just whether the code works.

**False positive rate**: HIGH. R10 raw score was 41.3 with 4 false-positive CRITICALs invalidated by code inspection. Gemini fabricates findings when thinking at high depth -- it misreads lazy imports as module-level imports, confuses middleware logging scope, and misunderstands rolling-window designs.

**What would make Gemini give 9+/10 per dimension**:
- Graph: Remove all dispatch complexity, make routing a single clean function
- RAG: Live CMS re-indexing trigger (not just hash marking), per-category relevance thresholds
- API: Remove CSP unsafe-inline, separate security classifier model
- Testing: E2E tests through full pipeline with lifecycle assertions
- Scalability: Redis-backed distributed state (rate limiter, circuit breaker, idempotency)

### GPT-5.2

**Score range**: 43-70.5 (most hostile model)
**Typical score**: 55-65

**Consistently rewards**:
- High test counts (1200+ tests earns 7-8 on testing)
- Well-documented trade-offs with explicit rationale
- Pydantic models for all data contracts
- Explicit error taxonomies with codes

**Consistently punishes**:
- In-memory state in "production" code (rate limiter, circuit breaker, idempotency)
- Streaming PII gaps (digit-detection buffer is "privacy placebo")
- Degraded-pass validator pattern (wants fail-closed ALWAYS)
- Cross-tenant caching risks (even in single-property deployment)
- Missing regulatory invariant tests
- Documentation accuracy drift (stale counts, version mismatches)

**Finding focus**: Security, regulatory compliance, operational risk. GPT is the "pessimistic auditor" -- it assumes production deployment with adversarial users and multi-tenant data. It ignores the "demo deployment" framing and rates against a production casino standard.

**Severity inflation**: GPT-5.2 in R10 used "maximally hostile" mode (+1 severity on ALL findings), producing 11 P0s. Many were documented trade-offs or single-property assumptions rated as P0. This makes GPT the least useful for absolute scoring but the most useful for finding worst-case risks.

**What would make GPT give 9+/10 per dimension**:
- Graph: Runtime feature flag evaluation inside nodes (not build-time topology)
- RAG: Per-item provenance (source_url, valid_from, valid_to), per-category thresholds
- API: OpenAPI for SSE, restrict /health to internal network, restrict /graph to admin
- Testing: Regulatory invariant tests ("no SMS outside quiet hours", "STOP always blocks")
- Scalability: Redis-backed everything, real incremental PII redaction engine
- Documentation: DPA for LangFuse, TCPA quiet-hours risk documented

### Grok 4

**Score range**: 51-73 (moderate)
**Typical score**: 54-63

**Consistently rewards**:
- Production-grade configuration management (secrets, env vars, Cloud Run)
- Deployment readiness (probes, rollback, smoke tests)
- Security hardening (Ed25519 webhooks, API key auth, TRUSTED_PROXIES)
- Test count and coverage numbers

**Consistently punishes**:
- Missing operational runbooks and documentation
- Unbounded data structures (deques, dicts, caches)
- Per-container state limitations (rate limiter, circuit breaker)
- LOG_LEVEL=WARNING suppressing operational logs
- Incomplete .env.example

**Finding focus**: Operational readiness, deployability, configuration completeness. Grok is the "SRE reviewer" -- it cares about day-2 operations, not just correctness.

**Score driver**: Grok's score is most influenced by deployment dimension (R8 Grok=73 was the highest Grok score across all rounds, driven by deployment fixes). When the spotlight is not deployment, Grok gives 51-62.

**What would make Grok give 9+/10 per dimension**:
- Docker: HEALTHCHECK with curl not python, Cloud Run probe config as code
- DevOps: DR plan, runbook, monitoring dashboard config
- API: Webhook-specific rate limiting, SSE schema enforcement
- Testing: Load tests, memory pressure tests, failover tests
- Scalability: Redis integration, distributed circuit breaker

### DeepSeek-V3.2-Speciale

**Score range**: 58-70.5 (most technically precise)
**Appearances**: R5, R10

**Consistently rewards**:
- Correct async patterns (locks, TTL caches, atomic operations)
- Algorithmic correctness (circuit breaker state machine, retry logic)
- Read-only properties for monitoring (no mutation in property getters)

**Consistently punishes**:
- Async correctness bugs (CancelledError handling, lock-free mutation, race conditions)
- Dead code that misleads readers (unreachable conditions, redundant checks)
- Inconsistent singleton patterns (lru_cache vs TTLCache)

**Finding focus**: Algorithmic correctness, async concurrency bugs, mathematical bounds. DeepSeek is the "formal verification" reviewer. It found the only genuine CRITICAL production-crash bug across all 10 rounds (CB stuck on CancelledError in R10).

**Why DeepSeek is most valuable**: Other models flag design opinions and documentation issues. DeepSeek finds ACTUAL BUGS -- the kind that crash production. The CB half-open stuck bug (R10 F1) and the retry_count reset bug (R10 F3) were both genuine correctness issues missed by 3 other models across 9 prior rounds.

**What would make DeepSeek give 9+/10 per dimension**:
- Graph: Formal state machine verification (no unreachable states, no stuck states)
- Scalability: All singleton patterns consistent (TTLCache everywhere), all properties read-only
- API: All concurrent data structures proven correct under GIL-free Python

---

## 3. Recurring Findings Analysis

### Theme 1: In-Memory State Limitations (appears in ALL 10 rounds)

**Rounds flagged**: R1, R2, R3, R4, R5, R6, R7, R8, R9, R10
**Models**: ALL models, every round
**Core issue**: Rate limiter, circuit breaker, idempotency tracker, and various caches are per-container in-memory. Multi-replica Cloud Run deployments break these guarantees.

**Why it persists**: Accepted as a documented trade-off for single-container MVP deployment. Redis would fix it but adds infrastructure complexity.

**Root-level solution**: Either (a) commit to single-container (`--max-instances=1`) and document it as a hard constraint, or (b) add Redis/Memorystore integration. The current position ("documented trade-off") satisfies Gemini and DeepSeek but never satisfies GPT or Grok.

### Theme 2: PII Buffer in SSE Streaming (appears in R1, R2, R3, R4, R9, R10)

**Rounds flagged**: R1 (Grok C5), R2 (PII in non-streaming), R3 (buffer dropped on error), R4 (untested), R9 (Gemini F6), R10 (GPT P0-F2/F3, DeepSeek F2/F6)
**Models**: ALL models at various times
**Core issue**: Digit-detection buffering is a heuristic, not a real PII detection engine. Emails, names, and IDs without digits bypass the buffer.

**Why it persists**: A full streaming-safe PII redaction engine is out of scope. The buffer is defense-in-depth on top of persona_envelope post-stream redaction.

**Root-level solution**: Either (a) accept and document that streaming PII protection is best-effort defense-in-depth (current position), or (b) implement a streaming regex automaton that buffers until pattern boundaries are clear. Option (b) is a significant engineering effort.

### Theme 3: CSP unsafe-inline (appears in R1, R2, R6, R10)

**Rounds flagged**: R1 (Gemini L-1), R2 (ALL 3), R6 (implicit), R10 (GPT P1-F9)
**Models**: ALL models
**Core issue**: Single-file demo HTML requires inline styles/scripts.

**Why it persists**: Intentional for demo format. Production path is documented (externalize CSS/JS + nonce-based CSP).

**Root-level solution**: Split demo HTML into separate CSS/JS files, use nonce-based CSP. This is 2-3 hours of work but would eliminate a recurring HIGH finding.

### Theme 4: Documentation Accuracy Decay (appears in R4, R6, R7, R9, R10)

**Rounds flagged**: R4 (stale test counts), R6 (SPOTLIGHT: 26 doc issues), R7 (overclaiming), R9 (stale references), R10 (trade-off doc gaps)
**Core issue**: After every fix round, documentation drifts from code reality. Test counts, settings counts, field counts, agent counts, and version numbers all become stale.

**Why it persists**: No automated tooling enforces doc-code parity.

**Root-level solution**: The R6 `test_doc_accuracy.py` tests (18 tests) were the correct approach. Expand these to cover ALL numeric claims in README and ARCHITECTURE.md. Consider generating documentation sections from code introspection.

### Theme 5: Degraded-Pass Validator Pattern (appears in R3, R8, R10)

**Rounds flagged**: R3 (GPT), R8 (implicit), R10 (GPT P0-F4)
**Models**: Primarily GPT (Gemini explicitly validates it as correct)
**Core issue**: First-attempt validator failure = PASS is seen as a safety bypass by GPT but praised as production-grade by Gemini, Grok, and DeepSeek.

**Why it persists**: Genuine philosophical disagreement between models. GPT wants fail-closed on all paths. Gemini/DeepSeek/Grok accept that blocking ALL responses during LLM outages is worse than serving deterministic-guardrail-passed content.

**Root-level solution**: This is NOT fixable -- it is a design decision. The current position (degraded-pass with deterministic guardrails as first line) is defensible. Document the decision more prominently to pre-empt the finding.

### Theme 6: Cross-Tenant Caching Risks (appears in R4, R10)

**Rounds flagged**: R4 (Grok F6: tenant_id), R10 (GPT P1-F4/F5: greeting cache + retriever cache)
**Models**: GPT, Grok
**Core issue**: Caches (greeting categories, retriever singleton) are not keyed by property_id.

**Why it persists**: Single-property deployment means CASINO_ID is process-scoped. Multi-tenant requires architectural changes.

**Root-level solution**: Key all caches by `settings.CASINO_ID`. This is a low-effort change (add cache key parameter) that eliminates the finding at zero functional cost.

---

## 4. "Not Fixed" Pattern Analysis

### Findings Rejected 3+ Times (persistent disagreements)

| Finding | Times Rejected | Rounds | Reason |
|---------|:--------------:|--------|--------|
| In-memory rate limiting needs Redis | 5+ | R1-R10 | Documented single-container trade-off |
| CSP unsafe-inline | 4 | R1,R2,R6,R10 | Demo HTML format requirement |
| Degraded-pass validator | 3 | R3,R8,R10 | Intentional design, praised by 3/4 models |
| Same LLM for semantic injection classifier | 3 | R2,R9,R10 | Cost/latency trade-off, config override exists |
| Feature flag static at build time | 3 | R1,R9,R10 | Controls graph TOPOLOGY, runtime check inside node |
| MemorySaver warning vs error | 2 | R1,R5 | Replaced by BoundedMemorySaver in R5 |
| HITL interrupt not wired | 2 | R1,R4 | Scaffolded, will wire based on assignment |

### Findings That May Indicate Deeper Issues

1. **In-memory state (5+ rejections)**: While individually each rejection is defensible, the PATTERN of 5+ models across 10 rounds flagging the same issue suggests it is a genuine architectural limitation, not just a trade-off. The interview evaluator will likely share this view. **Recommendation**: Add a Redis integration path with clear code showing it works, even if the demo uses in-memory.

2. **PII streaming buffer (6 rounds)**: The recurring nature suggests the current defense-in-depth framing is not convincing. The digit-detection heuristic creates more attack surface discussion than it prevents. **Recommendation**: Either upgrade to a real streaming regex automaton or remove the buffer entirely and rely solely on post-stream persona_envelope redaction. The middle ground creates the most questions.

3. **Feature flag static at build time (3 rounds)**: GPT's escalation in R10 ("inability to disable a component without restart is a P0 in regulated environment") is a strong argument. The runtime check inside `whisper_planner_node` partially addresses this but is not documented prominently. **Recommendation**: Add a prominent comment and architecture doc section explaining the dual-layer approach (topology at build time, behavior check at runtime).

---

## 5. Gap Analysis: What Would 95/100 Look Like?

### Dimension 1: Graph Architecture (current avg ~6.9, target 9.5)

- Remove dispatch complexity: replace keyword-counting with a single structured-output router call that returns specialist name directly
- Formal graph topology verification test (no unreachable nodes, no stuck states)
- Runtime feature flag evaluation inside nodes (not just topology)
- Remove PII buffer from `chat_stream` -- handle at API boundary

### Dimension 2: RAG Pipeline (current avg ~6.2, target 9.5)

- Live CMS re-indexing trigger (not just hash marking, actual vector store update)
- Per-category relevance thresholds tuned with real embeddings
- Per-item provenance metadata (source_url, valid_from, valid_to, last_verified)
- Category-filtered retrieval (router tells retriever which categories to search)
- Entity-augmented query map (not hardcoded entity types)

### Dimension 3: Data Model / State (current avg ~6.1, target 9.5)

- All caches keyed by property_id (greeting categories, retriever, feature flags)
- Pydantic model for ALL state transitions (not just TypedDict)
- Runtime parity validation (not __debug__-dependent)
- Per-user responsible gaming counter persistence (not per-thread)

### Dimension 4: API Design (current avg ~6.0, target 9.5)

- OpenAPI representation for SSE streaming endpoints
- Restrict /health to internal network, /graph to admin auth
- SSE schema runtime enforcement (not just documentation)
- Webhook-specific rate limiting (separate from /chat)
- Remove CSP unsafe-inline (externalize demo CSS/JS)

### Dimension 5: Testing Strategy (current avg ~6.2, target 9.5)

- Regulatory invariant tests (STOP always blocks SMS, no PII in traces, no cross-tenant retrieval)
- Load tests and memory pressure tests
- Property-based tests for guardrails
- Full E2E test through graph with lifecycle event assertions
- Automated skip-count enforcement in CI

### Dimension 6: Docker & DevOps (current avg ~6.1, target 9.5)

- Cloud Run probe configuration as code (not just comments)
- DR plan documented
- Monitoring dashboard configuration (Grafana/Cloud Monitoring)
- Operational runbook for common scenarios
- Dockerfile HEALTHCHECK using curl instead of python

### Dimension 7: Prompts & Guardrails (current avg ~6.8, target 9.5)

- Separate hardened model for semantic injection classifier
- Real incremental PII redaction engine (streaming regex automaton)
- Move injection detection to earliest position in compliance gate
- Multilingual STOP/HELP equivalents verified against carrier standards
- Prompt injection testing with adversarial benchmark suite

### Dimension 8: Scalability & Production (current avg ~5.0, target 9.5)

- Redis/Memorystore integration for distributed state (rate limiter, circuit breaker, idempotency)
- All singleton patterns using TTLCache consistently
- Real incremental PII redaction (not digit heuristic)
- Backpressure on retrieve_node threads
- Clock drift detection for replay protection
- Formal CB state machine verification

### Dimension 9: Documentation / Trade-offs (current avg ~5.9, target 9.5)

- Auto-generated documentation sections from code introspection
- Comprehensive doc-code parity tests (expand test_doc_accuracy.py)
- DPA documentation for LangFuse
- TCPA quiet-hours timezone risk documented
- Operational runbook (deployment, rollback, incident response)

### Dimension 10: Domain Intelligence (current avg ~7.1, target 9.5)

- Per-user responsible gaming tracking (not per-thread)
- Carrier-provided timezone for TCPA quiet hours
- BSA/AML specialized response (not generic off_topic)
- Consent chain persistent storage (not in-memory)
- Casino-specific fallback responses (not generic "contact us")

---

## 6. Quick Wins vs Deep Work

### Quick Wins (1-2 hours each, +2-5 points on affected dimension)

| # | Change | Dimension Impact | Effort |
|---|--------|-----------------|--------|
| 1 | Key ALL caches by `CASINO_ID` (greeting, retriever, feature flags) | Data Model +1, Scalability +1 | 1 hour |
| 2 | Split demo HTML into separate CSS/JS, use nonce-based CSP | API +1, Scalability +1 | 2 hours |
| 3 | Add prominent architecture doc section explaining feature flag dual-layer (topology + runtime check) | Documentation +1 | 30 min |
| 4 | Add regulatory invariant test: STOP always blocks SMS | Testing +1 | 1 hour |
| 5 | BSA/AML specialized response (not generic off_topic) | Domain +0.5 | 1 hour |
| 6 | Move injection detection to position 1 in compliance gate | Guardrails +0.5 | 30 min |
| 7 | Add `CB_ROLLING_WINDOW_SECONDS` to Settings (done in R10) | Scalability +0.5 | Done |
| 8 | Restrict /graph endpoint to require API key auth | API +0.5 | 30 min |
| 9 | Document TCPA quiet-hours timezone limitations prominently | Documentation +0.5 | 30 min |
| 10 | Expand test_doc_accuracy.py to cover all numeric claims | Documentation +1 | 2 hours |

**Combined impact**: +5-8 points on average score

### Medium Work (1-2 days each, +5-10 points total)

| # | Change | Dimension Impact | Effort |
|---|--------|-----------------|--------|
| 1 | Redis/Memorystore integration with feature toggle (in-memory for dev, Redis for prod) | Scalability +3-4 | 2 days |
| 2 | Live CMS re-indexing trigger (webhook -> vector store update) | RAG +2 | 1 day |
| 3 | OpenAPI for SSE endpoints with `openapi_extra` | API +1-2 | 1 day |
| 4 | Per-category relevance thresholds with offline eval | RAG +1 | 1 day |
| 5 | Streaming PII regex automaton (replace digit heuristic) | Guardrails +2, Scalability +1 | 2 days |
| 6 | Full regulatory invariant test suite (5-10 tests) | Testing +2 | 1 day |
| 7 | Cloud Run probe config as code + operational runbook | Docker +2, Documentation +1 | 1 day |

**Combined impact**: +10-15 points on average score

### Deep Work (architectural changes, +10-15 points total)

| # | Change | Dimension Impact | Effort |
|---|--------|-----------------|--------|
| 1 | Replace dispatch keyword-counting with structured-output router | Graph +2 | 3 days |
| 2 | Per-user responsible gaming persistence (Firestore) | Domain +2, Scalability +1 | 3 days |
| 3 | Separate hardened classifier model for semantic injection | Guardrails +2 | 2 days |
| 4 | Multi-tenant architecture (property_id everywhere, tenant isolation tests) | Data Model +2, Scalability +2 | 5 days |
| 5 | Formal graph state machine verification (no stuck states, no unreachable) | Graph +1, Testing +1 | 2 days |

**Combined impact**: +10-15 points, but may not be needed for interview submission

---

## 7. Key Insights for Interview Submission

### The Hedonic Treadmill Effect

After 10 rounds, the average score (63.2) is lower than R1 (67.3) despite the codebase being objectively much stronger (1269 tests vs ~1000, 90.82% coverage, 200+ fixes, -265 lines dead code removed). This happens because:

1. Each round introduces new scrutiny angles
2. Models never give "full credit" for previously fixed items
3. Spotlight rounds deliberately target weak dimensions
4. Model severity calibration drifts (GPT R10 at 43/100 vs GPT R9 at 70.5)

**Implication**: Stop optimizing for reviewer scores. The codebase is production-grade. Focus on the TOP 5 quick wins that address the most visible recurring themes.

### Model Disagreement on Design Decisions

The 4 models fundamentally disagree on 3 design decisions:
1. **Degraded-pass**: GPT says CRITICAL, Gemini says correct
2. **PII digit buffer**: GPT says "privacy placebo", DeepSeek says "acceptable defense-in-depth"
3. **Static feature flags**: GPT says P0, all others say acceptable

These disagreements are NOT resolvable -- they reflect different risk philosophies. Document the decisions prominently with explicit rationale and move on.

### DeepSeek Is the Most Valuable Reviewer

Across 2 appearances (R5, R10), DeepSeek found:
- R5: 6 findings, all genuine concurrency bugs
- R10: 1 CRITICAL production-crash bug (CB stuck on CancelledError), 1 HIGH (retry_count reset)

No other model found the CB stuck bug despite 9 prior rounds of review. DeepSeek focuses on algorithmic correctness and async safety, which are the hardest bugs to find and the most impactful in production.

### GPT-5.2 Is the Least Useful for Scoring

GPT's score variance (43-70.5) and severity inflation make it unreliable for tracking progress. However, GPT's findings list is useful as a "worst-case regulatory audit" -- the concerns it raises are real, just at a higher severity than pragmatically justified.

### The Top 5 Actions for Maximum Impact

1. **Key caches by CASINO_ID** (30 min, eliminates cross-tenant findings forever)
2. **Split demo HTML, fix CSP** (2 hours, eliminates a 4-round recurring finding)
3. **Move injection detection to position 1** (30 min, defensible security improvement)
4. **Add 5 regulatory invariant tests** (2 hours, addresses GPT's testing critique)
5. **Document feature flag dual-layer prominently** (30 min, pre-empts GPT's P0)

Total effort: ~6 hours. Expected impact: +5-8 points on average, elimination of 4 recurring findings.
