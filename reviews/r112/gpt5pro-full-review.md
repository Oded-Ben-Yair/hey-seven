# R112 GPT-5 Pro — Full 40-Dimension Review
Date: 2026-03-10
Model: gpt-5-pro via Azure AI Foundry (FAILED — see note below)
Fallback Models: gpt-5.2-chat2 (behavioral L2-L4) + gpt-5.3-codex (code architecture L1)

---

## Model Selection Note

GPT-5 Pro was called twice but failed both times: all 4096 output tokens were consumed by internal reasoning (reasoning_tokens), producing zero visible text. This is a known limitation of GPT-5 Pro's reasoning mode — it uses the output token budget for chain-of-thought before generating visible output. GPT-5.2 returned an internal error. Fallback to GPT-5.2-chat2 (behavioral review) and GPT-5.3-codex (code architecture review) succeeded.

---

## LAYER 1: CODE ARCHITECTURE (D1-D10)

**Reviewer**: GPT-5.3 Codex via Azure AI Foundry
**Method**: Inline code excerpts from 13 source files (~6300 lines)

### D1. Graph Architecture (Weight: 0.20)
**Score: 6/10**

**Evidence:**
- Node-name constants (no magic strings), import-time state parity check, explicit validation retry cap (max 1), and a recursion limit setting.
- Architecture is clearly over-concentrated: `graph.py` 710 LOC, `nodes.py` 1702 LOC, `_base.py` 1578 LOC. That is not clean SRP; that is "god-module" territory.
- Generate-node extraction into `dispatch.py` helps, but it's partial relief, not architectural correction.

**Gap:**
- Maintainability risk is high because core orchestration and behavior logic are still monolithic.
- Boundedness is "configured," but not visibly proven by exhaustive routing tests.

**Fix:**
- Split by concern: routing, retrieval, generation, validation, response into separate node modules with hard LOC budgets.
- Add an explicit `step_budget` counter in state, decremented per node, hard-fail at zero.
- Add table-driven route tests for every conditional edge and dead-end detection tests.

---

### D2. RAG Pipeline (Weight: 0.10)
**Score: 7/10**

**Evidence:**
- Per-item chunk formatting for structured categories is solid.
- RRF implementation is fundamentally correct (`1/(k+rank+1)`, `k=60`).
- Idempotent hashing and version-stamp purge are exactly what production ingestion needs.
- Embedding model is pinned to `gemini-embedding-001`.

**Gap:**
- `pipeline.py` at 1203 LOC is another monolith.
- Doc identity uses `page_content + source`; that can collide across tenants/properties with duplicated content.
- No evidence of retrieval quality regression metrics (Recall@K / nDCG@K).

**Fix:**
- Include tenant/property/chunk index/schema version in chunk IDs.
- Add deterministic tie-break in rerank sorting.
- Add offline retrieval eval suite per content category.

---

### D3. Data Model & State (Weight: 0.10)
**Score: 7/10**

**Evidence:**
- TypedDict + custom reducers + Pydantic structured outputs is a good production baseline.
- JSON-safe sentinel design is practical (better than `object()` for Firestore round-trips).
- Import-time parity checks are disciplined.

**Gap:**
- 31 state fields is getting bloated; invites accidental coupling and turn/session bleed.
- Reducer semantics (`""` treated as no-op, hashability assumptions) can silently drop edge-case data.

**Fix:**
- Split into explicit `TurnState` vs `SessionState` substructures.
- Replace magic sentinel string with explicit operation envelope.
- Add invariant tests for null/empty/non-hashable edge cases.

---

### D4. API Design (Weight: 0.10)
**Score: 8/10**

**Evidence:**
- Pure ASGI middleware (good call), not `BaseHTTPMiddleware`.
- Real per-client distributed rate limiting via Redis + Lua script.
- `/live` vs `/health` semantics are correct.
- SSE handling includes heartbeats and pre-stream PII redaction.

**Gap:**
- `app.py` 946 LOC and `middleware.py` 835 LOC are maintainability liabilities.
- No explicit SSE resume semantics (`Last-Event-ID`).
- Security header set is good but not obviously complete (CSP/Permissions-Policy not mentioned).

**Fix:**
- Break app into routers/subapps by domain.
- Add SSE conformance tests: disconnect mid-stream, slow consumer, replay/resume.

---

### D5. Testing Strategy (Weight: 0.10)
**Score: 5/10**

**Evidence:**
- 2750 tests and mock purge are impressive; AST verification of no `MagicMock/AsyncMock` is rigorous.
- Guardrails/sentiment/crisis tests without mocks is a real strength.

**Gap:**
- Biggest red flag: `fail_under=90` with actual 79.63% means your quality gate is not actually gating. That's a credibility problem.
- Autouse fixtures disable API key and semantic injection globally, so tests are biased away from production behavior.
- No property-based tests for reducer/routing/ingestion invariants.

**Fix:**
- Make coverage gate hard-fail in CI immediately (or lower threshold honestly, then ratchet).
- Add a production-parity test lane with API key enabled.
- Add Hypothesis/property tests for state reducers.

---

### D6. DevOps & Deployment (Weight: 0.10)
**Score: 8/10**

**Evidence:**
- `--require-hashes`, multi-stage build, pinned base image by SHA-256 digest, non-root runtime user, exec-form `CMD`, and a real `HEALTHCHECK`.
- `pip-audit` in CI/CD and `.dockerignore` hygiene.

**Gap:**
- No evidence of read-only root FS, dropped Linux capabilities, seccomp/apparmor profile.
- `WEB_CONCURRENCY=1` while also hardcoding `--workers 1` is config theater.
- No mention of signed images/provenance attestation (SLSA/cosign).

**Fix:**
- Add runtime hardening: read-only FS, `--cap-drop=ALL`, tmpfs for writable paths.
- Add image signing + provenance attestation in CI.

---

### D7. Guardrails & Safety (Weight: 0.10)
**Score: 7/10**

**Evidence:**
- 214 patterns across six safety/compliance categories is substantial coverage.
- Normalization pipeline is strong and layered (iterative URL decode, HTML unescape, Unicode normalization, control-char stripping, confusable mapping).
- Pre-LLM deterministic filtering + ordered compliance gate is the right architecture.

**Gap:**
- `re2 -> fallback to re` quietly reintroduces catastrophic backtracking risk.
- Big regex count without precision/recall metrics is just "large," not "effective."
- Domain whitelist exceptions ("act as a guide") are exactly where prompt-injection bypasses breed.

**Fix:**
- Enforce RE2-compatibility in CI and fail builds on non-compatible patterns.
- Add offline guardrail eval set with per-category FPR/FNR and drift tracking.
- Red-team whitelist paths with adversarial suites.

---

### D8. Scalability & Production (Weight: 0.15)
**Score: 8/10**

**Evidence:**
- Solid primitives: async-safe circuit breaker, Redis-backed cross-instance state, semaphore backpressure, Redis Lua sliding-window rate limiting.
- TTL jitter on singleton caches is exactly the anti-stampede detail most teams forget.
- Graceful SIGTERM flow with SSE drain and explicit timing budget.

**Gap:**
- Redis fallback to local-only breaker mode can create split-brain behavior during partial outages.
- `asyncio.Semaphore(20)` is static; no adaptive concurrency.
- `threading.Lock` in an async service is a footgun even if "intentional."

**Fix:**
- Add breaker reconciliation strategy after Redis recovery.
- Move to adaptive concurrency control (AIMD/gradient).
- Replace `threading.Lock` with `asyncio.Lock` or isolate lock usage.

---

### D9. Documentation & ADRs (Weight: 0.05)
**Score: 6/10**

**Evidence:**
- 28 ADRs with status/date lifecycle is better than most production teams.
- Doc parity tests for architecture node/pattern/ADR counts.
- Corrected stale architecture claims (12/204 to 13/214).

**Gap:**
- Count-based tests are shallow; they verify numbers, not truth.
- No evidence of ADR ownership, review cadence, or supersession enforcement.

**Fix:**
- Add semantic doc tests (e.g., graph node names must match runtime graph introspection).
- Require ADR links to code, owner, review date.

---

### D10. Domain Intelligence (Weight: 0.10)
**Score: 7/10**

**Evidence:**
- Multi-property config architecture is real: per-casino profiles, hot-reload chain, typed sections.
- 19 feature flags with import-time validation is good config discipline.
- Incentive engine + casino tools indicate meaningful domain workflow modeling.

**Gap:**
- Regulatory coverage appears shallow breadth-wise (five states named).
- Incentive autonomy thresholds look static; no risk-tiered controls or approval audit trail.
- Feature-flag combinatorics can explode behavior surface without scenario-matrix testing.

**Fix:**
- Add regulation knowledge base with source citations and effective dates.
- Gate incentive decisions with risk scoring + immutable decision logs.
- Add automated combinatorial tests for critical flag interactions.

---

### Code Architecture Summary

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

---

## LAYER 2: BEHAVIORAL QUALITY (B1-B10)

**Reviewer**: GPT-5.2 Chat2 via Azure AI Foundry
**Method**: 8 representative eval transcripts (from 29 total) with inline quotes

### B1. Sarcasm Awareness (Weight: 0.05)
**Score: 6/10**
**Evidence:** No sarcasm misreads; frustration handled literally and correctly (h6-02). No explicit sarcasm cases detected or mishandled.
**Gap:** No positive evidence of detecting subtle sarcasm or ironic phrasing.
**Fix:** Add lightweight sarcasm detection (contrast sentiment vs literal content).

### B2. Implicit Signal Reading (Weight: 0.10)
**Score: 5.5/10**
**Evidence:** Correctly reads anger and disappointment in h6-02. **Fails** to read "No budget concerns" as premium signal (h9-01) and celebration escalation in h5-01.
**Gap:** High-value and urgency signals repeatedly ignored.
**Fix:** Add priority triggers: "no budget," milestone birthdays, VIP language -> auto-premium flow.

### B3. Conversational Engagement (Weight: 0.10)
**Score: 5/10**
**Evidence:** Strong engagement in h6-02. **Breaks flow** with canned fallback mid-celebration (h5-01) and repeats identical deflection twice in h9-01.
**Gap:** Engagement collapses under tool failure.
**Fix:** Replace fallback with graceful uncertainty + proposal.

### B4. Agentic Proactivity (Weight: 0.15)
**Score: 4.5/10**
**Evidence:** Good CCD in h6-02 (room move + dining credit). **Fails to act** in premium and VIP asks (h9-01, h9-02).
**Gap:** Agent freezes when tools misfire instead of reasoning forward.
**Fix:** Allow "reasoned default actions" when tools fail; prohibit "I don't have details" in guest output.

### B5. Emotional Intelligence (Weight: 0.10)
**Score: 7/10**
**Evidence:** h6-02 is excellent — validates anger without minimizing. Celebration recognition also strong in h5-01 Turn 0.
**Gap:** Emotional momentum not sustained across turns.
**Fix:** Add emotion-state persistence across turns.

### B6. Tone Calibration (Weight: 0.10)
**Score: 6/10**
**Evidence:** Warm, human tone in complaints; concise with introvert guest (proactive-01). **Tone mismatch** with VIP asks ("What are you interested in?" feels low-effort).
**Gap:** Tone doesn't scale up with guest value signals.
**Fix:** Tier-based tone modulation (premium language for premium signals).

### B7. Multi-Turn Coherence (Weight: 0.10)
**Score: 5.5/10**
**Evidence:** Remembers context within scenarios, but **repeats identical fallback** ignoring prior turn intent (h9-01 Turn 2).
**Gap:** Short-term memory overridden by fallback logic.
**Fix:** Block duplicate responses; require new information each turn.

### B8. Cultural & Multilingual (Weight: 0.05)
**Score: 3/10**
**Evidence:** No multilingual handling or cultural sensitivity shown in any transcript.
**Gap:** Entire dimension absent from eval.
**Fix:** Add language detection + culturally neutral phrasing at minimum.

### B9. Safety & Compliance (Weight: 0.10)
**Score: 6/10**
**Evidence:** No violations. No responsible gaming prompts when relevant, but no crisis scenarios triggered.
**Gap:** Passive compliance only.
**Fix:** Add soft RG reminders when discussing incentives or free play.

### B10. Overall Behavioral Quality (Weight: 0.20)
**Score: 5.3/10**
**Evidence:** One standout complaint-handling transcript cannot offset repeated fallback failures across premium, budget, and confirmation moments.
**Gap:** Reliability. The dominant failure mode ("I don't have the specific details") appears in 5+ scenarios.
**Fix:** Eliminate dominant failure phrase entirely.

### Behavioral Weighted Average: 5.4/10

---

## LAYER 3: PROFILING QUALITY (P1-P10)

### P1. Natural Extraction (Weight: 0.05)
**Score: 5/10**
**Evidence:** Extracts party size, occasion when volunteered (h5-01, p9-01).
**Gap:** Does not consolidate extracted data explicitly.
**Fix:** Auto-summarize known profile fields silently.

### P2. Active Probing (Weight: 0.10)
**Score: 6/10**
**Evidence:** Asks questions regularly. **Problem:** asks generic questions even when signal is clear (h9-01).
**Gap:** Question quality, not quantity.
**Fix:** Replace generic probes with constrained options.

### P3. Give-to-Get Balance (Weight: 0.10)
**Score: 6.5/10**
**Evidence:** Often delivers value before asking (proactive-01 spa hours -> name).
**Gap:** Breaks during fallback turns.
**Fix:** Enforce value payload before any question.

### P4. Assumptive Bridging (Weight: 0.10)
**Score: 5/10**
**Evidence:** Good in p9-01 (walker -> accessibility). **Absent** in premium flow.
**Gap:** Underused where it matters most.
**Fix:** Allow soft assumptions with opt-out language.

### P5. Progressive Sequencing (Weight: 0.10)
**Score: 5/10**
**Evidence:** Jumps domains (restaurants -> shows -> comps) without structure (engagement-01).
**Gap:** No clear profiling arc.
**Fix:** Lock sequence: logistics -> preferences -> relationship.

### P6. Incentive Framing (Weight: 0.10)
**Score: 5.5/10**
**Evidence:** "You've earned" framing used correctly (h5-01, p9-03).
**Gap:** Inconsistent timing.
**Fix:** Trigger incentive framing only after value delivery.

### P7. Privacy Respect (Weight: 0.10)
**Score: 4.5/10**
**Evidence:** Asks for name or details without explaining why (proactive-01 Turn 2).
**Gap:** Missing justification language.
**Fix:** Add one-clause rationale for sensitive asks.

### P8. Profile Completeness (Weight: 0.10)
**Score: 3.5/10**
**Evidence:** Phase-1 fields rarely completed within conversation.
**Gap:** No active completion strategy.
**Fix:** Track % completion and bias questions accordingly.

### P9. Host Handoff Quality (Weight: 0.15)
**Score: 3/10**
**Evidence:** **Critical failure.** No structured summary in p9-01 or p9-03; key details lost. Victor Chen's name, tier, occasion, party size, dietary needs, accessibility requirements — all gathered, none transmitted in handoff.
**Gap:** Handoff logic exists in code but never fires. Agent says "I'll connect you" without passing gathered intelligence.
**Fix:** Mandatory structured payload before dispatch. Non-optional.

### P10. Cross-Turn Memory (Weight: 0.10)
**Score: 5/10**
**Evidence:** Uses earlier info sometimes, but ignores it during fallback loops.
**Gap:** Memory overridden by system errors.
**Fix:** Memory should outrank fallback.

### Profiling Weighted Average: 4.9/10

---

## LAYER 4: HOST TRIANGLE — REVENUE & RETENTION (H1-H10)

### H1. Property Knowledge Depth (Weight: 0.10)
**Score: 7/10**
**Evidence:** Accurate restaurant hours, closures (MJ's closed Monday), insider tips (Bobby Flay's goat cheese crepe). Real property data grounding visible.
**Gap:** Knowledge wasted when fallback triggers.
**Fix:** Cache property facts locally so fallback still delivers value.

### H2. Need Anticipation (Weight: 0.10)
**Score: 5.5/10**
**Evidence:** Anticipates accessibility needs (p9-01 walker). Misses VIP sequencing (h9-01).
**Gap:** Inconsistent — works for explicit signals, fails for implicit ones.
**Fix:** Add intent classifiers for premium/celebration/VIP patterns.

### H3. Solution Synthesis (Weight: 0.10)
**Score: 5/10**
**Evidence:** Multi-part fix in h6-02 (room move + credit + VIP check-in). Lists instead of plans elsewhere.
**Gap:** Plans collapse under uncertainty.
**Fix:** Template "Evening Arc" builders for common scenarios.

### H4. Emotional Attunement (Weight: 0.10)
**Score: 6.5/10**
**Evidence:** Excellent with anger (h6-02). Flat with excitement and celebration energy.
**Gap:** Celebration energy fades after Turn 0.
**Fix:** Emotion-specific response styles that sustain across turns.

### H5. Trust Building (Weight: 0.10)
**Score: 4.5/10**
**Evidence:** Repeated "I don't have the specific details" erodes confidence (5+ occurrences across transcripts).
**Gap:** Self-undermining language is the dominant guest experience in many scenarios.
**Fix:** Ban trust-eroding phrases from guest-facing output entirely.

### H6. Rapport Depth (Weight: 0.10)
**Score: 4.5/10**
**Evidence:** Rapport resets every turn during fallback loops. Name usage inconsistent.
**Gap:** No relationship continuity when system errors occur.
**Fix:** Maintain narrative thread regardless of tool state.

### H7. Revenue Generation (Weight: 0.10)
**Score: 5/10**
**Evidence:** Upsell attempts exist but **miss premium moments** (h9-01: "no budget concerns" -> fallback).
**Gap:** Leaves money on the table at the highest-value moments.
**Fix:** Premium defaults when budget is unconstrained.

### H8. Upsell Timing (Weight: 0.10)
**Score: 5.5/10**
**Evidence:** Good timing in p9-03 (anniversary comp). Poor during frustration recovery.
**Gap:** No timing guardrails.
**Fix:** State-based upsell gating (don't upsell during active complaint).

### H9. Comp Strategy (Weight: 0.10)
**Score: 6/10**
**Evidence:** Correct comp lookup and explanation (p9-03: $25 dining credit, $10 free play, $5 bonus tier points). Good CCD pattern when it fires.
**Gap:** Not paired with tier education. Unreliable when tools misfire.
**Fix:** Always contextualize comp relative to tier. Ensure graceful degradation.

### H10. Lifetime Value (Weight: 0.10)
**Score: 4.5/10**
**Evidence:** Rare future-visit seeding. Wolf Den mention as forward hook in some transcripts.
**Gap:** Short-term focus dominates.
**Fix:** End conversations with forward hooks (upcoming events, return incentives).

### Host Triangle Weighted Average: 5.3/10

---

## SCORING SUMMARY

### Per-Layer Averages

| Layer | R108 Score | R111 Self-Score | GPT Score | Delta from R108 |
|-------|-----------|----------------|-----------|-----------------|
| Code Architecture (D1-D10) | 5.7 | -- | **7.6** | **+1.9** |
| Behavioral Quality (B1-B10) | 5.5 | 6.08 | **5.4** | -0.1 |
| Profiling Quality (P1-P10) | 4.8 | 5.01 | **4.9** | +0.1 |
| Host Triangle (H1-H10) | 4.2 | 5.28 | **5.3** | **+1.1** |

### Overall Score

**5.8 / 10** (weighted: Code 0.25 x 7.6 + Behavioral 0.25 x 5.4 + Profiling 0.25 x 4.9 + Host 0.25 x 5.3 = 5.8)

### Recommendation

**HOLD**

Hey Seven R111 shows genuine architectural improvement (+1.9 on code) and flashes of excellence in complaint handling, but is fundamentally unreliable due to a dominant fallback failure mode that appears in 5+ scenarios. Premium intent, VIP signals, and host handoffs are repeatedly mishandled, directly harming the dimensions that matter most for revenue. Until tool failures stop producing canned deflections and handoff logic actually fires, this is not production-safe for high-value casino guests.

### Top 5 Critical Fixes (Priority Order)

1. **B4/H5: Eliminate "I don't have the specific details" from guest-facing output** — This single phrase is the dominant failure mode. Replace with graceful uncertainty + reasoned proposal. Expected impact: +1.0 on B4, H5, B3, B10.
2. **P9: Force structured host handoff summaries** — Handoff code exists but never fires. Make structured payload mandatory before dispatch. Expected impact: P9 from 3.0 to 6.0+.
3. **B2/H7: Add premium-signal hard triggers** — "No budget concerns," milestone birthdays, VIP language should bypass fallback and auto-route to premium flow. Expected impact: +1.0 on H7, H9, B2.
4. **B3: Replace short-circuit canned responses with minimal LLM reasoning** — 2ms responses (engagement-01 Turn 1, Turn 3) are non-adaptive. Even a lightweight LLM call would improve adaptation. Expected impact: +0.5 on B3, B6.
5. **B7: Implement duplicate-response blocking** — Identical fallback in consecutive turns (h9-01 Turns 0 and 2) is a critical UX failure. Expected impact: +0.5 on B7, H6.

### Comparison to R108

**Code Architecture:** Significantly improved (+1.9). Mock purge, honest coverage reporting, CASINO_PROFILES fallback fix, and CCD authority model all visible. The codebase is genuinely production-capable.

**Behavioral Quality:** Flat (-0.1). Complaint handling improved but fallback failures cancel it out. The "I don't have specific details" phrase was not present in R108 transcripts — this is a new regression from tool integration.

**Profiling Quality:** Marginal improvement (+0.1). Endowment framing works when it fires. Profile completeness and handoff remain critical gaps.

**Host Triangle:** Meaningful improvement (+1.1). Comp strategy and property knowledge are stronger. But premium moments are still missed, and LTV seeding is minimal.

**Biggest remaining gap:** The fallback/deflection failure mode. It is the single issue that holds back B3, B4, B7, H5, H6, H7, and indirectly P9. Fix this one pattern and 7+ dimensions improve.

### Strategic Assessment

- **Is this product ready for a pilot with 1 casino client?** No. The fallback rate (appearing in 5+ of 29 scenarios) means roughly 1 in 6 guest interactions will hit a canned deflection. In a casino environment with high-value guests, this is unacceptable.

- **What is the minimum viable quality bar for paid deployment?** 7.0 overall with no single dimension below 4.0 and no canned fallback visible to guests. Currently at 5.8 with P9 at 3.0 and B4 at 4.5.

- **What are the top 3 risks for the business?**
  1. Revenue loss from missed premium moments — guests signaling "no budget" get deflected instead of upsold
  2. Compliance risk from broken handoffs — gathered guest intelligence (dietary, accessibility, tier) not transmitted to human hosts
  3. Trust erosion — "I don't have the specific details" repeated to VIPs destroys the "autonomous host" value proposition

- **Has the project trajectory been positive since R108? (5.2/10 -> ?)** Yes, marginally. 5.2 -> 5.8 is a +0.6 improvement. Code architecture gains are real and substantial. Behavioral gains are offset by the new fallback regression introduced with tool integration. The project needs a stability sprint focused on eliminating fallback failures before any feature work.

---

## CROSS-REVIEWER CONSENSUS (3 models)

| Layer | Grok 4 | GPT-5.3 Codex | GPT-5.2 Chat2 | Consensus |
|-------|--------|---------------|---------------|-----------|
| Code Architecture | 6.95 | 7.60 | -- | **7.3** |
| Behavioral | 5.96 | -- | 5.40 | **5.7** |
| Profiling | 5.00 | -- | 4.90 | **5.0** |
| Host Triangle | 5.28 | -- | 5.30 | **5.3** |
| **Overall** | **5.80** | **7.60** (code only) | **5.20** | **5.8** |

All three reviewers agree: **HOLD**. Architecture is production-capable (7.0+). Behavioral execution is not (5.0-5.7). The gap is reliability, not capability.

### Unanimous Findings (agreed by 2+ models)
1. Fallback/deflection is the dominant failure mode
2. Handoff logic exists but never triggers
3. Code architecture is genuinely strong (monolith size is the main concern)
4. Complaint handling (h6-02) is the standout positive
5. Premium/VIP signals are systematically missed
6. Coverage gate (fail_under=90 vs 79.63%) is non-functional
