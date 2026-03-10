# Hey Seven R111 — 40-Dimension Product Review Prompt

**For**: GPT-5.4 Pro Deep Research (or equivalent frontier model)
**Mode**: Hostile external review — score honestly, find real problems, no flattery
**Repository**: https://github.com/Oded-Ben-Yair/hey-seven
**Date**: 2026-03-10
**Previous review**: R108 — scored **5.2/10 HOLD** (see "What Changed Since R108" below)

---

## Instructions

You are reviewing **Hey Seven**, a production AI casino host agent. This is NOT just a code review — it is a **complete product review** covering code quality, behavioral quality, business logic, regulatory compliance, deployment readiness, and strategic positioning.

**Ground rules:**
1. Score each dimension 1-10 with specific evidence (file:line or transcript quote)
2. Do NOT inflate scores — a 7 means "good, production-ready", 9 means "exceptional, best-in-class"
3. For each dimension, provide: Score, Evidence (what you saw), Gap (what's missing), Fix (specific action)
4. Read the FULL codebase, not just highlights. Clone the repo and explore.
5. After scoring all 40 dimensions, provide an overall INVEST / HOLD / PASS recommendation with 3-sentence justification
6. **Compare to your R108 scores** — for each dimension, state whether it improved, regressed, or stayed the same

---

## WHAT CHANGED SINCE YOUR R108 REVIEW (5.2/10 HOLD)

### Critical Changes You Requested — Now Implemented

| R108 Finding | Severity | What We Did | Evidence |
|-------------|----------|-------------|---------|
| Doc drift (pattern 204 vs 214, node 12 vs 13) | HIGH | Fixed ARCHITECTURE.md | `docs/ARCHITECTURE.md` |
| Mock tests inflating coverage (~350 skipped) | HIGH | **Full mock purge**: 19 files deleted, 37 cleaned, 0 mock imports | `git log bfde94c` — -20,978 LOC deleted |
| No tool execution transcripts | HIGH | **R111 eval**: 29 scenarios, 93 turns, 0 errors, GPT-5.4 judged | `tests/evaluation/r111-results/*.json` |
| Comp auto-approve below industry ($50) | MEDIUM | Raised to $100 regular / $250 VIP | `src/agent/incentives.py` |
| Profiling techniques too narrow (7) | MEDIUM | Added 4 new techniques → 11 total | `src/agent/profiling.py` |
| Endowment framing missing | MEDIUM | "You've earned" replaces "We'd like to offer" | `src/casino/config.py` Mohegan Sun rules |
| Handoff is flat | MEDIUM | 3-tier model (quick/standard/full) + hero_moment | `src/agent/behavior_tools/handoff.py` |
| Profiling questions appended, not embedded | LOW | Updated few-shot examples with embedded Qs | `src/agent/prompts.py` |

### R111 Test Suite (Post-Mock-Purge)
- **2750 tests passing, 0 failures, 2 skips** (re2 install + timing flaky)
- **Zero MagicMock/AsyncMock imports** (AST-verified across 92 test files)
- Coverage: 79.63% (honest — mock-inflated 90% was fake coverage)
- All tests are deterministic (guardrails, extraction, sentiment, incentives, config) or live eval

### R111 Eval Results (29 scenarios, Flash + tools, GPT-5.4 judge)

**Improvements vs R105 baselines:**

| Dimension | R105 | R111 | Delta | What Drove It |
|-----------|------|------|-------|---------------|
| H9 Comp Strategy | 2.35 | **4.14** | **+1.79** | Tool-based comp lookup + endowment framing |
| P6 Incentive Framing | 3.93 | **5.11** | **+1.18** | Per-casino incentive rules + natural framing |
| B6 Tone | 6.24 | **6.58** | +0.34 | 17 slop pattern filters + few-shot examples |
| P8 Profile Completeness | 3.62 | **3.88** | +0.26 | Contextual inference rules |
| H10 Lifetime Value | 3.87 | **4.10** | +0.23 | LTV nudge tool + event lookup |

**Regressions (under investigation):**

| Dimension | R105 | R111 | Delta | Hypothesis |
|-----------|------|------|-------|-----------|
| P9 Host Handoff | 4.30 | 2.22 | -2.08 | 3-tier handoff may not trigger correctly with Flash |
| H5 Trust Building | 5.30 | 4.00 | -1.30 | Tool hallucination ("search_restaurants" not a real tool) |
| B4 Agentic Proactivity | 6.32 | 5.17 | -1.15 | Increased validation strictness → more fallbacks |

**Sample transcript** (engagement-01 — h6-02 complaint recovery):
> **Guest**: "We waited 45 minutes between courses at Tuscany last night..."
> **Agent**: "I am incredibly sorry that you had such a long wait at Tuscany last night; forty-five minutes for a course is completely unacceptable. I am reporting this to the restaurant's management immediately..."
> **Agent** (turn 3): "Michael Jordan's Steak House at 7:30 PM — I'll get the host team to lock that in and flag it to ensure you receive white-glove service to make up for last night."

---

## LAYER 1: CODE ARCHITECTURE (10 dimensions)

### D1. Graph Architecture (Weight: 0.20)
Review `src/agent/graph.py`, `src/agent/state.py`, `src/agent/nodes.py`.
- Is the StateGraph correctly structured with validation loops?
- SRP: Are functions < 100 LOC? Is there proper extraction into helpers?
- Node naming: constants vs magic strings?
- Conditional edges: are all routing paths testable and bounded?
- Recursion limit: is it set? What prevents infinite loops?

### D2. RAG Pipeline (Weight: 0.10)
Review `src/rag/pipeline.py`, `src/rag/reranking.py`, `src/rag/embeddings.py`.
- Per-item chunking for structured data?
- RRF reranking implementation correctness?
- Idempotent ingestion with SHA-256 content hashing?
- Version-stamp purging for stale chunks?
- Embedding model version pinning?

### D3. Data Model & State (Weight: 0.10)
Review `src/agent/state.py`, `src/agent/extraction.py`, `src/agent/profiling.py`.
- TypedDict state with proper Annotated reducers?
- Per-turn vs persistent field separation?
- JSON serialization safety (no sets, no @property across boundaries)?
- Parity assertions between modules sharing field definitions?

### D4. API Design (Weight: 0.10)
Review `src/api/app.py`, `src/api/middleware.py`, `src/api/models.py`.
- Pure ASGI middleware (not BaseHTTPMiddleware)?
- SSE streaming correctness?
- Rate limiting: per-client or global? Atomic operations?
- Security headers: CORS, CSP, HSTS?
- Health endpoint: /live vs /health distinction?

### D5. Testing Strategy (Weight: 0.10)
Review `tests/` directory structure, `conftest.py`, coverage config.
- **NEW**: Zero mock imports (19 mock files deleted, 37 cleaned in R111)
- 2750 tests, 0 failures, 2 skips
- Are guardrails, sentiment, crisis tested WITHOUT mocks?
- Does at least 1 test exercise auth-enabled path?
- Property-based tests (Hypothesis) for regex patterns?

### D6. DevOps & Deployment (Weight: 0.10)
Review `Dockerfile`, CI/CD config, `requirements*.txt`.
- pip install --require-hashes?
- Multi-stage Dockerfile, non-root user?
- HEALTHCHECK in Dockerfile?
- pip-audit in CI/CD pipeline?
- Exec-form CMD (not shell form)?

### D7. Guardrails & Safety (Weight: 0.10)
Review `src/agent/guardrails.py`, `src/agent/compliance_gate.py`, `src/agent/crisis.py`.
- 214 guardrail patterns across 6 categories
- Multi-layer input normalization (URL decode, HTML unescape, NFKC, Cf strip)?
- Pre-LLM deterministic classification (before any LLM call)?
- Fail-closed behavior on classifier failure?
- re2-compatible patterns (no ReDoS risk)?

### D8. Scalability & Production (Weight: 0.15)
Review `src/agent/circuit_breaker.py`, `src/state_backend.py`, `src/api/middleware.py`.
- Circuit breaker with Redis L1/L2 sync?
- TTL cache jitter on singleton caches?
- asyncio.Lock not threading.Lock in async code?
- LLM backpressure via semaphore?
- Graceful shutdown / SIGTERM handling?
- Per-client rate limiting (not global lock)?

### D9. Documentation & ADRs (Weight: 0.05)
Review `docs/`, ADR files, README.md, ARCHITECTURE.md.
- How many ADRs? Do they have Status + Date?
- Is ARCHITECTURE.md accurate (13 nodes, 214 patterns)? **Fixed since R108**
- Are doc counts CI-testable (parity assertions)?

### D10. Domain Intelligence (Weight: 0.10)
Review `src/casino/config.py`, `knowledge-base/`, `src/agent/incentives.py`.
- Multi-property casino configurations?
- Per-casino feature flags (19 flags)?
- Regulatory accuracy (state-specific rules)?
- Incentive engine with tiered autonomy ($100/$250 thresholds)?

---

## LAYER 2: BEHAVIORAL QUALITY (10 dimensions — B1-B10)

Read eval transcripts from `tests/evaluation/r111-results/*.json` (29 scenarios, 93 turns, GPT-5.4 judged).

### B1. Sarcasm & Tone Awareness (Weight: 0.05)
R111: 6.25 | Does the agent detect sarcasm without mirroring it?

### B2. Implicit Signal Reading (Weight: 0.10)
R111: 6.12 | Does the agent pick up unstated needs (fatigue, urgency, VIP)?

### B3. Conversational Engagement (Weight: 0.10)
R111: 6.17 | Natural flow, adapts format based on feedback?

### B4. Agentic Proactivity (Weight: 0.15)
R111: 5.17 (REGRESSED from 6.32) | Uses tools to ground recommendations? CCD pattern?

### B5. Emotional Intelligence (Weight: 0.10)
R111: 7.40 | Handles distress, grief, celebration appropriately?

### B6. Tone Calibration (Weight: 0.10)
R111: 6.58 (+0.34) | No AI slop? Warmth from substance not enthusiasm?

### B7. Multi-Turn Coherence (Weight: 0.10)
R111: 5.88 | Remembers name, preferences, dietary needs across turns?

### B8. Cultural & Multilingual (Weight: 0.05)
(Not directly tested in R111 — score from R105 or independent assessment)

### B9. Safety & Compliance (Weight: 0.10)
R111: 6.33 | Crisis resources, responsible gaming, BSA/AML detection?

### B10. Overall Quality (Weight: 0.20)
R111: 5.67 | Composite behavioral quality

---

## LAYER 3: PROFILING QUALITY (10 dimensions — P1-P10)

### P1. Natural Extraction (Weight: 0.05)
R111: 4.88 | Extracts 2+ fields from volunteered info? No re-asking?

### P2. Active Probing (Weight: 0.10)
R111: 6.42 | At least 1 natural profiling question per response?

### P3. Give-to-Get Balance (Weight: 0.10)
R111: 7.46 | Value delivered WITH every profiling question?

### P4. Assumptive Bridging (Weight: 0.10)
R111: 5.04 | Makes contextual inferences, tests softly?

### P5. Progressive Sequencing (Weight: 0.10)
R111: 5.08 | Foundation → Preference → Relationship path?

### P6. Incentive Framing (Weight: 0.10)
R111: **5.11** (+1.18) | Incentives at natural moments? "You've earned" framing?

### P7. Privacy Respect (Weight: 0.10)
R111: 4.80 | Explains WHY when asking sensitive questions?

### P8. Profile Completeness (Weight: 0.10)
R111: 3.88 | 60%+ Phase 1 fields in first conversation?

### P9. Host Handoff Quality (Weight: 0.15)
R111: **2.22** (REGRESSED from 4.30) | Structured handoff with confidence levels?

### P10. Cross-Turn Memory (Weight: 0.10)
R111: 5.46 | Uses ALL earlier info in later turns?

---

## LAYER 4: HOST TRIANGLE — REVENUE & RETENTION (10 dimensions — H1-H10)

### H1. Property Knowledge Depth (Weight: 0.10)
R111: 6.70 | Specific venues, restaurants, shows from real data?

### H2. Need Anticipation (Weight: 0.10)
R111: 6.40 | Connects dots between what guest said and what they need?

### H3. Solution Synthesis (Weight: 0.10)
R111: 5.20 | Synthesizes multi-part plans, not lists?

### H4. Emotional Attunement (Weight: 0.10)
R111: 6.90 | Calibrates tone to guest's emotional state?

### H5. Trust Building (Weight: 0.10)
R111: **4.00** (REGRESSED from 5.30) | Tool-verified facts stated confidently? No overselling?

### H6. Rapport Depth (Weight: 0.10)
R111: 4.60 | Genuine rapport beyond transactional? Name usage?

### H7. Revenue Generation (Weight: 0.10)
R111: 5.00 | Natural higher-value suggestions?

### H8. Upsell Timing (Weight: 0.10)
R111: 5.78 | Appropriate timing for premium suggestions?

### H9. Comp Strategy (Weight: 0.10)
R111: **4.14** (+1.79 from 2.35) | Tool-based comp check? Appropriate for tier?

### H10. Lifetime Value (Weight: 0.10)
R111: 4.10 (+0.23) | Seeds for future visits? Event data?

---

## SCORING SUMMARY

After reviewing all 40 dimensions, provide:

### Per-Layer Averages
| Layer | R108 Score | R111 Self-Score | Your Score | Delta from R108 |
|-------|-----------|----------------|-----------|-----------------|
| Code Architecture (D1-D10) | 5.7 | — | ? / 10 | ? |
| Behavioral Quality (B1-B10) | 5.5 | 6.08 | ? / 10 | ? |
| Profiling Quality (P1-P10) | 4.8 | 5.01 | ? / 10 | ? |
| Host Triangle (H1-H10) | 4.2 | 5.28 | ? / 10 | ? |

### Overall Score
Weighted average across all 40 dimensions.

### Recommendation
**INVEST** / **HOLD** / **PASS** with 3-sentence justification.

### Top 5 Critical Fixes (Priority Order)
1. [Dimension] — [Specific fix] — [Expected impact]
2. ...
3. ...
4. ...
5. ...

### Comparison to R108
For each layer, state: What improved? What regressed? What's the single biggest remaining gap?

### Strategic Assessment
- Is this product ready for a pilot with 1 casino client?
- What is the minimum viable quality bar for paid deployment?
- What are the top 3 risks for the business?
- Has the project trajectory been positive since R108? (5.2/10 → ?)

---

## ARCHITECTURE CONTEXT

- **13-node LangGraph StateGraph** with validation loops (generate → validate → retry(1) → fallback)
- **31 state fields**, 19 feature flags (per-casino via Firestore)
- **4 casino tools**: check_comp_eligibility, lookup_tier_benefits, lookup_upcoming_events, check_incentive_eligibility
- **Tool-empowered CCD authority**: Checked (calls tool) → Confirmed (states result as fact) → Dispatched (decisive handoff to host team)
- **5-layer pre-LLM guardrails**: 214 regex patterns, fail-closed
- **Flash→Pro model routing** for complex/emotional queries
- **Guest profiling enrichment node** with 11 techniques, confidence gating, golden path sequencing
- **Incentive engine** with per-casino rules, $100/$250 auto-approve thresholds, endowment framing
- **3-tier handoff model** (quick/standard/full) with hero_moment field
- **27 few-shot examples** per specialist, 17 slop pattern filters, 15 cross-domain bridges
- **51 gold traces** in `data/training/` (CCD-compliant)
- **270 eval scenarios** across 35 YAML files

### Known Limitations (Don't penalize — acknowledged trade-offs)
- ChromaDB in dev, Vertex AI Vector Search in prod
- Frontend (Next.js) is MVP — backend focus
- Coverage 79.63% (honest post-mock-purge, was fake 90% with mocks)
- Gemini 3.1 Pro rate-limited at 250 RPD (free tier)
- No staging environment (deferred to R112)
- No load testing yet (deferred to R113)

### What We're NOT Asking You to Validate
- Frontend code quality (not the focus)
- GCP deployment configuration (not yet deployed)
- Fine-tuning pipeline (planned, not built)
