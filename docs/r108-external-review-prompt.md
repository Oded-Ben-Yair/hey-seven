# Hey Seven R108 — 40-Dimension Product Review Prompt

**For**: GPT-5.4 Pro Deep Research (or equivalent frontier model)
**Mode**: Hostile external review — score honestly, find real problems, no flattery
**Repository**: https://github.com/Oded-Ben-Yair/hey-seven

---

## Instructions

You are reviewing **Hey Seven**, a production AI casino host agent. This is NOT just a code review — it is a **complete product review** covering code quality, behavioral quality, business logic, regulatory compliance, deployment readiness, and strategic positioning.

**Ground rules:**
1. Score each dimension 1-10 with specific evidence (file:line or transcript quote)
2. Do NOT inflate scores — a 7 means "good, production-ready", 9 means "exceptional, best-in-class"
3. For each dimension, provide: Score, Evidence (what you saw), Gap (what's missing), Fix (specific action)
4. Read the FULL codebase, not just highlights. Clone the repo and explore.
5. After scoring all 40 dimensions, provide an overall INVEST / HOLD / PASS recommendation with 3-sentence justification

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
- Test count and 0-failure gate?
- Coverage configuration covers ALL src/ directories?
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
- How many guardrail categories? How many regex patterns?
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
- Is ARCHITECTURE.md accurate (node count, pattern count)?
- Are doc counts CI-testable (parity assertions)?

### D10. Domain Intelligence (Weight: 0.10)
Review `src/casino/config.py`, `knowledge-base/`, `src/agent/incentives.py`.
- Multi-property casino configurations?
- Per-casino feature flags?
- Regulatory accuracy (state-specific rules)?
- Incentive engine with tiered autonomy?

---

## LAYER 2: BEHAVIORAL QUALITY (10 dimensions — B1-B10)

For each dimension, read 3-5 eval transcripts from `tests/evaluation/results/r108-tools-streaming/` and the scenario YAML files.

### B1. Tone & Register (Weight: 0.05)
- Does the agent sound like a casino host, not a chatbot?
- No AI slop ("certainly!", "I'd be happy to!", "absolutely!")?
- Consistent voice across specialists?

### B2. Factual Accuracy (Weight: 0.10)
- Are venue names, hours, locations accurate per knowledge-base/?
- Does the agent fabricate facts not in RAG context?
- Does validation catch hallucinations?

### B3. Conversational Flow (Weight: 0.10)
- Natural multi-turn progression?
- Does the agent remember context from earlier turns?
- Smooth specialist handoffs (no topic resets)?

### B4. Tool Integration (Weight: 0.15)
- Does the agent call tools when appropriate (comp, tier, events)?
- Does it use tool results naturally in responses (CCD pattern)?
- Checked (calls tool) → Confirmed (states result) → Dispatched (decisive handoff)?

### B5. Safety Response (Weight: 0.10)
- Correct crisis escalation (4 levels)?
- Responsible gaming detection and response?
- BSA/AML compliance trigger handling?
- Age verification handling?

### B6. Emotional Intelligence (Weight: 0.10)
- Appropriate response to loss/disappointment?
- Grief detection and compassionate response?
- Celebration recognition and enhancement?

### B7. Multilingual (Weight: 0.05)
- Spanish language detection and response?
- Crisis resources in detected language?
- No language mixing within a response?

### B8. Cultural Sensitivity (Weight: 0.05)
- Appropriate handling of diverse cultural contexts?
- No assumptions about guest background?

### B9. Recovery & Fallback (Weight: 0.10)
- Graceful fallback when tools/LLM fails?
- Does fallback still feel like a host (not a system error)?
- Does the agent recover and continue conversation after fallback?

### B10. Authority & Decisiveness (Weight: 0.20)
- Does the agent decide (not hedge)?
- CCD pattern: "I'll get the team to..." not "You could try..."
- Specific recommendations vs generic suggestions?

---

## LAYER 3: PROFILING QUALITY (10 dimensions — P1-P10)

### P1. Name Extraction (Weight: 0.05)
- Does the agent correctly extract and use guest names?
- No false positives ("I'm done" → name "Done")?

### P2. Active Probing (Weight: 0.10)
- Does the agent ask profiling questions every turn?
- "What are we celebrating?" before "Here's what I recommend"?

### P3. Give-to-Get (Weight: 0.10)
- Does the agent give value before asking for information?
- Natural exchange, not interrogation?

### P4. Progressive Disclosure (Weight: 0.10)
- Does the agent build profile across turns?
- Does it avoid repeating questions already answered?

### P5. Occasion Detection (Weight: 0.10)
- Birthday, anniversary, celebration, loss — detected correctly?
- Used to personalize subsequent recommendations?

### P6. Incentive Framing (Weight: 0.10)
- Are incentives woven naturally, not as sales pitches?
- "Pleasant surprise" framing?
- Tool-based incentive eligibility checking?

### P7. Preference Learning (Weight: 0.10)
- Does the agent learn food/entertainment/gaming preferences?
- Does it use learned preferences in later turns?

### P8. Profile Accuracy (Weight: 0.10)
- Are extracted fields correct?
- Confidence gating: does it reject uncertain extractions?

### P9. Host Bridge (Weight: 0.15)
- Does the agent offer to connect with a human host?
- Structured handoff summaries with guest facts + next actions?
- Frustration detection → immediate handoff offer?

### P10. Profile Utilization (Weight: 0.10)
- Does the agent USE gathered profile data in recommendations?
- "Since you mentioned you love jazz, the Wolf Den tonight..."?

---

## LAYER 4: HOST TRIANGLE — REVENUE & RETENTION (10 dimensions — H1-H10)

### H1. Property Knowledge Depth (Weight: 0.10)
- Can the agent describe specific venues, restaurants, shows in detail?
- Real data from knowledge-base, not generic descriptions?

### H2. Win-Back Capability (Weight: 0.10)
- Can the agent re-engage a lapsed guest?
- Does it acknowledge past bad experiences and offer recovery?

### H3. Cross-Domain Suggestions (Weight: 0.10)
- After dinner recommendation → suggests show?
- Natural category bridging, not forced?

### H4. VIP Recognition (Weight: 0.10)
- Does the agent recognize and respond to VIP signals?
- Specific comp mechanics, not generic "valued guest"?

### H5. Trust Building (Weight: 0.10)
- Does the guest feel the agent cares?
- Consistency, follow-through promises?

### H6. Rapport Progression (Weight: 0.10)
- Does rapport build across turns?
- Phase-aware patterns (opening → building → deepening)?

### H7. Conflict Resolution (Weight: 0.10)
- How does the agent handle complaints?
- Acknowledge → validate → resolve pattern?

### H8. Memory & Continuity (Weight: 0.10)
- Does the agent reference earlier conversation context?
- "Earlier you mentioned..." type callbacks?

### H9. Comp Strategy (Weight: 0.10)
- Does the agent check comp eligibility via tools?
- Are comp offers appropriate for the guest's tier?
- Natural framing, not transactional?
- **R105 baseline: 2.35/10** — this is the most critical revenue dimension

### H10. Return Visit Seeding (Weight: 0.10)
- Does the agent plant reasons to come back?
- Upcoming events, seasonal offerings?
- LTV nudge integration with real event data?
- **R105 baseline: 3.87/10**

---

## SCORING SUMMARY

After reviewing all 40 dimensions, provide:

### Per-Layer Averages
| Layer | Avg Score | Dimensions Scored |
|-------|-----------|-------------------|
| Code Architecture (D1-D10) | ? / 10 | 10 |
| Behavioral Quality (B1-B10) | ? / 10 | 10 |
| Profiling Quality (P1-P10) | ? / 10 | 10 |
| Host Triangle (H1-H10) | ? / 10 | 10 |

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

### Strategic Assessment
- Is this product ready for a pilot with 1 casino client?
- What is the minimum viable quality bar for paid deployment?
- What are the top 3 risks for the business?

---

## CONTEXT FOR THE REVIEWER

### R108 Eval Results (5 scenarios, Flash + tools)
- Tool execution rate: 54% (tools bind and execute correctly)
- 0% error rate, 27% fallback rate (3rd-turn confirmations)
- All 5 scenarios had tool invocations (host, entertainment, comp agents)
- CCD language confirmed in transcripts ("I'll get the team to...")

### R105 Baselines (85 scenarios, Pro, GPT-5.2 judge)
- B-avg: 6.62 | P-avg: 5.18 | H-avg: 5.09
- Sub-5.0 dims: H9(2.35), P9(2.45→4.3 fixed), P6(3.93), P8(3.62), H10(3.87)
- Prompt engineering ceiling confirmed — architecture changes needed

### Architecture Highlights
- 13-node LangGraph StateGraph with validation loops
- 4 casino tools (comp, tier, events, incentives) via LangGraph bind_tools
- 5-layer pre-LLM guardrails (214 patterns)
- Flash→Pro model routing for complex/emotional queries
- Per-casino feature flags via Firestore with in-memory cache
- Guest profiling with 10-dimension framework
- 27 few-shot behavioral examples injected per specialist

### Known Issues (Reviewer should verify, not re-discover)
- 270 legacy mock tests broken by config change (ground rule: no mock fixes)
- Gemini 3.1 Pro rate-limited at 250 RPD (free tier)
- ChromaDB is dev-only; prod uses Vertex AI Vector Search
- Frontend (Next.js) is MVP — not the focus of this review

### What Changed in R107-R108
- R107: Authority model rewrite (CCD pattern), tool activation, prod fail-hard
- R108: CRITICAL config bug fix (tools weren't binding), eval confirmed tools work
- Gold traces updated for CCD compliance
- Export script expanded (+R98/R99 sources)
