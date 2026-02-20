# Production Rebrand + 10-Round Hostile Review Protocol

**Date**: 2026-02-20
**Status**: Approved
**Author**: Oded Ben-Yair + Claude Opus 4.6

---

## 1. Context

Hey Seven project is transitioning from interview assignment to production MVP. Oded passed the CTO interview and is joining Hey Seven as Senior AI/Backend Engineer.

The codebase has 23K LOC, 1070+ tests, 32 test files, and went through 20 rounds of hostile review (v2-r1 through v2-r20) reaching 92/100. It's tagged v1.0.0.

This design covers two workstreams:
1. **Production rebrand** — strip all interview language, delete interview-era files
2. **10 rounds of hostile LLM review** — using real models via MCP, targeting 95+ score

---

## 2. Production Rebrand

### Delete (complete removal from repo)

| Target | Reason |
|--------|--------|
| `assignment/` (entire directory) | Interview prep files |
| `reviews/` (entire directory) | Old review rounds — clean production repo |
| `boilerplate/` (entire directory) | Pre-built code already in `src/` |
| `Take-Home Assignment — Senior AI_Backend Engineer.pdf` | Interview document |
| `former session hey seven.txt` | Session transcript |
| `research/personas/` | Interview research |

### Rewrite

| File | Changes |
|------|---------|
| `CLAUDE.md` | Strip all interview language. Remove "Assignment Reception Protocol", "Oded's Strengths to Highlight", interview team templates. Reframe as production project. |
| `README.md` | Production README — product description, setup, deployment, API docs. No interview references. |

### Edit (targeted changes)

| File | Line | Change |
|------|------|--------|
| `ARCHITECTURE.md:813` | Remove "assignment/architecture.md" reference, reframe as production architecture |
| `src/api/middleware.py:180` | Remove "interview demo" comment |

### Keep (production value)

- `research/brand-design.md` — brand identity
- `research/casino-domain.md` — domain knowledge
- `research/company-intel.md` — company context
- `research/langgraph-*.md` — technical patterns
- `research/frontend-latest.md` — frontend stack
- `research/perplexity-deep/` — regulations, market research
- `knowledge-base/` — RAG data
- `src/` — production code
- `tests/` — test suite

---

## 3. 10-Round Review Protocol

### 3.1 Round Structure

Each round = 1 agent team ("prod-review-rN") with 4 teammates:

```
Team: "prod-review-rN"
├── gemini-reviewer  → gemini-query (thinking_level="high")
├── gpt-reviewer     → azure_chat (model="gpt-5.2")
├── grok-reviewer    → grok_reason (Grok 4)
└── fixer            → code-worker (blocked by 3 reviewers)
```

Rounds 5 and 10: Replace Grok with DeepSeek (`azure_deepseek_reason`).

### 3.2 Review Dimensions (10)

| # | Dimension | Checks |
|---|-----------|--------|
| 1 | Graph/Agent Architecture | StateGraph structure, specialist dispatch, validation loop, state management |
| 2 | RAG Pipeline | Chunking, retrieval, reranking, idempotent ingestion, multi-tenant safety |
| 3 | Data Model / State Design | TypedDict fields, reducers, serialization, guest profiles |
| 4 | API Design | Middleware, SSE, error handling, auth, rate limiting |
| 5 | Testing Strategy | Coverage, test quality, edge cases, integration tests |
| 6 | Docker & DevOps | Dockerfile, CI/CD, security scanning, deployment config |
| 7 | Prompts & Guardrails | System prompts, deterministic guardrails, injection defense |
| 8 | Scalability & Production | Circuit breakers, caching, async patterns, error recovery |
| 9 | Documentation & Code Quality | README, inline docs, naming, patterns consistency |
| 10 | Domain Intelligence | Casino ops, regulations, SMS/TCPA, comp system |

### 3.3 Scoring

- Each model scores each dimension 0-10
- Round score = average across 3 models
- **Consensus rule**: Finding must be flagged by 2/3 models to be "confirmed"
- **Severity levels**: CRITICAL > HIGH > MEDIUM > LOW
- **Target**: Round 10 score >= 95/100

### 3.4 Model Configuration

| Model | MCP Tool | Settings | Strengths |
|-------|----------|----------|-----------|
| Gemini 3 Pro | `gemini-query` | `thinking_level: "high"` | Architecture, patterns, deep reasoning |
| GPT-5.2 | `azure_chat` | `model: "gpt-5.2"` | Code quality, security, edge cases |
| Grok 4 | `grok_reason` | default | Practical issues, deployment, real-world gaps |
| DeepSeek | `azure_deepseek_reason` | default | Algorithmic correctness, mathematical rigor |

### 3.5 Reviewer Prompt Template

Each reviewer receives:
1. Full repo file tree
2. Key source files (read by the reviewer agent)
3. The 10 dimensions with scoring rubric (0-10 per dimension)
4. Previous round scores (trajectory tracking)
5. Round-specific spotlight area
6. Instruction: "Score each dimension 0-10. List findings as SEVERITY | file:line | description | specific fix."

### 3.6 Fix Protocol

The `fixer` teammate:
1. Reads all 3 review files from `reviews/prod-rN/`
2. Identifies consensus findings (2/3+ models)
3. Fixes CRITICAL first, then HIGH, then MEDIUM
4. Runs `pytest` — must pass with 0 failures
5. Writes `reviews/prod-rN/summary.md` (scores, fixes applied, remaining items)
6. Commits and pushes to GitHub

### 3.7 Context Safety

- Main lead **never reads** raw review files (only `summary.md`)
- Each team created and destroyed per round
- Between rounds: main lead reads summary, reports to user, creates next team
- All findings written to files, not team messages

---

## 4. Round Focus Areas (Spotlights)

| Round | Spotlight | Goal |
|-------|-----------|------|
| R1 | Production rebrand completeness | Verify all interview language gone, README production-ready |
| R2 | Security hardening | Injection, auth, secrets, PII redaction |
| R3 | Error handling & resilience | Circuit breakers, fallbacks, graceful degradation |
| R4 | Testing gaps | Missing edge cases, integration coverage |
| R5 | Scalability (+ DeepSeek) | Async patterns, memory, load behavior |
| R6 | API contract & documentation | README accuracy, API docs, code comments |
| R7 | RAG & domain data quality | Knowledge base accuracy, retrieval quality |
| R8 | Deployment readiness | Docker, Cloud Run, env vars, health checks |
| R9 | Code simplification | Dead code, over-engineering, unnecessary complexity |
| R10 | Final adversarial (+ DeepSeek) | Break everything. Stress test. Production sign-off |

---

## 5. Execution Plan

### Phase 1: Production Rebrand (1 session)
1. Delete all interview-era files
2. Rewrite CLAUDE.md and README.md
3. Edit ARCHITECTURE.md and middleware.py
4. Commit and push
5. Run tests to verify nothing broke

### Phase 2: Review Rounds (1 session, 10 teams sequentially)
1. Create team for round N
2. 3 reviewers execute in parallel
3. Fixer applies consensus fixes, runs tests, pushes
4. Main lead reads summary, shuts down team
5. Create team for round N+1
6. Repeat until round 10

### Phase 3: Final Validation
1. Full test suite passing
2. Docker build succeeds
3. README and ARCHITECTURE.md accurate
4. All 10 round summaries show trajectory toward 95+
5. Tag v2.0.0

---

## 6. Success Criteria

- [ ] Zero interview/assignment language in entire repo
- [ ] README.md describes production product
- [ ] CLAUDE.md frames production project
- [ ] 10 rounds completed with real model reviews
- [ ] Round 10 score >= 95/100
- [ ] All tests passing (1070+ baseline)
- [ ] Docker builds and runs
- [ ] All CRITICAL and HIGH findings fixed
- [ ] Tagged v2.0.0
