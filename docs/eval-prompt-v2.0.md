# Hey Seven Code Evaluation — v2.0 (Frozen)

**Version**: 2.0.0
**Frozen**: 2026-02-27
**Rationale**: ADR-023 — 3-Tier Reproducible Evaluation System

## Instructions

You are evaluating a production LangGraph casino host agent. Score based solely on
evidence in the code. Do not reference any previous scores or prior review rounds.

### Repo
GitHub: https://github.com/Oded-Ben-Yair/hey-seven
Commit: {commit_hash}

## Scoring Rubric (10 Dimensions, 0-10 each)

### Calibration Anchors

Use these examples to calibrate your scores consistently:

| Score | Meaning | Example |
|-------|---------|---------|
| 3 | Fundamental gaps | No tests, no auth, hardcoded secrets, mutable global state |
| 6 | Functional but fragile | Tests exist but low coverage, auth exists but bypassable, basic error handling |
| 9 | Production-grade | 90%+ coverage, multi-layer auth, circuit breakers, graceful degradation, documented trade-offs |

### Dimensions

1. **Graph/Agent Architecture** (0-10): StateGraph structure, specialist dispatch, validation loop, SRP compliance (<100 LOC per function), state management, conditional edges.
2. **RAG Pipeline** (0-10): Per-item chunking, RRF reranking, idempotent ingestion, multi-tenant metadata isolation, embedding model pinning, version-stamp purging.
3. **Data Model / State Design** (0-10): TypedDict with reducers, serialization safety, tombstone deletion support, parity checks across config locations.
4. **API Design** (0-10): Pure ASGI middleware, SSE streaming, error taxonomy, auth (timing-safe comparison), rate limiting (per-client), PII redaction (fail-closed).
5. **Testing Strategy** (0-10): Coverage >= 90%, zero failures, zero xfails, E2E graph tests with schema-dispatching mock LLM, auth-enabled test paths.
6. **Docker & DevOps** (0-10): Multi-stage, non-root, digest-pinned, --require-hashes, HEALTHCHECK /live, exec-form CMD, SBOM generation, pip-audit.
7. **Prompts & Guardrails** (0-10): Multi-layer normalization (URL + HTML + NFKD + Cf + confusables), 6 categories, re2-compatible syntax, multilingual coverage.
8. **Scalability & Production** (0-10): Circuit breaker with Redis sync, TTLCache with jitter, asyncio.Lock (not threading.Lock) in async paths, graceful shutdown, backpressure.
9. **Documentation & Trade-offs** (0-10): ADR count and quality, version parity, pattern count accuracy, regulatory references verified.
10. **Domain Intelligence** (0-10): Multi-property configs via get_casino_profile(), regulatory accuracy, onboarding checklists, emotional context handling.

## Review Process

1. Score each dimension 0-10 with a 1-2 sentence evidence-based justification.
2. Report all findings at MAJOR severity or above. If fewer than 2 findings, explain why the codebase is clean in that area.
3. Do NOT use a spotlight area or severity bumps. All dimensions are scored equally.
4. Do NOT reference previous scores. Score from a cold baseline.
5. All 4 reviewers score ALL 10 dimensions (enables ICC calculation).

### Finding Format

```
### Finding N (SEVERITY): Title
- **Location**: `file.py:line`
- **Problem**: What's wrong (specific, evidence-based)
- **Impact**: What breaks in production
- **Fix**: Specific code change
```

### Severity Levels

- **CRITICAL**: Security vulnerability, data leakage, regulatory violation, or production crash
- **MAJOR**: Correctness bug, scalability bottleneck, or missing safety mechanism
- **MINOR**: Code quality, documentation drift, or style inconsistency

## What This Prompt Removes (vs v1.0)

| v1.0 Element | Removed | Why |
|--------------|---------|-----|
| "Minimum 5 findings, look harder" | Yes | Forces manufactured criticisms at quality ceiling |
| `{previous_scores_table}` | Yes | Anchoring bias — new round anchors to old scores |
| "HOSTILE" framing | Yes | Primes for negativity, inflates severity |
| `{spotlight_area}` + severity bump | Yes | Changes instrument between rounds, breaks ICC |
| Specialist dimension assignment | Yes | Prevents inter-rater reliability measurement |

## Model Version Pinning

Record the exact model ID used for each review (not just "GPT" or "Gemini").
Example: `gemini-3.1-pro-preview-0226`, `gpt-5.2-codex-0220`, `grok-4-0215`.

## Weighted Score Calculation

| Dim | Weight |
|-----|--------|
| D1 Graph Architecture | 0.20 |
| D2 RAG Pipeline | 0.10 |
| D3 Data Model | 0.10 |
| D4 API Design | 0.10 |
| D5 Testing Strategy | 0.10 |
| D6 Docker & DevOps | 0.10 |
| D7 Prompts & Guardrails | 0.10 |
| D8 Scalability & Prod | 0.15 |
| D9 Trade-off Docs | 0.05 |
| D10 Domain Intelligence | 0.10 |

**Weighted score** = sum(score_i * weight_i) * 10
