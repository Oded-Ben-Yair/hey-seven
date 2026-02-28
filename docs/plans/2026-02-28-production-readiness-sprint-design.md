# Production-Readiness Sprint Design

**Date**: 2026-02-28
**Input**: External reviews from GPT-5.3 Codex (4 docs) + GPT Pro (1 doc)
**Target**: 9.0+ across all 20 dimensions
**Approach**: Fix real issues + rebuttal for false positives + live re-evaluation

## External Review Scores

| Source | Technical | Behavioral | Overall | CRITICALs | MAJORs |
|--------|-----------|------------|---------|-----------|--------|
| Codex V1 (Executive) | 8.2 | 6.6 | 7.4 | 1 | — |
| Codex V2 (Summary) | 6.7 | 6.3 | 6.5 | 1 | 14 |
| GPT Pro | 6.7 | 6.3 | 6.5 | 1 | 14 |

## Finding Validation

### Confirmed Real (8 findings)
1. **CRITICAL: RE2 enforcement** — `regex_engine.py` falls back to stdlib re without failing in production
2. **Version parity** — pyproject.toml=0.1.0, config.py=1.3.0
3. **ARCHITECTURE.md drift** — may describe 8-node graph vs current 11-node
4. **Degrade-pass guard** — validator error on attempt 0 can ship unvalidated response
5. **Crisis exit heuristic** — keyword list is brittle for exit detection
6. **B4 Proactivity** — infrastructure exists but not delivering (4/10)
7. **Behavioral CI gates** — eval tests API-key gated, regressions can ship silently
8. **B2/B3/B7** — implicit signals unstructured, domains_discussed not used for engagement

### False Positives (6 findings)
1. **RAG chunk ID collision** — `pipeline.py:244` already uses `\x00` delimiter (R36 fix A5)
2. **Docker --require-hashes** — `requirements-prod.txt` IS hash-generated, Dockerfile uses `--require-hashes`
3. **Circuit breaker race** — all mutations under `asyncio.Lock`, `is_open` is documented monitoring-only
4. **Requirements not hashed** — reviewer saw `requirements.txt` (dev), not `requirements-prod.txt` (prod)
5. **Retrieval pool hardcoded** — configurable via settings, semaphore backpressure exists
6. **Auth scope binding** — UUID thread IDs + API key is sufficient for MVP

## Sprint Waves

### Wave 1: Safety-Critical (CRITICAL fix)
- RE2 enforcement: fail-fast at startup if `ENVIRONMENT != "development"` and RE2 unavailable

### Wave 2: Technical Hardening (Real MAJORs)
- Version parity: sync pyproject.toml to 1.3.0
- ARCHITECTURE.md: update or delete
- Degrade-pass guard: require retrieval grounding when validator fails on attempt 0
- Doc drift tests: node count + version parity assertions

### Wave 3: Crisis/Safety Behavioral (B9)
- Crisis exit: replace keyword list with explicit safe confirmation pattern
- Crisis templates: per-severity response templates

### Wave 4: Behavioral Quality (B2/B3/B4/B7)
- B4: Wire whisper suggestions into specialist execution (gated)
- B2: Namespace extracted_fields (preferences, trip_context, constraints)
- B3: Use domains_discussed to vary engagement
- B7: Preference memory schema

### Wave 5: Rebuttal Document
- `docs/review-response.md` with code evidence for each false positive

### Wave 6: Live Re-evaluation
- 74 scenarios through real Gemini Flash
- 3-model judge panel

## Deferred (YAGNI)
- Semantic reranking after RRF
- Last-Event-ID SSE replay
- Staging → production promotion gate
- Multilingual response generation (ADR-005)
- Auth scope binding / thread fixation
