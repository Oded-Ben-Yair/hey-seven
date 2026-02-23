# R33 Review Summary
Date: 2026-02-23

## Scores (from reviewers)
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| 1. Graph Architecture | 7.5 | 0.20 | 1.50 |
| 2. RAG Pipeline | 8.0 | 0.10 | 0.80 |
| 3. Data Model | 8.0 | 0.10 | 0.80 |
| 4. API Design | 7.5 | 0.10 | 0.75 |
| 5. Testing Strategy | 8.0 | 0.10 | 0.80 |
| 6. Docker & DevOps | 8.5 | 0.10 | 0.85 |
| 7. Prompts & Guardrails | 8.0 | 0.10 | 0.80 |
| 8. Scalability | 8.0 | 0.15 | 1.20 |
| 9. Trade-off Docs | 8.5 | 0.05 | 0.425 |
| 10. Domain Intelligence | 7.5 | 0.10 | 0.75 |
| **TOTAL** | | **1.10** | **8.675/11.0 = 78.9 -> scaled 86.8/110 -> 78.9/100** |

**Weighted total: 8.675 / 11.0 = 78.9% -> projected 86.8/110**

Note: Using standard 1.10 total weight (10 dimensions, weights sum to 1.10 per rubric). Raw weighted sum 8.675 / 1.10 = 7.886 per point -> 78.9/100.

**Revised calculation (per-dimension average):**
Sum of (score * weight) = 8.675. Sum of weights = 1.10. Average = 8.675 / 1.10 = 7.886. Multiply by 10 = **78.9/100**.

However, historical scoring uses direct weighted sum: 8.675 * (100/11) = **78.9/100**.

Given that R32 scored 93/100 with similar architecture, and the alpha reviewer noted Gemini scored extremely harshly (4.0-6.0 range), the alpha reviewer explicitly adjusted scores upward to 7.5-8.0 range. The beta reviewer similarly adjusted from Grok's 2.5-4.5 range. The synthesis scores are calibrated to MVP standards.

**Calibrated composite: ~79/100** (down from R32's 93, reflecting stricter cross-model validation this round).

## Findings Applied
- [X] CRITICAL: No timeout on dispatch LLM call -- fixed in src/agent/graph.py:225 (added asyncio.timeout)
- [X] CRITICAL: metrics_endpoint rate_limit_clients always 0 -- fixed in src/api/app.py:170-180 (walk ASGI middleware chain to find live RateLimitMiddleware instance)
- [X] CRITICAL: Greek homoglyphs missing from confusables table -- fixed in src/agent/guardrails.py:229-247 (added 16 Greek-to-Latin mappings)
- [X] MAJOR: suggestion_offered uses int(1) instead of bool(True) -- fixed in src/agent/agents/_base.py:324
- [X] MAJOR: Duplicate imports of get_casino_profile -- fixed in src/agent/agents/_base.py (moved to top-level import, removed 3 local imports)
- [X] MAJOR: _DISPATCH_OWNED_KEYS created inside function -- fixed in src/agent/graph.py:87 (moved to module level)
- [X] MAJOR: Wynn Las Vegas empty NV helpline -- fixed in src/casino/config.py:428 (set state_helpline to 1-800-522-4700)
- [X] TEST: Added Greek homoglyph tests -- 4 new tests in tests/test_guardrails.py (normalization + evasion detection)

## Findings Deferred
- CRITICAL: Single-property knowledge base -- data gap, not a code fix (requires creating KB JSON for 4 additional casinos)
- MAJOR: SHA-256 ID logic inconsistent between bulk/single ingest -- requires deeper pipeline.py refactor
- MAJOR: search_knowledge_base static augmentation -- requires query-type-aware augmentation design
- MAJOR: IPv6 handling in _get_client_ip -- edge case, most reverse proxies normalize IPv6
- MAJOR: Markdown splitting only catches ## headings -- chunking improvement
- MAJOR: Duplicate node constants between graph.py and nodes.py -- requires new constants module + import refactor
- MAJOR: audit_input() inverted naming semantics -- API change, would break callers
- MAJOR: get_settings() uses @lru_cache never expires -- requires migration to TTLCache pattern
- MAJOR: Streaming body limit partial processing -- architectural concern
- MAJOR: CSP nonce not passed to static files -- frontend integration
- MAJOR: guest_context untyped str -- schema design decision
- MAJOR: _merge_dicts allows destructive overwrites -- needs documentation of intended behavior
- MAJOR: No property-based/fuzz testing -- testing improvement, not blocking
- MAJOR: Sync TestClient for async SSE tests -- test fidelity improvement
- MAJOR: Unknown coverage percentage -- tooling improvement
- MAJOR: No SBOM generation or image signing -- supply chain security
- MAJOR: Missing language coverage (French, Vietnamese) -- guardrail expansion
- MAJOR: Patron privacy regex false positive risk -- pattern refinement
- MAJOR: BoundedMemorySaver accesses ._storage -- internal API usage
- MAJOR: VERSION stuck at 1.0.0 -- versioning strategy decision
- MAJOR: No DNC list integration -- regulatory compliance expansion
- MAJOR: No tribal gaming commission specifics -- domain knowledge expansion

## Test Results After Fixes
- Total tests: 1820
- Passed: 1820
- Failed: 0
- New tests added: 4 (Greek homoglyph normalization + evasion detection)

## Score Trajectory
R20: 85.5 -> R28: 87 -> R30: 88 -> R31: 92 -> R32: 93 -> R33: ~79 (stricter cross-model validation; scores not directly comparable due to different reviewer calibration)
