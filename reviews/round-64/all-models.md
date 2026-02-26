# R64 Calibrated 4-Model Review

**Date**: 2026-02-26
**Strategy**: Longer code sections sent to each model to avoid false positives from truncated snippets.
**Models**: Gemini 3 Pro (D1-D3), GPT-5.2 Codex (D4-D6), DeepSeek V3.2 Speciale (D7-D8), Grok 4 (D9-D10)

---

## Scores

| Dim | Name | Weight | Score | Model | Delta vs R63 |
|-----|------|--------|-------|-------|--------------|
| D1 | Graph Architecture | 0.20 | **9.8** | Gemini 3 Pro | +0.1 |
| D2 | RAG Pipeline | 0.10 | **9.8** | Gemini 3 Pro | +0.1 |
| D3 | Data Model | 0.10 | **9.9** | Gemini 3 Pro | +0.2 |
| D4 | API Design | 0.10 | **7.5** | GPT-5.2 Codex | -0.5 |
| D5 | Testing Strategy | 0.10 | **9.0** | GPT-5.2 Codex | +0.3 |
| D6 | Docker & DevOps | 0.10 | **8.5** | GPT-5.2 Codex | +0.2 |
| D7 | Prompts & Guardrails | 0.10 | **9.8** | DeepSeek V3.2 | +0.1 |
| D8 | Scalability & Prod | 0.15 | **9.9** | DeepSeek V3.2 | +0.2 |
| D9 | Trade-off Docs | 0.05 | **9.5** | Grok 4 | +0.3 |
| D10 | Domain Intelligence | 0.10 | **9.8** | Grok 4 | +0.1 |

### Weighted Total

```
D1:  9.8 * 0.20 = 1.960
D2:  9.8 * 0.10 = 0.980
D3:  9.9 * 0.10 = 0.990
D4:  7.5 * 0.10 = 0.750
D5:  9.0 * 0.10 = 0.900
D6:  8.5 * 0.10 = 0.850
D7:  9.8 * 0.10 = 0.980
D8:  9.9 * 0.15 = 1.485
D9:  9.5 * 0.05 = 0.475
D10: 9.8 * 0.10 = 0.980
                  ------
TOTAL:            9.350 / 10 = 93.5
```

**Weighted Total: 93.5 / 100**

---

## Trajectory

| Round | Score | Notes |
|-------|-------|-------|
| R52 | 67.7 | Baseline after structural sprint |
| R53 | 84.3 | +16.6 (hardening fixes) |
| R54 | 85.7 | +1.4 |
| R55 | 88.7 | +3.0 |
| R56 | 90.1 | +1.4 |
| R57 | 92.4 | +2.3 |
| R58-R63 | 91-94 | Oscillation from calibration resets |
| **R64** | **93.5** | **Stable 93+ with calibrated review** |

---

## Findings by Model

### Gemini 3 Pro (D1, D2, D3)

**D1: 9.8** -- "Exceptional architectural guardrails for specialist dispatching."

Praised:
- Rigorous state protection via `_DISPATCH_OWNED_KEYS` collision detection + stripping
- Retry reuse pattern ("brilliantly optimized graph routing")
- Keyword fallback with deterministic tie-breaking
- Circuit breaker integration with correct error classification (parse errors != network errors)

Findings:
- **MINOR** `dispatch.py:34` -- `_extract_node_metadata` defined but not exported in `__all__`. Note: it IS used -- imported by `graph.py` line 58. Not dead code. False positive from not seeing the importing file.

**D2: 9.8** -- "Strong concurrent dual-strategy retrieval architecture."

Praised:
- `_safe_await` error isolation for asyncio.gather (one strategy can fail without killing the other)
- ThreadPoolExecutor(50) sized for Cloud Run concurrency
- RRF late-fusion with cosine score quality gate (not RRF rank score)

Findings: None.

**D3: 9.9** -- "Flawless use of LangGraph TypedDict schemas and Annotated reducers."

Praised:
- `UNSET_SENTINEL` tombstone for explicit field deletion (UUID-namespaced, JSON-serializable)
- `_merge_dicts` with None/empty-string filtering + tombstone support
- Pydantic models with `Literal` type constraints and `Field` bounds

Findings: None.

### GPT-5.2 Codex (D4, D5, D6)

**D4: 7.5** -- "Strong middleware design with security, logging, rate limiting."

Praised:
- Pure ASGI middleware (not BaseHTTPMiddleware)
- Consistent security headers across all error codes
- Rate limiting with LRU + Redis fallback + atomic Lua script
- Content-Encoding rejection + streaming byte counter

Findings:
- **MAJOR** D4-001: Standardize error response schema (RFC 7807 Problem+JSON). Currently returns JSON on 500 but different schemas on 401/413/415/429.
- **MAJOR** D4-002: Rate-limit response headers missing. Need `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` alongside `Retry-After`.
- **MINOR** D4-003: Auth error differentiation -- ensure 401 vs 403 semantics consistent.
- **MINOR** D4-004: API versioning strategy -- `X-API-Version` header exists but contract not documented.
- **MINOR** D4-005: Middleware ordering should be codified in a single app factory.

**D5: 9.0** -- "Excellent breadth: Hypothesis, chaos, load, security headers, encoding rejection."

Praised:
- Very strong test mix (2480+, Hypothesis fuzzing, chaos engineering, load tests)
- Security header tests on all error codes
- Content-Encoding rejection tests

Findings:
- **MINOR** D5-001: Add consumer-driven contract tests or OpenAPI schema validation.
- **MINOR** D5-002: Add post-deploy smoke tests against staging Cloud Run.
- **MINOR** D5-003: Add performance budgets (p95 latency regression thresholds).

**D6: 8.5** -- "Very solid container hygiene."

Praised:
- Multi-stage build + digest-pinned base image
- Non-root user, `--chown`, no curl, Python urllib HEALTHCHECK
- Graceful shutdown chain alignment

Findings:
- **MINOR** D6-001: Consider read-only filesystem, drop Linux capabilities.
- **MINOR** D6-002: CMD should use `${PORT:-8080}` for Cloud Run portability.
- **MINOR** D6-003: SBOM and image signing should be enforced in CI/CD, not just noted.

### DeepSeek V3.2 Speciale (D7, D8)

**D7: 9.8** -- "Robust and production-ready approach."

Praised:
- re2 adapter for ReDoS safety with fallback
- Extensive normalization pipeline (10x iterative URL decode, 2-pass HTML unescape, 136 confusables)
- Shared `_check_patterns` checking raw + normalized forms for ALL categories
- Semantic classifier with 3-failure degradation to restricted mode (not fail-open)
- Input size guard (8192 chars pre- and post-normalization)

Findings:
- **LOW** D7-001: NFKD + combining mark stripping may affect diacritic-heavy languages (Vietnamese, Arabic). Mitigated by checking raw input first. Monitor false positives.
- **LOW** D7-002: Input size limit of 8192 may restrict some long queries. Consider adjustable per use case.

**D8: 9.9** -- "Beyond many MVPs in resilience and production readiness."

Praised:
- Circuit breaker with Redis L1/L2 pipelining, bidirectional recovery, I/O outside lock
- asyncio.Semaphore(20) with acquired-flag CancelledError safety
- Redis Lua atomic rate limiter with in-memory fallback
- TTL jitter on 8+ caches
- SIGTERM drain chain (10s < 15s < 180s)
- Comprehensive observability (LangFuse, LangSmith, structured logging, metrics, alerts)
- Chaos tests (19) + load tests (4)

Findings:
- **LOW** D8-001: Redis CB sync_interval (2s) creates brief state inconsistency window. Acceptable trade-off.
- **LOW** D8-002: In-memory rate limit fallback allows higher effective rate under multi-instance when Redis is down.

### Grok 4 (D9, D10)

**D9: 9.5** -- "Comprehensive enough for a new engineer to onboard effectively."

Praised:
- 20 ADRs with clear lifecycle management
- Explicit supersession (ADR-016 by ADR-020)
- Supplementary docs (runbook, jurisdictional reference, output guardrails, onboarding checklist)
- Breadth covers architecture, operations, performance, and compliance

Findings:
- **MINOR** D9-001: Uniform 2026-02-25 review dates may signal incomplete reviews. Consider linking to review artifacts.

**D10: 9.8** -- "Nearly flawless for production. Deep, verifiable knowledge with no evident inaccuracies."

Praised:
- 5 casino profiles spanning tribal + commercial, 4 states
- Import-time completeness validation
- NGC Regulation 5.170 correction (from NRS 463.368 involuntary exclusion)
- Tribal vs commercial property_type distinction
- PROPERTY_STATE cross-validation
- Production secret validation
- Per-casino branding with safe_substitute templates

Findings:
- **MINOR** D10-001: Consider expanding to more profiles/states for broader coverage.
- **MINOR** D10-002: Explicit citations for regulatory data sources would strengthen audit readiness.

---

## Summary

**0 CRITICALs** across all 4 models. The codebase has reached a stable plateau above 93.

**Key gap**: D4 (API Design) at 7.5 is the weakest dimension. GPT-5.2 flagged missing RFC 7807 error schema standardization and rate-limit response headers. These are legitimate gaps that are straightforward to fix.

**Next actions to reach 95+**:
1. Implement RFC 7807 error envelope across all error responses (D4 +1.0)
2. Add rate-limit response headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`) (D4 +0.5)
3. Add OpenAPI contract tests (D5 +0.3)
4. Enforce SBOM + image signing in CI/CD (D6 +0.3)

Estimated impact: D4 7.5 -> 9.0 = +0.15 weighted, total ~94.7-95.0.

---

## Calibration Notes

This round used a strategy of sending COMPLETE files (not snippets) to each model. Results:
- Gemini: No false positives from truncated code. One minor finding was actually a false positive (function IS used, just imported elsewhere). Scores stable at 9.8-9.9.
- GPT-5.2: Consistent with its pattern of demanding RFC/standard compliance. D4 scoring reflects genuine API design gaps, not enterprise-grade demands.
- DeepSeek: Thorough analysis, no findings above LOW severity. Validated the normalization pipeline's correctness.
- Grok: Domain expertise confirmed. No regulatory inaccuracies found.

The 93.5 score is the most stable assessment to date, with clear actionable items for the remaining ~1.5 points to 95.
