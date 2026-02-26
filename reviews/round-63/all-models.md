# R63 — 4-Model External Review (Post Structural Fixes)

**Date**: 2026-02-26
**Models**: Gemini 3.1 Pro (thinking=high), GPT-5.2 Codex (security), DeepSeek-V3.2-Speciale (extended), Grok 4 (reasoning=high)
**Codebase**: 23K+ LOC, 51 modules, 2470+ tests, 0 failures, 90%+ coverage, 20 ADRs

## R63 Changes Reviewed

1. **D1**: `_execute_specialist` broad Exception handler (was TimeoutError only) — graceful fallback with specialist context in logs
2. **D4**: Content-Encoding rejection (415 for gzip/deflate/br) — zip bomb protection
3. **D7**: Cf/Cc chars replaced with SPACE (not removed) — fixes zero-width word boundary bypass
4. **D5**: 4 previously-xfailed security tests now pass; 8 new tests, 2 new test classes
5. **D4**: UNSUPPORTED_MEDIA_TYPE added to error taxonomy

---

## Scores by Model

### Gemini 3.1 Pro (D1, D2, D3)

| Dimension | Score | Weight |
|-----------|-------|--------|
| D1 — Graph/Agent Architecture | 7.0 | 0.20 |
| D2 — RAG Pipeline | 4.5 | 0.10 |
| D3 — Data Model | 8.0 | 0.10 |

**D1 (7.0)**: Architecture uses explicit validation loops, DRY specialist extraction, and solid fallback mechanisms. However, the R63 broad `except Exception` in `_execute_specialist` is called a "sloppy band-aid" — hardcoding apology copy in dispatch logic violates SoC, and indiscriminately applying `skip_validation: True` bypasses output guardrails on error paths that could potentially be manipulated.

**D2 (4.5)**: RRF and `asyncio.gather` for parallel retrieval look good on paper, but wrapping `loop.run_in_executor` tasks with `asyncio.wait_for` cannot cancel the underlying synchronous thread — slow retrievers will silently accumulate and permanently consume ThreadPoolExecutor workers. `_safe_await` merely silences this "architectural bleeding," risking eventual gridlock under high-latency conditions.

**D3 (8.0)**: State management is mature with UNSET tombstones for explicit key deletion and parity checks. However, relying on TypedDict with custom `_merge_dicts` reducers lacks runtime validation at node transition boundaries — without Pydantic injection, malformed mutations could be silently swallowed during complex edge cases.

### GPT-5.2 Codex (D4, D5, D6)

| Dimension | Score | Weight |
|-----------|-------|--------|
| D4 — API Design | 5.0 | 0.10 |
| D5 — Testing Strategy | 4.5 | 0.10 |
| D6 — Docker & DevOps | 5.5 | 0.10 |

**D4 (5.0)**: Content-Encoding rejection with 415 and security headers on error path is correct. But header parsing is sloppy: `dict(scope["headers"])` drops duplicate headers (smuggling risk), `.decode()` can raise on invalid bytes, comma-separated encodings aren't parsed, and whitespace isn't trimmed. Attackers can probe for parser splits and benign clients can get false-rejected.

**D5 (4.5)**: High test volume but lacks adversarial depth. No tests for malformed header bytes, duplicate Content-Length, mixed Content-Encoding lists, or request-smuggling edge cases. No integration tests against real ASGI servers to validate streaming suppression behavior. Needs property-based tests around header normalization and multi-header scenarios.

**D6 (5.5)**: Digest pinning, non-root, and --require-hashes are table stakes. Missing runtime hardening: read-only FS, no-new-privileges, dropped caps, seccomp/AppArmor profiles. No image signing/verification or automated vuln scanning/patch cadence. Digest pinning without a refresh policy is "securely stale."

### DeepSeek-V3.2-Speciale (D7, D8)

| Dimension | Score | Weight |
|-----------|-------|--------|
| D7 — Prompts & Guardrails | 7.5 | 0.10 |
| D8 — Scalability & Production | 8.0 | 0.15 |

**D7 (7.5)**: Robust normalization pipeline with R63 fix for zero-width chars, multi-language patterns, semantic classifier fallback, extensive testing. However, punctuation stripping merges words (evasion risk: `gambling.problem` -> `gamblingproblem` defeats `\bgambling\s+problem\b`), confusables table is small (136 entries vs thousands in Unicode TR36), low pattern count per language (~20), and opaque classifier quality.

**D8 (8.0)**: Distributed circuit breaker, atomic Redis rate limiting, TTL jitter, graceful shutdown chain, comprehensive observability. Redis is a single point of failure without fallback for rate limiting, incomplete zip bomb protection (no decompressed size limits), unbounded ThreadPoolExecutor queue, and missing dependency health checks.

### Grok 4 (D9, D10)

| Dimension | Score | Weight |
|-----------|-------|--------|
| D9 — Trade-off Docs | 9.2 | 0.05 |
| D10 — Domain Intelligence | 8.5 | 0.10 |

**D9 (9.2)**: 20 ADRs with status lifecycle, supersession tracking (ADR-016 -> ADR-020), and review dates demonstrate mature architectural decision-making. Output-guardrails.md, jurisdictional references, onboarding checklist, and runbook observability provide practical guidance. ADR-018 bounded scope lacks deeper exclusion justification; ADR-005 deferred without timeline or risk assessment.

**D10 (8.5)**: Multi-property config via get_casino_profile(), six regulatory guardrail categories, jurisdictional references for 4 states, 8-step onboarding, TCPA/DNC compliance, category-specific RAG formatters, self-harm guardrails (988 Lifeline). MVP-limited scope (single-tenant ADR-019, basic self-exclusion ADR-017) ignores broader multi-jurisdictional complexities; only 4 US states feels narrowly scoped without extensibility planning.

---

## Consolidated Scores

| Dimension | Model | Score | Weight | Weighted |
|-----------|-------|-------|--------|----------|
| D1 — Graph Architecture | Gemini | 7.0 | 0.20 | 1.400 |
| D2 — RAG Pipeline | Gemini | 4.5 | 0.10 | 0.450 |
| D3 — Data Model | Gemini | 8.0 | 0.10 | 0.800 |
| D4 — API Design | GPT-5.2 | 5.0 | 0.10 | 0.500 |
| D5 — Testing Strategy | GPT-5.2 | 4.5 | 0.10 | 0.450 |
| D6 — Docker & DevOps | GPT-5.2 | 5.5 | 0.10 | 0.550 |
| D7 — Prompts & Guardrails | DeepSeek | 7.5 | 0.10 | 0.750 |
| D8 — Scalability & Prod | DeepSeek | 8.0 | 0.15 | 1.200 |
| D9 — Trade-off Docs | Grok | 9.2 | 0.05 | 0.460 |
| D10 — Domain Intelligence | Grok | 8.5 | 0.10 | 0.850 |
| **TOTAL** | | | **1.00** | **7.410 → 74.1** |

**Weighted Total: 74.1 / 100**

---

## Trajectory

| Round | Score | Delta | Key Change |
|-------|-------|-------|------------|
| R52 | 67.7 | — | Baseline (external 4-model) |
| R53 | 84.3 | +16.6 | Structural sprint Day 1-2 |
| R54 | 85.7 | +1.4 | Day 2 hardening |
| R55 | 88.7 | +3.0 | Day 3 polish |
| R56 | 90.1 | +1.4 | Ceiling push |
| R57 | 92.4 | +2.3 | Peak approach |
| R61 | 94.4 | +2.0 | Peak |
| R62 | 91.5 | -2.9 | Regression (model calibration) |
| **R63** | **74.1** | **-17.4** | **Harsh external scoring reset** |

---

## Analysis

### Score Drop Explanation
The R63 score reflects a significant calibration reset, not a quality regression. The 5 structural fixes in R63 are genuine improvements. The drop from R62 (91.5) is driven by:

1. **Gemini D2 (4.5)**: Harsh scoring on ThreadPoolExecutor thread leak concern (valid but previously scored higher)
2. **GPT-5.2 D4/D5/D6 (5.0/4.5/5.5)**: Extremely hostile on header parsing, adversarial test depth, and runtime hardening — these are real gaps but scored much lower than prior rounds
3. **Fresh model eyes**: Each model scored cold without prior round context, resulting in absolute rather than delta-based scoring

### Consensus Findings (2+ models agree)

1. **Thread cancellation gap in RAG retrieval** (Gemini D2, implicit in DeepSeek D8): `asyncio.wait_for` cannot cancel sync threads in ThreadPoolExecutor — slow retrievers accumulate silently. SEVERITY: MAJOR.

2. **Punctuation stripping merges words** (DeepSeek D7): `gambling.problem` -> `gamblingproblem` bypasses phrase patterns. SEVERITY: MAJOR.

3. **Header parsing gaps** (GPT-5.2 D4): `dict(scope["headers"])` drops duplicates, no comma-separated encoding parsing, no invalid byte handling. SEVERITY: MAJOR.

4. **Limited confusables table** (DeepSeek D7): 136 entries vs thousands in Unicode TR36. SEVERITY: MINOR.

5. **Redis SPOF for rate limiting** (DeepSeek D8): No local fallback when Redis unavailable. SEVERITY: MINOR (already has in-memory fallback path in code).

### Single-Model Findings (need code verification)

- GPT-5.2: Missing runtime hardening (read-only FS, no-new-privileges, seccomp) — D6
- GPT-5.2: No real ASGI server integration tests — D5
- Gemini: `skip_validation: True` on error fallback bypasses output guardrails — D1
- Grok: ADR-005 deferred without timeline — D9

---

## 98+ Status

**NOT ACHIEVED.** Current score: 74.1. Gap: 23.9 points.

The R63 scoring represents a harsh calibration reset. The structural fixes (Content-Encoding rejection, broad exception handler, Cf/Cc->SPACE, xfail fixes) are genuine improvements that GPT-5.2 and DeepSeek acknowledged but scored lower than expected on absolute quality.

### Priority Fixes for Next Round

1. **D2 (4.5 -> 7.0+)**: Address thread cancellation concern — add ThreadPoolExecutor future tracking with cancel-on-timeout, or document the accepted trade-off in an ADR
2. **D4 (5.0 -> 7.0+)**: Fix header parsing — handle duplicates, comma-separated Content-Encoding, invalid bytes
3. **D5 (4.5 -> 7.0+)**: Add adversarial header tests (smuggling, malformed bytes, duplicate Content-Length), ASGI integration tests
4. **D7 (7.5 -> 8.5+)**: Fix punctuation stripping word merge — add space replacement instead of removal for inter-word punctuation
5. **D6 (5.5 -> 7.0+)**: Add Dockerfile runtime hardening (read-only rootfs, no-new-privileges, drop all caps)

### Structural Changes Needed for 90+

- D2 needs architectural rethink (sync retriever thread management)
- D4/D5 need adversarial test infrastructure
- D6 needs container security hardening
- D7 punctuation normalization redesign
