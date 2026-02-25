# R58 — 4-Model Review Results

**Date**: 2026-02-25
**Codebase**: 2462 tests, 0 failures, 90.6% coverage, 20 ADRs
**R58 Fix Applied**: Retrieval pool lifespan shutdown hook in app.py

---

## Scores by Model

| Dimension | Weight | Gemini 3 Pro | GPT-5.2 Codex | DeepSeek V3.2 | Grok 4 | Consensus |
|-----------|--------|-------------|---------------|---------------|--------|-----------|
| D1 Graph Architecture | 0.20 | 9.5 | — | — | — | 9.5 |
| D2 RAG Pipeline | 0.10 | 7.0 | — | — | — | 7.0 |
| D3 Data Model | 0.10 | 9.5 | — | — | — | 9.5 |
| D4 API Design | 0.10 | — | 9.4 | — | — | 9.4 |
| D5 Testing Strategy | 0.10 | — | 9.6 | — | — | 9.6 |
| D6 Docker & DevOps | 0.10 | — | 8.8 | — | — | 8.8 |
| D7 Prompts & Guardrails | 0.10 | — | — | 9.0 | — | 9.0 |
| D8 Scalability & Prod | 0.15 | — | — | 9.5 | — | 9.5 |
| D9 Trade-off Docs | 0.05 | — | — | — | 9.0 | 9.0 |
| D10 Domain Intelligence | 0.10 | — | — | — | 9.0 | 9.0 |

## Weighted Total

```
D1:  9.5 × 0.20 = 1.900
D2:  7.0 × 0.10 = 0.700
D3:  9.5 × 0.10 = 0.950
D4:  9.4 × 0.10 = 0.940
D5:  9.6 × 0.10 = 0.960
D6:  8.8 × 0.10 = 0.880
D7:  9.0 × 0.10 = 0.900
D8:  9.5 × 0.15 = 1.425
D9:  9.0 × 0.05 = 0.450
D10: 9.0 × 0.10 = 0.900
─────────────────────────
TOTAL:              10.005 → 100.1 (capped at 100)
                    ACTUAL: ~92.1 (weighted average = 9.21)
```

Wait — let me recalculate. The weights sum to 1.10, not 1.00 (the rubric has 10 dims with given weights that sum to 1.10).

Corrected (weights sum to 1.10, so divide by 1.10 for normalized score or just report raw weighted sum × 100/11):

Actually, looking at the rubric: weights are 0.20+0.10+0.10+0.10+0.10+0.10+0.10+0.15+0.05+0.10 = 1.10. The standard approach is weighted sum / sum of weights × 10:

```
Raw weighted sum = 10.005
Normalized = 10.005 / 1.10 = 9.095... → 90.95/100
```

**R58 WEIGHTED SCORE: 91.0/100**

---

## Trajectory

| Round | Score | Delta |
|-------|-------|-------|
| R52 | 67.7 | — |
| R53 | 84.3 | +16.6 |
| R54 | 85.7 | +1.4 |
| R55 | 88.7 | +3.0 |
| R56 | 90.1 | +1.4 |
| R57 | 92.4 | +2.3 |
| R58 | 91.0 | -1.4 |

**Note**: The -1.4 dip is driven by Gemini scoring D2 at 7.0 (down from implicit 6.0→7.0 improvement but still an anchor due to ThreadPoolExecutor/asyncio.to_thread layering concern) and GPT-5.2 scoring D6 at 8.8 (SBOM gap). The D2 concern about "event loop thread starvation" is partially valid for the dev path (ChromaDB is sync) but does NOT apply to production (Vertex AI Vector Search is async-native per ADR-006). See analysis below.

---

## Findings by Model

### Gemini 3 Pro (D1/D2/D3)

**D1 = 9.5** (+0.5 from R57)
- MAJOR: Unhandled exceptions in _execute_specialist (dispatch.py:140). TimeoutError caught but generic Exception not caught — could crash the graph. *Assessment: Valid but low-probability since agent_fn itself has full exception handling in _base.py. The outer timeout is a safety net.*
- MINOR: Unsafe dict unpacking if _inject_guest_context returns None. *Assessment: Valid theoretically; the function has try/except returning {} on all error paths.*

**D2 = 7.0** (+1.0 from R57)
- CRITICAL (disputed): Event loop thread starvation — asyncio.to_thread uses default executor (8-12 threads), then submits to _RETRIEVAL_POOL. Only 12 concurrent retrievals possible.
  - **REBUTTAL**: This concern applies to the ChromaDB dev path ONLY. In production, ADR-006 mandates Vertex AI Vector Search which is async-native. The ThreadPoolExecutor wraps ChromaDB's synchronous .similarity_search() for local dev. Production path uses async retriever.ainvoke() directly. The "eliminate ThreadPoolExecutor" recommendation is already the production architecture.
  - **Residual concern**: The dev path does have this layering issue, but it only affects local development with max_workers capped by default executor size. This is a valid MINOR for dev ergonomics, not a CRITICAL for production.

**D3 = 9.5** (+1.0 from R57)
- MINOR: guest_context lacks Annotated[..., _merge_dicts] reducer — full overwrite risk if any future node returns partial guest_context. *Assessment: Valid improvement opportunity.*

### GPT-5.2 Codex (D4/D5/D6)

**D4 = 9.4** (+0.8 from R57)
- No CRITICALs or MAJORs.
- MINOR: Consider publishing formal error schema in API docs.

**D5 = 9.6** (+0.6 from R57)
- No CRITICALs or MAJORs.
- MINOR: Ensure chaos/load tests run in CI on schedule.

**D6 = 8.8** (-0.7 from R57)
- MAJOR: Missing SBOM generation. *Assessment: Valid — no SBOM step documented. Add `pip-audit` or `syft` to CI.*
- MINOR: Document graceful shutdown contract in deployment runbooks. *Assessment: Already documented in Dockerfile comments and runbook, but could be more prominent.*

### DeepSeek V3.2 (D7/D8)

**D7 = 9.0** (-0.5 from R57)
- MEDIUM: Confusable table limited to 110 entries — comprehensive Unicode confusables data has thousands. *Assessment: Valid but scoped intentionally per ADR-018 (bounded scope). Adding all confusables would create false positives.*
- LOW: No case folding in normalization. *Assessment: Patterns use re.I flag; case folding is redundant.*
- LOW: Only inter-word punctuation removed. *Assessment: Boundary punctuation is handled by regex word boundary patterns.*
- LOW: URL decode loop limit of 10 iterations. *Assessment: 10 iterations handles realistic encoding depth; infinite loop risks DoS.*
- INFO: Double html.unescape. *Assessment: Intentional two-pass design documented in R52 fix comment.*

**D8 = 9.5** (unchanged from R57)
- LOW: Failure count not reset when entering half_open state. *Assessment: This is intentional — half_open decay (halve instead of clear) was a deliberate design choice per R35 fix.*
- LOW: Remote state read outside lock may cause stale updates. *Assessment: Acknowledged as by-design (R47/R49 fix pattern).*

### Grok 4 (D9/D10)

**D9 = 9.0** (+0.5 from R57)
- MEDIUM: All ADR review dates are the same (2026-02-25). Stagger reviews quarterly. *Assessment: Valid — dates reflect bulk review day, not ongoing cadence.*

**D10 = 9.0** (+0.5 from R57)
- LOW: Only 4 jurisdictions covered; no federal references (e.g., UIGEA). *Assessment: Valid for future expansion. Current 4 states match the 5 casino profiles (4 real + 1 demo).*

---

## Remaining Actionable Items (for 95+)

| Priority | Finding | Source | Fix Effort |
|----------|---------|--------|------------|
| 1 | Add SBOM generation to CI | GPT D6 | Low — add `pip-audit` step |
| 2 | Add Annotated reducer to guest_context | Gemini D3 | Low — 1 line change |
| 3 | Add generic Exception catch in _execute_specialist | Gemini D1 | Low — 3 lines |
| 4 | Stagger ADR review dates | Grok D9 | Low — update dates |
| 5 | Add federal regulatory overview to jurisdictional ref | Grok D10 | Medium — research needed |
| 6 | Expand confusable table or document why bounded | DeepSeek D7 | Low — already ADR-018 |

---

## 98+ Status

**Not reached.** Current consensus: 91.0/100. The main anchors are:
- D2 at 7.0 (Gemini's threading concern — partially valid for dev path)
- D6 at 8.8 (SBOM gap)
- D7 at 9.0 (confusable scope)

To reach 95+: Fix items 1-3 above, which would push D2→8.0, D6→9.5, D3→9.7, yielding ~93.5. For 98+, would need D2→9.5 which requires migrating dev retrieval to async (significant effort).
