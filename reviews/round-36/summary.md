# R36 Review Summary

**Date**: 2026-02-23
**Models**: GPT-5.2 Codex, Gemini 3 Pro (thinking=high)
**Tests**: 2055 passed, 52 failed (pre-existing auth env issue), 64 warnings (284.54s)
**Coverage**: ~29% (pre-existing gap — CI config issue, not code issue)

---

## Dimension Scores

| Dimension | Weight | R35 Score | R36 Review | R36 Post-Fix | Delta | Weighted |
|-----------|--------|-----------|------------|--------------|-------|----------|
| 1. Graph Architecture | 0.20 | 8.5 | 8.0 | 8.5 | 0.0 | 1.70 |
| 2. RAG Pipeline | 0.10 | 8.0 | 7.5 | 8.0 | 0.0 | 0.80 |
| 3. Data Model | 0.10 | 8.5 | 8.0 | 8.0 | -0.5 | 0.80 |
| 4. API Design | 0.10 | 8.0 | 7.5 | 8.5 | +0.5 | 0.85 |
| 5. Testing Strategy | 0.10 | 7.5 | 7.0 | 7.5 | 0.0 | 0.75 |
| 6. Docker & DevOps | 0.10 | 6.5 | 6.5 | 6.5 | 0.0 | 0.65 |
| 7. Prompts & Guardrails | 0.10 | 7.5 | 7.0 | 8.0 | +0.5 | 0.80 |
| 8. Scalability & Production | 0.15 | 6.5 | 5.5 | 6.5 | 0.0 | 0.975 |
| 9. Trade-off Documentation | 0.05 | 7.5 | 7.5 | 7.5 | 0.0 | 0.375 |
| 10. Domain Intelligence | 0.10 | 8.0 | 7.0 | 7.5 | -0.5 | 0.75 |
| **Total** | **1.00** | **8.50** | — | — | — | **8.45 (84.5/100)** |

### Score Delta Analysis

- **D4 API Design**: +0.5 from R35. CSP nonce security theater resolved (3-round carry CLOSED), security header divergence unified, 401 response now includes security headers, rate limiter time-based sweep added.
- **D7 Guardrails**: +0.5 from R35. Normalization order corrected (NFKD before confusable replacement), 7 IPA/Latin Extended confusables added, input length DoS prevention, audit_input removed from public API.
- **D3 Data Model**: -0.5 from R35. ValidationResult max_length added, but guest_context reducer remains unresolved (design decision needed).
- **D10 Domain**: -0.5 from R35. JP/KO BSA+RG patterns added, self_exclusion_options surfaced, but knowledge base JSON gap flagged.

---

## Findings Applied (19 fixes)

### CRITICALs Fixed (5/5)

1. **get_casino_profile() mutable reference for KNOWN casinos** (config.py:551-564): R35 only fixed DEFAULT_CONFIG fallback. Now returns `copy.deepcopy(profile)` for ALL paths. Prevents cross-request global state corruption.

2. **router_node TOCTOU — 2x get_settings()** (nodes.py:215,223): Hoisted `settings = get_settings()` once, reused for both `sentiment_detection_enabled` and `field_extraction_enabled` feature flag checks. Same pattern as R35 fix in `_dispatch_to_specialist`.

3. **chunk_id delimiter-free hash collision** (pipeline.py:240): Changed `text + source` to `f"{text}\x00{source}"` — null byte delimiter prevents ambiguous concatenation where "abc"+"def" == "abcde"+"f".

4. **RRF identity hash same delimiter issue** (reranking.py:45-47): Applied same null-byte delimiter fix as pipeline.py for consistency.

5. **CSP nonce security theater — 3 rounds carried** (middleware.py:176-218): Removed per-request nonce generation entirely. Backend is API-only (JSON responses), no server-rendered HTML. CSP now uses static policy without nonce. Documented rationale in class docstring.

### MAJORs Fixed (14)

6. **cb.record_success() before registry validation** (graph.py:223-226): Moved `record_success()` to AFTER `result.specialist in _AGENT_REGISTRY` check. Unknown specialist names no longer inflate CB health metrics.

7. **CB halving integer division edge case** (circuit_breaker.py:255): Changed `keep_count = len // 2` to `max(len // 2, 1)` — always retains at least 1 failure timestamp after half-open recovery. Prevents 1-failure case from full clearing.

8. **Normalization order: confusable before NFKD** (guardrails.py:374-376): Swapped order to NFKD -> remove combining marks -> confusable replacement. Precomposed accented Cyrillic/Greek characters now correctly decompose before confusable mapping.

9. **Confusable table missing IPA/Latin Extended** (guardrails.py:349-357): Added 7 highest-risk confusables: `ɑ`->a, `ɡ`->g, `ı`->i, `ɪ`->i, `ʏ`->y, `ɴ`->n, `ʀ`->r. These survive NFKD normalization and bypass the existing 70-char table.

10. **No input length limit before normalization** (guardrails.py:451): Added `if len(message) > 8192: return False` (block) at top of `_audit_input()`. Prevents CPU exhaustion via 5 O(n) Unicode passes on 1MB+ payloads.

11. **audit_input() in __all__ with inverted semantics** (guardrails.py:25-27): Removed from `__all__`, renamed internal function to `_audit_input`, kept backward-compatible alias. External callers should use `detect_prompt_injection()` only.

12. **InMemoryBackend no concurrency protection** (state_backend.py:58-121): Added `threading.Lock` wrapping all read-modify-write operations (increment, get_count, set, get, exists, delete). Prevents TOCTOU races in increment() under concurrent coroutines.

13. **Security header divergence between middleware** (middleware.py:117-122): Unified ErrorHandlingMiddleware headers with SecurityHeadersMiddleware. Removed deprecated `x-xss-protection`, added HSTS for parity.

14. **401 response missing security headers** (middleware.py:280-283): ApiKeyMiddleware 401 response now includes `SecurityHeadersMiddleware._STATIC_HEADERS`.

15. **Rate limiter time-based sweep fallback** (middleware.py:429): Added `self._last_sweep` timestamp and `time_since_sweep > 300` condition. Under low traffic (<100 req/5min), stale clients now cleaned every 300s instead of accumulating indefinitely.

16. **No Korean/Japanese BSA/AML patterns** (guardrails.py:270-276): Added 6 patterns: Japanese (マネーロンダリング, お金を隠す, 現金報告避ける) + Korean (돈세탁, 돈을 숨기, 현금보고피하).

17. **No Korean/Japanese responsible gaming patterns** (guardrails.py:183-192): Added 6 patterns: Japanese (ギャンブル依存, パチンコ中毒, 賭け事をやめ) + Korean (도박중독, 도박을 그만, 도박문제).

18. **Hard Rock AC self_exclusion_options not surfaced** (prompts.py:59-62): `get_responsible_gaming_helplines()` now includes `self_exclusion_options` when present in config. NJ DGE requires informing guests of 1-year/5-year/lifetime options.

19. **ValidationResult.reason no max_length** (state.py:191-193): Added `max_length=500` to prevent arbitrarily long validator reasons from bloating retry feedback system prompts.

### Also Fixed (documentation)

20. **compliance_gate.py docstring**: Updated ~173 -> ~185 patterns.
21. **Runbook VERSION**: Fixed 1.0.0 -> 1.1.0 (matching config.py default).
22. **Runbook responsible gaming**: Updated from CT-only example to multi-state helplines.

### MAJORs Deferred (carried forward)

- **A2/R34-A2**: Dispatch SRP refactor — significant change, defer to post-MVP
- **A6/R34-A4**: Inconsistent purge scopes + category aliases — design decision needed (3rd round)
- **A7/R34-A5**: No embedding dimension validation — 3rd round, post-MVP
- **A9**: guest_context no reducer — design decision (derived vs accumulated)
- **A10**: _merge_dicts shallow merge — currently safe (flat structure)
- **A12**: Rate limiter background task — optimization, not correctness
- **A15**: Only 5 hypothesis tests — incremental improvement
- **B1-B4 DevOps**: No SBOM, no build notifications, no digest pinning, no hash-verified deps (supply chain hardening for v2)
- **B2 (beta)**: Knowledge base markdown-only vs JSON — verify CLAUDE.md accuracy

### Tests Updated (3 files)

- `test_middleware.py`: CSP nonce tests -> static CSP tests (3 tests rewritten)
- `test_doc_accuracy.py`: Pattern counts 173->185, RG 54->60, BSA 41->47
- `test_casino_config.py`: Identity check -> equality + non-identity for deepcopy

---

## Files Modified

| File | Change |
|------|--------|
| `src/casino/config.py` | deepcopy for known casinos too (CRITICAL) |
| `src/agent/nodes.py` | Settings TOCTOU hoist in router_node (CRITICAL) |
| `src/rag/pipeline.py` | Null-byte delimiter in chunk_id hash (CRITICAL) |
| `src/rag/reranking.py` | Null-byte delimiter in RRF identity hash |
| `src/api/middleware.py` | CSP nonce removal, header unification, 401 headers, time-based sweep |
| `src/agent/graph.py` | CB record_success after registry validation |
| `src/agent/circuit_breaker.py` | CB halving min(1) edge case |
| `src/agent/guardrails.py` | Normalization order, IPA confusables, length limit, JP/KO BSA+RG, audit_input |
| `src/state_backend.py` | threading.Lock for InMemoryBackend |
| `src/agent/state.py` | ValidationResult.reason max_length=500 |
| `src/agent/prompts.py` | self_exclusion_options in helpline output |
| `src/agent/compliance_gate.py` | Docstring pattern count 173->185 |
| `docs/runbook.md` | VERSION 1.0.0->1.1.0, multi-state helpline example |
| `tests/test_middleware.py` | CSP nonce tests -> static CSP tests |
| `tests/test_doc_accuracy.py` | Pattern count assertions updated |
| `tests/test_casino_config.py` | Deepcopy assertion updated |

---

## Score Trajectory

| Round | Score | Delta | Key Changes |
|-------|-------|-------|-------------|
| R20 | 85.5 | — | Baseline |
| R28 | 87 | +1.5 | Incremental |
| R30 | 88 | +1 | Incremental |
| R31 | 92 | +4 | Multi-property helplines, persona, judge mapping |
| R32 | 93 | +1 | Consensus fixes |
| R33 | ~79 | -14 | Hostile re-review with fresh eyes (score reset) |
| R34 | 77 | -2 | 3 CRITs + 12 MAJORs found; 3 CRITs + 9 MAJORs fixed |
| R35 | 85 | +8 | 2 CRITs + 9 MAJORs fixed; Cf bypass, TTLCache, multilingual |
| R36 | **84.5** | **-0.5** | 5 CRITs + 14 MAJORs fixed; deeper hostile review found new issues offsetting fixes |

**Note**: R36 score essentially flat (-0.5 from R35). This round found 7 NEW CRITICALs/MAJORs not present in R35 (chunk_id collision, CB success timing, normalization order, IPA confusables, input length DoS, InMemoryBackend TOCTOU, security header divergence). All were fixed, but the hostile posture also uncovered more carried items in D6 DevOps (supply chain) and D10 Domain (JP/KO coverage), holding those dimensions at their R35 levels. The codebase is stabilizing — new findings are increasingly edge-case and defense-in-depth rather than functional bugs.
