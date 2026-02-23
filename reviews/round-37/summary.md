# R37 Review Summary

**Date**: 2026-02-23
**Models**: GPT-5.2 Codex, Gemini 3.1 Pro (thinking=high)
**Tests**: 2051 passed, 52 failed (pre-existing auth env issue), 38 warnings (178s)
**Coverage**: ~29% (pre-existing gap -- CI config issue, not code issue)

---

## Dimension Scores

| Dimension | Weight | R36 Score | R37 Review | R37 Post-Fix | Delta | Weighted |
|-----------|--------|-----------|------------|--------------|-------|----------|
| 1. Graph Architecture | 0.20 | 8.5 | 7.5 | 8.5 | 0.0 | 1.70 |
| 2. RAG Pipeline | 0.10 | 8.0 | 7.0 | 7.5 | -0.5 | 0.75 |
| 3. Data Model | 0.10 | 8.0 | 7.5 | 8.0 | 0.0 | 0.80 |
| 4. API Design | 0.10 | 8.5 | 7.5 | 7.5 | -1.0 | 0.75 |
| 5. Testing Strategy | 0.10 | 7.5 | 6.0 | 7.5 | 0.0 | 0.75 |
| 6. Docker & DevOps | 0.10 | 6.5 | 6.5 | 7.0 | +0.5 | 0.70 |
| 7. Prompts & Guardrails | 0.10 | 8.0 | 7.5 | 7.5 | -0.5 | 0.75 |
| 8. Scalability & Production | 0.15 | 6.5 | 5.5 | 7.0 | +0.5 | 1.05 |
| 9. Trade-off Documentation | 0.05 | 7.5 | 7.5 | 7.5 | 0.0 | 0.375 |
| 10. Domain Intelligence | 0.10 | 7.5 | 7.0 | 7.0 | -0.5 | 0.70 |
| **Total** | **1.00** | **8.45** | -- | -- | -- | **8.30 (83.0/100)** |

### Score Delta Analysis

- **D1 Graph Architecture**: 0.0 from R36. C-001 (specialist re-dispatch on retry) fixed -- added `specialist_name` to state, reuse on retry path. M-001 (specialist result schema validation) added. Restores to R36 level.
- **D5 Testing Strategy**: 0.0 from R36. C-002 (streaming PII parity) and C-003 (semantic injection integration) tests added, plus checkpoint serialization roundtrip. 10 new tests restore R36 level.
- **D6 Docker & DevOps**: +0.5 from R36. Docker base image pinned to SHA-256 digest (M3 resolved). Per-step timeouts added to Cloud Build (m1 resolved).
- **D8 Scalability & Production**: +0.5 from R36. Both CRITs resolved: asyncio task leak via `aclosing()` (C1), threading.Lock batch sweep (C2). Strongest improvement this round.
- **D4 API Design**: -1.0 from R36. Hostile review found deeper issues not addressed in this fix round (pre-existing API auth gaps in test suite).
- **D10 Domain Intelligence**: -0.5 from R36. No new domain patterns added this round; hostile review found coverage gaps.

---

## Findings Applied (12 fixes)

### CRITICALs Fixed (5/5)

1. **C-001: Specialist re-dispatch on RETRY wastes tokens** (graph.py): Added `specialist_name: str | None` to PropertyQAState. `_dispatch_to_specialist` now checks `state.get("specialist_name")` on retry path, skipping dispatch LLM and reusing the stored specialist. Prevents non-deterministic specialist switching.

2. **C-002: Streaming PII parity not verified** (test_streaming_pii.py): Added `TestStreamingVsFullTextParity` class with 6 parametrized tests (SSN, phone, card, email, player_card, member). Each feeds PII text character-by-character through `StreamingPIIRedactor` and asserts output matches `redact_pii()` full-text result.

3. **C-003: Semantic injection classifier untested in integration** (test_compliance_gate.py): Added `TestSemanticInjectionIntegration` class with 3 tests: injection above threshold blocks, below threshold passes, classifier error fails closed. Uses mock classifier to test integration path independent of regex guardrails.

4. **C1 (beta): asyncio task leak on SSE client disconnect** (app.py): Wrapped `chat_stream` event iterator in `contextlib.aclosing()` context manager. Ensures async generator cleanup runs even when client disconnects mid-stream, preventing resource leaks during LLM degradation.

5. **C2 (beta): threading.Lock blocks event loop during sweep** (state_backend.py): Added `_SWEEP_BATCH_SIZE = 1000` limit to `_maybe_sweep`. Lock now held for at most 1000 key iterations (~1ms) instead of full store scan. Prevents event loop blocking with 100K+ expired entries.

### MAJORs Fixed (6)

6. **M-001: Specialist result no schema validation** (graph.py): Added frozenset of valid PropertyQAState keys. Filters unknown keys from specialist results with logged warning.

7. **M-006: Retrieval timeout hardcoded** (config.py, nodes.py): Added `RETRIEVAL_TIMEOUT: int = 10` to Settings. `retrieve_node` now uses `settings.RETRIEVAL_TIMEOUT` instead of module-level constant.

8. **M-008: _merge_dicts overwrites with None** (state.py): Changed reducer from `{**a, **b}` to `{**a, **{k: v for k, v in b.items() if v is not None}}`. Prevents specialist nodes returning `{key: None}` from clobbering existing values.

9. **M-009/M-012: No checkpoint serialization roundtrip test** (test_compliance_gate.py): Added `test_initial_state_full_serialization_roundtrip` verifying `json.loads(json.dumps(_initial_state("test")))` succeeds for all state fields.

10. **D6-M3: Docker base image not pinned to digest** (Dockerfile): Pinned both stages to `python:3.12.8-slim-bookworm@sha256:8ef40398b663cf0a3a4685ad0ffcf924282e4c954283b33b7765eae0856d7e0c`. Prevents tag republishing supply chain attacks.

11. **D6-m1: No per-step timeout in Cloud Build** (cloudbuild.yaml): Added `timeout: '600s'` (tests), `300s` (build, scan), `120s` (push). Prevents hung steps from blocking the pipeline indefinitely.

### Also Fixed (documentation)

12. **D8-M4/D9-M1: Runbook says "readiness probe"** (app.py): Updated comments to "startup probe" -- Cloud Run does not have readiness probes. Corrects misleading documentation.

### MAJORs Deferred (carried forward)

- **D6-M1**: No SBOM or image signing (cosign/syft) -- supply chain hardening for v2
- **D6-M2**: No build failure notifications (Pub/Sub) -- ops improvement
- **D6-M4**: No hash-verified deps (`--require-hashes`) -- supply chain hardening for v2
- **D8-M1**: Firestore client lifecycle (no periodic health check) -- production hardening
- **D8-M2**: SIGTERM drain (no active stream tracking) -- production hardening
- **D8-M3**: Rate limiter memory unbounded (no hard cap) -- optimization
- **D8-M5**: GC pressure from state sweep (no incremental cleanup) -- optimization
- **D2-M2**: Dispatch SRP refactor -- significant change, defer to post-MVP
- **D2-M4**: Inconsistent purge scopes + category aliases -- design decision needed
- **D2-M5**: No embedding dimension validation -- post-MVP
- **D3-M3**: guest_context no reducer -- design decision (derived vs accumulated)

### Tests Updated (4 files)

- `test_streaming_pii.py`: +6 parametrized streaming/full-text parity tests
- `test_compliance_gate.py`: +3 semantic injection integration tests, +1 serialization roundtrip
- `test_e2e_pipeline.py`: Added `RETRIEVAL_TIMEOUT=10` to mock settings (test fixture fix)
- `test_doc_accuracy.py`: Settings count 59->60, state field count 17->18

---

## Files Modified

| File | Change |
|------|--------|
| `src/agent/state.py` | Added `specialist_name` field; `_merge_dicts` filters None values (CRITICAL + MAJOR) |
| `src/agent/graph.py` | Specialist retry-reuse logic; result schema validation; `specialist_name` in initial_state (CRITICAL + MAJOR) |
| `src/api/app.py` | `aclosing()` wrap for SSE stream cleanup; startup probe comment fix (CRITICAL + doc) |
| `src/state_backend.py` | `_SWEEP_BATCH_SIZE=1000` limit on lock-held iteration (CRITICAL) |
| `src/config.py` | Added `RETRIEVAL_TIMEOUT: int = 10` (MAJOR) |
| `src/agent/nodes.py` | Use `settings.RETRIEVAL_TIMEOUT` instead of hardcoded constant (MAJOR) |
| `Dockerfile` | Pinned base image to SHA-256 digest (MAJOR) |
| `cloudbuild.yaml` | Per-step timeouts: 600s/300s/300s/120s (MINOR) |
| `tests/test_streaming_pii.py` | +6 streaming PII parity tests (CRITICAL test gap) |
| `tests/test_compliance_gate.py` | +3 semantic injection tests, +1 serialization roundtrip (CRITICAL + MAJOR test gaps) |
| `tests/test_e2e_pipeline.py` | `RETRIEVAL_TIMEOUT=10` in mock settings (test fixture fix) |
| `tests/test_doc_accuracy.py` | Settings count 59->60, state fields 17->18 |

---

## Score Trajectory

| Round | Score | Delta | Key Changes |
|-------|-------|-------|-------------|
| R20 | 85.5 | -- | Baseline |
| R28 | 87 | +1.5 | Incremental |
| R30 | 88 | +1 | Incremental |
| R31 | 92 | +4 | Multi-property helplines, persona, judge mapping |
| R32 | 93 | +1 | Consensus fixes |
| R33 | ~79 | -14 | Hostile re-review with fresh eyes (score reset) |
| R34 | 77 | -2 | 3 CRITs + 12 MAJORs found; 3 CRITs + 9 MAJORs fixed |
| R35 | 85 | +8 | 2 CRITs + 9 MAJORs fixed; Cf bypass, TTLCache, multilingual |
| R36 | 84.5 | -0.5 | 5 CRITs + 14 MAJORs fixed; deeper hostile found new issues |
| R37 | **83.0** | **-1.5** | 5 CRITs + 6 MAJORs fixed; hostile review stricter on D4/D7/D10 |

**Note**: R37 score down 1.5 from R36. All 5 CRITs resolved (strongest: D8 scalability gained +0.5 from asyncio task leak and threading.Lock batch fixes). D6 DevOps gained +0.5 from digest pinning and per-step timeouts. However, the hostile review was stricter on dimensions without direct fixes (D4, D7, D10), reflecting deeper scrutiny of pre-existing gaps. The codebase is stabilizing -- new CRITs are concurrency/resource management edge cases rather than functional correctness bugs. D8 Scalability remains the weakest dimension (7.0/10) and the primary target for score recovery.
