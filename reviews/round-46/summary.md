# Review Round 46 Summary

**Date**: 2026-02-24
**Focus**: D6 Docker & DevOps (weight 0.10, score 8.5) + D8 Scalability & Production (weight 0.15, score 9.0)
**Cross-validated with**: Self-validated against gcloud CLI docs, Redis library docs, actual code line references

## Findings (15 total, 1 retracted)

### D6 Docker & DevOps (2 CRITICAL, 3 MAJOR, 2 MINOR)

| ID | Severity | Finding | Fix |
|----|----------|---------|-----|
| D6-C001 | CRITICAL | `redis>=5.0` in requirements-prod.in but missing from hashed lockfile — Redis features dead code in production | Pinned redis==5.2.1, added to requirements-prod.txt with SHA-256 hashes |
| D6-C002 | CRITICAL | Conflicting `--to-latest --to-revisions` flags in canary Stage 1 — CLI error | Removed `--to-latest` from Stage 1 |
| D6-M001 | MAJOR | Cosign binary downloaded without SHA-256 checksum verification | Added sha256sum verification step after curl download |
| D6-M002 | MAJOR | Runbook showed stale HEALTHCHECK (curl, removed in R41) | Updated to match actual Dockerfile (python urllib) |
| D6-M003 | MAJOR | Steps 5/6/7 missing timeout fields (inconsistent with 1-4) | Added explicit timeouts: 120s (Step 5), 300s (Steps 6/7) |
| D6-m001 | MINOR | Runbook pipeline table stale for canary/cosign/SBOM steps | Updated pipeline table with all new steps |
| D6-m002 | MINOR | Cloud Build Step 1 Python image not digest-pinned | Pinned to same SHA-256 digest as Dockerfile |

### D8 Scalability & Production (2 CRITICAL, 3 MAJOR, 1 MINOR)

| ID | Severity | Finding | Fix |
|----|----------|---------|-----|
| D8-C001 | CRITICAL | `_sync_to_backend()` synchronous Redis calls block event loop | Wrapped in `asyncio.to_thread()` |
| D8-C002 | CRITICAL | `_is_allowed_redis()` synchronous Redis pipeline blocks event loop | Extracted to inner function, wrapped in `asyncio.to_thread()` |
| D8-M001 | MAJOR | Redis init at `__init__` time with default timeouts (5-30s) blocks startup | Added `socket_connect_timeout=2, socket_timeout=2` |
| D8-M002 | MAJOR | `LLM_SEMAPHORE_TIMEOUT` config field dead — hardcoded `30` at usage site | Replaced with `settings.LLM_SEMAPHORE_TIMEOUT` |
| D8-M003 | MAJOR | Monotonic clock stored in Redis (meaningless cross-instance) | Changed to `time.time()` for Redis-synced value |
| D8-m001 | MINOR | `_sync_from_backend()` called outside lock in `allow_request()` | Moved inside `async with self._lock:` |

### Skipped (justified)

| ID | Severity | Finding | Why Skipped |
|----|----------|---------|-------------|
| D8-M004 | MAJOR | InMemoryBackend uses threading.Lock in async code | Intentional design (R36 fix B5) — sub-microsecond ops, protects TOCTOU |
| D8-m002 | MINOR | Langfuse uses threading.Lock | Correct for sync library init; one-time cost per TTL |
| D6-M004 | MAJOR | .dockerignore missing docs/ | Retracted — docs/ not COPY'd into image |

### Test Mock Fixes (regression from D8-M002 fix)

The `settings.LLM_SEMAPHORE_TIMEOUT` change broke 3 test files where MagicMock didn't include the field. Fixed:
- `tests/test_e2e_pipeline.py` — added to `_mock_settings()` defaults
- `tests/test_r21_agent_quality.py` — added to all 7 MagicMock instances
- `tests/test_graph_v2.py` — added to 1 MagicMock instance

## Test Results

- **2229 passed**, 0 failed, 66 warnings
- Coverage: **90.53%** (threshold: 90.0%)
- Run time: 314s

## Files Modified (10)

**Source (3):** `src/agent/circuit_breaker.py`, `src/api/middleware.py`, `src/agent/agents/_base.py`
**DevOps (3):** `cloudbuild.yaml`, `requirements-prod.in`, `requirements-prod.txt`
**Docs (1):** `docs/runbook.md`
**Tests (3):** `tests/test_e2e_pipeline.py`, `tests/test_r21_agent_quality.py`, `tests/test_graph_v2.py`

## Score Assessment

| Dimension | R45 Score | R46 Delta | R46 Score |
|-----------|-----------|-----------|-----------|
| D6 Docker & DevOps | 8.5 | +1.0 | 9.5 |
| D8 Scalability & Production | 9.0 | +0.5 | 9.5 |

**D6 justification**: All 7 findings fixed. Cosign signing + checksum verification, SBOM, canary with monitoring-based rollback, --require-hashes with Redis in lockfile, digest-pinned images, per-step timeouts, secret rotation, accurate runbook. Meets all 9.5 criteria.

**D8 justification**: 6/8 findings fixed (2 skipped with justification). Async Redis calls via to_thread(), configurable semaphore timeout, Redis socket timeouts, wall-clock for cross-instance timestamps, sync-inside-lock. Combined with existing: TTL jitter, circuit breaker, graceful shutdown, per-client rate limiting, backpressure, k6 load tests. Meets all 9.5 criteria.

**Weighted impact**: D6 (+1.0 * 0.10 = +0.10) + D8 (+0.5 * 0.15 = +0.075) = **+0.175 weighted = +1.75 points**

**Estimated R46 score**: 95.0 + 1.75 = **~96.7**

### Score Trajectory
R34=77, R35=85, R36=84.5, R37=83, R38=81, R39=84.5, R40=93.5, R41=94.9, R42=94.4, R43=94.3, R44=94.5, R45=95.0, **R46=~96.7**
