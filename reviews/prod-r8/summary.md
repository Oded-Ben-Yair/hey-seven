# Round 8 Fix Summary: Deployment Readiness

**Date**: 2026-02-20
**Fixer**: Claude Opus 4.6
**Scores**: Gemini=63, GPT=55, Grok=73 (avg=63.7)
**Previous**: R7 avg=54.3

---

## Consensus Analysis

| Finding | Gemini | GPT | Grok | Fix Applied |
|---------|--------|-----|------|-------------|
| Cloud Run probes not configured | CRITICAL | MEDIUM | HIGH | Yes - startup+liveness probes in cloudbuild.yaml |
| Unpinned deps (langfuse, cryptography) | HIGH | HIGH | HIGH | Yes - pinned exact versions |
| No post-deploy smoke test | HIGH | HIGH | HIGH | Yes - Step 7 smoke test with retries |
| No rollback strategy | -- | HIGH | CRITICAL | Yes - previous revision capture + auto-rollback |
| In-memory rate limiting | CRITICAL | HIGH | -- | Documented trade-off (no Redis per instructions) |
| Docker HEALTHCHECK ignored by CR | CRITICAL | MEDIUM | LOW | Commented + documented; CR probes added |
| Graceful shutdown 10s too short | MEDIUM | HIGH | -- | Bumped to 15s |
| Missing --cpu flag | MEDIUM | -- | MEDIUM | Added --cpu=2 |
| Missing --concurrency | -- | HIGH | MEDIUM | Added --concurrency=50 |
| Trivy scanner unpinned | -- | HIGH | -- | Pinned to 0.58.2 |
| Cloud Run timeout too short | HIGH | HIGH | -- | Bumped to 180s |
| LOG_LEVEL=WARNING in prod | MEDIUM | -- | -- | Changed to INFO |
| /health conflates liveness+readiness | -- | HIGH | -- | Added /live endpoint |
| SMS webhook no guard when disabled | MEDIUM | -- | -- | Returns 404 when SMS_ENABLED=False |
| No version in deploy env | HIGH | MEDIUM | -- | VERSION=$COMMIT_SHA |

---

## Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `requirements-prod.txt` | Modified | Pin langfuse==3.14.4, cryptography==46.0.5 |
| `requirements.txt` | Modified | Pin langfuse==3.14.4, cryptography==46.0.5 |
| `Dockerfile` | Modified | HEALTHCHECK documented as CR-ignored, PYTHONHASHSEED, graceful shutdown 15s |
| `cloudbuild.yaml` | Modified | Trivy pinned, --cpu=2, --concurrency=50, --timeout=180s, --no-traffic deploy, startup/liveness probes, smoke test, rollback, VERSION=$COMMIT_SHA, LOG_LEVEL=INFO |
| `src/api/app.py` | Modified | Added /live endpoint, health includes environment, SMS webhook 404 guard |
| `src/api/models.py` | Modified | Added LiveResponse model, HealthResponse.environment field |
| `src/api/errors.py` | Modified | Added ErrorCode.NOT_FOUND |
| `src/api/middleware.py` | Modified | Documented Cloud Run in-memory rate limiting trade-off |
| `tests/test_deployment.py` | Created | 46 deployment readiness tests (7 test classes) |
| `tests/test_doc_accuracy.py` | Modified | Updated field counts for new fields |
| `tests/test_phase2_integration.py` | Modified | SMS webhook tests enable SMS_ENABLED |

---

## Test Results

- **1262 passed**, 20 skipped, 0 failed
- **46 new deployment tests** added across 7 classes
- Coverage: **90.44%** (above 90% gate)

## Key Decisions

1. **Kept Dockerfile HEALTHCHECK** (commented with explanation) -- useful for local docker-compose even though Cloud Run ignores it
2. **Separate /live and /health** -- /live always returns 200 (liveness), /health returns 503 when degraded (startup/readiness). Prevents instance flapping.
3. **--no-traffic deploy + smoke test + rollback** -- canary-safe deployment pipeline
4. **In-memory rate limiting documented as accepted trade-off** -- per instructions, no Redis added
5. **SMS webhook returns 404 when SMS_ENABLED=False** -- closes security gap where unverified endpoint was always mountede
