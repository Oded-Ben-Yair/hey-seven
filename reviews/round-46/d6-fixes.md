# R46 D6 Fixes Applied

**Date**: 2026-02-24
**Fixer**: code-worker (Opus 4.6)

## CRITICALs (2/2 fixed)

### D6-C001: redis missing from hashed lockfile
- **File**: `requirements-prod.in` -- pinned `redis==5.2.1` (was `redis>=5.0`)
- **File**: `requirements-prod.txt` -- added `redis==5.2.1` with both wheel and sdist SHA-256 hashes
- **Verification**: `grep redis requirements-prod.txt` confirms entry present with hashes

### D6-C002: Conflicting --to-latest and --to-revisions flags
- **File**: `cloudbuild.yaml:263` -- removed `--to-latest` from canary Stage 1
- **Before**: `gcloud run services update-traffic ... --to-latest --to-revisions=LATEST=10`
- **After**: `gcloud run services update-traffic ... --to-revisions=LATEST=10`

## MAJORs (3/3 fixed, 1 retracted)

### D6-M001: Cosign binary without checksum verification
- **File**: `cloudbuild.yaml:69-70` -- added `sha256sum -c` verification after curl download
- **Checksum**: `8b24b946dd5809c6bd93de08033bcf6bc0ed7d336b7785787c080f574b89249b` (verified by downloading and computing locally)

### D6-M002: Runbook stale HEALTHCHECK command
- **File**: `docs/runbook.md:83-84` -- updated from `curl -f` to `python -c "import urllib.request; urllib.request.urlopen(...)"`
- Matches actual Dockerfile HEALTHCHECK (R41 removed curl from production image)

### D6-M003: Steps 5, 6, 7 missing timeout fields
- **File**: `cloudbuild.yaml` -- added `timeout: '120s'` to Step 5, `timeout: '300s'` to Steps 6 and 7
- Consistent with explicit timeout pattern used on Steps 1-4

### D6-M004: RETRACTED by reviewer
- `.dockerignore` docs/ exclusion is not needed since `docs/` is not COPY'd. No change made.

## MINORs (2/2 fixed)

### D6-m001: Runbook pipeline table stale for Step 8
- **File**: `docs/runbook.md` -- updated Step 8 description from "Route 100% traffic" to "Canary traffic rollout (10% -> 50% -> 100%) with error rate monitoring"
- Also added Steps 3b, 4b-4e to pipeline table (SBOM generation, cosign signing/attestation/verification)
- Updated traffic routing row in service config table to reflect canary strategy
- Updated automatic rollback description to include canary error rate trigger

### D6-m002: Cloud Build Step 1 Python image not pinned to digest
- **File**: `cloudbuild.yaml:16` -- pinned to `@sha256:8ef40398b663cf0a3a4685ad0ffcf924282e4c954283b33b7765eae0856d7e0c`
- Same digest as Dockerfile for consistency

## Test Results

- `tests/test_deployment.py`: **66 passed**, 0 failed, 0 errors
- Coverage gate (90%) not met because only deployment tests were run (expected)

## Files Modified

| File | Changes |
|------|---------|
| `requirements-prod.in` | Pinned `redis==5.2.1` |
| `requirements-prod.txt` | Added redis entry with SHA-256 hashes |
| `cloudbuild.yaml` | Removed conflicting flag, added cosign checksum, added timeouts, pinned Python digest |
| `docs/runbook.md` | Fixed HEALTHCHECK command, updated pipeline table, updated traffic routing description |
