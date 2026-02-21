# Production Review R12 — Grok Focus

**Reviewer**: Grok (hostile production reviewer)
**Commit**: a36a5e6
**Focus**: Operational readiness, deployment configuration, monitoring, day-2 operations, documentation accuracy (spotlight)
**Date**: 2026-02-21

---

## Dimension Scores

| # | Dimension | Score | Justification |
|---|-----------|-------|---------------|
| 1 | Graph/Agent Architecture | 8 | Solid 11-node StateGraph with well-documented dual-layer feature flag design, deterministic tie-breaking, and structured dispatch; no new architectural issues found. |
| 2 | RAG Pipeline | 7 | Per-item chunking, RRF reranking, and idempotent ingestion are solid; production vector DB (Vertex AI) is config-driven but not deployed -- `VECTOR_DB` not set in cloudbuild.yaml. |
| 3 | Data Model / State Design | 8 | PropertyQAState is clean with parity assertion, per-turn reset, and `_keep_max` reducer for responsible_gaming_count; `responsible_gaming_count: 0` in `_initial_state()` is harmless due to the reducer but semantically misleading. |
| 4 | API Design | 8 | Structured error taxonomy, SSE heartbeat, correct probe separation (/live vs /health), and Retry-After headers; solid for production. |
| 5 | Testing Strategy | 7 | 1461 tests collected (90%+ coverage gate in CI), but all documentation claims stale test counts (see Finding 1). |
| 6 | Docker & DevOps | 7 | Multi-stage build, non-root user, exec-form CMD, Trivy scan, 8-step pipeline with rollback; missing several production env vars in cloudbuild.yaml (see Findings 4, 5). |
| 7 | Prompts & Guardrails | 8 | 5-layer deterministic guardrails with structured output routing; compliance gate + router defense-in-depth is well-reasoned. |
| 8 | Scalability & Production | 6 | Single-container in-memory state for rate limiting, circuit breaker, and checkpointer; `SheetsClient.read_category()` blocks the event loop (see Finding 3). |
| 9 | Documentation & Code Quality (SPOTLIGHT) | 5 | Multiple stale numbers across README, CLAUDE.md, and runbook (see Findings 1, 2, 7, 8); code comments and docstrings are excellent, but the documentation drift is severe for a production codebase. |
| 10 | Domain Intelligence | 8 | TCPA compliance with consent hash chain, quiet hours with MNP caveats, 280+ area code mappings, bilingual keywords; production-grade for regulated casino domain. |

**Total Score: 72/100**

---

## Findings

### Finding 1 (HIGH — SPOTLIGHT +1 SEVERITY BUMP): Pervasive stale test count and file count across all documentation

- **Location**: `README.md:126,131,291`, `CLAUDE.md:19,97`
- **Problem**: README claims "1216 tests collected" and "35 test files". CLAUDE.md claims "1070+ tests across 32 test files". Actual pytest collection shows **1461 tests** across **42 test files**. The README count is 245 tests behind reality (17% drift). CLAUDE.md is 391 tests behind (37% drift). The project tree in README also says "1216 tests across 35 files".
- **Impact**: Stakeholders, reviewers, and new team members get a materially incorrect picture of test coverage. For a production codebase in a regulated domain, documentation accuracy is a trust signal. If test counts are wrong, what else is stale? Erodes confidence in all documentation claims.
- **Fix**: Automate test count extraction in CI. Add a cloudbuild.yaml post-test step:
  ```bash
  TEST_COUNT=$(python3 -m pytest --co -q 2>&1 | tail -1 | grep -oP '\d+')
  echo "Collected $TEST_COUNT tests"
  ```
  Update README.md, CLAUDE.md to use "1400+" or remove exact counts in favor of CI badge. Add a `make update-docs` target that patches counts automatically.

### Finding 2 (MEDIUM — SPOTLIGHT +1 SEVERITY BUMP): README claims "5-step pipeline" but cloudbuild.yaml has 8 steps

- **Location**: `README.md:294` — `cloudbuild.yaml # GCP Cloud Build CI/CD (5-step pipeline)`
- **Problem**: The project structure comment in README says `cloudbuild.yaml # GCP Cloud Build CI/CD (5-step pipeline)`. The actual `cloudbuild.yaml` has 8 steps: (1) test/lint, (2) Docker build, (3) Trivy scan, (4) push, (5) capture revision, (6) deploy, (7) smoke test, (8) route traffic. The runbook correctly documents all 8 steps.
- **Impact**: Developer onboarding confusion. Someone reading the README will expect 5 steps and find 8 -- or worse, assume 3 steps are unintentional and delete them.
- **Fix**: Update `README.md:294` to say `(8-step pipeline)`. Also update the runbook which says "5-step pipeline" in the table header but correctly lists 8 rows.

### Finding 3 (HIGH): SheetsClient.read_category() is a synchronous blocking call inside an async method

- **Location**: `src/cms/sheets_client.py:106-113`
- **Problem**: `read_category()` is declared `async def` but calls `self._service.spreadsheets().values().get(...).execute()` synchronously. The Google Sheets API client uses `httplib2` under the hood, which is synchronous and blocks the event loop. Unlike `retrieve_node()` which correctly wraps ChromaDB in `asyncio.to_thread()`, `SheetsClient` makes no such accommodation.
- **Impact**: When `read_all_categories()` is called (reads 8 categories sequentially), the event loop is blocked for the entire duration of 8 HTTP round-trips to Google Sheets API. During this time, all SSE streams, health checks, and concurrent requests are stalled. With typical Sheets API latency of 200-500ms per call, this could block for 1.6-4 seconds.
- **Fix**: Wrap the synchronous call in `asyncio.to_thread()`:
  ```python
  result = await asyncio.to_thread(
      lambda: self._service.spreadsheets()
          .values()
          .get(spreadsheetId=sheet_id, range=f"{tab_name}!A:P")
          .execute()
  )
  ```
  Or use `read_all_categories()` with `asyncio.gather()` on threaded calls for parallel reads.

### Finding 4 (HIGH): Production deployment missing VECTOR_DB, CASINO_ID, and PROPERTY_DATA_PATH env vars

- **Location**: `cloudbuild.yaml:65`
- **Problem**: The `--set-env-vars` line sets only `ENVIRONMENT`, `LOG_LEVEL`, and `VERSION`. It does NOT set `VECTOR_DB=firestore`, `CASINO_ID`, `PROPERTY_DATA_PATH`, or `FIRESTORE_PROJECT`. Since `VECTOR_DB` defaults to `chroma` in `config.py:71`, the production deployment will use ChromaDB (the local dev backend) instead of Firestore/Vertex AI Vector Search. The container has no ChromaDB installed (`requirements-prod.txt` excludes it), so RAG will crash silently at import time and the health endpoint will report `rag_ready: false`.
- **Impact**: Production deployment runs without a functional RAG pipeline. All queries route through the generate node with empty context, producing ungrounded responses. The system appears healthy except for `rag_ready: false` in the health endpoint, which is easy to miss since the overall status is still "degraded" (not "down").
- **Fix**: Add required env vars to cloudbuild.yaml Step 6:
  ```yaml
  --set-env-vars=ENVIRONMENT=production,LOG_LEVEL=INFO,VERSION=$COMMIT_SHA,VECTOR_DB=firestore,FIRESTORE_PROJECT=$PROJECT_ID,CASINO_ID=mohegan_sun,PROPERTY_DATA_PATH=data/mohegan_sun.json
  ```

### Finding 5 (MEDIUM): LangFuse secrets not configured in production deployment

- **Location**: `cloudbuild.yaml:64-65`
- **Problem**: The `--set-secrets` line includes `GOOGLE_API_KEY`, `API_KEY`, `CMS_WEBHOOK_SECRET`, and `TELNYX_PUBLIC_KEY`, but NOT `LANGFUSE_PUBLIC_KEY` or `LANGFUSE_SECRET_KEY`. Since these default to empty strings in `config.py:97-98`, observability is silently disabled in production.
- **Impact**: No LangFuse traces in production. The runbook mentions "LangFuse Dashboard: cloud.langfuse.com" and the README documents LangFuse integration with "10% sampling in production" -- but none of this works because the secrets are never provided. The health endpoint correctly reports `observability_enabled: false`, but this is unlikely to be noticed.
- **Fix**: Add LangFuse secrets to the `--set-secrets` line, or document in the runbook that LangFuse is not yet configured for production and remove claims of "10% sampling in production" from the README/runbook. Honest documentation is better than aspirational documentation.

### Finding 6 (MEDIUM): RateLimitMiddleware uses getattr for _request_counter initialization

- **Location**: `src/api/middleware.py:369`
- **Problem**: `self._request_counter = getattr(self, "_request_counter", 0) + 1` uses `getattr` to lazily initialize an instance attribute that should be set in `__init__`. This pattern bypasses the class constructor contract, makes the attribute invisible to type checkers, and creates a subtle race condition: between `getattr` returning 0 and `self._request_counter = 1`, another coroutine inside the same `async with self._lock` block cannot interfere (the lock protects it), but the pattern is still an anti-pattern that obscures the class's true state.
- **Impact**: Minor: the lock makes it safe in practice. But it signals hasty patching. Any future refactoring that moves the sweep outside the lock could introduce a real race.
- **Fix**: Initialize `self._request_counter = 0` in `__init__` alongside `self._requests` and `self._lock`.

### Finding 7 (MEDIUM — SPOTLIGHT +1 SEVERITY BUMP): Runbook claims Cloud Run does not support readinessProbe, but it does

- **Location**: `docs/runbook.md:52`
- **Problem**: The runbook states "Cloud Run does not support configuring a separate `readinessProbe` via `gcloud run deploy` flags." This is incorrect as of 2025. Cloud Run supports `--readiness-probe-path` via `gcloud run deploy`. The runbook and code comments both correctly identify `/health` as the readiness probe and `/live` as the liveness probe, but then claim the distinction cannot be implemented. The actual cloudbuild.yaml only configures startup and liveness probes, missing the readiness probe.
- **Impact**: When the circuit breaker opens, instances remain in the Cloud Run load balancer rotation because there is no readiness probe to remove them. Traffic is routed to degraded instances that will serve fallback responses instead of being drained. The `/health` endpoint returning 503 on CB open was designed for exactly this purpose -- but it is not wired as a readiness probe.
- **Fix**: Add `--readiness-probe-path=/health --readiness-probe-period=10` to cloudbuild.yaml Step 6. Update the runbook to remove the incorrect claim.

### Finding 8 (LOW — SPOTLIGHT +1 SEVERITY BUMP): README and CLAUDE.md describe different test and module counts

- **Location**: `README.md:131` vs `CLAUDE.md:19,97`
- **Problem**: README says "1216 tests, 35 test files, 5 layers". CLAUDE.md says "1070+ tests across 32 test files". Neither matches reality (1461 tests, 42 test files). The two authoritative documentation sources disagree with each other AND with reality. Additionally, the README claims "51 source modules across 10 packages" while this is not verifiable without counting, the test file count discrepancy suggests this may also be stale.
- **Impact**: Multiple sources of truth with different stale values. This is worse than a single stale document because it introduces confusion about which source to trust.
- **Fix**: Choose one canonical location for metrics (CLAUDE.md or README, not both). Add a comment pointing to the canonical source. Or use a CI-generated badge/script that auto-updates counts.

### Finding 9 (LOW): docker-compose.yml does not set VECTOR_DB or other production-relevant env vars

- **Location**: `docker-compose.yml:9-10`
- **Problem**: `docker-compose.yml` only sets `ENVIRONMENT=development` and `PORT=8080`. It relies on `.env` file for everything else. If a developer runs `docker compose up` without a `.env` file (or with a minimal one), they get silent defaults for all settings -- including `VECTOR_DB=chroma` which requires ChromaDB to be installed in the container, but `requirements-prod.txt` (used by Dockerfile) excludes it.
- **Impact**: `docker compose up --build` with a minimal `.env` (only `GOOGLE_API_KEY`) will fail to initialize RAG because ChromaDB is not in the production Docker image. The error is logged but the container starts successfully in degraded mode, which is confusing for new developers.
- **Fix**: Add a comment in docker-compose.yml explaining the `.env` dependency, or add `VECTOR_DB=chroma` explicitly to the docker-compose environment section with a comment noting it requires the dev Dockerfile (or a separate `docker-compose.dev.yml` with `requirements.txt` instead of `requirements-prod.txt`).

### Finding 10 (LOW): CMS webhook handler uses synchronous verify_webhook_signature but is called from async context

- **Location**: `src/cms/webhook.py:32-81`
- **Problem**: `verify_webhook_signature()` in `src/cms/webhook.py` is a synchronous function (no `async def`). It is called from the async `handle_cms_webhook()` function and from the async `cms_webhook()` endpoint in `app.py`. The HMAC computation is fast (microseconds), so this is not a performance issue. However, the SMS webhook's `verify_webhook_signature()` in `src/sms/webhook.py` is `async def`, creating an inconsistency. The CMS version does disk I/O-free HMAC; the SMS version does Ed25519 verification with `cryptography` library.
- **Impact**: Minimal performance impact (HMAC is CPU-bound and fast). But the inconsistency between CMS (sync) and SMS (async) webhook signature verification functions could confuse maintainers. The function is called without `await` in `handle_cms_webhook()` but WITH `await` in `app.py:403` for the SMS webhook -- the patterns diverge unnecessarily.
- **Fix**: Either make the CMS `verify_webhook_signature` async for consistency (trivial: add `async def` and it becomes a coroutine), or document why the patterns differ (HMAC is pure CPU, Ed25519 uses lazy import of `cryptography`).

### Finding 11 (LOW): Smoke test in cloudbuild.yaml does not fail on version mismatch

- **Location**: `cloudbuild.yaml:94-97`
- **Problem**: Step 7 extracts the deployed version and prints it alongside the expected commit SHA, but does NOT assert they match. The version check is informational only -- the pipeline proceeds to route traffic even if the deployed version is wrong. The runbook correctly documents that version assertion is mandatory after deployment, but the pipeline does not enforce it.
- **Impact**: If the health endpoint returns 200 but with a stale version (old container serving from a warm instance), the pipeline routes traffic to the wrong code. The operator must manually catch the mismatch from the log output.
- **Fix**: Add a version assertion after extracting the deployed version:
  ```bash
  if [ "$DEPLOYED_VERSION" != "$COMMIT_SHA" ]; then
    echo "VERSION MISMATCH: deployed=$DEPLOYED_VERSION expected=$COMMIT_SHA"
    echo "Rolling back..."
    # rollback logic
    exit 1
  fi
  ```

---

## Summary

The codebase demonstrates strong architectural fundamentals -- the 11-node StateGraph, specialist dispatch, circuit breaker, and TCPA compliance module are all well-engineered. The code quality (docstrings, type hints, error handling) is consistently high.

The primary concerns are operational:

1. **Documentation drift is severe**: Test counts, file counts, and pipeline step counts are stale across README, CLAUDE.md, and runbook. For a regulated domain product, documentation accuracy is a trust signal. When three separate documents disagree about the same metric, none of them is trustworthy.

2. **Production deployment gaps**: The cloudbuild.yaml is missing critical env vars (`VECTOR_DB`, `FIRESTORE_PROJECT`, `CASINO_ID`) and observability secrets. Deploying as-is would produce a production container running with local dev defaults.

3. **Event loop blocking**: `SheetsClient.read_category()` performs synchronous HTTP calls inside an async method, potentially stalling all concurrent requests during CMS reads.

4. **Missing readiness probe**: The `/health` endpoint was designed to gate traffic routing on degraded state, but no readiness probe is configured in Cloud Run to actually use it.

The codebase is code-ready but not fully deploy-ready for a production casino environment.
