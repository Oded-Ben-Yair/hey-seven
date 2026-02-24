# R46 Findings: D6 + D8

**Date**: 2026-02-24
**Reviewer**: Opus 4.6 (hostile)
**Cross-validation**: Self-validated against actual code line references, gcloud CLI docs, Redis library docs.

---

## D6 Docker & DevOps (current: 8.5)

### Findings

| ID | Severity | File:Line | Finding | Suggested Fix |
|----|----------|-----------|---------|---------------|
| D6-C001 | CRITICAL | `requirements-prod.in:32` | **`redis>=5.0` is unpinned AND missing from hashed lockfile.** `requirements-prod.in` declares `redis>=5.0` (unpinned range) but `requirements-prod.txt` (the pip-compile'd lockfile with `--generate-hashes`) contains zero `redis` entries. The Dockerfile uses `--require-hashes` on `requirements-prod.txt`, which means `redis` is NOT installed in the production image. All Redis-dependent R46 features (CB sync, distributed rate limiting) silently fall back to in-memory. This is not just a "nice-to-have" -- the entire D8 Redis integration is dead code in production. | 1. Pin `redis==5.2.1` in `requirements-prod.in`. 2. Re-run `pip-compile --generate-hashes --output-file=requirements-prod.txt requirements-prod.in` to generate hashed lockfile. 3. Verify `redis` appears in `requirements-prod.txt` with hashes. |
| D6-C002 | CRITICAL | `cloudbuild.yaml:263` | **`--to-latest` and `--to-revisions` are mutually exclusive flags.** Line 263: `gcloud run services update-traffic hey-seven --region=us-central1 --to-latest --to-revisions=LATEST=10`. The gcloud CLI treats `--to-latest` and `--to-revisions` as conflicting. `--to-latest` means "100% traffic to latest revision", `--to-revisions` means "route specific percentages". Passing both causes undefined behavior or CLI error. Stage 2 (line 268) and Stage 3 (line 273) use the flags correctly (single flag each). Only Stage 1 has the conflict. | Remove `--to-latest` from line 263. Correct command: `gcloud run services update-traffic hey-seven --region=us-central1 --to-revisions=LATEST=10` |
| D6-M001 | MAJOR | `cloudbuild.yaml:69` | **Cosign binary downloaded via HTTP without checksum verification.** `curl -Lo /workspace/cosign https://github.com/sigstore/cosign/releases/download/v2.4.1/cosign-linux-amd64` downloads the cosign binary over HTTPS but does not verify its SHA-256 digest. A compromised CDN/mirror or MITM (however unlikely with HTTPS) could supply a tampered signing tool. This undermines the entire image signing chain -- a malicious cosign binary could sign anything with any key. The irony: the project enforces `--require-hashes` for Python deps but not for the signing tool itself. | Add checksum verification: `echo "<expected_sha256>  /workspace/cosign" \| sha256sum -c` after the curl download. The expected hash is published on the cosign releases page. |
| D6-M002 | MAJOR | `docs/runbook.md:84` | **Runbook shows stale HEALTHCHECK command (curl, not python urllib).** Line 84 shows `CMD curl -f http://localhost:8080/health \|\| exit 1` but the actual Dockerfile (line 68-69) uses `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"`. R41 removed curl from the production image. Running the runbook's documented command would fail because curl is not installed. | Update runbook line 84 to match the actual Dockerfile HEALTHCHECK using python urllib. |
| D6-M003 | MAJOR | `cloudbuild.yaml:109-117` | **Steps 5, 6, and 7 have no `timeout` field.** Steps 1-4 all have explicit `timeout` fields (600s, 300s, etc.). Steps 5, 6, and 7 (capture revision, deploy, smoke test) use Cloud Build's default 10-minute timeout. Step 7 in particular (smoke test with 30s sleep + 3 retries * 15s + 3 rollback retries * 10s) could legitimately take 2+ minutes; the default 10-minute timeout is probably fine but inconsistent with the explicit-timeout pattern used elsewhere. Step 6 (Cloud Run deploy) can take 3-5 minutes for cold start. Missing timeout means a hung deploy blocks the pipeline for 10 minutes silently. | Add `timeout: '300s'` to Steps 5, 6, and 7 for consistency and explicit SLA documentation. |
| D6-M004 | MAJOR | `.dockerignore` | **`docs/` directory not excluded -- PNG screenshots (4 files) leak into production image.** The `.dockerignore` excludes `*.md` files but `docs/` contains 4 PNG screenshots (`screenshot-desktop.png`, `screenshot-mobile.png`, `screenshot-live-desktop.png`, `screenshot-desktop-v2.png`) and a file named `docs/plans/=` (likely a build artifact). These are copied into the image via `COPY src/ ./src/` -- wait, actually they're in `docs/` not `src/`. But the Dockerfile doesn't have a `COPY docs/` line, so this may not apply... Let me re-check: the Dockerfile copies `data/`, `static/`, and `src/`. `docs/` is not copied. **Retracted -- not a real issue.** The `.dockerignore` patterns still matter for `docker build` context size but no files from `docs/` end up in the image. | ~~Add `docs/` to `.dockerignore`~~ RETRACTED: Not a production issue since `docs/` is not COPY'd. Minor: add `docs/` to .dockerignore to reduce build context transfer time. |
| D6-m001 | MINOR | `docs/runbook.md:102` | **Runbook pipeline table still says "Step 8: Route 100% traffic" but actual Step 8 is now canary graduated rollout.** The runbook table at line 102 says `8 \| Route 100% traffic to new revision (\-\-to-latest)` but the actual Step 8 in cloudbuild.yaml is now a multi-stage canary (10% -> 50% -> 100%) with error rate checks. The one-liner description is misleading. | Update runbook pipeline table to reflect the canary deployment flow. |
| D6-m002 | MINOR | `cloudbuild.yaml:16` | **Cloud Build Step 1 uses Python 3.12.8 base but does not pin to digest.** Step 1 (`python:3.12.8-slim-bookworm`) uses a tag but not a digest, unlike the Dockerfile which pins both tag and digest. While this is the CI/test step (not production), a compromised Python base image in CI could inject malicious test passes. | Pin Step 1 Python image to the same digest used in the Dockerfile. |

### What's Good (credit improvements from aa5497d)

- **Cosign image signing (Steps 4b-4e)**: Full signing chain with GCP KMS, SBOM attestation, and inline verification. This is a genuine supply chain improvement. Well-documented ADR.
- **Canary deployment (Step 8)**: Graduated 10%/50%/100% traffic splitting with error rate monitoring between stages is a real production safety net. The `check_error_rate()` function is well-structured.
- **Secret rotation script**: `scripts/rotate-secret.sh` follows a correct flow: create version -> update service -> verify health -> disable old. No-delete policy for rollback.
- **SBOM generation**: CycloneDX format via Trivy, attached to signed image. Genuine compliance step.
- **VPC connector + Redis secret in deploy**: Infrastructure wiring for distributed state is correct (even though Redis dependency is missing from lockfile).
- **Per-step timeouts on Steps 1-4**: Explicit timeouts prevent hung pipeline stages.
- **Runbook expansion**: Canary, cosign, secret rotation sections are thorough and accurate (except stale HEALTHCHECK).

### Score Assessment
- Previous: 8.5
- Proposed: **8.3**
- Justification: D6-C001 (redis missing from lockfile) means the entire Redis integration is dead code in production -- this is a regression introduced by the R46 changes. D6-C002 (conflicting gcloud flags) means canary Stage 1 will fail or behave unpredictably. These two CRITICALs offset the genuine improvements from cosign, canary, and SBOM. The 4 MAJORs (cosign checksum, runbook stale, missing timeouts) are real but fixable in hours. Net effect: the new features are architecturally correct but not deployable due to the lockfile and flag issues.

---

## D8 Scalability & Production (current: 9.0)

### Findings

| ID | Severity | File:Line | Finding | Suggested Fix |
|----|----------|-----------|---------|---------------|
| D8-C001 | CRITICAL | `circuit_breaker.py:117-128` | **`_sync_to_backend()` calls synchronous Redis methods inside an async function without `await asyncio.to_thread()`.** `self._backend.set()` when backend is `RedisBackend` calls `self._client.setex(key, ttl, value)` which is a synchronous, blocking network call (the `redis` Python library is synchronous). This blocks the event loop for the duration of the Redis round-trip (~0.5-2ms on VPC, 5-50ms on degraded network). Called after EVERY `record_failure()` and `record_success()` (i.e., on every LLM call). With 50 concurrent SSE streams, this means 50 blocking Redis calls per LLM-call-cycle, starving heartbeats, health checks, and other SSE streams. Same issue in `_sync_from_backend()` (lines 149-151) which calls `self._backend.get()`. | Use `redis.asyncio.Redis` instead of `redis.Redis` for all async code paths. Or wrap synchronous Redis calls in `await asyncio.to_thread(self._backend.set, ...)`. The `StateBackend` ABC should have async variants (`async def aset`, `async def aget`) for async callers. |
| D8-C002 | CRITICAL | `middleware.py:564-569` | **`_is_allowed_redis()` calls synchronous Redis pipeline in async handler.** `pipe = self._redis_client.pipeline(); pipe.zremrangebyscore(...); pipe.zcard(...); pipe.zadd(...); pipe.expire(...); results = pipe.execute()` -- all synchronous, blocking calls from the `redis.Redis` client. This runs on EVERY rate-limited request (`/chat`, `/feedback`). Under load (50 concurrent connections), this blocks the event loop for 50 * 1-5ms = 50-250ms per burst, causing SSE heartbeat misses and health probe timeouts. Line 574: `self._redis_client.zrem(key, member)` is another synchronous blocking call. | Same as D8-C001: use `redis.asyncio.Redis` or wrap in `asyncio.to_thread()`. |
| D8-M001 | MAJOR | `middleware.py:386-389` | **`redis.Redis.from_url()` and `ping()` called at RateLimitMiddleware.__init__() time (import time).** `RateLimitMiddleware.__init__` is called during `create_app()` which happens at module import time (`app = create_app()` at line 677 of app.py). If Redis is unreachable during import, the exception is caught but adds 5-30s to container startup (default Redis connection timeout). This happens BEFORE the startup probe starts, potentially causing probe timeout. | Move Redis initialization to first request (lazy init pattern), or set a short connection timeout: `redis.Redis.from_url(url, socket_connect_timeout=2)`. |
| D8-M002 | MAJOR | `_base.py:327` | **LLM semaphore timeout hardcoded to 30 instead of reading `settings.LLM_SEMAPHORE_TIMEOUT`.** `config.py:76` defines `LLM_SEMAPHORE_TIMEOUT: int = 30` as a configurable setting, but `_base.py:327` hardcodes `await asyncio.wait_for(_LLM_SEMAPHORE.acquire(), timeout=30)`. The config field is dead -- changing `LLM_SEMAPHORE_TIMEOUT` via environment variable has no effect. This prevents runtime tuning during incidents. | Replace `timeout=30` with `timeout=get_settings().LLM_SEMAPHORE_TIMEOUT`. |
| D8-M003 | MAJOR | `circuit_breaker.py:117` | **CB `_sync_to_backend` stores `_last_failure_time` (monotonic clock) in Redis.** Line 124-128 writes `str(self._last_failure_time)` to Redis, where `_last_failure_time` is from `time.monotonic()`. Monotonic clocks are per-process -- they have different epoch origins on different machines. Storing a monotonic timestamp in Redis and reading it from another Cloud Run instance produces meaningless values. The stored value is currently only written, not consumed cross-instance (the `_sync_from_backend` only reads `state` and `failure_count`), so this is not a correctness bug today, but it's misleading data that will cause bugs if anyone tries to use it for cross-instance cooldown calculation. | Either: (1) remove the `_last_failure_time` write to Redis (it's unused), or (2) switch to `time.time()` (wall clock) for the Redis-synced value, keeping `time.monotonic()` for local use. |
| D8-M004 | MAJOR | `state_backend.py:64-69` | **`InMemoryBackend` uses `threading.Lock` but is called from async code paths.** `_sync_to_backend()` and `_sync_from_backend()` in circuit_breaker.py call `self._backend.set()` and `self._backend.get()`. When backend is `InMemoryBackend`, these acquire `threading.Lock` inside an async function. `threading.Lock` blocks the event loop thread. The InMemoryBackend operations are sub-microsecond (dict access), so this is negligible in practice, but under contention (50 concurrent requests hitting `_maybe_sweep()`), the lock hold time during sweep (up to 1ms per batch) blocks the event loop. | For InMemoryBackend used in async contexts, consider making operations lock-free (single-threaded Python GIL already provides atomicity for dict mutations) or use `asyncio.Lock`. The threading.Lock is only needed if the backend is accessed from multiple threads (e.g., `to_thread()` calls). |
| D8-m001 | MINOR | `circuit_breaker.py:301` | **`_sync_from_backend()` called outside the async lock in `allow_request()`.** Line 301: `await self._sync_from_backend()` runs before `async with self._lock:` on line 302. Two concurrent `allow_request()` calls can both read from Redis, both see "open", then both enter the lock and both try to update state. This is not a correctness bug (both set `_state = "open"` -- idempotent), but it means two unnecessary Redis reads instead of one. | Move `_sync_from_backend()` inside the lock, or accept as documented (it's rate-limited to 5s intervals anyway). |
| D8-m002 | MINOR | `langfuse_client.py:49` | **`_get_langfuse_client()` uses `threading.Lock` for double-checked locking pattern.** Same concern as other threading.Lock-in-async-context patterns. LangFuse client init can take 100-500ms (HTTP connection to langfuse server). If called from the event loop, `with _langfuse_lock:` blocks it. In practice, this only happens once (on first call after TTL expiry) and the init is fast, so impact is low. | Consider `asyncio.Lock` or lazy init pattern. Low priority since it's a one-time init cost. |

### What's Good (credit improvements from aa5497d)

- **Circuit breaker Redis backend sync architecture**: The L1 (local deque) / L2 (Redis) pattern is architecturally sound. One-directional promotion (only closed->open) is the correct design choice.
- **Distributed rate limiting via Redis sorted sets**: `_is_allowed_redis()` uses the correct ZRANGEBYSCORE sliding window pattern with pipeline for atomicity. The add-then-remove-on-reject pattern is correct.
- **Semaphore backpressure with timeout**: `asyncio.wait_for(_LLM_SEMAPHORE.acquire(), timeout=30)` is a good pattern for preventing request pile-up.
- **Redis fallback to in-memory**: Every Redis code path has a try/except that falls back to in-memory. Availability > consistency is the right choice for this use case.
- **StateBackend abstraction**: Clean ABC with InMemoryBackend and RedisBackend implementations. Proper dependency injection into CircuitBreaker.
- **TTL jitter on langfuse cache**: Parity with other singleton caches.
- **k6 load test scripts**: Both local and Cloud Run variants with realistic casino queries, proper thresholds, and custom metrics (first_token_time, scale_up_503s).

### Score Assessment
- Previous: 9.0
- Proposed: **8.7**
- Justification: D8-C001 and D8-C002 are the same root cause (synchronous Redis calls blocking the event loop in async code). This is a fundamental design flaw in the Redis integration -- the correct fix is using `redis.asyncio` or wrapping all calls in `to_thread()`. The architecture is sound but the implementation will block the event loop under any load. Combined with D6-C001 (redis not in lockfile), the Redis features are currently non-functional anyway, so these CRITICALs are latent. When Redis is actually wired in (lockfile fixed), these will manifest as event loop stalls. The 4 MAJORs are real but lower severity. The improvements (architecture, fallback, semaphore, load tests) are genuine and well-designed.

---

## Summary

| Category | Count |
|----------|-------|
| CRITICAL | 4 (2 D6, 2 D8) |
| MAJOR | 7 (3 D6, 4 D8) |
| MINOR | 4 (2 D6, 2 D8) |
| **Total** | **15** |
| Retracted | 1 (D6-M004 -- .dockerignore docs/ is not COPY'd) |

### Key Issues

1. **D6-C001**: `redis` package missing from hashed lockfile (`requirements-prod.txt`). All Redis features are dead code in production.
2. **D6-C002**: Conflicting `--to-latest --to-revisions` flags in canary Stage 1 will fail the gcloud CLI.
3. **D8-C001/C002**: Synchronous `redis.Redis` calls block the asyncio event loop in circuit breaker sync and rate limiter. Must use `redis.asyncio` or `to_thread()`.
4. **D8-M002**: `LLM_SEMAPHORE_TIMEOUT` config exists but is hardcoded at the usage site.

### Score Impact

| Dimension | Previous | Proposed | Delta | Justification |
|-----------|----------|----------|-------|---------------|
| D6 Docker & DevOps | 8.5 | 8.3 | -0.2 | 2 CRITICALs (lockfile, gcloud flags) offset genuine cosign/canary/SBOM improvements |
| D8 Scalability & Prod | 9.0 | 8.7 | -0.3 | 2 CRITICALs (sync Redis blocking event loop) are latent due to D6-C001 but will manifest when fixed |

### Estimated Weighted Score Impact

Using dimension weights (D6=0.10, D8=0.15):
- D6 delta: -0.2 * 0.10 = -0.02
- D8 delta: -0.3 * 0.15 = -0.045
- **Total weighted impact: -0.065 (from 95.0 to ~94.9)**

Note: All 4 CRITICALs are straightforward fixes (pip-compile, remove flag, use async redis). After fixing, the genuine improvements from cosign, canary, Redis architecture, and backpressure should push D6 to 9.0-9.2 and D8 to 9.3-9.5.
