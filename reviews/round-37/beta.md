# R37 Hostile Review — Dimensions 6-10 (reviewer-beta)

**Reviewer**: reviewer-beta (Opus 4.6)
**Cross-validation**: GPT-5.2 Codex (azure_code_review), Gemini 3.1 Pro (thinking=high)
**Date**: 2026-02-23
**Baseline**: R36 scored 84.5/100. D6=6.5, D7=7.0, D8=5.5, D9=7.5, D10=7.0

---

## D6: Docker & DevOps (weight 0.10) — Score: 6.5/10

### What works
- Multi-stage Dockerfile with non-root `appuser`, exec-form CMD (SIGTERM handling correct)
- 15s graceful shutdown timeout (`--timeout 15`) matches Cloud Run default
- `requirements-prod.txt` excludes ChromaDB (~200MB) — good prod/dev split
- 8-step Cloud Build pipeline with Trivy scan, smoke test, automatic rollback
- `--min-instances=1` prevents cold start for first request
- `--cpu-boost` during startup for faster container initialization
- Rollback captures previous revision tag before deploy, health-verifies before traffic shift

### CRITICAL findings: 0

### MAJOR findings: 4 (all carried from R36 — NONE resolved)

**M1 (CARRIED R36-B-M1): No SBOM or image signing**
- File: `cloudbuild.yaml`
- No `cosign sign` or `syft` step. Supply chain provenance is unverifiable.
- Casino operators under SOC 2 / PCI-DSS may require signed container images.
- Fix: Add `cosign sign` after push step, `syft` SBOM generation.

**M2 (CARRIED R36-B-M2): No build failure notifications**
- File: `cloudbuild.yaml`
- No Pub/Sub notification, no Slack/email webhook on build failure.
- A broken main branch is invisible until someone checks the console.
- Fix: Add `pubsubConfig` or Cloud Build trigger notification channel.

**M3 (CARRIED R36-B-M3): Base image tag not pinned to digest**
- File: `Dockerfile`, line 1: `FROM python:3.12.8-slim-bookworm`
- Tag `3.12.8-slim-bookworm` can be republished with different contents (supply chain risk).
- Fix: Pin to `python:3.12.8-slim-bookworm@sha256:<digest>`.

**M4 (CARRIED R36-B-M4): No hash-verified dependency installation**
- File: `Dockerfile`
- `pip install -r requirements-prod.txt` without `--require-hashes`.
- MITM or compromised PyPI mirror could inject malicious packages.
- Fix: Generate `requirements-prod.txt` with `pip-compile --generate-hashes`, use `--require-hashes` in Dockerfile.

### MINOR findings: 2

**m1 (CARRIED): No per-step timeout in Cloud Build**
- File: `cloudbuild.yaml`
- Individual steps have no `timeout:` field. A hung Trivy scan or smoke test blocks the pipeline indefinitely (only global 20min timeout).
- Fix: Add `timeout: '300s'` per step.

**m2 (NEW): HEALTHCHECK in Dockerfile is ignored by Cloud Run**
- File: `Dockerfile`
- `HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1` — Cloud Run ignores Dockerfile HEALTHCHECK instructions. This is dead configuration that misleads developers into thinking container self-healing exists.
- Impact: Low (documentation issue), but could delay incident diagnosis if operators believe container-level health checks are running.

### D6 Assessment
Score remains 6.5. All 4 R36 MAJORs carried forward with zero progress. The pipeline is functional and has good rollback semantics, but supply chain hardening (SBOM, signing, digest pinning, hash verification) remains completely unaddressed. These are table-stakes for regulated casino environments.

---

## D7: Prompts & Guardrails (weight 0.10) — Score: 7.5/10 (+0.5 from R36)

### What works
- 5-layer deterministic guardrails execute pre-LLM (injection, responsible gaming, age verification, BSA/AML, patron privacy)
- R36 normalization order fix applied: NFKD before confusable replacement (correct)
- R36 IPA/Latin Extended confusables added (7 new characters: ipa_a, ipa_g, dotless_i, etc.)
- R36 JP/KO patterns added for responsible gaming detection
- 8192 char input length limit prevents ReDoS via input size
- Semantic injection classifier with fail-closed + 5s asyncio.timeout
- `string.Template.safe_substitute()` throughout prompts (no format-string injection)
- HEART escalation framework for frustrated guests
- Per-casino helpline lookup with fallback logging (R35 fix)
- `_normalize_input()` uses `str.translate()` for O(n) confusable replacement

### CRITICAL findings: 0

### MAJOR findings: 2

**M1 (NEW): Potential ReDoS in complex alternation patterns**
- File: `src/agent/guardrails.py`
- Several patterns use nested alternations with overlapping character classes. While the 8192 char limit bounds absolute worst-case, patterns like the multilingual responsible gaming detector combine `(?:...)` groups with `\s*` separators that could exhibit super-linear behavior on crafted inputs near the 8192 limit.
- Specific concern: Hindi/Tagalog patterns added in pre-sprint research may not have been fuzz-tested for catastrophic backtracking.
- The 8192 limit provides a hard ceiling, but a 100ms+ regex evaluation on a single guardrail layer still degrades p99 latency under load.
- Fix: Run `regex-timeout` benchmarks on all 173 patterns with adversarial 8192-char inputs. Consider `re2` (Google's linear-time regex engine) for guardrail patterns where backreferences are not needed.

**M2 (NEW): Semantic injection classifier 5s timeout is too generous for SSE streaming**
- File: `src/agent/guardrails.py`
- The semantic injection classifier has a 5-second `asyncio.timeout`. For SSE streaming where users expect sub-second first-token latency, a 5s guardrail check before any LLM call creates unacceptable perceived latency.
- During classifier LLM degradation, every request pays the full 5s penalty before the fail-closed fallback activates. With 50 concurrent requests, this means 50 simultaneous 5s waits consuming Cloud Run CPU.
- Fix: Reduce timeout to 2s. Consider caching classifier results for identical/similar inputs (TTL 60s).

### MINOR findings: 2

**m1 (NEW): No guardrail execution time metrics**
- File: `src/agent/guardrails.py`
- Individual guardrail layers have no timing instrumentation. If a specific regex pattern becomes slow (e.g., after adding new language patterns), there is no observability to identify which layer is the bottleneck.
- Fix: Add per-layer `time.monotonic()` measurements, log when any layer exceeds 10ms.

**m2 (NEW): Confusable map is append-only with no coverage testing**
- File: `src/agent/guardrails.py`
- The confusable replacement map grows with each round (R36 added 7 IPA characters). There is no automated test that verifies coverage against a known Unicode confusables database (e.g., Unicode TR39 confusables.txt).
- New characters are added reactively when reviews find bypasses, not proactively from the Unicode standard.
- Fix: Add a test that cross-references the confusable map against Unicode TR39 skeleton mappings for characters used in the supported 11 languages.

### D7 Assessment
Score improves to 7.5 (+0.5). R36 normalization and IPA fixes are correctly applied. The 8192 char limit provides meaningful ReDoS protection. Main gaps: potential super-linear regex on crafted inputs near the limit, and the 5s semantic classifier timeout creating latency spikes during LLM degradation. The guardrail architecture (deterministic pre-LLM, fail-closed classifier) is sound.

---

## D8: Scalability & Production (weight 0.15) — Score: 5.5/10 (unchanged from R36)

### What works
- Circuit breaker with rolling window and half-open decay recovery
- TTL-cached singletons for credential rotation (GCP Workload Identity)
- Probabilistic sweep in InMemoryBackend (1% per write, forced at 50K)
- Pure ASGI middleware preserves SSE streaming (no BaseHTTPMiddleware)
- `--concurrency=50` on Cloud Run with `--max-instances=10` (500 max concurrent)
- Health endpoint returns 503 when degraded (CB open, RAG unavailable)
- Live endpoint always returns 200 (never fails liveness)
- CancelledError handling in SSE (re-raised, not swallowed)
- R36 fix: threading.Lock added to InMemoryBackend

### CRITICAL findings: 2

**C1 (NEW — Gemini 3 Pro cross-validated): asyncio task leak on TimeoutError in SSE streaming**
- File: `src/api/app.py`
- The SSE streaming endpoint uses `asyncio.wait_for(event.__anext__(), timeout=15)` for heartbeat. When `TimeoutError` fires, the inner async generator (`graph.astream_events()`) is NOT explicitly cleaned up. The generator's `__anext__()` coroutine is cancelled by `wait_for`, but the generator itself remains alive with internal state (LLM connections, Firestore handles).
- Under sustained load with slow LLM responses, each timeout leaks one async generator. With 50 concurrent SSE connections and p95 LLM latency of 3s, a 15s heartbeat rarely fires. But during LLM degradation (p95 > 15s), EVERY connection leaks a generator per heartbeat cycle.
- Gemini 3 Pro assessment: "The async generator is not wrapped in `contextlib.aclosing()` or explicitly `aclose()`d in the finally block. This is a resource leak that compounds during LLM degradation — exactly when you need resources most."
- Fix: Wrap `graph.astream_events()` in `contextlib.aclosing()` and ensure `aclose()` is called in the SSE endpoint's `finally` block.

**C2 (NEW — Gemini 3 Pro cross-validated): threading.Lock in InMemoryBackend blocks asyncio event loop**
- File: `src/state_backend.py`
- R36 added `threading.Lock` to InMemoryBackend for thread safety. However, `threading.Lock.acquire()` is a BLOCKING call. When the probabilistic sweep triggers (1% chance per write, or forced at 50K entries), the sweep iterates ALL entries under the lock.
- At 50K entries, the sweep involves iterating and filtering a dict of 50K items. Under CPython with GIL, this takes 10-50ms. During this time, the asyncio event loop is BLOCKED — no other coroutines can run, no SSE heartbeats are sent, no health checks respond.
- Gemini 3 Pro assessment: "A 50ms event loop block with 50 concurrent SSE connections means ALL connections stall simultaneously. Cloud Run liveness probe may fail if the sweep coincides with a probe."
- Fix: Replace `threading.Lock` with `asyncio.Lock` (InMemoryBackend is only used in async context). For the sweep, use `asyncio.to_thread()` to run the O(n) iteration off the event loop, or limit sweep batch size to 1000 entries per tick.

### MAJOR findings: 5

**M1 (NEW — Gemini 3 Pro cross-validated): Firestore AsyncClient singleton never closed**
- File: `src/casino/config.py`
- `_firestore_clients: dict[str, AsyncClient] = {}` caches Firestore clients permanently. Cloud Run suspends container CPU between requests. When CPU resumes, the gRPC channels inside AsyncClient may be dead (TCP keepalive timeout exceeded during suspension).
- The client will eventually reconnect, but the first request after suspension pays a 200-500ms reconnection penalty. With `--min-instances=1`, the minimum instance stays warm, but instances 2-10 spin up/down and will hit this.
- Fix: Add health check for Firestore client connectivity. Consider TTLCache for Firestore clients (1 hour TTL, matching LLM singleton pattern).

**M2 (NEW): No graceful drain on SIGTERM for in-flight SSE connections**
- File: `src/api/app.py`
- The Dockerfile correctly uses exec-form CMD and `--timeout 15` for graceful shutdown. But the FastAPI app has no shutdown handler that drains in-flight SSE connections.
- When Cloud Run sends SIGTERM (during scale-down or deploy), uvicorn starts shutdown. SSE connections receive `ServerDisconnectedError` immediately, losing any in-flight streamed response.
- The 15s timeout gives uvicorn time to finish HTTP requests, but long-lived SSE streams (30s+ for complex queries) will be cut mid-stream.
- Fix: Add lifespan shutdown handler that sets a `_shutting_down` flag. SSE endpoint checks this flag and sends a `{"event": "shutdown"}` message before closing. Reduce max SSE duration to 10s (< 15s SIGTERM timeout).

**M3 (NEW): Rate limiter memory unbounded under high-cardinality IP traffic**
- File: `src/api/middleware.py`
- `RateLimitMiddleware` uses `OrderedDict` keyed by IP address. The time-based sweep (R36 fix B6) runs periodically, but between sweeps, a DDoS from unique IPs grows the dict unboundedly.
- With Cloud Run's `--concurrency=50` and a botnet sending 1 request per unique IP, the dict grows at 50 IPs/second. Between sweeps (every N requests), this could reach 10K+ entries.
- Fix: Add `maxsize` cap to the OrderedDict (e.g., 10K entries). When exceeded, evict oldest entries regardless of TTL. This is an LRU rate limiter, not a perfect one, but it bounds memory.

**M4 (NEW): Runbook documents "readiness probe" but Cloud Run has no readiness probe**
- File: `docs/runbook.md`
- The runbook references "readiness probe" in operational procedures. Cloud Run supports startup probes and liveness probes, but NOT readiness probes (that's a Kubernetes concept).
- This creates operational confusion: an SRE following the runbook may waste time trying to configure a readiness probe that doesn't exist on Cloud Run.
- Fix: Replace "readiness probe" with "startup probe" throughout the runbook. Document that Cloud Run's startup probe serves the same purpose (blocking traffic until the container is ready).

**M5 (NEW): GC pressure from 50K in-memory state entries during SSE streaming**
- File: `src/state_backend.py`
- InMemoryBackend can hold up to 50K entries (dicts with timestamps). Python's GC (generational collector) runs periodically. With 50K+ live objects in gen2, a full GC collection can take 50-100ms.
- During SSE streaming, a GC pause means no tokens are sent for 50-100ms. While not catastrophic, this compounds with the threading.Lock sweep issue (C2) to create multi-hundred-ms stalls.
- Fix: For production, use RedisBackend (already implemented). For dev, reduce MAX_STORE_SIZE to 10K. Consider `gc.freeze()` on long-lived objects if InMemoryBackend is used in staging.

### MINOR findings: 2

**m1 (NEW): No connection pool size configuration for Firestore**
- File: `src/casino/config.py`
- Firestore AsyncClient uses default gRPC channel settings. Under sustained load (50 concurrent requests, each reading casino config), the default channel may saturate.
- Fix: Configure `channel_options` with appropriate max concurrent streams.

**m2 (NEW): Circuit breaker metrics not exported to Cloud Monitoring**
- File: `src/agent/circuit_breaker.py`
- `get_metrics()` returns a dict, but nothing exports this to Cloud Monitoring or Langfuse at regular intervals. The CB state is only visible through the health endpoint.
- Fix: Add periodic metrics export (every 60s) via structured logging or Langfuse trace.

### D8 Assessment
Score remains 5.5. Two new CRITICALs identified: asyncio task leak on SSE timeout (resource exhaustion during LLM degradation) and threading.Lock blocking the event loop during sweep (all connections stall). R36's threading.Lock fix for InMemoryBackend introduced a new problem (event loop blocking) while solving the race condition. Five MAJORs cover Firestore client lifecycle, SIGTERM drain, rate limiter memory bounds, runbook inaccuracy, and GC pressure. D8 remains the weakest dimension by a significant margin. The fundamental issue: the codebase is designed for single-request-response but deployed for concurrent SSE streaming, and the concurrency model has multiple impedance mismatches.

---

## D9: Trade-off Documentation (weight 0.05) — Score: 7.5/10 (unchanged)

### What works
- Comprehensive runbook covering all operational procedures
- ADRs embedded in code comments with round references (e.g., "R36 fix B5")
- Circuit breaker known limitation explicitly documented (multi-instance, short/medium/long-term path)
- InMemoryBackend vs RedisBackend trade-off documented with migration path
- Feature flag documentation with build-time vs runtime distinction
- Degraded-pass validation strategy documented with principled rationale
- Regulatory state requirements well-organized by jurisdiction
- Incident response playbooks for 5 failure scenarios

### CRITICAL findings: 0

### MAJOR findings: 1

**M1 (NEW): Runbook "readiness probe" is a documentation fantasy (also scored in D8-M4)**
- File: `docs/runbook.md`
- Cross-referenced from D8. The readiness probe documentation creates a false operational model. This is both a scalability issue (D8) and a documentation accuracy issue (D9).
- Impact on D9: documentation that describes non-existent infrastructure features is worse than missing documentation — it creates false confidence.

### MINOR findings: 2

**m1 (NEW): No ADR for threading.Lock vs asyncio.Lock decision in InMemoryBackend**
- File: `src/state_backend.py`
- R36 added `threading.Lock` but the rationale for choosing threading.Lock over asyncio.Lock is not documented. Given C2 above (event loop blocking), this decision needs an ADR explaining the trade-off and planned migration.

**m2 (NEW): Evaluation golden dataset rationale undocumented**
- File: `src/observability/evaluation.py`
- 20 golden test cases exist across 8 categories, but there's no documentation explaining how these were selected, what coverage gaps exist, or how new test cases should be added.
- Fix: Add a comment block explaining the test case selection methodology and coverage matrix.

### D9 Assessment
Score remains 7.5. Documentation is comprehensive and generally accurate. The readiness probe fiction is the only significant gap, and it's already scored in D8. Trade-off documentation in code comments (with round references) is excellent for maintainability.

---

## D10: Domain Intelligence (weight 0.10) — Score: 7.0/10 (unchanged)

### What works
- 5 casino profiles with state-specific regulatory data (CT, NJ, PA, NV)
- Tribal vs commercial casino distinction (Mohegan Sun = Mohegan Tribe sovereignty)
- Per-state responsible gaming helplines with correct numbers
- TCPA compliance with quiet hours, consent chains, bilingual keyword handling (EN+ES)
- BSA/AML guardrail layer for suspicious transaction language
- Self-exclusion program awareness (state-specific duration options)
- Age verification patterns across 11 languages
- Patron privacy guardrails (SSN, credit card, DOB patterns)
- Casino comp system and host workflow documentation in knowledge base
- Player psychology and retention playbook in knowledge base

### CRITICAL findings: 0

### MAJOR findings: 2

**M1 (NEW): Regulatory update tracking is manual and undocumented**
- File: `knowledge-base/regulations/state-requirements.md`
- State gaming regulations change frequently (NJ DGE updates quarterly, NV GCB annually). The current regulatory data is a point-in-time snapshot with no documented refresh cadence.
- The 2025 TCPA one-to-one consent rule vacatur is correctly noted, but there's no process to check for subsequent court decisions or FCC rulings.
- For a regulated product serving casino operators, stale regulatory data is a compliance risk.
- Fix: Document regulatory update cadence. Add `last_verified` dates to each state section. Consider a quarterly review checklist.

**M2 (NEW): No tribal gaming compact awareness beyond Mohegan Sun**
- File: `src/casino/config.py`, `knowledge-base/regulations/state-requirements.md`
- The codebase correctly identifies Mohegan Sun as a tribal casino under Mohegan Tribe sovereignty. But the regulatory framework doesn't generalize to other tribal casinos (e.g., Foxwoods under Mashantucket Pequot sovereignty).
- Foxwoods is in CASINO_PROFILES but its regulatory entry doesn't distinguish its tribal compact from Connecticut state law.
- Fix: Add tribal compact awareness to each tribal casino profile. Document which state regulations apply vs. tribal sovereignty exemptions.

### MINOR findings: 1

**m1 (NEW): No regional dialect awareness in multilingual guardrails**
- File: `src/agent/guardrails.py`
- Spanish patterns use Latin American vocabulary but may miss Peninsular Spanish dialectal variations. The Hindi patterns use Devanagari script but may miss Urdu cognates (same spoken language, different script).
- Impact is low for US casino market (Latin American Spanish dominates), but worth noting for future expansion.

### D10 Assessment
Score remains 7.0. Domain intelligence is solid for the current 5-property scope. Main gaps: regulatory update process (manual, no refresh cadence), and incomplete tribal gaming compact generalization. The multilingual guardrail coverage across 11 languages is impressive for an MVP.

---

## Summary

| Dimension | R36 Score | R37 Score | Change | CRITICALs | MAJORs | MINORs |
|-----------|-----------|-----------|--------|-----------|--------|--------|
| D6 Docker & DevOps | 6.5 | 6.5 | 0 | 0 | 4 (carried) | 2 |
| D7 Prompts & Guardrails | 7.0 | 7.5 | +0.5 | 0 | 2 | 2 |
| D8 Scalability & Production | 5.5 | 5.5 | 0 | 2 | 5 | 2 |
| D9 Trade-off Documentation | 7.5 | 7.5 | 0 | 0 | 1 | 2 |
| D10 Domain Intelligence | 7.0 | 7.0 | 0 | 0 | 2 | 1 |

**Total findings**: 2 CRITICALs + 14 MAJORs + 9 MINORs

### Top 3 Fixes by Impact (for fixer priority):

1. **C1 (D8): asyncio task leak on SSE timeout** — Wrap `graph.astream_events()` in `contextlib.aclosing()`, ensure `aclose()` in finally block. Resource exhaustion during LLM degradation.

2. **C2 (D8): threading.Lock blocks event loop during sweep** — Replace with `asyncio.Lock` or use `asyncio.to_thread()` for sweep. All 50 connections stall during sweep.

3. **D6 carried MAJORs (M1-M4): Supply chain hardening** — SBOM, image signing, digest pinning, hash-verified deps. Four rounds with zero progress on table-stakes security for regulated casinos.

### Cross-Validation Notes
- GPT-5.2 Codex: Response truncated (max_output_tokens reached, reasoning-only). Partial findings aligned with D8 concerns.
- Gemini 3.1 Pro (thinking=high): Full hostile review of D8 scalability. Identified both CRITICALs (asyncio task leak, threading.Lock blocking) and confirmed Firestore client lifecycle concern. Called the threading.Lock + event loop interaction "a classic sync-in-async antipattern that R36 introduced while fixing the race condition."
