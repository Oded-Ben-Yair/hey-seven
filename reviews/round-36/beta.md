# R36 Review -- Dimensions 6-10 (Reviewer Beta)

**Date**: 2026-02-23
**Reviewer**: reviewer-beta
**Cross-validation**: GPT-5.2 Codex (3 passes: security, performance, quality) + Gemini 3 Pro (thinking=high)
**Baseline**: R35 scored 6.5-7.5 across D6-D10. R35 fixes applied: Unicode Cf bypass fixed (category-based stripping), TTLCache for langfuse+state_backend, Hindi false positive matra fix, Tagalog BSA "labada" -> "paghuhugas" fix, Parx+Wynn self_exclusion_phone added, patron privacy ES/TL patterns added, get_responsible_gaming_helplines logging added, compliance_gate docstring updated, get_casino_profile DEFAULT_CONFIG deepcopy added.

---

## Dimension 6: Docker & DevOps (weight 0.10)

**Score: 6.5/10** (GPT-5.2: 6.5, Gemini: 6.5, Self: 6.5)

### Strengths
- Multi-stage Dockerfile with non-root user (`appuser`), exec-form CMD, 15s graceful shutdown
- 8-step Cloud Build pipeline: test/lint/mypy -> build -> Trivy scan (pinned 0.58.2) -> push -> capture rollback -> deploy --no-traffic -> smoke test (health + version + agent_ready) -> route traffic
- Rollback health verification with 3x retry + CRITICAL log on double-failure (cloudbuild.yaml:136-148)
- Staging ADR with clear "why not yet" rationale at top of cloudbuild.yaml
- `--min-instances=1`, `--max-instances=10` with `--cpu-boost` and startup/liveness probes
- Separate `/health` (readiness, 503 when CB open) and `/live` (liveness, always 200)
- `requirements-prod.txt` excludes ChromaDB (~200MB dev-only dependency)
- Version assertion in smoke test catches stale containers serving old code

### MAJOR

1. **No SBOM generation or image signing** (`cloudbuild.yaml`)
   - R34+R35 finding, still unresolved. Trivy scan is present but no `cosign sign` or `syft` SBOM generation.
   - Supply chain security gap: cannot verify image provenance after build.
   - GPT-5.2 Codex and Gemini 3 Pro both independently flagged supply-chain hardening as HIGH priority.
   - **Fix**: Add `cosign sign` after push step, `syft` SBOM as build artifact. Or use Cloud Build's built-in provenance (`--requested-verified-attestations`).

2. **No build failure notifications** (`cloudbuild.yaml`)
   - R34+R35 finding, still unresolved. Pipeline failure is silent -- no Slack/email/PagerDuty notification.
   - Team discovers failures only by manually checking Cloud Build console.
   - **Fix**: Add Cloud Build notification channel (Pub/Sub -> Cloud Function -> Slack). GCP has native Cloud Build Notifiers.

3. **Base image pinned by tag, not digest** (`Dockerfile:1,14`)
   - R35 finding, still unresolved. `python:3.12.8-slim-bookworm` is pinned to patch version but not to a specific image digest.
   - Tag drift or Docker Hub compromise could change the base image silently between builds.
   - GPT-5.2 Codex flagged this as HIGH supply-chain risk. Gemini 3 Pro confirmed.
   - **Fix**: Pin to digest: `FROM python:3.12.8-slim-bookworm@sha256:<digest>`. Add Renovate/Dependabot for automated updates.

4. **Python dependencies not hash-verified** (`Dockerfile:11`)
   - R35 finding, still unresolved. `pip install --no-cache-dir --target=/build/deps -r requirements-prod.txt` does not use `--require-hashes`.
   - Dependency confusion or supply chain attack could inject malicious packages.
   - Especially concerning given the AI/LangChain ecosystem's high rate of typo-squatting attacks.
   - **Fix**: Use `pip-compile --generate-hashes` and `pip install --require-hashes`.

### MINOR

5. **`curl` installed in production image** (`Dockerfile:18`)
   - R35 finding, still unresolved. Installed only for HEALTHCHECK (which Cloud Run ignores). Increases attack surface.
   - Gemini 3 Pro flagged as "Living off the Land Binary" -- if RCE is achieved via a LangChain vulnerability, `curl` provides trivial payload download and data exfiltration capability.
   - **Fix**: Use Python stdlib `http.client` for HEALTHCHECK or remove curl entirely since Cloud Run uses its own probes.

6. **No Cloud Build timeout configured** (`cloudbuild.yaml`)
   - No `timeout` field at the global build level or per-step. Default Cloud Build timeout is 10 minutes.
   - The test step (Step 1) runs pytest with full coverage. If a test hangs, the entire build blocks until the default timeout.
   - **Fix**: Add `timeout: 600s` (10min) to global options and `timeout: 300s` per step to catch hangs faster.

---

## Dimension 7: Prompts & Guardrails (weight 0.10)

**Score: 7.0/10** (GPT-5.2: 7.0, Gemini: 7.0, Self: 7.0)

### Strengths
- R35 fixes applied: Unicode Cf category-based stripping, compliance_gate docstring updated to ~173 patterns / 11 languages, patron privacy ES/TL patterns added, get_responsible_gaming_helplines logging added
- 5-layer deterministic guardrails: injection, responsible gaming, age verification, BSA/AML, patron privacy
- ~173 compiled regex patterns across 11 languages (EN, ES, PT, ZH, FR, VI, AR, JP, KO, Hindi, Tagalog)
- Unicode normalization pipeline: Cf removal -> confusable replacement (O(n) str.translate) -> NFKD -> combining mark removal -> whitespace collapse
- Cyrillic (23) + Greek (21) + Fullwidth Latin (26) = 70 confusable mappings
- Semantic injection classifier with fail-closed + 5s asyncio.timeout
- Compliance gate ordering rationale is excellent (injection before content guardrails, semantic last)
- Session-level responsible gaming escalation counter
- string.Template.safe_substitute() for user content in prompts

### MAJOR

1. **Normalization order: confusable replacement before NFKD decomposition** (`src/agent/guardrails.py:374-376`)
   - `str.translate(_CONFUSABLES_TABLE)` runs BEFORE `unicodedata.normalize("NFKD")`. This is suboptimal.
   - An attacker can use precomposed characters with diacritics (e.g., accented Cyrillic/Greek letters) that are NOT in the 70-char confusable table. Step 2 ignores them. Step 3 (NFKD) decomposes them into base character + combining marks. Step 4 removes combining marks. The base confusable character survives in the sanitized string, completely bypassing confusable replacement.
   - GPT-5.2 Codex and Gemini 3 Pro both independently identified this as a normalization order bug.
   - **Fix**: Swap order: run NFKD normalization FIRST, then confusable replacement. Or better: NFKD -> remove combining marks -> confusable replacement -> collapse whitespace. This ensures decomposed characters are handled before confusable mapping.

2. **Confusable table missing Armenian, Cherokee, Latin Extended, and IPA confusables** (`src/agent/guardrails.py:314-350`)
   - The 70-char confusable table covers Cyrillic, Greek, and Fullwidth Latin but misses other scripts with Latin-similar characters:
     - Armenian: `Ꜧ` (U+A726) and others
     - Cherokee: `Ꭺ` (U+13A2, looks like 'A'), `Ꭲ` (U+13A5, looks like 'T')
     - Latin Extended / IPA: `ɑ` (U+0251, looks like 'a'), `ɡ` (U+0261, looks like 'g'), `ı` (U+0131, dotless i)
   - NFKD does NOT decompose these to standard Latin -- they survive the entire normalization pipeline.
   - Gemini 3 Pro confirmed: manual confusable lists are inherently incomplete and will always be stale.
   - **Fix**: Replace the manual table with UTS #39 confusable skeleton logic (`confusable_homoglyphs` library or ICU `SpoofChecker`). Alternatively, add the highest-risk IPA/Latin Extended confusables: `ɑ`->'a', `ɡ`->'g', `ı`->'i', `ɪ`->'i', `ʏ`->'y', `ɴ`->'n', `ʀ`->'r'.

3. **No input length limit before normalization pipeline** (`src/agent/guardrails.py:358-381`)
   - The `_normalize_input()` function runs 5 O(n) passes (Cf check, translate, NFKD, combining check, regex) over the full input with no length cap.
   - An attacker submitting a 1MB Unicode payload forces ~5MB of processing per guardrail check. Combined with 5 guardrail categories, this amplifies to ~25MB of CPU-intensive Unicode processing per message.
   - R35 finding #6 (MINOR) suggested ReDoS protection but not normalization DoS. This is a separate attack vector.
   - GPT-5.2 Codex independently flagged this as a CPU exhaustion risk.
   - **Fix**: Add `if len(message) > 8192: return True` (block oversized input) at the top of `audit_input()` before normalization. Or add the length check inside `_normalize_input()`.

4. **`audit_input()` still in `__all__` with inverted semantics** (`src/agent/guardrails.py:25-27`)
   - R35 finding #5 (MINOR), still unresolved. Returns True=safe, False=injection detected. Opposite of all other `detect_*` functions.
   - The docstring now documents the inversion clearly, but `audit_input` is still in `__all__` as public API.
   - Any external caller using `if audit_input(msg):` (expecting True=detected like other guardrails) would PASS injection attempts.
   - **Fix**: Remove `audit_input` from `__all__` (keep as internal `_audit_input`). External callers should use `detect_prompt_injection()` only.

### MINOR

5. **No ReDoS protection on 173 regex patterns** (`src/agent/guardrails.py`)
   - R35 finding #6, still unresolved. Python's `re` module is vulnerable to catastrophic backtracking.
   - No input length cap before pattern matching (separate from normalization DoS above).
   - **Fix**: Add `if len(message) > 8192: return True` at the top of each `detect_*` function, or use `google-re2` for ReDoS-safe matching.

6. **Normalization skips normalized-check when input is already normalized** (`src/agent/guardrails.py:444`)
   - `if normalized != message:` skips the normalized-text check when input happens to already be in NFC form. While not a bypass (raw-text patterns already ran), it means an attack using pre-normalized confusable text (e.g., NFC Cyrillic `о` = U+043E which IS in the confusable table) might not trigger the "(normalized)" log category, making forensic analysis harder.
   - GPT-5.2 Codex flagged this as error-prone for clarity. Not a security bypass, but a logging gap.
   - **Fix**: Always run patterns on both raw and normalized text for consistent logging coverage.

---

## Dimension 8: Scalability & Production (weight 0.15)

**Score: 5.5/10** (GPT-5.2: 5.5, Gemini: 5.5, Self: 5.5)

### Strengths
- R35 fixes applied: TTLCache for `_get_langfuse_client()` (langfuse_client.py:30) and `get_state_backend()` (state_backend.py:173) with backward-compat shims, `get_casino_profile()` returns `copy.deepcopy(DEFAULT_CONFIG)` for unknown casino_id
- Circuit breaker with rolling window, asyncio.Lock, TTLCache singleton, CancelledError handling, half-open decay recovery
- LLM semaphore(20) for backpressure in `_base.py`
- Pure ASGI middleware (no BaseHTTPMiddleware -- correct for SSE streaming)
- ApiKeyMiddleware with hmac.compare_digest + 60s TTL key refresh
- RequestBodyLimitMiddleware with dual-layer validation
- BoundedMemorySaver with LRU eviction (MAX_ACTIVE_THREADS=1000)
- StateBackend abstraction with InMemory + Redis backends

### CRITICAL

1. **`get_casino_profile()` returns MUTABLE direct reference for KNOWN casinos** (`src/casino/config.py:551-564`)
   - R35 fixed the DEFAULT_CONFIG fallback path (line 563: `copy.deepcopy(DEFAULT_CONFIG)`), but the PRIMARY code path for known casinos still returns a direct reference to `CASINO_PROFILES[casino_id]`.
   - `CASINO_PROFILES.get(casino_id)` returns the dict OBJECT, not a copy. Any caller that mutates the returned dict (e.g., `profile["branding"]["persona_name"] = "Custom"`) corrupts the global profile for ALL subsequent requests across ALL users of that casino.
   - This is confirmed by reading callers: `_base.py:147` does `_casino_profile = get_casino_profile(settings.CASINO_ID)` then `_casino_profile.get("property_description")`. Currently read-only. But `persona.py:179` does `_profile = get_casino_profile(settings.CASINO_ID)` then `branding = _profile.get("branding", {})`. If ANY future code path does `branding["key"] = value`, the global profile is corrupted.
   - GPT-5.2 Codex flagged this as CRITICAL. Gemini 3 Pro confirmed: "cross-tenant data leakage and systemic corruption".
   - **Fix**: Either (a) `return copy.deepcopy(profile)` for ALL paths (both known and unknown), or (b) wrap `CASINO_PROFILES` values in `types.MappingProxyType` at module level to prevent accidental mutation. Option (b) is better (zero-cost, fail-fast on mutation attempt):
     ```python
     CASINO_PROFILES = {k: MappingProxyType(v) for k, v in _raw_profiles.items()}
     ```
     Note: this requires deep-freezing nested dicts too (`regulations`, `branding`, etc.).

### MAJOR

2. **`InMemoryBackend` has no concurrency protection** (`src/state_backend.py:41-117`)
   - R35 finding #4, still unresolved. `InMemoryBackend._store` is a plain `dict` accessed from async coroutines with no lock.
   - `increment()` has a TOCTOU race: read count (line 95), compute +1 (line 96), write -- another coroutine can read the same count in between.
   - Under burst loads, rate limits become ineffective. GPT-5.2 Codex confirmed: "50 concurrent requests will likely only increment counter by 1 or 2, trivially bypassing anti-abuse thresholds."
   - **Fix**: Add `asyncio.Lock` to `InMemoryBackend` and wrap all read-modify-write operations. Or use `threading.Lock` if called from sync contexts.

3. **RateLimitMiddleware stale-client sweep only on request count** (`src/api/middleware.py:429`)
   - R34+R35 finding, still unresolved. Sweep runs every 100 requests (`_request_counter % 100 == 0`).
   - Under low traffic (< 100 req/hour), stale clients accumulate without cleanup for hours. Under traffic spikes with many unique IPs, entries build up between sweeps.
   - GPT-5.2 Codex flagged: "slow memory leak" leading to OOM under high key cardinality.
   - **Fix**: Add time-based sweep fallback: `if time.monotonic() - self._last_sweep > 300:` as secondary trigger alongside the request-count trigger.

4. **Module-level `asyncio.Lock` objects in multiple modules** (`src/casino/config.py:30`, `src/casino/feature_flags.py:109`)
   - `_config_lock = asyncio.Lock()` and `_flag_lock = asyncio.Lock()` are created at module import time.
   - Python 3.10+ `asyncio.Lock` is lazy (binds to event loop on first acquire), so this is not a hard bug. However, if the module is imported from a thread context (e.g., during test discovery), the lock may bind to an unexpected event loop.
   - GPT-5.2 Codex flagged as a maintainability concern. Not a production bug given Python 3.12, but a test fragility risk.
   - **Fix**: Document the Python 3.10+ lazy-binding behavior as a comment, or initialize locks inside an async factory function.

### MINOR

5. **`_LLM_SEMAPHORE` is a module-level global** (`src/agent/agents/_base.py:43`)
   - `asyncio.Semaphore(20)` created at module level. Safe under normal uvicorn operation (single event loop), but if a test creates a new event loop, the semaphore's internal `_waiters` deque is bound to the original loop.
   - **Fix**: Instantiate in the FastAPI lifespan or use a factory function.

6. **Health endpoint calls `asyncio.to_thread(get_retriever)` on every request** (`src/api/app.py:316`)
   - `get_retriever` is already cached, but the `to_thread` call incurs thread pool overhead (~0.1ms) on every health check.
   - With liveness probes every 30s, this is negligible. But if external monitoring tools hit /health frequently, thread pool contention could occur.
   - **Fix**: Cache the `rag_ready` boolean for 60s instead of checking on every health request.

---

## Dimension 9: Trade-off Documentation (weight 0.05)

**Score: 7.5/10** (GPT-5.2: 7.5, Gemini: 7.5, Self: 7.5)

### Strengths
- R35 fixes preserved: Ops runbook (`docs/runbook.md`) at 469 lines covering all operational scenarios
- Cloud Run service config table (port, instances, memory, CPU, concurrency, timeout)
- Probe configuration with /health vs /live rationale
- Build pipeline step-by-step documentation
- Deployment playbook (standard + manual)
- Incident response for 8 scenarios
- Stateful components section with per-component upgrade paths (CB, Rate Limiter, InMemorySaver, Settings Cache, LLM Singleton Caches, Casino Config Cache)
- Alert thresholds table
- Security architecture with middleware stack execution order
- Complete environment variable reference (31 variables)
- Escalation matrix
- Staging ADR in cloudbuild.yaml with clear "why not yet" rationale
- In-memory rate limiting ADR with 3-tier upgrade path
- Degraded-pass validation strategy documented
- Circuit breaker multi-instance limitation documented with 3-tier upgrade path

### MAJOR

1. **Runbook stateful components still missing two entries** (`docs/runbook.md`)
   - R35 finding #1, still unresolved. The stateful components section documents: CB, Rate Limiter, InMemorySaver, Settings Cache, LLM Singleton Caches, Casino Config Cache.
   - **Missing**: `state_backend.py` InMemoryBackend (the pluggable state backend) and `feature_flags.py` flag cache (TTLCache maxsize=100, ttl=300).
   - Both are per-process with the same multi-instance implications as the other documented components.
   - **Fix**: Add both to the stateful components section with their upgrade paths:
     - StateBackend: "memory" (current) -> Redis (production)
     - Feature Flag Cache: TTLCache 5min (current) -> Firestore-backed (production)

### MINOR

2. **Runbook VERSION field still shows "1.0.0"** (`docs/runbook.md:411`)
   - R35 finding #2, still unresolved. Environment variable reference table says `VERSION` default is `1.0.0` but `src/config.py:97` shows default is `1.1.0`.
   - **Fix**: Update runbook to match config.py: `VERSION` default `1.1.0`.

3. **No CHANGELOG.md** (documentation gap)
   - R34+R35 finding, still unresolved. VERSION bumped to 1.1.0 but no changelog tracking what changed.
   - **Fix**: Create CHANGELOG.md following Keep a Changelog format.

4. **Runbook responsible gaming section hardcodes "DMHAS 1-888-789-7777 (Connecticut)"** (`docs/runbook.md:449`)
   - Runbook says "Auto-provides DMHAS 1-888-789-7777 (Connecticut)" but this is only correct for CT properties. NJ provides 1-800-GAMBLER, PA provides 1-800-GAMBLER, NV provides 1-800-MY-RESET.
   - The code correctly dispatches per-state via `get_responsible_gaming_helplines()`, but the runbook gives a CT-only example without noting the multi-state behavior.
   - **Fix**: Update to "Auto-provides state-specific helpline (e.g., CT: 1-888-789-7777, NJ: 1-800-GAMBLER, PA: 1-800-GAMBLER)".

---

## Dimension 10: Domain Intelligence (weight 0.10)

**Score: 7.0/10** (GPT-5.2: 7.0, Gemini: 7.0, Self: 7.0)

### Strengths
- R35 fixes applied: Parx self_exclusion_phone (PGCB 1-855-405-1429), Wynn self_exclusion_phone (NGCB 1-702-486-2000), Tagalog BSA fixed ("paghuhugas ng pera"), Hindi BSA matra required, patron privacy ES/TL patterns added
- 5 casino profiles across 4 states (CT x2, PA, NV, NJ) with real operational data
- Per-casino helplines via `get_responsible_gaming_helplines(casino_id=)` with per-state routing
- TCPA compliance: STOP/HELP/START in EN+ES, quiet hours per timezone, consent hash chain with HMAC-SHA256
- 280+ area code -> timezone mappings with MNP documentation
- BSA/AML patterns in 8 languages (EN, ES, PT, ZH, FR, VI, Hindi, Tagalog) -- 42+ patterns
- Age verification patterns in EN + Hindi + Tagalog
- Tribal gaming jurisdiction documented (Mohegan, Foxwoods compacts)
- NJ-specific SB 3401 push notification ban documented
- Property types distinguished: tribal (Mohegan, Foxwoods) vs commercial (Parx, Wynn, Hard Rock)
- Guest sentiment-gated proactive suggestions (positive-only, R27 fix)
- HEART framework for frustration escalation from message history
- Persona drift prevention with `_PERSONA_REINJECT_THRESHOLD=10`
- Knowledge base: 9 markdown files covering operations, regulations, psychology, company context

### MAJOR

1. **No Korean or Japanese BSA/AML patterns** (`src/agent/guardrails.py`)
   - R35 finding #5 (MINOR), elevating to MAJOR given the NV casino context.
   - Korean and Japanese have injection patterns (in `_NON_LATIN_INJECTION_PATTERNS`) but no BSA/AML, responsible gaming, or patron privacy patterns.
   - Wynn Las Vegas has significant Korean and Japanese high-roller clientele. A Japanese speaker asking `お金を隠す方法` ("how to hide money") or a Korean speaker asking `돈세탁 어떻게` ("how to launder money") bypasses BSA/AML detection entirely.
   - This is an asymmetric coverage gap: injection is caught in KO/JP but financial crime is not.
   - **Fix**: Add at minimum:
     - Japanese BSA/AML: `マネーロンダリング` (money laundering), `お金を隠す` (hide money), `現金.*報告.*避ける` (avoid cash report)
     - Korean BSA/AML: `돈세탁` (money laundering), `돈을 숨기` (hide money), `현금.*보고.*피하` (avoid cash report)

2. **No Korean or Japanese responsible gaming patterns** (`src/agent/guardrails.py`)
   - Same asymmetry as BSA/AML. Japanese and Korean speakers can discuss gambling addiction without triggering responsible gaming detection.
   - Japanese: `ギャンブル依存` (gambling addiction), `パチンコ中毒` (pachinko addiction -- relevant for casino context)
   - Korean: `도박 중독` (gambling addiction), `도박을 그만` (stop gambling)
   - **Fix**: Add 3-4 patterns per language for responsible gaming parity.

3. **Hard Rock AC missing `self_exclusion_options` in prompt delivery** (`src/casino/config.py:504`)
   - Hard Rock AC uniquely has `self_exclusion_options: "1-year, 5-year, or lifetime"` in its config.
   - However, `get_responsible_gaming_helplines()` (prompts.py:34-74) does NOT read or include `self_exclusion_options` in the returned helpline text. This NJ-specific data is defined in the config but never surfaced to the guest.
   - NJ DGE requires informing guests of their self-exclusion duration options.
   - **Fix**: Update `get_responsible_gaming_helplines()` to include `self_exclusion_options` when present: `f"- Self-Exclusion Options: {regulations.get('self_exclusion_options')}"`.

4. **Knowledge base is markdown-only, no structured JSON data files** (`knowledge-base/`)
   - CLAUDE.md states "Knowledge base: 5 JSON data files + 9 markdown files" but only 9 markdown files exist. No JSON data files found.
   - The RAG pipeline's per-item chunking (designed for structured JSON) cannot operate on the markdown files.
   - Either the JSON files were removed/never created, or the CLAUDE.md is stale.
   - **Fix**: Verify whether JSON data files are expected. If so, create them. If not, update CLAUDE.md to "9 markdown files" and verify the RAG pipeline handles markdown-only ingestion correctly.

### MINOR

5. **Wynn Las Vegas branding `exclamation_limit: 0` may produce awkward responses** (`src/casino/config.py:421`)
   - `exclamation_limit: 0` means ALL exclamation marks in LLM responses are replaced with periods.
   - "Welcome to Wynn." is fine, but "Happy birthday." (from a celebration query) loses warmth.
   - The "luxury" tone guide says "Exude refined elegance" which aligns with fewer exclamations, but zero seems overly restrictive.
   - **Fix**: Consider `exclamation_limit: 1` for Wynn (consistent with other properties) unless the brand team explicitly requested zero.

6. **State requirements doc references outdated TCPA one-to-one consent rule** (`knowledge-base/regulations/state-requirements.md:23`)
   - Correctly notes the rule was "VACATED by 11th Circuit (Jan 24, 2025)" but the paragraph structure could confuse a reader. The sentence "One-to-one consent rule:" followed by "VACATED" could be read as the rule existing then being vacated, when in fact it never took effect.
   - **Fix**: Rewrite to: "One-to-one consent rule: NEVER TOOK EFFECT -- vacated by 11th Circuit (Jan 24, 2025, *Insurance Marketing Coalition v. FCC*). Bundled consent remains permissible."

---

## Summary

| Dimension | R35 Score | R36 Score | Change | CRITICALs | MAJORs | MINORs |
|-----------|-----------|-----------|--------|-----------|--------|--------|
| D6: Docker & DevOps | 6.5 | 6.5 | 0 | 0 | 4 | 2 |
| D7: Prompts & Guardrails | 6.5 | 7.0 | +0.5 | 0 | 4 | 2 |
| D8: Scalability & Production | 5.5 | 5.5 | 0 | 1 | 3 | 2 |
| D9: Trade-off Documentation | 7.5 | 7.5 | 0 | 0 | 1 | 3 |
| D10: Domain Intelligence | 7.0 | 7.0 | 0 | 0 | 4 | 2 |
| **Total** | **6.6 avg** | **6.7 avg** | **+0.1** | **1** | **16** | **11** |

### Top Priority Fixes (for fixer)

1. **CRITICAL** `get_casino_profile()` returns mutable direct reference for KNOWN casinos (config.py:551-564) -- deepcopy only for DEFAULT_CONFIG fallback, not for actual casino profiles. Fix: deepcopy all paths or wrap in MappingProxyType.
2. **MAJOR** Normalization order: confusable replacement before NFKD decomposition (guardrails.py:374-376) -- swap order to NFKD first, then confusable replacement.
3. **MAJOR** Confusable table missing Armenian/Cherokee/Latin Extended/IPA characters (guardrails.py:314-350) -- add highest-risk confusables or use UTS #39 library.
4. **MAJOR** No input length limit before normalization pipeline (guardrails.py:358) -- add `if len(message) > 8192: return True` in audit_input().
5. **MAJOR** InMemoryBackend has no concurrency protection (state_backend.py:41-117) -- add asyncio.Lock.
6. **MAJOR** RateLimitMiddleware stale-client sweep still request-count-only (middleware.py:429) -- add time-based sweep fallback.
7. **MAJOR** No Korean/Japanese BSA/AML patterns (guardrails.py) -- add 3 patterns per language.
8. **MAJOR** No Korean/Japanese responsible gaming patterns (guardrails.py) -- add 3-4 patterns per language.
9. **MAJOR** Hard Rock AC self_exclusion_options not surfaced in helpline output (prompts.py:34-74).
10. **MAJOR** `audit_input()` still in `__all__` with inverted semantics (guardrails.py:25-27) -- remove from public API.

### R35 Findings Resolution Status

| R35 Finding | Status | Evidence |
|-------------|--------|----------|
| Unicode Cf category bypass (CRITICAL) | FIXED | guardrails.py:372 category-based stripping |
| TTLCache for langfuse+state_backend (CRITICAL) | FIXED | langfuse_client.py:30, state_backend.py:173 |
| get_casino_profile DEFAULT_CONFIG deepcopy (MAJOR) | FIXED (partial) | config.py:563 deepcopy for DEFAULT -- but known-casino path still mutable |
| Parx+Wynn self_exclusion_phone (MAJOR) | FIXED | config.py:363, 434 |
| Patron privacy ES/TL patterns (MAJOR) | FIXED | guardrails.py:295-303 |
| Tagalog BSA "labada" -> "paghuhugas" (MAJOR) | FIXED | guardrails.py:265 |
| Hindi BSA matra required (MAJOR) | FIXED | guardrails.py:260 |
| compliance_gate docstring updated (MAJOR) | FIXED | compliance_gate.py:8 ~173 patterns |
| get_responsible_gaming_helplines logging (MAJOR) | FIXED | prompts.py:69-73 |
| No SBOM/image signing | NOT FIXED | Still absent in cloudbuild.yaml |
| No build failure notifications | NOT FIXED | Still absent |
| Base image tag not digest | NOT FIXED | Still tag-pinned |
| Python deps not hash-verified | NOT FIXED | Still no --require-hashes |
| RateLimitMiddleware time-based sweep | NOT FIXED | Still request-count only |
| InMemoryBackend no concurrency protection | NOT FIXED | Still no lock |
| audit_input inverted semantics in __all__ | NOT FIXED | Still in __all__ |
| No ReDoS protection | NOT FIXED | Still no input length cap |
| Runbook missing state_backend + feature_flags | NOT FIXED | Still missing |
| Runbook VERSION 1.0.0 -> 1.1.0 | NOT FIXED | Still shows 1.0.0 |
| No CHANGELOG.md | NOT FIXED | Still absent |
| No Korean/Japanese BSA/AML | NOT FIXED | Still absent |

### Cross-Validation Agreement

GPT-5.2 Codex (3 passes) independently confirmed:
- `get_casino_profile()` mutable reference for known casinos as CRITICAL state corruption risk
- Normalization order (confusable before NFKD) as a bypass vector
- Confusable table coverage gaps (Armenian, Cherokee, IPA)
- Input length DoS via normalization pipeline
- InMemoryBackend TOCTOU race condition
- Module-level asyncio.Lock as test fragility risk
- Supply chain hardening gaps (no digest pinning, no hash-verified deps, no SBOM)
- curl as "Living off the Land Binary" post-exploitation tool

Gemini 3 Pro (thinking=high) independently confirmed:
- Normalization order as "classic logic error"
- Manual confusable lists as "inherently incomplete and stale"
- Mutable CASINO_PROFILES reference as "cross-tenant data leakage and systemic corruption"
- InMemoryBackend race condition makes "rate limits practically non-existent under burst loads"
- NFKD correctly handles Mathematical Alphanumeric Symbols, Enclosed Alphanumerics, and Tag characters (Cf)
- Combining Grapheme Joiner (U+034F) correctly caught by Mn removal

GPT-5.2 was more focused on supply-chain security (digest pinning, hash verification, SBOM) and performance (sharded locks, ordered eviction). Gemini 3 Pro was more focused on Unicode normalization theory (UTS #39 compliance, decomposition order) and state mutation analysis.
