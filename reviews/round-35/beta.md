# R35 Review — Dimensions 6-10 (Reviewer Beta)

**Date**: 2026-02-23
**Reviewer**: reviewer-beta
**Cross-validation**: GPT-5.2 Codex (azure_code_review, 3 passes: security, performance, quality)
**Baseline**: R34 scored 5.0-7.0 across D6-D10. R34 fixes applied: Spanish RG typo, Hindi/Tagalog 48 patterns, Foxwoods self-exclusion, FR/VI BSA/AML, TTLCache for get_settings(), rollback health verification, ops runbook, str.translate() normalization, unknown casino_id warning log.

---

## Dimension 6: Docker & DevOps (weight 0.10)

**Score: 6.5/10** (GPT-5.2: 6.5, Self: 6.5)

### Strengths
- Multi-stage Dockerfile with non-root user (`appuser`), exec-form CMD, 15s graceful shutdown
- 8-step Cloud Build pipeline: test/lint/mypy -> build -> Trivy scan (pinned 0.58.2) -> push -> capture rollback -> deploy --no-traffic -> smoke test (health + version + agent_ready) -> route traffic
- **R34 fix applied**: Rollback health verification now implemented — Step 7 polls health 3x after rollback with explicit `CRITICAL` log on double-failure (lines 136-148 of cloudbuild.yaml)
- Staging ADR at top of cloudbuild.yaml documents planned branch-based staging
- `--min-instances=1`, `--max-instances=10` with `--cpu-boost` and startup/liveness probes
- Separate `/health` (readiness, 503 when CB open) and `/live` (liveness, always 200) — correctly prevents instance cycling during LLM outages
- `requirements-prod.txt` excludes ChromaDB (~200MB dev-only dependency)
- HEALTHCHECK in Dockerfile documented as "Cloud Run ignores this, kept for local docker-compose"
- Version assertion in smoke test catches stale containers serving old code

### MAJOR

1. **No SBOM generation or image signing** (`cloudbuild.yaml`)
   - R34 finding #2 still unresolved. Trivy scan is present but no `cosign sign` or `syft` SBOM generation.
   - Supply chain security gap: cannot verify image provenance after build.
   - GPT-5.2 Codex independently flagged supply-chain hardening as HIGH priority.
   - **Fix**: Add `cosign sign` after push step, `syft` SBOM as build artifact. Or use Cloud Build's built-in provenance (`--requested-verified-attestations`).

2. **No build failure notifications** (`cloudbuild.yaml`)
   - R34 finding #3 still unresolved. Pipeline failure is silent — no Slack/email/PagerDuty notification.
   - Team discovers failures only by manually checking Cloud Build console.
   - **Fix**: Add Cloud Build notification channel (Pub/Sub -> Cloud Function -> Slack). GCP has native Cloud Build Notifiers.

3. **Base image pinned by tag, not digest** (`Dockerfile:1,14`)
   - `python:3.12.8-slim-bookworm` is pinned to patch version but not to a specific image digest.
   - Tag drift or Docker Hub compromise could change the base image silently between builds.
   - GPT-5.2 Codex flagged this as HIGH supply-chain risk.
   - **Fix**: Pin to digest: `FROM python:3.12.8-slim-bookworm@sha256:<digest>`. Add Renovate/Dependabot for automated updates.

4. **Python dependencies not hash-verified** (`Dockerfile:11`)
   - `pip install --no-cache-dir --target=/build/deps -r requirements-prod.txt` does not use `--require-hashes`.
   - Dependency confusion or supply chain attack could inject malicious packages.
   - **Fix**: Use `pip-compile --generate-hashes` and `pip install --require-hashes`.

### MINOR

5. **`curl` installed in production image** (`Dockerfile:18`)
   - Installed only for HEALTHCHECK (which Cloud Run ignores). Increases attack surface.
   - **Fix**: Use Python stdlib `http.client` for HEALTHCHECK or remove curl entirely since Cloud Run uses its own probes.

---

## Dimension 7: Prompts & Guardrails (weight 0.10)

**Score: 6.5/10** (GPT-5.2: 6.5, Self: 6.5)

### Strengths
- **R34 fixes applied**: Spanish RG pattern typo fixed (`perd[ií]` on line 147), Hindi/Tagalog patterns added (48 new patterns), FR/VI BSA/AML patterns added, str.translate() normalization (O(n) single-pass)
- 5-layer deterministic guardrails: injection, responsible gaming, age verification, BSA/AML, patron privacy
- ~166 compiled regex patterns across 11 languages (EN, ES, PT, ZH, FR, VI, AR, JP, KO, Hindi, Tagalog)
- Unicode normalization pipeline: zero-width removal -> confusable replacement (O(n) str.translate) -> NFKD -> combining mark removal -> whitespace collapse
- Cyrillic (23) + Greek (21, including R34 Nu/Rho uppercase fixes) + Fullwidth Latin (26) = 70 confusable mappings
- Semantic injection classifier with fail-closed + 5s asyncio.timeout
- Compliance gate ordering is well-reasoned: injection before content guardrails, semantic classifier last
- Session-level responsible gaming escalation counter (`responsible_gaming_count`)
- string.Template.safe_substitute() for user content in prompts (no format() KeyError DoS)

### CRITICAL

1. **Normalization pipeline misses Unicode format characters (Cf category)** (`src/agent/guardrails.py:358`)
   - Zero-width removal only covers `\u200b-\u200f`, `\u2028-\u202f`, `\ufeff`.
   - **Missing**: `\u2060` (Word Joiner), `\u2066-\u2069` (Bidi Isolates), `\u202A-\u202E` (Bidi Override), `\u180E` (Mongolian Vowel Separator), `\uFFF9-\uFFFB` (Interlinear Annotation).
   - **Bypass vector**: Insert `\u2060` between characters of injection payload: `i\u2060gnore previous instructions` — passes zero-width removal regex, passes injection regex (word broken by invisible char), and is NOT caught by NFKD normalization.
   - GPT-5.2 Codex independently flagged this as the primary normalization bypass.
   - **Fix**: Replace the targeted regex with category-based stripping: `text = "".join(c for c in text if unicodedata.category(c) != "Cf")`

### MAJOR

2. **compliance_gate.py docstring claims "84 patterns across 4 languages"** (`src/agent/compliance_gate.py:7-8`)
   - Actual count is ~166 patterns across 11 languages. The docstring is 6 review rounds stale.
   - R34 added Hindi/Tagalog (48 patterns) + FR/VI BSA/AML (6 patterns) but docstring was not updated.
   - Documentation accuracy matters for audit — a regulator reading this docstring would underestimate the guardrail coverage.
   - **Fix**: Update to "~166 compiled regex patterns across 11 languages" or compute count dynamically.

3. **Patron privacy patterns only in English** (`src/agent/guardrails.py:281-295`)
   - All 10 patron privacy patterns are English-only. No Spanish, Hindi, Tagalog, or other language coverage.
   - A guest asking "donde esta mi esposo?" (where is my husband?) or "nasaan ang asawa ko?" (Tagalog) bypasses patron privacy detection entirely.
   - Other guardrail categories (injection, RG, BSA/AML) have multilingual coverage; patron privacy is the outlier.
   - **Fix**: Add at least Spanish (3-5 patterns) and Tagalog (2-3 patterns) for patron privacy.

4. **`except Exception: pass` in get_responsible_gaming_helplines** (`src/agent/prompts.py:62-63`)
   - Bare except swallows ALL errors silently, including bugs in the helpline lookup logic.
   - If a code regression breaks the per-casino helpline lookup, it silently falls back to CT defaults — NJ guests receive CT helplines without any log warning.
   - This exact pattern caused the R25-R31 multi-tenant helpline bug to persist for 6 rounds.
   - **Fix**: Log the exception: `except Exception: logger.warning("Helpline lookup failed for casino_id=%s", casino_id, exc_info=True)`

### MINOR

5. **`audit_input()` inverted semantics** (`src/agent/guardrails.py:403-436`)
   - Returns True=safe, False=injection detected. Opposite of all other `detect_*` functions.
   - `detect_prompt_injection()` wraps it with `not audit_input()` but `audit_input()` is still in `__all__` as public API.
   - GPT-5.2 Codex flagged this as error-prone for external callers.
   - **Fix**: Remove `audit_input` from `__all__` and add deprecation warning, or rename to `is_input_safe()`.

6. **No ReDoS protection on 166 regex patterns** (`src/agent/guardrails.py`)
   - Python's `re` module is vulnerable to catastrophic backtracking on certain patterns.
   - No input length cap before pattern matching.
   - **Fix**: Add `if len(message) > 8192: return True` (block oversized input) at the top of each `detect_*` function, or use `google-re2` for ReDoS-safe matching.

---

## Dimension 8: Scalability & Production (weight 0.15)

**Score: 5.5/10** (GPT-5.2: 5.5, Self: 5.5)

### Strengths
- **R34 fix applied**: `get_settings()` migrated from `@lru_cache` to `TTLCache(maxsize=1, ttl=3600)` with double-checked locking via `threading.Lock` (correct for synchronous Settings)
- Circuit breaker with rolling window, asyncio.Lock, TTLCache singleton, CancelledError handling
- LLM semaphore(20) for backpressure in `_base.py`
- Pure ASGI middleware (no BaseHTTPMiddleware — correct for SSE streaming)
- ApiKeyMiddleware with hmac.compare_digest + 60s TTL key refresh
- RequestBodyLimitMiddleware with dual-layer validation (Content-Length + streaming byte counting)
- BoundedMemorySaver with LRU eviction (MAX_ACTIVE_THREADS=1000) for dev
- FirestoreSaver integration for production checkpointing
- StateBackend abstraction with InMemory + Redis backends, probabilistic sweep
- Ops runbook with stateful components section (R34 fix) documenting CB, rate limiter, InMemorySaver, settings cache, LLM caches, casino config cache

### CRITICAL

1. **Two remaining `@lru_cache` singletons not migrated to TTLCache** (`src/observability/langfuse_client.py:26`, `src/state_backend.py:169`)
   - `get_settings()` was migrated to TTLCache in R34, but two other singletons still use `@lru_cache(maxsize=1)`:
     - `_get_langfuse_client()` — LangFuse credentials never refresh. If secret rotates, observability breaks permanently until container restart.
     - `get_state_backend()` — State backend never refreshes. If Redis connection drops and `@lru_cache` holds the dead client, all distributed state operations fail permanently.
   - GPT-5.2 Codex independently identified the LangFuse cache inconsistency.
   - **Fix**: Replace both with `TTLCache(maxsize=1, ttl=3600)` + appropriate lock, consistent with all other singleton caches.

### MAJOR

2. **`get_casino_profile()` returns mutable `DEFAULT_CONFIG` reference** (`src/casino/config.py:555`)
   - When `casino_id` is not found, returns `DEFAULT_CONFIG` directly (not a copy).
   - Any caller that mutates the returned dict (e.g., `profile["branding"]["persona_name"] = "Custom"`) corrupts the global default for ALL subsequent calls.
   - `get_casino_config()` (async) correctly uses `copy.deepcopy(DEFAULT_CONFIG)` (line 702), but `get_casino_profile()` (sync) does not.
   - **Fix**: `return copy.deepcopy(DEFAULT_CONFIG)` or wrap `DEFAULT_CONFIG` in `MappingProxyType` (already used for `DEFAULT_FEATURES`).

3. **RateLimitMiddleware stale-client sweep only on request count** (`src/api/middleware.py:429`)
   - Sweep runs every 100 requests (`_request_counter % 100 == 0`).
   - Under low traffic (< 100 req/hour), stale clients accumulate without cleanup for hours.
   - R34 finding #8 still unresolved — no time-based sweep fallback added.
   - **Fix**: Add `if time.monotonic() - self._last_sweep > 300:` as secondary trigger.

4. **InMemoryBackend in state_backend.py has no concurrency protection** (`src/state_backend.py:41-117`)
   - `InMemoryBackend._store` is a plain `dict` accessed from async coroutines with no lock.
   - Under concurrent async requests, `increment()` has a TOCTOU race: read count, compute +1, write — another coroutine can read the same count in between.
   - The `RateLimitMiddleware` has its own `asyncio.Lock`, but `InMemoryBackend` does not.
   - **Fix**: Add `asyncio.Lock` to `InMemoryBackend` or document that it must only be accessed under caller-side locking.

### MINOR

5. **`_get_langfuse_client` backward-compat shim missing** (`src/observability/langfuse_client.py:136-138`)
   - `clear_langfuse_cache()` calls `_get_langfuse_client.cache_clear()` which depends on `@lru_cache` API.
   - If migrated to TTLCache, this call will break unless a shim is added (same pattern as `get_settings.cache_clear`).
   - **Fix**: When migrating, add `_get_langfuse_client.cache_clear = _langfuse_cache.clear`.

---

## Dimension 9: Trade-off Documentation (weight 0.05)

**Score: 7.5/10** (GPT-5.2: 7.5, Self: 7.5)

### Strengths
- **R34 fix applied**: Ops runbook (`docs/runbook.md`) now exists with 469 lines covering:
  - Cloud Run service config table (port, instances, memory, CPU, concurrency, timeout)
  - Probe configuration with /health vs /live rationale
  - Build pipeline step-by-step documentation
  - Deployment playbook (standard + manual)
  - Secrets configuration (GCP Secret Manager)
  - Incident response for 8 scenarios (LLM outage, RAG failure, high error rate, OOM, SSE timeout, validation loop, cold start, rate limit)
  - **Stateful components section** (R34 fix) with per-component upgrade paths
  - Alert thresholds table
  - Security architecture (middleware stack, API key auth, webhook security)
  - Complete environment variable reference (31 variables)
  - Escalation matrix
  - API endpoints reference
- Staging ADR at top of cloudbuild.yaml with clear "why not yet" rationale
- In-memory rate limiting ADR with 3-tier upgrade path in middleware.py
- Dual-layer feature flags documented in feature_flags.py with MappingProxyType
- Degraded-pass validation strategy documented
- HITL interrupt trade-off documented (config-toggled, default off)
- Circuit breaker multi-instance limitation documented with 3-tier upgrade path
- Compliance gate ordering rationale is excellent (why injection before content guardrails, why semantic last)

### MAJOR

1. **Ops runbook stateful components missing two entries** (`docs/runbook.md`)
   - The stateful components section documents: CB, Rate Limiter, InMemorySaver, Settings Cache, LLM Singleton Caches, Casino Config Cache.
   - **Missing**: `state_backend.py` InMemoryBackend (the pluggable state backend) and `feature_flags.py` flag cache (TTLCache maxsize=100, ttl=300).
   - Both are per-process with the same multi-instance implications.
   - **Fix**: Add both to the stateful components section with their upgrade paths.

### MINOR

2. **Runbook VERSION field shows "1.0.0"** (`docs/runbook.md:411`)
   - Environment variable reference table says `VERSION` default is `1.0.0` but `src/config.py:97` shows default is `1.1.0`.
   - **Fix**: Update runbook to match config.py: `VERSION` default `1.1.0`.

3. **No CHANGELOG.md** (documentation gap)
   - R34 finding #12 still unresolved. VERSION bumped to 1.1.0 but no changelog tracking what changed.
   - **Fix**: Create CHANGELOG.md following Keep a Changelog format.

---

## Dimension 10: Domain Intelligence (weight 0.10)

**Score: 7.0/10** (GPT-5.2: 7.0, Self: 7.0)

### Strengths
- 5 casino profiles across 4 states (CT x2, PA, NV, NJ) with real operational data
- **R34 fixes applied**: Foxwoods self-exclusion now complete (url + phone), FR/VI BSA/AML patterns added, unknown casino_id warning log added
- Per-casino helplines via `get_responsible_gaming_helplines(casino_id=)` with per-state routing
- TCPA compliance: STOP/HELP/START in EN+ES, quiet hours per timezone, consent hash chain with HMAC-SHA256
- 280+ area code -> timezone mappings with MNP documentation and fallback strategy
- BSA/AML patterns in 7 languages (EN, ES, PT, ZH, FR, VI, Hindi) — 42+ patterns
- Age verification patterns in EN + Hindi + Tagalog
- Tribal gaming jurisdiction documented in state-requirements.md (Mohegan, Foxwoods compacts)
- NJ-specific SB 3401 push notification ban documented
- Property types distinguished: tribal (Mohegan, Foxwoods) vs commercial (Parx, Wynn, Hard Rock)
- Quiet hours configurable per-state (NV 22:00, others 21:00)
- Guest sentiment-gated proactive suggestions (positive-only)
- HEART framework for frustration escalation from message history
- Persona drift prevention with `_PERSONA_REINJECT_THRESHOLD=10`
- Knowledge base: 5 JSON data files + 9 markdown files covering operations, regulations, psychology, company context

### MAJOR

1. **Parx Casino and Wynn Las Vegas missing `self_exclusion_phone`** (`src/casino/config.py:361-362, 431-432`)
   - Parx has `self_exclusion_authority` (PGCB) and `self_exclusion_url` but no phone number.
   - Wynn has `self_exclusion_authority` (NGCB) and `self_exclusion_url` but no phone number.
   - A guest asking "how do I self-exclude?" at these properties gets incomplete information.
   - Foxwoods R34 fix added url+phone, but the same gap exists for Parx and Wynn.
   - **Fix**: Add Parx `self_exclusion_phone`: PGCB Self-Exclusion Program `1-855-405-1429`. Add Wynn `self_exclusion_phone`: NGCB `1-702-486-2000`.

2. **Tagalog BSA/AML pattern `labada ng pera` is not standard Filipino** (`src/agent/guardrails.py:265`)
   - "Labada" means "to do laundry" (literal). The term "money laundering" in standard Filipino is "paghuhugas ng pera" or the direct loanword "money laundering".
   - "Labada ng pera" is uncommon/incorrect — native speakers would NOT use this phrase for money laundering.
   - The other Tagalog BSA/AML patterns (`paano mag-launder`, `itago ang pera`, etc.) are more accurate.
   - **Fix**: Replace `labada ng pera` with `paghuhugas ng pera` or `money\s+laundering\s+(?:ng|sa)` for Taglish.

3. **Hindi BSA/AML pattern "काल[ाे]?" has regex issue** (`src/agent/guardrails.py:260`)
   - The character class `[ाे]` uses Devanagari vowel signs. The `?` makes the entire class optional, but the regex intent is to match "काला" (kaalaa) or "काले" (kaale).
   - However, `काल` alone means "time/death" — not "black". Without the vowel sign, this pattern matches innocent text about time.
   - **Fix**: Make the vowel sign required: `काल[ाे]\s*(?:धन|पैसे?)` (require at least one matra).

4. **Responsible gaming helplines `except Exception: pass` silently swallows errors** (`src/agent/prompts.py:62-63`)
   - Already flagged in Dim 7, but the domain impact is worse: if any regression breaks the helpline lookup, NJ/PA/NV guests silently receive CT helplines (wrong state, potentially wrong phone numbers).
   - This is a regulatory compliance risk — NJ DGE requires `1-800-GAMBLER`, not CT Council helpline.
   - **(Cross-reference with Dim 7 finding #4)**

### MINOR

5. **No Korean or Japanese BSA/AML patterns** (`src/agent/guardrails.py`)
   - Korean and Japanese have injection patterns (in `_NON_LATIN_INJECTION_PATTERNS`) but no BSA/AML patterns.
   - Korean and Japanese speakers at US casinos (especially NV) could discuss money laundering without triggering BSA/AML detection.
   - **Fix**: Add 2-3 Korean and 2-3 Japanese BSA/AML patterns for coverage parity.

6. **Wynn Las Vegas timezone is `America/Los_Angeles` but NV uses `America/Los_Angeles`** (`src/casino/config.py:435`)
   - This is technically correct (Las Vegas is in Pacific time zone through the `America/Los_Angeles` IANA identifier). No bug, but consider adding a comment: `"timezone": "America/Los_Angeles",  # NV is Pacific time (no state-level IANA ID)` for clarity.

---

## Summary

| Dimension | R34 Score | R35 Score | Change | CRITICALs | MAJORs | MINORs |
|-----------|-----------|-----------|--------|-----------|--------|--------|
| D6: Docker & DevOps | 6.0 | 6.5 | +0.5 | 0 | 4 | 1 |
| D7: Prompts & Guardrails | 5.5 | 6.5 | +1.0 | 1 | 3 | 2 |
| D8: Scalability & Production | 5.0 | 5.5 | +0.5 | 1 | 3 | 1 |
| D9: Trade-off Documentation | 7.0 | 7.5 | +0.5 | 0 | 1 | 2 |
| D10: Domain Intelligence | 6.5 | 7.0 | +0.5 | 0 | 4 | 2 |
| **Total** | **6.0 avg** | **6.6 avg** | **+0.6** | **2** | **15** | **8** |

### Top Priority Fixes (for fixer)

1. **CRITICAL** Unicode Cf category bypass in normalization pipeline (guardrails.py:358) — replace targeted regex with `unicodedata.category(c) != "Cf"` stripping
2. **CRITICAL** Two `@lru_cache` singletons not migrated to TTLCache (langfuse_client.py:26, state_backend.py:169) — inconsistent with all other singletons
3. **MAJOR** `get_casino_profile()` returns mutable `DEFAULT_CONFIG` — add `copy.deepcopy()` or `MappingProxyType`
4. **MAJOR** Parx + Wynn missing `self_exclusion_phone` — add phone numbers
5. **MAJOR** `except Exception: pass` in get_responsible_gaming_helplines — add logging
6. **MAJOR** compliance_gate.py docstring claims "84 patterns across 4 languages" — update to ~166 across 11
7. **MAJOR** Patron privacy patterns English-only — add Spanish + Tagalog
8. **MAJOR** Tagalog BSA/AML "labada ng pera" linguistically incorrect — fix to "paghuhugas ng pera"
9. **MAJOR** Hindi BSA/AML "काल[ाे]?" matches "काल" (time) without vowel sign — make matra required
10. **MAJOR** No SBOM/image signing in CI/CD — add cosign + syft or Cloud Build provenance

### R34 Findings Resolution Status

| R34 Finding | Status | Evidence |
|-------------|--------|----------|
| Spanish RG typo `per[ií]` -> `perd[ií]` | FIXED | guardrails.py:147 |
| Rollback health validation | FIXED | cloudbuild.yaml:136-148 |
| Per-process CB fragmentation doc | FIXED | runbook.md stateful components |
| No SBOM/image signing | NOT FIXED | Still absent in cloudbuild.yaml |
| No build failure notifications | NOT FIXED | Still absent |
| Missing Hindi/Tagalog patterns | FIXED | 48 patterns added |
| Incomplete Greek confusables | FIXED | Nu (U+039D) + Rho (U+03A1) added |
| `_normalize_input` O(n*m) performance | FIXED | str.translate() on line 360 |
| Missing FR/VI BSA/AML patterns | FIXED | 6 patterns added |
| Foxwoods self-exclusion url/phone | FIXED | config.py:291-292 |
| Unknown casino_id silent fallback | FIXED | warning log on line 551-554 |
| get_settings() @lru_cache -> TTLCache | FIXED | config.py:181 TTLCache |
| Stale-client sweep time-based fallback | NOT FIXED | Still request-count only |
| PII exc_info leak | FIXED | pii_redaction.py:107-113 uses length only |
| Ops runbook | FIXED | 469-line runbook.md |

### Cross-Validation Agreement

GPT-5.2 Codex (3 passes) independently confirmed:
- Unicode format character bypass (Cf category) as primary normalization gap
- `@lru_cache` -> TTLCache inconsistency for Langfuse client
- Supply chain hardening gaps (no digest pinning, no hash-verified deps, no SBOM)
- `audit_input()` inverted semantics as maintainability risk
- Mutable `DEFAULT_CONFIG` return as correctness risk
- RateLimitMiddleware global lock contention under scale
- Per-process state fragmentation as architectural limitation

GPT-5.2 was more focused on supply-chain security (digest pinning, hash verification) and ReDoS risk. Self-review was more focused on domain accuracy (Tagalog linguistics, Hindi regex correctness, self-exclusion completeness).
