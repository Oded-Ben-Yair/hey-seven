# R34 Review — Dimensions 6-10 (Reviewer Beta)

**Date**: 2026-02-23
**Reviewer**: reviewer-beta
**Cross-validation**: Grok 4 (reasoning_effort=high) + GPT-5.2 Codex (azure_code_review)
**Baseline**: R33 scored 7.5-8.5 across dimensions. R33 fixes: +15 MAJORs resolved (French/Vietnamese injection+RG patterns, fullwidth Latin confusables, patron privacy regex refined, 112 total patterns, hypothesis property-based tests).

---

## Dimension 6: Docker & DevOps (weight 0.10)

**Score: 6/10** (Grok: 6, GPT: 6)

### Strengths
- Multi-stage Dockerfile with non-root user (`appuser`), exec-form CMD, graceful shutdown (15s)
- 8-step Cloud Build pipeline: test/lint -> build -> Trivy scan -> push -> capture rollback revision -> deploy --no-traffic -> smoke test (health + version assertion + agent_ready) -> route traffic
- requirements-prod.txt excludes ChromaDB (dev-only dependency)
- Staging ADR at top of cloudbuild.yaml documents why branch-based staging not yet implemented
- HEALTHCHECK with curl in Dockerfile
- --min-instances=1, --max-instances=10 with cpu-boost and startup/liveness probes

### CRITICAL

1. **No rollback health validation** (`cloudbuild.yaml`)
   - Step 4 captures `$_ROLLBACK_REVISION` but auto-rollback in step 7 failure does not verify rollback succeeded
   - If rollback itself fails (corrupt revision, quota exhaustion), service stays on broken revision with no alert
   - **Fix**: Add rollback verification step: after `gcloud run services update-traffic --to-revisions=$_ROLLBACK_REVISION=100`, poll health endpoint for 60s, fail-loud if unhealthy

### MAJOR

2. **No SBOM generation or image signing** (`cloudbuild.yaml`)
   - Trivy scan is present but no `cosign sign` or `syft` SBOM generation
   - Supply chain security gap: cannot verify image provenance after build
   - **Fix**: Add `cosign sign` after push step, `syft` SBOM as build artifact

3. **No build failure notifications** (`cloudbuild.yaml`)
   - Pipeline failure is silent — no Slack/email/PagerDuty notification
   - Team discovers failures only by manually checking Cloud Build console
   - **Fix**: Add Cloud Build notification channel (Pub/Sub -> Cloud Function -> Slack)

4. **Per-process rate limiter state fragmentation** (`src/api/middleware.py`)
   - `RateLimitMiddleware` uses in-memory `OrderedDict` — each Cloud Run instance has independent rate limit state
   - 10 max instances = effective rate limit is 10x the configured value (200 req/min instead of 20)
   - ADR documents this with 3-tier upgrade path (Redis -> GCS -> Memorystore) but no implementation timeline
   - **Fix**: Acceptable for MVP if documented as known limitation in ops runbook. Priority for multi-instance production.

### MINOR

5. **Python 3.12.8-slim-bookworm pinning** (`Dockerfile`)
   - Pinned to patch version (good), but no automated base image update mechanism
   - Security patches require manual Dockerfile update + rebuild
   - **Fix**: Consider Dependabot/Renovate for base image updates

---

## Dimension 7: Prompts & Guardrails (weight 0.10)

**Score: 5.5/10** (Grok: 5, GPT: 6)

### Strengths
- 5-layer deterministic guardrails with 112 re.compile() patterns across 9 languages
- Unicode normalization pipeline: zero-width removal -> confusable replacement -> NFKD -> combining mark removal -> whitespace collapse
- Cyrillic (23) + Greek (16) + Fullwidth Latin (26) = 65 confusable mappings
- Semantic injection classifier with fail-closed + 5s asyncio.timeout
- Property-based hypothesis tests for guardrails (200 examples, never crashes on arbitrary input)
- French (5 injection + 3 RG) and Vietnamese (4 injection + 3 RG) patterns added since R33
- Comprehensive test coverage in test_guardrails.py (adversarial bypass tests for each script)

### CRITICAL

1. **Spanish responsible gaming pattern typo** (`src/agent/guardrails.py:130`)
   - Pattern `per[ií]` is missing the letter `d` — should be `perd[ií]` to match "perdi" / "perdí" (Spanish for "I lost")
   - Current pattern matches "peri" (irrelevant word) instead of the intended gambling distress signal
   - `r"(?:per[ií]|perdiendo|arruinado)"` — the `per[ií]` branch is wrong; `perdiendo` and `arruinado` are correct
   - **Fix**: Change `per[ií]` to `perd[ií]` on line 130
   - **Impact**: Spanish-speaking guests expressing gambling distress ("perdi todo mi dinero") bypass responsible gaming detection

### MAJOR

2. **Missing Hindi/Tagalog guardrail patterns** (`src/agent/guardrails.py`)
   - 9 languages covered (EN, ES, PT, ZH, FR, VI, AR, JP, KO) but no Hindi or Tagalog
   - Hindi and Tagalog are significant US casino patron demographics (especially in NV, NJ, CA properties)
   - At minimum, responsible gaming and injection patterns needed for these languages
   - **Fix**: Add Hindi (5-7 RG patterns) and Tagalog (3-5 RG patterns) for responsible gaming layer

3. **Incomplete Greek confusables** (`src/agent/guardrails.py`)
   - Greek lowercase has 7 mappings, uppercase has 9, but missing:
     - Nu (N/v) uppercase: U+039D -> N
     - Pi uppercase: U+03A0 -> (visually similar to TT/H)
     - Rho uppercase: U+03A1 -> P (already mapped as lowercase, but uppercase missing from dict)
   - **Fix**: Add missing Greek uppercase confusables to `_CONFUSABLES` dict

4. **_normalize_input performance** (`src/agent/guardrails.py`)
   - Per-character iteration through `_CONFUSABLES` dict (65 entries) on every input
   - For a 500-char input: 500 * 65 = 32,500 dict lookups per request
   - **Fix**: Use `str.maketrans()` + `str.translate()` for O(n) single-pass replacement instead of O(n*m) nested loop
   - GPT-5.2 specifically flagged this as actionable performance improvement

### MINOR

5. **Non-Latin injection patterns not counted in _INJECTION_PATTERNS** (`src/agent/guardrails.py`)
   - `_INJECTION_PATTERNS` has 11 patterns, `_NON_LATIN_INJECTION_PATTERNS` has 22 patterns
   - `detect_prompt_injection()` checks both lists, but test_doc_accuracy.py only asserts `_INJECTION_PATTERNS == 11`
   - Not a bug (both are checked), but documentation gap — total injection patterns is 33, not 11
   - **Fix**: Add doc-accuracy test for `_NON_LATIN_INJECTION_PATTERNS` count

---

## Dimension 8: Scalability & Production (weight 0.15)

**Score: 5/10** (Grok: 4, GPT: 6)

### Strengths
- Circuit breaker with rolling window, asyncio.Lock for state transitions, TTLCache singleton
- LLM semaphore(20) for backpressure in _base.py
- Concurrent LLM judge calls with Semaphore(5)
- Pure ASGI middleware (no BaseHTTPMiddleware — correct for SSE)
- ApiKeyMiddleware with hmac.compare_digest + 60s TTL key refresh
- RequestBodyLimitMiddleware with dual-layer validation (Content-Length + streaming byte counting)

### CRITICAL

6. **Per-process circuit breaker state fragmentation** (`src/agent/circuit_breaker.py`)
   - Circuit breaker uses `TTLCache(maxsize=1, ttl=3600)` singleton per process
   - With 10 Cloud Run instances, each has independent CB state
   - LLM outage triggers CB open on instance A, but instances B-J continue hammering the failed LLM
   - 10x failure amplification during outages
   - **Fix**: For MVP, document as known limitation. For production: shared CB state via Redis or Firestore with TTL

### MAJOR

7. **get_settings() uses @lru_cache, not TTLCache** (`src/config.py:180`)
   - All other singletons (LLM, CB, retriever) use `TTLCache(ttl=3600)` for credential rotation
   - `get_settings()` uses `@lru_cache(maxsize=1)` which never expires
   - `clear_settings_cache()` exists for manual clearing but requires incident response action
   - Inconsistency: if GCP Workload Identity credentials rotate, settings cache holds stale values until manual clear
   - **Fix**: Replace `@lru_cache` with `TTLCache(maxsize=1, ttl=3600)` to match other singletons, or document the exception with rationale (Settings reads env vars, not rotatable credentials)

8. **RateLimitMiddleware stale-client sweep interval** (`src/api/middleware.py`)
   - Sweep runs every 100 requests (`_sweep_counter % 100 == 0`)
   - Under low traffic (< 100 req/min), stale clients accumulate without cleanup
   - With `RATE_LIMIT_MAX_CLIENTS=10000` this is bounded, but memory grows silently until eviction
   - **Fix**: Add time-based sweep fallback (every 5 minutes regardless of request count)

### MINOR

9. **SecurityHeadersMiddleware nonce allocation** (`src/api/middleware.py`)
   - `secrets.token_urlsafe(16)` per request for CSP nonce
   - Under high load, this adds ~0.1ms per request for CSPRNG
   - Not blocking, but could be pre-generated in batches for hot paths
   - **Fix**: Low priority. Acceptable for current scale.

---

## Dimension 9: Trade-off Documentation (weight 0.05)

**Score: 7/10** (Grok: 7, GPT: 7)

### Strengths
- Staging ADR at top of cloudbuild.yaml (planned branch-based staging, why not yet)
- In-memory rate limiting ADR with 3-tier upgrade path documented in middleware.py
- InMemorySaver ADR with MAX_ACTIVE_THREADS guard documented
- Dual-layer feature flags (build-time topology + runtime behavior) with MappingProxyType documented in feature_flags.py
- ChromaDB dev-only / Vertex AI prod distinction documented in CLAUDE.md
- HITL interrupt trade-off documented (config-toggled, default off)
- Degraded-pass validation strategy documented in langgraph-patterns.md rule

### MAJOR

10. **Missing ops runbook for multi-instance state fragmentation** (documentation gap)
    - CB, rate limiter, and InMemorySaver are all per-process — documented individually but no unified ops runbook
    - Operators need a single document listing all stateful components and their multi-instance behavior
    - **Fix**: Create `docs/ops-runbook.md` with stateful component inventory and expected behavior under scale

### MINOR

11. **Foxwoods missing self-exclusion details** (`src/casino/config.py`)
    - mohegan_sun has full self_exclusion config (authority, url, phone)
    - foxwoods has self_exclusion.authority = "Connecticut DOSR" but missing url and phone
    - Other casinos (parx, wynn, hard_rock_ac) have complete self_exclusion configs
    - **Fix**: Add url and phone to foxwoods self_exclusion config

12. **No version history or changelog** (documentation gap)
    - VERSION bumped to 1.1.0 but no CHANGELOG.md tracking what changed per version
    - Git log is the only history — not accessible to non-developers
    - **Fix**: Create CHANGELOG.md following Keep a Changelog format

---

## Dimension 10: Domain Intelligence (weight 0.10)

**Score: 6.5/10** (Grok: 6, GPT: 7)

### Strengths
- 5 casino profiles across 4 states (CT, PA, NV, NJ) with per-state regulations
- Per-casino helplines via `get_responsible_gaming_helplines(casino_id=)` (R31 fix)
- TCPA compliance: STOP/HELP/START in EN+ES, quiet hours, consent hash chain with HMAC-SHA256
- 280+ area code -> timezone mappings with MNP documentation
- BSA/AML patterns in 4 languages (EN, ES, PT, ZH) — 25 patterns
- Age verification patterns (6) with state-specific minimum ages
- Guest sentiment-gated proactive suggestions (positive-only)
- HEART framework for frustration escalation from message history
- Persona drift prevention with _PERSONA_REINJECT_THRESHOLD=10

### MAJOR

13. **Foxwoods self-exclusion incomplete** (`src/casino/config.py`)
    - foxwoods regulations.self_exclusion has authority but missing url/phone
    - A guest asking "how do I self-exclude at Foxwoods?" gets incomplete information
    - **Fix**: Add Connecticut DOSR self-exclusion URL and phone to foxwoods config

14. **PII redaction exc_info=True may leak data** (`src/api/pii_redaction.py`)
    - `logger.error("PII redaction failed", exc_info=True)` logs full stack trace
    - Stack trace may contain the original text (function arguments in traceback)
    - PII redaction failure + stack trace logging = PII in logs (defeats purpose)
    - **Fix**: Log `logger.error("PII redaction failed for input length=%d", len(text))` without exc_info, or redact the exception message

15. **BSA/AML patterns missing Vietnamese and French** (`src/agent/guardrails.py`)
    - French and Vietnamese were added for injection (5+4) and responsible gaming (3+3) in R33
    - But BSA/AML still only has EN, ES, PT, ZH — no FR or VI
    - Language coverage gap: French/Vietnamese speakers can discuss money laundering without triggering BSA/AML
    - **Fix**: Add French (3-5) and Vietnamese (3-5) BSA/AML patterns

### MINOR

16. **Casino config returns DEFAULT_CONFIG for unknown casino_id** (`src/casino/config.py`)
    - `get_casino_profile(casino_id)` returns `DEFAULT_CONFIG` (Mohegan Sun) for unrecognized IDs
    - Silent fallback to wrong property — no logging or warning
    - **Fix**: Add `logger.warning("Unknown casino_id=%s, falling back to default", casino_id)` on fallback

---

## Summary

| Dimension | Score | CRITICALs | MAJORs | MINORs |
|-----------|-------|-----------|--------|--------|
| D6: Docker & DevOps | 6.0 | 1 | 3 | 1 |
| D7: Prompts & Guardrails | 5.5 | 1 | 3 | 1 |
| D8: Scalability & Production | 5.0 | 1 | 2 | 1 |
| D9: Trade-off Documentation | 7.0 | 0 | 1 | 2 |
| D10: Domain Intelligence | 6.5 | 0 | 3 | 1 |
| **Total** | **6.0 avg** | **3** | **12** | **6** |

### Top Priority Fixes (for fixer)

1. **CRITICAL** Spanish RG pattern typo: `per[ií]` -> `perd[ií]` (guardrails.py:130) — 1-line fix
2. **CRITICAL** Rollback health validation in cloudbuild.yaml — add verification step after rollback
3. **CRITICAL** Per-process CB fragmentation — document as known limitation with mitigation timeline
4. **MAJOR** PII redaction exc_info leak — remove exc_info=True from PII error logging
5. **MAJOR** Incomplete Greek confusables — add missing Nu, Pi uppercase
6. **MAJOR** Missing FR/VI BSA/AML patterns — add 6-10 patterns
7. **MAJOR** Foxwoods self-exclusion url/phone — add missing fields
8. **MAJOR** _normalize_input performance — switch to str.translate()
9. **MAJOR** Hindi/Tagalog RG patterns — add for key demographics
10. **MAJOR** Unknown casino_id silent fallback — add warning log

### Cross-Validation Agreement

Both Grok 4 and GPT-5.2 Codex independently identified:
- Spanish RG pattern typo (CRITICAL)
- Per-process state fragmentation (CB + rate limiter)
- PII exc_info leak risk
- _normalize_input performance concern
- Missing language coverage gaps in guardrails

Grok was more pessimistic on D8 (Scalability) due to multi-instance fragmentation. GPT was more focused on concrete code improvements (str.translate, exc_info). Both agreed D9 (Trade-off Documentation) is the strongest dimension in this group.
