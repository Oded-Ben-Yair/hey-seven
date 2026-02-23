# R39 Hostile Review: Dimensions 6-10

**Reviewer**: reviewer-beta
**Date**: 2026-02-23
**Baseline**: R38 scores (D6=7.0, D7=7.0, D8=6.5, D9=7.5, D10=7.5)
**Cross-Validation**: GPT-5.2 Codex (security), Gemini 3.1 Pro (performance)

---

## D6: Docker & DevOps — Score: 7.5 (+0.5)

### Strengths
- 8-step Cloud Build pipeline with --no-traffic canary, smoke test gate, automatic rollback with health verification (cloudbuild.yaml:95-156)
- SHA-256 pinned base image in Dockerfile, non-root user, exec-form CMD
- Trivy scan pinned to 0.58.2 for reproducible builds (cloudbuild.yaml:35-43)
- Version verification in smoke test prevents stale container deployment (cloudbuild.yaml:112-118)
- Per-step timeouts prevent hung pipeline stages (cloudbuild.yaml:18,31,36,47)
- Staging strategy ADR documents the conscious trade-off and upgrade path (cloudbuild.yaml:1-12)

### Findings

**MAJOR M-001: Smoke test does not validate /chat endpoint functionality**
The smoke test (cloudbuild.yaml:95-156) checks /health, version, and agent_ready. But it never sends a test message to /chat. A deployment could pass smoke test with a broken LLM connection (e.g., wrong GOOGLE_API_KEY secret version). The agent is "ready" (graph compiled) but non-functional (LLM unreachable).
- Severity: MAJOR
- Fix: Add a /chat smoke test with a simple greeting query ("Hi") and verify a non-error SSE response is returned within 30s.

**MAJOR M-002: No container resource limits in Dockerfile**
Dockerfile has no `--ulimit`, no `--pids-limit`, no memory constraints. Cloud Run sets these externally via deploy flags, but local `docker run` has no protection. A developer running `docker run` locally could exhaust host resources.
- Severity: MAJOR
- Fix: Add resource constraints to Makefile docker-run target or document that Cloud Run handles limits.

**MINOR m-001: pip install in test step not using lockfile**
cloudbuild.yaml:22 runs `pip install --no-cache-dir -r requirements-dev.txt`. No `--constraint` or lockfile ensures deterministic versions across builds. A transitive dependency update could break the build non-deterministically.
- Severity: MINOR
- Fix: Generate `requirements-dev.lock` with `pip-compile` and use it in CI.

---

## D7: Prompts & Guardrails — Score: 7.5 (+0.5)

### Strengths
- 5-layer deterministic guardrails with 11-language coverage (EN, ES, PT, ZH, FR, VI, AR, JP, KO, Hindi, Tagalog)
- ~185 compiled regex patterns covering injection, responsible gaming, age verification, BSA/AML, patron privacy
- Input normalization pipeline: URL decode -> HTML unescape -> strip Cf chars -> NFKD -> strip combining marks -> confusable table -> strip delimiters -> collapse whitespace (guardrails.py:386-422)
- Semantic injection classifier with 5s timeout and fail-closed behavior (guardrails.py:581-649)
- 8-step compliance gate with documented priority ordering (compliance_gate.py)
- PII redaction fails closed with "[PII_REDACTION_ERROR]" placeholder
- Streaming PII redactor operates on redacted buffer to prevent length misalignment (streaming_pii.py:124-134)
- Confusable table covers Cyrillic, Greek, Fullwidth Latin, IPA/Latin Extended with O(n) str.maketrans (guardrails.py:333-383)

### Findings

**CRITICAL C-001: Single-pass URL decoding allows double-encoding bypass**
`_normalize_input()` calls `urllib.parse.unquote(text)` exactly once (guardrails.py:398). Double-encoded payloads bypass this completely:
- Input: `ignore%2520previous%2520instructions`
- After single unquote: `ignore%20previous%20instructions` (still encoded)
- The regex sees `%20` not spaces, so the pattern `ignore\s+(all\s+)?(previous|prior|above)` does NOT match.
A second unquote would produce: `ignore previous instructions` which WOULD match.
- GPT-5.2 Codex cross-validation independently flagged this as security gap #1.
- Severity: CRITICAL
- Fix: Iterative decode with a 3-round cap: `for _ in range(3): new = unquote(html.unescape(text)); if new == text: break; text = new`

**CRITICAL C-002: unquote() does not decode form-encoded `+` as space**
`urllib.parse.unquote()` leaves `+` intact (guardrails.py:398). `urllib.parse.unquote_plus()` decodes `+` as space. An attacker can send `ignore+previous+instructions` and the injection pattern fails because `\s+` does not match `+`.
- GPT-5.2 Codex cross-validation independently flagged this as security gap #2.
- Severity: CRITICAL
- Fix: Replace `urllib.parse.unquote(text)` with `urllib.parse.unquote_plus(text)`.

**MAJOR M-001: Delimiter stripping is too narrow**
`re.sub(r"(?<=\w)[._\-](?=\w)", "", text)` (guardrails.py:419) only strips `.`, `_`, `-`. Attackers can use other delimiters: `i:g:n:o:r:e`, `i/g/n/o/r/e`, `i;g;n;o;r;e`, `i~g~n~o~r~e`, `i|g|n|o|r|e`.
- GPT-5.2 Codex cross-validation flagged this as gap #5.
- Severity: MAJOR
- Fix: Expand the character class to cover all ASCII punctuation: `r"(?<=\w)[^\w\s](?=\w)"` or use Unicode-aware `[\p{P}\p{S}]` with the regex module.

**MAJOR M-002: Non-Latin injection patterns not checked on normalized input**
`_audit_input()` checks `_NON_LATIN_INJECTION_PATTERNS` only on raw `message` (guardrails.py:494), not on `normalized`. If normalization alters the text (e.g., stripping Cf chars around Arabic/Japanese text), the patterns might not match the original but would match the normalized form.
- GPT-5.2 Codex cross-validation flagged this as gap #6.
- Severity: MAJOR
- Fix: Also run `_check_patterns(normalized, _NON_LATIN_INJECTION_PATTERNS, ...)` when `normalized != message`.

**MAJOR M-003: No post-normalization length check allows expansion-based DoS**
NFKD normalization can expand input (e.g., ligatures decompose). The 8192-char limit (guardrails.py:482) is checked BEFORE normalization. A crafted input of 8191 ligature characters could expand to 16K+ after NFKD, creating expensive regex operations.
- GPT-5.2 Codex cross-validation flagged this as gap #3.
- Severity: MAJOR
- Fix: Add `if len(normalized) > 8192: return False` after `_normalize_input()`.

**MINOR m-001: _audit_input inverted semantics is a footgun**
The function returns True for safe, False for injection. The docstring warns about this, and `detect_prompt_injection()` wraps it. But `audit_input = _audit_input` (guardrails.py:513) preserves the public alias with inverted semantics.
- Severity: MINOR
- Fix: Remove `audit_input` alias entirely. Any caller should use `detect_prompt_injection()`.

---

## D8: Scalability & Production — Score: 7.0 (+0.5)

### Strengths
- Pure ASGI middleware (not BaseHTTPMiddleware) preserves SSE streaming (middleware.py:1-5)
- Per-client asyncio.Lock rate limiting (R38 fix from global lock bottleneck) (middleware.py:364-372)
- Background sweep task decoupled from request path (middleware.py:427-458)
- Circuit breaker with rolling time window and half-open recovery with decay (circuit_breaker.py)
- TTLCache singletons for credential rotation (nodes.py, config.py)
- Client disconnect detection via asyncio.CancelledError (ErrorHandlingMiddleware:143-152)
- IP normalization for consistent rate keying (middleware.py:383-397)
- 3-tier ADR for rate limiting upgrade path (middleware.py:340-351)
- Cloud Run probes: startup (/health) + liveness (/live) properly configured

### Findings

**CRITICAL C-001: asyncio.Lock is unnecessary in single-threaded asyncio**

**Theoretical Max Concurrent Users Calculation:**
- Cloud Run: --concurrency=50, --max-instances=10 = 500 max TCP connections
- Each SSE stream holds one connection for its duration (30-120s)
- Each /chat request triggers 1-6 LLM calls (router + optional specialist dispatch + generate + optional validate + retry)
- LLM calls: ~2-15s each, no visible asyncio.Semaphore limiting concurrency
- Firestore: AsyncClient with default connection pool (not explicitly sized)
- Memory: 2Gi per instance. Each SSE stream: ~1-5 MB (messages + context + buffers)
- **Theoretical max: 500 connections, ~50-100 active LLM calls simultaneously** (bottleneck: LLM API rate limits, not application)

The per-client `asyncio.Lock` approach (middleware.py:364-372) is architecturally unnecessary. Python asyncio is single-threaded cooperative multitasking. Context switches only happen at `await` points. Inside the `_is_allowed` method, the deque operations (popleft, append, len) have ZERO await points, making them inherently atomic. The locks add:
- 10,000 Lock objects in memory at max_clients capacity
- Event loop scheduling overhead per lock acquire/release
- Gemini 3.1 Pro cross-validation scored this 4/10 and called it a "fundamental misunderstanding of how the Python event loop executes code."

However: the `_requests_lock` IS needed for the structural dict mutation path (adding new keys, LRU eviction) because `await self._ensure_sweep_task()` can cause a context switch between checking `client_ip not in self._requests` and the dict mutation. So the _requests_lock is correct; the _client_locks are unnecessary overhead.
- Severity: CRITICAL (performance penalty at scale)
- Fix: Remove `_client_locks` dict entirely. The per-client deque ops are atomic in single-threaded asyncio. Keep `_requests_lock` for dict structural mutations only.

**MAJOR M-001: No asyncio.Semaphore for LLM call concurrency control**
There is no visible Semaphore limiting concurrent LLM calls. With --concurrency=50 per instance and 1-6 LLM calls per request, a burst of 50 simultaneous /chat requests could trigger 50-300 concurrent LLM API calls. This will hit Gemini API rate limits and cause cascading timeouts.
- Severity: MAJOR
- Fix: Add `asyncio.Semaphore(20)` for LLM calls in `_get_llm()` or `execute_specialist()` to bound concurrent API calls per instance.

**MAJOR M-002: max_clients=10000 mismatched with concurrency=50**
`RATE_LIMIT_MAX_CLIENTS=10000` (config.py:49) means the rate limiter stores up to 10,000 IP entries per instance. But each instance handles max 50 concurrent connections. The 10,000 limit is 200x the actual connection capacity, wasting ~2-4 MB of memory for deques and locks that will never be concurrently active.
- Severity: MAJOR
- Fix: Set `RATE_LIMIT_MAX_CLIENTS` to `1000` (20x concurrency, adequate buffer for unique IPs within the 60s window).

**MAJOR M-003: In-memory rate limiting provides no cross-instance protection**
Each Cloud Run instance maintains independent counters (documented in ADR). With max-instances=10, effective rate limit = 200 req/min/IP (not 20). The ADR acknowledges this as a demo limitation, but no progress toward Cloud Armor migration is visible.
- Severity: MAJOR (known, but still a gap)
- Fix: Add Cloud Armor rate limiting policy (zero code change, GCP-native) or document concrete timeline.

**MINOR m-001: No readiness probe distinct from startup probe**
Cloud Run config has startup-probe-path=/health and liveness-probe-path=/live. But there's no readiness probe. During instance draining (graceful shutdown), the instance should stop accepting new connections while completing in-flight SSE streams. Without a readiness probe, Cloud Run may route new connections to a draining instance.
- Severity: MINOR
- Fix: Add /ready endpoint that returns 503 when shutdown signal received but before process exit.

---

## D9: Trade-off Documentation — Score: 7.5 (0.0)

### Strengths
- Rate limiting ADR with 3-tier upgrade path (middleware.py:317-351)
- Staging strategy ADR with MVP justification and planned production path (cloudbuild.yaml:1-12)
- Degraded-pass validation strategy documented inline with first/retry attempt rationale (nodes.py)
- Circuit breaker rolling window ADR (circuit_breaker.py)
- Feature flags dual-layer design (build-time topology vs runtime behavior) with cross-module parity check (feature_flags.py)
- Multi-property architecture documented with Firestore async config and TTL cache (config.py)
- Semantic classifier fail-closed rationale documented (guardrails.py:591-605)
- --allow-unauthenticated defense-in-depth explanation (cloudbuild.yaml:61-65)

### Findings

**MAJOR M-001: No ADR for LLM concurrency limits**
The system lacks documentation of how many concurrent LLM calls a single instance can sustain. Cloud Run --concurrency=50 is documented, but the relationship between HTTP concurrency and LLM API concurrency is never analyzed. This is the most important scalability trade-off for an LLM-based system.
- Severity: MAJOR
- Fix: Document: (1) Gemini API rate limits per project, (2) expected LLM calls per /chat request (1-6), (3) max concurrent LLM calls = concurrency * avg_calls_per_request, (4) whether a Semaphore is needed.

**MAJOR M-002: No ADR for checkpointer choice trade-offs**
The system uses MemorySaver (dev) and FirestoreSaver (prod) for conversation state. But there's no documentation of:
- FirestoreSaver read/write latency impact on graph execution
- Document size limits (Firestore 1MB max) vs conversation history growth
- Cost implications per active conversation
- Comparison with Redis-based alternatives
- Severity: MAJOR
- Fix: Add ADR in graph.py or memory.py documenting checkpointer choice rationale.

**MINOR m-001: Feature flag documentation lacks per-flag rationale**
feature_flags.py documents the dual-layer design pattern but not WHY each of the 12 flags has its specific default. For example, `outbound_campaigns_enabled: False` and `sms_enabled: False` suggest incomplete features, but the rationale is not documented.
- Severity: MINOR

---

## D10: Domain Intelligence — Score: 7.5 (0.0)

### Strengths
- 5 real casino profiles with property-specific branding, regulations, and operational config (config.py:178-535)
- State-specific responsible gaming helplines: CT (1-800-MY-RESET), PA (1-800-GAMBLER), NV (1-800-MY-RESET/1-800-GAMBLER), NJ (1-800-GAMBLER) (config.py)
- Self-exclusion authorities with phone numbers and URLs for all 5 states (config.py)
- NJ self-exclusion options (1-year, 5-year, or lifetime) only in hard_rock_ac profile (config.py:504)
- Per-casino persona names: Seven (Mohegan), Foxy (Foxwoods), Lucky (Parx), Wynn Host (Wynn), Ace (Hard Rock) (config.py)
- Wynn Las Vegas correctly uses "formal" formality_level and exclamation_limit=0 (luxury tier) (config.py:417-422)
- Tribal vs commercial property_type correctly assigned (config.py:231,301,373,444,519)
- deepcopy prevents global mutation on both known and unknown casino IDs (config.py:560-564)

### Findings

**MAJOR M-001: self_exclusion_options missing from 4 of 5 casino profiles**
Only `hard_rock_ac` (NJ) has `self_exclusion_options: "1-year, 5-year, or lifetime"` (config.py:504). The other 4 profiles lack this field entirely. Every state has specific self-exclusion durations:
- CT: 1-year minimum (can request longer)
- PA: 1-year, 5-year, or lifetime
- NV: 1-year minimum (revocable after 1 year), or lifetime (irrevocable)
Without `self_exclusion_options`, the agent cannot accurately inform guests about their self-exclusion choices when they request it.
- Severity: MAJOR
- Fix: Add `self_exclusion_options` to all 5 profiles with state-accurate durations.

**MAJOR M-002: Wynn Las Vegas responsible_gaming_helpline may be incorrect**
Wynn Las Vegas (NV) has `responsible_gaming_helpline: "1-800-MY-RESET"` (config.py:430). The 1-800-MY-RESET number is the Connecticut Council on Problem Gambling helpline, specific to CT. Nevada's standard helpline is 1-800-522-4700 (National Council on Problem Gambling) or the state_helpline 1-800-GAMBLER. Using CT's helpline for a NV property would give guests a helpline that may not be equipped for NV self-exclusion processes.
- Severity: MAJOR
- Fix: Verify the correct NV responsible gaming helpline. Standard for NV commercial casinos is 1-800-522-4700 (NCPG national hotline) or 1-800-GAMBLER.

**MAJOR M-003: No tribal gaming compact-specific language**
Mohegan Sun and Foxwoods are tribal casinos operating under compacts with the state of CT. Tribal casinos have unique regulatory nuances:
- Self-exclusion is through the Mohegan Tribal Gaming Commission / Mashantucket Pequot Tribal Nation, NOT CT DCP
- The `self_exclusion_authority: "CT Department of Consumer Protection"` (config.py:219,290) is incorrect for tribal casinos. CT DCP handles commercial gaming; tribal casinos have their own gaming commissions.
- AI disclosure law "CT SB 2" may not apply to tribal sovereign land operations
This could give guests incorrect information about the self-exclusion process.
- Severity: MAJOR
- Fix: Update Mohegan Sun to `self_exclusion_authority: "Mohegan Tribal Gaming Commission"` and Foxwoods to `self_exclusion_authority: "Mashantucket Pequot Tribal Nation Gaming Commission"`. Verify CT SB 2 applicability to tribal operations.

**MINOR m-001: No gaming-specific operational data per property**
Casino profiles have `property_size_gaming_sqft` and `dining_venues` but lack:
- Table game count / slot count
- Poker room availability
- Sportsbook availability (Parx has Xcite Center sportsbook; Wynn has Wynn Sports)
- VIP/high-limit room availability
These are common guest questions that the agent should be able to answer.
- Severity: MINOR
- Fix: Add gaming_tables, slot_machines, has_poker_room, has_sportsbook fields to operational config.

---

## Score Summary

| Dimension | R38 Score | R39 Score | Delta | Findings |
|-----------|-----------|-----------|-------|----------|
| D6: Docker & DevOps | 7.0 | 7.5 | +0.5 | 2 MAJOR, 1 MINOR |
| D7: Prompts & Guardrails | 7.0 | 7.5 | +0.5 | 2 CRITICAL, 3 MAJOR, 1 MINOR |
| D8: Scalability & Production | 6.5 | 7.0 | +0.5 | 1 CRITICAL, 3 MAJOR, 1 MINOR |
| D9: Trade-off Documentation | 7.5 | 7.5 | 0.0 | 2 MAJOR, 1 MINOR |
| D10: Domain Intelligence | 7.5 | 7.5 | 0.0 | 3 MAJOR, 1 MINOR |

**Totals**: 3 CRITICAL, 13 MAJOR, 5 MINOR

### Critical Findings Summary
1. **D7-C001**: Double-encoding bypass in _normalize_input (single-pass unquote)
2. **D7-C002**: Form-encoded `+` not decoded as space (unquote vs unquote_plus)
3. **D8-C001**: Unnecessary per-client asyncio.Lock objects (10K Lock overhead)

### Cross-Validation Agreement
- GPT-5.2 Codex independently identified D7-C001, D7-C002, D7-M001, D7-M002, D7-M003 (5/5 match)
- Gemini 3.1 Pro independently identified D8-C001 and the max_clients mismatch, scored rate limiter 4/10
