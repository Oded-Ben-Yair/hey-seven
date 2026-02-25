# R52 Hostile Review: Grok 4 — D9 Trade-off Docs, D10 Domain Intelligence

**Date**: 2026-02-25
**Reviewer**: Grok 4 (reasoning_effort=high)
**Model**: grok-4
**Dimensions**: D9 (Trade-off Documentation, weight 0.05), D10 (Domain Intelligence, weight 0.10)

---

## Scores

| Dimension | Score | Verdict |
|-----------|-------|---------|
| **D9: Trade-off Documentation** | **5.5 / 10.0** | Needs significant work |
| **D10: Domain Intelligence** | **4.0 / 10.0** | Needs urgent fixes |

---

## D9 Score Justification: 5.5

Trade-off Documentation falls short of excellence due to multiple gaps in completeness and accuracy, which undermine trust in the documentation as a reliable reference. While ADRs are generally well-structured and most runbook sections are present, critical issues like an unindexed ADR (despite it being well-written), a stale ADR index title, an incomplete "Complete Reference" env var table (missing ~35 vars, which is a massive omission), and stale pattern counts indicate sloppy maintenance. Env var defaults that are documented do match the code, but the table's incompleteness defeats the purpose of a "complete" reference. No deferred decisions lack ADRs (e.g., ADR 005 is deferred and documented), but the overall staleness and gaps make this unacceptable without significant work. Harshly scored below 6.0 as it needs major updates for accuracy and completeness.

## D10 Score Justification: 4.0

Domain Intelligence has severe regulatory inaccuracies and incompleteness, which are critical for a casino AI system where errors could lead to legal/compliance risks. Profiles are validated at runtime, multi-property configs exist for 5 casinos with deepcopies handled correctly, and onboarding checklists are detailed -- but regulatory details are outdated (e.g., wrong NV helpline, misleading NV self-exclusion wording, incorrect CT tribal helpline in defaults) and incomplete (e.g., missing Resorts World profile despite CLAUDE.md mention, unvalidated DEFAULT_CONFIG fallback, no codification of "defer to human" for self-exclusion). Onboarding mismatches (e.g., undocumented requirements) and failure to cover new-state updates compound this. These are not minor; they could mislead operators or users on sensitive topics like responsible gaming. Harshly scored well below 6.0 due to critical regulatory flaws needing urgent fixes.

---

## Findings (13 total: 2 CRITICAL, 7 MAJOR, 4 MINOR)

### CRITICAL Findings (2)

#### C1. CT tribal self-exclusion default still references Dept. of Consumer Protection
- **Severity**: CRITICAL
- **File**: `src/agent/prompts.py:29`
- **Issue**: The `RESPONSIBLE_GAMING_HELPLINES_DEFAULT` constant says "CT Self-Exclusion Program: ct.gov/selfexclusion (Dept. of Consumer Protection)". This is inaccurate for CT tribal casinos (Mohegan Sun, Foxwoods), which self-exclude through their own tribal gaming commissions, NOT CT DCP. The R39 fix D10-M003 corrected this in `CASINO_PROFILES` but the DEFAULT constant in `prompts.py` was never updated. Any code path that falls back to this default (e.g., `get_responsible_gaming_helplines()` with no `casino_id` or when profile lookup fails) serves wrong regulatory information.
- **Fix**: Update `RESPONSIBLE_GAMING_HELPLINES_DEFAULT` to use generic/accurate language, or remove the DCP reference entirely. Example: "CT Self-Exclusion: Contact your casino's tribal gaming commission or visit ct.gov/selfexclusion".

#### C2. Outdated NV responsible gaming helpline number
- **Severity**: CRITICAL
- **File**: `src/casino/config.py:443` (wynn_las_vegas profile), `docs/jurisdictional-reference.md:9`
- **Issue**: NV `responsible_gaming_helpline` is "1-800-522-4700" in both the Wynn profile and the jurisdictional reference. NCPG rebranded from 1-800-522-4700 to 1-800-GAMBLER (1-800-426-2537) nationally in 2022. The old number may still route correctly but is officially deprecated. For a production casino system, using an outdated helpline number is a regulatory inaccuracy that could cause compliance issues.
- **Fix**: Update to "1-800-GAMBLER" (or "1-800-GAMBLER (1-800-426-2537)") in both `config.py` wynn_las_vegas profile and `jurisdictional-reference.md` NV section.

### MAJOR Findings (7)

#### M1. ADR index missing ADR-0001-dispatch-srp-refactor
- **Severity**: MAJOR
- **File**: `docs/adr/README.md` (entire file)
- **Issue**: `ADR-0001-dispatch-srp-refactor.md` exists on disk (Accepted, 2026-02-23, covers R43 SRP refactor of `_dispatch_to_specialist`) but is NOT listed in the ADR index. This creates an incomplete ADR catalog -- anyone reading the index would miss this significant architectural decision.
- **Fix**: Add entry to README.md index. Suggested row: `| ADR-0001 | [_dispatch_to_specialist SRP Refactor](ADR-0001-dispatch-srp-refactor.md) | Accepted | graph.py |`

#### M2. "Complete Reference" env var table is incomplete (~35 vars missing)
- **Severity**: MAJOR
- **File**: `docs/runbook.md:545-571` (Environment Variables section)
- **Issue**: The table titled "Environment Variables (Complete Reference)" lists only ~23 vars but `src/config.py` defines ~58 settings. Missing vars include: `LANGFUSE_HOST`, `COMP_COMPLETENESS_THRESHOLD`, `SEMANTIC_INJECTION_ENABLED`, `SEMANTIC_INJECTION_THRESHOLD`, `SEMANTIC_INJECTION_MODEL`, `LLM_SEMAPHORE_TIMEOUT`, `CANARY_ERROR_THRESHOLD`, `CANARY_STAGE_WAIT_SECONDS`, `KMS_KEY_PATH`, `FIRESTORE_PROJECT`, `FIRESTORE_COLLECTION`, `GOOGLE_SHEETS_ID`, `CONSENT_HMAC_SECRET`, `PERSONA_MAX_CHARS`, `TELNYX_API_KEY`, `TELNYX_MESSAGING_PROFILE_ID`, `SMS_FROM_NUMBER`, `TRUSTED_PROXIES`, `MAX_REQUEST_BODY_SIZE`, `REDIS_URL`, `CHROMA_PERSIST_DIR`, `RAG_TOP_K`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_MIN_RELEVANCE_SCORE`, `RRF_K`, `MODEL_TEMPERATURE`, `MODEL_TIMEOUT`, `MODEL_MAX_RETRIES`, `MODEL_MAX_OUTPUT_TOKENS`, `WHISPER_LLM_TEMPERATURE`, `MAX_HISTORY_MESSAGES`, `ENABLE_HITL_INTERRUPT`, `RETRIEVAL_TIMEOUT`, `RATE_LIMIT_MAX_CLIENTS`, `PROPERTY_NAME`, `PROPERTY_WEBSITE`, `PROPERTY_PHONE`, `PROPERTY_STATE`. This misleads operators about the full configuration surface.
- **Fix**: Either expand the table to include ALL env vars from `src/config.py`, or rename from "Complete Reference" to "Key Environment Variables" and add a note pointing to `src/config.py` for the full list.

#### M3. DEFAULT_CONFIG has empty state_helpline, not validated
- **Severity**: MAJOR
- **File**: `src/casino/config.py:146-155`
- **Issue**: `DEFAULT_CONFIG` regulations section has `"state_helpline": ""` (empty string). When an unknown `casino_id` falls back to `DEFAULT_CONFIG` via `get_casino_profile()`, the returned config has no state helpline. The `_validate_profiles_completeness()` function only validates `CASINO_PROFILES` entries, not `DEFAULT_CONFIG` itself. The fallback config would fail the same validation if checked.
- **Fix**: Either populate `DEFAULT_CONFIG` with safe defaults (e.g., "1-800-GAMBLER" national) and extend `_validate_profiles_completeness()` to also check `DEFAULT_CONFIG`, or make the fallback explicitly return a warning message instead of empty strings for regulatory fields.

#### M4. NV self-exclusion wording is misleading
- **Severity**: MAJOR
- **File**: `docs/jurisdictional-reference.md:36`, `src/casino/config.py:449`
- **Issue**: NV self-exclusion says "1-year minimum (revocable after 1 year)". Per NRS 463.368, after the 1-year minimum period the person must petition for reinstatement -- it is not automatically revocable. The wording could mislead guests into thinking their self-exclusion automatically expires. In a responsible gaming context, this misunderstanding could have real consequences.
- **Fix**: Revise to "1-year minimum; requires petition for reinstatement after minimum period" or "1-year minimum (removal requires written petition to NGCB after 1 year)".

#### M5. "Defer to human host" for self-exclusion is not codified
- **Severity**: MAJOR
- **File**: `docs/jurisdictional-reference.md:51`, `src/agent/guardrails.py`, `src/agent/compliance_gate.py`
- **Issue**: The jurisdictional reference states "AI host must defer to human host for all self-exclusion requests" but this behavioral requirement is not codified as a specific guardrail action. Self-exclusion mentions trigger the generic "responsible_gaming" query_type, which auto-provides helplines. There is no explicit "defer to human" mechanism (e.g., escalation to human host, or a distinct `self_exclusion` query_type that triggers escalation).
- **Fix**: Either add a dedicated `self_exclusion` query_type with explicit human escalation, or document that the current responsible_gaming response (providing helplines) is the intentional design for MVP and create an ADR for this trade-off.

#### M6. Onboarding checklist mismatch: ai_disclosure_law vs validation
- **Severity**: MAJOR
- **File**: `docs/casino-onboarding.md:23`, `src/casino/config.py:557-563`
- **Issue**: Onboarding checklist step 1 lists `ai_disclosure_law` as a required field in the regulations section. However, `_REQUIRED_PROFILE_FIELDS` in config.py does NOT include `ai_disclosure_law`. Three of 5 profiles have it empty (`""`) and pass validation. This mismatch means the onboarding doc claims stricter requirements than the code actually enforces.
- **Fix**: Either add `ai_disclosure_law` to `_REQUIRED_PROFILE_FIELDS` (and populate for all profiles), or remove it from the onboarding checklist's "required" column and mark it optional.

#### M7. Onboarding checklist missing jurisdictional-reference.md update step
- **Severity**: MAJOR
- **File**: `docs/casino-onboarding.md` (entire checklist)
- **Issue**: The 7-step onboarding checklist does not include updating `docs/jurisdictional-reference.md` when adding a casino in a new state. If a new casino in Michigan, for example, is onboarded, the jurisdictional reference would remain stale with only CT/NJ/NV/PA coverage.
- **Fix**: Add step 8: "Update docs/jurisdictional-reference.md — add new state section if the casino's state is not already covered (gaming age, self-exclusion authority, helplines, CTR/SAR, alcohol/smoking rules)."

### MINOR Findings (4)

#### m1. ADR 009 index title is stale
- **Severity**: MINOR
- **File**: `docs/adr/README.md:28`
- **Issue**: Index title says "UNSET_SENTINEL as object()" but the actual ADR 009 content describes the evolution to UUID-namespaced string (R49 superseded R48 object() approach). The index title reflects the R48 design, not the current accepted approach.
- **Fix**: Update index title to "UNSET_SENTINEL (UUID-namespaced String)" to match ADR content.

#### m2. Stale regex pattern count in runbook
- **Severity**: MINOR
- **File**: `docs/runbook.md:541`
- **Issue**: Runbook says "204 regex patterns across 12 languages". Actual count of regex r-strings in `guardrails.py` is 209. The count is stale by 5 patterns.
- **Fix**: Update count to 209 (or implement an automated count in CI to prevent future drift).

#### m3. CLAUDE.md mentions "Resorts World" but no profile exists
- **Severity**: MINOR
- **File**: Project `CLAUDE.md` (D10 description)
- **Issue**: CLAUDE.md says "Multi-property casino config (Mohegan Sun, Resorts World, Hard Rock)" but `CASINO_PROFILES` has no `resorts_world` entry. The actual profiles are: mohegan_sun, foxwoods, parx_casino, wynn_las_vegas, hard_rock_ac. CLAUDE.md is inaccurate about which casinos are configured.
- **Fix**: Update CLAUDE.md to list the actual 5 configured casinos, or add a Resorts World profile if it was intended.

#### m4. NFKC vs NFKD inconsistency between CLAUDE.md rules and code
- **Severity**: MINOR
- **File**: CLAUDE.md rules, `src/agent/guardrails.py:509`, `docs/runbook.md:284`
- **Issue**: The global CLAUDE.md rules reference "NFKC" normalization but both the code (`guardrails.py:509`) and runbook correctly use NFKD. NFKD (decomposition) is the correct choice for security normalization as it decomposes characters into base + combining marks, making bypass harder. The code is correct; the external rules doc is wrong.
- **Fix**: Update CLAUDE.md rules to reference NFKD instead of NFKC for alignment with actual implementation.

---

## Summary

| Category | Count |
|----------|-------|
| CRITICAL | 2 |
| MAJOR | 7 |
| MINOR | 4 |
| **Total** | **13** |

### Priority Fix Order

1. **C2** — Update NV helpline to 1-800-GAMBLER (regulatory accuracy, 5-minute fix)
2. **C1** — Fix CT DCP reference in prompts.py default helpline (regulatory accuracy)
3. **M2** — Complete the env var table or rename it (operator trust)
4. **M3** — Validate and populate DEFAULT_CONFIG regulatory fields (fallback safety)
5. **M4** — Fix NV self-exclusion wording (regulatory accuracy)
6. **M5** — Document or implement self-exclusion escalation (compliance gap)
7. **M6** — Align onboarding checklist with code validation (doc-code parity)
8. **M7** — Add jurisdictional-reference update to onboarding (process completeness)
9. **M1** — Add missing ADR to index (catalog completeness)
10. **m1-m4** — Minor fixes (staleness cleanup)
