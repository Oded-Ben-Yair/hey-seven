# R69 Grok Review -- D9, D10

## Date: 2026-02-26
## R68 Baseline: D9=9.0, D10=9.0

---

## D9: Trade-off Documentation (weight 0.05)

**Score: 7.5/10**

### Findings

#### MAJOR

- **[MAJOR] docs/runbook.md:577 -- Pattern count drift (204 vs 206).**
  Runbook claims "204 regex patterns across 12 languages" but `src/agent/guardrails.py` contains 206 `regex_engine.compile` calls (verified via grep count). This is a 3-way count drift: docs say one number, code has another. Any security metric claim in docs must match code exactly, or the documentation cannot be trusted for compliance audits. Add `assert len(patterns) == N` test and reference it in docs.

- **[MAJOR] docs/runbook.md:629 -- Helpline primary/alternate contradiction.**
  Runbook's "Responsible Gaming Escalation" section says: "Problem gambling: Auto-provides 1-800-MY-RESET (NCPG)." But `src/agent/prompts.py:27-30` (R68 fix D10) explicitly changed the primary NCPG helpline to **1-800-GAMBLER**, demoting 1-800-MY-RESET to alternate. The runbook is stale post-R68 and describes the wrong primary helpline. An operator reading the runbook would believe 1-800-MY-RESET is the primary NCPG number, contradicting R68's own fix rationale.

- **[MAJOR] src/config.py:106 -- Stale VERSION string.**
  `VERSION: str = "1.1.0"` but project is at v1.3.0 per CLAUDE.md and git commit history. The VERSION field is exposed via `/health` endpoint in development mode. While production overrides with `$COMMIT_SHA`, the default value creates confusion: any local dev testing, any Dockerfile HEALTHCHECK, or any staging environment running without the override will report the wrong version. VERSION should be bumped with every release, not left stale across 3 minor versions.

- **[MAJOR] docs/runbook.md:670-675 -- Regulatory quick reference table incomplete.**
  The "Per-State Regulatory Quick Reference" table lists only `state_helpline` per state (e.g., CT: 1-888-789-7777) but omits the `responsible_gaming_helpline` field entirely. In code, CT casinos have `responsible_gaming_helpline: "1-800-MY-RESET"` which is a DIFFERENT number from `state_helpline: "1-888-789-7777"`. The table is incomplete and could mislead an operator into thinking the state_helpline is the only helpline configured per jurisdiction. Add a column for `responsible_gaming_helpline`.

#### MINOR

- **[MINOR] ADR-019 (docs/adr/019-single-tenant-deployment.md) -- Missing fallback jurisdiction risk.**
  ADR-019 documents the single-tenant-per-deployment decision but does not mention the regulatory risk of DEFAULT_CONFIG falling back to CT jurisdiction (config.py:147, state="CT"). An unknown casino_id silently receives CT-specific helplines and regulations. The code warns at runtime (config.py:691-696 logger.warning) but the ADR should document this as an accepted risk with mitigation.

- **[MINOR] docs/runbook.md:569 vs guardrails.py:29-37 -- "Six" vs seven exports.**
  Runbook says "Six deterministic guardrail layers" but `guardrails.py.__all__` exports 7 functions. The "six categories" interpretation is defensible (classify_injection_semantic is Layer 2 of injection, not a separate category), but the language should be "six categories" not "six layers" since the semantic classifier IS a separate layer (regex + semantic = 2 layers for injection alone). Phrasing invites confusion.

- **[MINOR] ADR-018 title says "135 entries" -- VERIFIED CORRECT.**
  Code has exactly 135 confusable entries in `_CONFUSABLES` dict. Title matches. However, the ADR body originally said "~110 entries" in earlier rounds; the title was updated to 135 but the ADR body text still says "maps 135 highest-risk confusables" which is now consistent. No issue.

### D9 Summary

Strengths: 22 ADRs with proper status lifecycle (Proposed/Accepted/Deferred/Superseded/Deprecated) and review dates. Comprehensive runbook (739 lines) covering 15+ incident scenarios, probe configs, canary deployment, secret rotation, and a 10-step onboarding checklist. ADR-022 adds regulatory risk rationale with enforcement precedent. ADRs reference source code locations.

Weaknesses: Multiple doc-code parity drifts (pattern count, helpline hierarchy, version string, regulatory table completeness). The runbook was not fully updated after R68 fixes, creating contradictions between the runbook's helpline section and the code's R68 fix. Documentation that contradicts code is WORSE than no documentation.

---

## D10: Domain Intelligence (weight 0.10)

**Score: 6.5/10**

### Findings

#### CRITICAL

- **[CRITICAL] config.py:228, config.py:314 -- Tribal self_exclusion_url points to state page.**
  Both Mohegan Sun (`self_exclusion_url: "ct.gov/selfexclusion"`) and Foxwoods (`self_exclusion_url: "ct.gov/selfexclusion"`) point guests to the Connecticut STATE self-exclusion page. But the code's own comments (config.py:225-226, 311-312) correctly state: "Tribal casinos self-exclude through their own gaming commissions, NOT CT DCP." The ct.gov/selfexclusion page is for CT DCP-regulated commercial gaming venues. Tribal self-exclusion is handled by the Mohegan Tribal Gaming Commission and the Mashantucket Pequot Tribal Nation Gaming Commission respectively. Directing a guest seeking self-exclusion to the WRONG authority is a regulatory failure. The `self_exclusion_authority` field correctly names the tribal commission, but the `self_exclusion_url` contradicts it. A guest following the URL would be on the wrong page. This needs to either point to the tribal commission's own page or say "Contact property directly" (matching the phone field pattern).

#### MAJOR

- **[MAJOR] config.py:241, config.py:326 -- Tribal commission_url points to state website.**
  Both CT tribal casinos set `enforcement_context.commission_url: "https://ct.gov/gaming"`. The ct.gov/gaming page is for the Connecticut Department of Consumer Protection's gaming division, which regulates commercial (non-tribal) gaming. The Mohegan Tribal Gaming Authority and the Mashantucket Pequot Tribal Gaming Commission are sovereign entities with their own regulatory frameworks. Using the state URL in the `commission_url` field misrepresents the jurisdictional relationship. While tribal gaming commissions may not have publicly accessible websites, the field should either reference the actual tribal authority or be marked as "N/A (tribal jurisdiction -- contact property)".

- **[MAJOR] config.py:110-161 -- DEFAULT_CONFIG falls back to CT jurisdiction.**
  DEFAULT_CONFIG has `"state": "CT"`, `"responsible_gaming_helpline": "1-800-MY-RESET"`, and `"state_helpline": "1-800-GAMBLER"`. Any unknown casino_id receives CT-specific regulatory information. While the code logs a warning (config.py:691-696), serving wrong-jurisdiction helplines is a regulatory risk. For example, a PA casino accidentally configured with an unknown casino_id would receive CT helplines. The DEFAULT_CONFIG should use jurisdiction-neutral national helplines only (1-800-GAMBLER is national, but 1-800-MY-RESET is marketed as CT-associated by the CT Council on Problem Gambling context), or better, the default should hard-fail for unknown casino_ids in production (`ENVIRONMENT != "development"`).

- **[MAJOR] docs/runbook.md:628-629 -- "Problem gambling: Auto-provides 1-800-MY-RESET (NCPG)" is wrong post-R68.**
  After R68, the primary NCPG number is 1-800-GAMBLER. Serving this stale information to operations staff creates a domain knowledge gap between the runbook and the actual system behavior. In a regulated environment, operators must know exactly what the system tells guests.

#### MINOR

- **[MINOR] config.py:500 -- Redundant Ohm sign confusable.**
  The confusable map includes `"\u2126": "O"` (Ohm sign mapped to "O"), but under NFKD normalization, U+2126 decomposes to U+03A9 (Greek Omega), which is already mapped at config.py line 435 as `"\u039f": "O"`. Wait -- actually U+03A9 is Omega which is NOT in the confusable table. U+039F (Omicron) IS mapped. The Ohm sign decomposes to Omega (U+03A9), and Omega is NOT mapped. So the U+2126 entry serves a purpose after all: it catches the Ohm sign by mapping it directly before NFKD normalization. This finding is WITHDRAWN -- the entry is NOT redundant because the normalization order in code is NFKD first then confusable replacement (guardrails.py comment at line 571-574), meaning U+2126 decomposes to U+03A9 (Omega), and Omega is NOT in the confusable table. So the U+2126 entry is indeed redundant -- NFKD converts it to Omega, and Omega is unmapped, so "O" substitution is lost. This is actually a BUG: the Ohm sign decomposes to Omega (U+03A9) via NFKD, but Omega is not in the confusable table, so the mapping of U+2126 to "O" is dead code (NFKD runs first per line 571 comment, decomposing Ohm to Omega before confusable replacement runs). Adding U+03A9 (Omega) to the confusable table would fix this. Upgrading to MINOR.

- **[MINOR] Limited casino coverage -- 5 casinos, all in 4 states (CT, PA, NV, NJ).**
  No tribal casinos outside CT. No multi-jurisdiction operators (MGM, Caesars). No midwest or southern states. While MVP scope is acknowledged in ADR-019, the onboarding checklist (runbook:679-739) should note that new states may require entirely new regulatory patterns not covered by the current 4-state configuration (e.g., Mississippi, Louisiana, Michigan all have different self-exclusion frameworks).

- **[MINOR] config.py:153 -- 1-800-MY-RESET described as NCPG in DEFAULT_CONFIG but is CT-associated.**
  The comment on line 154 says "R52 fix M3: National fallback -- 1-800-GAMBLER is the rebranded NCPG number." This correctly identifies 1-800-GAMBLER as the national NCPG number. But the `responsible_gaming_helpline` field is set to 1-800-MY-RESET, which while also an NCPG line, is marketed by the NCPG as an alternate/secondary line and is prominently used by the CT Council on Problem Gambling. Using it as the "default" for all unknown casinos creates a CT-centric bias in what should be a jurisdiction-neutral fallback.

### D10 Summary

Strengths: 5 casino profiles with per-state regulatory data. Enforcement context with recent precedent cases. Varied escalation thresholds by strict liability status (PA/NJ=2, CT/NV=3). NGC Reg. 5.170 correctly referenced for NV self-exclusion. Tribal vs commercial property_type distinction. Per-casino persona names and branding. Import-time validation of required regulatory fields.

Weaknesses: The tribal jurisdiction handling is fundamentally flawed -- the code correctly SAYS tribal commissions handle self-exclusion (via comments and `self_exclusion_authority` field) but then POINTS GUESTS to the state page via `self_exclusion_url` and `commission_url`. This is a direct contradiction between documented intent and actual configuration. In a regulated casino environment, directing a guest to the wrong self-exclusion authority could expose the operator to liability. The DEFAULT_CONFIG CT fallback for unknown casinos creates silent regulatory risk.

---

## R68 Fix Verification

- **[VERIFIED] Helpline number doc-code parity** -- PARTIAL. Code (prompts.py) correctly updated to make 1-800-GAMBLER primary NCPG helpline (R68 fix). BUT runbook.md:629 still says "Auto-provides 1-800-MY-RESET (NCPG)" -- doc-code parity NOT fully achieved.
- **[NOT VERIFIED] CT tribal self-exclusion phone fix** -- config.py:232 and :318 correctly say "Contact property directly" for tribal self_exclusion_phone. However, self_exclusion_url STILL points to ct.gov/selfexclusion (state page), contradicting the tribal jurisdiction fix.
- **[VERIFIED] ADR-022 regulatory risk rationale** -- ADR-022 exists with guardrail-to-regulation mapping, enforcement precedent table, and liability exposure analysis. Comprehensive and well-structured.
- **[VERIFIED] Runbook onboarding checklist** -- 10-step onboarding checklist present (runbook lines 679-739) covering regulatory info, casino profile creation, feature flags, RAG data, env vars, validation, testing, guardrail verification, secrets, and deployment.
- **[VERIFIED] enforcement_context + escalation_threshold per jurisdiction** -- Present for all 5 casino profiles with strict_liability flag, recent enforcement case, last_regulatory_review date, and commission_url. Thresholds correctly varied (PA/NJ=2 vs CT/NV=3).

---

## Summary

| Severity | D9 Count | D10 Count | Total |
|----------|----------|-----------|-------|
| CRITICAL | 0 | 1 | 1 |
| MAJOR | 4 | 3 | 7 |
| MINOR | 2 | 3 | 5 |

**D9 Score: 7.5/10** (down from 9.0 baseline -- 4 MAJORs for doc-code parity drift)
**D10 Score: 6.5/10** (down from 9.0 baseline -- 1 CRITICAL tribal URL contradiction, 3 MAJORs)

**Weighted impact on overall score**: D9 contributes 0.05 * 7.5 = 0.375, D10 contributes 0.10 * 6.5 = 0.65. Combined = 1.025 (vs baseline 0.05 * 9.0 + 0.10 * 9.0 = 1.35). Net impact: -0.325 on overall weighted score.

### Top 3 Action Items

1. **Fix tribal self_exclusion_url** (CRITICAL): Replace `ct.gov/selfexclusion` with `"Contact property directly"` or actual tribal commission URL for Mohegan Sun and Foxwoods. The current configuration contradicts the code's own comments about tribal sovereignty. Similarly fix commission_url for both tribal properties.

2. **Update runbook helpline section** (MAJOR): Change "Problem gambling: Auto-provides 1-800-MY-RESET (NCPG)" to reflect R68 fix making 1-800-GAMBLER the primary NCPG helpline. Add `responsible_gaming_helpline` column to the Per-State Regulatory Quick Reference table.

3. **Fix pattern count and version drift** (MAJOR): Update runbook pattern count from 204 to 206 (or add CI assertion `assert len(_ALL_PATTERNS) == N`). Bump config.py VERSION from 1.1.0 to 1.3.0.
