# R68 Grok Review -- D9/D10

**Reviewer**: Grok (hostile domain expert)
**Date**: 2026-02-26
**Baseline**: R67 scores (D9=9.0, D10=9.2)

## Scores

| Dim | Name | Score | Delta from R67 |
|-----|------|-------|----------------|
| D9 | Trade-off Docs | 9.0 | +0.0 |
| D10 | Domain Intelligence | 9.0 | -0.2 |

**Weighted contribution**: D9 (0.05 * 9.0 = 0.45) + D10 (0.10 * 9.0 = 0.90) = 1.35

---

## D9 Trade-off Documentation (9.0/10)

### Summary

23 ADR files exist (including ADR-0001 through ADR-022 plus the README index). All ADRs have status fields, most have review dates of 2026-02-25 or 2026-02-26. The status lifecycle is well-defined (Proposed, Accepted, Deferred, Superseded, Deprecated). The runbook is comprehensive at 739 lines covering Cloud Run config, deployment, incident response, observability, security, regulatory mapping, and onboarding. ADR quality is generally high with alternatives considered, consequences documented, and cross-references to review rounds.

### CRITICALs

None.

### MAJORs

**D9-M001: ADR-022 missing from README index**
- `docs/adr/022-regulatory-risk-rationale.md` exists on disk and is referenced in the review instructions, but the `docs/adr/README.md` index stops at entry 021. The index is the primary discovery mechanism for ADRs. A missing entry means regulatory counsel reviewing the ADR catalog would not find the guardrail-to-regulation mapping -- the single most important ADR for legal review.
- File: `docs/adr/README.md` line 41 (end of index)
- Fix: Add ADR-022 entry to the README index table.

**D9-M002: Runbook helpline numbers contradict source code**
- `docs/runbook.md` line 628 states: `NV: 1-800-MY-RESET` but `src/casino/config.py` line 467 says NV `state_helpline` is `1-800-GAMBLER` (the correct post-2022 NCPG rebrand). The runbook is stale.
- `docs/runbook.md` line 628 also says `PA: 1-800-GAMBLER` but `src/casino/config.py` line 385 says PA `state_helpline` is `1-800-848-1880` (PA Council on Compulsive Gambling line).
- An operator following the runbook during an incident would provide incorrect helpline information for NV and PA guests. In a regulated environment, wrong helpline numbers are a compliance risk.
- File: `docs/runbook.md` line 628
- Fix: Update line 628 to match code: `CT: 1-888-789-7777, NJ: 1-800-GAMBLER, PA: 1-800-848-1880, NV: 1-800-GAMBLER`

**D9-M003: Regex pattern count claims are inconsistent across documents**
- Actual count (verified by import): **171 patterns** across 6 guardrail categories.
- `src/agent/compliance_gate.py` docstring line 8: claims "~185 compiled regex patterns"
- `docs/runbook.md` line 577: claims "204 regex patterns across 12 languages"
- `CLAUDE.md` (project root): claims "204 regex patterns"
- Three different documents claim three different counts, none matching reality. Pattern count is a security posture metric used for audit and compliance reporting. Overstating it is worse than understating it -- an auditor who verifies the count loses trust in all other claims.
- Fix: Grep all documents for pattern count claims and update to the actual count (171). Consider adding a test that asserts the count to prevent drift.

### MINORs

**D9-m001: ADR-011 through ADR-015 are thin**
- ADRs 011 (RRF k=60), 012 (retrieval timeout), 013 (SSE timeout), 014 (message limits), and 015 (circuit breaker parameters) are configuration parameter ADRs with minimal context. They document the "what" but not the "why not other values" -- for example, ADR-012 says "10-second timeout" but doesn't explain why not 5s or 15s. ADR-015 is the strongest of these (includes a rationale table). The others could benefit from rejected alternatives.
- Impact: Low. These are supplementary configuration records, not architectural decisions.

**D9-m002: ADR-018 confusable count in title (135) differs from code-level sum**
- ADR-018 title says "135 entries" which does match the runtime count. However, the ADR body lists 7 scripts with individual counts (23+18+52+16+10+7+9 = 135). This is internally consistent. No action needed -- this is a verification pass.
- Status: No finding (verified correct).

**D9-m003: Some ADRs lack explicit "Last Reviewed" field**
- ADRs 001-007 and ADR-0001 have review dates in the README index but not in the ADR files themselves. ADRs 009, 011-015 also lack inline review dates. While the README index provides centralized tracking, best practice per the project's own rubric is to have the date in both locations.
- Impact: Low. The centralized index compensates.

**D9-m004: Runbook references `docs/output-guardrails.md` (line 531) -- file existence not verified**
- Line 531 says "See `docs/output-guardrails.md` for full architecture" but this file was not in the review scope. If the file does not exist, the reference is broken documentation.
- Impact: Low (documentation cross-reference).

---

## D10 Domain Intelligence (9.0/10)

### Summary

Five casino profiles (Mohegan Sun, Foxwoods, Parx Casino, Wynn Las Vegas, Hard Rock AC) covering 4 states (CT, NJ, NV, PA). Each profile has comprehensive regulatory fields including state, gaming age, helplines, self-exclusion authority/URL/phone/options, enforcement context, escalation thresholds, and operational data (timezone, property type, size). The `get_casino_profile()` function returns `copy.deepcopy()` preventing mutable global state corruption. Import-time validation via `_validate_profiles_completeness()` catches missing required fields. The jurisdictional reference document provides accurate regulatory comparisons. The onboarding checklist in the runbook is thorough (10 steps). Per-jurisdiction responses in `off_topic_node` cover self-harm (988 Lifeline), responsible gaming, BSA/AML, patron privacy, and age verification.

### CRITICALs

None.

### MAJORs

**D10-M001: CT tribal casino self-exclusion phone number is CT DCP, not tribal commission**
- Both `mohegan_sun` and `foxwoods` profiles set `self_exclusion_phone: "1-860-713-6300"`. This is the CT Department of Consumer Protection (DCP) number. However, the code comments (line 221-223, 304-306) explicitly state: "Tribal casinos self-exclude through their own gaming commissions, NOT CT DCP."
- The `self_exclusion_authority` field correctly says "Mohegan Tribal Gaming Commission" and "Mashantucket Pequot Tribal Nation Gaming Commission" respectively, but the phone number still points to CT DCP.
- The jurisdictional reference (`docs/jurisdictional-reference.md` line 24) also uses `1-860-713-6300` as the CT phone.
- This is a real regulatory inaccuracy: a guest calling the CT DCP number to self-exclude from a tribal casino would be directed to the wrong authority. Tribal casinos operate under sovereign jurisdiction and their self-exclusion programs are administered by their own gaming commissions, not by CT DCP. The guest would need to call the tribal gaming commission directly.
- Note: This has been present since R39 when the authority names were corrected but the phone numbers were not updated. The phone numbers should be the direct lines for each tribal gaming commission.
- Fix: Update `self_exclusion_phone` for each CT tribal casino to the correct tribal gaming commission phone number. If the tribal commissions do not publish separate phone numbers, document this in the profile and link to the tribal commission website instead.

**D10-M002: DEFAULT_CONFIG missing `self_exclusion_authority` field**
- Runtime validation (confirmed by import output) warns: `DEFAULT_CONFIG missing required regulatory fields: frozenset({'self_exclusion_authority'})`.
- While `DEFAULT_CONFIG` is only used as a fallback for unknown casino IDs, a guest hitting the default path would receive no self-exclusion authority information. The `_REQUIRED_PROFILE_FIELDS` frozenset explicitly includes `self_exclusion_authority`, but `DEFAULT_CONFIG.regulations` at line 146-157 does not contain it.
- The validation only logs a warning (`logger.warning`) but does not raise, so the application starts successfully with an incomplete fallback config.
- Fix: Add `"self_exclusion_authority": "Contact your state gaming commission"` (or similar generic guidance) to `DEFAULT_CONFIG.regulations`.

**D10-M003: `get_responsible_gaming_helplines()` labels 1-800-MY-RESET as "National Problem Gambling Helpline"**
- `src/agent/prompts.py` line 60: The dynamic helpline function outputs `"- National Problem Gambling Helpline: 1-800-MY-RESET (1-800-699-7378)"`. However, 1-800-MY-RESET is the NCPG number, not the "National Problem Gambling Helpline" brand. The `config.py` comments (line 154, 465) correctly note that the national number rebranded to 1-800-GAMBLER in 2022.
- Meanwhile, `RESPONSIBLE_GAMING_HELPLINES_DEFAULT` (line 27) lists both numbers with distinct labels: `1-800-GAMBLER` as "National Problem Gambling Helpline" and `1-800-MY-RESET` as "National Council on Problem Gambling". The dynamic function (line 60) uses a different label for the same number.
- This creates a confusing inconsistency: the static default helplines list both numbers with correct labels, but the dynamic per-state function outputs only `1-800-MY-RESET` with the wrong label and omits `1-800-GAMBLER` entirely.
- Fix: Update the dynamic function to list `1-800-GAMBLER` as "National Problem Gambling Helpline" (the rebranded primary number) and optionally keep `1-800-MY-RESET` as an alternate NCPG line, matching the static default format.

### MINORs

**D10-m001: `self_exclusion_url` for CT tribal casinos points to `ct.gov/selfexclusion`**
- The CT state government self-exclusion page (`ct.gov/selfexclusion`) is the CT DCP program page. Tribal casinos have their own self-exclusion processes. While the CT DCP page may provide general information, it does not administer tribal casino self-exclusion.
- Both Mohegan Sun (line 224) and Foxwoods (line 307) use this URL. This is consistent with the jurisdictional reference doc (line 23) but may be misleading for tribal casino guests seeking self-exclusion through their tribal gaming commission.
- Impact: Medium-low. The URL provides some useful information even if the authority is different.

**D10-m002: Foxwoods `enforcement_context.commission_url` is `https://ct.gov/gaming`**
- Foxwoods is a Mashantucket Pequot tribal casino. The `commission_url` should point to the Mashantucket Pequot Tribal Nation's gaming commission, not the CT state gaming page. The Mohegan Sun profile has the same issue.
- Both profiles (lines 234, 316) use `https://ct.gov/gaming` which is a state-level page, not a tribal commission page.
- Impact: Low for MVP since this field is used for documentation, not guest-facing responses.

**D10-m003: All 5 profiles have identical `_updated_at: ""`**
- The `_updated_at` field is empty for all profiles. While these are static profiles (not Firestore-backed), the field exists in `CasinoConfig` TypedDict. Having an empty string for all profiles suggests the field is not maintained. Either populate it with the actual last-modified date or remove it from static profiles.
- Impact: Low. Cosmetic/metadata.

**D10-m004: `RESPONSIBLE_GAMING_HELPLINES_DEFAULT` references "CT Self-Exclusion" with DCP contact**
- Line 30 in `prompts.py`: `"CT Self-Exclusion: Contact your casino's tribal gaming commission or visit ct.gov/selfexclusion"`. This correctly directs guests to their tribal gaming commission first, with the DCP URL as a secondary resource. However, the phrasing "your casino's tribal gaming commission" assumes the guest is at a tribal casino -- the default helplines should be jurisdiction-neutral since they serve as the fallback for unknown casinos.
- Impact: Low. The phrasing is reasonable for the CT-defaulting configuration.

**D10-m005: Escalation threshold varies without documented rationale**
- Mohegan Sun (CT): threshold 3, Foxwoods (CT): threshold 3, Wynn (NV): threshold 3, but Parx (PA): threshold 2, Hard Rock (NJ): threshold 2.
- PA and NJ have lower thresholds than CT and NV. No documented rationale exists for why PA/NJ should escalate after 2 mentions vs 3 for CT/NV. If this is intentional (stricter state requirements), it should be documented. If it's arbitrary, it should be standardized.
- Impact: Low. The thresholds are reasonable either way.

---

## Cross-Dimension Observations

1. **Documentation-code parity is the weakest link**: Three findings (D9-M002, D9-M003, D10-M003) stem from documentation not being updated when code changed. This is a systematic issue -- code changes propagate through tests automatically, but documentation updates require manual effort. Consider adding CI checks that verify pattern counts and helpline numbers against source code.

2. **Tribal sovereignty nuance is partially handled**: The code correctly identifies tribal vs commercial property types and names the correct tribal gaming commissions as self-exclusion authorities. However, the phone numbers, URLs, and commission URLs still point to CT state government resources. This is the remaining gap in tribal casino regulatory accuracy.

3. **ADR quality is bimodal**: The core architectural ADRs (001-010, ADR-0001) are excellent with detailed context, alternatives, and consequences. The parameter ADRs (011-015) are thin by comparison. ADR-022 (regulatory risk) is strong but not indexed.
