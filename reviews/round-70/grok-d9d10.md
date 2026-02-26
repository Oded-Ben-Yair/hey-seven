# R70 Review: Grok 4 (D9, D10)

**Model**: Grok 4 (flagship)
**Date**: 2026-02-26
**Reviewer**: mcp__grok__grok_chat

---

## D9: Trade-off Documentation (weight 0.05)
**Score: 8.7**

### Findings:
- [MAJOR] **Doc-code parity drift on pattern count**: Runbook claims "205 regex patterns" but actual codebase count is 204 (`regex_engine.compile()` count verified). Test `test_doc_accuracy.py` asserts `== 204`. The runbook (line 577) and ADR-018 both reference 205, creating a 3-way inconsistency. Also, `test_guardrail_fuzz.py` docstring says "205 regex patterns". **Auditor-verified: actual = 204.**
- [MAJOR] **Missing cross-references for known limitations**: Limitations like `_last_backend_sync` race and `_latency_samples` process-scoped are documented in-code, but there's no linkage to ADRs or ARCHITECTURE.md for broader context. Fragments knowledge for new contributors.
- [MINOR] **Stale alternate helpline framing in prompts.py**: `RESPONSIBLE_GAMING_HELPLINES_DEFAULT` lists "NCPG Alternate Line: 1-800-MY-RESET" — while 1-800-GAMBLER is correctly primary, the "alternate" framing could confuse guests. NCPG's current branding positions both as co-equal contact points, not primary/alternate.

### R69 Fix Verification:
- **Pattern count 205 in runbook**: REGRESSION. Actual count is 204. Runbook, ADR-018, and fuzz test docstring all say 205. Test asserts 204. 3-way inconsistency.
- **Helpline: 1-800-GAMBLER as primary**: CONFIRMED. prompts.py correctly lists it first.
- **VERSION 1.3.0 parity**: CONFIRMED. config.py=1.3.0, middleware x-api-version=1.3.0, .env.example=1.3.0. Full parity.

**Weighted: 0.44**

---

## D10: Domain Intelligence (weight 0.10)
**Score: 9.2**

### Findings:
- [MINOR] **Incomplete jurisdictional nuances for tribal casinos**: mohegan_sun and foxwoods correctly use "Contact property directly" for self_exclusion_url, but no documentation or validation covers edge cases like multi-jurisdiction overlaps (e.g., CT tribal properties with federal IGRA oversight). Feature flags lack tribal-specific toggles.
- [MINOR] **CT helpline in default fallback for all tribal casinos**: Default helplines include "CT Council on Problem Gambling: 1-888-789-7777" which may not align with non-CT tribal properties if they are ever onboarded. Per-casino helpline lookup exists but default fallback is CT-specific.
- [MINOR] **Missing feature flag completeness for regulatory compliance**: 12 feature flags per casino are not audited against jurisdictional requirements (e.g., no flag for NJ DGE self-exclusion reminder frequency). Small gap but could compound.

### R69 Fix Verification:
- **Tribal self_exclusion_url = "Contact property directly"**: CONFIRMED. Both mohegan_sun and foxwoods profiles use this correctly.
- **commission_url = "Contact property directly — tribal jurisdiction"**: CONFIRMED. Tribal casino profiles use this verbatim.

**Weighted: 0.92**

---

## Summary Table

| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| D9: Trade-off Documentation | 8.7 | 0.05 | 0.44 |
| D10: Domain Intelligence | 9.2 | 0.10 | 0.92 |
| **Subtotal** | **8.88** | **0.15** | **1.36** |

### Critical Finding — Auditor-Verified:
**Pattern count drift is a confirmed regression**: The actual `regex_engine.compile()` count in `guardrails.py` is **204** (verified by running `inspect.getsource()` + regex count). The runbook (line 577), ADR-018 (line 21), and `test_guardrail_fuzz.py` (line 3 docstring) all claim **205**. The test `test_doc_accuracy.py:285` correctly asserts `== 204`. This is a 3-way doc inconsistency that should be resolved: either add the missing pattern or update all docs to 204.
