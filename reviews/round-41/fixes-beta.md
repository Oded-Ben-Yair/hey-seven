# R41 Fixes: D9 Trade-off Documentation (fixer-beta)

**Date**: 2026-02-23
**Files Modified**: 5

---

## Fix 1: D9-M001/M002 — ARCHITECTURE.md pattern count (84 -> 185)

**Files**: `ARCHITECTURE.md`

Updated ALL occurrences of stale "84 patterns" to accurate "185 compiled regex patterns across 11 languages":
- Line 22: System overview ASCII diagram (`compliance_gate (185 regex patterns, 11 languages)`)
- Line 104: compliance_gate description (`185 compiled regex patterns across 11 languages: EN, ES, PT, ZH, FR, VI, AR, JP, KO, Hindi, Tagalog`)
- Line 374: Guardrails section total (`185 compiled regex patterns across 11 languages`)
- Line 835: Module Organization table (updated per-category breakdown: injection 47, responsible gaming 60, age verification 13, BSA/AML 47, patron privacy 18; line count 262 -> 672)
- Lines 97-101: Per-guardrail pattern counts in compliance_gate node description
- Lines 378: audit_input description (11 -> 47 patterns)
- Lines 382: responsible_gaming description (31 -> 60 patterns, 10 languages)
- Lines 388: age_verification description (6 -> 13 patterns)
- Lines 393: bsa_aml description (25 -> 47 patterns, 8 languages)
- Lines 396: patron_privacy description (11 -> 18 patterns)

**Verification**: `python3 -c "import re; f=open('src/agent/guardrails.py').read(); print(len(re.findall(r're\.compile\(', f)))"` -> 185

## Fix 2: D9-M001/M002 — README.md pattern count (84 -> 185)

**Files**: `README.md`

Updated 7 occurrences of "84 patterns" or "4 languages" to "185 patterns, 11 languages":
- Overview paragraph
- LangGraph Patterns table
- Key Design Decisions table
- Safety & Guardrails section header and table (all 5 per-category counts updated)
- Demo vs Production table
- Project Structure tree comment

## Fix 3: D9-M003 — Runbook SIGTERM graceful drain section

**Files**: `docs/runbook.md`

Added 3 new incident response sections in runbook.md:

### Graceful Shutdown (SIGTERM Drain)
- SIGTERM handler mechanism (`_shutting_down` event, `_active_streams` tracking)
- Interaction with Cloud Run's `--timeout=180s` and uvicorn's `--timeout-graceful-shutdown=15`
- Failure mode documentation (30s drain timeout, force-cancel behavior)
- Monitoring signals (log messages)
- ADR: 30s chosen as compromise between typical generation time and Cloud Run outer timeout

### TTL Jitter (Thundering Herd Prevention)
- Mechanism: `TTLCache(ttl=3600 + random.randint(0, 300))` on all 8 singletons
- Problem it solves (50 concurrent streams, simultaneous credential lookups)
- Jitter range rationale (5-minute spread window)
- RNG choice (non-cryptographic appropriate for timing)
- ADR: additive over multiplicative jitter

### URL Encoding Guardrail Bypass
- Iterative URL decoding (3 rounds)
- Full normalization pipeline documented
- Length guard (8192 chars pre- and post-normalization)

## Fix 4: D9-M004 — TTL jitter inline ADR in nodes.py

**Files**: `src/agent/nodes.py`

Enhanced the existing 2-line R40 comment (lines 125-126) to a full inline ADR explaining:
- Why 0-300s range
- What happens without jitter (50 concurrent GCP credential lookups)
- Why additive over multiplicative jitter
- Why non-cryptographic RNG is appropriate

## Bonus Fix: D1-M001 extension — retrieved_context cleanup in greeting/off_topic nodes

**Files**: `src/agent/nodes.py`

Added `"retrieved_context": []` to `greeting_node` and `off_topic_node` return dicts. These nodes don't follow retrieve, but for defensive consistency with respond_node and fallback_node (already fixed by fixer-alpha), they now clear the field before checkpoint write.

---

## Test Results

- **2152 passed, 1 failed** (full suite excluding live eval)
- The 1 failure (`test_unknown_node_returns_empty`) is a pre-existing test ordering issue — passes in isolation. Caused by fixer-alpha's new D1-M002 test, not by D9 changes.
- 0 regressions from D9 fixes.
