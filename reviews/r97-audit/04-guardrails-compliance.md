# Component 4: Guardrails + Compliance Gate

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/agent/guardrails.py` | 1,297 | 6 guardrail categories: prompt injection (Latin + non-Latin), responsible gaming, age verification, BSA/AML, patron privacy, self-harm. 214 compiled regex patterns across 11 languages. Multi-layer input normalization (`_normalize_input`). Semantic injection classifier (`classify_injection_semantic`). Confusable mapping (145 entries). |
| `src/agent/compliance_gate.py` | 652 | Compliance gate graph node: priority chain of 10 checks (turn-limit, empty, injection, RG, age, BSA, crisis persistence, patron privacy, grief/celebration detection, crisis levels, self-harm, confirmation, semantic injection). Structured JSON audit logging for every trigger. |
| `src/agent/regex_engine.py` | 110 | RE2/stdlib regex adapter. Prefers `google-re2` for linear-time guarantee (ReDoS-safe). Falls back to stdlib `re` for unsupported features. `enforce_re2_in_production()` fails fast in non-dev environments. |

**Total: 2,059 lines across 3 files.**

## Wiring Verification

**Fully wired.**

- `guardrails.py`: Imported by `compliance_gate.py:34-43` (all 7 `detect_*` functions + `audit_input` + `classify_injection_semantic`)
- `compliance_gate.py`: `compliance_gate_node` imported by `graph.py:55`, registered as `NODE_COMPLIANCE_GATE` in the StateGraph
- `regex_engine.py`: Imported by `guardrails.py:22` (`regex_engine.compile`) for all 214 pattern compilations. `enforce_re2_in_production()` called at app startup (`app.py:84`)

**Entry point chain**: `app.py` startup -> `enforce_re2_in_production()`. Every request: `START` -> `compliance_gate_node` -> all guardrail checks -> router (or short-circuit to terminal node).

## Architectural Strengths

1. **Defense-in-depth**: Two layers — regex (zero-cost, deterministic) then LLM semantic classifier (configurable, fail-closed). Semantic runs last so classifier outage doesn't block legitimate safety queries.
2. **Priority chain with explicit rationale** (`compliance_gate.py:55-93`): Injection runs at position 3 (before all content guardrails) because injection failure compromises all downstream checks. Documented reasoning for every position.
3. **Multi-layer normalization** (`guardrails.py`): URL decode (iterative 3-pass), HTML unescape, NFKD, invisible char strip (Cf category), confusable replacement (145 entries), punctuation-to-space, single-char merging. Checks both raw AND normalized forms.
4. **ReDoS protection** (`regex_engine.py`): RE2 preferred for linear-time regex execution. `enforce_re2_in_production()` fails fast if RE2 unavailable in non-dev environments.
5. **Grief/celebration pass-through** (`compliance_gate.py:348-457`): Emotional contexts are NOT blocked — they set `guest_sentiment` and pass through to the router. This is the correct design: grief needs empathetic response, not a wall.
6. **Crisis persistence with de-escalation** (`compliance_gate.py:219-330`): Once `crisis_active` is set, follow-up messages stay in crisis mode UNLESS the guest both confirms they're OK AND asks a property question, OR agrees to seek help. Prevents premature transition.
7. **Confirmation detection** (`compliance_gate.py:550-615`): Short acknowledgments ("great", "ok") route to greeting instead of RAG, reducing fallback rate. Only triggers for <8-word messages to avoid false matches.
8. **Structured JSON audit logging**: Every guardrail trigger produces a structured `audit_event` JSON log with category, query_type, timestamp, action, and severity. Enables systematic auditing.
9. **Input size DoS protection** (`guardrails.py:1002`): Blocks inputs >8192 chars before normalization to prevent CPU exhaustion from 5 O(n) Unicode passes.
10. **Act-as whitelist** (`guardrails.py:1025`): "Act as a guide" is whitelisted in casino context (where "guide" is a legitimate domain term), while "act as a hacker" triggers injection detection.

## Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `test_guardrails.py` | 91 | All 6 categories: injection, RG, age, BSA, patron privacy, self-harm. Language coverage (EN, ES, PT, ZH, etc.). Normalization bypass resistance. Confusable attacks. Semantic classifier mocking. |
| `test_compliance_gate.py` | 40 | Full compliance gate priority chain: turn limit, empty, injection, RG, BSA, crisis, patron privacy, grief, celebration, confirmation. |
| `test_crisis_detection.py` | 72 | Graduated crisis levels (concern/urgent/immediate). Pattern coverage. |
| `test_crisis_progression.py` | 37 | Multi-turn crisis progression and de-escalation. |
| `test_guardrail_fuzz.py` | 45 | Hypothesis fuzzing for normalization and pattern coverage. |
| `test_guardrail_redos.py` | 7 | ReDoS timing tests (patterns complete under time limit). |
| `test_guardrail_patterns.py` | 10 | Pattern count assertions (214 total). |
| `test_property_based.py` | 9 | Hypothesis property-based tests for guardrail invariants. |
| `test_e2e_security_enabled.py` | 15 | E2E with semantic injection classifier enabled. |
| `test_regulatory_invariants.py` | 16 | Regulatory compliance invariants. |
| `test_regulatory_response_content.py` | 9 | Response content for regulated categories. |
| `test_multilingual.py` | 21 | Language detection and multilingual guardrail coverage. |

**Total: ~372 tests covering guardrails + compliance.**

## Live vs Mock Assessment

**Mostly deterministic (no mocks needed) with targeted mocks for semantic classifier.**

- **Deterministic tests (majority)**: `test_guardrails.py` (91 tests) directly call `detect_prompt_injection()`, `detect_responsible_gaming()`, etc. — pure Python functions, no LLM calls. These are genuinely live tests of the regex engine.
- **Compliance gate tests**: `test_compliance_gate.py` (40 tests) mocks `get_settings` to control `SEMANTIC_INJECTION_ENABLED=False` and `MAX_MESSAGE_LIMIT`. When semantic injection is disabled, all tests are deterministic.
- **Semantic classifier**: Mocked in `test_guardrails.py:394-406` when testing `classify_injection_semantic()`. The LLM call is mocked. `test_e2e_security_enabled.py` tests with semantic enabled but still mocks the LLM.
- **Fuzz tests**: `test_guardrail_fuzz.py` uses Hypothesis to generate random inputs — no mocks, just deterministic guardrail functions.

**This is appropriate**: Guardrails are fundamentally regex-based. The only LLM-dependent part is the semantic injection classifier, which is correctly tested via mocks for unit tests and would need a live integration test.

## Known Gaps

1. **`_audit_input` inverted semantics** (`guardrails.py:978`): Returns True=safe, False=injection. Documented but confusing. The wrapper `detect_prompt_injection()` fixes this for external callers, but internal code referencing `_audit_input` directly is error-prone.
2. **`_normalize_for_matching` uses stdlib `re`** (`guardrails.py:916-919`): The `_merge_single_chars` regex replacement at line 919 uses `re.sub` (stdlib) inside a function called on every guardrail check. Not using `regex_engine.compile`. Potential ReDoS vector on crafted input.
3. **No live test for semantic injection classifier**: `classify_injection_semantic()` LLM call is only tested with mocks. No `@pytest.mark.live` test verifies the Gemini API accepts the `InjectionClassification` schema.
4. **Patron privacy `is...here` false positive documented but partially fixed** (`compliance_gate.py:332`): R95 added `(?!there\b)` negative lookahead for "Is there anything else to do here?" but the comment says this is a partial fix — other similar constructions may still false-match.
5. **No rate limiting on audit log emission**: Every guardrail trigger emits a structured JSON log. Under adversarial load (attacker sending 1000 injection attempts/second), this generates massive log volume. No sampling or rate-limiting on audit logs.
6. **Grief keyword matching is broad**: `"her favorite"` and `"his favorite"` (`compliance_gate.py:387-388`) match for grief detection. These could false-match on non-grief messages like "This is her favorite restaurant."
7. **`_CONFIRMATION_PATTERNS` not re2-compiled**: Uses plain string matching (`in` operator), not regex. This is fine for exact matches but the `startswith` checks at line 590-601 could drift from the exact patterns.

## Confidence: 92%

The guardrail system is the strongest component in the codebase. 214 regex patterns across 11 languages, multi-layer normalization, RE2 enforcement in production, structured audit logging, defense-in-depth with semantic classifier, and 372+ tests including fuzz and ReDoS coverage. The priority chain in compliance_gate is well-reasoned and documented. The main gaps are minor: inverted semantics internal function, no live test for semantic classifier, and some broad grief keyword matching.

## Verdict: production-ready

This is production-grade security infrastructure. The guardrail system has been hardened across 90+ review rounds with specific fixes for every bypass discovered (URL encoding, Unicode confusables, double-encoding, form-encoded+, zero-width characters, character-level smuggling, non-Latin language injection). The RE2 enforcement and structured audit logging are best practices for regulated environments.
