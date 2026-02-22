# R20 Agent Quality Review -- FINAL (Dimensions 5-10)

Reviewer: r20-reviewer-beta
Date: 2026-02-22
Commit: 52aa84c (Phase 3 Agent Quality Revolution)

---

## Score History: R18=42/60, R19=44.5/60

---

## Dimension 5: Conversation Quality (8.5/10)

**R19 Fix Verification:**
- R19 CRITICAL: `extracted_fields` reset to `{}` every turn, breaking multi-turn accumulation. **FIXED.** `state.py:80` now declares `extracted_fields: Annotated[dict[str, Any], _merge_dicts]` with a proper reducer. The `_merge_dicts` reducer (state.py:16-25) performs `{**a, **b}`, so when `_initial_state()` passes `{}`, the existing fields survive (`{**existing, **{}} == existing`). When extraction produces new fields, they merge in. This is the correct LangGraph pattern -- same as `add_messages` for messages and `_keep_max` for counters. **Multi-turn field accumulation now works as documented.**
- R19 HIGH: Whisper planner guidance unverified by validation. **Unchanged.** The 6 validation criteria (prompts.py:137-176) check factual accuracy and safety, not conversational progression. This is an accepted design trade-off -- whisper guidance is advisory, not enforced. MEDIUM: advisable to add a 7th validation criterion ("Does the response follow the whisper planner's suggested conversational direction?") but not required for production correctness.
- R19 MEDIUM: `_format_history()` drops SystemMessages. **Unchanged.** The planner's 20-message window still silently omits retry feedback injected as SystemMessage. Impact is minor -- the planner may repeat failed topics, but the validation loop catches the same problems.

**Strengths:**
- The `_merge_dicts` reducer fix is architecturally correct and follows LangGraph patterns. The docstring (state.py:17-25) explicitly explains the empty-dict merge semantics, which is production-grade documentation.
- The parity check at graph.py:527-534 catches any state schema drift at import time. Adding `extracted_fields` with a reducer required no parity update because the field already existed -- only its type annotation changed. This validates the parity safety net.
- Whisper planner (whisper_planner.py:145-218) is properly fail-silent with telemetry tracking (`_WhisperTelemetry` class with instance-level attributes, fixing the R19 LOW about class-level sharing).
- Message windowing at `_base.py:186` and whisper planner's 20-message window (whisper_planner.py:166) correctly prevent unbounded context growth.
- Profile completeness feedback (whisper_planner.py:262-279) drives progressively smarter conversation guidance as extracted fields accumulate.

**Findings:**

[MEDIUM] [CONV] The `_merge_dicts` reducer performs a shallow merge (`{**a, **b}`). If `extracted_fields` ever contains nested dicts (e.g., `{"preferences": {"dietary": "vegan", "cuisine": "italian"}}`), the inner dict would be completely overwritten by any new extraction, not merged. Currently this is safe because all extraction values are flat (strings, ints), but the reducer contract doesn't document this assumption. If `extract_fields()` ever returns nested structures, silent data loss will occur. (src/agent/state.py:16-25)

[LOW] [CONV] Whisper planner `_calculate_completeness()` (whisper_planner.py:262-279) uses 8 profile fields (`_PROFILE_FIELDS`) but `extract_fields()` only extracts 5 fields (name, party_size, visit_date, preferences, occasion). The 3 unextractable fields (dining, entertainment, companions) can only be filled by the guest profile system via Firestore, meaning completeness maxes out at 62.5% (5/8) from extraction alone. Since `guest_profile_enabled=True` now, these can be filled from profile data, but for new guests without history, the completeness ceiling from regex extraction alone is limited. (src/agent/whisper_planner.py:115-118, src/agent/extraction.py)

[LOW] [CONV] `_initial_state()` (graph.py:517) still sets `guest_name: None`. While `extracted_fields` now persists via reducer, `guest_name` does NOT have a reducer -- it's a plain `str | None` in state.py:85. This means if extraction sets `guest_name="Sarah"` in turn 1, it's reset to `None` in turn 2 by `_initial_state()`. The persona envelope node reads `state.get("guest_name")` (persona.py:176), so name injection only works within the turn where the name was first spoken. However, the name IS preserved in `extracted_fields` (via reducer), and the router re-extracts from `extracted_fields["name"]` to set `guest_name` (nodes.py:209-210). Wait -- no: nodes.py:209 only sets `guest_name` when `extracted.get("name")` is truthy, meaning only when the CURRENT message contains a name pattern. If the guest said their name in turn 1 but not turn 2, `guest_name` will be `None` in turn 2 despite `extracted_fields` still containing it. Fix: read `guest_name` from `extracted_fields["name"]` in the persona node or graph dispatch, not just from state. (src/agent/graph.py:517, src/agent/persona.py:176, src/agent/nodes.py:209-210)

---

## Dimension 6: Persona & Voice (8.5/10)

**R19 Fix Verification:**
- R19 MEDIUM: `_inject_guest_name()` lowercased proper nouns ("Mohegan" -> "mohegan"). **FIXED.** persona.py:126-134 now checks `first_word in _LOWERCASE_STARTERS` (a frozenset of 18 common sentence starters: I, We, Our, You, The, There, etc.). Only these are lowercased. Proper nouns like "Mohegan", "Bobby", "Todd" are preserved. This is a clean, maintainable solution.
- R19 MEDIUM: Exclamation replacement with periods still produces awkward text. **Unchanged.** persona.py:64-76 still replaces `!` with `.`. "Welcome! Enjoy your stay. We have great restaurants." The period after "stay" reads unnaturally. MEDIUM: but this is a style preference, not a correctness issue.

**Strengths:**
- 5 truly distinct specialist personas with domain-specific vocabulary, interaction styles, and behavioral guidance. Each persona has a unique identity label (Foodie Insider, Excitement Builder, Comfort Expert, Strategic Advisor, Master Host) and matching behavioral instructions. This is NOT just name-swapping -- each has genuinely different advice on how to communicate (e.g., dining agent paints sensory pictures; hotel agent creates sanctuary language; entertainment agent builds hype).
- The `_LOWERCASE_STARTERS` frozenset approach for name injection is both correct and maintainable. Adding new starter words is trivial. The frozenset is defined inside the function (persona.py:127-131), which is slightly wasteful (recreated per call) but irrelevant at this scale.
- BrandingConfig enforcement in persona_envelope_node correctly processes in safety-first order: PII redaction -> branding -> name injection -> SMS truncation (persona.py:1-10 docstring, persona.py:168-181).
- `get_persona_style()` (prompts.py:241-270) cleanly maps 4 tone options and 3 formality levels to natural-language prompt guidance. The mapping is well-structured and extensible.

**Findings:**

[MEDIUM] [PERSONA] `_LOWERCASE_STARTERS` is defined as a `frozenset` INSIDE `_inject_guest_name()` (persona.py:127-131), meaning it's recreated on every function call. While frozenset creation from a literal is fast (~microseconds), this is a module-level constant pattern and should be defined at module scope for clarity and to signal immutability. Also, the list is incomplete -- missing "Yes", "No", "So", "Well", "Now", "Actually", "Absolutely" which are common sentence starters that should be lowercased. If the LLM generates "Absolutely, we have great dining options" and the guest is "Sarah", the result is "Sarah, Absolutely, we have..." (uppercase A preserved because "Absolutely" is not in the set). (src/agent/persona.py:127-131)

[LOW] [PERSONA] The neutral sentiment tone guide is still empty (prompts.py:291). Since the majority of guest messages are neutral, ~70%+ of interactions receive zero tone guidance beyond the base system prompt. The base persona prompts (e.g., dining_agent.py:17-61) contain rich behavioral guidance, so this is not a gap for specialist agents. But the general host_agent uses `CONCIERGE_SYSTEM_PROMPT` (prompts.py:45-92) which has less specific tone guidance than specialist prompts. For neutral-sentiment + host-agent interactions, the tone calibration is essentially the base prompt alone.

[LOW] [PERSONA] All 5 specialist agents share the same BrandingConfig (DEFAULT_CONFIG). There is infrastructure for per-casino BrandingConfig via Firestore (`get_casino_config()`), but no infrastructure for per-SPECIALIST persona variations within the same casino. The dining agent and entertainment agent could legitimately have different tone settings (dining=luxury, entertainment=casual), but the current architecture applies the same branding to all. This is a documented trade-off.

---

## Dimension 7: Emotional Intelligence (8.0/10)

**R19 Fix Verification:**
- R19 HIGH: Sarcasm detection absent. **FIXED.** sentiment.py:36-51 adds `_SARCASM_PATTERNS` with 6 compiled regex patterns covering: "Great, another...", "Oh wonderful...", "Just great/wonderful/fantastic/perfect/lovely", "Thanks for nothing / Thanks a lot", "Yeah right / Sure, that helps", "Love waiting / Love how...". The patterns fire AFTER frustration and BEFORE VADER (sentiment.py:90-94). This returns "frustrated" for sarcastic input, which is the correct routing -- sarcastic guests need empathetic handling, not enthusiastic matching.
- R19 MEDIUM: No frustration escalation analogous to responsible gaming. **Unchanged.** No session-level frustration counter or escalation mechanism exists. Responsible gaming has `_keep_max` reducer + 3-trigger escalation (compliance_gate.py:113-125). Frustration has no equivalent.
- R19 LOW: Mixed sentiment ("I love this place but the wait is killing me"). **Partially addressed.** "killing" is not a frustrated pattern, but "wait" doesn't trigger either. However, adding sarcasm patterns like "Love waiting" (sentiment.py:50) covers the most common mixed-sentiment sarcasm patterns.

**Strengths:**
- The sarcasm pattern set is well-chosen for casino context. "Great, another 30-minute wait" now correctly returns "frustrated" instead of VADER's false-positive "positive". This was the #1 sarcasm false-positive example cited in R18 and R19.
- Processing order is correct: frustration patterns (deterministic, highest priority) -> sarcasm patterns (deterministic, catches VADER false positives) -> casino overrides (domain-specific) -> VADER (general NLP). Each layer catches what the next would miss.
- VADER singleton fix: `_get_vader_analyzer()` (sentiment.py:57-68) now uses a module-level `_vader_analyzer = None` with lazy initialization, avoiding per-call instantiation. This addresses the R18 MEDIUM about wasteful re-instantiation.
- Responsible gaming escalation (3+ trigger threshold) with `_keep_max` reducer is architecturally sound and works correctly across turns.
- 36 sentiment tests (test_sentiment.py) cover edge inputs, casino overrides, frustration patterns (4 groups), VADER thresholds, priority ordering, and fail-silent behavior. Excellent coverage.

**Findings:**

[MEDIUM] [EQ] Sarcasm patterns have NO test coverage. Despite adding 6 sarcasm patterns to sentiment.py:38-51, test_sentiment.py has ZERO tests for any sarcasm pattern. The grep confirms no "sarcasm" or "_SARCASM" string appears in any test file. This means: (a) no verification that "Great, another 30-minute wait" returns "frustrated", (b) no regression protection if patterns are modified, (c) no false-positive testing for legitimate positive statements like "That's just wonderful, thank you!" (which should NOT be classified as sarcasm but matches the "just wonderful" pattern). The pattern `r"(?i)\bjust\s+(?:great|wonderful|fantastic|perfect|lovely)\b"` would incorrectly classify sincere appreciation: "That was just wonderful, the chef outdid himself" -> "frustrated". This is a false positive in production. (tests/test_sentiment.py -- no sarcasm tests)

[MEDIUM] [EQ] No frustration escalation mechanism. A guest who sends 5 frustrated messages in a row receives the same `SENTIMENT_TONE_GUIDES["frustrated"]` text on every turn. No counter increments, no human escalation, no behavior change. Responsible gaming has a well-designed escalation (3+ triggers -> stronger language, live support suggestion). Frustration needs the same pattern: 3+ frustrated turns -> proactively offer "I can see this hasn't been a great experience. Would you like me to connect you with a live team member who can help directly?" (src/agent -- no frustration escalation)

[LOW] [EQ] `detect_sentiment()` returns a single label without confidence. The function (sentiment.py:71-117) returns exactly one of "positive", "negative", "neutral", "frustrated" with no confidence score. VADER provides a compound score that could be useful for downstream decisions (e.g., compound=-0.31 vs compound=-0.95 are both "negative" but the latter is much stronger). The whisper planner and specialist agents cannot distinguish mildly negative from extremely negative.

---

## Dimension 8: Guest Experience (8.0/10)

**R19 Fix Verification:**
- R19 CRITICAL: `extracted_fields` reset every turn. **FIXED.** `_merge_dicts` reducer preserves fields across turns (state.py:80). This was the highest-priority fix and it's correctly implemented.
- R19 HIGH: `guest_profile_enabled` defaults to False. **FIXED.** feature_flags.py:73 now reads `"guest_profile_enabled": True` with the comment "enabled after R19 review". DEFAULT_CONFIG in config.py:125 also reads `True`. The parity check at feature_flags.py:92-97 ensures these stay in sync.
- R19 MEDIUM: Proactive suggestions absent. **Unchanged.** No specialist agent proactively suggests related items.
- R19 MEDIUM: Task completion tracking absent. **Unchanged.** No request fulfillment tracking exists.

**Strengths:**
- The `_merge_dicts` reducer + `guest_profile_enabled=True` combination means the FULL guest profiling pipeline is now functional: regex extraction (sub-1ms) -> field accumulation across turns -> whisper planner guidance -> guest context injection into specialist prompts -> name injection in persona envelope. This is the end-to-end pipeline that R18 and R19 identified as broken/disabled. It now works.
- Guest profile system (guest_profile.py, 425+ LOC) is production-grade: Firestore persistence with TTL cache, CCPA cascade delete, confidence tracking with decay, bounded in-memory fallback (10K max). The `get_agent_context()` function applies confidence decay and filters low-confidence fields.
- Comp agent profile completeness gate (comp_agent.py:90-102) with 60% threshold (config.py:61: `COMP_COMPLETENESS_THRESHOLD: float = 0.60`) meaningfully gates comp recommendations. With 8 profile fields, this requires 5+ fields before providing personalized comp information. This is genuinely useful hospitality design.
- Guest context injection in `_base.py:128-143` formats 5 field types (name, party_size, visit_date, preferences, occasion) into the system prompt, enabling ALL specialist agents to personalize based on accumulated guest data.
- Name injection (persona.py:97-134) with the proper noun fix is a nice personalization touch. "Sarah, we have several excellent dining options" is warmer than "We have several excellent dining options."

**Findings:**

[MEDIUM] [GUEST] `guest_name` does NOT persist across turns -- only `extracted_fields` does (via reducer). If the guest says "I'm Sarah" in turn 1 and asks about restaurants in turn 2, `state["guest_name"]` is `None` in turn 2 (reset by `_initial_state()` at graph.py:519). The router only sets `guest_name` when the CURRENT message contains a name (nodes.py:209-210). The persona envelope node reads `state.get("guest_name")` (persona.py:176), so name injection fails on turn 2+. The name IS in `extracted_fields["name"]` (via reducer), but the persona envelope doesn't read from there. Fix: in `_dispatch_to_specialist()` or `persona_envelope_node`, fall back to `state.get("extracted_fields", {}).get("name")` when `guest_name` is None. (src/agent/persona.py:176, src/agent/graph.py:519)

[MEDIUM] [GUEST] Proactive suggestions remain entirely absent. No specialist agent proactively cross-references guest data with available offerings. Example: guest mentions anniversary in turn 1, asks about restaurants in turn 2. The dining agent has both pieces of data (via extracted_fields), but the system prompt says nothing about proactive suggestions. A great casino host would say "For your anniversary, I'd especially recommend the candlelit ambiance at Tuscany." The whisper planner's `conversation_note` could theoretically prompt this, but the specialist prompts don't instruct the LLM to act on unsolicited recommendations based on profile data. (src/agent/agents/*.py -- all reactive)

[LOW] [GUEST] Task completion tracking remains absent. Every turn is treated independently (per-turn reset of non-message fields). The whisper planner's `offer_readiness` measures profile completeness, not request fulfillment. For a production casino host, tracking whether "the guest got what they needed" is important for both UX and analytics.

---

## Dimension 9: Domain Expertise (8.5/10)

**R18/R19 Issues Unchanged (by design):**
- Knowledge base: 5 strategic files (comp-system.md, host-workflow.md, retention-playbook.md, state-requirements.md, hey-seven-overview.md). No operational guest-facing data (restaurant menus, entertainment schedules, hotel room types). **This is documented as intentional for the MVP** -- the RAG pipeline is production-grade, but the knowledge base is thin. Real deployment would load property-specific JSON via the ingestion pipeline.
- Responsible gaming helplines: still hardcoded to Connecticut. `get_responsible_gaming_helplines()` (prompts.py:32-38) always returns CT helplines.
- Guardrails: CT-centric (21+, CT self-exclusion). State-specific behavior not implemented.

**Strengths:**
- 5-layer deterministic guardrails (guardrails.py) with 84+ compiled regex patterns across 4 languages (English, Spanish, Portuguese, Mandarin). This is genuinely impressive coverage for a casino AI:
  - Prompt injection (11 patterns + Unicode normalization)
  - Responsible gaming (24 patterns in 4 languages)
  - Age verification (6 patterns)
  - BSA/AML (16 patterns in 4 languages including chip walking and structuring)
  - Patron privacy (12 patterns including social media surveillance)
- Semantic injection classifier (guardrails.py:345-399) provides LLM-based second layer with fail-closed semantics. On classifier failure, returns `is_injection=True` with `confidence=1.0` -- correct for regulated environment.
- Casino comp knowledge in knowledge-base/casino-operations/comp-system.md includes ADT formula with game-specific parameters, reinvestment rates by tier, comp approval authority matrix, real-time vs bounce-back offers. This is operationally accurate.
- COMP_COMPLETENESS_THRESHOLD is now visible: 0.60 (config.py:61). With 8 profile fields, this requires 5+ fields. This is a meaningful gate -- not trivially passed.
- Compliance gate ordering (compliance_gate.py) correctly prioritizes: injection detection -> responsible gaming -> age verification -> BSA/AML -> patron privacy. The docstrings explain WHY each position matters.

**Findings:**

[MEDIUM] [DOMAIN] Knowledge base has ZERO operational guest-facing data. A guest asking "What restaurants do you have?" would get zero retrieval results unless a separate property JSON was loaded. The knowledge-base/ contains ONLY strategic content: comp theory, host workflow, player psychology, regulations, company overview. This is the gap between "casino host agent" and "casino strategy research tool." The RAG ingestion pipeline (pipeline.py) supports per-item chunking of structured data (menus, hours), but no such data exists in the repository. For the PRODUCTION MVP claim, this is a significant gap. (knowledge-base/ -- 5 files, all strategic)

[MEDIUM] [DOMAIN] State-specific regulatory behavior is not implemented beyond documentation. The knowledge-base/regulations/state-requirements.md documents 6 states' requirements, but the code enforces only CT rules. guardrails.py hardcodes 21+ age (CT-specific for Mohegan Sun). For multi-state deployment, the guardrail patterns, helplines, age requirements, and self-exclusion program references would all need property-specific routing. The `get_responsible_gaming_helplines()` function (prompts.py:32-38) is the single point of change but currently returns a static string.

[LOW] [DOMAIN] The comp agent's cautious language rules (comp_agent.py:47-53) are well-designed but rely entirely on prompt compliance. There is no structural guardrail preventing the LLM from promising specific comp amounts. The validation prompt (prompts.py:123-176) checks 6 criteria but none specifically check "no specific dollar amounts promised" or "no eligibility guarantees made." The validation criteria focus on groundedness, on-topic, no gambling advice, read-only, accuracy, and responsible gaming. A comp-specific validation criterion would strengthen the safety net.

---

## Dimension 10: Evaluation Framework (8.0/10)

**R19 Fix Verification:**
- R19 HIGH: LLM-as-judge still offline-only. **Unchanged.** llm_judge.py:532-534 still logs "not yet implemented" and falls back to offline scoring. This is a known limitation, honestly documented.
- R19 HIGH: Circular scenario tests. **Unchanged.** `_build_mock_response()` in test_conversation_scenarios.py still constructs responses FROM expected keywords.
- R19 MEDIUM: No test_persona.py. **FIXED.** tests/test_persona.py exists with thorough tests covering:
  - `_inject_guest_name`: 9 tests (no name, empty name, name present, short response, apology, generic lowercase, proper noun, Bobby Flay, case-insensitive)
  - `_enforce_branding`: 5 tests (exclamation limit default, limit 2, no excess, emoji removed, emoji kept)
  - `_validate_output`: 2 tests (clean text, PII redacted)
  - `persona_envelope_node` integration: 2 tests (name injection pipeline, SMS truncation order)
  - Total: 18 tests. Excellent coverage of the persona pipeline.

**Strengths:**
- New test_persona.py (18 tests) covers the full persona pipeline: PII redaction, branding enforcement, name injection (including the proper noun edge case that R19 flagged), and integration through persona_envelope_node. The SMS truncation order test (test_sms_truncation_after_name_injection) verifies processing order correctness.
- test_sentiment.py (36 tests) and test_extraction.py (52 tests) remain solid. Together with test_persona.py (18 tests), Phase 3 features now have 106 dedicated unit tests.
- Phase 3 baseline document (docs/phase3-baseline.md) is honest about limitations. The per-category breakdown (10 categories x 5 metrics) provides genuine visibility into quality distribution.
- Deterministic golden dataset (20 cases) with weighted scoring (Safety 0.35, Groundedness 0.25, Helpfulness 0.25, Persona 0.15) is appropriate for regulated domain.
- 55 conversation scenarios across 10 well-chosen categories provide structural coverage of the guest journey.
- Total test count: ~1580, up from 1452 pre-Phase 3 (+128 new tests).

**Findings:**

[HIGH] [EVAL] Sarcasm patterns (sentiment.py:38-51) have ZERO test coverage. 6 new regex patterns were added to production code but no tests verify they work correctly or check for false positives. The pattern `r"(?i)\bjust\s+(?:great|wonderful|fantastic|perfect|lovely)\b"` would classify sincere gratitude ("That was just wonderful, thank you so much!") as "frustrated" -- a false positive that would trigger empathetic de-escalation tone for a happy guest. This is the ONLY untested production code path added in the recent fixes. All other Phase 3 additions have dedicated test files. (tests/test_sentiment.py -- no sarcasm tests)

[MEDIUM] [EVAL] LLM-as-judge remains offline-only (llm_judge.py:532-534). The 5 quality metrics are keyword-heuristic proxies. Empathy baseline 0.30 for any non-empty response compresses scoring range. Cultural sensitivity baseline 0.70 further compresses. These baselines mean a cold "The restaurant is on floor 2" response scores 0.30 empathy -- only 0.2 points below the average. The evaluation framework measures keyword presence, not actual behavioral quality. Honestly documented but limits the "evaluation framework" dimension's value.

[MEDIUM] [EVAL] Conversation scenario tests remain tautological. `_build_mock_response()` constructs responses FROM expected keywords, then tests verify those keywords appear. This is circular: 55 "tests" pass trivially because the mocks are built to contain what's being tested. These tests validate mock construction logic, not agent quality. For real quality measurement, at least a subset should test with actual LLM responses (or at minimum, responses NOT derived from the expected keywords).

[MEDIUM] [EVAL] No regression detection framework. The evaluation system takes snapshots (phase3-baseline.md) but has no automated mechanism to detect quality drops. No CI gate fails if empathy drops below 0.40, no comparison against baseline runs automatically. Quality regressions between versions are invisible without manual review.

[LOW] [EVAL] The `_WhisperTelemetry` class (whisper_planner.py:94-110) has been fixed to use `__init__` instance attributes instead of class-level attributes (R19 LOW resolved). This is now correct.

---

## Agent Quality Subtotal: 49.5/60

| Dimension | R18 | R19 | R20 | Delta R19->R20 | Key Improvements |
|-----------|-----|-----|-----|----------------|------------------|
| 5. Conversation Quality | 7.0 | 7.5 | 8.5 | +1.0 | extracted_fields reducer FIXED -- multi-turn profiling now works |
| 6. Persona & Voice | 8.0 | 8.0 | 8.5 | +0.5 | Proper noun name injection FIXED, _LOWERCASE_STARTERS approach clean |
| 7. Emotional Intelligence | 6.0 | 7.0 | 8.0 | +1.0 | Sarcasm detection ADDED (6 patterns), VADER singleton FIXED |
| 8. Guest Experience | 6.0 | 6.5 | 8.0 | +1.5 | extracted_fields persist + guest_profile_enabled=True -- full pipeline works |
| 9. Domain Expertise | 8.0 | 8.0 | 8.5 | +0.5 | COMP_COMPLETENESS_THRESHOLD visible (0.60), guardrails comprehensive |
| 10. Evaluation Framework | 7.0 | 7.5 | 8.0 | +0.5 | test_persona.py ADDED (18 tests), 106 dedicated Phase 3 tests total |

## Trajectory: R18=42/60 -> R19=44.5/60 (+2.5) -> R20=49.5/60 (+5.0)

Total improvement: **+7.5 points** across 3 rounds. Strongest improvement in Guest Experience (+2.0 from R18) driven by the extracted_fields reducer and guest_profile_enabled fixes.

## Critical/High Finding Summary

| # | Severity | Dimension | Finding |
|---|----------|-----------|---------|
| 1 | HIGH | EVAL | Sarcasm patterns (6 regexes in sentiment.py:38-51) have ZERO test coverage -- false positive risk for sincere positive statements |
| 2 | MEDIUM | CONV | guest_name does not persist across turns (no reducer), name injection fails on turn 2+ unless name re-spoken |
| 3 | MEDIUM | EQ | No sarcasm test coverage -- "Just wonderful, thank you!" would be false-positive classified as frustrated |
| 4 | MEDIUM | EQ | No frustration escalation (responsible gaming has 3-trigger escalation, frustration has none) |
| 5 | MEDIUM | GUEST | Proactive suggestions absent -- agent is purely reactive, never cross-references profile with offerings |
| 6 | MEDIUM | DOMAIN | Knowledge base has zero operational guest-facing data (no menus, schedules, room types) |
| 7 | MEDIUM | EVAL | LLM-as-judge still offline-only, scenario tests still tautological, no regression detection |

## Remaining Actionable Fixes (Priority Order)

1. **[HIGH] Add sarcasm tests to test_sentiment.py**: Test all 6 patterns for correct detection AND test false-positive cases (sincere "Just wonderful, thank you!" should be positive, not frustrated). The "just wonderful" pattern is particularly risky for false positives.

2. **[MEDIUM] Fix guest_name cross-turn persistence**: In persona_envelope_node, fall back to `state.get("extracted_fields", {}).get("name")` when `guest_name` is None. This is a 2-line fix that completes the personalization pipeline.

3. **[MEDIUM] Add frustration escalation**: Track frustrated sentiment count with `_keep_max` reducer (same pattern as `responsible_gaming_count`). At 3+ frustrated turns, proactively offer human escalation.

4. **[MEDIUM] Move `_LOWERCASE_STARTERS` to module scope** and add missing starters (Yes, No, So, Well, Actually, Absolutely).

## Final Assessment: Would This Agent Be a GREAT Casino Host?

**As a TECHNICAL system**: Yes. The architecture is production-grade. The 11-node StateGraph with validation loops, 5-layer guardrails, circuit breakers, degraded-pass validation, streaming PII redaction, and multi-tenant feature flags is genuinely well-engineered. The code quality, documentation, and test coverage (1580 tests) are strong.

**As an ACTUAL casino host**: Good, not yet great. Three gaps prevent "great":

1. **Reactive, not proactive**: A great casino host says "Since you mentioned your anniversary, you should check out the spa's couples package." This agent answers questions but never volunteers connections between what it knows about the guest and what the property offers. The whisper planner infrastructure is there; the specialist prompts just need instructions to act on guest profile data proactively.

2. **Guest name doesn't stick**: If Sarah introduces herself in turn 1, the host should call her "Sarah" in turns 2, 3, 4, etc. Currently, name injection only works in the turn where the name was spoken. The fix is small (2 lines), but until it's done, the personalization feels forgetful.

3. **Knowledge base is strategic, not operational**: A casino host needs to know restaurant menus, show schedules, room types, and spa services. The knowledge base contains comp theory and regulatory analysis -- valuable for configuring the system, but not for answering "What's for dinner?" The RAG pipeline is ready for operational data; it just hasn't been fed any.

Despite these gaps, the agent handles the hard stuff well: responsible gaming escalation, BSA/AML deflection, patron privacy, multilingual guardrails, and regulated-environment safety. For a seed-stage MVP, this is a strong foundation. The gaps are features to add, not architectural flaws to fix.
