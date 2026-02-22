# R19 Agent Quality Review (Dimensions 5-10)

Reviewer: r19-reviewer-beta
Date: 2026-02-22
Commit: 52aa84c (Phase 3 Agent Quality Revolution)

---

## Previous R18 Scores: Conv 7.0, Persona 8.0, EQ 6.0, Guest 6.0, Domain 8.0, Eval 7.0 = 42/60

---

## Dimension 5: Conversation Quality (7.5/10)

**R18 Fix Verification:**
- R18 flagged `_inject_guest_name()` as a no-op (persona.py:122 returned `content` unchanged). **FIXED.** Line 123 now reads `return f"{guest_name}, {content[0].lower()}{content[1:]}"`. The function correctly prepends guest name when conditions are met. (+0.5 from R18)
- R18 flagged missing test_sentiment.py and test_extraction.py. **FIXED.** Both files exist with 36 and 52 tests respectively.

**Strengths:**
- WhisperPlan structured output (whisper_planner.py:114-131) with `next_topic`, `extraction_targets`, `offer_readiness`, `conversation_note` is a well-designed multi-turn guidance contract.
- Profile completeness feedback loop (whisper_planner.py:255-272) enables progressively smarter conversation guidance.
- Field accumulation across turns (nodes.py:204-206) correctly merges new extractions without overwriting prior fields.
- 55 conversation scenarios across 10 categories (test_conversation_scenarios.py) provide solid coverage.

**Findings:**

[HIGH] [CONV] Whisper planner output is consumed by ALL specialist agents (include_whisper=True in all 5 agents), but there is no mechanism to verify whether the LLM actually followed the whisper guidance. The `WhisperPlan.next_topic` field tells the agent "explore dining preferences next," but the generated response is validated only for factual accuracy (validation_prompt.py 6 criteria: grounded, on-topic, no gambling, read-only, accurate, responsible gaming). None of these criteria check whether the agent followed the whisper planner's conversational guidance. This means the multi-turn progression system is advisory only -- the validation loop cannot enforce it. (src/agent/prompts.py:123-176 validation criteria, src/agent/agents/_base.py:146-149 whisper injection)

[MEDIUM] [CONV] `_format_history()` (whisper_planner.py:280-297) silently drops SystemMessage objects from the conversation view. When a retry occurs, the retry feedback is injected as a SystemMessage (_base.py:172-178), but the whisper planner's next turn will see the message history WITHOUT that retry context. The planner may therefore suggest the same topic that just caused a validation failure, creating a retry loop that the planner cannot learn from. (src/agent/whisper_planner.py:291-296)

[MEDIUM] [CONV] The `_initial_state()` function (graph.py:493-520) resets `extracted_fields` to `{}` every turn. While `messages` persists via the `add_messages` reducer, extracted fields do NOT persist across turns through the checkpointer -- they only accumulate WITHIN a single turn's processing via `existing.update(extracted)` in nodes.py:205-206. If the user says "My name is Sarah" in turn 1 and "party of 4" in turn 2, the name from turn 1 is LOST because `extracted_fields` was reset to `{}` before turn 2 started. The accumulation logic only works for MULTIPLE extractions in a SINGLE message. This fundamentally breaks the claimed multi-turn field accumulation. (src/agent/graph.py:513, src/agent/nodes.py:204-206)

[LOW] [CONV] `_calculate_completeness()` (whisper_planner.py:255-272) uses equal weight for all 8 profile fields (`name`, `visit_date`, `party_size`, `dining`, `entertainment`, `gaming`, `occasions`, `companions`). In practice, `name` and `visit_date` are far more likely to be offered by guests than `gaming` or `companions`. Equal weighting means a profile with name+date+party (3/8 = 37.5%) looks 37.5% complete, but from a host's perspective it has the 3 most critical fields. The whisper planner prompt says "Prioritize high-weight fields (name, visit_date, party_size)" (prompts.py:207) but the completeness metric doesn't reflect this weighting.

---

## Dimension 6: Persona & Voice (8.0/10)

**R18 Fix Verification:**
- R18 flagged `_inject_guest_name()` no-op. **FIXED** (persona.py:123). Name injection now functional.
- R18 noted only one BrandingConfig exists. **Still true** -- single DEFAULT_CONFIG (casino/config.py). Documented as intentional.

**Strengths:**
- 5 genuinely distinct specialist personas with domain-specific vocabulary and behavioral guidance (dining_agent.py:16-61, comp_agent.py:22-75, hotel_agent, entertainment_agent, host_agent prompts).
- Persona style injection via `get_persona_style()` (prompts.py:241-270) maps BrandingConfig to natural language prompt guidance with 4 tone and 3 formality options.
- `PERSONA_STYLE_TEMPLATE` is injected into ALL specialist agents via `_base.py:151-159`, ensuring consistent persona expression.
- BrandingConfig enforcement in `persona_envelope_node` (persona.py:47-94) correctly enforces exclamation limits and emoji removal at the output layer.

**Findings:**

[MEDIUM] [PERSONA] The exclamation limit enforcement (persona.py:61-76) replaces excess `!` with `.`, which produces grammatically awkward output. Example: input "Welcome! Enjoy your stay! We have great restaurants!" with limit=1 becomes "Welcome! Enjoy your stay. We have great restaurants." The `.` after "stay" creates an unnaturally abrupt sentence where an exclamation was clearly intended by the LLM. A better approach would be to simply strip excess exclamation marks (replace with empty string) and let the LLM's sentence structure stand, or replace with nothing to join sentences. (src/agent/persona.py:66-76)

[MEDIUM] [PERSONA] Guest name injection (persona.py:122-123) lowercases the first character of the content: `return f"{guest_name}, {content[0].lower()}{content[1:]}"`. This is correct for content starting with "I" or "We" but WRONG for content starting with a proper noun or "Hi". Example: if the LLM generates "Mohegan Sun has wonderful dining options" and guest_name="Sarah", the result is "Sarah, mohegan Sun has wonderful dining options" -- lowercase "m" in a proper noun. The function should NOT blindly lowercase the first character. (src/agent/persona.py:123)

[LOW] [PERSONA] The `SENTIMENT_TONE_GUIDES` dict (prompts.py:277-292) has an empty string for "neutral" sentiment. Since the majority of guest messages are neutral, most interactions receive ZERO tone guidance. This means tone calibration only activates for emotionally charged messages. While arguably intentional (don't over-instruct), it means the persona's warmth depends entirely on the base system prompt for ~70%+ of interactions.

---

## Dimension 7: Emotional Intelligence (7.0/10)

**R18 Fix Verification:**
- R18 flagged missing test_sentiment.py. **FIXED.** `tests/test_sentiment.py` exists with 36 tests covering: edge inputs (empty, whitespace, None), casino overrides (8 phrases), frustration patterns (explicit words, can't-find, exhaustion, what-a), VADER thresholds, frustration priority over VADER, fail-silent behavior.
- R18 flagged missing test_extraction.py. **FIXED.** `tests/test_extraction.py` exists with 52 tests covering: edge inputs, name patterns + false positives, party size boundaries, visit dates, preferences/dietary, occasions, multi-field extraction, fail-silent.
- R18 flagged VADER `SentimentIntensityAnalyzer()` instantiated per call. **NOT FIXED.** Line 65-66 still creates a new instance per call. However, this is functionally minor (VADER instantiation is fast after first import).

**Strengths:**
- VADER + casino-specific overrides (sentiment.py:17-26) with 8 casino-positive phrases. These correctly handle "killing it", "hit the jackpot", "on fire" etc. that VADER would misclassify.
- Frustration detection (sentiment.py:29-33) runs BEFORE VADER with 4 compiled regex pattern groups, giving deterministic priority to strong frustration signals.
- Sentiment propagation chain is complete: detect_sentiment in router (nodes.py:192-195) -> state.guest_sentiment -> SENTIMENT_TONE_GUIDES injection in _base.py:161-166.
- Responsible gaming escalation with session-level counter via `_keep_max` reducer (state.py:78, compliance_gate.py:113-125).
- New test suite (test_sentiment.py) has excellent coverage of casino-specific edge cases.

**Findings:**

[HIGH] [EQ] Sarcasm detection remains completely absent (unchanged from R18). VADER is notoriously poor with sarcasm. "Great, another 30-minute wait for a table" scores positive with VADER (compound ~0.6 due to "great"). The casino domain has HIGH sarcasm incidence from frustrated guests. Without sarcasm handling, the agent will respond enthusiastically ("Glad you're having a great time!") to guests who are clearly frustrated. Even a simple negation-before-positive heuristic ("great" preceded by "another", "just", "oh", or followed by negative context) would significantly improve this. (src/agent/sentiment.py -- no sarcasm patterns)

[MEDIUM] [EQ] Frustration handling has no structural behavior change -- it only adds tone guidance text to the system prompt (_base.py:161-166). A frustrated guest still gets the same retrieval depth, the same validation strictness, the same retry budget, and the same response latency. A production casino host would: (a) skip validation to serve faster, (b) proactively offer human escalation, (c) increase retrieval breadth to provide more options. The current implementation trusts the LLM to "be empathetic" based on a text prompt injection. (src/agent/agents/_base.py:161-166)

[MEDIUM] [EQ] There is no frustration escalation path analogous to responsible gaming escalation. Responsible gaming has a session counter with escalation at 3+ triggers (compliance_gate.py:113-125). But a guest who sends 5 frustrated messages in a row gets the same tone guidance every time with no escalation to human support. The `off_topic_node` has escalation for gambling issues but nothing for general frustration. (src/agent/nodes.py:548-649 -- no frustration escalation)

[LOW] [EQ] The `detect_sentiment()` function (sentiment.py:37-77) has no handling for mixed sentiment ("I love this place but the wait is killing me"). The frustration patterns fire first because "killing" is not in `_FRUSTRATED_PATTERNS`, and VADER would score the compound as slightly positive due to "love". This specific example returns "positive" when the guest is actually expressing frustration. The function returns a single label with no confidence score.

---

## Dimension 8: Guest Experience (6.5/10)

**R18 Fix Verification:**
- R18 flagged `_inject_guest_name()` no-op. **FIXED.** Name injection now works (persona.py:123).
- R18 flagged `guest_profile_enabled` defaults to False. **STILL FALSE** (casino/config.py:125). The entire guest profile system remains disabled by default. However, extracted_fields still accumulate in state via nodes.py:201-210.

**Strengths:**
- Field extraction (extraction.py) covers 5 key data points with well-crafted regex and false-positive prevention (`_COMMON_WORDS` exclusion).
- Guest profile system (guest_profile.py) is production-grade: Firestore persistence, CCPA cascade delete, confidence tracking with decay, bounded in-memory fallback.
- Comp agent profile completeness gate (comp_agent.py:90-102) asks for more info before recommending offers.
- Guest context injection in `_base.py:128-143` formats extracted fields into the system prompt.
- New test_extraction.py covers boundary values, false positive exclusions, multi-field extraction.

**Findings:**

[CRITICAL] [GUEST] The `extracted_fields` accumulation across turns is BROKEN. `_initial_state()` in graph.py:513 resets `extracted_fields` to `{}` every turn. The `add_messages` reducer persists messages across turns, but `extracted_fields` has NO reducer -- it is a plain `dict[str, Any]` in PropertyQAState (state.py:67). The accumulation code in nodes.py:204-206 (`existing = dict(state.get("extracted_fields", {}) or {})`) starts from an empty dict every turn because `_initial_state` wiped it. This means: (a) guest name said in turn 1 is lost by turn 2, (b) party size from turn 1 is lost, (c) the whisper planner sees a fresh empty profile every turn, (d) field accumulation "across turns" only works within a single turn. This is a FUNDAMENTAL break in the claimed multi-turn profiling capability. The phase3-baseline.md doc says "Fields accumulate across conversation turns" (line 127) but this is FALSE. (src/agent/graph.py:513, src/agent/state.py:67 -- no reducer for extracted_fields)

[HIGH] [GUEST] `guest_profile_enabled` still defaults to False (casino/config.py:125). This means `get_agent_context()` in graph.py:274 is never called. The 425 LOC guest_profile.py module with Firestore persistence, CCPA cascade delete, confidence tracking, and decay -- ALL of this is scaffolded infrastructure that never activates in default configuration. The extracted fields from nodes.py go into `state["extracted_fields"]` but are NEVER processed through the profile system. The guest profile system exists but is NOT wired to production. (src/casino/config.py:125, src/agent/graph.py:274)

[MEDIUM] [GUEST] Proactive suggestions remain entirely absent. No specialist agent proactively suggests related items. A dining agent answering "What are your restaurant hours?" never says "By the way, since you mentioned your anniversary earlier, Tuscany's candlelit dining is wonderful for celebrations." The whisper planner's `conversation_note` could theoretically trigger proactive suggestions, but no specialist prompt has instructions to act on unsolicited recommendations. Every interaction is purely reactive: question in -> answer out. (src/agent/agents/*.py -- all reactive)

[MEDIUM] [GUEST] Task completion tracking is absent. There is no mechanism to track whether a guest's request was fulfilled. The agent treats every turn as independent (per-turn reset). A real host tracks: "Guest asked about restaurants -> recommended Tuscany -> guest asked for hours -> provided hours -> TASK COMPLETE." The `offer_readiness` field in WhisperPlan measures profile completeness, not request fulfillment. (src/agent/state.py -- no task_status or request_tracking field)

[LOW] [GUEST] Handoff to human support is limited to static contact info in fallback_node (nodes.py:436-444). There is no warm handoff mechanism, no CRM ticket creation, no notification to a live host. The escalation for frustrated guests simply includes the same phone number as any other fallback.

---

## Dimension 9: Domain Expertise (8.0/10)

**R18 Issues Unchanged:**
- Knowledge base still only 5 files across 4 directories (casino-operations/comp-system.md, casino-operations/host-workflow.md, player-psychology/retention-playbook.md, regulations/state-requirements.md, company-context/hey-seven-overview.md). No specific restaurant menus, entertainment schedules, hotel room details, spa menus, or gaming floor layout.
- Responsible gaming helplines still hardcoded to Connecticut (prompts.py:22-26).

**Strengths:**
- Casino comp knowledge (knowledge-base/casino-operations/comp-system.md) includes ADT formula, reinvestment rates, comp authority matrix. Genuinely deep.
- Host workflow knowledge covers portfolio structure, daily schedules, KPIs. Operationally accurate.
- Regulatory knowledge (knowledge-base/regulations/state-requirements.md) is current: TCPA, NJ SB 3401, PA KYC, CCPA.
- 5-layer deterministic guardrails (guardrails.py) with 84 compiled regex patterns across 4 languages.
- BSA/AML patterns (guardrails.py:129-159) cover real casino-specific financial crime: structuring, chip walking, multi-buy-in structuring.
- Compliance gate ordering (compliance_gate.py:46-91) is correctly prioritized with injection detection before content guardrails. The docstring explains WHY each position matters.

**Findings:**

[MEDIUM] [DOMAIN] Knowledge base has NO property-specific operational data (restaurant menus, hours, prices, entertainment schedules, room types, spa services). The knowledge-base/ directory contains ONLY strategic/regulatory content: comp system theory, host workflow theory, player psychology, regulations, and company overview. An actual guest asking "What restaurants do you have?" would get zero retrieval results unless a separate property data JSON was loaded via RAG ingestion. The RAG pipeline is production-grade but the knowledge base is strategic documentation, not operational guest-facing data. This is the biggest gap between "casino host agent" and "casino strategy research tool." (knowledge-base/ -- 5 files, all strategic)

[MEDIUM] [DOMAIN] State-specific regulatory behavior is not implemented. The guardrails (guardrails.py) are hardcoded for Mohegan Sun CT: 21+ age requirement, CT self-exclusion program. For deployment to a Nevada casino (gaming age 21, different self-exclusion program, different regulatory body) or NJ (self-exclusion list managed by DGE, different TCPA requirements), the code would need modifications. The `get_responsible_gaming_helplines()` function exists (prompts.py:32-38) but always returns CT helplines. (src/agent/prompts.py:32-38, src/agent/guardrails.py)

[LOW] [DOMAIN] The `COMP_COMPLETENESS_THRESHOLD` in comp_agent.py:93 references `settings.COMP_COMPLETENESS_THRESHOLD` but its default value is not visible in the comp agent code. With 8 profile fields in `_PROFILE_FIELDS`, a threshold of even 0.1 means 1/8 fields suffices. Without knowing the threshold value, it's unclear whether the completeness gate is meaningful or trivially passed.

---

## Dimension 10: Evaluation Framework (7.5/10)

**R18 Fix Verification:**
- R18 flagged missing test_sentiment.py and test_extraction.py. **FIXED.** Both exist with 36 and 52 tests respectively.
- R18 flagged LLM-as-judge is offline-only. **STILL OFFLINE.** Line 533-534 of llm_judge.py still says "not yet implemented" and falls back to offline scoring.
- R18 flagged circular scenario tests. **NOT FIXED.** `_build_mock_response()` in test_conversation_scenarios.py:120-249 still constructs responses FROM expected keywords. Tests still validate mocks, not agent behavior.

**Strengths:**
- New test_sentiment.py (36 tests) has excellent coverage: edge inputs, casino overrides (8 parametrized cases), frustration patterns (4 pattern groups), VADER thresholds, priority ordering, fail-silent.
- New test_extraction.py (52 tests) covers: edge inputs, name patterns with false-positive prevention, party size boundaries, visit dates, dietary preferences, occasions, multi-field extraction.
- 5 LLM-as-judge metrics (empathy, cultural_sensitivity, conversation_flow, persona_consistency, guest_experience) with offline deterministic heuristics.
- Phase 3 baseline document (docs/phase3-baseline.md) honestly records per-category breakdowns and acknowledges heuristic limitations.
- Evaluation weights (Safety 0.35, Groundedness 0.25, Helpfulness 0.25, Persona 0.15) are appropriate for regulated domain.
- Test count progression: 1452 -> 1580 (+128 tests across Phases 2-5).

**Findings:**

[HIGH] [EVAL] LLM-as-judge remains offline-only with keyword heuristics (llm_judge.py:532-534). The `EVAL_LLM_ENABLED=true` code path logs "not yet implemented" and falls back to offline scoring. All 5 quality metrics are keyword proxies. The baseline scores in phase3-baseline.md are keyword-heuristic scores dressed as quality metrics. Empathy 0.50, conversation flow 0.59 -- these numbers measure keyword presence, not actual empathy or conversational quality. The gap between keyword heuristic and actual LLM-judge scoring is likely 20-40%. (src/observability/llm_judge.py:533-534)

[HIGH] [EVAL] Conversation scenario tests (test_conversation_scenarios.py:324-335) remain circular. `_build_mock_response()` (lines 120-249) is a 130-line function that constructs mock LLM responses FROM the expected keywords defined in the YAML scenarios. The test then verifies "at least one expected keyword appears" (line 329). This is tautological: the mock is built to contain the keywords, then the test checks for the keywords. The test validates the mock construction logic, not the agent's response quality. 55 scenario "tests" pass trivially because the mocks are designed to pass. Real validation requires testing with actual or at least unseen LLM responses. (tests/test_conversation_scenarios.py:120-249, 324-335)

[MEDIUM] [EVAL] No dedicated persona test file. Despite persona being a major Phase 3 feature (5 personas, BrandingConfig enforcement, name injection, tone calibration), there is NO `tests/test_persona.py`. The `_inject_guest_name()` fix was a R18 CRITICAL finding that is now verified only by grep-searching test_graph_v2.py where there are 2 tests (test_persona_returns_empty, test_persona_constant). These test passthrough behavior, not the injection logic, branding enforcement, exclamation limiting, or emoji removal. The persona envelope node handles safety-critical processing (PII redaction order, branding, name injection, SMS truncation) and has zero dedicated unit tests. (tests/ -- no test_persona.py)

[MEDIUM] [EVAL] Empathy scoring baseline is 0.30 for ANY non-empty response (llm_judge.py:242). Cultural sensitivity baseline is 0.70 (llm_judge.py:307). These high baselines compress the dynamic range. The empathy scale effectively becomes 0.30-1.00 (range of 0.70), and cultural sensitivity 0.70-1.00 (range of 0.30). This makes cultural sensitivity appear "strong" (0.80 average) when the actual discrimination power is minimal -- the score can only vary by 0.30 points. The R18 finding about this is unchanged.

[MEDIUM] [EVAL] No regression detection framework. The evaluation system measures snapshots but has no CI gate that fails if quality drops below baseline. `phase3-baseline.md` records numbers but nothing compares current scores against these baselines automatically. Quality regressions between versions are invisible without manual comparison. (docs/phase3-baseline.md -- baseline exists, no automated comparison)

[LOW] [EVAL] The `_WhisperTelemetry` class (whisper_planner.py:94-100) uses class-level attributes `count: int = 0` and `alerted: bool = False` that look like class variable annotations but are actually shared across ALL instances. Since only `_telemetry = _WhisperTelemetry()` is instantiated (line 103), this works correctly, but the pattern is fragile -- if anyone instantiated a second `_WhisperTelemetry()`, both instances would share the same count. Should use `__init__` or a simpler module-level dict.

---

## Agent Quality Subtotal: 44.5/60

| Dimension | R18 | R19 | Delta | Key Issues |
|-----------|-----|-----|-------|-----------|
| 5. Conversation Quality | 7.0 | 7.5 | +0.5 | Name injection fixed, but extracted_fields don't persist across turns, whisper guidance unverified |
| 6. Persona & Voice | 8.0 | 8.0 | 0.0 | Strong personas, but name injection lowercases proper nouns, exclamation replacement awkward |
| 7. Emotional Intelligence | 6.0 | 7.0 | +1.0 | Excellent new test coverage (88 tests), but no sarcasm detection, no frustration escalation |
| 8. Guest Experience | 6.0 | 6.5 | +0.5 | Name injection works, but extracted_fields reset per turn BREAKS multi-turn profiling, profile system still disabled |
| 9. Domain Expertise | 8.0 | 8.0 | 0.0 | Deep regulatory/strategic knowledge, but knowledge base has zero operational guest-facing data |
| 10. Evaluation Framework | 7.0 | 7.5 | +0.5 | 88 new sentiment+extraction tests, but LLM judge still offline, scenario tests still circular, no test_persona.py |

## Delta from R18: +2.5

## Critical/High Finding Summary

| # | Severity | Dimension | Finding |
|---|----------|-----------|---------|
| 1 | CRITICAL | GUEST/CONV | `extracted_fields` reset to `{}` every turn by `_initial_state()` -- multi-turn field accumulation is broken (graph.py:513, state.py:67) |
| 2 | HIGH | CONV | Whisper planner guidance has no validation enforcement -- 6 validation criteria check factual accuracy, not conversational progression |
| 3 | HIGH | EQ | Sarcasm detection completely absent -- VADER misclassifies sarcastic frustrated guests as positive |
| 4 | HIGH | GUEST | `guest_profile_enabled` still defaults to False -- 425 LOC profile system is scaffolded, not active |
| 5 | HIGH | EVAL | LLM-as-judge still offline-only -- keyword heuristics are proxies, not quality measurement |
| 6 | HIGH | EVAL | Conversation scenario tests are tautological -- mock responses built from expected keywords, tests validate mocks |

## Actionable Fixes (Priority Order)

1. **[CRITICAL] Add reducer for `extracted_fields`** or persist them via checkpointer. Without this, the entire multi-turn profiling claim is false. Options: (a) use `Annotated[dict, merge_dicts]` reducer in PropertyQAState, (b) store extracted fields in a separate persistence layer, (c) reconstruct from message history each turn.
2. **Fix `_inject_guest_name()` proper noun handling** (persona.py:123): Check if `content[0].isupper()` and the first word is a proper noun before lowercasing. Or only lowercase when the first character is a capital letter that should be lowercase (e.g., sentence-initial "I").
3. **Add basic sarcasm detection** to sentiment.py: at minimum, detect negation + positive word patterns ("Great, another...", "Oh wonderful, I love waiting...").
4. **Enable `guest_profile_enabled` by default** or document clearly why it's disabled and what's blocking it.
5. **Add `tests/test_persona.py`**: Unit tests for exclamation enforcement, emoji removal, name injection (including proper noun edge case), PII redaction ordering, SMS truncation.
6. **Implement actual LLM-as-judge** or at minimum remove the "evaluation framework" scoring claim and call it "keyword heuristic check."
