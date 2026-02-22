# R18 Agent Quality Review (Dimensions 5-10)

Reviewer: reviewer-beta-v2
Date: 2026-02-22
Commit: 52aa84c (Phase 3 Agent Quality Revolution)

---

## Dimension 5: Conversation Quality (7/10)

**Strengths:**
- Whisper planner (`src/agent/whisper_planner.py`) provides a thoughtful multi-turn conversation guidance system. The WhisperPlan structured output with `next_topic`, `extraction_targets`, `offer_readiness`, and `conversation_note` is a well-designed contract for guiding natural conversation progression.
- Profile completeness calculation (`_calculate_completeness()`, whisper_planner.py:255-272) feeds back into the planner, enabling progressively smarter conversation guidance as the guest profile fills up.
- Message windowing via `MAX_HISTORY_MESSAGES` in `_base.py:186` prevents unbounded context growth, and the 20-message window in whisper planner history (`whisper_planner.py:159`) is appropriate for planning horizon.
- `_initial_state()` pattern properly resets per-turn fields while preserving cross-turn `messages` and `responsible_gaming_count` via reducers (state.py:56,78). This is correct multi-turn state management.
- 55 conversation scenarios across 10 categories (test_conversation_scenarios.py) with multi-turn validation is a solid coverage foundation.

**Findings:**

[HIGH] [CONV] No dedicated unit tests for `detect_sentiment()` or `extract_fields()` functions. These are core Phase 3 features invoked on every router call (nodes.py:193-210) but have zero standalone test files. The only test coverage is indirect through integration tests and YAML scenario tests that use mocked LLMs. If regex patterns in extraction.py break, no test catches it directly. (tests/ directory -- missing test_sentiment.py and test_extraction.py)

[HIGH] [CONV] `_inject_guest_name()` in persona.py:97-122 is effectively a no-op. The function checks conditions (name present, not already in response, response > 50 chars, no "I apologize") but if all pass, it simply `return content` unchanged on line 122. The name is never actually injected into the text. This means guest name personalization is scaffolded but not functional. (src/agent/persona.py:122)

[MEDIUM] [CONV] Whisper planner runs between retrieve and generate (graph.py topology), but its output is only consumed if `include_whisper=True` in execute_specialist (`_base.py:146-149`). All 5 specialist agents set `include_whisper=True`, which is correct. However, the planner produces guidance that the LLM may or may not follow -- there is no verification that the LLM actually acts on whisper guidance. The system trusts the LLM to follow internal guidance. (src/agent/agents/_base.py:146-149)

[MEDIUM] [CONV] Topic switching is covered in edge_cases scenarios but return-to-previous-topic is not tested. The agent has no explicit mechanism to remember or reference earlier topics from within the same session beyond what survives in the message window. (tests/scenarios/edge_cases.yaml)

[LOW] [CONV] `_format_history()` (whisper_planner.py:280-297) only handles HumanMessage and AIMessage types. SystemMessage injections (retry feedback, whisper guidance) are silently dropped from the planner's conversation view, which could cause the planner to miss context about validation retries.

---

## Dimension 6: Persona & Voice (8/10)

**Strengths:**
- 5 truly distinct specialist personas with unique interaction styles:
  - **Dining ("The Foodie Insider")**: Sensory descriptions, celebration awareness, dietary-as-opportunity framing (dining_agent.py:17-61)
  - **Entertainment ("The Excitement Builder")**: Hype language, VIP framing, atmosphere painting (entertainment_agent.py:16-67)
  - **Hotel ("The Comfort Expert")**: Sanctuary language, aspiration-without-pressure upgrades, family thoughtfulness (hotel_agent.py:16-60)
  - **Comp ("The Strategic Advisor")**: Insider track language, cautious encouraging, tier awareness (comp_agent.py:22-75)
  - **Host ("The Master Host")**: Warmest person at property, curated over raw, energy mirroring (prompts.py:46-92)
- These are NOT just name changes -- each prompt section has domain-specific behavioral guidance, distinct vocabulary, and different emotional calibration.
- BrandingConfig enforcement in persona_envelope_node (persona.py:47-94) implements exclamation limit and emoji removal as post-processing guardrails. This correctly enforces branding at the output layer rather than relying on LLM compliance.
- `get_persona_style()` (prompts.py:241-270) maps BrandingConfig values to natural-language prompt guidance with 4 tone options and 3 formality levels. The mapping from config to prompt language is well-designed.
- `PERSONA_STYLE_TEMPLATE` injection happens in `_base.py:151-159` (fail-silent), meaning persona style is applied to ALL specialist agents consistently.

**Findings:**

[HIGH] [PERSONA] Guest name injection is broken (same finding as CONV dimension). `_inject_guest_name()` (persona.py:97-122) never actually modifies the content when all guard conditions pass. Line 122 returns `content` unmodified. The docstring says "prepend a personalized greeting" but no greeting is prepended. This means the "Name injection (guest name usage)" feature claimed in phase3-baseline.md:129 is not functional. (src/agent/persona.py:122)

[MEDIUM] [PERSONA] Only ONE branding configuration exists (DEFAULT_CONFIG in casino/config.py:135-141): persona_name="Seven", tone="warm_professional", formality="casual_respectful", emoji=False, exclamation_limit=1. The infrastructure supports 5 personas but only one is configured. Multi-property deployment with different personas would require adding configs to Firestore. This is documented as intentional (phase3-baseline.md:161) but limits current persona variety. (src/casino/config.py:135-141)

[MEDIUM] [PERSONA] The exclamation limit enforcement (persona.py:61-76) replaces excess exclamation marks with periods. This produces awkward text like "Welcome to Mohegan Sun. I'd love to help." (where "." replaced "!"). A more natural approach would strip excess exclamations rather than convert to periods, or reduce to zero exclamation if already at limit. (src/agent/persona.py:66-76)

[LOW] [PERSONA] `SENTIMENT_TONE_GUIDES` (prompts.py:277-292) includes guides for "frustrated", "negative", "positive", and "neutral" (empty string). The guides are injected into specialist prompts via `_base.py:161-166`. However, the neutral guide is empty, meaning for the majority of messages (most guests are neutral), no tone guidance is applied. This is likely intentional but means tone calibration only kicks in for emotionally charged messages.

---

## Dimension 7: Emotional Intelligence (6/10)

**Strengths:**
- VADER + casino-specific overrides (sentiment.py:17-26) is a solid approach. Casino-positive phrases like "killing it", "hit the jackpot", "on fire", "on a roll" are correctly overridden to positive sentiment, preventing VADER from misclassifying gambling-context excitement as violence/negativity.
- Frustration detection (sentiment.py:29-33) runs BEFORE VADER with compiled regex patterns, giving deterministic priority to strong frustration signals. Patterns cover both explicit ("frustrated", "ridiculous", "unacceptable") and implicit ("can't find", "waste of", "sick of") frustration.
- Sentiment detection is wired into the router node (nodes.py:192-195) and propagated to specialist agents via state (`guest_sentiment` field). The tone guidance injection in `_base.py:161-166` means the LLM actually receives sentiment-adaptive instructions.
- Responsible gaming escalation (nodes.py:590-600) escalates after 3+ triggers in a session with stronger language encouraging live support. The `responsible_gaming_count` persists across turns via `_keep_max` reducer (state.py:78). This is genuine emotional intelligence for a high-stakes scenario.

**Findings:**

[HIGH] [EQ] No dedicated test file for sentiment detection. `detect_sentiment()` is called on every router invocation (nodes.py:194) but has zero standalone unit tests. No tests verify that casino overrides fire correctly, that VADER thresholds (0.3/-0.3) produce expected classifications, or that the frustration regex patterns work. The function has 4 distinct return paths (frustrated/positive/negative/neutral) and multiple edge cases (empty input, VADER import failure) that are completely untested in isolation. (tests/ directory -- no test_sentiment.py)

[HIGH] [EQ] No dedicated test file for field extraction. `extract_fields()` is called on every router invocation (nodes.py:201) but has zero standalone unit tests. No tests verify name regex patterns, party size boundaries (1-50), visit date parsing, preference extraction, or occasion matching. The `_COMMON_WORDS` exclusion list and `_NAME_PATTERNS` regex are untested for false positives. (tests/ directory -- no test_extraction.py)

[MEDIUM] [EQ] Sarcasm detection is absent. The sentiment module (sentiment.py) has no sarcasm handling. VADER is notoriously poor with sarcasm ("Great, another 30-minute wait for a table" would score positive). No sarcasm-specific patterns or overrides exist. For a casino environment where frustrated guests may use heavy sarcasm, this is a meaningful gap. (src/agent/sentiment.py -- no sarcasm handling)

[MEDIUM] [EQ] Sentiment-driven behavior change is prompt-only, not structural. The agent appends tone guidance text to the system prompt (`_base.py:161-166`) but does not structurally change behavior based on sentiment. For example, a frustrated guest still gets the same number of retrieval results, the same validation strictness, and the same retry budget. A production-grade system might skip validation for frustrated guests (serve faster) or increase retrieval (give more options). (src/agent/agents/_base.py:161-166)

[MEDIUM] [EQ] VADER `SentimentIntensityAnalyzer()` is instantiated fresh on every call (sentiment.py:65-66). While the module uses lazy import (first load ~20ms, subsequent <1ms), creating a new analyzer instance per message is wasteful. Should be a module-level singleton with lazy initialization. (src/agent/sentiment.py:65-66)

[LOW] [EQ] Frustration handling does not offer alternatives or escalate to human. When a guest is detected as frustrated (sentiment returns "frustrated"), the agent receives tone guidance to be empathetic but has no mechanism to proactively offer human escalation or alternative solutions. The off_topic_node has escalation for responsible gaming (3+ triggers), but no equivalent for general frustration escalation.

---

## Dimension 8: Guest Experience (6/10)

**Strengths:**
- Field extraction (extraction.py) covers 5 key guest data points: name, party_size, visit_date, preferences (dietary/cuisine), and occasion. The regex patterns are well-crafted with false-positive prevention via `_COMMON_WORDS` exclusion list and size boundaries (party 1-50, name 2-30 chars).
- Fields accumulate across turns (nodes.py:204-206) -- if a guest says their name in turn 1 and party size in turn 3, both are preserved via `existing.update(extracted)`. This mirrors natural conversation profiling.
- Guest profile system (guest_profile.py) is production-grade with Firestore persistence, CCPA cascade delete, confidence tracking with decay, and bounded in-memory fallback. The `get_agent_context()` function applies confidence decay and filters low-confidence fields before injecting into the LLM context.
- Comp agent has a profile completeness gate (comp_agent.py:90-102) that asks for more information before recommending offers when the profile is incomplete. This is genuinely useful guest experience design.
- Guest context injection in `_base.py:128-143` formats extracted fields (name, party_size, visit_date, preferences, occasion) into the system prompt so specialist agents can personalize their responses.

**Findings:**

[HIGH] [GUEST] Guest name injection is a no-op (third occurrence of this finding). `_inject_guest_name()` at persona.py:122 returns `content` unchanged even when all conditions for injection are met. This means the entire personalization pipeline (extraction -> state -> persona envelope) culminates in... nothing. The guest's name is extracted, stored in state, and then never used in the response. (src/agent/persona.py:122)

[HIGH] [GUEST] `guest_profile_enabled` feature flag defaults to `False` (casino/config.py:125). This means the guest profile context injection in graph.py:274 (`if await is_feature_enabled(..., "guest_profile_enabled")`) is disabled by default. The entire guest profile system (425 LOC in guest_profile.py + data models) is scaffolded infrastructure that is not active in the default configuration. Extracted fields still accumulate in state (nodes.py:201-210) but are not fed through `get_agent_context()`. (src/casino/config.py:125)

[MEDIUM] [GUEST] Proactive suggestions are entirely absent from the agent architecture. No specialist agent proactively suggests things the guest did not ask for. The architecture is reactive: guest asks a question -> retrieval -> response. The whisper planner could theoretically guide proactive suggestions via `conversation_note`, but the specialist agents do not have instructions to act on unsolicited recommendations. A real casino host proactively says "By the way, since you mentioned your anniversary, you should check out the Spa's couples package." (src/agent/agents/*.py -- all reactive)

[MEDIUM] [GUEST] Task completion tracking is absent. There is no mechanism to know when a guest's request has been fulfilled. The agent treats every message as independent (per-turn reset of non-message fields). A real host tracks: "Guest asked about restaurants -> recommended Tuscany -> guest asked for hours -> provided hours -> TASK COMPLETE." The whisper planner's `offer_readiness` field is the closest analogue but measures profile completeness, not request fulfillment. (src/agent/state.py -- no task_status field)

[LOW] [GUEST] Handoff patterns are limited to static contact information. The fallback_node (nodes.py:426-445) and all no-context-fallback messages provide phone/website but no warm handoff mechanism. A production system should integrate with the casino's CRM to create a ticket or notify a live host when escalation is needed.

---

## Dimension 9: Domain Expertise (8/10)

**Strengths:**
- Casino comp knowledge in knowledge-base/casino-operations/comp-system.md is genuinely deep: ADT formula with game-specific parameters, reinvestment rates by tier, comp approval authority matrix, real-time vs bounce-back offers, trip-level vs cumulative theoretical. This is not generic -- it reflects real casino operations knowledge.
- Host workflow knowledge (knowledge-base/casino-operations/host-workflow.md) covers portfolio structure (300-450 players, 3 tiers), daily schedule breakdown, shift differences, prioritization framework, KPIs, and technology stack. A casino host reviewing this would recognize the accuracy.
- Regulatory knowledge (knowledge-base/regulations/state-requirements.md) is comprehensive and current: TCPA one-to-one consent vacating (Jan 2025), NJ SB 3401 (Feb 2026), PA enhanced KYC, CCPA casino applicability. The compliance architecture checklist at the end is production-ready.
- 5-layer deterministic guardrails (guardrails.py) covering prompt injection, responsible gaming, age verification, BSA/AML, and patron privacy with multilingual patterns (English, Spanish, Portuguese, Mandarin). The BSA/AML patterns (guardrails.py:129-159) cover money laundering, structuring, CTR avoidance, chip walking, and multi-buy-in structuring -- these are real casino-specific financial crime patterns, not generic AML.
- Comp agent uses cautious language rules (comp_agent.py:47-53): "based on available information", "you may be eligible", "subject to availability and terms". This is correct for the regulated environment where promising specific comp amounts creates liability.
- Property-specific differentiation: each specialist prompt explicitly names `$property_name` and the system prompt includes property-specific context (prompts.py:88-92). Greeting node derives categories from actual property data file (nodes.py:470-509).

**Findings:**

[MEDIUM] [DOMAIN] Knowledge base has only 5 files across 4 directories. For a production casino host agent, this is thin. Missing: specific restaurant menus/hours, entertainment schedules, hotel room details, spa service menus, gaming floor layout, loyalty program tiers with specific benefits. The RAG pipeline is production-grade but the data feeding it is sparse. (knowledge-base/ -- 5 files)

[MEDIUM] [DOMAIN] Responsible gaming helplines are hardcoded to Connecticut (prompts.py:22-26). The `get_responsible_gaming_helplines()` function (prompts.py:32-38) always returns CT helplines regardless of property. For multi-state deployment, this needs property-specific helpline routing. The architecture supports it (function exists) but the implementation is static. (src/agent/prompts.py:32-38)

[LOW] [DOMAIN] Property-specific advice differentiation is limited to `$property_name` substitution. All specialist prompts assume Mohegan Sun CT context (21+ age, CT self-exclusion). For deployment to a Nevada or NJ casino, the guardrail patterns would need state-specific regulatory adjustments. The knowledge base states-requirements.md documents this gap but the code does not implement state-aware behavior. (src/agent/guardrails.py -- CT-centric)

[LOW] [DOMAIN] The comp agent profile completeness gate threshold is configurable via `COMP_COMPLETENESS_THRESHOLD` (comp_agent.py:93) but defaults to a settings value that is not visible in the code. The threshold logic uses `_PROFILE_FIELDS` as denominator (8 fields), so even 1/8 = 12.5% completeness may suffice if the threshold is low. The relationship between the threshold and real comp eligibility assessment is unclear.

---

## Dimension 10: Evaluation Framework (7/10)

**Strengths:**
- 5 LLM-as-judge metrics (llm_judge.py:30-42): empathy, cultural_sensitivity, conversation_flow, persona_consistency, guest_experience. Each has a dedicated offline scoring function with keyword/regex heuristics that work without API keys in CI.
- 55 conversation scenarios across 10 well-chosen categories (test_conversation_scenarios.py:9-19). Categories cover the full guest journey: dining, hotel, entertainment, profile building, comp eligibility, sentiment shifts, cultural sensitivity, escalation paths, edge cases, greeting-to-deep progressions.
- ConversationEvalReport (llm_judge.py:107-137) provides aggregate metrics across all scenarios, enabling version-over-version comparison. The `phase3-baseline.md` document records the first baseline measurement with per-category breakdowns.
- Test infrastructure is well-designed: YAML scenario files with parametrized pytest (test_conversation_scenarios.py:266-345), fixture-based property data, grounded mock responses built from test data rather than hardcoded strings.
- Deterministic golden dataset (20 cases) in evaluation.py with weighted scoring (Groundedness 0.25, Helpfulness 0.25, Safety 0.35, Persona 0.15). Safety getting the highest weight is appropriate for a regulated environment.
- Phase 3 baseline document (docs/phase3-baseline.md) is honest about limitations: "Offline heuristic scores are intentionally conservative", "Empathy is lowest because keyword matching cannot fully capture emotional attunement." This is good evaluation practice -- know your measurement tool's limits.

**Findings:**

[HIGH] [EVAL] LLM-as-judge is offline-only (hardcoded keyword heuristics). The `EVAL_LLM_ENABLED=true` code path (llm_judge.py:532-534) logs a message and falls back to offline scoring. The actual LLM-based evaluation is "not yet implemented" per the log message. This means all 5 quality metrics are keyword-matching proxies, not actual LLM quality assessments. The baseline scores in phase3-baseline.md are keyword-heuristic scores, not LLM-judge scores. This is honestly documented but limits evaluation fidelity significantly. (src/observability/llm_judge.py:533-534)

[HIGH] [EVAL] No dedicated unit tests for `detect_sentiment()` or `extract_fields()` -- the two core Phase 3 features. The conversation scenario tests (test_conversation_scenarios.py) use fully mocked LLMs, so they validate the mock machinery, not actual sentiment/extraction behavior. The test_phase2_integration.py tests WhisperPlan integration but does not test the sentiment or extraction functions themselves. 128 new tests were added in Phase 3, but the two new production modules have zero direct test coverage. (tests/ directory)

[MEDIUM] [EVAL] Conversation scenario tests (test_conversation_scenarios.py:324-335) verify "at least one expected keyword appears" in the mocked response. But the mock response is constructed FROM the expected keywords (`_build_mock_response()` at line 120-249). This is circular: the test passes because the mock is built to contain the keywords being tested. The test validates the mock construction logic, not the agent's ability to produce relevant responses. Real validation would require testing against actual (or at least more realistic) LLM outputs. (tests/test_conversation_scenarios.py:120-335)

[MEDIUM] [EVAL] Empathy scoring baseline is 0.30 for ANY non-empty response (llm_judge.py:243). This means a completely cold, unhelpful response like "Restaurant is on floor 2. Hours 5-10." scores 0.30 empathy. The cultural sensitivity baseline is even higher at 0.70 (llm_judge.py:307). These high baselines compress the scoring range, making it harder to distinguish genuinely empathetic/sensitive responses from mediocre ones. (src/observability/llm_judge.py:243, 307)

[MEDIUM] [EVAL] No A/B testing or regression detection framework. The evaluation system can measure a snapshot (current baseline) but has no mechanism to detect quality regressions across versions. There is no CI gate that fails if empathy drops below a threshold, no comparison of current scores against the baseline in phase3-baseline.md. (docs/phase3-baseline.md -- baseline exists but no regression mechanism)

[LOW] [EVAL] Guest experience score (llm_judge.py:458-497) uses component weights that sum to 0.85 (empathy 0.25 + cultural 0.15 + flow 0.25 + persona 0.20 = 0.85), then normalizes by dividing by `sum(weights.values())` (0.85) and multiplying by 0.85. This means the component contribution is effectively 0.85/0.85 * 0.85 = 0.85, which is mathematically equivalent to just 0.85 * weighted_average + 0.15 * helpfulness. The intermediate normalization is unnecessary but not wrong. (src/observability/llm_judge.py:466-497)

---

## Agent Quality Subtotal: 42/60

| Dimension | Score | Key Issues |
|-----------|-------|-----------|
| 5. Conversation Quality | 7/10 | Missing unit tests for core functions, broken name injection |
| 6. Persona & Voice | 8/10 | Strong persona differentiation, but name injection no-op |
| 7. Emotional Intelligence | 6/10 | Good VADER+casino approach, but no tests and no sarcasm handling |
| 8. Guest Experience | 6/10 | Profile system is scaffolded but guest_profile_enabled=False by default |
| 9. Domain Expertise | 8/10 | Deep casino knowledge, strong guardrails, thin knowledge base |
| 10. Evaluation Framework | 7/10 | Good infrastructure, circular scenario tests, offline-only judge |

## Critical/High Finding Summary

| # | Severity | Dimension | Finding |
|---|----------|-----------|---------|
| 1 | HIGH | CONV/PERSONA/GUEST | `_inject_guest_name()` is a no-op -- returns content unchanged (persona.py:122) |
| 2 | HIGH | EQ/EVAL | No dedicated unit tests for `detect_sentiment()` (sentiment.py) |
| 3 | HIGH | EQ/EVAL | No dedicated unit tests for `extract_fields()` (extraction.py) |
| 4 | HIGH | GUEST | `guest_profile_enabled` defaults to False -- profile system inactive (config.py:125) |
| 5 | HIGH | EVAL | LLM-as-judge is offline-only, no actual LLM evaluation implemented (llm_judge.py:533) |

## Actionable Fixes (Priority Order)

1. **Fix `_inject_guest_name()` no-op** (persona.py:122): Add actual name prepend logic, e.g., `return f"{guest_name}, {content[0].lower()}{content[1:]}"` or contextual injection.
2. **Add test_sentiment.py**: Unit tests for VADER thresholds, casino overrides, frustration patterns, empty input, fail-silent behavior.
3. **Add test_extraction.py**: Unit tests for each regex pattern, boundary values, false positive exclusions, common words list.
4. **Consider enabling guest_profile_enabled by default** or document why it is disabled (latency concern? Firestore dependency?).
5. **Implement actual LLM-as-judge scoring** or document a timeline. The offline heuristics are a good CI foundation but insufficient for quality measurement.
