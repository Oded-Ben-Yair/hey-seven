# Component 7: Incentives + Crisis + Sentiment

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/agent/incentives.py` | 556 | Per-casino incentive rules engine (5 casinos + defaults), trigger conditions, auto-approve thresholds, framing templates, system prompt section builder |
| `src/agent/crisis.py` | 333 | Graduated 4-level crisis detection (none/concern/urgent/immediate), English + Spanish patterns, crisis response templates with verified 988/741741 resources |
| `src/agent/sentiment.py` | 349 | VADER-based sentiment (positive/negative/neutral/frustrated), casino-domain overrides, sarcasm detection (pattern + context-contrast), LLM augmentation for ambiguous band |
| `src/agent/slang.py` | 208 | Gambling slang normalization for RAG search (75 slang terms, 53 drunk-typing corrections), multi-word regex + single-word dict lookup |
| **Total** | **1446** | |

## Wiring Verification

All 4 files are wired to production code:

**incentives.py:**
- `src/agent/agents/_base.py:1152` — `from src.agent.incentives import get_incentive_prompt_section` (specialist agents inject incentive offers into system prompt)

**crisis.py:**
- `src/agent/nodes.py:1542` — `from src.agent.crisis import get_crisis_followup_es, get_crisis_response_es` (off_topic_node Spanish crisis path)
- Note: `detect_crisis_level` is imported in compliance_gate.py (verified by grep for the function name)

**sentiment.py:**
- `src/agent/agents/_base.py:31` — `from src.agent.sentiment import detect_sentiment, detect_sarcasm_context` (specialist agents)
- `src/agent/nodes.py:366` — `from src.agent.sentiment import detect_sentiment_augmented` (router node)

**slang.py:**
- `src/agent/nodes.py:36` — `from .slang import normalize_for_search` (module-level import)
- `src/agent/nodes.py:446` — `query = normalize_for_search(query)` (called in retrieve_node)

**Verdict: All 4 files are wired to production paths. crisis.py is wired to compliance_gate AND off_topic_node. incentives.py is wired through specialist _base.py. sentiment.py is wired in both router (augmented) and specialist agents (base + sarcasm). slang.py is wired in retrieve_node.**

## Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `tests/test_incentives.py` | 43 | IncentiveEngine: trigger conditions, per-casino rules, format_incentive_offer, auto-approve threshold, completeness thresholds, unknown casino defaults |
| `tests/test_crisis_detection.py` | 72 | All 3 crisis levels (immediate/urgent/concern), English + Spanish patterns, false positive avoidance, edge cases |
| `tests/test_semantic_sarcasm.py` | 22 | Context-contrast sarcasm detection, positive signal words, conversation history dependency |
| `tests/test_slang_normalization.py` | 34 | Multi-word slang, single-word slang, drunk-typing, contraction handling (YOLO'd), no-op for clean text |
| `tests/test_sentiment.py` | 34 | VADER integration, casino-domain overrides, frustrated patterns, sarcasm patterns, neutral band |
| **Total** | **205** | |

## Live vs Mock Assessment

**All deterministic — no LLM calls needed:**

- `test_incentives.py`: Pure business logic (rule evaluation, template formatting). Zero mocks, zero API calls. Tests are clean assertions against deterministic IncentiveEngine.
- `test_crisis_detection.py`: Pure regex pattern matching. Tests feed strings through `detect_crisis_level()` and assert levels. No mocks needed.
- `test_semantic_sarcasm.py`: VADER + context list comparison. No LLM calls.
- `test_slang_normalization.py`: String replacement tests. No external dependencies.
- `test_sentiment.py`: VADER library (local, no API). Casino overrides and frustrated patterns are all regex-based.

**Assessment: Perfect fit — all 4 modules are deterministic (regex, VADER, business rules). No mocking needed or used. LLM augmentation paths (`detect_sentiment_augmented`, LLM-augmented extraction) are opt-in features behind feature flags and tested separately.**

## Known Gaps

1. **H9 (Comp Strategy: 1.9)**: The incentive engine has per-casino rules (birthday=$25 dining, 75% completeness=$10 free play, etc.) but lacks **deterministic comp policy encoding**. It can offer pre-defined incentives but cannot calculate comps based on player worth, ADT, or trip history. The R96 strategy correctly identifies this: "H9 is missing business logic, not model capability. Needs CompStrategy tool."

2. **Incentive trigger thresholds may be too conservative for 3-turn conversations**: `profile_completeness_50` triggers at 25% (R88 fix lowered from 50%). Birthday/anniversary triggers require explicit mention. In behavioral evaluation, many scenarios never trigger any incentive because guests don't mention occasions or reach 25% completeness in 3 turns.

3. **Crisis detection has no formal test against known NCPG protocols**: Patterns were designed based on R72 domain research, but there's no validation against official NCPG intervention guideline word lists or state-specific terminology databases. The 988 Lifeline numbers are verified (hardcoded correctly).

4. **Sarcasm detection false positive rate unknown**: `detect_sarcasm_context()` uses heuristics (positive words + negative history). No quantitative evaluation of precision/recall on real casino guest data. The 17 sarcasm regex patterns may over-fire on sincere positive messages in frustrated contexts.

5. **Slang normalization dictionary is static**: 75 gambling slang terms + 53 drunk-typing corrections are hardcoded. No mechanism to update the dictionary from production data. New slang (especially crypto/modern gambling) requires code changes.

6. **`detect_sentiment_augmented` LLM fallback is opt-in (flag=False)**: The LLM sentiment path exists but is disabled by default via `sentiment_llm_augmented` feature flag. If enabled, it adds an LLM call for every "neutral" VADER result with 10+ words. Cost impact unclear.

7. **H10 (Lifetime Value: 3.5)**: No return-visit seeding logic exists in any of these modules. Incentive engine handles one-time offers but has no concept of guest lifetime value, visit frequency-based offers, or "come back next month" framing.

## Confidence: 85%

All four modules are clean, well-tested deterministic code. Incentives engine is properly per-casino with immutable `MappingProxyType` rules and `string.Template.safe_substitute()`. Crisis detection covers English + Spanish with graduated levels. Sentiment detection combines VADER + casino overrides + sarcasm heuristics. Slang normalization is search-only (never corrupts stored text). The gaps are business logic gaps (comp strategy, LTV), not code quality issues.

## Verdict: needs-new-tool

Code is production-ready. H9 (1.9) requires a CompStrategy tool with deterministic comp policies based on player worth/ADT. H10 (3.5) requires an LTV Nudge Engine for return-visit seeding. Both are new LangGraph tools per the R96 Phase 2 strategy. The existing incentive engine is the correct foundation to build on.
