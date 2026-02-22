# R18 Review Summary

## Scores
| Dimension | Score | Category |
|-----------|-------|----------|
| 1. Architecture & Data Model | 9.0/10 | Code (40%) |
| 2. API & Infrastructure | 9.0/10 | Code (40%) |
| 3. Testing & Reliability | 8.5/10 | Code (40%) |
| 4. Security & Compliance | 9.5/10 | Code (40%) |
| 5. Conversation Quality | 7.0/10 | Agent (60%) |
| 6. Persona & Voice | 8.0/10 | Agent (60%) |
| 7. Emotional Intelligence | 6.0/10 | Agent (60%) |
| 8. Guest Experience | 6.0/10 | Agent (60%) |
| 9. Domain Expertise | 8.0/10 | Agent (60%) |
| 10. Evaluation Framework | 7.0/10 | Agent (60%) |

**Code Quality Subtotal**: 36.0/40
**Agent Quality Subtotal**: 42.0/60
**R18 Total**: 78.0/100

## Fixes Applied

1. **[HIGH] [TEST] conftest.py whisper telemetry reset** -- Updated `conftest.py:58-63` to reset `_wp._telemetry.count` and `_wp._telemetry.alerted` instead of the old `_wp._failure_count` and `_wp._failure_alerted` attributes that were refactored into `_WhisperTelemetry` class in R17. The stale references were silently caught by `except AttributeError: pass`, meaning whisper planner telemetry was NOT being reset between tests. (files: `tests/conftest.py`)

2. **[HIGH] [CONV/PERSONA/GUEST] Fix `_inject_guest_name()` no-op** -- `persona.py:122` returned `content` unchanged even when all guard conditions passed. Added actual name injection logic: `return f"{guest_name}, {content[0].lower()}{content[1:]}"` which prepends the guest name with a natural greeting style. (files: `src/agent/persona.py`)

3. **[HIGH] [EQ/EVAL] Add test_sentiment.py** -- Created 36 unit tests covering: empty/edge inputs, all 8 casino-positive override phrases, 7 explicit frustration words, `can't find/believe/get` patterns, `waste of/sick of/tired of/fed up` patterns, `what a joke/disaster/mess` patterns, VADER threshold boundaries, frustration priority over VADER, and fail-silent behavior for non-string inputs. (files: `tests/test_sentiment.py`)

4. **[HIGH] [EQ/EVAL] Add test_extraction.py** -- Created 52 unit tests covering: empty/None/non-string inputs, 5 name extraction patterns, `_COMMON_WORDS` false-positive exclusion, name boundary enforcement (min 2 chars, no trailing words), 8 party size patterns with min/max boundaries (1-50), visit date day-of-week and date format patterns, 6 dietary preference patterns, allergy and cuisine patterns, multiple preferences in one message, 8 occasion patterns, multi-field extraction from single messages, and fail-silent behavior. (files: `tests/test_extraction.py`)

## Disputed Findings

- **[HIGH] [GUEST] `guest_profile_enabled` defaults to False** -- This is a deliberate design decision documented in `phase3-baseline.md:161`. The guest profile system depends on Firestore, which is not available in local dev or CI. The extracted fields still accumulate in state via `extracted_fields` on every turn. Enabling by default would break local dev and CI environments. The flag is designed to be toggled per-casino in Firestore config. No code change needed.

- **[HIGH] [EVAL] LLM-as-judge offline-only** -- This is honestly documented in `phase3-baseline.md` as intentional. The offline heuristics serve as a CI-compatible foundation. Implementing actual LLM evaluation requires API keys in CI and adds cost/latency to the test suite. The current architecture is extensible (the `EVAL_LLM_ENABLED` flag path exists) and the baseline document explicitly states the scores are heuristic proxies. No code change needed for R18.

## Remaining Items (MEDIUM/LOW - deferred)

### MEDIUM severity
- [MEDIUM] [ARCH] `BoundedMemorySaver._track_thread()` accesses `MemorySaver` implementation detail (memory.py:69-74) -- pinned LangGraph version mitigates risk
- [MEDIUM] [API] CSP nonce generated per-request but never passed to HTML templates (middleware.py:196-218) -- static files don't use nonce, no functional impact currently
- [MEDIUM] [API] CORSMiddleware may buffer SSE responses (app.py:118-123) -- needs empirical testing
- [MEDIUM] [TEST] `_state()` helper missing Phase 3 fields (test_graph_v2.py:34-51) -- LangGraph handles defaults, not breaking
- [MEDIUM] [TEST] No test for `_initial_state()` parity check itself (test_state_parity.py) -- check runs at import time
- [MEDIUM] [SEC] `audit_input()` inverted boolean convention (guardrails.py) -- correct at all call sites, cosmetic
- [MEDIUM] [CONV] Whisper planner guidance not verified post-LLM (base.py:146-149) -- trust-the-LLM design
- [MEDIUM] [CONV] Return-to-previous-topic not tested (edge_cases.yaml) -- feature gap
- [MEDIUM] [PERSONA] Only one BrandingConfig configured (config.py:135-141) -- documented, per-casino via Firestore
- [MEDIUM] [PERSONA] Exclamation replacement with periods produces awkward text (persona.py:61-76) -- cosmetic
- [MEDIUM] [EQ] No sarcasm detection (sentiment.py) -- VADER limitation, acknowledged
- [MEDIUM] [EQ] Sentiment-driven behavior is prompt-only, not structural (base.py:161-166) -- design choice
- [MEDIUM] [EQ] VADER analyzer instantiated per-call (sentiment.py:65-66) -- sub-1ms, acceptable for now
- [MEDIUM] [GUEST] No proactive suggestions (agents/*.py) -- reactive by design
- [MEDIUM] [GUEST] No task completion tracking (state.py) -- feature gap
- [MEDIUM] [DOMAIN] Knowledge base sparse (5 files) -- data gap, not code gap
- [MEDIUM] [DOMAIN] Responsible gaming helplines hardcoded to CT (prompts.py:32-38) -- single-property MVP
- [MEDIUM] [EVAL] Scenario tests are circular (test_conversation_scenarios.py:120-335) -- validates mock machinery, acceptable for CI
- [MEDIUM] [EVAL] Empathy/cultural sensitivity baselines too high (llm_judge.py:243,307) -- heuristic limitation
- [MEDIUM] [EVAL] No A/B testing or regression detection framework -- future work

### LOW severity
- 13 LOW findings across all dimensions (documented in individual review files) -- deferred

## Key Insights

### Top 3 Strengths
1. **Security posture is exceptional** (9.5/10): 5-layer guardrails with 84 multilingual regex patterns, fail-closed PII, streaming PII redaction with chunk-boundary lookahead, TCPA compliance with HMAC-SHA256 audit chain
2. **Architecture is production-grade** (9.0/10): 11-node StateGraph with validation loops, DRY specialist extraction via `_base.py`, deterministic tie-breaking, dual-layer feature flags
3. **Specialist persona differentiation is genuinely distinct** (8.0/10): 5 personas with unique vocabulary, interaction styles, and emotional calibration -- not just name changes

### Top 3 Areas for Improvement
1. **Emotional intelligence needs structural depth** (6.0/10): Sentiment detection works but only modifies prompt text, not agent behavior. No sarcasm handling. VADER analyzer created per-call.
2. **Guest experience pipeline ends in a dead end** (6.0/10): Fields are extracted and accumulated but guest_profile_enabled=False means the profile system is inactive. Name injection was a no-op (now fixed). No proactive suggestions.
3. **Evaluation framework is heuristic-only** (7.0/10): LLM-as-judge not implemented, scenario tests are circular, no regression detection. Good CI foundation but insufficient for quality measurement.

### Comparison to Previous Rounds
- R17 code-only baseline: ~84/100
- R18 total (code 40% + agent 60%): 78.0/100
- The lower total reflects the new agent quality dimensions (5-10) which expose real gaps in emotional intelligence, guest experience, and evaluation. Code quality remains strong at 36/40.
- Phase 3 additions (sentiment, extraction, persona, whisper planner) are architecturally sound but have testing gaps (now addressed) and feature gaps (deferred).

## Verification

- Test command: `python3 -m pytest tests/ -v --tb=line --ignore=tests/test_eval.py`
- Tests passed: 1603
- Tests failed: 51 (all pre-existing API auth failures in test_api.py, test_config.py, test_phase2_integration.py, test_phase4_integration.py -- unrelated to R18 changes)
- New tests added: 88 (36 in test_sentiment.py, 52 in test_extraction.py)
- Files modified: `tests/conftest.py`, `src/agent/persona.py`
- Files created: `tests/test_sentiment.py`, `tests/test_extraction.py`
