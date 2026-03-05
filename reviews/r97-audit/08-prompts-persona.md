# Component 8: Prompts + Persona

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/agent/prompts.py` | 1052 | All prompt templates (CONCIERGE_SYSTEM_PROMPT EN/ES, ROUTER_PROMPT, VALIDATION_PROMPT, WHISPER_PLANNER_PROMPT), responsible gaming helplines (per-casino lookup), greeting/off-topic templates, few-shot examples, sentiment tone guides, booking context |
| `src/agent/persona.py` | 333 | Persona envelope node: PII redaction (fail-closed), branding enforcement (exclamation limit, emoji), guest name injection, performative opener stripping, SMS truncation, BSA/AML threshold redaction |
| **Total** | **1385** | |

## Wiring Verification

Both files are heavily wired across the production graph:

**prompts.py:**
- `src/agent/agents/host_agent.py:9` — `from src.agent.prompts import CONCIERGE_SYSTEM_PROMPT`
- `src/agent/agents/_base.py:23` — imports multiple prompt templates (CONCIERGE_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, SENTIMENT_TONE_GUIDES, etc.)
- `src/agent/agents/_base.py:816` — dynamic import of booking context templates
- `src/agent/whisper_planner.py:26` — `from src.agent.prompts import WHISPER_PLANNER_PROMPT`
- `src/agent/nodes.py:1150` — `from src.agent.prompts import GREETING_TEMPLATE_ES` (Spanish greeting)
- `src/agent/nodes.py:1308` — `from src.agent.prompts import get_responsible_gaming_helplines_es`
- `src/agent/nodes.py:1595` — `from src.agent.prompts import OFF_TOPIC_RESPONSE_ES`

**persona.py:**
- `src/agent/graph.py:193` — `graph.add_node(NODE_PERSONA, persona_envelope_node)` (wired as graph node)

**Verdict: Both files are critical production components. prompts.py is the central prompt repository used by 5+ modules. persona.py is a graph node in the processing chain between validate and respond.**

## Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `tests/test_prompts.py` | 7 | Template variable substitution, helpline content, router prompt structure |
| `tests/test_persona.py` | 18 | PII redaction, branding enforcement, guest name injection, SMS truncation, performative stripping, persona_envelope_node integration |
| `tests/test_prompt_parameterization.py` | 12 | Template variables across all prompt templates, placeholder coverage |
| `tests/test_tone_calibration.py` | 16 | _strip_performative_openers, exclamation reduction, opener chaining |
| `tests/test_few_shot_examples.py` | 9 | Few-shot example structure, specialist coverage, format consistency |
| `tests/test_intent_validation.py` | 7 | VALIDATION_PROMPT template integrity, variable presence |
| **Total** | **69** | |

## Live vs Mock Assessment

**All deterministic — no LLM calls needed:**

- `test_prompts.py`: Template substitution tests — verifies `$property_name`, `$current_time` etc. are correctly handled by `string.Template.safe_substitute()`. Pure string tests.
- `test_persona.py`: PII redaction, branding, name injection are all regex/string operations. `persona_envelope_node` tests construct state dicts and verify output transformations.
- `test_tone_calibration.py`: Tests `_strip_performative_openers` against known patterns. Pure regex.
- `test_few_shot_examples.py`: Validates the structure of `FEW_SHOT_EXAMPLES` dict (keys, format, specialist coverage).
- `test_intent_validation.py`: Template variable presence checks.

**Assessment: Appropriate — prompts are templates (string operations) and persona is a post-processing node (regex/string manipulation). No LLM calls needed.**

## Known Gaps

1. **prompts.py at 1052 LOC**: Large file combining 8+ prompt templates, helpline lookups, greeting templates, few-shot examples, sentiment tone guides, and booking context. While organized with section headers, it could benefit from splitting into: (a) system prompts, (b) routing/validation prompts, (c) helpline/compliance data, (d) few-shot examples. Currently navigable but approaching maintenance burden.

2. **H6 (Rapport Depth: 4.0)**: The system prompt has "Grounded Warmth" instructions and 4 calibration examples, but no **micro-pattern retrieval** for rapport building. The prompt tells the LLM to "be warm" but doesn't provide specific rapport techniques for different guest types (first-timers, VIPs, families, couples, grieving). The R96 strategy identifies this as needing a Rapport Ladder tool.

3. **Few-shot examples (27)**: `FEW_SHOT_EXAMPLES` provides 5-7 examples per specialist (dining, hotel, comp, entertainment, host). These are the primary behavioral calibration mechanism. However, they are injected only when `few_shot_examples_enabled` feature flag is True. No A/B testing framework exists to measure their impact on behavioral scores.

4. **No prompt versioning**: Prompts are directly in source code with no version tracking beyond git. For a product where prompt quality directly determines behavioral scores, a prompt management system (versioned prompts, A/B testing, rollback) would be valuable for Phase 3 optimization.

5. **Persona node does NOT strip Gemini list format artifacts**: Gemini sometimes produces markdown bullet points or numbered lists. The persona node strips performative openers and emoji but doesn't normalize markdown formatting for SMS/chat contexts. Minor issue for web chat, bigger issue for SMS channel.

6. **BSA/AML threshold redaction (R86)**: `persona.py:81` redacts `$10,000` as `[regulatory threshold]`. This is a single regex that catches `$10,000` and `$10.000` but may miss `$10000` (no separator), `10,000 dollars`, or `ten thousand dollars`. Narrow but functional.

7. **Spanish prompt parity**: `CONCIERGE_SYSTEM_PROMPT_ES` is a full translation of the English prompt. Changes to the English prompt must be manually propagated to the Spanish version. No automated parity check exists.

8. **Validation prompt grounding vs proactive suggestions**: The VALIDATION_PROMPT allows "category-level suggestions" without specific facts (R76 fix). The boundary between allowed ("After dinner, you might enjoy a show") and disallowed ("The Wolf Den has a show tonight at 8 PM" without RAG context) is defined in prose, not deterministic rules. Relies on the validator LLM's interpretation.

## Confidence: 78%

prompts.py is the behavioral heart of the agent — every word matters for scores. The current prompts are well-crafted with anti-patterns, emotional intelligence, conversation adaptation, and guest intelligence protocol sections. persona.py is a clean post-processing node with the correct processing order (PII -> branding -> name -> truncation). The main gaps are structural (large file, no versioning, no A/B testing) and behavioral (H6 rapport depth needs micro-pattern retrieval, not better prompts).

## Verdict: needs-fixes

The code works correctly but has structural issues:
1. prompts.py should be split for maintainability (1052 LOC)
2. Spanish prompt parity needs automated checking
3. H6 improvement requires a Rapport Ladder tool (R96 Phase 2), not prompt changes
4. Few-shot example impact measurement needs A/B testing infrastructure

No critical bugs. The "fixes" are structural improvements and tooling, not correctness issues.
