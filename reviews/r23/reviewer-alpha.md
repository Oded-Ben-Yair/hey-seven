# R23 Review: EQ + Guest + Persona Quality

Reviewer: reviewer-alpha (hostile)
Date: 2026-02-22
Files reviewed: `_base.py`, `whisper_planner.py`, `prompts.py`, `test_r21_agent_quality.py`

---

## Critical Issues (must fix)

### C-001: Proactive suggestion injected TWICE into system prompt (BUG)

**File**: `src/agent/agents/_base.py` lines 178-181 and 219-229

When `include_whisper=True` (which is TRUE for all 5 specialist agents — host, dining, entertainment, comp, hotel), the proactive suggestion is injected into the system prompt **twice**:

1. **First injection** (line 178-181): `format_whisper_plan()` is called, which includes the suggestion line `"Proactive suggestion (90% confidence): Try Todd English's..."` inside the Whisper Track Guidance section (see `whisper_planner.py` lines 270-276).

2. **Second injection** (line 219-229): A separate block checks `whisper.get("proactive_suggestion")` and injects a whole new `## Proactive Suggestion` section with the same text.

**Impact**: The LLM receives the same suggestion twice with different framing. The first says "Proactive suggestion (90% confidence): ..." as a data point. The second says "## Proactive Suggestion (weave naturally, don't force)" as a directive. This increases the chance the LLM forces the suggestion because it appears to be heavily emphasized. Directly contradicts the "Never push; if the guest doesn't bite, drop it" instruction.

**Fix**: Either (a) remove the suggestion from `format_whisper_plan()` since the dedicated section in `_base.py` has better framing, or (b) remove the dedicated section in `_base.py` and let `format_whisper_plan()` handle it. One path, not two.

### C-002: Proactive suggestion sentiment gate has different logic than escalation gate (BUG)

**File**: `src/agent/agents/_base.py` lines 205 and 220

The frustration escalation check (line 205) requires:
```python
frustrated_count >= 2 AND guest_sentiment in ("frustrated", "negative")
```

The proactive suggestion gate (line 220) blocks on:
```python
guest_sentiment not in ("frustrated", "negative")
```

**Bug**: When `guest_sentiment is None` (the default from `_initial_state()`), the proactive suggestion **IS injected** because `None not in ("frustrated", "negative")` evaluates to `True`. This means if sentiment detection is disabled, broken, or the feature flag is off, proactive suggestions will still be injected with zero emotional context. The system cannot know if the guest is frustrated or not, yet it proceeds to suggest.

**Impact**: A frustrated guest whose sentiment was not detected (e.g., because sentiment_detection_enabled feature flag is False, or VADER fails silently returning "neutral" on a regex-only-frustrated message that doesn't match patterns) will receive a proactive upsell suggestion. This is the exact scenario the feature was designed to prevent.

**Fix**: Change line 220 to explicitly require a non-negative sentiment:
```python
if whisper and state.get("guest_sentiment") in ("positive", "neutral"):
```
This makes the gate explicit: suggestions only when we have positive evidence of non-negative sentiment.

### C-003: "Max 1 per conversation" claim is unenforced (BUG)

**File**: `src/agent/agents/_base.py` line 218 comment, `src/agent/prompts.py` line 222

The comment says "Max 1 per conversation (tracked by whisper planner)" and the WHISPER_PLANNER_PROMPT says "Maximum 1 proactive_suggestion per conversation session". But there is NO tracking mechanism anywhere in the codebase:

1. No `suggestion_offered` boolean in `PropertyQAState` (confirmed: grep returned no matches).
2. No counter in state or config.
3. The whisper planner prompt instructs "Maximum 1" but this is a soft LLM instruction, not a hard gate.
4. `whisper_plan` is reset to `None` every turn via `_initial_state()`, so even if the LLM remembers from conversation history, the planner has no state tracking prior suggestions.

**Impact**: The LLM may suggest something every single turn if the confidence remains high. A guest asking about dinner on turn 1 gets a restaurant suggestion. Same guest asking about show times on turn 2 gets another suggestion. Turn 3, spa suggestion. This is the pushy sales experience the design explicitly wanted to avoid.

**Fix**: Add `suggestion_offered: bool` to `PropertyQAState` (with a `_keep_max`-style reducer or a custom `_keep_true` reducer). Set it to True after injecting a suggestion. Check it before injection:
```python
if whisper and not state.get("suggestion_offered", False) and ...:
```

---

## High Issues (should fix)

### H-001: `_count_consecutive_frustrated` skips over AIMessages, which inflates the frustrated count

**File**: `src/agent/agents/_base.py` lines 47-70

The function iterates messages in reverse and only evaluates `HumanMessage` instances. It skips `AIMessage` entirely. Consider this conversation:

```
HumanMessage("This is terrible")        # frustrated
AIMessage("I sincerely apologize...")    # skipped
AIMessage("Let me look into that...")    # skipped (e.g., retry response)
HumanMessage("I still can't find it")   # frustrated
```

The function returns 2 (consecutive frustrated), which triggers escalation. But if the agent sent two good apology messages between the frustrated messages, the guest may have calmed down. The function treats all non-HumanMessages as invisible.

More critically, if the validation loop retries (adding an extra AIMessage), or if the graph produces multiple AIMessages per turn, the consecutive count is unaffected — which is probably correct. But the docstring says "count of consecutive frustrated messages before the first positive/neutral one" — this implies AIMessages should break the chain, but they don't.

**Fix**: Either (a) update the docstring to clarify AIMessages are intentionally skipped, or (b) decide whether consecutive truly means "no positive HumanMessage in between" vs "adjacent HumanMessages". The current behavior is reasonable but the intent is ambiguous.

### H-002: Persona drift prevention uses message count instead of human turn count

**File**: `src/agent/agents/_base.py` lines 237-243

The threshold (`_PERSONA_REINJECT_THRESHOLD = 10`) is compared against `len(history)` where `history` includes both `HumanMessage` and `AIMessage` objects. The comment says "~5 human turns" — this is correct only if every human turn produces exactly 1 AI response. But:

1. **Validation retries**: A retry adds an extra AIMessage, so the count grows faster.
2. **Whisper planner failures**: No extra messages, but other paths might.
3. **Multiple system messages**: These are filtered out by the `isinstance(m, (HumanMessage, AIMessage))` check, so this is fine.

The real issue: a 3-turn conversation with 2 retries produces `len(history) = 8` (3 human + 3 AI + 2 retry AI = 8), which is close to triggering at 10. A 4-turn conversation with retries could trigger the reminder very early.

**Impact**: Persona reminder may be injected earlier than intended, wasting tokens and adding noise. Not a bug, but a calibration issue.

**Fix**: Count only `HumanMessage` instances: `if sum(1 for m in history if isinstance(m, HumanMessage)) > _PERSONA_REINJECT_THRESHOLD // 2:`

### H-003: Persona drift reminder is generic, not property-specific

**File**: `src/agent/agents/_base.py` lines 238-243

The reminder says: "You are Seven, the AI concierge for {property_name}." But the persona name is hardcoded as "Seven". The actual persona name comes from `BrandingConfig.persona_name` (injected via `get_persona_style()` on line 187-189). If a client configures a different persona name (e.g., "Lucky" for a different casino), the drift prevention reminder will contradict the persona style section.

**Impact**: For multi-property deployments with different persona names, the LLM receives conflicting identity signals. The main system prompt says "You are Lucky" via persona style, but the reminder says "You are Seven."

**Fix**: Read `persona_name` from the branding config:
```python
try:
    from src.casino.config import DEFAULT_CONFIG
    persona_name = DEFAULT_CONFIG.get("branding", {}).get("persona_name", "Seven")
except Exception:
    persona_name = "Seven"
```

### H-004: Escalation and suggestion run sentiment detection TWICE on the same messages

**File**: `src/agent/agents/_base.py` lines 203-205 and 194-198

Line 203 builds `history` and calls `_count_consecutive_frustrated(history)` which runs `detect_sentiment()` on every HumanMessage. But `guest_sentiment` in the state (line 194) was already computed by `nodes.py` line 193-195 using `detect_sentiment()` on the CURRENT message only.

So:
1. The **current** message's sentiment is computed once in the router node and stored in state.
2. The **history** sentiments are recomputed from scratch in `_count_consecutive_frustrated()` every time `execute_specialist` runs.
3. VADER is fast (sub-1ms), but this is redundant work and — more importantly — if the sentiment detection logic ever changes (e.g., new sarcasm patterns), the historical recomputation will retroactively reclassify old messages differently than they were classified in real-time.

**Impact**: Minor performance issue, but a real consistency risk. A message classified as "neutral" at turn 3 could be reclassified as "frustrated" at turn 7 if sarcasm patterns are updated between deploys.

**Fix**: Store per-message sentiment in state (e.g., as metadata on the HumanMessage) rather than recomputing from content. Or accept the trade-off and document it.

### H-005: Test file uses `pytest.raises(Exception)` instead of `pytest.raises(ValidationError)`

**File**: `tests/test_r21_agent_quality.py` lines 524 and 535

```python
with pytest.raises(Exception):  # ValidationError
```

This catches ANY exception, not specifically Pydantic's `ValidationError`. If the code raises a `TypeError` or `RuntimeError` for a completely different reason, the test still passes. This is a weak assertion.

**Fix**: Import `pydantic.ValidationError` and use `pytest.raises(ValidationError)`.

### H-006: `format_whisper_plan` does not gate on `guest_sentiment`

**File**: `src/agent/whisper_planner.py` lines 270-276

`format_whisper_plan()` includes the proactive suggestion in its output based solely on `suggestion_confidence >= 0.8`. It has no access to `guest_sentiment` and cannot filter on it. This means the Whisper Track Guidance section will always include the suggestion text if confidence is high enough, even when the guest is frustrated.

Combined with C-001 (double injection), this means: even if the dedicated proactive suggestion block in `_base.py` correctly gates on sentiment, the `format_whisper_plan()` path does NOT gate, so the suggestion still leaks through in the Whisper Track Guidance section.

**Impact**: Frustrated guest sees a suggestion via the Whisper guidance path even when the dedicated suggestion path correctly blocks it.

**Fix**: Either (a) pass `guest_sentiment` to `format_whisper_plan()` and gate there too, or (b) remove the suggestion from `format_whisper_plan()` entirely and only use the dedicated section (recommended — aligns with C-001 fix).

---

## Medium Issues (nice to fix)

### M-001: WHISPER_PLANNER_PROMPT does not see prior suggestions from conversation history

**File**: `src/agent/prompts.py` lines 183-222

The prompt includes `$conversation_history` (last 20 messages) and `$guest_profile`, but does not include any record of previously offered suggestions. The instruction "Maximum 1 proactive_suggestion per conversation session" depends entirely on the LLM inferring from conversation history whether a suggestion was already made. If the suggestion was woven subtly (as instructed), the LLM may not recognize it as a "suggestion" and generate another one.

**Fix**: Add a `$prior_suggestions` variable to the prompt template and populate it from state.

### M-002: `_count_consecutive_frustrated` does not handle multimodal message content

**File**: `src/agent/agents/_base.py` line 64

```python
content = msg.content if isinstance(msg.content, str) else str(msg.content)
```

When `msg.content` is a list (multimodal LangChain format, e.g., `[{"type": "text", "text": "..."}]`), `str()` produces `"[{'type': 'text', 'text': '...'}]"` — a dict-like string. VADER will return near-zero scores for this malformed input, always classifying as neutral. This breaks frustration detection for multimodal inputs.

**Fix**: Extract text parts from multimodal content:
```python
if isinstance(msg.content, list):
    content = " ".join(p.get("text", "") for p in msg.content if isinstance(p, dict) and p.get("type") == "text")
elif isinstance(msg.content, str):
    content = msg.content
else:
    content = str(msg.content)
```

### M-003: Test `test_sarcasm_detected_as_frustrated` is fragile

**File**: `tests/test_r21_agent_quality.py` lines 140-147

The test asserts `_count_consecutive_frustrated(messages) == 2`, meaning BOTH "Oh great, another thing that doesn't work" AND "Thanks for nothing" must be classified as frustrated. This depends on the exact sarcasm patterns in `sentiment.py`. If someone adjusts the sarcasm regex (e.g., tightens "Oh great, another"), this test fails for a reason unrelated to what it's testing (frustration counting logic).

**Fix**: Use messages known to trigger the simpler `_FRUSTRATED_PATTERNS` (e.g., "This is ridiculous", "I'm fed up") instead of relying on sarcasm detection.

### M-004: No test for suggestion injection when `guest_sentiment is None`

**File**: `tests/test_r21_agent_quality.py`

Tests cover sentiment = "frustrated" (blocked) and sentiment = "positive" (allowed). Missing test for sentiment = `None` (the default). Per C-002, `None` allows injection, which may be unintended.

### M-005: No test for the double-injection scenario (C-001)

**File**: `tests/test_r21_agent_quality.py`

No test verifies that when `include_whisper=True` AND the whisper plan has a high-confidence suggestion, the suggestion text does not appear twice in the system prompt sent to the LLM. This is the most impactful bug in the review.

### M-006: No test for persona name mismatch across branding config

**File**: `tests/test_r21_agent_quality.py`

The persona drift prevention test (lines 431-470) verifies "PERSONA REMINDER" and "Mohegan Sun" appear, but does not test that the persona name matches the branding config. If someone sets `persona_name="Lucky"` in branding, the test would still pass even though the reminder says "Seven" (contradicting the persona style section).

### M-007: `_calculate_completeness` treats falsy values as unfilled

**File**: `src/agent/whisper_planner.py` line 302

```python
filled = sum(1 for field in _PROFILE_FIELDS if profile.get(field))
```

If `party_size` is `0` (e.g., solo guest explicitly stated), it counts as unfilled because `0` is falsy. Same for an empty-string name after PII redaction. `profile.get(field) is not None` would be more accurate.

### M-008: Escalation guidance hardcodes the escalation language

**File**: `src/agent/agents/_base.py` lines 206-214

The escalation offer template includes a specific phrase: "would you like me to connect you with one of our dedicated hosts who can assist you personally?" This is injected as a system prompt example. If different properties want different escalation language, there's no configuration point. Currently fine for single-property MVP, but becomes tech debt at multi-property.

---

## Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **EQ (emotional intelligence)** | 6/10 | Frustration detection is solid (VADER + regex + sarcasm), but the double-gate bug (C-002: None sentiment allows suggestions to frustrated guests), the double-injection (C-001), and the uncapped suggestion count (C-003) undermine the emotional intelligence. The escalation logic itself is well-designed with the HEART framework reference. |
| **Guest (guest experience)** | 5/10 | Proactive suggestions are a great concept, but three critical bugs (double injection, no max-1 enforcement, sentiment gate allows None) mean the production behavior will be pushy and repetitive — the exact opposite of the stated intent. The confidence threshold (0.8) and the "weave naturally" framing are good design. |
| **Persona (drift prevention)** | 7/10 | The threshold-based re-injection is a simple, effective pattern. The hardcoded "Seven" name is a real issue for multi-property but acceptable for MVP. Message-count-based threshold is slightly miscalibrated due to retry inflation, but functionally adequate. Coverage is good — all 5 specialist agents inherit the behavior via `execute_specialist()`. |

**Overall**: 6/10 — Three critical bugs (C-001 through C-003) all stem from the same root cause: the proactive suggestion feature was implemented in two places without coordination. The frustration escalation and persona drift features are well-implemented with minor calibration issues.

---

## Missing Tests

1. **Double injection test**: Verify suggestion appears exactly ONCE in system prompt when `include_whisper=True` and suggestion confidence is high (covers C-001).
2. **None sentiment + suggestion test**: Verify behavior when `guest_sentiment is None` and suggestion confidence is high (covers C-002).
3. **Max-1 suggestion test**: Verify that across a multi-turn conversation, the suggestion is offered at most once (covers C-003 — though this requires state tracking to be implemented first).
4. **Escalation + suggestion mutual exclusivity test**: Verify that when escalation guidance is injected, proactive suggestions are NOT injected (edge case: frustrated_count >= 2 but guest_sentiment is something weird).
5. **Multimodal content test**: Verify `_count_consecutive_frustrated` handles `msg.content` as a list (covers M-002).
6. **Persona name from branding test**: Verify the PERSONA REMINDER uses the configured persona name, not hardcoded "Seven" (covers H-003).
7. **Empty string suggestion test**: Verify that `proactive_suggestion=""` (empty string, not None) is treated as no suggestion. Currently `if suggestion and conf >= 0.8` would correctly skip empty string, but no test confirms this.
8. **format_whisper_plan with frustrated sentiment**: Verify that `format_whisper_plan()` behavior when it should NOT include the suggestion (covers H-006 — requires either passing sentiment or removing suggestion from this function).
9. **Boundary: exactly 10 messages in history**: Test persona reminder at exactly the threshold boundary (`len(history) == 10` should NOT trigger; `== 11` should).
10. **Whisper planner structured output with new fields**: Integration test that the LLM's structured output correctly populates `proactive_suggestion` and `suggestion_confidence` (currently only unit tests for Pydantic validation, no LLM integration test).
