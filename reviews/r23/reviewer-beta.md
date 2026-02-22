# R23 Review: Eval + Testing Quality

Reviewer: reviewer-beta (hostile code review)
Date: 2026-02-22
Files reviewed:
- `src/observability/llm_judge.py`
- `tests/test_r22_llm_judge.py`
- `src/observability/evaluation.py`

---

## Critical Issues (must fix)

### C1. NaN silently passes regression detection (llm_judge.py:606)

**Bug**: `detect_regression()` compares `drop > threshold` where `drop = baseline_score - current_score`. When `current_score` is `float('nan')` (e.g., from a division error in averaging), `nan > 0.05` evaluates to `False` in Python. A NaN score will NEVER trigger a regression, silently passing the CI gate.

```python
# Current: NaN comparison silently returns False
drop = 0.4 - float('nan')  # nan
drop > 0.05  # False -- regression goes undetected
```

**Fix**: Add NaN check at the start of the loop:
```python
if math.isnan(current_score) or math.isnan(baseline_score):
    regressions.append(f"{metric}: NaN detected (current={current_score}, baseline={baseline_score})")
    continue
```

**Impact**: A broken scoring function returning NaN would pass CI undetected. This is a data integrity hole in the quality gate.

### C2. Empathy baseline margin is below detection threshold (llm_judge.py:619-620)

**Bug**: `QUALITY_BASELINE["empathy"] = 0.40`. Actual golden conversation avg_empathy = 0.4333. The margin is only **0.0333**, which is BELOW the default regression threshold of 0.05. This means:

1. Empathy can drop from 0.4333 to 0.3501 (a 19% relative decline) without triggering any regression.
2. Two golden conversations score empathy at 0.300 (persona-drift-01 and proactive-01). If a code change causes one more to drop to 0.300, average drops to ~0.383 -- still no regression detected (0.40 - 0.383 = 0.017 < 0.05).

**Fix**: Either tighten threshold to 0.02 for empathy or raise baseline to 0.42 (with a 0.013 margin at default threshold). Better: per-metric thresholds since some metrics have tighter margins than others.

**Impact**: The empathy dimension of the CI gate is effectively non-functional. A significant empathy degradation would pass undetected.

### C3. `detect_regression()` accepts arbitrary baseline keys without validation (llm_judge.py:602)

**Bug**: If `baseline` contains a key that doesn't match any metric (e.g., typo `"emphaty"` instead of `"empathy"`), `detect_regression` constructs `current_key = "avg_emphaty"`, gets `current_score = 0.0` from `current_dict.get()`, and reports a false regression. Conversely, the real metric `empathy` would be silently unchecked.

```python
# Typo in baseline silently causes false regression + missed real metric
detect_regression(report, {"emphaty": 0.4})  # False positive for "emphaty"
# Real "empathy" never checked
```

**Fix**: Validate baseline keys against `ALL_METRICS` at function entry:
```python
unknown = set(baseline.keys()) - set(ALL_METRICS)
if unknown:
    raise ValueError(f"Unknown baseline metrics: {unknown}")
```

---

## High Issues (should fix)

### H1. `guest_experience` score collapses when evaluated without component metrics (llm_judge.py:558-561)

**Bug**: `evaluate_conversation(..., metrics=[METRIC_GUEST_EXPERIENCE])` passes an empty `scores` dict to `_score_guest_experience_offline()`. Since no component metrics were computed, `weighted_sum = 0.0` and guest_experience relies only on the helpfulness bonus (max 0.15). The 85% weighted-component portion is zeroed out.

Actual result: `guest_experience = 0.105` for a response that scores 0.63 when all metrics are evaluated.

**Fix**: Either force component metric evaluation when guest_experience is requested, or document this dependency explicitly and raise a warning.

### H2. Data contamination when last conversation turn is from user (llm_judge.py:826-838)

**Bug**: In `run_conversation_evaluation()`, the loop sets `last_response` to the last assistant turn's content. But `prior_messages` includes ALL turns. The code only removes the last message from `prior_messages` if it's an assistant message (line 837). When the last turn is from the user, `prior_messages` still contains the assistant response being evaluated.

```python
# Turns: [user, assistant("Welcome!"), user("Thanks!")]
# After loop: prior_messages = [user, assistant, user]  # assistant still in context
# last_response = "Welcome!"  # same text is in both context and response
```

This inflates `conversation_flow` scores because word overlap between context and response is artificially high.

**Fix**: Remove ALL messages after (and including) the last evaluated assistant turn from `prior_messages`, not just the last element:
```python
# Find index of last assistant message and slice
for i in range(len(prior_messages) - 1, -1, -1):
    if prior_messages[i].get("role") == "assistant":
        prior_messages = prior_messages[:i]
        break
```

### H3. Test comments use stale/incorrect baseline values (test_r22_llm_judge.py:214-236)

**Bug**: Multiple test comments reference baseline values that don't match `QUALITY_BASELINE`:

- Line 214: `avg_empathy=0.52` comment says "Baseline 0.55, drop is 0.03" -- actual baseline is 0.40 (drop is -0.12, ABOVE baseline)
- Line 229: `avg_empathy=0.30` comment says "-0.25 from baseline" -- actual drop is 0.10
- Lines 230-233: All drop comments are wrong (0.30 stated vs 0.25 actual for cultural, etc.)

The tests still pass because the assertion logic is correct, but incorrect comments will mislead developers into thinking the thresholds are different than they actually are. This is a maintenance hazard.

**Fix**: Update all comments to reflect actual `QUALITY_BASELINE` values.

### H4. `test_no_regression_within_threshold` doesn't actually test the threshold boundary (test_r22_llm_judge.py:212-223)

**Bug**: The test claims to verify "no regression when drop is within threshold" but every test value is ABOVE the baseline (positive margin), not within the threshold window. There is no test for the exact boundary condition: `drop == threshold` (should NOT regress) vs `drop == threshold + epsilon` (SHOULD regress).

```python
# Current: all values are above baseline, testing nothing about the threshold
avg_empathy=0.52  # 0.12 ABOVE baseline 0.40 -- not testing threshold at all

# Should test: values slightly below baseline but within threshold
avg_empathy=0.37  # 0.03 below baseline 0.40, within threshold 0.05 -- no regression
```

**Fix**: Use values that are actually between `baseline` and `baseline - threshold` to exercise the threshold logic.

### H5. No negative threshold validation in `detect_regression()` (llm_judge.py:578)

**Bug**: A negative `threshold` (e.g., `threshold=-0.1`) would make the condition `drop > -0.1` true even when the score IMPROVED. This would flag improvements as regressions. No validation prevents this.

**Fix**: Add `assert threshold >= 0` or `threshold = abs(threshold)` at function entry.

### H6. Two parallel evaluation systems with no cross-reference (evaluation.py + llm_judge.py)

**Issue**: `evaluation.py` provides single-turn Q&A evaluation with 4 dimensions (groundedness, helpfulness, safety, persona). `llm_judge.py` provides multi-turn conversation evaluation with 5 different dimensions (empathy, cultural_sensitivity, conversation_flow, persona_consistency, guest_experience).

There is NO documentation, code comment, or test that verifies these systems produce consistent results. The persona evaluation in `evaluation.py` (`score_persona`) and `llm_judge.py` (`_score_persona_consistency_offline`) use the same violation patterns but completely different scoring logic (subtractive from 1.0 vs additive from 0.6 base). A response could pass persona in one system and fail in the other.

**Fix**: Add a cross-system validation test that evaluates the same response through both systems and asserts their persona dimensions agree within a tolerance.

---

## Medium Issues (nice to fix)

### M1. Missing golden conversation categories

The `GOLDEN_CONVERSATIONS` dataset (6 cases) is missing several important conversation patterns:

1. **Topic switching mid-conversation**: "Tell me about restaurants" then "Actually, what about the spa?" -- tests context switching without losing prior context.
2. **Return to previous topic**: User asks about dining, asks about shows, then returns to dining -- tests memory retention across topic changes.
3. **Multi-language**: Spanish/Chinese/Arabic greeting followed by English -- tests graceful language handling.
4. **Adversarial in conversation**: User starts with legitimate question then attempts prompt injection mid-conversation -- tests guardrail consistency across turns.
5. **Ambiguous/vague follow-up**: "What about that other one?" after discussing multiple options -- tests clarification behavior.
6. **Sarcasm/frustration escalation**: "Oh sure, another 30 minutes, just great" -- tests sarcasm detection in multi-turn.
7. **VIP recognition and comp inquiry**: "I'm a Diamond member, what perks do I get?" -- tests loyalty program awareness.
8. **Group decision making**: "We can't agree on a restaurant, half want Italian, half want seafood" -- tests compromise suggestion ability.

### M2. Empathy scoring doesn't differentiate emotional intensity (llm_judge.py:266-284)

The emotional word list uses binary detection (`any(w in last_user_msg)`). "I'm a little confused" and "I'm absolutely furious and devastated" both trigger the same +0.2 bonus. The `expected_empathy_level` field in `ConversationTestCase` is defined but never used in scoring.

### M3. `_score_conversation_flow_offline` stop-word list is incomplete (llm_judge.py:364-366)

The exclusion set for "significant words" is small (20 words). Common words like "could", "would", "should", "there", "these", "their", "just", "some", "more", "into" would all count as significant topic words (4+ chars), inflating relevance scores for generic responses.

### M4. No test for `ConversationEvalReport.to_dict()` rounding (test_r22_llm_judge.py)

`ConversationEvalReport.to_dict()` rounds values to 4 decimal places, but no test verifies this rounding behavior. Only `ConversationEvalScore.to_dict()` rounding is tested (line 378-386).

### M5. `_response_lower` variable computed but unused in guest_experience scoring (llm_judge.py:473)

Line 473: `response_lower = response.lower()` is computed but never used. The `info_patterns` all have `(?i)` flags, making the lowered version unnecessary. This is dead code.

### M6. Persona violation scoring can go negative before clamping (llm_judge.py:414-418)

If a response has 5+ persona violations (e.g., multiple emoji blocks + informal words), the score drops: `0.6 - 5*0.15 = -0.15`. This gets clamped to 0.0 by `max(0.0, ...)`, but the intermediate negative value means the positive markers (lines 420-423) are wasted -- their bonus is absorbed into bringing the score back to 0.0 rather than improving it. Consider clamping between penalty and bonus stages.

### M7. `run_conversation_evaluation` divides by `n` without ZeroDivisionError protection in edge case (llm_judge.py:843-857)

The early return for empty conversations (line 812) protects against n=0. But if `conversations` contains only items where ALL turns are filtered out (theoretically impossible given the data structure, but defensive coding would check), `all_scores` would be non-empty with zero-value scores, and `n` would be > 0. Low risk but worth noting.

### M8. `evaluate_conversation` doesn't validate `metrics` parameter (llm_judge.py:524)

If `metrics=["nonexistent"]` is passed, the function silently returns all-zeros with no warning. This makes integration errors hard to diagnose.

---

## Scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Eval (evaluation quality)** | 5/10 | NaN hole (C1), empathy baseline gap (C2), and no cross-system validation (H6) undermine regression detection. The gate would miss real regressions in empathy and any metric returning NaN. |
| **Testing (test coverage)** | 6/10 | Good coverage of happy paths and basic edge cases. Missing: NaN handling, threshold boundary conditions, cross-system consistency, topic-switching golden conversations, data contamination edge case. Stale comments (H3) are a maintenance risk. |
| **Architecture** | 7/10 | Clean separation between offline/LLM modes. Dataclass design is solid. Two evaluation systems (evaluation.py for single-turn, llm_judge.py for multi-turn) complement each other well in theory. But lack of cross-reference, the guest_experience dependency issue (H1), and the missing `expected_empathy_level` integration (M2) show the architecture isn't fully connected yet. |

**Overall**: 6/10 -- The framework has good bones but the regression detection has real gaps. C1 (NaN) and C2 (empathy margin) could both allow quality degradation to pass CI undetected.

---

## Missing Golden Conversations

| ID Suggestion | Category | Pattern | Why Important |
|---|---|---|---|
| `multi-topic-switch-01` | topic_switch | Dining -> Spa mid-conversation | Tests context switching, common in real usage |
| `multi-return-topic-01` | retention | Dining -> Shows -> Back to dining | Tests long-range memory, high-value for casino host |
| `multi-language-01` | multilingual | Spanish greeting -> English conversation | Casino guests are multilingual, tests graceful handling |
| `multi-adversarial-01` | adversarial | Legitimate question -> prompt injection attempt | Tests guardrail consistency across turns |
| `multi-vague-followup-01` | ambiguous | Multiple options discussed -> "What about that one?" | Tests clarification behavior, very common pattern |
| `multi-sarcasm-01` | complaint | Sarcastic frustration ("Oh great, another wait") | Tests tone detection beyond keyword matching |
| `multi-vip-comp-01` | vip | Diamond member asking about comps/perks | Core casino host function, not represented |
| `multi-group-01` | group | Group with conflicting preferences | Tests compromise/suggestion ability |
