"""Tests for _should_inject_suggestion() helper in _base.py.

R74 B4: Proactive suggestion injection gating logic. Extracted to a pure
function for testability — no LLM mocking needed.

Tests the 5 gating conditions:
  1. Whisper plan has a non-empty proactive_suggestion
  2. suggestion_confidence >= 0.6 (R82 1F: lowered from 0.8)
  3. Sentiment not negative/frustrated/None
  4. suggestion_offered is False (max 1 per session)
  5. retrieved_context is non-empty (grounding exists)
"""

import pytest

from src.agent.agents._base import _should_inject_suggestion


def _base_state(**overrides):
    """Build a minimal state dict that passes ALL suggestion gates."""
    base = {
        "whisper_plan": {
            "proactive_suggestion": "Try Todd English's Tuscany for Italian cuisine",
            "suggestion_confidence": 0.9,
        },
        "suggestion_offered": False,
        "retrieved_context": [
            {"content": "Italian restaurant info", "metadata": {}, "score": 0.9},
        ],
        "guest_sentiment": "positive",
        "extracted_fields": {},
        "messages": [],
    }
    base.update(overrides)
    return base


def _base_dynamics(**overrides):
    """Build dynamics dict with defaults."""
    base = {
        "terse_replies": 0,
        "repeated_question": False,
        "brevity_preference": False,
        "turn_count": 1,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path: all conditions met
# ---------------------------------------------------------------------------


class TestProactiveSuggestionInjected:
    def test_injected_when_all_conditions_met(self):
        """Suggestion appears when: confidence >= 0.8, positive sentiment,
        not offered yet, grounding exists."""
        state = _base_state()
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is True
        assert "Todd English" in section
        assert "Proactive Suggestion" in section
        assert "don't force it" in section

    def test_injected_on_neutral_with_occasion(self):
        """Neutral sentiment + occasion context allows injection."""
        state = _base_state(extracted_fields={"occasion": "anniversary"})
        section, mark = _should_inject_suggestion(state, "neutral", _base_dynamics())

        assert mark is True
        assert "Todd English" in section

    def test_injected_on_neutral_with_engagement(self):
        """Neutral sentiment + 3+ turns of engagement allows injection."""
        state = _base_state()
        dynamics = _base_dynamics(turn_count=3)
        section, mark = _should_inject_suggestion(state, "neutral", dynamics)

        assert mark is True
        assert "Todd English" in section

    def test_exact_confidence_threshold(self):
        """Confidence exactly at 0.6 is accepted (R82: lowered from 0.8, >= not >)."""
        state = _base_state(
            whisper_plan={
                "proactive_suggestion": "Check out the spa",
                "suggestion_confidence": 0.6,
            }
        )
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is True
        assert "spa" in section


# ---------------------------------------------------------------------------
# Suppression: each gating condition individually
# ---------------------------------------------------------------------------


class TestProactiveSuggestionSuppressed:
    def test_suppressed_on_negative_sentiment(self):
        """Suggestion NOT injected when sentiment is negative."""
        state = _base_state()
        section, mark = _should_inject_suggestion(state, "negative", _base_dynamics())

        assert mark is False
        assert section == ""

    def test_suppressed_on_frustrated_sentiment(self):
        """Suggestion NOT injected when sentiment is frustrated."""
        state = _base_state()
        section, mark = _should_inject_suggestion(state, "frustrated", _base_dynamics())

        assert mark is False
        assert section == ""

    def test_suppressed_on_none_sentiment(self):
        """Suggestion NOT injected when sentiment detection failed (None)."""
        state = _base_state()
        section, mark = _should_inject_suggestion(state, None, _base_dynamics())

        assert mark is False
        assert section == ""

    def test_suppressed_when_already_offered(self):
        """Suggestion NOT injected when suggestion_offered is True (max 1 per session)."""
        state = _base_state(suggestion_offered=True)
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is False
        assert section == ""

    def test_suppressed_without_grounding(self):
        """Suggestion NOT injected when retrieved_context is empty."""
        state = _base_state(retrieved_context=[])
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is False
        assert section == ""

    def test_suppressed_on_low_confidence(self):
        """Suggestion NOT injected when confidence < 0.6 (R82: lowered from 0.8)."""
        state = _base_state(
            whisper_plan={
                "proactive_suggestion": "Try the buffet",
                "suggestion_confidence": 0.59,
            }
        )
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is False
        assert section == ""

    def test_suppressed_when_no_whisper_plan(self):
        """Suggestion NOT injected when whisper_plan is None."""
        state = _base_state(whisper_plan=None)
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is False
        assert section == ""

    def test_suppressed_when_suggestion_is_empty_string(self):
        """Suggestion NOT injected when proactive_suggestion is empty string."""
        state = _base_state(
            whisper_plan={
                "proactive_suggestion": "",
                "suggestion_confidence": 0.9,
            }
        )
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is False
        assert section == ""

    def test_suppressed_when_suggestion_is_none(self):
        """Suggestion NOT injected when proactive_suggestion is None."""
        state = _base_state(
            whisper_plan={
                "proactive_suggestion": None,
                "suggestion_confidence": 0.9,
            }
        )
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is False
        assert section == ""

    def test_neutral_without_contextual_signal_now_passes(self):
        """R82 1F: Neutral sentiment now passes without occasion or 3+ turns."""
        state = _base_state()
        dynamics = _base_dynamics(turn_count=1)
        section, mark = _should_inject_suggestion(state, "neutral", dynamics)

        assert mark is True  # R82: was False, now True
        assert "Todd English" in section

    def test_suppressed_on_missing_confidence_key(self):
        """Missing suggestion_confidence key defaults to 0.0 (blocked)."""
        state = _base_state(
            whisper_plan={
                "proactive_suggestion": "Try the spa",
                # no suggestion_confidence key
            }
        )
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is False
        assert section == ""


# ---------------------------------------------------------------------------
# Edge cases and combinations
# ---------------------------------------------------------------------------


class TestProactiveSuggestionEdgeCases:
    def test_retrieved_context_none_treated_as_empty(self):
        """retrieved_context=None is treated as no grounding."""
        state = _base_state(retrieved_context=None)
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is False
        assert section == ""

    def test_multiple_conditions_fail_simultaneously(self):
        """When multiple conditions fail, still returns empty."""
        state = _base_state(
            whisper_plan={
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": 0.5,  # low confidence
            },
            suggestion_offered=True,  # already offered
            retrieved_context=[],  # no grounding
        )
        section, mark = _should_inject_suggestion(state, "frustrated", _base_dynamics())

        assert mark is False
        assert section == ""

    def test_suggestion_text_preserved_exactly(self):
        """The suggestion text from whisper plan is injected verbatim."""
        suggestion_text = "Visit Elemis Spa for their signature aromatherapy treatment"
        state = _base_state(
            whisper_plan={
                "proactive_suggestion": suggestion_text,
                "suggestion_confidence": 0.95,
            }
        )
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is True
        assert suggestion_text in section

    def test_confidence_just_below_threshold(self):
        """Confidence at 0.599... is rejected (R82: threshold is now 0.6)."""
        state = _base_state(
            whisper_plan={
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": 0.5999,
            }
        )
        section, mark = _should_inject_suggestion(state, "positive", _base_dynamics())

        assert mark is False
        assert section == ""
