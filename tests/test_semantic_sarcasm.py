"""Tests for context-aware sarcasm detection.

R72 Phase C1: Verifies detect_sarcasm_context() correctly identifies
sarcasm when positive/neutral messages occur in negative conversation contexts.
"""

import pytest

from src.agent.sentiment import detect_sarcasm_context, detect_sentiment


class TestSarcasmContextDetection:
    """Context-contrast sarcasm detection."""

    def test_positive_after_two_negative_is_sarcasm(self):
        """'Great service' after 2 frustrated messages = sarcasm."""
        assert detect_sarcasm_context(
            current_text="Great service here",
            current_sentiment="positive",
            recent_sentiments=["frustrated", "negative"],
        ) is True

    def test_positive_after_one_negative_short_message(self):
        """Short positive after 1 frustrated = likely sarcasm."""
        assert detect_sarcasm_context(
            current_text="Very helpful",
            current_sentiment="positive",
            recent_sentiments=["frustrated"],
        ) is True

    def test_positive_after_positive_not_sarcasm(self):
        """Positive after positive = genuine."""
        assert detect_sarcasm_context(
            current_text="This is wonderful!",
            current_sentiment="positive",
            recent_sentiments=["positive", "positive"],
        ) is False

    def test_neutral_after_negative_no_positive_words(self):
        """Neutral text without positive words = not sarcasm."""
        assert detect_sarcasm_context(
            current_text="ok where's the pool",
            current_sentiment="neutral",
            recent_sentiments=["frustrated", "negative"],
        ) is False

    def test_negative_sentiment_not_checked(self):
        """Already negative messages don't need sarcasm check."""
        assert detect_sarcasm_context(
            current_text="This is terrible",
            current_sentiment="negative",
            recent_sentiments=["frustrated"],
        ) is False

    def test_frustrated_sentiment_not_checked(self):
        """Already frustrated messages don't need sarcasm check."""
        assert detect_sarcasm_context(
            current_text="What a joke",
            current_sentiment="frustrated",
            recent_sentiments=["frustrated"],
        ) is False

    def test_empty_history(self):
        """No history = no sarcasm detection possible."""
        assert detect_sarcasm_context(
            current_text="Great!",
            current_sentiment="positive",
            recent_sentiments=[],
        ) is False

    def test_empty_text(self):
        """Empty text = no sarcasm."""
        assert detect_sarcasm_context(
            current_text="",
            current_sentiment="positive",
            recent_sentiments=["frustrated"],
        ) is False

    def test_none_text(self):
        """None text = no sarcasm."""
        assert detect_sarcasm_context(
            current_text=None,
            current_sentiment="positive",
            recent_sentiments=["frustrated"],
        ) is False


class TestSarcasmPositiveWordDetection:
    """Positive words that commonly appear in sarcastic messages."""

    def test_great_detected(self):
        assert detect_sarcasm_context(
            "Oh great, this is great",
            "positive",
            ["frustrated", "negative"],
        ) is True

    def test_wonderful_detected(self):
        assert detect_sarcasm_context(
            "Wonderful experience",
            "positive",
            ["frustrated", "negative"],
        ) is True

    def test_amazing_detected(self):
        assert detect_sarcasm_context(
            "Amazing job so far",
            "positive",
            ["frustrated", "frustrated"],
        ) is True

    def test_helpful_detected(self):
        assert detect_sarcasm_context(
            "Very helpful",
            "positive",
            ["frustrated"],
        ) is True

    def test_perfect_detected(self):
        assert detect_sarcasm_context(
            "Perfect",
            "positive",
            ["frustrated", "negative"],
        ) is True


class TestSarcasmRealisticScenarios:
    """Realistic multi-turn sarcasm scenarios."""

    def test_hotel_complaint_sarcasm(self):
        """Guest complains across 2 turns then gives sarcastic 'compliment'."""
        # 2 prior negative turns + positive words in sarcastic message
        s1 = detect_sentiment("The room was terrible and the AC was awful")
        s2 = detect_sentiment("This is the worst hotel experience I've ever had")
        assert s1 in ("frustrated", "negative"), f"Expected negative, got {s1}"
        assert s2 in ("frustrated", "negative"), f"Expected negative, got {s2}"
        # "Amazing" is in the positive words set — common sarcastic word
        assert detect_sarcasm_context(
            "Amazing job so far, truly excellent service",
            "positive",
            [s1, s2],  # 2 previous frustrated turns
        ) is True

    def test_wait_time_sarcasm(self):
        """Guest waited too long, now sarcastic."""
        s1 = detect_sentiment("I've been waiting for 30 minutes for a table")
        assert detect_sarcasm_context(
            "No rush, take your time, it's not like we're hungry",
            "neutral",
            [s1],
        ) is False  # "hungry" not in positive words set, correct

    def test_genuine_positive_after_resolution(self):
        """Guest was frustrated but issue resolved — genuine positive."""
        # After resolution, guest genuinely happy
        assert detect_sarcasm_context(
            "That actually sounds wonderful, thank you so much for helping!",
            "positive",
            ["positive"],  # Previous turn was already resolved
        ) is False


class TestDomainTrackingState:
    """Verify the domains_discussed reducer works correctly."""

    def test_append_unique_basic(self):
        from src.agent.state import _append_unique

        result = _append_unique([], ["dining"])
        assert result == ["dining"]

    def test_append_unique_no_duplicates(self):
        from src.agent.state import _append_unique

        result = _append_unique(["dining"], ["dining"])
        assert result == ["dining"]

    def test_append_unique_accumulates(self):
        from src.agent.state import _append_unique

        result = _append_unique(["dining"], ["entertainment"])
        assert result == ["dining", "entertainment"]

    def test_append_unique_none_inputs(self):
        from src.agent.state import _append_unique

        assert _append_unique(None, None) == []
        assert _append_unique(None, ["dining"]) == ["dining"]
        assert _append_unique(["dining"], None) == ["dining"]

    def test_append_unique_empty_strings_filtered(self):
        from src.agent.state import _append_unique

        result = _append_unique([], ["", "dining", ""])
        assert result == ["dining"]
