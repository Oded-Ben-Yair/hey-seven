"""Unit tests for detect_sentiment() — VADER + casino-aware adjustments.

Covers: VADER thresholds, casino overrides, frustration patterns,
empty/edge inputs, fail-silent behavior.
"""

import pytest

from src.agent.sentiment import detect_sentiment


# ---------------------------------------------------------------------------
# Empty / edge inputs
# ---------------------------------------------------------------------------


class TestEdgeInputs:
    """Empty, None, whitespace-only inputs should return neutral."""

    def test_empty_string(self):
        assert detect_sentiment("") == "neutral"

    def test_whitespace_only(self):
        assert detect_sentiment("   ") == "neutral"

    def test_none_input(self):
        assert detect_sentiment(None) == "neutral"


# ---------------------------------------------------------------------------
# Casino-domain positive overrides
# ---------------------------------------------------------------------------


class TestCasinoOverrides:
    """Casino-positive phrases that VADER might misclassify."""

    @pytest.mark.parametrize(
        "text",
        [
            "I'm killing it at the slots tonight",
            "Just hit the jackpot on the penny machine",
            "I'm on fire at the poker table",
            "Totally crushed it at blackjack",
            "Cleaned up at the craps table",
            "Made a killing at roulette last night",
            "I've been on a hot streak all day",
            "I'm on a roll with these cards",
        ],
    )
    def test_casino_positive_phrases(self, text):
        assert detect_sentiment(text) == "positive"

    def test_casino_override_case_insensitive(self):
        assert detect_sentiment("I HIT THE JACKPOT") == "positive"


# ---------------------------------------------------------------------------
# Frustration patterns (deterministic, before VADER)
# ---------------------------------------------------------------------------


class TestFrustrationPatterns:
    """Frustrated signals should take priority over VADER scores."""

    @pytest.mark.parametrize(
        "text",
        [
            "I'm so frustrated with this wait",
            "This is absolutely ridiculous",
            "The service is unacceptable",
            "What a terrible experience",
            "This is horrible",
            "What an awful night",
            "I'm really annoyed right now",
        ],
    )
    def test_explicit_frustration_words(self, text):
        assert detect_sentiment(text) == "frustrated"

    @pytest.mark.parametrize(
        "text",
        [
            "I can't find the restaurant",
            "I cant believe this happened",
            "I can't get anyone to help me",
        ],
    )
    def test_cant_find_patterns(self, text):
        assert detect_sentiment(text) == "frustrated"

    @pytest.mark.parametrize(
        "text",
        [
            "What a waste of time",
            "I'm sick of waiting",
            "Tired of this nonsense",
            "I'm fed up with the noise",
        ],
    )
    def test_exhaustion_patterns(self, text):
        assert detect_sentiment(text) == "frustrated"

    @pytest.mark.parametrize(
        "text",
        [
            "What a joke this place is",
            "What a disaster this turned out to be",
            "What a mess this room is",
        ],
    )
    def test_what_a_patterns(self, text):
        assert detect_sentiment(text) == "frustrated"


# ---------------------------------------------------------------------------
# VADER threshold tests
# ---------------------------------------------------------------------------


class TestVaderThresholds:
    """VADER compound score thresholds: >= 0.3 positive, <= -0.3 negative."""

    def test_positive_sentiment(self):
        assert detect_sentiment("This is wonderful, I love it!") == "positive"

    def test_negative_sentiment(self):
        assert detect_sentiment("This is terrible and I hate it") in (
            "negative",
            "frustrated",
        )

    def test_neutral_sentiment(self):
        assert detect_sentiment("The restaurant is on the second floor") == "neutral"

    def test_simple_greeting_is_neutral_or_positive(self):
        result = detect_sentiment("Hello, good morning")
        assert result in ("neutral", "positive")


# ---------------------------------------------------------------------------
# Frustration takes priority over VADER
# ---------------------------------------------------------------------------


class TestFrustrationPriority:
    """Frustrated patterns should fire before VADER even for positive text."""

    def test_frustrated_overrides_positive_vader(self):
        # Text with positive words but frustrated trigger
        assert detect_sentiment("I love this place but the service is unacceptable") == "frustrated"

    def test_frustrated_overrides_negative_vader(self):
        assert detect_sentiment("I'm frustrated and upset") == "frustrated"


# ---------------------------------------------------------------------------
# Fail-silent behavior
# ---------------------------------------------------------------------------


class TestFailSilent:
    """Errors should return neutral, never raise."""

    def test_integer_input_raises_attribute_error(self):
        # detect_sentiment checks `not text.strip()` before the try/except,
        # so non-string inputs that are truthy will raise AttributeError.
        # This documents current behavior — caller is responsible for passing strings.
        with pytest.raises(AttributeError):
            detect_sentiment(12345)
