"""Tests for LLM-augmented sentiment detection (B1).

Covers: fast-path skip for clear sentiments, ambiguous band triggers LLM,
short message bypass, LLM failure fallback, structured output schema.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.sentiment import (
    SentimentOutput,
    detect_sentiment,
    detect_sentiment_augmented,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(sentiment: str = "frustrated", confidence: float = 0.85):
    """Create a mock LLM that returns a SentimentOutput via structured output."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        return_value=SentimentOutput(sentiment=sentiment, confidence=confidence)
    )
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


async def _mock_get_llm(sentiment: str = "frustrated", confidence: float = 0.85):
    """Async factory that returns a mock LLM."""
    return _make_mock_llm(sentiment, confidence)


# ---------------------------------------------------------------------------
# SentimentOutput schema validation
# ---------------------------------------------------------------------------


class TestSentimentOutputSchema:
    """Validate the Pydantic model for structured LLM output."""

    def test_valid_positive(self):
        out = SentimentOutput(sentiment="positive", confidence=0.9)
        assert out.sentiment == "positive"
        assert out.confidence == 0.9

    def test_valid_frustrated(self):
        out = SentimentOutput(sentiment="frustrated", confidence=0.75)
        assert out.sentiment == "frustrated"

    def test_confidence_bounds_low(self):
        with pytest.raises(Exception):
            SentimentOutput(sentiment="neutral", confidence=-0.1)

    def test_confidence_bounds_high(self):
        with pytest.raises(Exception):
            SentimentOutput(sentiment="neutral", confidence=1.1)

    def test_invalid_sentiment(self):
        with pytest.raises(Exception):
            SentimentOutput(sentiment="angry", confidence=0.5)


# ---------------------------------------------------------------------------
# Fast path: clear VADER results skip LLM
# ---------------------------------------------------------------------------


class TestFastPathSkipsLLM:
    """When VADER returns a clear result, LLM should NOT be called."""

    @pytest.mark.asyncio
    async def test_clear_positive_skips_llm(self):
        mock_llm = _make_mock_llm()
        get_llm = AsyncMock(return_value=mock_llm)

        result = await detect_sentiment_augmented(
            "I'm so happy and excited about this amazing trip!",
            get_llm,
            vader_result="positive",
        )
        assert result == "positive"
        get_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_negative_skips_llm(self):
        mock_llm = _make_mock_llm()
        get_llm = AsyncMock(return_value=mock_llm)

        result = await detect_sentiment_augmented(
            "This is terrible and disappointing service",
            get_llm,
            vader_result="negative",
        )
        assert result == "negative"
        get_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_frustrated_skips_llm(self):
        mock_llm = _make_mock_llm()
        get_llm = AsyncMock(return_value=mock_llm)

        result = await detect_sentiment_augmented(
            "I've been waiting forever and this is ridiculous",
            get_llm,
            vader_result="frustrated",
        )
        assert result == "frustrated"
        get_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Short message bypass
# ---------------------------------------------------------------------------


class TestShortMessageBypass:
    """Messages under 10 words skip LLM even if VADER is neutral."""

    @pytest.mark.asyncio
    async def test_short_neutral_skips_llm(self):
        mock_llm = _make_mock_llm()
        get_llm = AsyncMock(return_value=mock_llm)

        result = await detect_sentiment_augmented(
            "What time is dinner?",
            get_llm,
            vader_result="neutral",
        )
        assert result == "neutral"
        get_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_exactly_9_words_skips_llm(self):
        mock_llm = _make_mock_llm()
        get_llm = AsyncMock(return_value=mock_llm)

        # 9 words
        result = await detect_sentiment_augmented(
            "I would like to know about the restaurant please",
            get_llm,
            vader_result="neutral",
        )
        assert result == "neutral"
        get_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Ambiguous band triggers LLM
# ---------------------------------------------------------------------------


class TestAmbiguousTrigger:
    """Neutral VADER + 10+ words should invoke LLM."""

    @pytest.mark.asyncio
    async def test_ambiguous_triggers_llm(self):
        mock_llm = _make_mock_llm(sentiment="frustrated", confidence=0.9)
        get_llm = AsyncMock(return_value=mock_llm)

        # 14 words, VADER neutral
        result = await detect_sentiment_augmented(
            "I have been waiting here for quite a while now and nobody seems to care about us",
            get_llm,
            vader_result="neutral",
        )
        assert result == "frustrated"
        get_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_ambiguous_returns_llm_positive(self):
        mock_llm = _make_mock_llm(sentiment="positive", confidence=0.8)
        get_llm = AsyncMock(return_value=mock_llm)

        result = await detect_sentiment_augmented(
            "We are looking forward to visiting the resort next week and spending time at the spa and casino",
            get_llm,
            vader_result="neutral",
        )
        assert result == "positive"


# ---------------------------------------------------------------------------
# LLM failure fallback
# ---------------------------------------------------------------------------


class TestLLMFailureFallback:
    """When LLM fails, detect_sentiment_augmented returns VADER result."""

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_vader(self):
        get_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        result = await detect_sentiment_augmented(
            "I have been waiting here for quite a while now and nobody seems to notice or care",
            get_llm,
            vader_result="neutral",
        )
        assert result == "neutral"

    @pytest.mark.asyncio
    async def test_llm_invoke_error_falls_back(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=ValueError("Bad response"))
        mock_llm.with_structured_output.return_value = mock_structured
        get_llm = AsyncMock(return_value=mock_llm)

        result = await detect_sentiment_augmented(
            "I have been waiting here for quite a while now and nobody seems to notice or care",
            get_llm,
            vader_result="neutral",
        )
        assert result == "neutral"


# ---------------------------------------------------------------------------
# VADER auto-computation
# ---------------------------------------------------------------------------


class TestVADERAutoCompute:
    """When vader_result is None, detect_sentiment is called automatically."""

    @pytest.mark.asyncio
    async def test_auto_computes_vader(self):
        mock_llm = _make_mock_llm()
        get_llm = AsyncMock(return_value=mock_llm)

        # "killing it" triggers casino positive override
        result = await detect_sentiment_augmented(
            "I am killing it at the tables tonight",
            get_llm,
            vader_result=None,
        )
        assert result == "positive"
        get_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Regression: existing detect_sentiment unchanged
# ---------------------------------------------------------------------------


class TestDetectSentimentRegression:
    """Verify existing detect_sentiment behavior is unchanged."""

    def test_positive(self):
        assert detect_sentiment("I love this place, it is amazing!") == "positive"

    def test_negative(self):
        # "terrible" triggers frustrated pattern; use pure negative without frustration keywords
        assert detect_sentiment("I am very unhappy with the situation") == "negative"

    def test_neutral(self):
        assert detect_sentiment("What time is dinner?") == "neutral"

    def test_frustrated(self):
        assert detect_sentiment("This is ridiculous, I can't believe this!") == "frustrated"

    def test_empty(self):
        assert detect_sentiment("") == "neutral"
