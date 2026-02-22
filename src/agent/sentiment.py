"""VADER-based sentiment detection for casino guest messages.

Sub-1ms sentiment analysis with casino-aware adjustments.
Fail-silent: returns "neutral" on any error.

Feature flag: ``sentiment_detection_enabled`` (default True).
"""

import logging
import re

logger = logging.getLogger(__name__)

__all__ = ["detect_sentiment"]

# Casino-domain positive phrases that VADER may misclassify
_CASINO_POSITIVE_OVERRIDES: set[str] = {
    "killing it",
    "hit the jackpot",
    "on fire",
    "crushed it",
    "cleaned up",
    "made a killing",
    "on a hot streak",
    "on a roll",
}

# Frustrated signal phrases (stronger than generic negative)
_FRUSTRATED_PATTERNS: list[str] = [
    r"(?i)\b(frustrated|annoyed|ridiculous|unacceptable|terrible|horrible|awful)\b",
    r"(?i)\bcan'?t\s+(find|believe|get|figure)\b",
    r"(?i)\b(waste\s+of|sick\s+of|tired\s+of|fed\s+up)\b",
    r"(?i)\bwhat\s+a\s+(joke|disaster|mess)\b",
]


def detect_sentiment(text: str) -> str:
    """Detect sentiment of a guest message using VADER with casino-aware adjustments.

    Args:
        text: The guest's message text.

    Returns:
        One of: "positive", "negative", "neutral", "frustrated".
        Returns "neutral" on any error (fail-silent).
    """
    if not text or not text.strip():
        return "neutral"

    try:
        # Check for frustrated patterns first (deterministic, before VADER)
        for pattern in _FRUSTRATED_PATTERNS:
            if re.search(pattern, text):
                return "frustrated"

        # Check casino-domain overrides
        text_lower = text.lower()
        for phrase in _CASINO_POSITIVE_OVERRIDES:
            if phrase in text_lower:
                return "positive"

        # Lazy import VADER (sub-1ms after first load)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.3:
            return "positive"
        if compound <= -0.3:
            return "negative"
        return "neutral"

    except Exception:
        logger.warning("Sentiment detection failed, returning neutral", exc_info=True)
        return "neutral"
