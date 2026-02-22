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

# Sarcasm patterns: negation + positive word combos that VADER misclassifies.
# These fire AFTER frustration and BEFORE VADER to catch sarcastic complaints.
_SARCASM_PATTERNS: list[str] = [
    # "Great, another..." / "Oh great, ..."
    r"(?i)\b(?:oh\s+)?great[,!]?\s+another\b",
    # "Oh wonderful, now I have to wait..." (sarcastic wonderful)
    r"(?i)\boh\s+wonderful\b",
    # "Just great" / "Just wonderful" / "Just fantastic" (standalone sarcasm)
    r"(?i)\bjust\s+(?:great|wonderful|fantastic|perfect|lovely)\b",
    # "Thanks for nothing" / "Thanks a lot" (sarcastic thanks)
    r"(?i)\bthanks\s+(?:for\s+nothing|a\s+lot)\b",
    # "Yeah right" / "Sure, that helps"
    r"(?i)\b(?:yeah\s+right|sure[,!]?\s+that\s+(?:helps?|works?))\b",
    # "Love waiting" / "Love how everything takes forever" (sarcastic love)
    r"(?i)\blove\s+(?:waiting|how\s+(?:long|slow|everything|nothing))\b",
]


_vader_analyzer = None


def _get_vader_analyzer():
    """Lazy singleton for VADER SentimentIntensityAnalyzer.

    Avoids re-parsing the 7000-entry lexicon file on every call.
    Thread-safe: polarity_scores() is read-only after init.
    """
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


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

        # Check for sarcasm patterns (negation + positive word combos)
        # VADER misclassifies these as positive; we catch them deterministically.
        for pattern in _SARCASM_PATTERNS:
            if re.search(pattern, text):
                return "frustrated"

        # Check casino-domain overrides
        text_lower = text.lower()
        for phrase in _CASINO_POSITIVE_OVERRIDES:
            if phrase in text_lower:
                return "positive"

        # Lazy import VADER (sub-1ms after first load).
        # Module-level singleton avoids re-parsing the 7000-entry lexicon
        # on every call. Thread-safe: VADER's polarity_scores() is read-only.
        analyzer = _get_vader_analyzer()
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
