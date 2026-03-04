"""VADER-based sentiment detection for casino guest messages.

Sub-1ms sentiment analysis with casino-aware adjustments.
Fail-silent: returns "neutral" on any error.

Feature flag: ``sentiment_detection_enabled`` (default True).
"""

import logging
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "detect_sentiment",
    "detect_sarcasm_context",
    "detect_sentiment_augmented",
    "SentimentOutput",
]

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
    # "Just great" / "Just wonderful" / "Just fantastic" (standalone sarcasm).
    # Anchored to sentence start or post-comma to avoid false positives on
    # sincere mid-sentence usage like "That was just wonderful, thank you!"
    r"(?i)(?:^|[.,!?]\s*)\s*just\s+(?:great|wonderful|fantastic|perfect|lovely)\b",
    # "Thanks for nothing" / "Thanks a lot" (sarcastic thanks)
    r"(?i)\bthanks\s+(?:for\s+nothing|a\s+lot)\b",
    # "Yeah right" / "Sure, that helps"
    r"(?i)\b(?:yeah\s+right|sure[,!]?\s+that\s+(?:helps?|works?))\b",
    # "Love waiting" / "Love how everything takes forever" (sarcastic love)
    r"(?i)\blove\s+(?:waiting|how\s+(?:long|slow|everything|nothing))\b",
    # R70 B1 fixes: backhanded compliments and passive resignation
    # "I suppose" / "I guess" as qualifier after statement — hedging dissatisfaction
    r"(?i)(?:clean|fine|ok(?:ay)?|good|nice|decent)\s+I\s+(?:suppose|guess)\b",
    # "Could have been worse" / "Could be worse" — damning with faint praise
    r"(?i)\bcould\s+(?:have\s+been|be)\s+worse\b",
    # Standalone "Very helpful" / "Very nice" / "Very useful" — ironic when isolated
    # Anchored to sentence boundary to avoid false positives on sincere usage
    r"(?i)(?:^|[.,!?]\s+)very\s+(?:helpful|nice|useful|informative)\s*[.!?]?\s*$",
    # "Whatever" / "If you say so" / "Sure whatever" — passive resignation
    r"(?i)\b(?:sure\s+)?whatever\b(?:\s*[.,!]?\s*$)",
    r"(?i)\bif\s+you\s+say\s+so\b",
    # R88: Casino-context sarcasm patterns
    # "Oh wonderful, another chatbot" / "This should be fun" — chatbot skepticism
    r"(?i)\b(?:oh\s+)?wonderful[,!]?\s+another\b",
    r"(?i)\bthis\s+should\s+be\s+(?:fun|interesting|great)\b",
    # "Wow, really helpful" / "How helpful" — dripping sarcasm
    r"(?i)\b(?:wow[,!]?\s+)?(?:really|how|so)\s+helpful\b",
    # "That's exactly what I needed" — sarcastic when frustrated
    r"(?i)\bthat'?s?\s+exactly\s+what\s+I\s+(?:needed|wanted)\b",
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


# ---------------------------------------------------------------------------
# R72 B1: Context-aware sarcasm detection (semantic incongruity)
# ---------------------------------------------------------------------------
# When VADER returns "positive" or "neutral" for the current message,
# but the conversation history shows negative signals (prior frustrated
# messages, complaints, corrections), the current message may be sarcastic.
#
# This is a zero-LLM-cost, sub-1ms approach that uses VADER compound
# scores to detect incongruity between the current message and recent
# conversation context. No additional API calls needed.
#
# Research basis (R72 A6): Embedding-based incongruity detection achieves
# 70-80% sarcasm F1, but adds 10-30ms latency per call. Context-contrast
# using existing VADER scores is free and catches the most common pattern:
# positive-sounding text in a negative conversation context.

# Positive-leaning words that appear in sarcastic messages
_SARCASM_POSITIVE_SIGNALS: set[str] = {
    "great",
    "wonderful",
    "fantastic",
    "perfect",
    "lovely",
    "amazing",
    "awesome",
    "brilliant",
    "excellent",
    "beautiful",
    "helpful",
    "useful",
    "impressive",
    "nice",
    "good",
    "love",
    "enjoy",
    "fun",
    "terrific",
    "outstanding",
    "marvelous",
    "superb",
    "delightful",
}


def detect_sarcasm_context(
    current_text: str,
    current_sentiment: str,
    recent_sentiments: list[str],
) -> bool:
    """Detect likely sarcasm via context contrast (no LLM call).

    When the current message reads as positive/neutral but the recent
    conversation history has been negative/frustrated, the positive tone
    may be sarcastic. This catches patterns like:

    - Guest: [frustrated] "I've been waiting forever"
    - Guest: [frustrated] "The room was terrible"
    - Guest: "Oh yeah this is really great service" → sarcasm

    Args:
        current_text: The current guest message.
        current_sentiment: VADER sentiment for current message
            ("positive", "negative", "neutral", "frustrated").
        recent_sentiments: Sentiments of the last 2-3 human messages
            (most recent first).

    Returns:
        True if the message is likely sarcastic, False otherwise.
    """
    if not current_text or not recent_sentiments:
        return False

    try:
        # Only check when current message reads positive or neutral
        if current_sentiment not in ("positive", "neutral"):
            return False

        # Need at least 1 recent negative/frustrated message for contrast
        negative_count = sum(
            1 for s in recent_sentiments[:3] if s in ("frustrated", "negative")
        )
        if negative_count == 0:
            return False

        # Check if current message contains positive-leaning words
        # (sarcasm uses positive words in negative context)
        text_lower = current_text.lower()
        words = set(text_lower.split())
        has_positive_words = bool(words & _SARCASM_POSITIVE_SIGNALS)

        if not has_positive_words:
            return False

        # Context contrast: positive words + recent negative history = likely sarcasm
        # Require at least 2 negative signals for stronger confidence
        # OR 1 negative + very short message (terse positive = more likely sarcastic)
        word_count = len(current_text.split())
        if negative_count >= 2:
            logger.debug(
                "Sarcasm detected via context contrast: positive words + %d recent negative",
                negative_count,
            )
            return True
        if negative_count >= 1 and word_count <= 8:
            # Short message with positive words after negative history
            # e.g., "Great service" after "I've been waiting forever"
            logger.debug(
                "Sarcasm detected: short positive (%d words) + recent negative",
                word_count,
            )
            return True

        return False

    except Exception:
        logger.debug("Sarcasm context detection failed", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# R75 B1: LLM-augmented sentiment for VADER's ambiguous band
# ---------------------------------------------------------------------------
# When VADER returns "neutral" for a 10+ word message, the compound score
# is in the ambiguous range (-0.3 to 0.3). An LLM call provides refined
# classification for these borderline cases only.
#
# Feature flag: ``sentiment_llm_augmented`` (default False — opt-in).
# Fast path: Clear VADER results (positive/negative/frustrated) skip LLM.
# Fallback: If LLM fails, VADER result is returned (never worse than current).


class SentimentOutput(BaseModel):
    """Structured output for LLM sentiment classification."""

    sentiment: Literal["positive", "negative", "neutral", "frustrated"] = Field(
        description="The detected sentiment of the guest message"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in classification"
    )


async def detect_sentiment_augmented(
    text: str,
    get_llm_fn: Callable,
    vader_result: str | None = None,
) -> str:
    """LLM-augmented sentiment for VADER's ambiguous band.

    Fast path: If VADER result is clear (positive/negative/frustrated),
    return immediately without invoking the LLM.

    Slow path: If VADER is neutral AND text has 10+ words, invoke LLM
    for refined classification using structured output.

    Fallback: If LLM fails, return VADER result (never degrades quality).

    Args:
        text: The guest's message text.
        get_llm_fn: Async callable that returns an LLM instance.
        vader_result: Pre-computed VADER result, or None to compute it.

    Returns:
        One of: "positive", "negative", "neutral", "frustrated".
    """
    # Fast path: use existing VADER detection
    if vader_result is None:
        vader_result = detect_sentiment(text)

    # Only augment ambiguous cases
    if vader_result != "neutral":
        return vader_result

    # Short messages don't benefit from LLM analysis
    if len(text.split()) < 10:
        return vader_result

    try:
        llm = await get_llm_fn()
        sentiment_llm = llm.with_structured_output(SentimentOutput)
        prompt = (
            "Classify the sentiment of this casino guest message. "
            "Consider context: the guest is interacting with an AI concierge at a casino resort.\n\n"
            f"Message: {text}\n\n"
            "Classify as: positive (happy, excited, satisfied), negative (unhappy, disappointed), "
            "neutral (informational, no emotion), frustrated (annoyed, impatient, complaining)."
        )
        result: SentimentOutput = await sentiment_llm.ainvoke(prompt)
        logger.info(
            "LLM sentiment augmentation: %s -> %s (conf=%.2f)",
            vader_result,
            result.sentiment,
            result.confidence,
        )
        return result.sentiment
    except Exception:
        logger.debug(
            "LLM sentiment augmentation failed, using VADER result", exc_info=True
        )
        return vader_result
