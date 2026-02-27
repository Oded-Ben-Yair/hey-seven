"""Gambling slang and drunk-typing normalization for improved RAG retrieval.

Normalizes casino-specific slang, abbreviations, and common misspellings
into standard English BEFORE the RAG retrieval step. This improves
embedding quality and search relevance without affecting the guest's
original message in conversation history.

**Important**: Normalization is for SEARCH ONLY — never store or display
the normalized text. The original message is preserved in state.

Feature flag: ``slang_normalization_enabled`` (default True).

R72 Phase C3: Based on domain research into real VIP-to-host SMS patterns.
"""

import logging
import re

logger = logging.getLogger(__name__)

__all__ = ["normalize_for_search"]

# ---------------------------------------------------------------------------
# Gambling slang → plain English (for embedding/search quality)
# ---------------------------------------------------------------------------
# Ordered from most specific to most general to prevent partial matches.
# All patterns are case-insensitive via re.IGNORECASE in the compiled regex.

_GAMBLING_SLANG: dict[str, str] = {
    # Table games
    "on tilt": "frustrated after losing",
    "tilting": "frustrated after losing",
    "the cooler": "bad luck person",
    "whale": "high-value VIP guest",
    "high roller": "VIP guest",
    "big spender": "VIP guest",
    "grind": "steady gambling",
    "grinding": "steady gambling",
    "degen": "frequent gambler",
    "run bad": "losing streak",
    "running bad": "losing streak",
    "the nuts": "best hand",
    "cold streak": "losing streak",
    "hot streak": "winning streak",
    "on a heater": "winning streak",
    "busted": "lost all money",
    "busted out": "lost all money",
    "felted": "lost all money at poker",
    "crapped out": "lost at craps",
    "snake eyes": "rolled two ones at craps",
    "boxcars": "rolled two sixes at craps",
    # Comps and host language
    "RFB": "room food and beverage complimentary",
    "rfb": "room food and beverage complimentary",
    "ADT": "average daily theoretical",
    "adt": "average daily theoretical",
    "theo": "theoretical expected value",
    "markers": "casino credit line",
    "marker": "casino credit",
    "front money": "cash deposit for gambling",
    "comp": "complimentary benefit",
    "comped": "given for free as a benefit",
    "comp me": "give me a complimentary benefit",
    "junket": "organized gambling trip",
    # Crypto/modern gambling
    "ape in": "invest aggressively",
    "aped in": "invested aggressively",
    "YOLO": "risking everything",
    "yolo": "risking everything",
    "diamond hands": "holding through losses",
    "paper hands": "selling at a loss",
    "moon": "big win",
    "rekt": "lost everything",
    "degen play": "risky gamble",
}

# ---------------------------------------------------------------------------
# Common drunk-typing / SMS abbreviations → standard English
# ---------------------------------------------------------------------------

_DRUNK_TYPING: dict[str, str] = {
    # Common misspellings from SMS with hosts
    "resturant": "restaurant",
    "restaraunt": "restaurant",
    "restraunt": "restaurant",
    "resterant": "restaurant",
    "restuarant": "restaurant",
    "dinning": "dining",
    "steakhoue": "steakhouse",
    "steakhous": "steakhouse",
    "buffett": "buffet",
    "bufet": "buffet",
    "buffe": "buffet",
    "enterainment": "entertainment",
    "entertanment": "entertainment",
    "accomodation": "accommodation",
    "acommodation": "accommodation",
    "reccomend": "recommend",
    "recomend": "recommend",
    "reccommend": "recommend",
    "reservaton": "reservation",
    "reservatn": "reservation",
    # SMS abbreviations
    "rm": "room",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "thnx": "thanks",
    "nite": "night",
    "tonite": "tonight",
    "tmrw": "tomorrow",
    "tmw": "tomorrow",
    "2nite": "tonight",
    "2moro": "tomorrow",
    "2morrow": "tomorrow",
    "ur": "your",
    "u": "you",
    "cn": "can",
    "wht": "what",
    "wher": "where",
    "lmk": "let me know",
    "asap": "as soon as possible",
    "btw": "by the way",
    "idk": "I don't know",
    "rn": "right now",
    "cuz": "because",
    "bc": "because",
    "tho": "though",
    "b4": "before",
    "w/": "with",
    "w/o": "without",
    "upgrde": "upgrade",
    "upgrd": "upgrade",
}

# Pre-compile patterns for O(1) lookup per word
# For multi-word slang, use regex; for single-word, use dict lookup
_MULTI_WORD_SLANG: list[tuple[re.Pattern, str]] = []
_SINGLE_WORD_SLANG: dict[str, str] = {}

for phrase, replacement in _GAMBLING_SLANG.items():
    if " " in phrase:
        # Multi-word: compile as regex pattern with word boundaries
        _MULTI_WORD_SLANG.append((
            re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE),
            replacement,
        ))
    else:
        _SINGLE_WORD_SLANG[phrase.lower()] = replacement

# Single-word drunk typing corrections (case-insensitive lookup)
_SINGLE_WORD_DRUNK: dict[str, str] = {k.lower(): v for k, v in _DRUNK_TYPING.items()}


def normalize_for_search(text: str) -> str:
    """Normalize gambling slang and drunk-typing for improved RAG retrieval.

    Applies normalization to produce a search-friendly version of the guest
    message. The original message is NEVER modified in conversation state.

    Processing order:
    1. Multi-word gambling slang (longest match first)
    2. Single-word gambling slang
    3. Drunk-typing corrections

    Args:
        text: The guest's original message text.

    Returns:
        Normalized text for search. Returns original text on any error.
    """
    if not text or not isinstance(text, str):
        return text or ""

    try:
        result = text

        # 1. Multi-word slang substitution (preserves surrounding text)
        for pattern, replacement in _MULTI_WORD_SLANG:
            result = pattern.sub(replacement, result)

        # 2. Single-word substitution (word-by-word, preserves spacing)
        words = result.split()
        normalized_words = []
        for word in words:
            # Strip trailing punctuation for lookup, re-attach after.
            # Also handle contractions like "YOLO'd" — strip 'd/'s/'t
            stripped = word.rstrip(".,!?;:'\")")
            # Handle common contractions: YOLO'd → YOLO, it's → it
            if stripped.endswith("'d") or stripped.endswith("'s") or stripped.endswith("'t"):
                stripped = stripped[:-2]
                punct = word[len(stripped):]
            else:
                punct = word[len(stripped):]
            lookup = stripped.lower()

            if lookup in _SINGLE_WORD_SLANG:
                normalized_words.append(_SINGLE_WORD_SLANG[lookup] + punct)
            elif lookup in _SINGLE_WORD_DRUNK:
                normalized_words.append(_SINGLE_WORD_DRUNK[lookup] + punct)
            else:
                normalized_words.append(word)

        return " ".join(normalized_words)

    except Exception:
        logger.debug("Slang normalization failed, returning original", exc_info=True)
        return text
