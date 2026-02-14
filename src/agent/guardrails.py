"""Deterministic input guardrails (pre-LLM safety nets).

Prompt injection detection and responsible gaming detection run before any
LLM call, providing a deterministic first line of defense independent of
model behavior.

Extracted from ``nodes.py`` to separate guardrail concerns from graph node
logic.  Both functions are stateless and side-effect-free (aside from logging).
"""

import logging
import re

logger = logging.getLogger(__name__)

__all__ = [
    "audit_input",
    "detect_responsible_gaming",
    "detect_age_verification",
    "detect_bsa_aml",
]

# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------

#: Regex patterns for prompt injection detection.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\b", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"\bDAN\b.*\bmode\b", re.I),
    re.compile(r"pretend\s+(?:you(?:'re|\s+are)\s+)?(?:a|an|the)\b", re.I),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|your)\b", re.I),
    # "act as" — require role-play framing (article + noun), exclude hospitality
    # phrases like "act as a guide" which are legitimate casino-context queries.
    re.compile(r"act\s+as\s+(?:if\s+)?(?:you(?:'re|\s+are)\s+)?(?:a|an|the)\s+(?!guide\b|concierge\b|host\b)", re.I),
]

# ---------------------------------------------------------------------------
# Responsible gaming patterns
# ---------------------------------------------------------------------------

#: Regex patterns for responsible gaming detection (pre-LLM safety net).
#: Includes English and Spanish patterns for multilingual guest populations.
_RESPONSIBLE_GAMING_PATTERNS = [
    # English patterns
    re.compile(r"gambling\s+problem", re.I),
    re.compile(r"problem\s+gambl", re.I),
    re.compile(r"addict(?:ed|ion)?\s+(?:to\s+)?gambl", re.I),
    re.compile(r"self[- ]?exclu", re.I),
    re.compile(r"can'?t\s+stop\s+gambl", re.I),
    re.compile(r"help\s+(?:with|for)\s+gambl", re.I),
    re.compile(r"gambling\s+helpline", re.I),
    re.compile(r"compulsive\s+gambl", re.I),
    re.compile(r"gambl(?:ing)?\s+addict", re.I),
    re.compile(r"lost\s+(?:all|everything)\s+gambl", re.I),
    re.compile(r"gambl(?:ing)?\s+(?:is\s+)?ruin", re.I),
    re.compile(r"(?:want|need)\s+to\s+(?:ban|exclude)\s+(?:myself|me)", re.I),
    re.compile(r"limit\s+my\s+(?:gambl|play|betting)", re.I),
    re.compile(r"take\s+a\s+break\s+from\s+gambl", re.I),
    re.compile(r"spend(?:ing)?\s+too\s+much\s+(?:at\s+(?:the\s+)?casino|gambl)", re.I),
    re.compile(r"(?:my\s+)?family\s+(?:says?|thinks?)\s+I\s+gambl", re.I),
    re.compile(r"cool(?:ing)?[- ]?off\s+period", re.I),
    # Spanish patterns (US casino diverse clientele)
    re.compile(r"problema\s+de\s+juego", re.I),
    re.compile(r"adicci[oó]n\s+al\s+juego", re.I),
    re.compile(r"no\s+puedo\s+(?:parar|dejar)\s+de\s+jugar", re.I),
    re.compile(r"ayuda\s+con\s+(?:el\s+)?juego", re.I),
    re.compile(r"juego\s+compulsivo", re.I),
    # Mandarin patterns (CT casino significant Asian clientele)
    re.compile(r"赌博\s*(?:成瘾|上瘾|问题)", re.I),  # gambling addiction/problem
    re.compile(r"戒\s*赌", re.I),                     # quit gambling
    re.compile(r"赌瘾", re.I),                         # gambling addiction (colloquial)
]

# ---------------------------------------------------------------------------
# Age verification patterns (casino guests must be 21+)
# ---------------------------------------------------------------------------

#: Regex patterns for detecting underage-related queries.
#: Mohegan Sun requires guests to be 21+ for gaming and most venues.
_AGE_VERIFICATION_PATTERNS = [
    re.compile(r"\b(?:my|our)\s+(?:\d{1,2}[- ]?year[- ]?old|kid|child|teen|son|daughter|minor)", re.I),
    re.compile(r"\b(?:under\s*(?:age|21|18)|underage|too\s+young)\b", re.I),
    re.compile(r"\bcan\s+(?:my\s+)?(?:kid|child|teen|minor)s?\s+(?:play|gamble|enter|go)", re.I),
    re.compile(r"\b(?:minimum|legal)\s+(?:gambling|gaming|casino)\s+age\b", re.I),
    re.compile(r"\bhow\s+old\s+(?:do\s+you\s+have\s+to\s+be|to\s+(?:gamble|play|enter))", re.I),
    re.compile(r"\bminors?\b.*\b(?:allow|enter|visit|casino|gambl|play)", re.I),
]

# ---------------------------------------------------------------------------
# BSA/AML financial crime patterns (Bank Secrecy Act compliance)
# ---------------------------------------------------------------------------

#: Regex patterns for detecting queries related to financial crime, money
#: laundering, or structuring.  Casinos are MSBs under BSA and must report
#: CTRs (>$10 000 cash) and SARs.  The agent must never provide advice that
#: could facilitate structuring or help circumvent reporting requirements.
_BSA_AML_PATTERNS = [
    re.compile(r"\b(?:money\s+)?launder", re.I),
    re.compile(r"\bstructur(?:e|ing)\s+(?:cash|transaction|deposit|chip)", re.I),
    re.compile(r"\bavoid\s+(?:report|ctr|sar|detection|tax)", re.I),
    re.compile(r"\bcurrency\s+transaction\s+report", re.I),
    re.compile(r"\bsuspicious\s+activity\s+report", re.I),
    re.compile(r"\b(?:under|below)\s+\$?\s*10[\s,]?000\b", re.I),
    re.compile(r"\bsmur(?:f|fing)\b", re.I),
    re.compile(r"\bcash\s+out\s+(?:without|no)\s+(?:id|report|track)", re.I),
    re.compile(r"\bhide\s+(?:my\s+)?(?:money|cash|income|winnings)\b", re.I),
    re.compile(r"\b(?:un)?traceable\b.*\b(?:funds?|cash|money)\b", re.I),
    re.compile(r"\b(?:funds?|cash|money)\b.*\b(?:un)?traceable\b", re.I),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_input(message: str) -> bool:
    """Check user input for prompt injection patterns.

    Deterministic regex-based guardrail that runs before any LLM call.
    Logs a warning if injection patterns are detected.

    Args:
        message: The raw user input message.

    Returns:
        True if the input looks safe, False if injection detected.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(message):
            logger.warning("Prompt injection detected (pattern: %s)", pattern.pattern[:60])
            return False
    return True


def detect_responsible_gaming(message: str) -> bool:
    """Check if user message indicates a gambling problem or self-exclusion need.

    Deterministic regex-based safety net that ensures responsible gaming
    helplines are always provided, regardless of LLM routing.

    Args:
        message: The raw user input message.

    Returns:
        True if responsible gaming support is needed.
    """
    for pattern in _RESPONSIBLE_GAMING_PATTERNS:
        if pattern.search(message):
            logger.info("Responsible gaming query detected (pattern: %s)", pattern.pattern[:60])
            return True
    return False


def detect_age_verification(message: str) -> bool:
    """Check if user message involves underage guests or age verification.

    Casino guests must be 21+ for gaming at Mohegan Sun (CT law). This
    deterministic guardrail ensures age-related queries always include the
    legal age requirement, independent of LLM behavior.

    Args:
        message: The raw user input message.

    Returns:
        True if age verification information should be included.
    """
    for pattern in _AGE_VERIFICATION_PATTERNS:
        if pattern.search(message):
            logger.info("Age verification query detected (pattern: %s)", pattern.pattern[:60])
            return True
    return False


def detect_bsa_aml(message: str) -> bool:
    """Check if user message relates to money laundering or BSA/AML evasion.

    Casinos are Money Services Businesses (MSBs) under the Bank Secrecy Act.
    They must file Currency Transaction Reports (CTRs) for cash transactions
    over $10,000 and Suspicious Activity Reports (SARs) for structuring.
    The agent must never provide guidance that could facilitate financial crime.

    Args:
        message: The raw user input message.

    Returns:
        True if BSA/AML compliance response should be triggered.
    """
    for pattern in _BSA_AML_PATTERNS:
        if pattern.search(message):
            logger.warning("BSA/AML query detected (pattern: %s)", pattern.pattern[:60])
            return True
    return False
