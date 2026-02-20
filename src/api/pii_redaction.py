"""PII redaction for logs, traces, and structured output.

Regex-based redaction of personally identifiable information before
the data reaches logging, LangFuse traces, or any external system.

Redaction is applied to:
- Structured log output (via middleware)
- LangFuse trace metadata (before sending to LangFuse)
- Error messages in API responses

Casino-domain PII patterns:
- Phone numbers (E.164 and US formats)
- Email addresses
- Credit card numbers (Visa, MC, Amex, Discover)
- SSN (full and partial)
- Player card / loyalty card numbers
- Names preceded by common identifiers ("Mr.", "Mrs.", "my name is")

Design: Fails CLOSED — if redaction fails, a placeholder is returned
instead of the original text. PII never leaks to logs on error.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redaction patterns (compiled for performance)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Phone numbers: E.164, US formats
    (re.compile(r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[PHONE]', 'phone'),
    # Email addresses
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), '[EMAIL]', 'email'),
    # Credit cards: Visa, MC, Amex, Discover (with optional separators)
    (re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'), '[CARD]', 'credit_card'),
    # Amex: 15 digits
    (re.compile(r'\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b'), '[CARD]', 'amex'),
    # SSN: full (XXX-XX-XXXX)
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]', 'ssn'),
    # SSN: no separators (9 consecutive digits in SSN context)
    (re.compile(r'(?i)(?:ssn|social\s+security)[:\s]*(\d{9})\b'), '[SSN]', 'ssn_raw'),
    # Player/loyalty card numbers (6-12 digit sequences with common prefixes)
    (re.compile(r'(?i)(?:player|loyalty|rewards?|member)\s*(?:card\s*(?:number|#|id)?|number|#|id)[:\s]*(\d{6,12})\b'), '[PLAYER_ID]', 'player_card'),
]

# Name patterns: conservative — case-insensitive prefix but the captured
# name group must start with an actual uppercase letter (proper noun).
# This prevents false positives like "I'm your concierge" → "[NAME]"
# when redaction is applied to bot-generated template text.
# Python's (?i) affects the entire pattern including character classes, so
# we use a post-match validation lambda instead (see redact_pii / contains_pii).
_NAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(?i)\b(?:my\s+name\s+is|i\'?m|this\s+is)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)'), 'name_self_id'),
    (re.compile(r'(?i)\b(?:mr\.?|mrs\.?|ms\.?|dr\.?)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)'), 'name_honorific'),
]


def _is_proper_name(match: re.Match) -> bool:
    """Check if captured name group starts with uppercase (proper noun).

    Under (?i) flag, [A-Z] matches both cases. This post-match validation
    ensures only actual proper nouns are treated as names, preventing
    false positives on lowercase words like "your" in "I'm your concierge".
    """
    name = match.group(1)
    return bool(name and name[0].isupper())


def redact_pii(text: str) -> str:
    """Redact PII patterns from text.

    Applies all regex patterns and replaces matches with redaction tokens.
    Fails CLOSED: if redaction errors, returns a safe placeholder instead
    of the original text. PII never leaks to downstream consumers on error.

    Args:
        text: Input text potentially containing PII.

    Returns:
        Text with PII replaced by redaction tokens.
    """
    if not text:
        return text

    try:
        result = text

        # Apply standard patterns
        for pattern, replacement, _name in _PATTERNS:
            result = pattern.sub(replacement, result)

        # Apply name patterns (replace the captured group only).
        # Post-match validation: only redact if captured name starts with uppercase
        # (proper noun). Prevents "I'm your concierge" → "I'm [NAME]".
        for pattern, _name in _NAME_PATTERNS:
            result = pattern.sub(
                lambda m: m.group(0).replace(m.group(1), '[NAME]') if _is_proper_name(m) else m.group(0),
                result,
            )

        return result
    except Exception:
        logger.warning("PII redaction failed; returning redacted placeholder", exc_info=True)
        return "[PII_REDACTION_ERROR]"


def redact_dict(data: dict[str, Any], *, keys_to_redact: set[str] | None = None) -> dict[str, Any]:
    """Redact PII from string values in a dict (shallow).

    Only processes string values. Does not recurse into nested dicts
    unless they are in the keys_to_redact set.

    Args:
        data: Dict with potentially PII-containing string values.
        keys_to_redact: Specific keys to redact. If None, redacts all string values.

    Returns:
        New dict with redacted string values.
    """
    result = {}
    target_keys = keys_to_redact or set(data.keys())

    for key, value in data.items():
        if key in target_keys and isinstance(value, str):
            result[key] = redact_pii(value)
        else:
            result[key] = value

    return result


def contains_pii(text: str) -> bool:
    """Check if text contains any PII patterns.

    Useful for validation/alerting without modifying the text.

    Args:
        text: Text to check.

    Returns:
        True if any PII pattern matches.
    """
    if not text:
        return False

    for pattern, _, _ in _PATTERNS:
        if pattern.search(text):
            return True

    for pattern, _ in _NAME_PATTERNS:
        match = pattern.search(text)
        if match and _is_proper_name(match):
            return True

    return False
