"""Deterministic field extraction from guest messages.

Regex-based extraction of structured data from natural language messages.
Sub-1ms, no LLM call, fail-silent (returns empty dict on any error).

Extracted fields populate ``extracted_fields`` in state, which feeds
the whisper planner and guest profile systems.

Feature flag: ``field_extraction_enabled`` (default: True).
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Name patterns
# ---------------------------------------------------------------------------

_NAME_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(?:my name is|i'm|i am|call me|this is)\s+([A-Z][a-z]+)(?:\s|,|\.|\!|\?|$)"),
    re.compile(r"(?i)(?:^|\.\s+)([A-Z][a-z]+) here\b"),
]

# ---------------------------------------------------------------------------
# Party size patterns
# ---------------------------------------------------------------------------

_PARTY_SIZE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(?:party of|group of|there(?:'s| is| are))\s+(\d{1,2})"),
    re.compile(r"(?i)(\d{1,2})\s+(?:of us|people|guests|persons)"),
    re.compile(r"(?i)(?:we are|we're)\s+(\d{1,2})"),
    re.compile(r"(?i)(?:for|table for)\s+(\d{1,2})(?:\s+(?:people|guests))?"),
]

# ---------------------------------------------------------------------------
# Visit date patterns
# ---------------------------------------------------------------------------

_VISIT_DATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(?:next|this)\s+((?:friday|saturday|sunday|monday|tuesday|wednesday|thursday|weekend))"),
    re.compile(r"(?i)(?:visiting|arriving|coming|staying|checking in)\s+(?:on\s+)?((?:next|this)\s+\w+|\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)"),
    re.compile(r"(?i)(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)"),
]

# ---------------------------------------------------------------------------
# Preference / dietary patterns
# ---------------------------------------------------------------------------

_PREFERENCE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(?:i'm|i am|we're|we are)\s+(vegetarian|vegan|gluten[- ]free|kosher|halal|pescatarian|dairy[- ]free)"),
    re.compile(r"(?i)(?:allergic to|allergy to|can't eat|cannot eat)\s+(\w+(?:\s+\w+)?)"),
    re.compile(r"(?i)(?:prefer|looking for|interested in)\s+(italian|chinese|japanese|mexican|seafood|steakhouse|sushi|thai|indian|french)"),
]

# ---------------------------------------------------------------------------
# Occasion patterns
# ---------------------------------------------------------------------------

_OCCASION_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(?:celebrating|it's|for)\s+(?:our |my |a )?(anniversary|birthday|wedding|honeymoon|graduation|retirement|bachelor(?:ette)?\s*party|promotion|engagement)"),
    re.compile(r"(?i)(anniversary|birthday|wedding|honeymoon|graduation|retirement|bachelor(?:ette)?\s*party|promotion|engagement)"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_fields(text: str) -> dict[str, Any]:
    """Extract structured fields from a guest message.

    Returns a dict with any of: name, party_size, visit_date, preferences,
    occasion. Empty dict if nothing extracted. Fail-silent on any error.

    Args:
        text: The guest's message text.

    Returns:
        Dict of extracted field name -> value. Empty if nothing found.
    """
    if not text or not isinstance(text, str):
        return {}

    try:
        fields: dict[str, Any] = {}

        # Name extraction
        for pattern in _NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip()
                # Basic sanity: name should be 2-30 chars and not a common word
                if 2 <= len(name) <= 30 and name.lower() not in _COMMON_WORDS:
                    fields["name"] = name
                    break

        # Party size
        for pattern in _PARTY_SIZE_PATTERNS:
            match = pattern.search(text)
            if match:
                size = int(match.group(1))
                if 1 <= size <= 50:  # Reasonable party size
                    fields["party_size"] = size
                    break

        # Visit date
        for pattern in _VISIT_DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                fields["visit_date"] = match.group(1).strip()
                break

        # Preferences / dietary
        all_prefs = []
        for pattern in _PREFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                pref = match.group(1).strip()
                if pref and pref not in all_prefs:
                    all_prefs.append(pref)
        if all_prefs:
            fields["preferences"] = ", ".join(all_prefs)

        # Occasion
        for pattern in _OCCASION_PATTERNS:
            match = pattern.search(text)
            if match:
                fields["occasion"] = match.group(1).strip().lower()
                break

        return fields

    except Exception:
        logger.debug("Field extraction failed, returning empty", exc_info=True)
        return {}


# Common words to exclude from name extraction (false positives)
_COMMON_WORDS: frozenset = frozenset({
    "here", "there", "just", "really", "very", "please", "thanks",
    "sorry", "sure", "okay", "hello", "help", "good", "great",
    "looking", "visiting", "staying", "coming", "going", "wondering",
    "vegetarian", "vegan", "pescatarian", "kosher", "halal",
    "allergic", "interested", "celebrating", "planning",
})
