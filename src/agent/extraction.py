"""Deterministic field extraction from guest messages.

Regex-based extraction of structured data from natural language messages.
Sub-1ms, no LLM call, fail-silent (returns empty dict on any error).

Extracted fields populate ``extracted_fields`` in state, which feeds
the whisper planner and guest profile systems.

Feature flag: ``field_extraction_enabled`` (default: True).
"""

import logging
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

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
# Loyalty signal patterns (B2: implicit loyalty/VIP recognition)
# ---------------------------------------------------------------------------

_LOYALTY_PATTERNS: list[re.Pattern] = [
    # "20 years" / "member for 10 years" / "coming here for 5 years"
    re.compile(r"(?i)(?:member|coming here|visiting|been a (?:guest|member))\s+(?:for\s+)?(\d+)\s+years?"),
    # "Momentum member" / "Gold tier" / "Platinum member"
    re.compile(r"(?i)(momentum|gold|platinum|silver|diamond|elite|vip)\s+(?:member|tier|status|level)"),
    re.compile(r"(?i)(?:member|tier|status|level)\s+(?:is\s+)?(momentum|gold|platinum|silver|diamond|elite|vip)"),
    # "I spend a lot" / "high roller" / "big spender"
    re.compile(r"(?i)(?:spend\s+a\s+lot|high\s+roller|big\s+spender|whale|i\s+come\s+(?:here\s+)?every)"),
    # "used to be Gold" / "was a Platinum member" (with or without "member/tier" suffix)
    re.compile(r"(?i)(?:used\s+to\s+be|was\s+(?:a\s+)?)\s*(gold|platinum|silver|diamond|elite|vip)(?:\s+(?:member|tier))?"),
]

# ---------------------------------------------------------------------------
# Urgency signal patterns (B2: implicit urgency detection)
# ---------------------------------------------------------------------------

_URGENCY_PATTERNS: list[re.Pattern] = [
    # "checking out in an hour" / "leaving soon" / "flight in 3 hours"
    re.compile(r"(?i)(?:checking out|leaving|departing|flight|checkout)\s+(?:in\s+)?(?:\d+\s+|an?\s+)?(?:hour|minute|soon)"),
    # "leaving soon" without time reference
    re.compile(r"(?i)(?:checking out|leaving|departing)\s+soon"),
    # "quick" / "fast" / "hurry" / "rush"
    re.compile(r"(?i)\b(?:quick(?:ly)?|fast|hurry|rush(?:ed|ing)?|right\s+now|immediately|asap)\b"),
    # "don't have much time" / "limited time" / "short on time"
    re.compile(r"(?i)(?:don'?t\s+have\s+(?:much|a\s+lot\s+of)\s+time|limited\s+time|short\s+on\s+time|running\s+late)"),
]

# ---------------------------------------------------------------------------
# Fatigue signal patterns (B2: implicit fatigue/comfort needs)
# ---------------------------------------------------------------------------

_FATIGUE_PATTERNS: list[re.Pattern] = [
    # "exhausted" / "tired" / "long day" / "on our feet all day"
    re.compile(r"(?i)\b(?:exhausted|tired|wiped(?:\s+out)?|beat|drained|fatigued)\b"),
    re.compile(r"(?i)(?:long\s+(?:day|drive|flight|trip)|on\s+(?:our|my)\s+feet\s+all\s+day)"),
    re.compile(r"(?i)(?:drove|traveled|flew)\s+(?:\d+\s+)?hours?"),
    re.compile(r"(?i)\b(?:need\s+to\s+(?:unwind|relax|rest|decompress)|want\s+to\s+(?:unwind|relax|rest))\b"),
]

# ---------------------------------------------------------------------------
# Budget signal patterns (B2: implicit budget consciousness)
# ---------------------------------------------------------------------------

_BUDGET_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)\b(?:cheap(?:er)?|affordable|budget|inexpensive|economical)\b"),
    re.compile(r"(?i)(?:nothing\s+(?:too\s+)?expensive|not\s+(?:too\s+)?pricey|on\s+a\s+budget)"),
    re.compile(r"(?i)(?:free|complimentary|no\s+(?:cover|charge)|don'?t\s+(?:want\s+to\s+)?spend\s+(?:too\s+)?much)"),
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

        # Loyalty signals (B2: implicit VIP/loyalty recognition)
        for pattern in _LOYALTY_PATTERNS:
            match = pattern.search(text)
            if match:
                fields["loyalty_signal"] = match.group(0).strip()
                break

        # Urgency signals (B2: implicit urgency detection)
        for pattern in _URGENCY_PATTERNS:
            if pattern.search(text):
                fields["urgency"] = True
                break

        # Fatigue signals (B2: implicit fatigue/comfort needs)
        for pattern in _FATIGUE_PATTERNS:
            if pattern.search(text):
                fields["fatigue"] = True
                break

        # Budget signals (B2: implicit budget consciousness)
        for pattern in _BUDGET_PATTERNS:
            if pattern.search(text):
                fields["budget_conscious"] = True
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


def get_guest_profile_summary(extracted_fields: dict[str, Any]) -> str:
    """Format extracted fields as a human-readable profile summary.

    Designed for human host handoff: when the AI concierge transfers
    a conversation to a human casino host, this summary provides context
    about everything learned from the guest during the AI interaction.

    R72 Phase C4: Based on research into Ritz-Carlton preference pads
    and Four Seasons anticipatory service — human hosts need structured
    guest intelligence, not raw data dumps.

    Args:
        extracted_fields: Accumulated dict from the _merge_dicts reducer.

    Returns:
        Formatted multi-line string suitable for host display.
        Returns empty string if no fields are populated.
    """
    if not extracted_fields:
        return ""

    lines: list[str] = []

    # Identity & party
    if extracted_fields.get("name"):
        lines.append(f"Guest name: {extracted_fields['name']}")
    if extracted_fields.get("party_size"):
        lines.append(f"Party size: {extracted_fields['party_size']}")

    # Occasion & visit
    if extracted_fields.get("occasion"):
        lines.append(f"Occasion: {extracted_fields['occasion']}")
    if extracted_fields.get("visit_date"):
        lines.append(f"Visit date: {extracted_fields['visit_date']}")

    # Preferences & dietary
    if extracted_fields.get("preferences"):
        lines.append(f"Dietary/preferences: {extracted_fields['preferences']}")

    # Loyalty signals
    if extracted_fields.get("loyalty_signal"):
        lines.append(f"Loyalty signal: {extracted_fields['loyalty_signal']}")

    # Behavioral signals
    signals: list[str] = []
    if extracted_fields.get("urgency"):
        signals.append("time-constrained")
    if extracted_fields.get("fatigue"):
        signals.append("fatigued/travel-weary")
    if extracted_fields.get("budget_conscious"):
        signals.append("budget-conscious")
    if signals:
        lines.append(f"Behavioral signals: {', '.join(signals)}")

    if not lines:
        return ""

    return "Guest Profile Summary:\n" + "\n".join(f"  - {line}" for line in lines)


# ---------------------------------------------------------------------------
# R75 B2: LLM-augmented field extraction for conversational messages
# ---------------------------------------------------------------------------
# Regex extraction misses conversational paraphrases like "birthday next
# Saturday for four people". LLM fallback fires only when regex returns
# empty AND text has 15+ words (cost control).
#
# Feature flag: ``extraction_llm_augmented`` (default False — opt-in).
# Merge: LLM results fill gaps only — regex wins on conflicts (deterministic
# is more reliable than probabilistic).


class ExtractionOutput(BaseModel):
    """Structured output for LLM field extraction."""

    name: str | None = Field(default=None, description="Guest name if mentioned")
    party_size: int | None = Field(default=None, description="Number of guests")
    visit_date: str | None = Field(default=None, description="When they're visiting")
    occasion: str | None = Field(default=None, description="Special occasion")
    preferences: str | None = Field(default=None, description="Dietary or venue preferences")


async def extract_fields_augmented(
    text: str,
    regex_result: dict[str, Any],
    get_llm_fn: Callable,
) -> dict[str, Any]:
    """LLM fallback extraction for conversational messages regex misses.

    Only fires when regex returns empty AND text has 15+ words.
    Merges LLM results with regex (regex wins on conflicts).

    Args:
        text: The guest's message text.
        regex_result: Pre-computed regex extraction result from extract_fields().
        get_llm_fn: Async callable that returns an LLM instance.

    Returns:
        Dict of extracted field name -> value. May be augmented with LLM fields.
    """
    # If regex already found fields, no need for LLM
    if regex_result:
        return regex_result

    # Short messages unlikely to have extractable info
    if len(text.split()) < 15:
        return regex_result

    try:
        llm = await get_llm_fn()
        extraction_llm = llm.with_structured_output(ExtractionOutput)
        prompt = (
            "Extract any guest information from this casino concierge message. "
            "Only extract information that is explicitly stated, never infer.\n\n"
            f"Message: {text}"
        )
        result: ExtractionOutput = await extraction_llm.ainvoke(prompt)

        # Merge: only add non-None LLM fields that regex didn't find
        merged = dict(regex_result)
        for field_name in ("name", "party_size", "visit_date", "occasion", "preferences"):
            value = getattr(result, field_name, None)
            if value is not None and field_name not in merged:
                merged[field_name] = value

        if merged != regex_result:
            logger.info("LLM extraction augmented: %s new fields", len(merged) - len(regex_result))
        return merged
    except Exception:
        logger.debug("LLM extraction augmentation failed, using regex result", exc_info=True)
        return regex_result
