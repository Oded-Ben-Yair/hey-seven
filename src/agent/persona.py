"""Persona envelope node — output guardrails + branding enforcement + SMS truncation.

Sits between ``validate`` (PASS) and ``respond`` in the v2 graph.

Processing order (each step operates on the output of the previous):
1. PII redaction (fail-closed — safety-critical)
2. BrandingConfig enforcement (exclamation limit, emoji check)
3. Guest name injection (when available)
4. SMS truncation (when PERSONA_MAX_CHARS > 0)
"""

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage

from src.api.pii_redaction import redact_pii
from src.casino.config import get_casino_profile
from src.config import get_settings

from .nodes import _normalize_content
from .state import PropertyQAState

logger = logging.getLogger(__name__)

__all__ = ["persona_envelope_node"]

# Common sentence starters that should be lowercased when prepending guest name.
# e.g., "We have great restaurants" -> "Sarah, we have great restaurants".
# Proper nouns (Mohegan, Bobby) are NOT in this set, so they stay capitalized.
_LOWERCASE_STARTERS: frozenset[str] = frozenset(
    {
        "I",
        "We",
        "Our",
        "You",
        "Your",
        "The",
        "There",
        "Here",
        "It",
        "This",
        "That",
        "As",
        "For",
        "With",
        "At",
        "To",
        "A",
        "An",
        "Yes",
        "No",
        "So",
        "Well",
        "Now",
        "Actually",
        "Absolutely",
        "Of",
        "Sure",
    }
)


def _validate_output(response_text: str) -> str:
    """Post-generation output guardrail. Catches accidental PII leakage.

    Applies PII redaction to the LLM response text. If the redacted text
    differs from the original, a warning is logged for observability.

    Args:
        response_text: The raw LLM response text.

    Returns:
        Redacted text (unchanged if no PII was detected).
    """
    redacted = redact_pii(response_text)
    if redacted != response_text:
        logger.warning("Output guardrail: PII detected in LLM response, redacting")
    # R86: CTR threshold redaction — prevent BSA/AML threshold disclosure
    redacted = re.sub(r"\$\s*10[,.]?000", "[regulatory threshold]", redacted)
    return redacted


def _enforce_branding(content: str, branding: dict) -> str:
    """Enforce BrandingConfig rules on response content.

    - Exclamation limit: reduces excess exclamation marks to the configured max.
    - Emoji check: strips emoji when emoji_allowed is False.

    Args:
        content: The response text after PII redaction.
        branding: BrandingConfig dict from casino config.

    Returns:
        Content with branding rules enforced.
    """
    # Exclamation limit enforcement
    exclamation_limit = branding.get("exclamation_limit", 1)
    exclamation_count = content.count("!")
    if exclamation_count > exclamation_limit:
        # Replace excess exclamation marks with periods, keeping the first N
        result_parts = []
        found = 0
        for char in content:
            if char == "!":
                found += 1
                if found <= exclamation_limit:
                    result_parts.append(char)
                else:
                    result_parts.append(".")
            else:
                result_parts.append(char)
        content = "".join(result_parts)

    # Emoji removal when not allowed
    if not branding.get("emoji_allowed", False):
        # Remove common emoji ranges (supplementary multilingual plane)
        content = re.sub(
            r"[\U0001F600-\U0001F64F"  # emoticons
            r"\U0001F300-\U0001F5FF"  # symbols & pictographs
            r"\U0001F680-\U0001F6FF"  # transport & map
            r"\U0001F1E0-\U0001F1FF"  # flags
            r"\U00002702-\U000027B0"  # dingbats
            r"\U0000FE00-\U0000FE0F"  # variation selectors
            r"\U0000200D"  # zero-width joiner
            r"]+",
            "",
            content,
        )

    return content


def _inject_guest_name(content: str, guest_name: str | None) -> str:
    """Inject guest name into the response for personalization.

    If the response doesn't already contain the guest's name and the name
    is available, prepend a personalized greeting to make the response
    feel more personal.

    Args:
        content: The response text.
        guest_name: The guest's name, or None if unknown.

    Returns:
        Content with guest name injected if appropriate.
    """
    if not guest_name:
        return content

    # Don't inject if the name is already in the response
    if guest_name.lower() in content.lower():
        return content

    # Don't inject into very short responses or fallback messages
    if len(content) < 50 or "I apologize" in content:
        return content

    # Prepend personalized greeting with guest name.
    # Only lowercase the first character when it begins a generic sentence
    # (e.g., "We have great..." → "Sarah, we have great...").
    # Preserve case for proper nouns (e.g., "Mohegan Sun has..." stays capitalized).
    first_word = content.split()[0] if content.split() else ""
    if first_word in _LOWERCASE_STARTERS:
        return f"{guest_name}, {content[0].lower()}{content[1:]}"
    return f"{guest_name}, {content}"


# Performative openers that make responses sound artificial
_PERFORMATIVE_OPENER_PATTERNS: list[re.Pattern] = [
    # "Absolutely! " / "Absolutely, "
    re.compile(r"^Absolutely[!,]\s*", re.IGNORECASE),
    # "Oh, " / "Oh! " at start of response
    re.compile(r"^Oh[,!]\s*", re.IGNORECASE),
    # "I'd be absolutely delighted" / "I'd be happy to" / "I'd be glad to"
    re.compile(
        r"^I'?d be (?:absolutely )?(?:delighted|happy|glad) to[!.]?\s*", re.IGNORECASE
    ),
    # "What a wonderful question!" / "What a great question!"
    re.compile(
        r"^What a (?:wonderful|great|fantastic|excellent) question[!.]?\s*",
        re.IGNORECASE,
    ),
    # "Great question!" at start
    re.compile(
        r"^(?:Great|Excellent|Wonderful|Fantastic) question[!.]?\s*", re.IGNORECASE
    ),
    # "Of course!" at start
    re.compile(r"^Of course[!,]\s*", re.IGNORECASE),
    # "Sure thing!" at start
    re.compile(r"^Sure thing[!,]\s*", re.IGNORECASE),
    # R88: "I appreciate you asking!" (patronizing)
    re.compile(r"^I appreciate you asking[!.]\s*", re.IGNORECASE),
]


def _strip_performative_openers(content: str) -> str:
    """Remove performative/artificial openers from LLM responses.

    Strips patterns like "Absolutely!", "Oh,", "I'd be delighted to",
    "Great question!" from the beginning of responses. These make AI
    responses sound artificial and performative.

    Also reduces exclamation clustering in the first paragraph (3+ → max 1).

    Args:
        content: The response text after PII redaction.

    Returns:
        Content with performative openers stripped.
    """
    if not content:
        return content

    # Strip performative openers (iterate: patterns can chain, e.g., "Oh, absolutely!")
    for _ in range(3):  # Max 3 passes (handles "Oh, absolutely! Great question!")
        original = content
        for pattern in _PERFORMATIVE_OPENER_PATTERNS:
            content = pattern.sub("", content, count=1)
        if content == original:
            break

    # Capitalize first letter after stripping (if it was lowered)
    if content and content[0].islower():
        content = content[0].upper() + content[1:]

    # Reduce exclamation clustering in first paragraph
    first_para_end = content.find("\n\n")
    if first_para_end == -1:
        first_para_end = len(content)
    first_para = content[:first_para_end]
    excl_count = first_para.count("!")
    if excl_count >= 3:
        # Keep only the first exclamation mark, replace rest with periods
        found = 0
        chars = list(first_para)
        for i, ch in enumerate(chars):
            if ch == "!":
                found += 1
                if found > 1:
                    chars[i] = "."
        content = "".join(chars) + content[first_para_end:]

    return content


async def persona_envelope_node(state: PropertyQAState) -> dict[str, Any]:
    """Apply output guardrails and persona formatting to the final response.

    Processing order:
    1. Output PII guardrail (always active, fail-closed)
    2. BrandingConfig enforcement (exclamation limit, emoji check)
    3. Guest name injection (when available)
    4. SMS truncation (when PERSONA_MAX_CHARS > 0)

    Args:
        state: The current graph state.

    Returns:
        Empty dict (passthrough) or dict with modified messages.
    """
    settings = get_settings()
    max_chars = settings.PERSONA_MAX_CHARS

    # Find the last AI message
    messages = state.get("messages", [])
    last_ai_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_msg = msg
            break

    if last_ai_msg is None:
        return {}

    content = _normalize_content(last_ai_msg.content)

    # Step 1: Output PII guardrail (always active, fail-closed)
    content = _validate_output(content)

    # Step 2: BrandingConfig enforcement
    # R29/R31 fix: use property-specific profile, never DEFAULT_CONFIG
    try:
        _profile = get_casino_profile(settings.CASINO_ID)
        branding = _profile.get("branding", {})
    except Exception:
        branding = get_casino_profile("").get("branding", {})
    content = _enforce_branding(content, branding)

    # Step 2b: Strip performative openers (R75 B6: tone calibration)
    content = _strip_performative_openers(content)

    # Step 3: Guest name injection
    # Fall back to extracted_fields["name"] when guest_name is None.
    # guest_name has no reducer so it resets per-turn; extracted_fields
    # persists via _merge_dicts reducer across turns.
    guest_name = state.get("guest_name") or (state.get("extracted_fields") or {}).get(
        "name"
    )
    content = _inject_guest_name(content, guest_name)

    # Step 4: SMS truncation (only when max_chars > 0)
    # R49 fix (Gemini CRITICAL-D10-001): Truncate at sentence boundary, not mid-word.
    # Hard truncation can chop TCPA-required compliance footer (STOP/HELP keywords)
    # and destroy actionable info (phone numbers, reservation codes).
    if max_chars > 0 and len(content) > max_chars:
        # Try sentence boundary first (period, exclamation, question mark)
        truncated = content[: max_chars - 3]
        last_sentence_end = max(
            truncated.rfind(". "),
            truncated.rfind("! "),
            truncated.rfind("? "),
            truncated.rfind(".\n"),
        )
        if last_sentence_end > max_chars // 2:
            # Found a sentence boundary in the second half — use it
            content = truncated[: last_sentence_end + 1].rstrip()
        else:
            # No good sentence boundary — fall back to word boundary
            last_space = truncated.rfind(" ")
            if last_space > max_chars // 2:
                content = truncated[:last_space].rstrip() + "..."
            else:
                # Very long word or no spaces — hard truncate as last resort
                content = truncated + "..."

    # Only return messages update if content actually changed
    original = _normalize_content(last_ai_msg.content)
    if content != original:
        return {"messages": [AIMessage(content=content)]}

    return {}
