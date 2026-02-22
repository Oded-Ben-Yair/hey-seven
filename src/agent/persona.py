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
from src.casino.config import DEFAULT_CONFIG
from src.config import get_settings

from .state import PropertyQAState

logger = logging.getLogger(__name__)

__all__ = ["persona_envelope_node"]


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
            r"\U0001F300-\U0001F5FF"   # symbols & pictographs
            r"\U0001F680-\U0001F6FF"   # transport & map
            r"\U0001F1E0-\U0001F1FF"   # flags
            r"\U00002702-\U000027B0"   # dingbats
            r"\U0000FE00-\U0000FE0F"   # variation selectors
            r"\U0000200D"              # zero-width joiner
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

    content = last_ai_msg.content if isinstance(last_ai_msg.content, str) else str(last_ai_msg.content)

    # Step 1: Output PII guardrail (always active, fail-closed)
    content = _validate_output(content)

    # Step 2: BrandingConfig enforcement
    branding = DEFAULT_CONFIG.get("branding", {})
    content = _enforce_branding(content, branding)

    # Step 3: Guest name injection
    guest_name = state.get("guest_name")
    content = _inject_guest_name(content, guest_name)

    # Step 4: SMS truncation (only when max_chars > 0)
    if max_chars > 0 and len(content) > max_chars:
        content = content[: max_chars - 3] + "..."

    # Only return messages update if content actually changed
    original = last_ai_msg.content if isinstance(last_ai_msg.content, str) else str(last_ai_msg.content)
    if content != original:
        return {"messages": [AIMessage(content=content)]}

    return {}
