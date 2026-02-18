"""Persona envelope node — output guardrails + optional SMS truncation layer.

Sits between ``validate`` (PASS) and ``respond`` in the v2 graph.

Output guardrails (always active):
- PII redaction: catches accidental PII leakage in LLM-generated responses.

Channel formatting:
- Web (``PERSONA_MAX_CHARS=0``): pass through after guardrails.
- SMS (``PERSONA_MAX_CHARS=160``): truncate last AI message with ellipsis.
"""

import logging

from langchain_core.messages import AIMessage

from src.api.pii_redaction import redact_pii
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


async def persona_envelope_node(state: PropertyQAState) -> dict:
    """Apply output guardrails and persona formatting to the final response.

    Processing order:
    1. Output PII guardrail (always active)
    2. SMS truncation (when PERSONA_MAX_CHARS > 0)

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

    # Step 1: Output PII guardrail (always active)
    content = _validate_output(content)

    # Step 2: SMS truncation (only when max_chars > 0)
    if max_chars > 0 and len(content) > max_chars:
        content = content[: max_chars - 3] + "..."

    # Only return messages update if content actually changed
    original = last_ai_msg.content if isinstance(last_ai_msg.content, str) else str(last_ai_msg.content)
    if content != original:
        return {"messages": [AIMessage(content=content)]}

    return {}
