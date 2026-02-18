"""Persona envelope node — optional SMS truncation layer.

Sits between ``validate`` (PASS) and ``respond`` in the v2 graph.
For web (``PERSONA_MAX_CHARS=0``): pass through unchanged.
For SMS (``PERSONA_MAX_CHARS=160``): truncate the last AI message.
"""

import logging

from langchain_core.messages import AIMessage

from src.config import get_settings

from .state import PropertyQAState

logger = logging.getLogger(__name__)

__all__ = ["persona_envelope_node"]


async def persona_envelope_node(state: PropertyQAState) -> dict:
    """Apply persona formatting to the final response.

    For web (PERSONA_MAX_CHARS=0): pass through unchanged.
    For SMS (PERSONA_MAX_CHARS>0): truncate last AI message with ellipsis.

    Args:
        state: The current graph state.

    Returns:
        Empty dict (passthrough) or dict with truncated messages.
    """
    settings = get_settings()
    max_chars = settings.PERSONA_MAX_CHARS

    if max_chars <= 0:
        # Web mode: no transformation
        return {}

    # SMS mode: truncate last AI message
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > max_chars:
                truncated = content[: max_chars - 3] + "..."
                return {"messages": [AIMessage(content=truncated)]}
            return {}
    return {}
