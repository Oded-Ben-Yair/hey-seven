"""Deterministic compliance gate node for the property Q&A graph.

Runs all 5 guardrail layers (prompt injection, responsible gaming, age
verification, BSA/AML, patron privacy) as a single pre-router node.  Zero
LLM calls — pure regex-based classification.  Returns ``query_type``
directly when a guardrail triggers, or ``None`` to signal the downstream
router should perform LLM classification.

Also includes the turn-limit guard (moved from ``router_node`` to centralize
all deterministic checks before the LLM router).
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.config import get_settings

from .guardrails import (
    audit_input,
    detect_age_verification,
    detect_bsa_aml,
    detect_patron_privacy,
    detect_responsible_gaming,
)
from .state import PropertyQAState

logger = logging.getLogger(__name__)

__all__ = ["compliance_gate_node"]


def _get_last_human_message(messages: list) -> str:
    """Extract the content of the last HumanMessage from a message list."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


async def compliance_gate_node(state: PropertyQAState) -> dict[str, Any]:
    """Run all deterministic guardrails before the LLM router.

    Priority order (first match wins):
    1. Turn-limit guard → off_topic (conversation too long)
    2. Empty message → greeting
    3. Prompt injection → off_topic
    4. Responsible gaming → gambling_advice
    5. Age verification → age_verification
    6. BSA/AML → off_topic
    7. Patron privacy → patron_privacy
    8. All pass → query_type=None (router does LLM classification)

    Args:
        state: The current graph state.

    Returns:
        Dict with ``query_type`` and ``router_confidence`` updates.
        ``query_type=None`` signals the downstream router to classify via LLM.
    """
    settings = get_settings()
    messages = state.get("messages", [])

    # 1. Turn-limit guard
    if len(messages) > settings.MAX_MESSAGE_LIMIT:
        logger.warning(
            "Message limit exceeded (%d messages), forcing off_topic",
            len(messages),
        )
        return {"query_type": "off_topic", "router_confidence": 1.0}

    # 2. Empty message → greeting
    user_message = _get_last_human_message(messages)
    if not user_message:
        return {"query_type": "greeting", "router_confidence": 1.0}

    # 3. Prompt injection (audit_input returns False when injection detected)
    if not audit_input(user_message):
        return {"query_type": "off_topic", "router_confidence": 1.0}

    # 4. Responsible gaming
    if detect_responsible_gaming(user_message):
        return {"query_type": "gambling_advice", "router_confidence": 1.0}

    # 5. Age verification
    if detect_age_verification(user_message):
        return {"query_type": "age_verification", "router_confidence": 1.0}

    # 6. BSA/AML
    if detect_bsa_aml(user_message):
        return {"query_type": "off_topic", "router_confidence": 1.0}

    # 7. Patron privacy
    if detect_patron_privacy(user_message):
        return {"query_type": "patron_privacy", "router_confidence": 1.0}

    # 8. All guardrails passed — signal LLM router to classify
    return {"query_type": None, "router_confidence": 0.0}
