"""Compliance gate node for the property Q&A graph.

Two-layer pre-router safety net:

**Layer 1 — Deterministic regex (zero LLM cost, zero latency):**
5 guardrail categories (prompt injection, responsible gaming, age verification,
BSA/AML, patron privacy) using 84 compiled regex patterns across 4 languages.

**Layer 2 — Semantic injection classifier (configurable LLM second layer):**
When ``SEMANTIC_INJECTION_ENABLED=True`` (default), an LLM-based classifier
runs on messages that pass ALL Layer 1 guardrails to catch sophisticated
injection attempts that evade pattern matching.  **Fails closed** on error
(returns synthetic injection classification to block suspicious input).
Disable via ``SEMANTIC_INJECTION_ENABLED=False`` to eliminate LLM cost/latency.

Returns ``query_type`` directly when a guardrail triggers, or ``None`` to
signal the downstream router should perform LLM classification.

Also includes the turn-limit guard (moved from ``router_node`` to centralize
all deterministic checks before the LLM router).
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.config import get_settings

from .guardrails import (
    audit_input,
    classify_injection_semantic,
    detect_age_verification,
    detect_bsa_aml,
    detect_patron_privacy,
    detect_responsible_gaming,
)
from .nodes import _get_last_human_message
from .state import PropertyQAState

logger = logging.getLogger(__name__)

__all__ = ["compliance_gate_node"]


async def compliance_gate_node(state: PropertyQAState) -> dict[str, Any]:
    """Run all deterministic guardrails before the LLM router.

    Priority order (first match wins):
    1. Turn-limit guard → off_topic (conversation too long)
    2. Empty message → greeting
    3. Prompt injection (regex) → off_topic
    4. Responsible gaming → gambling_advice
    5. Age verification → age_verification
    6. BSA/AML → off_topic
    7. Patron privacy → patron_privacy
    8. Semantic injection (LLM, fail-closed) → off_topic
    9. All pass → query_type=None (router does LLM classification)

    Semantic injection (step 8) runs AFTER all deterministic guardrails
    so that safety-critical responses (responsible gaming helplines,
    age verification, BSA/AML refusal) are never blocked by a semantic
    classifier failure.  On classifier error, the fail-closed behavior
    blocks ONLY messages that passed all deterministic checks — a genuine
    "unknown" that warrants caution.

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

    # 3. Prompt injection — regex (fast deterministic first layer)
    if not audit_input(user_message):
        return {"query_type": "off_topic", "router_confidence": 1.0}

    # 4. Responsible gaming (with session-level escalation)
    if detect_responsible_gaming(user_message):
        rg_count = state.get("responsible_gaming_count", 0) + 1
        if rg_count >= 3:
            logger.warning(
                "Responsible gaming escalation: %d triggers in session, "
                "adding live-support escalation",
                rg_count,
            )
        return {
            "query_type": "gambling_advice",
            "router_confidence": 1.0,
            "responsible_gaming_count": rg_count,
        }

    # 5. Age verification
    if detect_age_verification(user_message):
        return {"query_type": "age_verification", "router_confidence": 1.0}

    # 6. BSA/AML
    if detect_bsa_aml(user_message):
        return {"query_type": "off_topic", "router_confidence": 1.0}

    # 7. Patron privacy
    if detect_patron_privacy(user_message):
        return {"query_type": "patron_privacy", "router_confidence": 1.0}

    # 8. Semantic injection — LLM second layer (configurable, fail-closed).
    # Runs AFTER all deterministic guardrails to ensure safety-critical
    # responses (helplines, age info, BSA/AML refusal) are never blocked
    # by classifier failure.  On error, returns synthetic injection=True
    # which blocks only "clean" messages that no deterministic rule caught.
    if settings.SEMANTIC_INJECTION_ENABLED:
        semantic_result = await classify_injection_semantic(user_message)
        if (
            semantic_result
            and semantic_result.is_injection
            and semantic_result.confidence >= settings.SEMANTIC_INJECTION_THRESHOLD
        ):
            logger.warning(
                "Semantic injection detected (confidence=%.2f): %s",
                semantic_result.confidence,
                semantic_result.reason[:100],
            )
            return {
                "query_type": "off_topic",
                "router_confidence": semantic_result.confidence,
            }

    # 9. All guardrails passed — signal LLM router to classify
    return {"query_type": None, "router_confidence": 0.0}
