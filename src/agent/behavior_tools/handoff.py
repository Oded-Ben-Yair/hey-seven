"""Structured host handoff summary builder.

Aggregates guest profile, conversation history, risk flags, and
recommended actions into a structured summary for human casino hosts.

Enhances the existing HandoffRequest (src/agent/handoff.py) with richer
context: conversation narrative, key preferences, risk flags, recommended
actions, and urgency classification.

Pure business logic — no LLM calls, no I/O.

Targets P9 (Host Handoff): 2.1 → 5.0+
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "HandoffSummary",
    "build_handoff_summary",
    "format_handoff_for_prompt",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class HandoffSummary(BaseModel):
    """Structured handoff summary for human casino hosts."""

    guest_name: str | None = None
    profile_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    conversation_summary: str = ""
    key_preferences: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    urgency: Literal["routine", "priority", "urgent"] = "routine"
    handoff_reason: str = ""
    domains_discussed: list[str] = Field(default_factory=list)
    turn_count: int = 0


# ---------------------------------------------------------------------------
# Risk flag detection
# ---------------------------------------------------------------------------

_RISK_KEYWORDS = {
    "crisis": "Guest showed signs of crisis or emotional distress",
    "self_harm": "Self-harm indicators detected — handle with care",
    "responsible_gaming": "Responsible gaming concerns raised",
    "frustrated": "Guest expressed frustration during conversation",
    "complaint": "Guest had complaints about service or experience",
    "intoxicated": "Potential intoxication signals detected",
}


def _detect_risk_flags(
    state: dict[str, Any],
    messages: list,
) -> list[str]:
    """Detect risk flags from state and conversation history."""
    flags: list[str] = []

    if state.get("crisis_active"):
        flags.append(_RISK_KEYWORDS["crisis"])

    sentiment = state.get("guest_sentiment")
    if sentiment == "frustrated":
        flags.append(_RISK_KEYWORDS["frustrated"])

    # Check for responsible gaming mentions in recent messages
    rg_count = state.get("responsible_gaming_count", 0)
    if rg_count > 0:
        flags.append(
            f"Responsible gaming topic raised {rg_count} time(s) — "
            "review conversation before engaging"
        )

    # Check message content for complaint signals
    for msg in messages[-6:]:
        if isinstance(msg, HumanMessage):
            from src.agent.nodes import _normalize_content

            content = _normalize_content(msg.content).lower()
            if any(
                w in content
                for w in ("complaint", "manager", "unacceptable", "terrible", "awful")
            ):
                if _RISK_KEYWORDS["complaint"] not in flags:
                    flags.append(_RISK_KEYWORDS["complaint"])
                break

    return flags


# ---------------------------------------------------------------------------
# Conversation summary builder
# ---------------------------------------------------------------------------


def _build_conversation_narrative(
    messages: list,
    extracted: dict[str, Any],
    domains: list[str],
) -> str:
    """Build a 3-5 sentence narrative summary of the conversation."""
    from src.agent.nodes import _normalize_content

    parts: list[str] = []

    # Guest identity
    name = extracted.get("name")
    party_size = extracted.get("party_size")
    occasion = extracted.get("occasion")

    if name:
        intro = f"Guest {name}"
    else:
        intro = "Guest"

    if occasion:
        intro += f" (visiting for {occasion})"
    if party_size:
        intro += f" with a party of {party_size}"
    parts.append(f"{intro}.")

    # Topics discussed
    if domains:
        domain_str = ", ".join(domains[:5])
        parts.append(f"Discussed: {domain_str}.")

    # Key asks from recent messages
    human_messages = [
        _normalize_content(m.content) for m in messages if isinstance(m, HumanMessage)
    ]
    if human_messages:
        last_ask = human_messages[-1]
        if len(last_ask) > 100:
            last_ask = last_ask[:97] + "..."
        parts.append(f'Last request: "{last_ask}"')

    # Preferences
    prefs: list[str] = []
    if extracted.get("preferences"):
        prefs.append(f"dining: {extracted['preferences']}")
    if extracted.get("gaming"):
        prefs.append(f"gaming: {extracted['gaming']}")
    if extracted.get("dietary"):
        prefs.append(f"dietary: {extracted['dietary']}")
    if prefs:
        parts.append(f"Preferences: {', '.join(prefs)}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Recommended actions builder
# ---------------------------------------------------------------------------


def _build_recommended_actions(
    state: dict[str, Any],
    extracted: dict[str, Any],
    risk_flags: list[str],
) -> list[str]:
    """Build recommended actions for the receiving host."""
    actions: list[str] = []

    # Crisis-related actions
    if state.get("crisis_active"):
        actions.append("Prioritize guest welfare — follow responsible gaming protocol")
        actions.append("Have supervisor review conversation transcript")

    sentiment = state.get("guest_sentiment")

    # Frustrated guest actions
    if sentiment == "frustrated":
        actions.append(
            "Acknowledge previous frustration before addressing new requests"
        )
        actions.append("Consider offering a service recovery gesture")

    # Occasion-related actions
    occasion = extracted.get("occasion")
    if occasion:
        actions.append(f"Review comp eligibility for {occasion} celebration")

    # Loyalty-related actions
    loyalty = extracted.get("loyalty_tier") or extracted.get("loyalty_signal")
    if loyalty:
        actions.append(f"Check loyalty status: {loyalty}")
    else:
        actions.append("Offer rewards program enrollment if not already a member")

    # If no specific actions, add generic
    if not actions:
        actions.append("Review conversation history for context before engaging")
        actions.append("Greet guest by name if available")

    return actions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_handoff_summary(
    state: dict[str, Any],
    handoff_reason: str = "",
) -> HandoffSummary:
    """Build a structured handoff summary from graph state.

    Aggregates data from extracted_fields, messages, crisis state,
    sentiment, and domains_discussed into actionable intelligence
    for human casino hosts.

    Args:
        state: Current LangGraph state dict.
        handoff_reason: Why the AI is handing off.

    Returns:
        HandoffSummary with all fields populated.
    """
    extracted = state.get("extracted_fields") or {}
    messages = state.get("messages", [])
    domains = state.get("domains_discussed", [])
    completeness = state.get("profile_completeness_score", 0.0)

    # Count human turns
    turn_count = sum(1 for m in messages if isinstance(m, HumanMessage))

    # Risk flags
    risk_flags = _detect_risk_flags(state, messages)

    # Urgency classification
    urgency: Literal["routine", "priority", "urgent"] = "routine"
    if state.get("crisis_active"):
        urgency = "urgent"
    elif state.get("guest_sentiment") in ("frustrated", "negative"):
        urgency = "priority"
    elif any("complaint" in f.lower() for f in risk_flags):
        urgency = "priority"

    # Build narrative
    narrative = _build_conversation_narrative(messages, extracted, domains)

    # Key preferences
    key_prefs: list[str] = []
    for field in ("preferences", "dietary", "gaming", "entertainment", "spa"):
        val = extracted.get(field)
        if val:
            key_prefs.append(f"{field}: {val}")

    # Recommended actions
    actions = _build_recommended_actions(state, extracted, risk_flags)

    return HandoffSummary(
        guest_name=extracted.get("name"),
        profile_completeness=completeness,
        conversation_summary=narrative,
        key_preferences=key_prefs,
        risk_flags=risk_flags,
        recommended_actions=actions,
        urgency=urgency,
        handoff_reason=handoff_reason,
        domains_discussed=list(domains),
        turn_count=turn_count,
    )


def format_handoff_for_prompt(summary: HandoffSummary) -> str:
    """Format a HandoffSummary into a prompt section for the agent.

    Injects handoff context into the agent system prompt so the AI
    can naturally transition the conversation to a human host with
    appropriate context sharing.

    Args:
        summary: The handoff summary to format.

    Returns:
        Formatted prompt section string.
    """
    lines: list[str] = ["\n\n## Handoff Preparation"]

    lines.append(
        "You are preparing to hand this guest off to a human host. "
        "Share a brief summary of what you've discussed so the guest "
        "doesn't have to repeat themselves."
    )

    if summary.guest_name:
        lines.append(f"- Guest: {summary.guest_name}")

    if summary.conversation_summary:
        lines.append(f"- Context: {summary.conversation_summary}")

    if summary.key_preferences:
        pref_str = "; ".join(summary.key_preferences[:4])
        lines.append(f"- Preferences noted: {pref_str}")

    if summary.risk_flags:
        lines.append("- **Important context for host**:")
        for flag in summary.risk_flags[:3]:
            lines.append(f"  - {flag}")

    if summary.recommended_actions:
        lines.append("- Suggested next steps:")
        for action in summary.recommended_actions[:3]:
            lines.append(f"  - {action}")

    lines.append(
        f"\nUrgency: {summary.urgency}. Turns in conversation: {summary.turn_count}."
    )

    return "\n".join(lines)
