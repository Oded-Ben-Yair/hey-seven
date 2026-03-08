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
    """Structured handoff summary for human casino hosts.

    R103 fix P9: Added guest_stated_facts/agent_inferences partition
    and next_actions for actionable follow-up.
    """

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
    # R103 fix P9: Stated vs inferred partition
    guest_stated_facts: list[str] = Field(default_factory=list)
    agent_inferences: list[str] = Field(default_factory=list)
    # R103 fix P9: Concrete next actions for receiving host
    next_actions: list[str] = Field(default_factory=list)


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
    "repeated_question": "Guest repeated the same question — AI may not have answered adequately",
}


def _detect_risk_flags(
    state: dict[str, Any],
    messages: list,
) -> list[str]:
    """Detect risk flags from state and conversation history."""
    from src.agent.nodes import _normalize_content

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

    # R100 fix P9: Check for repeated questions (same question asked twice)
    human_msgs = [
        _normalize_content(m.content).lower().strip()
        for m in messages
        if isinstance(m, HumanMessage)
    ]
    if len(human_msgs) >= 2:
        # Check if any of the last 3 messages repeats an earlier one
        recent = human_msgs[-3:]
        earlier = human_msgs[:-3] if len(human_msgs) > 3 else []
        for msg_text in recent:
            if len(msg_text) > 10 and (
                msg_text in earlier or human_msgs.count(msg_text) > 1
            ):
                flags.append(_RISK_KEYWORDS["repeated_question"])
                break

    # Check message content for complaint signals
    for msg in messages[-6:]:
        if isinstance(msg, HumanMessage):
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
# R103 fix P9: Stated vs inferred partition
# ---------------------------------------------------------------------------

# Fields that are typically stated explicitly by the guest
_STATED_FIELDS = frozenset(
    {
        "name",
        "party_size",
        "occasion",
        "visit_date",
        "visit_duration",
        "preferences",
        "dietary",
        "loyalty_tier",
        "loyalty_signal",
    }
)

# Fields that are typically inferred from context by the agent
_INFERRED_FIELDS = frozenset(
    {
        "visit_purpose",
        "party_composition",
        "budget_signal",
        "gaming",
        "entertainment",
        "spa",
        "home_market",
        "visit_frequency",
        "occasion_details",
        "urgency",
        "fatigue",
        "budget_conscious",
    }
)


def _partition_stated_inferred(
    extracted: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Partition extracted fields into guest-stated facts and agent inferences.

    Uses field-name heuristics: fields like name, party_size, occasion are
    typically stated explicitly. Fields like visit_purpose, party_composition,
    budget_signal are typically inferred from context.

    Args:
        extracted: Accumulated extracted fields dict.

    Returns:
        Tuple of (stated_facts, inferences) as human-readable strings.
    """
    stated: list[str] = []
    inferred: list[str] = []

    for field, value in extracted.items():
        if value is None or value == "" or isinstance(value, bool):
            continue

        label = field.replace("_", " ").title()
        entry = f"{label}: {value}"

        if field in _STATED_FIELDS:
            stated.append(entry)
        elif field in _INFERRED_FIELDS:
            inferred.append(entry)
        else:
            # Unknown field — default to inferred (safer)
            inferred.append(entry)

    return stated, inferred


def _build_next_actions(
    state: dict[str, Any],
    extracted: dict[str, Any],
    risk_flags: list[str],
    domains: list[str],
) -> list[str]:
    """Build concrete next actions for the receiving host.

    R103 fix P9: Conversation-specific actions instead of generic templates.

    Args:
        state: Current graph state.
        extracted: Extracted profile fields.
        risk_flags: Detected risk flags.
        domains: Domains discussed in conversation.

    Returns:
        List of actionable next steps.
    """
    actions: list[str] = []

    # Crisis/safety first
    if state.get("crisis_active"):
        actions.append(
            "IMMEDIATE: Review conversation for crisis indicators, follow RG protocol"
        )

    # Name-based personalization
    name = extracted.get("name")
    if name:
        actions.append(
            f"Greet as {name} — guest shared their name during AI conversation"
        )
    else:
        actions.append("Introduce yourself and ask for guest's name")

    # Occasion-specific actions
    occasion = extracted.get("occasion")
    if occasion:
        actions.append(f"Check comp eligibility for {occasion} celebration")

    # Unresolved requests — check if last message was a question
    from src.agent.nodes import _normalize_content

    messages = state.get("messages", [])
    last_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = _normalize_content(msg.content)
            break
    if last_human and "?" in last_human:
        actions.append(f'Follow up on unanswered question: "{last_human[:80]}"')

    # Domain-specific follow-ups
    if "dining" in domains and not extracted.get("preferences"):
        actions.append(
            "Ask about dining preferences — guest discussed dining but no specifics captured"
        )
    if "hotel" in domains:
        actions.append("Verify room reservation status and any upgrade opportunities")

    # Loyalty/comp follow-up
    if extracted.get("loyalty_signal") or extracted.get("loyalty_tier"):
        tier = extracted.get("loyalty_tier") or extracted.get("loyalty_signal")
        actions.append(f"Pull up loyalty record: guest mentioned '{tier}'")
    elif state.get("guest_sentiment") == "frustrated":
        actions.append(
            "Consider service recovery gesture — guest expressed frustration"
        )

    return actions[:5]  # Cap at 5 actions


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

    # R103 fix P9: Stated vs inferred partition
    stated, inferred = _partition_stated_inferred(extracted)

    # R103 fix P9: Concrete next actions
    next_actions = _build_next_actions(state, extracted, risk_flags, domains)

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
        guest_stated_facts=stated,
        agent_inferences=inferred,
        next_actions=next_actions,
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

    # R103 fix P9: Stated vs inferred partition
    if summary.guest_stated_facts:
        lines.append("- **Guest told us** (confirmed facts):")
        for fact in summary.guest_stated_facts[:5]:
            lines.append(f"  - {fact}")

    if summary.agent_inferences:
        lines.append("- **We inferred** (verify with guest):")
        for inference in summary.agent_inferences[:5]:
            lines.append(f"  - {inference}")

    if summary.recommended_actions:
        lines.append("- Suggested next steps:")
        for action in summary.recommended_actions[:3]:
            lines.append(f"  - {action}")

    # R103 fix P9: Concrete next actions
    if summary.next_actions:
        lines.append("- **Your first actions**:")
        for action in summary.next_actions[:5]:
            lines.append(f"  - {action}")

    lines.append(
        f"\nUrgency: {summary.urgency}. Turns in conversation: {summary.turn_count}."
    )

    return "\n".join(lines)
