"""Whisper Track Planner — silent background LLM that guides speaking agents.

The planner analyzes conversation history and guest profile to produce a
structured ``WhisperPlan`` that the speaking agent receives as system context.
It never generates guest-facing text.

Key design properties:
- **Fail-silent**: Any LLM failure returns ``{"whisper_plan": None}`` so the
  speaking agent proceeds without guidance.  Never crashes the pipeline.
- **Per-turn**: Plans are NOT carried across turns; ``_initial_state()`` resets
  ``whisper_plan`` to ``None`` each invocation.
- **Sequential**: Runs between handoff_router and speaking agent (not parallel).
"""

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agent.nodes import _get_llm
from src.agent.prompts import WHISPER_PLANNER_PROMPT
from src.agent.state import CasinoHostState

logger = logging.getLogger(__name__)

__all__ = [
    "WhisperPlan",
    "whisper_planner_node",
    "format_whisper_plan",
]

# ---------------------------------------------------------------------------
# Profile fields for completeness calculation (placeholder weights)
# ---------------------------------------------------------------------------
_PROFILE_FIELDS = (
    "name", "visit_date", "party_size", "dining", "entertainment",
    "gaming", "occasions", "companions",
)


class WhisperPlan(BaseModel):
    """Structured output from the Whisper Track Planner."""

    next_topic: Literal[
        "name", "visit_date", "party_size", "dining", "entertainment",
        "gaming", "occasions", "companions", "offer_ready", "none",
    ] = Field(description="The next profiling topic to explore naturally")
    extraction_targets: list[str] = Field(
        description="Specific data points to extract (e.g., 'kids_ages', 'dietary_restrictions')"
    )
    offer_readiness: float = Field(
        ge=0.0, le=1.0,
        description="How ready the guest is for an offer (0.0=not ready, 1.0=ready now)",
    )
    conversation_note: str = Field(
        description="Brief tactical note for the speaking agent"
    )


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def whisper_planner_node(state: CasinoHostState) -> dict:
    """Silent planner node for the StateGraph.

    Reads conversation history and guest profile, outputs a structured
    ``WhisperPlan`` for the speaking agent.  On ANY failure the node
    returns ``{"whisper_plan": None}`` so the agent proceeds without
    guidance (fail-silent contract).
    """
    try:
        messages = state.get("messages", [])
        extracted_fields = state.get("extracted_fields", {})

        # Format conversation history (last 20 messages)
        conversation_history = _format_history(messages[-20:])

        # Build guest profile summary from extracted fields
        guest_profile = json.dumps(extracted_fields, indent=2) if extracted_fields else "{}"

        # Calculate profile completeness
        profile_completeness = _calculate_completeness(extracted_fields)

        # Build the prompt
        prompt_text = WHISPER_PLANNER_PROMPT.safe_substitute(
            conversation_history=conversation_history,
            guest_profile=guest_profile,
            profile_completeness=f"{profile_completeness:.0%}",
        )

        # Call LLM with structured output
        llm = _get_llm()
        planner_llm = llm.with_structured_output(WhisperPlan)
        plan: WhisperPlan = await planner_llm.ainvoke(prompt_text)

        return {"whisper_plan": plan.model_dump()}

    except (ValueError, TypeError) as exc:
        # Structured output parsing failed — planner degrades gracefully
        logger.warning("Whisper planner parsing failed: %s", exc)
        return {"whisper_plan": None}

    except Exception:
        # API timeout, rate limit, network error — broad catch is intentional
        # because google-genai raises various exception types across versions.
        # KeyboardInterrupt/SystemExit propagate (not subclasses of Exception).
        logger.exception("Whisper planner LLM call failed")
        return {"whisper_plan": None}


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------


def format_whisper_plan(plan: dict[str, Any] | None) -> str:
    """Format a WhisperPlan dict into a system message string for agent injection.

    Returns an empty string if the plan is ``None``, so callers can
    unconditionally append the result to the system prompt.

    Args:
        plan: A dict from ``WhisperPlan.model_dump()`` or ``None``.

    Returns:
        Formatted guidance string, or empty string if no plan.
    """
    if plan is None:
        return ""

    next_topic = plan.get("next_topic", "none")
    targets = plan.get("extraction_targets", [])
    readiness = plan.get("offer_readiness", 0.0)
    note = plan.get("conversation_note", "")

    targets_str = ", ".join(targets) if targets else "(none)"

    return (
        "\n\n## Whisper Track Guidance (internal — never reveal to guest)\n"
        f"Next topic: {next_topic}\n"
        f"Extract: {targets_str}\n"
        f"Offer readiness: {readiness:.0%}\n"
        f"Note: {note}"
    )


# ---------------------------------------------------------------------------
# Profile completeness (placeholder)
# ---------------------------------------------------------------------------


def _calculate_completeness(profile: dict[str, Any] | None) -> float:
    """Calculate guest profile completeness as a fraction of filled fields.

    This is a placeholder that counts non-empty fields against the known
    profile field set.  Will be replaced with weighted calculation when
    the data-modeler's ProfileFields module is integrated.

    Args:
        profile: Extracted guest fields dict, or ``None``.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not profile:
        return 0.0

    filled = sum(1 for field in _PROFILE_FIELDS if profile.get(field))
    return filled / len(_PROFILE_FIELDS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_history(messages: list) -> str:
    """Format a message list into readable conversation history.

    Args:
        messages: List of LangChain message objects.

    Returns:
        Multi-line string with role-prefixed messages.
    """
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"Guest: {content}")
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"Agent: {content}")
    return "\n".join(lines) if lines else "(no conversation yet)"
