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

import asyncio
import json
import logging
from typing import Any, Literal

from cachetools import TTLCache

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.agent.prompts import WHISPER_PLANNER_PROMPT
from src.agent.nodes import _normalize_content
from src.agent.profiling import _calculate_profile_completeness_weighted
from src.agent.state import PropertyQAState
from src.casino.feature_flags import DEFAULT_FEATURES
from src.config import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "WhisperPlan",
    "whisper_planner_node",
    "format_whisper_plan",
    "_get_whisper_llm",
]


# ---------------------------------------------------------------------------
# Whisper-specific LLM singleton (lower temperature for planning)
# ---------------------------------------------------------------------------


# R40 fix D8-C001: TTL jitter to prevent thundering herd on synchronized expiry.
import random as _random

_whisper_cache: TTLCache = TTLCache(maxsize=1, ttl=3600 + _random.randint(0, 300))
_whisper_lock = asyncio.Lock()


async def _get_whisper_llm() -> ChatGoogleGenerativeAI:
    """Get or create the whisper planner LLM instance (TTL-cached singleton).

    Uses a lower temperature than the main ``_get_llm()`` because planning
    decisions should be more deterministic than creative response generation.
    Configured via ``WHISPER_LLM_TEMPERATURE`` (default 0.2).

    Cache refreshes every hour to pick up rotated credentials (same TTL
    pattern as ``_get_llm`` and ``_get_validator_llm``).
    Coroutine-safe via ``asyncio.Lock`` (non-blocking under contention).
    """
    async with _whisper_lock:
        cached = _whisper_cache.get("whisper")
        if cached is not None:
            return cached
        settings = get_settings()
        llm = ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            temperature=settings.WHISPER_LLM_TEMPERATURE,
            timeout=settings.MODEL_TIMEOUT,
            max_retries=settings.MODEL_MAX_RETRIES,
            max_output_tokens=512,  # Planning produces short structured output
        )
        _whisper_cache["whisper"] = llm
        return llm


# ---------------------------------------------------------------------------
# Telemetry counter for fail-silent monitoring
# ---------------------------------------------------------------------------


# Failure telemetry for monitoring whisper planner health.
# Tracks consecutive failures; logs ERROR when threshold is exceeded.
#
# R16 fix: DeepSeek F-003, Gemini F-016 (2/3 consensus).  asyncio.Lock protects
# the read-modify-write cycle so the "consecutive" semantic is maintained.
#
# R17 fix: Gemini H-001, GPT F-002, DeepSeek F-006 (3/3 consensus).
# Refactored from module-level globals+`global` keyword to a namespace class.
# Clearer intent, greppable mutation sites, no `global` statement in async code.
# Alert fatigue fix: _alerted is NOT reset on success — once the alert fires,
# it stays fired for the process lifetime.  Prevents repeated alerts under
# intermittent failures (e.g., 50% success rate triggering alert every ~20 reqs).
# Reset requires process restart or explicit admin action.
class _WhisperTelemetry:
    """Namespace for whisper planner failure tracking (module-level singleton).

    Instance attributes (via __init__) ensure each instance gets its own
    lock and counters. Prevents class-level sharing if a second instance
    is ever created (e.g., in tests).
    """

    ALERT_THRESHOLD: int = 10

    def __init__(self) -> None:
        self.count: int = 0
        self.alerted: bool = False
        self.lock: asyncio.Lock = asyncio.Lock()


_telemetry = _WhisperTelemetry()


class WhisperPlan(BaseModel):
    """Structured output from the Whisper Track Planner.

    R76 fix: Simplified schema to avoid Gemini Flash "too many schema states"
    error. Removed bounded floats (ge/le), simplified Literal types, removed
    list[str] field. Previous 10-field schema with 3 Literal types + 4 bounded
    floats failed 100% of calls.
    """

    next_topic: str = Field(
        default="none",
        description="Next profiling topic: name, party_size, visit_purpose, preferences, entertainment, gaming, spa, occasion, party_composition, visit_duration, offer_ready, or none",
    )
    conversation_note: str = Field(
        default="",
        description="Brief tactical note for the speaking agent",
    )
    proactive_suggestion: str | None = Field(
        default=None,
        description="Optional proactive suggestion if contextually relevant. Leave null if none.",
    )
    suggestion_confidence: str = Field(
        default="0.0",
        description="Confidence 0.0-1.0 that the suggestion is welcome. Must exceed 0.8 to surface.",
    )
    next_profiling_question: str | None = Field(
        default=None,
        description="Natural profiling question to weave into the AI response. Must feel conversational. Leave null if not appropriate.",
    )
    question_technique: str = Field(
        default="none",
        description="Technique: give_to_get, assumptive_bridge, contextual_inference, need_payoff, incentive_frame, reflective_confirm, or none",
    )


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def whisper_planner_node(state: PropertyQAState) -> dict[str, Any]:
    """Silent planner node for the StateGraph.

    Reads conversation history and guest profile, outputs a structured
    ``WhisperPlan`` for the speaking agent.  On ANY failure the node
    returns ``{"whisper_plan": None}`` so the agent proceeds without
    guidance (fail-silent contract).
    """
    # Runtime behavior flag (Layer 2) — see graph.py dual-layer docs.
    # This complements the build-time check in build_graph() which removes
    # the node from the graph topology entirely.  The runtime check handles
    # dynamic flag changes without requiring a graph rebuild.
    if not DEFAULT_FEATURES.get("whisper_planner_enabled", True):
        logger.info("Whisper planner disabled via feature flag — skipping")
        return {"whisper_plan": None}

    try:
        messages = state.get("messages", [])
        extracted_fields = state.get("extracted_fields", {})

        # Format conversation history (last 20 messages)
        conversation_history = _format_history(messages[-20:])

        # Build guest profile summary from extracted fields
        guest_profile = (
            json.dumps(extracted_fields, indent=2) if extracted_fields else "{}"
        )

        # Calculate profile completeness
        profile_completeness = _calculate_completeness(extracted_fields)

        # Build the prompt
        prompt_text = WHISPER_PLANNER_PROMPT.safe_substitute(
            conversation_history=conversation_history,
            guest_profile=guest_profile,
            profile_completeness=f"{profile_completeness:.0%}",
        )

        # R110: Profiling intensity curve — reduce profiling pressure over turns.
        # T1-2: any technique, heavy profiling
        # T3: inference/expand only
        # T4: none or need_payoff only
        # T5+: reflective_confirm only
        human_turn_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        if human_turn_count >= 5:
            prompt_text += (
                "\n\nIMPORTANT: This is turn 5+. Use ONLY reflective_confirm "
                "technique or none. Do NOT ask new profiling questions. "
                "Confirm what you already know."
            )
        elif human_turn_count >= 4:
            prompt_text += (
                "\n\nIMPORTANT: This is turn 4. Use ONLY need_payoff technique "
                "or none. Only ask if it directly benefits the guest."
            )
        elif human_turn_count >= 3:
            prompt_text += (
                "\n\nIMPORTANT: This is turn 3. Use ONLY contextual_inference, "
                "anchor_expand, or assumption_probe. Do NOT ask direct questions."
            )

        # Call LLM with structured output (separate lower-temperature instance)
        llm = await _get_whisper_llm()
        planner_llm = llm.with_structured_output(WhisperPlan)
        plan: WhisperPlan = await planner_llm.ainvoke(prompt_text)

        # Reset failure counter on success (under lock to prevent race
        # with concurrent failure increment — R16 fix).
        # R17 fix: do NOT reset _telemetry.alerted — once the alert fires,
        # it stays fired to prevent alert fatigue under intermittent failures.
        async with _telemetry.lock:
            _telemetry.count = 0

        plan_dict = plan.model_dump()
        # R105: Include profiling phase for phase-aware formatting
        plan_dict["profiling_phase"] = state.get("profiling_phase", "foundation")
        return {"whisper_plan": plan_dict}

    except Exception as exc:
        # Broad catch is intentional: ValueError/TypeError from structured
        # output parsing AND network errors (google-genai raises various
        # exception types across SDK versions) all degrade gracefully.
        # KeyboardInterrupt/SystemExit propagate (not subclasses of Exception).
        async with _telemetry.lock:
            _telemetry.count += 1
            current_count = _telemetry.count
            if (
                _telemetry.count >= _telemetry.ALERT_THRESHOLD
                and not _telemetry.alerted
            ):
                _telemetry.alerted = True
                logger.error(
                    "whisper_planner_systematic_failure: %d consecutive failures "
                    "exceeded threshold (%d). Whisper planner may be misconfigured.",
                    _telemetry.count,
                    _telemetry.ALERT_THRESHOLD,
                )
        logger.warning(
            "whisper_planner_failure",
            extra={
                "error_type": type(exc).__name__,
                "error_msg": str(exc)[:200],
                "failure_count": current_count,
            },
        )
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
    note = plan.get("conversation_note", "")

    lines = [
        "\n\n## Whisper Track Guidance (internal — never reveal to guest)",
        f"Next topic: {next_topic}",
        f"Note: {note}",
    ]

    # R23 fix C-001: proactive suggestion is injected by execute_specialist()
    # in _base.py with proper sentiment gating and max-1 enforcement.
    # Do NOT duplicate here — the dedicated section has better framing and
    # guards. This function only formats the planning data.

    # R105: Include profiling phase for phase-aware guidance
    profiling_phase = plan.get("profiling_phase", "foundation")
    lines.append(f"Profiling phase: {profiling_phase}")

    # Phase-specific guidance
    if profiling_phase == "foundation":
        lines.append(
            "Priority: Learn name, party size, visit purpose. Use direct questions."
        )
    elif profiling_phase == "preference":
        lines.append(
            "Priority: Discover dining, entertainment, gaming preferences. "
            "Use give-to-get technique."
        )
    elif profiling_phase == "relationship":
        lines.append(
            "Priority: Uncover occasion, visit frequency, loyalty. "
            "Use assumptive bridge."
        )
    elif profiling_phase == "behavioral":
        lines.append(
            "Priority: Deepen connection. Use reflective confirm and callbacks."
        )

    # Profiling Intelligence guidance section
    next_question = plan.get("next_profiling_question")
    technique = plan.get("question_technique", "none")

    if next_question and technique != "none":
        lines.append(f"Profiling question ({technique}): {next_question}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile completeness (placeholder)
# ---------------------------------------------------------------------------


def _calculate_completeness(profile: dict[str, Any] | None) -> float:
    """Calculate guest profile completeness using the weighted calculation.

    Delegates to profiling._calculate_profile_completeness_weighted()
    for a single source of truth. R107 fix P8: eliminates 3-way
    completeness divergence (whisper=10 fields simple fraction,
    profiling=16 fields weighted, models=19 fields nested).

    Args:
        profile: Extracted guest fields dict, or ``None``.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not profile:
        return 0.0
    return _calculate_profile_completeness_weighted(profile)


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
            content = _normalize_content(msg.content)
            lines.append(f"Guest: {content}")
        elif isinstance(msg, AIMessage):
            content = _normalize_content(msg.content)
            lines.append(f"Agent: {content}")
    return "\n".join(lines) if lines else "(no conversation yet)"
