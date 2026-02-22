"""Shared specialist agent execution logic.

Extracts the ~80% duplicated code from the 4 specialist agents
(host, dining, entertainment, comp) into a single ``execute_specialist()``
function. Each agent file becomes a thin wrapper that passes its unique
system prompt template and configuration.

Dependency injection (``get_llm_fn``, ``get_cb_fn``) preserves existing
test mock paths -- tests that ``patch("src.agent.agents.host_agent._get_llm")``
continue to work because the agent passes its own module-level reference.
"""

import asyncio
import logging
from collections.abc import Callable
from string import Template

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.nodes import _format_context_block
from src.agent.prompts import (
    SENTIMENT_TONE_GUIDES,
    get_persona_style,
    get_responsible_gaming_helplines,
)
from src.agent.state import PropertyQAState
from src.agent.whisper_planner import format_whisper_plan
from src.config import get_settings

logger = logging.getLogger(__name__)

# LLM concurrency backpressure: limits concurrent LLM API calls across
# all specialist agents. Prevents: (1) LLM provider 429 rate limiting
# cascading into circuit breaker trips, (2) httpx connection pool
# exhaustion, (3) memory pressure from concurrent response buffering.
# Gemini API typical QPS limits: 60-300 RPM for Flash, so 20 concurrent
# is conservative. R5 fix per Gemini F3 analysis.
_LLM_SEMAPHORE = asyncio.Semaphore(20)


def _fallback_message(reason: str = "trouble generating a response") -> str:
    """Build a fallback message with property contact info.

    Single source of truth for all specialist agent fallback messages.
    Prevents duplication across circuit breaker, ValueError, and network
    error handlers.
    """
    settings = get_settings()
    return Template(
        "I apologize, but I'm having $reason. "
        "Please try again, or contact $property_name directly at $property_phone."
    ).safe_substitute(
        reason=reason,
        property_name=settings.PROPERTY_NAME,
        property_phone=settings.PROPERTY_PHONE,
    )


async def execute_specialist(
    state: PropertyQAState,
    *,
    agent_name: str,
    system_prompt_template: Template,
    context_header: str,
    no_context_fallback: str,
    get_llm_fn: Callable,
    get_cb_fn: Callable,
    include_whisper: bool = False,
) -> dict:
    """Execute the shared specialist agent logic.

    Args:
        state: Current graph state.
        agent_name: Agent name for logging (e.g. "dining", "host").
        system_prompt_template: string.Template with $property_name,
            $current_time, ${responsible_gaming_helplines} placeholders.
        context_header: Label for the retrieved context section
            (e.g. "Retrieved Dining Knowledge Base Context").
        no_context_fallback: Pre-formatted fallback message for empty
            retrieved context. Property placeholders ($property_name etc.)
            must already be substituted by the caller.
        get_llm_fn: Callable that returns the LLM instance (injected for testability).
        get_cb_fn: Callable that returns the CircuitBreaker instance (injected for testability).
        include_whisper: If True, append whisper planner guidance to the system prompt.

    Returns:
        Dict with ``messages`` (and optionally ``skip_validation``).
    """
    settings = get_settings()
    retrieved = state.get("retrieved_context", [])
    current_time = state.get("current_time", "unknown")
    retry_count = state.get("retry_count", 0)
    retry_feedback = state.get("retry_feedback")

    # Cache circuit breaker instance — avoids repeated get_cb_fn() calls.
    # R15 fix: await for async _get_circuit_breaker() with lock protection.
    cb = await get_cb_fn()

    # Circuit breaker check -- early exit before building prompts.
    # Uses lock-protected allow_request() instead of is_open property
    # to ensure atomic state transition (open -> half_open -> probe).
    if not await cb.allow_request():
        logger.warning("Circuit breaker open — %s agent returning fallback", agent_name)
        return {
            "messages": [AIMessage(content=_fallback_message("temporary technical difficulties"))],
            "skip_validation": True,
        }

    system_prompt = system_prompt_template.safe_substitute(
        property_name=settings.PROPERTY_NAME,
        current_time=current_time,
        responsible_gaming_helplines=get_responsible_gaming_helplines(),
    )

    # Format and append retrieved context
    if retrieved:
        context_block = _format_context_block(retrieved)
        system_prompt += Template(
            "\n\n## $header\n$context"
        ).safe_substitute(header=context_header, context=context_block)
    else:
        # No context found -- return domain-specific fallback
        return {
            "messages": [AIMessage(content=no_context_fallback)],
            "skip_validation": True,
        }

    # Phase 3: Inject guest context from profile (when non-empty)
    guest_context = state.get("guest_context", {})
    if guest_context:
        context_parts = []
        if guest_context.get("name"):
            context_parts.append(f"- Guest name: {guest_context['name']}")
        if guest_context.get("party_size"):
            context_parts.append(f"- Party size: {guest_context['party_size']}")
        if guest_context.get("visit_date"):
            context_parts.append(f"- Visit date: {guest_context['visit_date']}")
        if guest_context.get("preferences"):
            context_parts.append(f"- Preferences: {guest_context['preferences']}")
        if guest_context.get("occasion"):
            context_parts.append(f"- Occasion: {guest_context['occasion']}")
        if context_parts:
            system_prompt += "\n\n## Guest Context\n" + "\n".join(context_parts)

    # Inject whisper planner guidance
    if include_whisper:
        whisper_guidance = format_whisper_plan(state.get("whisper_plan"))
        if whisper_guidance:
            system_prompt += whisper_guidance

    # Phase 3: Inject persona style from BrandingConfig (fail-silent)
    try:
        from src.casino.config import DEFAULT_CONFIG
        branding = DEFAULT_CONFIG.get("branding", {})
        persona_style = get_persona_style(branding)
        if persona_style:
            system_prompt += persona_style
    except Exception:
        logger.debug("Persona style injection failed, continuing without", exc_info=True)

    # Phase 3: Inject sentiment-adaptive tone guidance
    guest_sentiment = state.get("guest_sentiment")
    if guest_sentiment:
        tone_guide = SENTIMENT_TONE_GUIDES.get(guest_sentiment, "")
        if tone_guide:
            system_prompt += f"\n\n## Tone Guidance\n{tone_guide}"

    # Build message list
    llm_messages = [SystemMessage(content=system_prompt)]

    # On retry, inject feedback
    if retry_count > 0 and retry_feedback:
        llm_messages.append(SystemMessage(
            content=Template(
                "IMPORTANT: Your previous response failed validation. Reason: $feedback. "
                "Please generate a corrected response that addresses this issue."
            ).safe_substitute(feedback=retry_feedback)
        ))

    # Sliding window on conversation history.
    # On retry, exclude the last AIMessage (the one that failed validation)
    # to prevent the LLM from parroting the invalid response.
    history = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
    if retry_count > 0 and history and isinstance(history[-1], AIMessage):
        history = history[:-1]
    window = history[-settings.MAX_HISTORY_MESSAGES:]
    llm_messages.extend(window)

    llm = await get_llm_fn()

    try:
        async with _LLM_SEMAPHORE:
            response = await llm.ainvoke(llm_messages)
        await cb.record_success()
        content = response.content if isinstance(response.content, str) else str(response.content)
        return {"messages": [AIMessage(content=content)]}
    except (ValueError, TypeError) as exc:
        await cb.record_failure()
        logger.warning("%s agent LLM response parsing failed: %s", agent_name.capitalize(), exc)
        # ValueError/TypeError may produce malformed content that still
        # warrants validation.  Incrementing retry_count ensures the
        # validator runs; the validate_node's "retry_count < 1" check
        # prevents unbounded retries.  Only circuit-breaker-open and
        # network-error paths use skip_validation=True.
        # R10 fix (DeepSeek F3): was hard-coded to 1 which reset the
        # retry budget when retry_count was already >= 1.
        return {
            "messages": [AIMessage(content=_fallback_message("trouble processing that response"))],
            "skip_validation": False,
            "retry_count": retry_count + 1,
        }
    except asyncio.CancelledError:
        # Client disconnect (normal for SSE) — NOT an LLM failure.
        # R11 fix (DeepSeek F-005): use record_cancellation() instead of
        # record_failure() to avoid inflating CB failure count from normal
        # SSE disconnects. record_cancellation() resets the half_open probe
        # flag (R10 DeepSeek F1 concern) without counting toward threshold.
        await cb.record_cancellation()
        raise
    except Exception:
        # Broad catch: httpx.HTTPError, asyncio.TimeoutError, ConnectionError,
        # and google-genai SDK exceptions (GoogleAPICallError, ResourceExhausted,
        # DeadlineExceeded, RuntimeError, AttributeError on malformed responses).
        # Without this catch, unhandled exceptions bypass record_failure() so the
        # circuit breaker never trips, and the SSE stream crashes.
        # KeyboardInterrupt/SystemExit propagate (not subclasses of Exception).
        await cb.record_failure()
        logger.exception("%s agent LLM call failed", agent_name.capitalize())
        return {
            "messages": [AIMessage(content=_fallback_message("trouble generating a response right now"))],
            "skip_validation": True,
        }
