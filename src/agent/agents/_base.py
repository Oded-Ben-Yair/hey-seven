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
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.nodes import _format_context_block
from src.agent.prompts import (
    EMOTIONAL_CONTEXT_GUIDES,
    HEART_ESCALATION_LANGUAGE,
    SENTIMENT_TONE_GUIDES,
    get_persona_style,
    get_responsible_gaming_helplines,
)
from src.agent.sentiment import detect_sentiment, detect_sarcasm_context
from src.agent.state import PropertyQAState
from src.agent.whisper_planner import format_whisper_plan
from src.casino.config import get_casino_profile
from src.config import get_settings

logger = logging.getLogger(__name__)

# ADR: LLM Concurrency Limits (R39 fix D9-M001)
#
# LLM concurrency backpressure: limits concurrent LLM API calls across
# all specialist agents per Cloud Run instance. Each /chat request triggers
# 1-6 LLM calls (router + optional specialist dispatch + generate +
# optional validate + retry). With --concurrency=50 per instance, a burst
# of 50 simultaneous /chat requests could trigger 50-300 concurrent LLM
# API calls without this gate.
#
# Prevents: (1) LLM provider 429 rate limiting cascading into circuit
# breaker trips, (2) httpx connection pool exhaustion, (3) memory pressure
# from concurrent response buffering.
#
# Calculation: Gemini Flash rate limit = 300 RPM project-wide. With max 10
# Cloud Run instances (cloudbuild.yaml --max-instances=10), worst case =
# 10 * 20 = 200 concurrent requests, well under 300 RPM. 67% safety margin.
# R5 fix per Gemini F3 analysis.
_LLM_SEMAPHORE = asyncio.Semaphore(20)

# Phase 4: Persona drift prevention threshold.
# Research: persona consistency drops 20-40% over 10-15 LLM turns.
# Re-inject condensed persona reminder after this many messages in context.
_PERSONA_REINJECT_THRESHOLD = 10  # ~5 human turns


def _detect_conversation_dynamics(messages: list) -> dict[str, Any]:
    """Analyze conversation history for behavioral dynamics.

    Returns a dict with detected dynamics:
    - terse_replies: int — count of recent terse (< 5 words) human replies
    - repeated_question: bool — last 2 human messages are semantically similar
    - brevity_preference: bool — guest consistently uses short messages
    - turn_count: int — number of human messages in conversation

    Sub-1ms for typical conversation lengths (< 40 messages).
    """
    dynamics: dict[str, Any] = {
        "terse_replies": 0,
        "repeated_question": False,
        "brevity_preference": False,
        "turn_count": 0,
    }

    human_messages: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            human_messages.append(content)

    dynamics["turn_count"] = len(human_messages)

    if not human_messages:
        return dynamics

    # Detect terse replies (< 5 words in recent messages)
    recent = human_messages[-3:] if len(human_messages) >= 3 else human_messages
    terse_count = sum(1 for msg in recent if len(msg.split()) < 5)
    dynamics["terse_replies"] = terse_count

    # Detect brevity preference (all messages are short)
    if len(human_messages) >= 2:
        avg_words = sum(len(m.split()) for m in human_messages) / len(human_messages)
        dynamics["brevity_preference"] = avg_words < 6

    # Detect repeated question (last 2 human messages share key words)
    if len(human_messages) >= 2:
        last = set(human_messages[-1].lower().split())
        prev = set(human_messages[-2].lower().split())
        # Remove common stop words for comparison
        stop = {"i", "the", "a", "an", "is", "are", "was", "were", "do", "does",
                "did", "can", "could", "what", "where", "when", "how", "about",
                "for", "to", "at", "in", "on", "of", "and", "or", "my", "me",
                "you", "your", "it", "that", "this", "with"}
        last_content = last - stop
        prev_content = prev - stop
        if last_content and prev_content:
            overlap = last_content & prev_content
            # If 50%+ of content words overlap, likely repeated question
            min_len = min(len(last_content), len(prev_content))
            if min_len > 0 and len(overlap) / min_len >= 0.5:
                dynamics["repeated_question"] = True

    return dynamics


def _count_consecutive_frustrated(messages: list) -> int:
    """Count consecutive frustrated/negative sentiments from recent HumanMessages.

    Iterates messages in reverse, running sub-1ms VADER sentiment detection
    on each HumanMessage. Returns count of consecutive frustrated/negative
    messages before the first positive/neutral one.

    Used for frustration escalation: when count >= 2, the speaking agent
    injects a soft escalation offer to connect with a human host.

    Research: HEART framework — guests in sustained distress need human
    empathy, not AI persistence. 40-50% of escalations fail because they
    happen too late.
    """
    count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            sentiment = detect_sentiment(content)
            if sentiment in ("frustrated", "negative"):
                count += 1
            else:
                break
    return count


def _build_behavioral_prompt_sections(
    state: PropertyQAState,
    user_msg: str,
    user_msg_lower: str,
    extracted: dict[str, Any],
    guest_sentiment: str | None,
) -> tuple[str, str | None, dict[str, Any], int]:
    """Build all behavioral prompt sections from state analysis.

    R72 C6: Extracted from execute_specialist to reduce function size
    and isolate behavioral signal detection from LLM execution.

    Returns:
        Tuple of (behavioral_prompt_sections, effective_sentiment,
                  conversation_dynamics, frustrated_count).
    """
    sections: list[str] = []

    # Sarcasm detection: override VADER when conversation context contradicts
    effective_sentiment = guest_sentiment
    if effective_sentiment in ("positive", "neutral"):
        recent_sentiments: list[str] = []
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage) and len(recent_sentiments) < 3:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                recent_sentiments.append(detect_sentiment(content))
        prior_sentiments = recent_sentiments[1:] if len(recent_sentiments) > 1 else []
        if detect_sarcasm_context(user_msg, effective_sentiment, prior_sentiments):
            effective_sentiment = "frustrated"
            logger.info("Sarcasm override: %s → frustrated (context contrast)", guest_sentiment)

    # Sentiment tone guide
    if effective_sentiment:
        tone_guide = SENTIMENT_TONE_GUIDES.get(effective_sentiment, "")
        if tone_guide:
            sections.append(f"## Tone Guidance\n{tone_guide}")

    # Emotional context guides (grief, anxiety, celebration, allergy, gambling frustration)
    emotional_guides: list[str] = []
    if any(kw in user_msg_lower for kw in ("passed away", "passed on", "lost my", "funeral", "in memory", "rest in peace")):
        emotional_guides.append(EMOTIONAL_CONTEXT_GUIDES["grief"])
    if any(kw in user_msg_lower for kw in ("first time", "never been", "nervous", "anxious", "intimidat", "overwhelm")):
        emotional_guides.append(EMOTIONAL_CONTEXT_GUIDES["anxiety"])
    if extracted.get("occasion"):
        emotional_guides.append(EMOTIONAL_CONTEXT_GUIDES["celebration"])
    if any(kw in user_msg_lower for kw in ("allerg", "anaphyla", "epipen", "celiac")):
        emotional_guides.append(EMOTIONAL_CONTEXT_GUIDES["allergy_concern"])
    if any(kw in user_msg_lower for kw in ("losing", "lost all", "bad day", "bad luck", "cold streak", "down a lot")):
        emotional_guides.append(EMOTIONAL_CONTEXT_GUIDES["gambling_frustration"])
    if emotional_guides:
        sections.append("## Emotional Context\n" + "\n\n".join(emotional_guides))

    # Implicit signal guides (loyalty, urgency, fatigue, budget)
    if extracted.get("loyalty_signal"):
        sections.append(
            "## Loyalty Context\n"
            f"The guest has signaled loyalty: \"{extracted['loyalty_signal']}\". "
            "Treat them as a valued long-term guest. Acknowledge their history warmly. "
            "Recommend elevated, VIP-appropriate experiences."
        )
    if extracted.get("urgency"):
        sections.append(
            "## Urgency Context\n"
            "The guest is short on time. Give short, direct answers. "
            "Prioritize proximity and speed. Skip marketing language and preamble. "
            "Lead with the single best option, not a list."
        )
    if extracted.get("fatigue"):
        sections.append(
            "## Fatigue Context\n"
            "The guest is tired or has traveled far. Recommend restful, low-effort options: "
            "spa, quiet dining, pool. Avoid high-energy suggestions like gaming floor, "
            "loud venues, or activities requiring lots of walking."
        )
    if extracted.get("budget_conscious"):
        sections.append(
            "## Budget Context\n"
            "The guest has signaled budget consciousness. Lead with value options, "
            "free activities (Wolf Den, pool), and casual dining (buffet). "
            "Do NOT suggest premium/expensive options first. Maintain this filter "
            "throughout the conversation."
        )

    # Cross-domain awareness
    domains_discussed = state.get("domains_discussed", [])
    if domains_discussed:
        _all_domains = {"dining", "entertainment", "comp", "hotel", "host"}
        _not_discussed = _all_domains - set(domains_discussed) - {"host"}
        if _not_discussed:
            sections.append(
                "## Domain Awareness\n"
                f"You've already helped with: {', '.join(sorted(domains_discussed))}. "
                f"If the guest asks 'what else' or seems done with this topic, "
                f"naturally suggest exploring: {', '.join(sorted(_not_discussed))}."
            )

    # Conversation dynamics
    history = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
    dynamics = _detect_conversation_dynamics(history)

    dynamics_guides: list[str] = []
    if dynamics["terse_replies"] >= 2:
        dynamics_guides.append(
            "The guest is giving very short replies — they may be disengaged or prefer "
            "brevity. Switch to short, direct answers. Ask a focused either/or question "
            "instead of listing options. Make ONE confident recommendation."
        )
    if dynamics["repeated_question"]:
        dynamics_guides.append(
            "The guest appears to be repeating a question — your prior answer may not "
            "have been clear. Acknowledge this ('Let me be more specific'), then "
            "answer in a different format: use a direct fact (time, location) "
            "instead of prose."
        )
    if dynamics["brevity_preference"]:
        dynamics_guides.append(
            "The guest consistently uses short messages. Match their style: "
            "keep responses concise and functional. Skip pleasantries and marketing."
        )
    if dynamics_guides:
        sections.append("## Conversation Style Guidance\n" + "\n".join(dynamics_guides))

    # Frustration escalation (HEART framework)
    frustrated_count = _count_consecutive_frustrated(history)
    if frustrated_count >= 2 and state.get("guest_sentiment") in ("frustrated", "negative"):
        heart = HEART_ESCALATION_LANGUAGE
        if frustrated_count >= 3:
            heart_steps = (
                f"1. HEAR: {heart['hear']}\n"
                f"2. EMPATHIZE: {heart['empathize']}\n"
                f"3. APOLOGIZE: {heart['apologize']}\n"
                f"4. RESOLVE: {heart['resolve']}\n"
                f"5. THANK: {heart['thank']}"
            )
        else:
            heart_steps = (
                f"1. HEAR: {heart['hear']}\n"
                f"2. EMPATHIZE: {heart['empathize']}\n"
                "3. Gently offer to connect with a human host for personalized help."
            )
        sections.append(
            "## Escalation Guidance (HEART Framework)\n"
            "The guest has expressed frustration across multiple messages. "
            "Follow these steps in your response:\n"
            f"{heart_steps}\n"
            "After addressing their concern, offer: \"Would you like me to "
            "connect you with one of our dedicated hosts who can assist you "
            "personally?\""
        )

    prompt_text = ""
    if sections:
        prompt_text = "\n\n" + "\n\n".join(sections)

    return prompt_text, effective_sentiment, dynamics, frustrated_count


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

    # R29 fix: Inject property_description from casino profile for multi-property support.
    # Avoids hardcoding Mohegan Sun description in the system prompt template.
    _casino_profile = get_casino_profile(settings.CASINO_ID)
    _property_description = _casino_profile.get("property_description", "")

    system_prompt = system_prompt_template.safe_substitute(
        property_name=settings.PROPERTY_NAME,
        current_time=current_time,
        responsible_gaming_helplines=get_responsible_gaming_helplines(casino_id=settings.CASINO_ID),
        property_description=_property_description,
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
    # R27 fix H-003: use property-specific profile instead of DEFAULT_CONFIG
    try:
        _profile = get_casino_profile(settings.CASINO_ID)
        branding = _profile.get("branding", {})
        persona_style = get_persona_style(branding)
        if persona_style:
            system_prompt += persona_style
    except Exception:
        logger.debug("Persona style injection failed, continuing without", exc_info=True)

    # R72 C6: Extract behavioral signals into dedicated helper (SRP refactor).
    # Detects sarcasm, emotional context, implicit signals, conversation dynamics,
    # frustration escalation, and domain tracking — returns prompt sections to inject.
    extracted = state.get("extracted_fields") or {}
    user_msg = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    user_msg_lower = user_msg.lower()

    behavioral_sections, guest_sentiment, dynamics, frustrated_count = (
        _build_behavioral_prompt_sections(
            state, user_msg, user_msg_lower, extracted, state.get("guest_sentiment"),
        )
    )
    system_prompt += behavioral_sections

    # Phase 4: Proactive suggestion injection from whisper plan.
    # R71 B4 fix: Allow suggestions on neutral sentiment when contextual signals
    # are strong (occasion detected, building multi-turn plan, positive dynamics).
    # Still block on frustrated/negative and None (detection failed).
    # R23 fix C-003: check suggestion_offered to enforce max-1-per-conversation.
    whisper = state.get("whisper_plan")
    suggestion_already_offered = state.get("suggestion_offered", False)
    _sentiment = guest_sentiment  # Use effective sentiment (may be overridden by sarcasm detection)
    _allow_suggestion = (
        _sentiment == "positive"
        or (
            _sentiment == "neutral"
            and (
                # Strong contextual signals that make proactive suggestion welcome:
                extracted.get("occasion")  # Guest celebrating something
                or dynamics["turn_count"] >= 3  # Multi-turn engaged conversation
            )
        )
    )
    if (
        whisper
        and not suggestion_already_offered
        and _allow_suggestion
    ):
        suggestion = whisper.get("proactive_suggestion")
        conf = whisper.get("suggestion_confidence", 0.0)
        if suggestion and conf >= 0.8:
            system_prompt += (
                "\n\n## Proactive Suggestion (weave naturally, don't force)\n"
                f"{suggestion}\n"
                "Only mention this if it fits naturally in your response. "
                "Never push; if the guest doesn't bite, drop it."
            )
            # R23 fix C-003: Mark suggestion as offered to enforce max-1.
            # _keep_max reducer: max(True, False) = True, persists across turns.
            suggestion_already_offered = True  # Used in return dict below

    # Build message list
    llm_messages = [SystemMessage(content=system_prompt)]

    # Compute conversation history for persona reinject and sliding window
    history = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]

    # Phase 4: Persona drift prevention — re-inject condensed persona reminder
    # when conversation history exceeds threshold. Research: persona consistency
    # drops 20-40% over 10-15 turns without reinforcement.
    # R23 fix H-002: count only HumanMessage instances for threshold check
    # (retries add extra AIMessages that inflate the count).
    human_turn_count = sum(1 for m in history if isinstance(m, HumanMessage))
    if human_turn_count > _PERSONA_REINJECT_THRESHOLD // 2:
        # R23/R29 fix: read persona name from property-specific profile
        try:
            _persona_name = get_casino_profile(
                settings.CASINO_ID
            ).get("branding", {}).get("persona_name", "Seven")
        except Exception:
            _persona_name = "Seven"
        llm_messages.append(SystemMessage(content=(
            f"PERSONA REMINDER: You are {_persona_name}, the AI concierge for "
            f"{settings.PROPERTY_NAME}. Maintain your warm, professional tone. "
            "Never provide gambling advice or discuss competitors. "
            "Always stay on-property topics. Be concise and helpful."
        )))

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
    if retry_count > 0 and history and isinstance(history[-1], AIMessage):
        history = history[:-1]
    window = history[-settings.MAX_HISTORY_MESSAGES:]
    llm_messages.extend(window)

    llm = await get_llm_fn()

    # R46 D8: Semaphore acquisition with timeout for backpressure.
    # If all 20 LLM slots are busy, return a fallback instead of
    # queueing indefinitely. Prevents request pile-up during LLM slowdowns.
    # R46 fix D8-M002: Read timeout from config instead of hardcoding.
    #
    # R52 fix D8: Single try/finally with acquired flag. Previous structure had
    # acquire() outside try/finally — CancelledError between acquire() completing
    # and entering try block leaked the semaphore count permanently.
    semaphore_timeout = settings.LLM_SEMAPHORE_TIMEOUT
    acquired = False
    try:
        try:
            await asyncio.wait_for(_LLM_SEMAPHORE.acquire(), timeout=semaphore_timeout)
            acquired = True
        except asyncio.TimeoutError:
            logger.warning(
                "%s agent semaphore acquire timeout (%ds) — returning backpressure fallback",
                agent_name.capitalize(),
                semaphore_timeout,
            )
            return {
                "messages": [AIMessage(content=_fallback_message("high demand right now"))],
                "skip_validation": True,
            }

        try:
            response = await llm.ainvoke(llm_messages)
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
    finally:
        if acquired:
            _LLM_SEMAPHORE.release()

    await cb.record_success()
    content = response.content if isinstance(response.content, str) else str(response.content)
    result: dict = {"messages": [AIMessage(content=content)]}
    # R23 fix C-003: persist suggestion_offered flag across turns
    if suggestion_already_offered:
        result["suggestion_offered"] = True  # _keep_truthy: once True, stays True
    return result
