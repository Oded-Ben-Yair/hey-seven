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
import re
from collections.abc import Callable
from string import Template
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.nodes import _format_context_block, _normalize_content
from src.agent.prompts import (
    EMOTIONAL_CONTEXT_GUIDES,
    FEW_SHOT_EXAMPLES,
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
        "list_format": False,
        "conversation_phase": "opening",
    }

    human_messages: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = _normalize_content(msg.content)
            human_messages.append(content)

    dynamics["turn_count"] = len(human_messages)

    # Conversation phase based on turn count
    turn_count = dynamics["turn_count"]
    if turn_count <= 2:
        dynamics["conversation_phase"] = "opening"
    elif turn_count <= 6:
        dynamics["conversation_phase"] = "exploring"
    else:
        dynamics["conversation_phase"] = "deciding"

    if not human_messages:
        return dynamics

    # Detect list format in last human message (numbered or bullet lists)
    last_msg = human_messages[-1]
    has_numbers = bool(re.search(r"(?:^|\n)\s*\d+[.\)]\s", last_msg))
    has_bullets = bool(re.search(r"(?:^|\n)\s*[-*\u2022]\s", last_msg))
    dynamics["list_format"] = has_numbers or has_bullets

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
        stop = {
            "i",
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "can",
            "could",
            "what",
            "where",
            "when",
            "how",
            "about",
            "for",
            "to",
            "at",
            "in",
            "on",
            "of",
            "and",
            "or",
            "my",
            "me",
            "you",
            "your",
            "it",
            "that",
            "this",
            "with",
        }
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
            content = _normalize_content(msg.content)
            sentiment = detect_sentiment(content)
            if sentiment in ("frustrated", "negative"):
                count += 1
            else:
                break
    return count


# R74 B3: Full domain set for cross-domain engagement variation.
# Broader than the original specialist agent set (dining, entertainment, comp, hotel,
# host) — includes spa, gaming, shopping, and promotions to cover the full casino
# experience.  "host" is excluded from suggestions since it's meta (not a domain
# the guest "explores").
_ALL_GUEST_DOMAINS = frozenset(
    {
        "dining",
        "entertainment",
        "hotel",
        "spa",
        "gaming",
        "shopping",
        "promotions",
        "comp",
    }
)

# R82 Track 2C: Cross-domain bridge templates.
# Specific transition phrases between domain pairs, injected into the system
# prompt when the guest has discussed domain A and domain B is available.
# Format: {(from_domain, to_domain): "bridge text"}
CROSS_DOMAIN_BRIDGES: dict[tuple[str, str], str] = {
    # R88: Updated with property-specific venue names for Mohegan Sun.
    # Generic bridges like "the spa is nice" scored B4=3-4 in judge panel.
    # Concrete venue references improve specificity and trust.
    (
        "dining",
        "entertainment",
    ): "After dinner, the Wolf Den has free live music every night — it's right in the Casino of the Earth and the energy is always good.",
    (
        "dining",
        "spa",
    ): "After dinner, the Mandara Spa is open until 8 PM if you want to fully unwind — their relaxation lounge alone is worth the visit.",
    (
        "dining",
        "hotel",
    ): "If you're making an evening of it, the Sky Tower rooms have great views and you're steps from everything.",
    (
        "entertainment",
        "dining",
    ): "Before the show, Todd English's Tuscany is a short walk from the Wolf Den and their wood-fired dishes pair well with a night out.",
    (
        "entertainment",
        "hotel",
    ): "For late shows, staying at the resort means you can walk back to your room instead of the drive home.",
    (
        "entertainment",
        "spa",
    ): "After a high-energy show, the Elemis Spa in the Earth Tower is a great contrast — their express treatments are perfect for winding down.",
    (
        "hotel",
        "dining",
    ): "As a hotel guest, you're steps from every restaurant — SolToro for casual Mexican, Tuscany for Italian, or MJ's for steaks.",
    (
        "hotel",
        "spa",
    ): "Hotel guests can easily walk to the Mandara Spa — it's a Balinese-inspired retreat that feels worlds away from the casino floor.",
    (
        "hotel",
        "entertainment",
    ): "Staying over means you can catch the Wolf Den shows at 8 PM without worrying about the drive home.",
    (
        "spa",
        "dining",
    ): "After a spa session, Caputo Trattoria is a low-key spot for comfort food, or Tuscany if you want something more refined.",
    (
        "spa",
        "hotel",
    ): "If you're planning a full spa day, combining it with an overnight stay in the Sky Tower makes it a real retreat.",
    (
        "comp",
        "dining",
    ): "With your play history, you might be closer to the next tier than you think — dining credits at Tuscany and SolToro open up at each level.",
    (
        "comp",
        "entertainment",
    ): "Each Momentum tier unlocks more — priority Arena access and VIP Wolf Den seating are closer than you might think.",
    (
        "comp",
        "spa",
    ): "Spa credits at Mandara and Elemis open up as you move through tiers — you might already qualify based on your visits.",
    (
        "gaming",
        "dining",
    ): "When you're ready for a break from the floor, Bobby's Burger Palace is quick, or Michael Jordan's Steak House for something special.",
}


def _build_cross_domain_hint(domains_discussed: list[str]) -> str:
    """Build cross-domain engagement hint with specific bridge templates.

    R82 Track 2C: Enhanced with specific bridge text instead of generic
    "you could mention X" hints. The LLM gets a concrete transition phrase
    it can use or adapt, rather than inventing one from scratch.

    Args:
        domains_discussed: List of specialist domain names already discussed
            in this session (e.g. ``["dining", "entertainment"]``).

    Returns:
        A prompt section string (without leading ``\\n\\n``) if there are
        both explored and unexplored domains, or ``""`` if no hint is needed.
    """
    if not domains_discussed:
        return ""

    unexplored = sorted(_ALL_GUEST_DOMAINS - set(domains_discussed))
    if not unexplored:
        return ""

    # Find specific bridge templates from discussed -> unexplored
    bridges: list[str] = []
    last_domain = domains_discussed[-1] if domains_discussed else ""
    for target in unexplored[:3]:  # Max 3 suggestions
        bridge_text = CROSS_DOMAIN_BRIDGES.get((last_domain, target))
        if bridge_text:
            bridges.append(f'- {target}: "{bridge_text}"')

    if not bridges:
        # Fallback to generic hint if no specific bridges found
        explored_str = ", ".join(sorted(domains_discussed))
        suggest_str = ", ".join(unexplored[:3])
        return (
            "## Cross-Domain Awareness (internal context)\n"
            f"Guest has already explored: {explored_str}.\n"
            f"If it fits naturally, you could mention: {suggest_str}.\n"
            "Do NOT force these — only mention if genuinely relevant to the conversation."
        )

    explored_str = ", ".join(sorted(domains_discussed))
    return (
        "## Cross-Domain Awareness (internal context)\n"
        f"Guest has explored: {explored_str}.\n"
        "If natural, you can use one of these transitions:\n"
        + "\n".join(bridges)
        + "\n"
        "Use these as inspiration — adapt to the conversation, don't copy verbatim."
    )


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
                content = _normalize_content(msg.content)
                recent_sentiments.append(detect_sentiment(content))
        prior_sentiments = recent_sentiments[1:] if len(recent_sentiments) > 1 else []
        if detect_sarcasm_context(user_msg, effective_sentiment, prior_sentiments):
            effective_sentiment = "frustrated"
            logger.info(
                "Sarcasm override: %s → frustrated (context contrast)", guest_sentiment
            )

    # Sentiment tone guide
    if effective_sentiment:
        tone_guide = SENTIMENT_TONE_GUIDES.get(effective_sentiment, "")
        if tone_guide:
            sections.append(f"## Tone Guidance\n{tone_guide}")

    # Emotional context guides (grief, anxiety, celebration, allergy, gambling frustration)
    emotional_guides: list[str] = []
    if any(
        kw in user_msg_lower
        for kw in (
            "passed away",
            "passed on",
            "lost my",
            "funeral",
            "in memory",
            "rest in peace",
        )
    ):
        emotional_guides.append(EMOTIONAL_CONTEXT_GUIDES["grief"])
    if any(
        kw in user_msg_lower
        for kw in (
            "first time",
            "never been",
            "nervous",
            "anxious",
            "intimidat",
            "overwhelm",
        )
    ):
        emotional_guides.append(EMOTIONAL_CONTEXT_GUIDES["anxiety"])
    if extracted.get("occasion"):
        emotional_guides.append(EMOTIONAL_CONTEXT_GUIDES["celebration"])
    if any(kw in user_msg_lower for kw in ("allerg", "anaphyla", "epipen", "celiac")):
        emotional_guides.append(EMOTIONAL_CONTEXT_GUIDES["allergy_concern"])
    if any(
        kw in user_msg_lower
        for kw in (
            "losing",
            "lost all",
            "bad day",
            "bad luck",
            "cold streak",
            "down a lot",
        )
    ):
        emotional_guides.append(EMOTIONAL_CONTEXT_GUIDES["gambling_frustration"])
    # R94: Loss recovery — lead with empathy, not upsell
    if any(
        kw in user_msg_lower
        for kw in (
            "lost $",
            "down $",
            "dropped $",
            "blew $",
            "lost a lot",
            "bad night",
            "rough night",
            "lost everything",
            "lost big",
        )
    ):
        emotional_guides.append(
            "**Loss Recovery**: The guest has experienced a significant loss. "
            "Lead with genuine empathy: 'That's a tough night.' "
            "Do NOT suggest dining, spa, or entertainment UNTIL the guest opens the door. "
            "Wait for them to ask for a distraction. If they vent, just listen and acknowledge. "
            "NEVER say 'cheer up' or immediately pivot to 'but we have great shows!'"
        )
    # R94: Disappointment/dissatisfaction (not frustration — more subtle)
    if any(
        kw in user_msg_lower
        for kw in (
            "disappointed",
            "letdown",
            "let down",
            "expected more",
            "not what i expected",
            "underwhelm",
            "mediocre",
        )
    ):
        emotional_guides.append(
            "**Disappointment Recovery**: The guest is disappointed (not angry — subtle). "
            "Acknowledge: 'I hear you — that's not the experience we want for you.' "
            "Then offer a specific alternative: 'If you're open to it, [X] might be more what you're looking for.' "
            "Don't dismiss or argue with their assessment."
        )
    if emotional_guides:
        sections.append("## Emotional Context\n" + "\n\n".join(emotional_guides))

    # Implicit signal guides (loyalty, urgency, fatigue, budget)
    if extracted.get("loyalty_signal"):
        sections.append(
            "## VIP Recognition (Action, Not Words)\n"
            f'The guest has signaled loyalty: "{extracted["loyalty_signal"]}". '
            "Do NOT just say 'valued guest' — that's generic. Instead:\n"
            "- Reference specific benefits they'd qualify for: 'With your play history, "
            "you'd typically qualify for complimentary dining at Bobby's or Tuscany.'\n"
            "- Offer to check their Momentum status: 'I can look into your tier — "
            "that would tell us exactly what comps are available.'\n"
            "- Suggest elevated experiences: private gaming salons, VIP events, host introduction.\n"
            "- Use their history to personalize: 'Guests who visit as often as you do "
            "usually have a dedicated host — I can connect you.'"
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

    # R91: Tier curiosity — aspirational framing for loyalty/tier questions
    if any(
        kw in user_msg_lower
        for kw in (
            "tier",
            "momentum",
            "rewards",
            "loyalty",
            "points",
            "upgrade",
            "benefits",
            "perks",
        )
    ):
        sections.append(
            "## Tier Advancement (Aspirational)\n"
            "The guest is curious about tiers/rewards. Frame tier info ASPIRATIONALLY, not as brochure:\n"
            "- 'You're already earning toward your next tier just by being here.'\n"
            "- 'At [X] credits you hit [Tier], which opens up [specific benefit].'\n"
            "- 'How often do you visit? I can give you a sense of where you might be.'\n"
            "NEVER just list tiers like a brochure. Connect their current behavior to what they unlock next."
        )

    # R91: Loss/security ownership — host takes action, never deflects
    if any(
        kw in user_msg_lower
        for kw in (
            "lost my",
            "stolen",
            "theft",
            "stole",
            "missing",
            "left my",
            "someone took",
        )
    ) and not any(
        # Exclude grief/gambling contexts (already handled by their own guides)
        kw in user_msg_lower
        for kw in ("passed away", "passed on", "lost all", "lost everything")
    ):
        sections.append(
            "## Loss/Security Response\n"
            "The guest is reporting a lost or stolen item. Take OWNERSHIP:\n"
            "- 'I'll flag this for our security team right away.'\n"
            "- Ask WHEN and WHERE it happened so you can pass along details.\n"
            "- Provide the security desk location and phone number.\n"
            "- NEVER say 'I can't report this' or 'you'll need to contact security yourself'.\n"
            "- Frame yourself as the facilitator: 'Let me get this to the right people.'"
        )

    # R74 B3: Cross-domain engagement variation via extracted helper.
    domains_discussed = state.get("domains_discussed", [])
    cross_domain_hint = _build_cross_domain_hint(domains_discussed)
    if cross_domain_hint:
        sections.append(cross_domain_hint)

    # Conversation dynamics
    history = [
        m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))
    ]
    dynamics = _detect_conversation_dynamics(history)

    dynamics_guides: list[str] = []
    if dynamics["terse_replies"] >= 2:
        dynamics_guides.append(
            "The guest is giving very short replies. CAP your response at 2 sentences maximum. "
            "No lists, no bullet points, no preamble. One direct answer or one focused question."
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

    # Conversation phase guidance
    phase = dynamics.get("conversation_phase", "exploring")
    if phase == "opening":
        sections.append(
            "## Conversation Phase: Opening\n"
            "The guest is just starting. Be welcoming and orient them to what's available. "
            "Ask what brings them in today if they haven't said."
        )
    elif phase == "deciding":
        sections.append(
            "## Conversation Phase: Deciding\n"
            "The guest has been exploring for several turns. They're likely narrowing down. "
            "Make ONE confident recommendation instead of listing options. Be decisive."
        )

    # Format adaptation (match guest's format)
    if dynamics.get("list_format"):
        sections.append(
            "## Format Matching\n"
            "The guest used a list/numbered format. Match their style — respond with "
            "a structured list or numbered items."
        )

    # Frustration escalation (HEART framework)
    frustrated_count = _count_consecutive_frustrated(history)
    if frustrated_count >= 2 and state.get("guest_sentiment") in (
        "frustrated",
        "negative",
    ):
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
            'After addressing their concern, offer: "Would you like me to '
            "connect you with one of our dedicated hosts who can assist you "
            'personally?"'
        )

    prompt_text = ""
    if sections:
        prompt_text = "\n\n" + "\n\n".join(sections)

    return prompt_text, effective_sentiment, dynamics, frustrated_count


def _should_inject_suggestion(
    state: PropertyQAState,
    effective_sentiment: str | None,
    dynamics: dict[str, Any],
) -> tuple[str, bool]:
    """Determine whether to inject a proactive suggestion into the system prompt.

    R82 Track 1F: Added per-gate logging + lowered thresholds.

    Gating conditions (ALL must be true):
      1. Whisper plan has a non-empty ``proactive_suggestion``
      2. ``suggestion_confidence`` >= 0.6 (R82 1F: lowered from 0.8)
      3. ``effective_sentiment`` is not negative, frustrated, or None
      4. ``suggestion_offered`` is False (max 1 per session)
      5. ``retrieved_context`` is non-empty (grounding exists)

    R82 1F: Neutral sentiment now passes without extra gates (removed
    occasion/engagement requirement that was too restrictive).

    Returns:
        Tuple of (proactive_prompt_section, should_mark_offered).
        If conditions are not met, returns ("", False).
    """
    whisper = state.get("whisper_plan")
    if not whisper:
        logger.debug("Proactivity gate: NO whisper_plan")
        return "", False

    suggestion = whisper.get("proactive_suggestion")
    if not suggestion:
        logger.debug("Proactivity gate: NO proactive_suggestion in whisper")
        return "", False

    # R82 1F: Lowered from 0.8 to 0.6 — 0.8 was too strict, almost never passed
    # R76 fix: suggestion_confidence is now a str (simplified WhisperPlan schema).
    try:
        conf = float(whisper.get("suggestion_confidence", 0.0))
    except (ValueError, TypeError):
        conf = 0.0
    if conf < 0.6:
        logger.debug("Proactivity gate: confidence %.2f < 0.6 threshold", conf)
        return "", False

    if state.get("suggestion_offered", False):
        logger.debug("Proactivity gate: suggestion already offered this session")
        return "", False

    if not state.get("retrieved_context"):
        logger.debug("Proactivity gate: no retrieved_context for grounding")
        return "", False

    # Sentiment gate: block on negative, frustrated, or unknown.
    # R82 1F: Allow on neutral unconditionally (removed extra gate for neutral).
    # The old code required occasion or 3+ turns for neutral — too restrictive.
    if effective_sentiment == "positive":
        pass  # allowed
    elif effective_sentiment == "neutral":
        pass  # R82 1F: Allow on neutral without extra gates
    else:
        logger.debug("Proactivity gate: sentiment '%s' blocked", effective_sentiment)
        return "", False

    logger.info(
        "Proactivity gate: ALL PASSED (conf=%.2f, sentiment=%s, turn=%d)",
        conf,
        effective_sentiment,
        dynamics.get("turn_count", 0),
    )

    section = (
        "\n\n## Proactive Suggestion (weave naturally into your response — don't force it)\n"
        f"{suggestion}\n"
        "Only mention this if it flows naturally from the conversation."
    )
    return section, True


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
            "messages": [
                AIMessage(content=_fallback_message("temporary technical difficulties"))
            ],
            "skip_validation": True,
        }

    # R29 fix: Inject property_description from casino profile for multi-property support.
    # Avoids hardcoding Mohegan Sun description in the system prompt template.
    _casino_profile = get_casino_profile(settings.CASINO_ID)
    _property_description = _casino_profile.get("property_description", "")

    system_prompt = system_prompt_template.safe_substitute(
        property_name=settings.PROPERTY_NAME,
        current_time=current_time,
        responsible_gaming_helplines=get_responsible_gaming_helplines(
            casino_id=settings.CASINO_ID
        ),
        property_description=_property_description,
    )

    # Phase 1: Language-aware prompt selection for Spanish support.
    # When guest is speaking Spanish and spanish_support_enabled flag is on,
    # replace the English system prompt with the Spanish equivalent.
    # Feature flag allows instant rollback per-casino.
    detected_lang = state.get("detected_language")
    if detected_lang == "es":
        from src.casino.feature_flags import is_feature_enabled as _is_feature_enabled

        if await _is_feature_enabled(settings.CASINO_ID, "spanish_support_enabled"):
            from src.agent.prompts import (
                CONCIERGE_SYSTEM_PROMPT_ES,
                get_responsible_gaming_helplines_es,
            )

            system_prompt = CONCIERGE_SYSTEM_PROMPT_ES.safe_substitute(
                property_name=settings.PROPERTY_NAME,
                current_time=current_time,
                responsible_gaming_helplines=get_responsible_gaming_helplines_es(
                    casino_id=settings.CASINO_ID,
                ),
                property_description=_property_description,
            )

    # Format and append retrieved context
    if retrieved:
        context_block = _format_context_block(retrieved)
        system_prompt += Template("\n\n## $header\n$context").safe_substitute(
            header=context_header, context=context_block
        )

        # Phase 5: Annotate retrieved context with real-time open/closed status.
        from src.agent.hours import is_open_now as _is_open_now

        _casino_tz = _casino_profile.get("operational", {}).get(
            "timezone", "America/New_York"
        )
        hours_annotations: list[str] = []
        for chunk in retrieved:
            meta = chunk.get("metadata", {})
            hours_str = meta.get("hours", "")
            item_name = meta.get("item_name", meta.get("name", ""))
            if hours_str and item_name:
                open_status = _is_open_now(hours_str, timezone=_casino_tz)
                if open_status is True:
                    hours_annotations.append(f"- {item_name}: OPEN ({hours_str})")
                elif open_status is False:
                    hours_annotations.append(f"- {item_name}: CLOSED ({hours_str})")
        if hours_annotations:
            system_prompt += "\n\n## Current Availability\n" + "\n".join(
                hours_annotations
            )
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

    # R79: Inject accumulated profiling data (extracted_fields) as additional context.
    # This data comes from the profiling enrichment node and is more comprehensive
    # than guest_context (which only has user-submitted form data).
    if not guest_context:  # Don't double-inject if guest_context already covered it
        _profile_fields = state.get("extracted_fields") or {}
        if _profile_fields:
            profile_parts: list[str] = []
            if _profile_fields.get("name"):
                profile_parts.append(f"- Guest name: {_profile_fields['name']}")
            if _profile_fields.get("party_size"):
                profile_parts.append(f"- Party size: {_profile_fields['party_size']}")
            if _profile_fields.get("visit_purpose"):
                profile_parts.append(
                    f"- Visit purpose: {_profile_fields['visit_purpose']}"
                )
            if _profile_fields.get("preferences"):
                profile_parts.append(
                    f"- Dining preferences: {_profile_fields['preferences']}"
                )
            if _profile_fields.get("occasion"):
                profile_parts.append(f"- Occasion: {_profile_fields['occasion']}")
            if _profile_fields.get("dietary"):
                profile_parts.append(
                    f"- Dietary restrictions: {_profile_fields['dietary']}"
                )
            if _profile_fields.get("budget_signal"):
                profile_parts.append(f"- Budget: {_profile_fields['budget_signal']}")
            if profile_parts:
                system_prompt += (
                    "\n\n## Guest Profile (from conversation)\n"
                    + "\n".join(profile_parts)
                )

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
        logger.debug(
            "Persona style injection failed, continuing without", exc_info=True
        )

    # R72 C6: Extract behavioral signals into dedicated helper (SRP refactor).
    # Detects sarcasm, emotional context, implicit signals, conversation dynamics,
    # frustration escalation, and domain tracking — returns prompt sections to inject.
    extracted = state.get("extracted_fields") or {}
    user_msg = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_msg = _normalize_content(msg.content)
            break
    user_msg_lower = user_msg.lower()

    behavioral_sections, guest_sentiment, dynamics, frustrated_count = (
        _build_behavioral_prompt_sections(
            state,
            user_msg,
            user_msg_lower,
            extracted,
            state.get("guest_sentiment"),
        )
    )
    system_prompt += behavioral_sections

    # R85: Behavioral uplift — anti-deflection, signal reading, format adaptation,
    # natural follow-up, and cross-domain suggestions. Injected between behavioral
    # sections and few-shot examples so the LLM sees instructions before examples.
    system_prompt += (
        "\n\n## Agent Behavior — You ARE the Host\n"
        "You are their casino host, not a help desk. NEVER deflect to phone, "
        "website, or 'contact us' unless the guest explicitly asks for a number. "
        "Answer directly with what you know.\n\n"
        "CRITICAL: NEVER say 'I can't make reservations' or 'While I can't book'. "
        "Say 'I've got you' and provide specific recommendations. After giving info, "
        "offer to connect them with the host team to finalize. "
        "You facilitate — you don't refuse.\n\n"
        "## Reading the Room\n"
        "Adapt to implicit signals:\n"
        '- "drove 3 hours" / "long day" → exhausted, recommend restful options\n'
        '- "been coming for years" → VIP treatment, acknowledge loyalty\n'
        '- Short replies / "just tell me one" → be decisive, ONE pick, not a list\n'
        '- "I suppose it was fine" → hidden dissatisfaction, probe gently\n'
        '- "just landed" → hungry and quick, prioritize speed\n\n'
        "## Response Format\n"
        "Match your format to the guest's energy:\n"
        "- Short question → direct answer (2-3 sentences max)\n"
        "- Enthusiastic exploration → richer detail\n"
        "- Follow-up → build on context, don't restart\n"
        "- Frustration → ONE definitive pick with confidence\n\n"
        "## Relationship Building (EVERY TURN)\n"
        "End every response with ONE natural question that helps you know the guest "
        "better. Always offer something useful FIRST, then ask.\n\n"
        "Pattern: [answer/recommendation] + [question that personalizes the experience]\n\n"
        "Examples:\n"
        '- "Bobby\'s is our best celebration spot — how many in your group?"\n'
        '- "Wolf Den has a great show tonight — are you more into live music or comedy?"\n'
        '- "Spa opens at 9 AM, morning slots are quietest — would that fit your schedule?"\n\n'
        "When you learn something (name, occasion, party size, food preference), USE IT "
        "immediately: 'Since you mentioned the anniversary, Tuscany's waterfall table "
        "would be perfect.'\n\n"
        "## Cross-Domain Awareness\n"
        "After answering, suggest ONE related activity from a different domain "
        "when natural:\n"
        "- After dining → entertainment or spa\n"
        "- After hotel → dining or entertainment\n"
        "- After entertainment → dining or late-night options\n"
        'Frame concretely: "After dinner, Wolf Den usually has great live music" '
        'not "Would you also like entertainment?"'
    )

    # R92: Booking context — when guest wants to make a reservation,
    # inject qualifying question guidance for the specialist.
    if state.get("booking_intent"):
        _BOOKING_QUALIFIERS = {
            "dining": "party size, date/time, occasion, and dietary restrictions",
            "hotel": "check-in/check-out dates, room type, and number of guests",
            "entertainment": "number of tickets, preferred seating, and date",
            "spa": "preferred treatment, date/time, and number of guests",
        }
        _qualifier = _BOOKING_QUALIFIERS.get(
            agent_name, "date, party size, and preferences"
        )
        system_prompt += (
            "\n\n## Booking Context\n"
            "The guest wants to make a reservation or booking. Your response should:\n"
            "1. Provide specific venue/option recommendations based on what they've told you\n"
            "2. Confirm any details you already know (party size, date, preferences)\n"
            f"3. Ask for missing qualifying details: {_qualifier}\n"
            "4. When they're ready, offer to connect them with the host team to finalize\n\n"
            "Do NOT just redirect to a phone number. Give them real, helpful information first.\n\n"
            "## Recommendation→Question Micro-Flow (REQUIRED for booking)\n"
            "After every venue/option recommendation, ALWAYS ask ONE qualifying question.\n"
            "Examples:\n"
            '- "Todd English\'s Tuscany has amazing wood-fired pasta — how many will be joining you?"\n'
            '- "The Sky Tower rooms have mountain views starting at $199 — what dates are you looking at?"\n'
            '- "The Mandara Spa\'s couples massage is our most popular — would morning or afternoon work better?"'
        )

    # R83: Inject few-shot behavioral examples for the current specialist.
    # Examples show the model the exact tone and style expected — more effective
    # than descriptive instructions alone. Gated by few_shot_examples_enabled flag.
    # Cap at 3 examples to avoid prompt bloat.
    from src.casino.feature_flags import is_feature_enabled as _is_few_shot_enabled

    if await _is_few_shot_enabled(settings.CASINO_ID, "few_shot_examples_enabled"):
        specialist_examples = FEW_SHOT_EXAMPLES.get(agent_name, [])
        if specialist_examples:
            # Cap at 3 most relevant examples to limit prompt size
            capped = specialist_examples[:3]
            examples_parts = []
            for user_ex, response_ex in capped:
                examples_parts.append(f'Guest: "{user_ex}"\nYou: "{response_ex}"')
            examples_text = "\n\n".join(examples_parts)
            system_prompt += (
                f"\n\n## Response Examples (match this style)\n\n{examples_text}"
            )
            logger.info(
                "R83: Injected %d few-shot examples for %s agent",
                len(capped),
                agent_name,
            )

    # R82 Track 1E: Frustration/crisis suppression of promotional content.
    # When the guest is frustrated/negative, override promotional specialist prompts
    # with factual, empathy-first guidance. This is a HARD override — the specialist
    # prompt template is already loaded but this section takes priority.
    # Note: `guest_sentiment` here is the effective sentiment (post-sarcasm-override)
    # returned by _build_behavioral_prompt_sections.
    _PROMOTIONAL_AGENTS = frozenset({"comp", "promotions"})
    if (
        guest_sentiment in ("frustrated", "negative")
        and agent_name in _PROMOTIONAL_AGENTS
    ):
        frustration_override = (
            "\n\n## OVERRIDE: Guest Is Frustrated — Suppress Promotional Tone\n"
            "The guest is upset or frustrated. Your ENTIRE response must follow these rules:\n"
            "1. NO promotional language. No 'explore rewards', 'benefits shine', 'exciting perks', "
            "'you're going to love this'.\n"
            "2. Be factual and direct. State what they qualify for, not how great it is.\n"
            "3. Acknowledge their frustration FIRST before any comp/loyalty information.\n"
            "4. Keep response under 3 sentences.\n"
            "5. End with: 'Would you like me to connect you with a dedicated host who can "
            "assist you personally?'\n"
            "6. If you don't have specific comp information for them, just empathize and "
            "offer the human host connection. Do NOT default to generic promotions."
        )
        system_prompt += frustration_override
        logger.info(
            "R82 1E: Frustration suppression activated for %s agent (sentiment=%s)",
            agent_name,
            guest_sentiment,
        )

    # R82 Track 1E: Crisis state suppression across ALL specialists.
    # If crisis_active is True, ALL specialists should avoid any promotional content.
    if state.get("crisis_active", False):
        crisis_override = (
            "\n\n## OVERRIDE: Crisis State Active\n"
            "The guest may be in distress. Your response must:\n"
            "1. NOT include any promotional content, upselling, or rewards mentions.\n"
            "2. Be compassionate and brief.\n"
            "3. Include crisis resources if relevant.\n"
            "4. Offer to connect with a human host."
        )
        system_prompt += crisis_override
        logger.info("R82 1E: Crisis suppression activated for %s agent", agent_name)

    # Profiling Intelligence: inject guest profile summary and profiling guidance
    from src.casino.feature_flags import DEFAULT_FEATURES as _DEFAULT_FEATURES

    if _DEFAULT_FEATURES.get("profiling_enabled", True):
        from src.agent.extraction import get_guest_profile_summary

        profile_summary = get_guest_profile_summary(extracted)
        if profile_summary:
            system_prompt += f"\n\n## {profile_summary}"

        # R78 fix: Inject profiling question guidance from whisper plan.
        # Strengthened from "weave naturally" (ignored by LLM 90%+ of the time)
        # to explicit mandatory instruction with format guidance.
        whisper = state.get("whisper_plan")
        if whisper and whisper.get("next_profiling_question"):
            technique = whisper.get("question_technique", "none")
            if technique != "none" and guest_sentiment not in (
                "frustrated",
                "negative",
                "grief",
            ):
                from src.agent.profiling import PROFILING_TECHNIQUE_PROMPTS

                technique_guide = PROFILING_TECHNIQUE_PROMPTS.get(technique, "")
                question = whisper["next_profiling_question"]
                system_prompt += (
                    "\n\n## REQUIRED: Ask This Question\n"
                    "After answering the guest's question, you MUST include the following "
                    "question naturally at the end of your response. Do NOT skip it.\n\n"
                    f'Question to ask: "{question}"\n'
                    f"Technique: {technique_guide}\n\n"
                    "Format: Answer the guest's question first (2-3 sentences). Then add a "
                    "natural transition and ask the question. Example format:\n"
                    "[Your answer here]\n\n"
                    "[Natural transition + question]\n\n"
                    "If the guest is upset, rushed, or gave a one-word reply, SKIP the question."
                )

    # R85 fix: Wire incentive engine into specialist prompt.
    # get_incentive_prompt_section is sync (no I/O) — safe to call inline.
    # Only injects when profile_completeness >= 50% and matching rules exist.
    # R87: Returns tuple (prompt_section, approval_request_or_none).
    _incentive_approval: dict | None = None
    if _DEFAULT_FEATURES.get("incentives_enabled", True):
        from src.agent.incentives import get_incentive_prompt_section

        _completeness = state.get("profile_completeness_score", 0.0)
        incentive_section, _incentive_approval = get_incentive_prompt_section(
            casino_id=settings.CASINO_ID,
            profile_completeness=_completeness,
            extracted_fields=extracted,
        )
        if incentive_section:
            system_prompt += f"\n\n{incentive_section}"
            logger.info(
                "R85: Incentive section injected (completeness=%.2f, agent=%s)",
                _completeness,
                agent_name,
            )

    # R98: CompStrategy — deterministic comp policy injection for comp agent.
    # R99 lesson: expanding to ALL agents caused 1.43-point H-avg regression
    # (prompt pollution). Comp context stays comp-only. For H9, improve dispatch
    # routing to send comp-related queries to comp agent instead.
    if agent_name == "comp":
        from src.agent.behavior_tools.comp_strategy import get_comp_prompt_section

        comp_section = get_comp_prompt_section(state, casino_id=settings.CASINO_ID)
        if comp_section:
            system_prompt += comp_section
            logger.info("R98: CompStrategy section injected (agent=%s)", agent_name)

    # R98: Rapport Ladder — micro-pattern retrieval for rapport building.
    # Provides context-specific conversation techniques per guest type.
    from src.agent.behavior_tools.rapport_ladder import get_rapport_prompt_section

    rapport_section = get_rapport_prompt_section(state)
    if rapport_section:
        system_prompt += rapport_section

    # R98: LTV Nudge Engine — return-visit seeding.
    # Plants forward-looking hooks for lifetime value optimization.
    from src.agent.behavior_tools.ltv_nudge import get_ltv_prompt_section

    ltv_section = get_ltv_prompt_section(state, casino_id=settings.CASINO_ID)
    if ltv_section:
        system_prompt += ltv_section

    # R74 B4 / R82 1F: Proactive suggestion injection via extracted helper.
    # Gated by: confidence >= 0.6 (R82: lowered from 0.8), sentiment not
    # negative/frustrated, max 1 per session (suggestion_offered), grounding exists.
    proactive_section, suggestion_already_offered = _should_inject_suggestion(
        state,
        guest_sentiment,
        dynamics,
    )
    if proactive_section:
        system_prompt += proactive_section

    # Build message list
    llm_messages = [SystemMessage(content=system_prompt)]

    # Compute conversation history for persona reinject and sliding window
    history = [
        m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))
    ]

    # Phase 4: Persona drift prevention — re-inject condensed persona reminder
    # when conversation history exceeds threshold. Research: persona consistency
    # drops 20-40% over 10-15 turns without reinforcement.
    # R23 fix H-002: count only HumanMessage instances for threshold check
    # (retries add extra AIMessages that inflate the count).
    human_turn_count = sum(1 for m in history if isinstance(m, HumanMessage))
    if human_turn_count > _PERSONA_REINJECT_THRESHOLD // 2:
        # R23/R29 fix: read persona name from property-specific profile
        try:
            _persona_name = (
                get_casino_profile(settings.CASINO_ID)
                .get("branding", {})
                .get("persona_name", "Seven")
            )
        except Exception:
            _persona_name = "Seven"
        llm_messages.append(
            SystemMessage(
                content=(
                    f"PERSONA REMINDER: You are {_persona_name}, the AI concierge for "
                    f"{settings.PROPERTY_NAME}. Maintain your warm, professional tone. "
                    "Never provide gambling advice or discuss competitors. "
                    "Always stay on-property topics. Be concise and helpful."
                )
            )
        )

    # On retry, inject feedback
    if retry_count > 0 and retry_feedback:
        llm_messages.append(
            SystemMessage(
                content=Template(
                    "IMPORTANT: Your previous response failed validation. Reason: $feedback. "
                    "Please generate a corrected response that addresses this issue."
                ).safe_substitute(feedback=retry_feedback)
            )
        )

    # Sliding window on conversation history.
    # On retry, exclude the last AIMessage (the one that failed validation)
    # to prevent the LLM from parroting the invalid response.
    if retry_count > 0 and history and isinstance(history[-1], AIMessage):
        history = history[:-1]
    window = history[-settings.MAX_HISTORY_MESSAGES :]
    llm_messages.extend(window)

    # R83: Model routing — use Pro model when dispatch layer determined complex routing.
    # The dispatch layer sets model_used in the state dict passed to the agent.
    # If it matches COMPLEX_MODEL_NAME, use _get_complex_llm() instead.
    _model_used = state.get("model_used")
    if _model_used and _model_used == settings.COMPLEX_MODEL_NAME:
        from src.agent.nodes import _get_complex_llm

        llm = await _get_complex_llm()
        logger.info("R83: Using Pro model (%s) for %s agent", _model_used, agent_name)
    else:
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
                "messages": [
                    AIMessage(content=_fallback_message("high demand right now"))
                ],
                "skip_validation": True,
            }

        try:
            response = await llm.ainvoke(llm_messages)
        except (ValueError, TypeError) as exc:
            await cb.record_failure()
            logger.warning(
                "%s agent LLM response parsing failed: %s", agent_name.capitalize(), exc
            )
            # ValueError/TypeError may produce malformed content that still
            # warrants validation.  Incrementing retry_count ensures the
            # validator runs; the validate_node's "retry_count < 1" check
            # prevents unbounded retries.  Only circuit-breaker-open and
            # network-error paths use skip_validation=True.
            # R10 fix (DeepSeek F3): was hard-coded to 1 which reset the
            # retry budget when retry_count was already >= 1.
            return {
                "messages": [
                    AIMessage(
                        content=_fallback_message("trouble processing that response")
                    )
                ],
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
                "messages": [
                    AIMessage(
                        content=_fallback_message(
                            "trouble generating a response right now"
                        )
                    )
                ],
                "skip_validation": True,
            }
    finally:
        if acquired:
            _LLM_SEMAPHORE.release()

    await cb.record_success()
    content = _normalize_content(response.content)

    # R82 Track 1G (partial): Response length budgets per intent.
    # Applied post-generation to enforce limits the LLM ignores from prompt instructions.
    # Truncates to the nearest sentence boundary within the word budget.
    _MAX_WORDS_BY_INTENT: dict[str, int] = {
        "greeting": 50,
        "acknowledgment": 40,
        "confirmation": 40,
        "off_topic": 60,
    }
    _query_type = state.get("query_type", "")
    _max_words = _MAX_WORDS_BY_INTENT.get(_query_type or "", 0)
    if _max_words > 0 and content:
        words = content.split()
        if len(words) > _max_words:
            truncated = " ".join(words[:_max_words])
            # Find last sentence boundary — don't cut mid-sentence
            for punct in (".", "?", "!"):
                last_punct = truncated.rfind(punct)
                if last_punct > len(truncated) // 2:
                    truncated = truncated[: last_punct + 1]
                    break
            logger.info(
                "R82 1G: Response truncated from %d to %d words for %s",
                len(words),
                len(truncated.split()),
                _query_type,
            )
            content = truncated

    result: dict = {"messages": [AIMessage(content=content)]}
    # R23 fix C-003: persist suggestion_offered flag across turns
    if suggestion_already_offered:
        result["suggestion_offered"] = True  # _keep_truthy: once True, stays True

    # R87: Wire incentive approval handoff for above-threshold comps.
    # Lower priority than frustration handoff (frustration overwrites below).
    if _incentive_approval:
        result["handoff_request"] = _incentive_approval

    # Phase 5: Wire handoff for persistent frustration (overwrites incentive handoff).
    # R98: Enhanced with structured handoff summary from HandoffOrchestrator.
    # R100 fix P9: Lowered threshold from 3 to 2 consecutive frustrated messages.
    # Also detect repeated questions (guest asking same thing twice = agent failed).
    _repeated = dynamics.get("repeated_question", False)
    _handoff_trigger = frustrated_count >= 2 or (frustrated_count >= 1 and _repeated)
    if _handoff_trigger:
        from src.agent.handoff import build_handoff_request
        from src.agent.behavior_tools.handoff import build_handoff_summary

        _reason = (
            f"Guest frustrated ({frustrated_count} consecutive) + repeated question"
            if _repeated
            else f"Guest frustrated across {frustrated_count}+ consecutive messages"
        )
        handoff_req = build_handoff_request(
            department="vip_services",
            reason=_reason,
            extracted_fields=state.get("extracted_fields"),
            urgency="high",
        )
        summary = build_handoff_summary(state, handoff_reason=_reason)
        handoff_dict = handoff_req.model_dump()
        handoff_dict["structured_summary"] = summary.model_dump()
        result["handoff_request"] = handoff_dict

    return result
