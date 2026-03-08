"""Return-visit seeding engine for lifetime value optimization.

Plants forward-looking hooks in specialist agent responses: upcoming
events, seasonal offers, loyalty milestones, and personal callbacks.

Pure business logic — no LLM calls, no I/O. Per-casino nudge rules
stored as immutable module-level data.

Skips nudges for grief/crisis/frustrated sentiment (safety first).

Targets H10 (Lifetime Value): 3.5 → 5.0+
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "LTVNudge",
    "get_ltv_nudges",
    "get_ltv_prompt_section",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class LTVNudge(BaseModel):
    """A single return-visit nudge to inject into agent response."""

    nudge_type: Literal[
        "upcoming_event",
        "seasonal_offer",
        "loyalty_milestone",
        "personal_callback",
        "new_experience",
    ]
    message_fragment: str
    timing: str  # "next visit", "this weekend", "next month"
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Per-casino nudge catalogs (immutable)
# ---------------------------------------------------------------------------

_MOHEGAN_SUN_NUDGES: tuple[LTVNudge, ...] = (
    LTVNudge(
        nudge_type="upcoming_event",
        message_fragment=(
            "By the way, the Mohegan Sun Arena has some great shows coming up "
            "— I can share what's on the calendar if you're interested."
        ),
        timing="upcoming",
        relevance=0.7,
    ),
    LTVNudge(
        nudge_type="seasonal_offer",
        message_fragment=(
            "We often have seasonal dining promotions — next time you visit, "
            "ask about what's running and I can point you to the best deals."
        ),
        timing="next visit",
        relevance=0.6,
    ),
    LTVNudge(
        nudge_type="new_experience",
        message_fragment=(
            "Have you tried our spa? It's a nice complement to a gaming weekend "
            "— something to keep in mind for your next trip."
        ),
        timing="next visit",
        relevance=0.5,
    ),
    LTVNudge(
        nudge_type="loyalty_milestone",
        message_fragment=(
            "If you keep building those tier credits, you'll unlock some really "
            "nice perks at the next level — worth checking your progress."
        ),
        timing="ongoing",
        relevance=0.6,
    ),
    LTVNudge(
        nudge_type="personal_callback",
        message_fragment=(
            "Next time you're planning a visit, feel free to reach out ahead "
            "of time — I can help set things up so everything's ready when you arrive."
        ),
        timing="next visit",
        relevance=0.7,
    ),
)

_FOXWOODS_NUDGES: tuple[LTVNudge, ...] = (
    LTVNudge(
        nudge_type="upcoming_event",
        message_fragment=(
            "Foxwoods always has something happening — from concerts at the Grand "
            "Theater to special gaming tournaments. Worth checking the calendar."
        ),
        timing="upcoming",
        relevance=0.7,
    ),
    LTVNudge(
        nudge_type="seasonal_offer",
        message_fragment=(
            "We run seasonal promotions throughout the year — next visit, ask me "
            "what's current and I'll point you to the best options."
        ),
        timing="next visit",
        relevance=0.6,
    ),
    LTVNudge(
        nudge_type="new_experience",
        message_fragment=(
            "If you haven't explored our outdoor spaces, the Tanger Outlets and "
            "the Foxwoods Golf Course are great additions to a resort weekend."
        ),
        timing="next visit",
        relevance=0.5,
    ),
    LTVNudge(
        nudge_type="personal_callback",
        message_fragment=(
            "Feel free to reach out before your next trip — I can help with "
            "reservations, show tickets, or anything else to make the visit smooth."
        ),
        timing="next visit",
        relevance=0.7,
    ),
)

_WYNN_NUDGES: tuple[LTVNudge, ...] = (
    LTVNudge(
        nudge_type="upcoming_event",
        message_fragment=(
            "The Wynn frequently hosts exclusive events and residency shows. "
            "I'd be happy to share upcoming highlights."
        ),
        timing="upcoming",
        relevance=0.7,
    ),
    LTVNudge(
        nudge_type="seasonal_offer",
        message_fragment=(
            "Our Forbes-rated restaurants sometimes feature special seasonal menus "
            "— a wonderful reason to plan a return visit."
        ),
        timing="next visit",
        relevance=0.7,
    ),
    LTVNudge(
        nudge_type="new_experience",
        message_fragment=(
            "If you haven't experienced the Lake of Dreams, it's one of those "
            "things that makes a Wynn visit truly memorable."
        ),
        timing="next visit",
        relevance=0.6,
    ),
    LTVNudge(
        nudge_type="personal_callback",
        message_fragment=(
            "For your next visit, I can help arrange everything in advance — "
            "dining reservations, spa appointments, show tickets."
        ),
        timing="next visit",
        relevance=0.8,
    ),
)

_DEFAULT_NUDGES: tuple[LTVNudge, ...] = (
    LTVNudge(
        nudge_type="personal_callback",
        message_fragment=(
            "Next time you're planning a visit, reach out ahead of time "
            "and I can help set things up for you."
        ),
        timing="next visit",
        relevance=0.6,
    ),
    LTVNudge(
        nudge_type="loyalty_milestone",
        message_fragment=(
            "The rewards program has some great benefits as you tier up — "
            "worth keeping an eye on your progress."
        ),
        timing="ongoing",
        relevance=0.5,
    ),
)

NUDGE_CATALOG: MappingProxyType[str, tuple[LTVNudge, ...]] = MappingProxyType(
    {
        "mohegan_sun": _MOHEGAN_SUN_NUDGES,
        "foxwoods": _FOXWOODS_NUDGES,
        "wynn_las_vegas": _WYNN_NUDGES,
        "parx_casino": _DEFAULT_NUDGES,
        "hard_rock_ac": _DEFAULT_NUDGES,
    }
)


# Sentiments that suppress nudges (safety first)
_SUPPRESS_SENTIMENTS = frozenset({"grief", "crisis", "frustrated", "negative"})

# Domains that trigger specific nudge types
_DOMAIN_NUDGE_AFFINITY: MappingProxyType[str, str] = MappingProxyType(
    {
        "dining": "seasonal_offer",
        "entertainment": "upcoming_event",
        "gaming": "loyalty_milestone",
        "spa": "new_experience",
        "hotel": "personal_callback",
        "comp": "loyalty_milestone",
    }
)


# ---------------------------------------------------------------------------
# Nudge selection engine
# ---------------------------------------------------------------------------


def get_ltv_nudges(
    casino_id: str,
    domains_discussed: list[str] | None = None,
    guest_sentiment: str | None = None,
    occasion: str | None = None,
    turn_count: int = 0,
    extracted_fields: dict[str, Any] | None = None,
) -> list[LTVNudge]:
    """Select relevant LTV nudges based on context and guest profile.

    Returns up to 2 nudges ranked by relevance to current conversation.
    Returns empty list for suppressed sentiments or very short conversations.

    R105: Profile-aware nudge selection. Uses extracted_fields to boost
    nudges matching guest interests (entertainment, gaming, spa, visit
    frequency). Occasion-specific nudge text is generated when occasion
    is present.

    Args:
        casino_id: Casino identifier.
        domains_discussed: List of conversation domains.
        guest_sentiment: Current guest sentiment.
        occasion: Guest's occasion (birthday, anniversary, etc.).
        turn_count: Number of human turns in conversation.
        extracted_fields: Guest profile data from extraction (R105).

    Returns:
        List of up to 2 relevant LTVNudge instances.
    """
    # Safety: no nudges for distressed guests
    if guest_sentiment in _SUPPRESS_SENTIMENTS:
        return []

    # No nudges for very short conversations (< 2 turns)
    if turn_count < 2:
        return []

    catalog = NUDGE_CATALOG.get(casino_id, _DEFAULT_NUDGES)
    domains = domains_discussed or []
    profile = extracted_fields or {}

    # Score nudges by relevance to conversation context
    scored: list[tuple[float, LTVNudge]] = []
    for nudge in catalog:
        score = nudge.relevance

        # Boost nudges matching discussed domains
        for domain in domains:
            affinity = _DOMAIN_NUDGE_AFFINITY.get(domain)
            if affinity and affinity == nudge.nudge_type:
                score += 0.3
                break

        # Boost personal_callback for longer conversations (rapport built)
        if nudge.nudge_type == "personal_callback" and turn_count >= 4:
            score += 0.2

        # Boost occasion-related nudges
        if occasion and nudge.nudge_type in ("seasonal_offer", "upcoming_event"):
            score += 0.15

        # R105: Profile-aware boosts from extracted_fields
        score += _profile_boost(nudge.nudge_type, profile)

        scored.append((score, nudge))

    # Sort by score descending, take top 2
    scored.sort(key=lambda x: x[0], reverse=True)
    return [nudge for _, nudge in scored[:2]]


# R105: Keywords that indicate spa/relaxation interest in preferences
_SPA_KEYWORDS: frozenset[str] = frozenset(
    {
        "spa",
        "massage",
        "relax",
        "relaxation",
        "wellness",
        "treatment",
        "sauna",
        "facial",
    }
)

# R105: Keywords for regular/frequent visit patterns
_REGULAR_VISIT_KEYWORDS: frozenset[str] = frozenset(
    {"weekly", "regular", "every week", "frequent", "often", "all the time", "always"}
)


def _profile_boost(nudge_type: str, profile: dict[str, Any]) -> float:
    """Calculate score boost from guest profile data.

    Pure deterministic. Returns 0.0 when no relevant profile data exists.
    """
    boost = 0.0

    # Entertainment interest boosts upcoming_event
    if nudge_type == "upcoming_event":
        entertainment = profile.get("entertainment_interests") or profile.get(
            "entertainment"
        )
        if entertainment:
            boost += 0.4

    # Gaming preference boosts loyalty_milestone
    if nudge_type == "loyalty_milestone":
        gaming = profile.get("gaming_preferences") or profile.get("gaming")
        if gaming:
            boost += 0.4

    # Spa/relaxation preference boosts new_experience
    if nudge_type == "new_experience":
        preferences = str(profile.get("preferences", "")).lower()
        if preferences and any(kw in preferences for kw in _SPA_KEYWORDS):
            boost += 0.3

    # Regular visitor boosts personal_callback
    if nudge_type == "personal_callback":
        visit_freq = str(profile.get("visit_frequency", "")).lower()
        if visit_freq and any(kw in visit_freq for kw in _REGULAR_VISIT_KEYWORDS):
            boost += 0.3

    return boost


# ---------------------------------------------------------------------------
# Integration: system prompt section builder
# ---------------------------------------------------------------------------


def get_ltv_prompt_section(
    state: dict[str, Any],
    casino_id: str,
) -> str:
    """Build an LTV nudge prompt section for specialist agents.

    Called by execute_specialist() in _base.py to inject return-visit
    hooks into specialist responses.

    Args:
        state: Current graph state.
        casino_id: Casino identifier.

    Returns:
        Formatted prompt section string, or "" if no nudges apply.
    """
    domains = state.get("domains_discussed", [])
    sentiment = state.get("guest_sentiment")
    extracted = state.get("extracted_fields") or {}
    occasion = extracted.get("occasion")
    messages = state.get("messages", [])

    from langchain_core.messages import HumanMessage as _HM

    turn_count = sum(1 for m in messages if isinstance(m, _HM))

    nudges = get_ltv_nudges(
        casino_id=casino_id,
        domains_discussed=domains,
        guest_sentiment=sentiment,
        occasion=occasion,
        turn_count=turn_count,
        extracted_fields=extracted,  # R105: pass profile data for profile-aware nudges
    )

    if not nudges:
        return ""

    lines: list[str] = [
        "\n\n## Return Visit Seeding (weave ONE naturally, never force)"
    ]
    lines.append(
        "If the conversation reaches a natural close or transition, "
        "include ONE of these forward-looking hooks:"
    )
    for nudge in nudges:
        lines.append(f'- "{nudge.message_fragment}"')

    lines.append(
        "Do NOT list multiple nudges. Pick the one that fits the conversation flow."
    )

    return "\n".join(lines)
