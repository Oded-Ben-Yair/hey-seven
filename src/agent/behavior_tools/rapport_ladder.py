"""Rapport building micro-pattern retrieval for specialist agents.

Provides context-specific conversation techniques based on guest type
and conversation dynamics. Each technique includes a concrete example
the agent can adapt.

Pure business logic — no LLM calls, no I/O.

Targets H6 (Rapport Depth): 4.0 → 5.0+
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "RapportPattern",
    "get_rapport_patterns",
    "get_rapport_prompt_section",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

GuestType = Literal[
    "first_timer",
    "regular",
    "vip",
    "family",
    "couple",
    "solo",
    "grieving",
    "celebrating",
]


class RapportPattern(BaseModel):
    """A micro-pattern for building rapport with a specific guest type."""

    technique: str
    description: str
    example: str
    applicable_guest_types: list[GuestType] = Field(default_factory=list)
    conversation_phase: Literal["opening", "exploring", "deciding", "closing"] = (
        "exploring"
    )


# ---------------------------------------------------------------------------
# Rapport technique catalog (immutable)
# ---------------------------------------------------------------------------

_TECHNIQUES: tuple[RapportPattern, ...] = (
    RapportPattern(
        technique="callback",
        description="Reference something the guest mentioned earlier in the conversation",
        example=(
            "You mentioned earlier that you love Italian food — "
            "our Italian restaurant just launched a new seasonal menu you might enjoy."
        ),
        applicable_guest_types=["regular", "vip", "couple", "family"],
        conversation_phase="exploring",
    ),
    RapportPattern(
        technique="knowledge_flex",
        description="Share a specific insider detail that shows deep property knowledge",
        example=(
            "Fun fact — the chef at that restaurant actually trained in Tuscany "
            "and brings in fresh pasta ingredients weekly."
        ),
        applicable_guest_types=["first_timer", "regular", "vip", "couple", "solo"],
        conversation_phase="exploring",
    ),
    RapportPattern(
        technique="anticipatory",
        description="Proactively suggest something before the guest asks",
        example=(
            "Since you're celebrating, I'd suggest making reservations early — "
            "our fine dining spots fill up fast on weekends."
        ),
        applicable_guest_types=["celebrating", "couple", "vip", "family"],
        conversation_phase="deciding",
    ),
    RapportPattern(
        technique="insider_tip",
        description="Share a local tip or lesser-known feature",
        example=(
            "Pro tip — if you're here on a weekday, the brunch at our bistro "
            "is a hidden gem that most guests don't know about."
        ),
        applicable_guest_types=["first_timer", "regular", "solo"],
        conversation_phase="exploring",
    ),
    RapportPattern(
        technique="shared_excitement",
        description="Match the guest's enthusiasm about their plans",
        example=(
            "That show is going to be amazing — I've heard great things. "
            "Dinner before at our steakhouse would make it a perfect evening."
        ),
        applicable_guest_types=["celebrating", "couple", "family", "first_timer"],
        conversation_phase="exploring",
    ),
    RapportPattern(
        technique="empathetic_mirror",
        description="Acknowledge and validate the guest's emotional state",
        example=(
            "I understand this is a meaningful trip for you. Let me make sure "
            "we get everything just right."
        ),
        applicable_guest_types=["grieving", "celebrating", "solo"],
        conversation_phase="opening",
    ),
    RapportPattern(
        technique="name_callback",
        description="Use the guest's name naturally in context",
        example=(
            "Sarah, based on what you've told me, I think you'd really "
            "enjoy the sunset views from our rooftop bar."
        ),
        applicable_guest_types=[
            "first_timer",
            "regular",
            "vip",
            "family",
            "couple",
            "solo",
            "celebrating",
        ],
        conversation_phase="exploring",
    ),
    RapportPattern(
        technique="experience_bridge",
        description="Connect the current topic to a related experience they'd enjoy",
        example=(
            "Since you enjoyed the spa, you might also like our yoga classes "
            "on the terrace — they're complimentary for hotel guests."
        ),
        applicable_guest_types=["regular", "vip", "couple", "solo"],
        conversation_phase="exploring",
    ),
    RapportPattern(
        technique="return_anchor",
        description="Plant a reason to come back without being pushy",
        example=(
            "That restaurant is launching a new tasting menu next month — "
            "worth keeping in mind for your next visit."
        ),
        applicable_guest_types=["regular", "vip", "couple", "first_timer"],
        conversation_phase="closing",
    ),
    RapportPattern(
        technique="family_focus",
        description="Acknowledge the family dynamic and tailor suggestions",
        example=(
            "With the kids, you might enjoy our family arcade "
            "— it's a nice break between meals and they'll love it."
        ),
        applicable_guest_types=["family"],
        conversation_phase="exploring",
    ),
)


# ---------------------------------------------------------------------------
# Guest type inference
# ---------------------------------------------------------------------------

# Sentiment → guest type mapping for emotional context
_SENTIMENT_GUEST_TYPE: MappingProxyType[str, GuestType] = MappingProxyType(
    {
        "grief": "grieving",
        "celebration": "celebrating",
        "excited": "celebrating",
    }
)


def _infer_guest_type(
    extracted: dict[str, Any],
    sentiment: str | None,
    turn_count: int,
) -> GuestType:
    """Infer guest type from profile data and sentiment."""
    # Sentiment-driven types take priority
    if sentiment and sentiment in _SENTIMENT_GUEST_TYPE:
        return _SENTIMENT_GUEST_TYPE[sentiment]

    # Check occasion
    occasion = extracted.get("occasion", "")
    if isinstance(occasion, str):
        lower_occ = occasion.lower()
        if any(
            w in lower_occ
            for w in ("birthday", "anniversary", "celebration", "wedding")
        ):
            return "celebrating"

    # Check party composition
    party = extracted.get("party_composition", "")
    party_size = extracted.get("party_size")
    if isinstance(party, str) and any(
        w in party.lower() for w in ("kid", "child", "family", "son", "daughter")
    ):
        return "family"
    if isinstance(party, str) and any(
        w in party.lower()
        for w in ("wife", "husband", "partner", "girlfriend", "boyfriend")
    ):
        return "couple"

    # Check loyalty for VIP/regular
    loyalty = extracted.get("loyalty_tier", "")
    loyalty_signal = extracted.get("loyalty_signal", "")
    if isinstance(loyalty, str) and any(
        w in loyalty.lower() for w in ("soar", "ascend", "platinum", "diamond")
    ):
        return "vip"
    if isinstance(loyalty, str) and any(
        w in loyalty.lower() for w in ("leap", "ignite", "gold", "silver")
    ):
        return "regular"
    if isinstance(loyalty_signal, str) and any(
        w in loyalty_signal.lower() for w in ("regular", "always", "years", "frequent")
    ):
        return "regular"

    # Conversation length as proxy
    if turn_count <= 1:
        return "first_timer"

    # Default: solo guest
    if party_size and str(party_size).strip() == "1":
        return "solo"

    return "first_timer"


# ---------------------------------------------------------------------------
# Pattern selection engine
# ---------------------------------------------------------------------------


def get_rapport_patterns(
    guest_type: GuestType,
    conversation_phase: str = "exploring",
    guest_name: str | None = None,
    domains_discussed: list[str] | None = None,
) -> list[RapportPattern]:
    """Select rapport patterns for a specific guest type and phase.

    Returns up to 3 patterns ranked by relevance.

    Args:
        guest_type: Inferred guest type.
        conversation_phase: Current conversation phase.
        guest_name: Guest name if known (enables name_callback).
        domains_discussed: Topics already covered.

    Returns:
        List of up to 3 relevant RapportPattern instances.
    """
    candidates: list[tuple[float, RapportPattern]] = []

    for pattern in _TECHNIQUES:
        if guest_type not in pattern.applicable_guest_types:
            continue

        score = 0.5

        # Boost patterns matching conversation phase
        if pattern.conversation_phase == conversation_phase:
            score += 0.3

        # Boost callback/name_callback if guest name is known
        if pattern.technique in ("callback", "name_callback") and guest_name:
            score += 0.25

        # Skip name_callback if no name known
        if pattern.technique == "name_callback" and not guest_name:
            continue

        # Boost experience_bridge if multiple domains discussed
        if (
            pattern.technique == "experience_bridge"
            and domains_discussed
            and len(domains_discussed) >= 2
        ):
            score += 0.2

        # Boost return_anchor for closing phase
        if pattern.technique == "return_anchor" and conversation_phase == "closing":
            score += 0.3

        candidates.append((score, pattern))

    # Sort by score descending, take top 3
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates[:3]]


# ---------------------------------------------------------------------------
# Integration: system prompt section builder
# ---------------------------------------------------------------------------


def get_rapport_prompt_section(
    state: dict[str, Any],
) -> str:
    """Build a rapport technique prompt section for specialist agents.

    Called by execute_specialist() in _base.py to inject rapport-building
    techniques into the specialist system prompt.

    Args:
        state: Current graph state.

    Returns:
        Formatted prompt section string, or "" if no patterns apply.
    """
    extracted = state.get("extracted_fields") or {}
    sentiment = state.get("guest_sentiment")
    domains = state.get("domains_discussed", [])
    messages = state.get("messages", [])
    guest_name = extracted.get("name") or state.get("guest_name")

    from langchain_core.messages import HumanMessage as _HM

    turn_count = sum(1 for m in messages if isinstance(m, _HM))

    # Don't inject rapport techniques for crisis/frustrated
    if sentiment in ("grief", "crisis"):
        # Grieving guests get empathetic_mirror only
        guest_type: GuestType = "grieving"
    elif sentiment in ("frustrated", "negative"):
        return ""  # No rapport techniques for frustrated guests
    else:
        guest_type = _infer_guest_type(extracted, sentiment, turn_count)

    # R105: Use profiling phase from state for better phase mapping,
    # with turn-count as fallback when profiling_phase is not set.
    profiling_phase = state.get("profiling_phase")

    if profiling_phase == "foundation":
        phase = "opening"
    elif profiling_phase == "preference":
        phase = "exploring"
    elif profiling_phase == "relationship":
        phase = "deciding"
    elif profiling_phase == "behavioral":
        phase = "closing"
    else:
        # Fallback to turn-count-based phase detection
        if turn_count <= 1:
            phase = "opening"
        elif turn_count <= 4:
            phase = "exploring"
        elif turn_count <= 7:
            phase = "deciding"
        else:
            phase = "closing"

    patterns = get_rapport_patterns(
        guest_type=guest_type,
        conversation_phase=phase,
        guest_name=guest_name,
        domains_discussed=domains,
    )

    if not patterns:
        return ""

    lines: list[str] = [
        "\n\n## Rapport Technique (use ONE naturally, adapt to context)"
    ]
    lines.append(f"Guest type: {guest_type} | Phase: {phase}")

    for pattern in patterns[:2]:  # Inject max 2 to avoid overwhelming
        lines.append(f"\n**{pattern.technique}**: {pattern.description}")
        lines.append(f'Example: "{pattern.example}"')

    lines.append(
        "\nAdapt the technique to fit what the guest is actually asking about. "
        "Do NOT copy examples verbatim — use them as inspiration."
    )

    return "\n".join(lines)
