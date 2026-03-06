"""Deterministic comp policy engine for specialist agents.

Calculates comp eligibility based on guest tier, ADT, visit frequency,
and occasion. Returns structured prompt sections with specific comp
offers, talking points, and restrictions.

Pure business logic — no LLM calls, no I/O. Per-casino comp policies
stored as immutable module-level data.

Feature flag: ``comp_agent_enabled`` (checked via DEFAULT_FEATURES).

Targets H9 (Comp Strategy): 1.9 → 5.0+
"""

from __future__ import annotations

import logging
from string import Template
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "CompTier",
    "CompOffer",
    "CompStrategyInput",
    "CompStrategyOutput",
    "get_comp_strategy",
    "get_comp_prompt_section",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CompOffer(BaseModel):
    """A single comp offer with value and conditions."""

    comp_type: str  # "dining_credit", "room_upgrade", "free_play", "show_tickets", "spa_credit"
    description: str
    estimated_value: float = Field(ge=0.0)
    auto_approve: bool = True
    conditions: list[str] = Field(default_factory=list)
    framing: str  # Natural language for the agent to use


class CompTier(BaseModel):
    """Comp tier definition with ADT ranges and available offers."""

    tier_name: str
    adt_min: float = Field(ge=0.0)
    adt_max: float | None = None
    base_offers: list[CompOffer] = Field(default_factory=list)
    occasion_multiplier: float = Field(default=1.0, ge=1.0)


class CompStrategyInput(BaseModel):
    """Input for comp strategy calculation."""

    guest_tier: Literal["new", "regular", "vip", "high_roller"] = "new"
    estimated_adt: float | None = None
    visit_frequency: Literal["first", "occasional", "regular", "weekly"] = "first"
    current_occasion: str | None = None
    property_id: str = "mohegan_sun"
    guest_sentiment: str | None = None


class CompStrategyOutput(BaseModel):
    """Output from comp strategy calculation."""

    eligible_comps: list[CompOffer] = Field(default_factory=list)
    approval_required: bool = False
    talking_points: list[str] = Field(default_factory=list)
    restrictions: list[str] = Field(default_factory=list)
    tier_name: str = "exploration"


# ---------------------------------------------------------------------------
# Per-casino comp policies (immutable)
# ---------------------------------------------------------------------------

_EXPLORATION_OFFERS: tuple[CompOffer, ...] = (
    CompOffer(
        comp_type="dining_credit",
        description="Complimentary appetizer or dessert at select restaurants",
        estimated_value=15.0,
        auto_approve=True,
        conditions=["Valid at casual dining venues", "One per visit"],
        framing="I can arrange a complimentary appetizer or dessert at one of our casual dining spots.",
    ),
    CompOffer(
        comp_type="free_play",
        description="Introductory free play bonus",
        estimated_value=10.0,
        auto_approve=True,
        conditions=["New loyalty program members", "Slots or video poker"],
        framing="As a new guest, you're eligible for a $10 introductory free play bonus when you sign up for our rewards program.",
    ),
)

_REGULAR_OFFERS: tuple[CompOffer, ...] = (
    CompOffer(
        comp_type="dining_credit",
        description="Dining credit at any restaurant",
        estimated_value=25.0,
        auto_approve=True,
        conditions=["Based on recent play activity", "Valid for 30 days"],
        framing="Based on your recent visits, I can offer you a $25 dining credit at any of our restaurants.",
    ),
    CompOffer(
        comp_type="show_tickets",
        description="Complimentary show tickets",
        estimated_value=40.0,
        auto_approve=True,
        conditions=["Subject to availability", "Select shows only"],
        framing="I may be able to arrange complimentary tickets to an upcoming show — let me check availability for you.",
    ),
    CompOffer(
        comp_type="free_play",
        description="Loyalty free play",
        estimated_value=25.0,
        auto_approve=True,
        conditions=["Based on tier status", "Valid same day"],
        framing="Your play history qualifies you for a $25 free play offer.",
    ),
)

_VIP_OFFERS: tuple[CompOffer, ...] = (
    CompOffer(
        comp_type="room_upgrade",
        description="Suite upgrade on next stay",
        estimated_value=150.0,
        auto_approve=False,
        conditions=["Subject to availability", "Based on loyalty tier"],
        framing="Given your loyalty status, I'd like to look into a suite upgrade for your next stay. Let me check with our reservations team.",
    ),
    CompOffer(
        comp_type="dining_credit",
        description="Fine dining comp",
        estimated_value=75.0,
        auto_approve=False,
        conditions=["Fine dining venues", "Based on ADT"],
        framing="I can look into a dining comp at one of our fine dining restaurants for you.",
    ),
    CompOffer(
        comp_type="spa_credit",
        description="Spa treatment credit",
        estimated_value=50.0,
        auto_approve=True,
        conditions=["Select treatments", "Reservation required"],
        framing="You're eligible for a $50 spa credit — would you like me to share what treatments are available?",
    ),
    CompOffer(
        comp_type="show_tickets",
        description="Premium show tickets with VIP seating",
        estimated_value=100.0,
        auto_approve=False,
        conditions=["VIP seating section", "Select headliner shows"],
        framing="I can check on VIP seating for an upcoming headliner show.",
    ),
)

_HIGH_ROLLER_OFFERS: tuple[CompOffer, ...] = (
    CompOffer(
        comp_type="room_upgrade",
        description="Premium suite comp",
        estimated_value=500.0,
        auto_approve=False,
        conditions=["Premium suites", "Based on ADT and stay history"],
        framing="Let me have your host look into our premium suite availability for your visit.",
    ),
    CompOffer(
        comp_type="dining_credit",
        description="Full dining experience comp",
        estimated_value=200.0,
        auto_approve=False,
        conditions=["Any venue including fine dining", "Based on play history"],
        framing="I can arrange a fully comped dining experience — which of our restaurants interests you?",
    ),
    CompOffer(
        comp_type="free_play",
        description="High-value free play",
        estimated_value=250.0,
        auto_approve=False,
        conditions=["Based on ADT and recent play", "Valid 7 days"],
        framing="Based on your play level, I'll have your host review a free play offer for you.",
    ),
)


def _build_casino_tiers() -> dict[str, tuple[CompTier, ...]]:
    """Build per-casino comp tiers. Each casino can have custom tiers."""
    # Shared tier structure across casinos (customize per-casino as needed)
    base_tiers: tuple[CompTier, ...] = (
        CompTier(
            tier_name="exploration",
            adt_min=0.0,
            adt_max=50.0,
            base_offers=list(_EXPLORATION_OFFERS),
            occasion_multiplier=1.0,
        ),
        CompTier(
            tier_name="regular",
            adt_min=50.0,
            adt_max=200.0,
            base_offers=list(_REGULAR_OFFERS),
            occasion_multiplier=1.25,
        ),
        CompTier(
            tier_name="vip",
            adt_min=200.0,
            adt_max=1000.0,
            base_offers=list(_VIP_OFFERS),
            occasion_multiplier=1.5,
        ),
        CompTier(
            tier_name="high_roller",
            adt_min=1000.0,
            adt_max=None,
            base_offers=list(_HIGH_ROLLER_OFFERS),
            occasion_multiplier=2.0,
        ),
    )

    return {
        "mohegan_sun": base_tiers,
        "foxwoods": base_tiers,
        "parx_casino": base_tiers,
        "wynn_las_vegas": base_tiers,
        "hard_rock_ac": base_tiers,
    }


COMP_TIERS: MappingProxyType[str, tuple[CompTier, ...]] = MappingProxyType(
    _build_casino_tiers()
)

_DEFAULT_TIERS: tuple[CompTier, ...] = (
    CompTier(
        tier_name="exploration",
        adt_min=0.0,
        adt_max=50.0,
        base_offers=list(_EXPLORATION_OFFERS),
    ),
)

# Occasion keywords that trigger multiplier
_OCCASION_KEYWORDS: MappingProxyType[str, str] = MappingProxyType(
    {
        "birthday": "birthday",
        "anniversary": "anniversary",
        "celebration": "celebration",
        "honeymoon": "celebration",
        "wedding": "celebration",
        "bachelor": "celebration",
        "bachelorette": "celebration",
        "retirement": "celebration",
        "promotion": "celebration",
        "graduation": "celebration",
    }
)

# Tier mapping from guest_tier string to ADT estimate when ADT is unknown
_TIER_ADT_ESTIMATES: MappingProxyType[str, float] = MappingProxyType(
    {
        "new": 25.0,
        "regular": 100.0,
        "vip": 500.0,
        "high_roller": 2000.0,
    }
)

# Auto-approve threshold (comps above this require host approval)
_AUTO_APPROVE_THRESHOLD = 50.0


# ---------------------------------------------------------------------------
# Comp strategy engine
# ---------------------------------------------------------------------------


def _resolve_tier(
    casino_id: str,
    estimated_adt: float,
) -> CompTier:
    """Resolve the comp tier based on casino and ADT."""
    tiers = COMP_TIERS.get(casino_id, _DEFAULT_TIERS)
    for tier in reversed(tiers):  # Check highest tier first
        if estimated_adt >= tier.adt_min:
            return tier
    return tiers[0]


def _detect_occasion(occasion_str: str | None) -> str | None:
    """Detect occasion type from free-text occasion string."""
    if not occasion_str:
        return None
    lower = occasion_str.lower()
    for keyword, occasion_type in _OCCASION_KEYWORDS.items():
        if keyword in lower:
            return occasion_type
    return None


def get_comp_strategy(input_data: CompStrategyInput) -> CompStrategyOutput:
    """Calculate comp strategy based on guest profile and casino policy.

    Pure deterministic business logic. No LLM calls, no I/O.

    Args:
        input_data: Guest tier, ADT, visit frequency, occasion, property.

    Returns:
        CompStrategyOutput with eligible comps, talking points, restrictions.
    """
    # Resolve ADT (use estimate if not provided)
    adt = input_data.estimated_adt
    if adt is None:
        adt = _TIER_ADT_ESTIMATES.get(input_data.guest_tier, 25.0)

    # Find matching tier
    tier = _resolve_tier(input_data.property_id, adt)

    # Detect occasion for multiplier
    occasion = _detect_occasion(input_data.current_occasion)
    multiplier = tier.occasion_multiplier if occasion else 1.0

    # Filter offers: skip high-value items for negative sentiment
    skip_high_value = input_data.guest_sentiment in ("frustrated", "negative", "grief")

    eligible: list[CompOffer] = []
    any_requires_approval = False

    for offer in tier.base_offers:
        adjusted_value = offer.estimated_value * multiplier

        # Don't push high-value comps on frustrated/grieving guests
        if skip_high_value and adjusted_value > _AUTO_APPROVE_THRESHOLD:
            continue

        auto = adjusted_value <= _AUTO_APPROVE_THRESHOLD
        if not auto:
            any_requires_approval = True

        eligible.append(
            CompOffer(
                comp_type=offer.comp_type,
                description=offer.description,
                estimated_value=adjusted_value,
                auto_approve=auto,
                conditions=list(offer.conditions),
                framing=offer.framing,
            )
        )

    # Build talking points
    talking_points: list[str] = []
    if occasion:
        talking_points.append(
            f"Acknowledge the {occasion} — enhanced offers available for special occasions."
        )
    if input_data.visit_frequency in ("regular", "weekly"):
        talking_points.append(
            "Recognize their loyalty and regular visits — mention how their play contributes to comp eligibility."
        )
    if input_data.guest_tier == "new":
        talking_points.append(
            "Welcome them warmly. Mention the rewards program as a way to earn comps over time."
        )
    if any_requires_approval:
        talking_points.append(
            "For higher-value comps, let the guest know you'll check with their host or player services."
        )

    # Build restrictions
    restrictions: list[str] = [
        "Never promise specific dollar amounts — say 'I can look into' or 'you may be eligible'.",
        "All comps subject to availability and terms.",
    ]
    if input_data.guest_tier == "new":
        restrictions.append(
            "New guests: focus on rewards program signup and introductory offers."
        )

    return CompStrategyOutput(
        eligible_comps=eligible,
        approval_required=any_requires_approval,
        talking_points=talking_points,
        restrictions=restrictions,
        tier_name=tier.tier_name,
    )


# ---------------------------------------------------------------------------
# Integration: system prompt section builder
# ---------------------------------------------------------------------------


def get_comp_prompt_section(
    state: dict[str, Any],
    casino_id: str,
) -> str:
    """Build a comp strategy prompt section for the comp agent.

    Called by comp_agent to inject specific comp policy into the LLM
    system prompt. Returns empty string when no comps are applicable.

    Args:
        state: Current graph state (for extracted_fields, sentiment).
        casino_id: Casino identifier.

    Returns:
        Formatted prompt section string, or "" if no comps apply.
    """
    extracted = state.get("extracted_fields") or {}
    sentiment = state.get("guest_sentiment")

    # Infer guest tier from loyalty signals
    loyalty_signal = extracted.get("loyalty_signal", "")
    loyalty_tier = extracted.get("loyalty_tier", "")
    guest_tier: Literal["new", "regular", "vip", "high_roller"] = "new"
    if loyalty_tier:
        lower_tier = loyalty_tier.lower()
        if any(w in lower_tier for w in ("soar", "ascend", "platinum", "diamond")):
            guest_tier = "high_roller"
        elif any(w in lower_tier for w in ("leap", "gold")):
            guest_tier = "vip"
        elif any(w in lower_tier for w in ("ignite", "silver", "core")):
            guest_tier = "regular"
    elif loyalty_signal:
        lower_signal = loyalty_signal.lower()
        if any(w in lower_signal for w in ("regular", "years", "frequent", "always")):
            guest_tier = "regular"

    # Infer visit frequency
    visit_freq: Literal["first", "occasional", "regular", "weekly"] = "first"
    if guest_tier in ("vip", "high_roller"):
        visit_freq = "regular"
    elif guest_tier == "regular":
        visit_freq = "occasional"

    input_data = CompStrategyInput(
        guest_tier=guest_tier,
        estimated_adt=None,  # No ADT data from conversation
        visit_frequency=visit_freq,
        current_occasion=extracted.get("occasion"),
        property_id=casino_id,
        guest_sentiment=sentiment,
    )

    result = get_comp_strategy(input_data)

    if not result.eligible_comps:
        return ""

    lines: list[str] = [
        "\n\n## Comp Strategy (use naturally, never force or list all at once)"
    ]
    lines.append(f"Guest tier: {result.tier_name}")

    lines.append("\n### Available Comps")
    for offer in result.eligible_comps:
        approval_note = "" if offer.auto_approve else " [CHECK WITH HOST FIRST]"
        lines.append(
            f"- {offer.description} (~${offer.estimated_value:.0f}){approval_note}"
        )
        lines.append(f'  Say: "{offer.framing}"')

    if result.talking_points:
        lines.append("\n### Talking Points")
        for point in result.talking_points:
            lines.append(f"- {point}")

    if result.restrictions:
        lines.append("\n### Rules")
        for restriction in result.restrictions:
            lines.append(f"- {restriction}")

    return "\n".join(lines)
