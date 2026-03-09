"""LangGraph tool definitions for casino domain operations.

Provides 4 @tool functions that specialist agents can call mid-conversation
via LangGraph's tool-use pattern. Each tool wraps existing pure business
logic or parses structured knowledge-base data.

These tools replace prompt-section injection for dimensions where the LLM
needs REAL data to integrate naturally (H9, H10, H3, P6).

R106: Architecture shift — tool-use instead of prompt engineering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Knowledge-base data (parsed once at import time, immutable)
# ---------------------------------------------------------------------------

_KB_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "knowledge-base"
    / "casino-operations"
)


def _load_tier_data() -> MappingProxyType[str, dict[str, str]]:
    """Parse momentum-tiers.md into structured tier lookup."""
    tiers: dict[str, dict[str, str]] = {}
    tier_path = _KB_ROOT / "momentum-tiers.md"

    if not tier_path.exists():
        logger.warning("momentum-tiers.md not found at %s", tier_path)
        return MappingProxyType({})

    content = tier_path.read_text(encoding="utf-8")
    current_tier = ""
    current_text: list[str] = []

    for line in content.split("\n"):
        if line.startswith("## Momentum Tier:"):
            if current_tier and current_text:
                tiers[current_tier] = {
                    "name": current_tier,
                    "description": "\n".join(current_text).strip(),
                }
            current_tier = line.replace("## Momentum Tier:", "").strip().lower()
            current_text = []
        elif line.startswith("## ") and current_tier:
            # New non-tier section — flush current tier
            if current_tier and current_text:
                tiers[current_tier] = {
                    "name": current_tier,
                    "description": "\n".join(current_text).strip(),
                }
            current_tier = ""
            current_text = []

            # Capture non-tier sections (Promotions, Redemption, etc.)
            section_name = line.replace("## ", "").strip().lower()
            section_lines: list[str] = []
            # We'll capture these in the next iteration
            tiers[section_name] = {"name": section_name, "description": ""}
        elif current_tier:
            current_text.append(line)

    # Flush last tier
    if current_tier and current_text:
        tiers[current_tier] = {
            "name": current_tier,
            "description": "\n".join(current_text).strip(),
        }

    return MappingProxyType(tiers)


def _load_entertainment_data() -> MappingProxyType[str, dict[str, str]]:
    """Parse entertainment-guide.md into structured venue lookup."""
    venues: dict[str, dict[str, str]] = {}
    ent_path = _KB_ROOT / "entertainment-guide.md"

    if not ent_path.exists():
        logger.warning("entertainment-guide.md not found at %s", ent_path)
        return MappingProxyType({})

    content = ent_path.read_text(encoding="utf-8")
    current_venue = ""
    current_text: list[str] = []

    for line in content.split("\n"):
        if line.startswith("## "):
            if current_venue and current_text:
                venues[current_venue] = {
                    "name": current_venue,
                    "description": "\n".join(current_text).strip(),
                }
            current_venue = line.replace("## ", "").strip()
            current_text = []
        elif current_venue:
            current_text.append(line)

    # Flush last venue
    if current_venue and current_text:
        venues[current_venue] = {
            "name": current_venue,
            "description": "\n".join(current_text).strip(),
        }

    return MappingProxyType(venues)


_TIER_DATA: MappingProxyType[str, dict[str, str]] = _load_tier_data()
_ENTERTAINMENT_DATA: MappingProxyType[str, dict[str, str]] = _load_entertainment_data()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@tool
def check_comp_eligibility(
    guest_tier: str = "new",
    occasion: str = "",
) -> str:
    """Check what comps and rewards a guest may be eligible for based on their loyalty tier and any special occasion.

    Use this when the guest asks about rewards, comps, free play, or loyalty benefits.
    Also use when you want to proactively mention a comp opportunity.

    Args:
        guest_tier: Guest's loyalty tier level. One of: new, regular, vip, high_roller.
        occasion: Special occasion if any (birthday, anniversary, celebration, etc.).
    """
    from src.agent.behavior_tools.comp_strategy import (
        CompStrategyInput,
        get_comp_strategy,
    )

    # Map guest_tier to valid Literal type
    valid_tiers: dict[str, Literal["new", "regular", "vip", "high_roller"]] = {
        "new": "new",
        "regular": "regular",
        "vip": "vip",
        "high_roller": "high_roller",
    }
    tier = valid_tiers.get(guest_tier.lower().replace(" ", "_"), "new")

    input_data = CompStrategyInput(
        guest_tier=tier,
        current_occasion=occasion if occasion else None,
        property_id="mohegan_sun",
    )
    result = get_comp_strategy(input_data)

    if not result.eligible_comps:
        return "No specific comp offers found for this tier level. The guest can sign up for the Momentum Rewards program to start earning toward comps."

    lines: list[str] = [f"Guest tier: {result.tier_name}"]
    lines.append("Eligible comps:")
    for offer in result.eligible_comps:
        approval = (
            " (auto-approved)" if offer.auto_approve else " (check with host first)"
        )
        lines.append(f"- {offer.description} (~${offer.estimated_value:.0f}){approval}")
        lines.append(f"  Natural framing: {offer.framing}")

    if result.talking_points:
        lines.append("\nTalking points:")
        for point in result.talking_points:
            lines.append(f"- {point}")

    if result.restrictions:
        lines.append("\nRules:")
        for r in result.restrictions:
            lines.append(f"- {r}")

    return "\n".join(lines)


@tool
def check_tier_status(
    tier_name: str = "",
    query: str = "",
) -> str:
    """Look up benefits and perks for a specific Momentum Rewards loyalty tier.

    Use this when the guest asks about their tier benefits, how to advance tiers,
    or what perks are available at different loyalty levels.

    Args:
        tier_name: Tier to look up (core, ignite, leap, ascend, soar). Leave empty for general info.
        query: Optional specific question about the tier (e.g., "how many credits to advance").
    """
    if not _TIER_DATA:
        return "Tier information is not available right now. Please check with the Momentum Desk for current tier details."

    # Normalize tier name
    tier_key = tier_name.lower().strip() if tier_name else ""

    if tier_key and tier_key in _TIER_DATA:
        tier_info = _TIER_DATA[tier_key]
        return f"Momentum Tier: {tier_info['name'].title()}\n{tier_info['description']}"

    # Fuzzy match — check if query mentions a tier
    all_tier_names = ["core", "ignite", "leap", "ascend", "soar"]
    search_text = (tier_key + " " + query).lower()

    for name in all_tier_names:
        if name in search_text and name in _TIER_DATA:
            tier_info = _TIER_DATA[name]
            return f"Momentum Tier: {tier_info['name'].title()}\n{tier_info['description']}"

    # No specific tier — return overview of all tiers
    lines = ["Momentum Rewards Tiers (from entry to highest):"]
    for name in all_tier_names:
        if name in _TIER_DATA:
            desc = _TIER_DATA[name]["description"]
            # First sentence only for overview
            first_sentence = desc.split(". ")[0] + "." if ". " in desc else desc[:150]
            lines.append(f"- {name.title()}: {first_sentence}")

    # Check for non-tier sections (promotions, redemption, etc.)
    for key in (
        "momentum points redemption",
        "current promotions",
        "tier reset schedule",
    ):
        if key in search_text and key in _TIER_DATA:
            info = _TIER_DATA[key]
            if info["description"]:
                lines.append(f"\n{info['name'].title()}:\n{info['description']}")

    return "\n".join(lines)


@tool
def lookup_upcoming_events(
    venue_type: str = "all",
    interest: str = "",
) -> str:
    """Find entertainment venues, shows, and events at the property.

    Use this when the guest asks about shows, concerts, comedy, entertainment,
    or things to do. Also use to suggest entertainment as part of an evening plan.

    Args:
        venue_type: Type of venue to search (arena, comedy, wolf_den, outdoor, family, all).
        interest: Guest's specific interest (e.g., "comedy", "rock music", "family friendly").
    """
    if not _ENTERTAINMENT_DATA:
        return "Entertainment information is not available right now. Check mohegansun.com/entertainment for current listings."

    venue_key = venue_type.lower().strip() if venue_type else "all"
    interest_lower = interest.lower().strip() if interest else ""

    # Map venue_type aliases to actual section names
    venue_aliases: dict[str, list[str]] = {
        "arena": ["Mohegan Sun Arena"],
        "comedy": ["Comix Comedy Club"],
        "wolf_den": ["Wolf Den"],
        "outdoor": ["Outdoor Events & Seasonal Programming"],
        "family": ["Family-Friendly Entertainment"],
        "tickets": ["Tickets & VIP Packages"],
        "all": list(_ENTERTAINMENT_DATA.keys()),
    }

    target_venues = venue_aliases.get(venue_key, list(_ENTERTAINMENT_DATA.keys()))

    # If interest is specified, search all venues for relevant content
    if interest_lower and venue_key == "all":
        matching: list[str] = []
        for venue_name, venue_info in _ENTERTAINMENT_DATA.items():
            if (
                interest_lower in venue_info["description"].lower()
                or interest_lower in venue_name.lower()
            ):
                matching.append(
                    f"**{venue_info['name']}**\n{venue_info['description']}"
                )
        if matching:
            return "\n\n".join(matching)
        # No match — return all venues
        target_venues = list(_ENTERTAINMENT_DATA.keys())

    lines: list[str] = []
    for venue_name in target_venues:
        if venue_name in _ENTERTAINMENT_DATA:
            info = _ENTERTAINMENT_DATA[venue_name]
            lines.append(f"**{info['name']}**\n{info['description']}")

    if not lines:
        return "No matching entertainment venues found. The Mohegan Sun Box Office at mohegansun.com/entertainment has the latest listings."

    return "\n\n".join(lines)


@tool
def check_incentive_eligibility(
    occasion: str = "",
    profile_completeness: float = 0.0,
    guest_tier: str = "new",
) -> str:
    """Check if the guest qualifies for any incentive offers based on their profile information.

    Use this when you've learned something about the guest (birthday, anniversary,
    gaming preferences) and want to check if there's a matching incentive to offer.

    Args:
        occasion: Special occasion mentioned by guest (birthday, anniversary, etc.).
        profile_completeness: How complete the guest's profile is (0.0 to 1.0).
        guest_tier: Guest's loyalty tier (new, regular, vip, high_roller).
    """
    from src.agent.incentives import IncentiveEngine
    from src.config import get_settings

    settings = get_settings()
    casino_id = settings.CASINO_ID
    property_name = settings.PROPERTY_NAME

    engine = IncentiveEngine(casino_id)

    # Build extracted_fields from the tool inputs
    extracted: dict[str, str] = {}
    if occasion:
        occasion_lower = occasion.lower()
        if "birthday" in occasion_lower:
            extracted["birthday"] = occasion
            extracted["occasion"] = occasion
        elif "anniversar" in occasion_lower:
            extracted["anniversary"] = occasion
            extracted["occasion"] = occasion
        else:
            extracted["occasion"] = occasion

    applicable = engine.get_applicable_incentives(
        profile_completeness=profile_completeness,
        extracted_fields=extracted,
    )

    if not applicable:
        return "No specific incentive offers match the current guest profile. Continue building rapport and gathering information — offers may become available as we learn more."

    lines: list[str] = ["Available incentive offers:"]
    for rule in applicable:
        offer_text = engine.format_incentive_offer(
            rule,
            {
                "property_name": property_name,
                "value": f"{rule.incentive_value:.0f}"
                if rule.incentive_value
                else "a complimentary",
                "incentive_type": rule.incentive_type.replace("_", " "),
            },
        )
        auto = engine.check_auto_approve(rule)
        approval_note = (
            " (you can offer this directly)" if auto else " (check with host first)"
        )
        lines.append(f"- {offer_text}{approval_note}")

    lines.append(
        "\nRemember: weave this in naturally as a pleasant surprise, not a sales pitch."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# All tools list (for import by tool_binding.py)
# ---------------------------------------------------------------------------

ALL_CASINO_TOOLS = [
    check_comp_eligibility,
    check_tier_status,
    lookup_upcoming_events,
    check_incentive_eligibility,
]
