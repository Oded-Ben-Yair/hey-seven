"""Incentive engine for guest profile-driven offers.

Determines which incentives to surface based on guest profile completeness
and extracted fields. Pure business logic -- no LLM calls, no I/O.

Each casino has its own incentive rules (trigger conditions, values, framing).
The engine evaluates rules against the current profile state and returns
applicable incentives for the specialist agent's system prompt.

Feature flag: ``incentives_enabled`` (checked via DEFAULT_FEATURES sync lookup).
"""

from __future__ import annotations

import copy
import logging
from string import Template
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "IncentiveRule",
    "IncentiveEngine",
    "get_incentive_prompt_section",
    "INCENTIVE_RULES",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class IncentiveRule(BaseModel):
    """A single incentive rule tied to a profile trigger.

    Attributes:
        trigger_field: Which profile field triggers this rule. One of:
            ``"birthday"``, ``"email"``, ``"gaming_preference"``,
            ``"profile_completeness_75"``.
        incentive_type: Category of incentive. One of:
            ``"dining_credit"``, ``"free_play"``, ``"tier_points"``,
            ``"comp_upgrade"``.
        incentive_value: Dollar value of the incentive (0.0 for non-monetary).
        max_per_guest: Maximum times this incentive can be offered per guest.
            Prevents gaming the system. Default 1.
        auto_approve_threshold: Maximum dollar value the agent can offer
            without requiring host approval. Default 50.0.
        framing_template: Template string for natural language presentation.
            Uses ``string.Template`` placeholders: ``$property_name``,
            ``$value``, ``$incentive_type``. Use ``$$$value`` to render
            as ``$25`` (``$$`` = literal dollar sign, ``$value`` = placeholder).
    """

    trigger_field: str
    incentive_type: str
    incentive_value: float = Field(ge=0.0)
    max_per_guest: int = Field(default=1, ge=1)
    auto_approve_threshold: float = Field(default=50.0, ge=0.0)
    framing_template: str


# ---------------------------------------------------------------------------
# Per-casino incentive configurations (immutable at module level)
# ---------------------------------------------------------------------------

_MOHEGAN_SUN_RULES: tuple[IncentiveRule, ...] = (
    IncentiveRule(
        trigger_field="birthday",
        incentive_type="dining_credit",
        incentive_value=25.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "Since your birthday is coming up, $property_name would love to "
            "treat you to a $$$value dining credit at any of our restaurants."
        ),
    ),
    IncentiveRule(
        trigger_field="profile_completeness_75",
        incentive_type="free_play",
        incentive_value=10.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "As a thank you for sharing your preferences, here's a $$$value "
            "free play bonus to enjoy on the gaming floor at $property_name."
        ),
    ),
    IncentiveRule(
        trigger_field="anniversary",
        incentive_type="comp_upgrade",
        incentive_value=0.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "Happy anniversary! Let me see about upgrading your experience "
            "at $property_name to make it extra special."
        ),
    ),
)

_FOXWOODS_RULES: tuple[IncentiveRule, ...] = (
    IncentiveRule(
        trigger_field="birthday",
        incentive_type="dining_credit",
        incentive_value=20.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "Happy birthday! $property_name would like to offer you a $$$value "
            "dining credit to celebrate."
        ),
    ),
    IncentiveRule(
        trigger_field="profile_completeness_75",
        incentive_type="tier_points",
        incentive_value=15.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "Thanks for letting us get to know you! Here are $$$value bonus "
            "tier points for your rewards account at $property_name."
        ),
    ),
)

_PARX_RULES: tuple[IncentiveRule, ...] = (
    IncentiveRule(
        trigger_field="birthday",
        incentive_type="free_play",
        incentive_value=15.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "Happy birthday from $property_name! Enjoy $$$value in free play "
            "on us."
        ),
    ),
    IncentiveRule(
        trigger_field="profile_completeness_75",
        incentive_type="dining_credit",
        incentive_value=10.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "Thanks for sharing your preferences with us. Here's a $$$value "
            "dining credit at $property_name."
        ),
    ),
)

_WYNN_RULES: tuple[IncentiveRule, ...] = (
    IncentiveRule(
        trigger_field="birthday",
        incentive_type="dining_credit",
        incentive_value=50.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "In celebration of your birthday, $property_name is pleased to "
            "extend a $$$value dining credit for any of our Forbes-rated restaurants."
        ),
    ),
    IncentiveRule(
        trigger_field="profile_completeness_75",
        incentive_type="free_play",
        incentive_value=25.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "Thank you for sharing your preferences. Please enjoy $$$value "
            "in complimentary play at $property_name."
        ),
    ),
)

_HARD_ROCK_AC_RULES: tuple[IncentiveRule, ...] = (
    IncentiveRule(
        trigger_field="birthday",
        incentive_type="free_play",
        incentive_value=20.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "Rock on -- it's your birthday! $property_name is hooking you up "
            "with $$$value in free play."
        ),
    ),
    IncentiveRule(
        trigger_field="profile_completeness_75",
        incentive_type="tier_points",
        incentive_value=10.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "Thanks for telling us what you like! Here are $$$value bonus "
            "tier points at $property_name."
        ),
    ),
    IncentiveRule(
        trigger_field="gaming_preference",
        incentive_type="free_play",
        incentive_value=5.0,
        max_per_guest=1,
        auto_approve_threshold=50.0,
        framing_template=(
            "Since you mentioned you enjoy $incentive_type, here's a $$$value "
            "free play bonus to try your luck at $property_name."
        ),
    ),
)

# Conservative defaults for unknown casinos
_DEFAULT_RULES: tuple[IncentiveRule, ...] = (
    IncentiveRule(
        trigger_field="birthday",
        incentive_type="dining_credit",
        incentive_value=10.0,
        max_per_guest=1,
        auto_approve_threshold=25.0,
        framing_template=(
            "Happy birthday! $property_name would like to offer you a $$$value "
            "dining credit."
        ),
    ),
)

# Immutable mapping: casino_id -> tuple of rules.
# MappingProxyType prevents accidental mutation of module-level data.
INCENTIVE_RULES: MappingProxyType[str, tuple[IncentiveRule, ...]] = MappingProxyType({
    "mohegan_sun": _MOHEGAN_SUN_RULES,
    "foxwoods": _FOXWOODS_RULES,
    "parx_casino": _PARX_RULES,
    "wynn_las_vegas": _WYNN_RULES,
    "hard_rock_ac": _HARD_ROCK_AC_RULES,
})

_DEFAULT_INCENTIVE_RULES: tuple[IncentiveRule, ...] = _DEFAULT_RULES


# ---------------------------------------------------------------------------
# Incentive engine
# ---------------------------------------------------------------------------


class IncentiveEngine:
    """Evaluates incentive rules against guest profile state.

    Pure business logic -- no I/O, no LLM calls. Instantiate per-request
    with the casino_id to load the appropriate rule set.

    Args:
        casino_id: Casino identifier (e.g., ``"mohegan_sun"``).
    """

    def __init__(self, casino_id: str) -> None:
        self._casino_id = casino_id
        self._rules: tuple[IncentiveRule, ...] = INCENTIVE_RULES.get(
            casino_id, _DEFAULT_INCENTIVE_RULES
        )

    @property
    def casino_id(self) -> str:
        return self._casino_id

    @property
    def rules(self) -> tuple[IncentiveRule, ...]:
        return self._rules

    def get_applicable_incentives(
        self,
        profile_completeness: float,
        extracted_fields: dict[str, Any],
    ) -> list[IncentiveRule]:
        """Return incentive rules whose trigger conditions are met.

        Trigger conditions:
        - ``"birthday"``: ``extracted_fields`` contains a truthy ``birthday`` key.
        - ``"email"``: ``extracted_fields`` contains a truthy ``email`` key.
        - ``"gaming_preference"``: ``extracted_fields`` contains a truthy
          ``gaming_preference`` key.
        - ``"anniversary"``: ``extracted_fields`` contains a truthy
          ``anniversary`` or ``occasion`` key with anniversary-related value.
        - ``"profile_completeness_75"``: ``profile_completeness >= 0.75``.

        Args:
            profile_completeness: Float 0.0-1.0 representing profile fill rate.
            extracted_fields: Dict of extracted guest data from conversation.

        Returns:
            List of applicable IncentiveRule instances (may be empty).
        """
        applicable: list[IncentiveRule] = []

        for rule in self._rules:
            if self._is_triggered(rule, profile_completeness, extracted_fields):
                applicable.append(rule)

        return applicable

    def _is_triggered(
        self,
        rule: IncentiveRule,
        profile_completeness: float,
        extracted_fields: dict[str, Any],
    ) -> bool:
        """Check if a single rule's trigger condition is met."""
        trigger = rule.trigger_field

        if trigger == "profile_completeness_75":
            return profile_completeness >= 0.75

        if trigger == "anniversary":
            if extracted_fields.get("anniversary"):
                return True
            occasion = extracted_fields.get("occasion", "")
            if isinstance(occasion, str) and "anniversar" in occasion.lower():
                return True
            return False

        # Generic field presence check (birthday, email, gaming_preference, etc.)
        return bool(extracted_fields.get(trigger))

    def format_incentive_offer(
        self,
        rule: IncentiveRule,
        context: dict[str, Any],
    ) -> str:
        """Format a natural language incentive offer from a rule's template.

        Uses ``string.Template.safe_substitute()`` to prevent KeyError on
        missing placeholders.

        Args:
            rule: The incentive rule to format.
            context: Dict with substitution values. Expected keys:
                ``property_name``, ``value``, ``incentive_type``.

        Returns:
            Formatted offer string.
        """
        return Template(rule.framing_template).safe_substitute(context)

    def check_auto_approve(self, rule: IncentiveRule) -> bool:
        """Check if an incentive can be auto-approved without host review.

        Returns True if the incentive's dollar value is at or below the
        auto-approve threshold. Non-monetary incentives (value=0.0) are
        always auto-approved.

        Args:
            rule: The incentive rule to check.

        Returns:
            True if auto-approved, False if host approval required.
        """
        return rule.incentive_value <= rule.auto_approve_threshold

    def build_host_approval_request(
        self,
        rule: IncentiveRule,
        guest_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a structured approval request for high-value incentives.

        Used when ``check_auto_approve()`` returns False. The returned dict
        can be serialized and sent to a human host dashboard.

        Args:
            rule: The incentive rule requiring approval.
            guest_context: Guest information (name, phone, loyalty tier, etc.).

        Returns:
            Dict with approval request fields.
        """
        return {
            "casino_id": self._casino_id,
            "incentive_type": rule.incentive_type,
            "incentive_value": rule.incentive_value,
            "trigger_field": rule.trigger_field,
            "auto_approve_threshold": rule.auto_approve_threshold,
            "guest_context": copy.deepcopy(guest_context),
            "requires_approval": True,
        }


# ---------------------------------------------------------------------------
# Integration point: system prompt section builder
# ---------------------------------------------------------------------------


def get_incentive_prompt_section(
    casino_id: str,
    profile_completeness: float,
    extracted_fields: dict[str, Any],
) -> str:
    """Build a system prompt section listing applicable incentives.

    Called by specialist agents to inject incentive awareness into the LLM
    system prompt. Returns an empty string when no incentives apply or when
    the feature is disabled.

    Feature gate: checks ``incentives_enabled`` from DEFAULT_FEATURES
    (synchronous, no I/O). This is a Layer 1 (build-time) check -- runtime
    per-casino overrides via Firestore are not consulted here to avoid
    async in a sync function.

    Args:
        casino_id: Casino identifier.
        profile_completeness: Float 0.0-1.0.
        extracted_fields: Dict of extracted guest data.

    Returns:
        A prompt section string (may be empty).
    """
    from src.casino.feature_flags import DEFAULT_FEATURES

    if not DEFAULT_FEATURES.get("incentives_enabled", True):
        return ""

    engine = IncentiveEngine(casino_id)
    applicable = engine.get_applicable_incentives(
        profile_completeness, extracted_fields,
    )

    if not applicable:
        return ""

    from src.config import get_settings

    settings = get_settings()
    property_name = settings.PROPERTY_NAME

    lines: list[str] = ["## Available Incentives (offer naturally, do not force)"]

    for rule in applicable:
        offer_text = engine.format_incentive_offer(rule, {
            "property_name": property_name,
            "value": f"{rule.incentive_value:.0f}" if rule.incentive_value else "a complimentary",
            "incentive_type": rule.incentive_type.replace("_", " "),
        })
        approval_note = ""
        if not engine.check_auto_approve(rule):
            approval_note = " [REQUIRES HOST APPROVAL -- do not promise, say you will check]"
        lines.append(f"- {offer_text}{approval_note}")

    lines.append(
        "Weave these naturally into the conversation. Do NOT lead with incentives "
        "or list them unprompted. Only mention when contextually relevant."
    )

    return "\n".join(lines)
