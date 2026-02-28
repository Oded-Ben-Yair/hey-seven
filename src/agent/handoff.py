"""Warm handoff protocol for transferring guests to human casino hosts.

When the AI concierge determines that a guest needs human assistance
(crisis situations, persistent frustration, complex requests beyond
AI capability), this module structures the handoff with guest context.

The handoff is emitted as an SSE event of type "handoff" so the frontend
can display appropriate UI (e.g., "Connecting you with a host...").
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = ["HandoffRequest", "build_handoff_request"]


class HandoffRequest(BaseModel):
    """Structured handoff request for human host transfer."""

    type: Literal["handoff"] = "handoff"
    department: str = Field(
        description="Target department: responsible_gaming, vip_services, front_desk, general"
    )
    reason: str = Field(
        description="Brief reason for handoff (shown to receiving host)"
    )
    guest_summary: str = Field(
        default="",
        description="Summary of guest profile and conversation context",
    )
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Urgency level for host queue prioritization",
    )
    consent_given: bool = Field(
        default=False,
        description="Whether the guest explicitly consented to the handoff",
    )


def build_handoff_request(
    department: str,
    reason: str,
    extracted_fields: dict[str, Any] | None = None,
    urgency: str = "medium",
    consent_given: bool = False,
) -> HandoffRequest:
    """Build a structured handoff request with guest context.

    Args:
        department: Target department for the handoff.
        reason: Why the handoff is needed.
        extracted_fields: Guest profile data from state.
        urgency: Priority level.
        consent_given: Whether guest agreed to transfer.

    Returns:
        HandoffRequest ready for SSE emission.
    """
    summary_parts = []
    if extracted_fields:
        if extracted_fields.get("name"):
            summary_parts.append(f"Guest: {extracted_fields['name']}")
        if extracted_fields.get("party_size"):
            summary_parts.append(f"Party size: {extracted_fields['party_size']}")
        if extracted_fields.get("occasion"):
            summary_parts.append(f"Occasion: {extracted_fields['occasion']}")
        if extracted_fields.get("loyalty_signal"):
            summary_parts.append(f"Loyalty: {extracted_fields['loyalty_signal']}")

    guest_summary = " | ".join(summary_parts) if summary_parts else ""

    return HandoffRequest(
        department=department,
        reason=reason,
        guest_summary=guest_summary,
        urgency=urgency,
        consent_given=consent_given,
    )
