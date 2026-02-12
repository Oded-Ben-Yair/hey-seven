"""Casino-specific tool definitions for the Host Agent.

Each tool represents a discrete capability the agent can invoke during
conversations. Tools use the @tool decorator from langchain_core and return
structured data. In production, these would call Firestore, the casino's
PMS (Property Management System), and external APIs. The current
implementations return realistic stub data for demonstration.
"""

import random
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Player Management
# ---------------------------------------------------------------------------


@tool
def check_player_status(player_id: str) -> dict:
    """Look up a player's profile, loyalty tier, and recent visit history.

    Queries the casino's player tracking system to retrieve the full player
    profile. This is typically the first tool called when a host begins
    interacting with or discussing a specific player.

    Args:
        player_id: The unique player tracking number (e.g., "PLY-482910").

    Returns:
        A dict containing:
            - player_id: Echo of the input ID.
            - name: Player's full name.
            - tier: Loyalty tier (Diamond, Platinum, Gold, Silver, Base).
            - adt: Average Daily Theoretical (in USD).
            - total_theo_ytd: Year-to-date theoretical win.
            - comp_balance: Available comp dollar balance.
            - last_visit: ISO date of most recent property visit.
            - visit_count_ytd: Number of visits this year.
            - preferences: Dict of known preferences (game, drink, room, dining).
            - host_assigned: Name of assigned human host, if any.
            - status: "active", "inactive", or "self_excluded".
    """
    # Stub: realistic demo data keyed by player_id suffix
    profiles = {
        "PLY-482910": {
            "name": "Michael Chen",
            "tier": "Diamond",
            "adt": 2850.00,
            "total_theo_ytd": 342000.00,
            "comp_balance": 4200.00,
            "last_visit": "2025-01-28",
            "visit_count_ytd": 3,
            "preferences": {
                "game": "Baccarat",
                "drink": "Macallan 18 neat",
                "room": "Palazzo Suite",
                "dining": "SW Steakhouse",
            },
            "host_assigned": "Jennifer Martinez",
            "status": "active",
        },
        "PLY-119204": {
            "name": "Sarah Williams",
            "tier": "Platinum",
            "adt": 950.00,
            "total_theo_ytd": 87400.00,
            "comp_balance": 1150.00,
            "last_visit": "2025-02-01",
            "visit_count_ytd": 8,
            "preferences": {
                "game": "Blackjack",
                "drink": "Cabernet Sauvignon",
                "room": "Deluxe King",
                "dining": "Nobu",
            },
            "host_assigned": None,
            "status": "active",
        },
        "PLY-330087": {
            "name": "Robert Taylor",
            "tier": "Gold",
            "adt": 420.00,
            "total_theo_ytd": 28560.00,
            "comp_balance": 380.00,
            "last_visit": "2024-12-15",
            "visit_count_ytd": 0,
            "preferences": {
                "game": "Slots - Dragon Link",
                "drink": "Bud Light",
                "room": "Standard King",
                "dining": "Buffet",
            },
            "host_assigned": None,
            "status": "active",
        },
    }

    profile = profiles.get(player_id)
    if profile is None:
        return {
            "player_id": player_id,
            "error": "Player not found in tracking system.",
            "suggestion": "Verify the player ID or search by name.",
        }

    return {"player_id": player_id, **profile}


# ---------------------------------------------------------------------------
# Comp Calculations
# ---------------------------------------------------------------------------


@tool
def calculate_comp(player_id: str, comp_type: str) -> dict:
    """Calculate an eligible comp for a player based on theoretical win.

    Uses the casino's reinvestment matrix to determine what comp value a
    player qualifies for. The reinvestment percentage varies by tier and
    comp type. Common comp types: room, dining, show, freeplay, travel.

    Args:
        player_id: The player tracking number.
        comp_type: Type of comp to calculate. One of:
            "room", "dining", "show", "freeplay", "travel".

    Returns:
        A dict containing:
            - player_id: Echo of the input.
            - comp_type: The requested comp type.
            - theoretical_win: The player's theoretical win used for calculation.
            - reinvestment_pct: The reinvestment percentage applied.
            - comp_value: The calculated comp dollar amount.
            - eligible: Whether the player qualifies.
            - notes: Any relevant notes (tier-based caps, blackout dates, etc.).
    """
    # Reinvestment matrix: tier -> comp_type -> reinvestment percentage
    reinvestment_matrix: dict[str, dict[str, float]] = {
        "Diamond": {
            "room": 0.30,
            "dining": 0.25,
            "show": 0.20,
            "freeplay": 0.15,
            "travel": 0.10,
        },
        "Platinum": {
            "room": 0.22,
            "dining": 0.18,
            "show": 0.15,
            "freeplay": 0.12,
            "travel": 0.07,
        },
        "Gold": {
            "room": 0.15,
            "dining": 0.12,
            "show": 0.10,
            "freeplay": 0.08,
            "travel": 0.00,
        },
        "Silver": {
            "room": 0.10,
            "dining": 0.08,
            "show": 0.06,
            "freeplay": 0.05,
            "travel": 0.00,
        },
    }

    # Stub: look up player tier and theo from check_player_status data
    tier_lookup = {
        "PLY-482910": ("Diamond", 2850.00),
        "PLY-119204": ("Platinum", 950.00),
        "PLY-330087": ("Gold", 420.00),
    }

    player_info = tier_lookup.get(player_id)
    if player_info is None:
        return {
            "player_id": player_id,
            "comp_type": comp_type,
            "eligible": False,
            "error": "Player not found. Cannot calculate comp.",
        }

    tier, adt = player_info
    matrix = reinvestment_matrix.get(tier, reinvestment_matrix["Silver"])
    pct = matrix.get(comp_type, 0.0)

    if pct == 0.0:
        return {
            "player_id": player_id,
            "comp_type": comp_type,
            "theoretical_win": adt,
            "reinvestment_pct": 0.0,
            "comp_value": 0.0,
            "eligible": False,
            "notes": f"{tier} tier players are not eligible for {comp_type} comps.",
        }

    comp_value = round(adt * pct, 2)
    notes = []
    if comp_type == "room" and tier == "Diamond":
        notes.append("Diamond: eligible for suite upgrade at host discretion.")
    if comp_type == "travel" and tier in ("Gold", "Silver"):
        notes.append("Travel comps require Platinum tier or above.")

    return {
        "player_id": player_id,
        "comp_type": comp_type,
        "tier": tier,
        "theoretical_win": adt,
        "reinvestment_pct": pct,
        "comp_value": comp_value,
        "eligible": True,
        "notes": "; ".join(notes) if notes else "Standard comp calculation.",
    }


# ---------------------------------------------------------------------------
# Reservations
# ---------------------------------------------------------------------------


@tool
def make_reservation(
    player_id: str, venue: str, date: str, party_size: int
) -> dict:
    """Book a restaurant, show, or hotel room for a player.

    Checks availability and creates a reservation in the property management
    system. For VIP players, preferred seating and room preferences are
    automatically applied from their profile.

    Args:
        player_id: The player tracking number.
        venue: Name of the venue or room type. Examples:
            "SW Steakhouse", "Nobu", "O by Cirque du Soleil",
            "Palazzo Suite", "Deluxe King".
        date: Reservation date in ISO format (YYYY-MM-DD).
        party_size: Number of guests (including the player).

    Returns:
        A dict containing:
            - confirmation_number: Unique reservation ID.
            - player_id: Echo of the input.
            - venue: The booked venue.
            - date: The reservation date.
            - party_size: Number of guests.
            - time: Assigned time slot (for dining/shows).
            - special_notes: VIP preferences applied.
            - status: "confirmed", "waitlisted", or "unavailable".
    """
    confirmation = f"RES-{random.randint(100000, 999999)}"

    # Simulate availability check
    try:
        res_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return {
            "player_id": player_id,
            "venue": venue,
            "error": f"Invalid date format: {date}. Use YYYY-MM-DD.",
        }

    today = datetime.now(tz=timezone.utc).date()
    if res_date < today:
        return {
            "player_id": player_id,
            "venue": venue,
            "error": "Cannot book reservations in the past.",
        }

    # Simulate occasional unavailability
    if party_size > 8:
        return {
            "confirmation_number": None,
            "player_id": player_id,
            "venue": venue,
            "date": date,
            "party_size": party_size,
            "status": "waitlisted",
            "special_notes": "Large party — pending manager approval.",
        }

    venue_times = {
        "SW Steakhouse": "7:30 PM",
        "Nobu": "8:00 PM",
        "O by Cirque du Soleil": "9:30 PM",
        "Buffet": "12:00 PM",
    }
    time_slot = venue_times.get(venue, "6:00 PM")

    special_notes = []
    if player_id == "PLY-482910":
        special_notes.append("Diamond VIP: preferred booth, Macallan 18 at table.")
    elif player_id == "PLY-119204":
        special_notes.append("Platinum: preferred seating assigned.")

    return {
        "confirmation_number": confirmation,
        "player_id": player_id,
        "venue": venue,
        "date": date,
        "party_size": party_size,
        "time": time_slot,
        "special_notes": "; ".join(special_notes) if special_notes else "Standard reservation.",
        "status": "confirmed",
    }


# ---------------------------------------------------------------------------
# Messaging
# ---------------------------------------------------------------------------


@tool
def send_message(player_id: str, channel: str, message: str) -> dict:
    """Send a personalized message to a player via SMS or email.

    Messages are logged in the CRM and comply with opt-in/opt-out
    regulations. The casino host persona and branding are applied
    automatically based on the channel.

    Args:
        player_id: The player tracking number.
        channel: Delivery channel. One of: "sms", "email".
        message: The message body. Should be personalized and professional.

    Returns:
        A dict containing:
            - message_id: Unique message tracking ID.
            - player_id: Echo of the input.
            - channel: The delivery channel used.
            - status: "sent", "queued", "opted_out", or "failed".
            - timestamp: ISO timestamp of when the message was dispatched.
            - notes: Compliance notes if applicable.
    """
    if channel not in ("sms", "email"):
        return {
            "player_id": player_id,
            "channel": channel,
            "status": "failed",
            "error": f"Unsupported channel: {channel}. Use 'sms' or 'email'.",
        }

    # Simulate opt-out check
    opted_out_players = {"PLY-999001", "PLY-999002"}
    if player_id in opted_out_players:
        return {
            "message_id": None,
            "player_id": player_id,
            "channel": channel,
            "status": "opted_out",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "notes": "Player has opted out of marketing communications.",
        }

    message_id = f"MSG-{random.randint(100000, 999999)}"
    return {
        "message_id": message_id,
        "player_id": player_id,
        "channel": channel,
        "status": "sent",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "notes": "Message dispatched successfully.",
    }


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------


@tool
def check_compliance(player_id: str, action: str) -> dict:
    """Check regulatory compliance before taking an action for a player.

    Validates the proposed action against self-exclusion lists, responsible
    gaming flags, cooling-off periods, and state-specific regulations. This
    tool MUST be called before comps, reservations, or messages for flagged
    players.

    Args:
        player_id: The player tracking number.
        action: The proposed action. One of:
            "comp", "reservation", "message", "freeplay", "marker".

    Returns:
        A dict containing:
            - player_id: Echo of the input.
            - action: The proposed action.
            - compliant: Whether the action is allowed (True/False).
            - flags: List of compliance flags, if any.
            - required_actions: Steps the host must take (if non-compliant).
            - regulation_refs: Relevant regulation references.
    """
    # Simulate self-exclusion and responsible gaming database
    self_excluded = {"PLY-550001", "PLY-550002"}
    responsible_gaming_flags = {
        "PLY-330087": ["excessive_loss_velocity", "session_duration_alert"],
    }

    flags: list[str] = []
    required_actions: list[str] = []
    regulation_refs: list[str] = []

    if player_id in self_excluded:
        flags.append("SELF_EXCLUSION_ACTIVE")
        required_actions.append("BLOCK all promotional contact and comps.")
        required_actions.append("Do NOT acknowledge player's gambling history.")
        regulation_refs.append("NRS 463.368 (Nevada Self-Exclusion Program)")
        return {
            "player_id": player_id,
            "action": action,
            "compliant": False,
            "flags": flags,
            "required_actions": required_actions,
            "regulation_refs": regulation_refs,
        }

    rg_flags = responsible_gaming_flags.get(player_id, [])
    if rg_flags:
        flags.extend(rg_flags)
        if action in ("comp", "freeplay", "marker"):
            required_actions.append(
                "Responsible gaming flags detected. Escalate to compliance officer "
                "before issuing comp, freeplay, or marker."
            )
            regulation_refs.append("AGA Responsible Gaming Code of Conduct, Section 4")
            return {
                "player_id": player_id,
                "action": action,
                "compliant": False,
                "flags": flags,
                "required_actions": required_actions,
                "regulation_refs": regulation_refs,
            }
        # Messages and reservations allowed but flagged
        flags.append("PROCEED_WITH_CAUTION")
        required_actions.append("Note responsible gaming flags in CRM entry.")

    return {
        "player_id": player_id,
        "action": action,
        "compliant": True,
        "flags": flags if flags else ["CLEAR"],
        "required_actions": required_actions if required_actions else ["None"],
        "regulation_refs": regulation_refs if regulation_refs else ["N/A"],
    }


@tool
def lookup_regulations(state_code: str, topic: str) -> str:
    """Retrieve relevant gaming regulation information for a specific state and topic.

    Searches the regulatory knowledge base for applicable rules, limits,
    and requirements. Used by the compliance checker node and by the agent
    when a host asks about regulatory constraints.

    Args:
        state_code: Two-letter US state code (e.g., "NV", "NJ", "PA").
        topic: Regulatory topic to look up. Examples:
            "self_exclusion", "comp_limits", "responsible_gaming",
            "marker_regulations", "advertising_rules".

    Returns:
        A string containing the relevant regulatory summary. Returns a
        "no data" message if the state/topic combination is not in the
        knowledge base.
    """
    regulations = {
        ("NV", "self_exclusion"): (
            "Nevada Self-Exclusion Program (NRS 463.368): Players may voluntarily "
            "exclude themselves for 1 year, 5 years, or lifetime. Casinos must remove "
            "self-excluded persons from marketing lists within 30 days. Violations: "
            "up to $100,000 fine per incident. Host must not acknowledge gambling "
            "history or offer inducements."
        ),
        ("NV", "comp_limits"): (
            "Nevada does not impose statutory caps on complimentary goods and services. "
            "However, NGC Regulation 6A requires accurate recordkeeping of all comps "
            "exceeding $10,000 aggregate per player per year. Comp value must be "
            "reasonable relative to player's theoretical win (industry standard: "
            "15-40% reinvestment depending on tier)."
        ),
        ("NV", "responsible_gaming"): (
            "Nevada Responsible Gaming Regulations: Casinos must provide responsible "
            "gaming information at all entry points. Staff training required annually. "
            "Problem gambling hotline (1-800-522-4700) must be posted. Cashiers must "
            "offer self-exclusion information upon request. Hosts should be trained "
            "to recognize signs of problem gambling."
        ),
        ("NJ", "self_exclusion"): (
            "New Jersey Self-Exclusion Program (NJAC 13:69D-1.6): Players may self-"
            "exclude for 1 year, 5 years, or lifetime. Casinos must forfeit winnings "
            "of self-excluded players to the Casino Revenue Fund. NJ is stricter than "
            "NV: casinos must make reasonable efforts to prevent self-excluded persons "
            "from gambling, not just from receiving marketing."
        ),
        ("PA", "comp_limits"): (
            "Pennsylvania Gaming Control Board: Comps must be reported for any player "
            "receiving $5,000+ in aggregate complimentaries per year. Casinos must "
            "maintain detailed comp logs including theoretical win justification."
        ),
    }

    result = regulations.get((state_code.upper(), topic.lower()))
    if result:
        return result

    return (
        f"No regulatory data found for state={state_code}, topic={topic}. "
        f"Available states: NV, NJ, PA. Available topics: self_exclusion, "
        f"comp_limits, responsible_gaming, marker_regulations, advertising_rules."
    )


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


@tool
def escalate_to_human(player_id: str, reason: str, context: dict) -> dict:
    """Route a conversation or action to a human casino host.

    Creates an escalation ticket in the host management system. Used when
    the AI agent encounters situations requiring human judgment: high-value
    comp approvals, VIP complaints, complex regulatory questions, or when
    explicitly requested by the caller.

    Args:
        player_id: The player tracking number.
        reason: Brief description of why escalation is needed. Examples:
            "High-value comp approval ($5,000+)",
            "Player complaint about service",
            "Complex regulatory question".
        context: Dict with relevant conversation context to hand off:
            - summary: Brief conversation summary.
            - comp_details: Comp calculation if applicable.
            - compliance_flags: Any flags raised.

    Returns:
        A dict containing:
            - ticket_id: Escalation ticket number.
            - player_id: Echo of the input.
            - assigned_host: Human host the case is routed to.
            - priority: "urgent", "high", "normal".
            - estimated_response: Expected response time.
            - status: "routed" or "queued".
    """
    # Priority based on reason keywords
    priority = "normal"
    estimated_response = "2 hours"
    if any(kw in reason.lower() for kw in ("complaint", "upset", "angry", "vip")):
        priority = "urgent"
        estimated_response = "15 minutes"
    elif any(kw in reason.lower() for kw in ("high-value", "regulatory", "marker")):
        priority = "high"
        estimated_response = "30 minutes"

    # Route to assigned host if known, otherwise to duty host
    host_assignments = {
        "PLY-482910": "Jennifer Martinez",
        "PLY-119204": "David Kim",
    }
    assigned_host = host_assignments.get(player_id, "Duty Host (on-shift)")

    ticket_id = f"ESC-{random.randint(10000, 99999)}"

    return {
        "ticket_id": ticket_id,
        "player_id": player_id,
        "assigned_host": assigned_host,
        "reason": reason,
        "priority": priority,
        "estimated_response": estimated_response,
        "context_summary": context.get("summary", "No summary provided."),
        "status": "routed",
    }


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

# Import the RAG search tool so it's available to the agent.
# In the Docker container, both langgraph_agent/ and rag/ are top-level
# packages under /app. For local dev with symlinks, ensure boilerplate/
# is on PYTHONPATH or use `pip install -e .`.
try:
    from rag.retriever import search_knowledge_base
except ImportError:
    # Fallback: try relative import if running as a sub-package
    from ..rag.retriever import search_knowledge_base  # type: ignore[no-redef]

ALL_TOOLS: list = [
    check_player_status,
    calculate_comp,
    make_reservation,
    send_message,
    check_compliance,
    lookup_regulations,
    escalate_to_human,
    search_knowledge_base,
]
"""Complete list of tools available to the Casino Host agent.

Includes all casino-domain tools plus the RAG knowledge base search tool
for answering questions about casino operations, regulations, and player
development best practices.
"""
