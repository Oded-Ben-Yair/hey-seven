"""TCPA compliance module for SMS communications.

Handles mandatory STOP/HELP/START keyword processing, quiet-hours enforcement
per guest timezone, US area-code-to-timezone mapping, consent level checking,
and a tamper-evident SHA-256 consent hash chain.

All keyword handling is deterministic -- no LLM calls. This module executes
BEFORE the agent graph to ensure regulatory compliance at zero cost and
zero latency.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mandatory SMS keywords (CTIA / 10DLC requirement)
# English + Spanish per design spec section 3.4
# ---------------------------------------------------------------------------

STOP_KEYWORDS: frozenset[str] = frozenset(
    {
        "stop",
        "stopall",
        "unsubscribe",
        "cancel",
        "end",
        "quit",
        # Spanish
        "parar",
        "detener",
        "cancelar",
    }
)

HELP_KEYWORDS: frozenset[str] = frozenset(
    {
        "help",
        "info",
        # Spanish
        "ayuda",
        "informacion",
    }
)

START_KEYWORDS: frozenset[str] = frozenset(
    {
        "start",
        "subscribe",
        # Spanish
        "iniciar",
        "comenzar",
    }
)

# ---------------------------------------------------------------------------
# Area-code to IANA timezone mapping (major US area codes)
# ---------------------------------------------------------------------------

_AREA_CODE_TZ: dict[str, str] = {
    # Eastern
    "201": "America/New_York",
    "202": "America/New_York",
    "203": "America/New_York",
    "212": "America/New_York",
    "215": "America/New_York",
    "216": "America/New_York",
    "267": "America/New_York",
    "301": "America/New_York",
    "302": "America/New_York",
    "305": "America/New_York",
    "315": "America/New_York",
    "347": "America/New_York",
    "401": "America/New_York",
    "407": "America/New_York",
    "410": "America/New_York",
    "412": "America/New_York",
    "443": "America/New_York",
    "516": "America/New_York",
    "518": "America/New_York",
    "561": "America/New_York",
    "570": "America/New_York",
    "585": "America/New_York",
    "609": "America/New_York",
    "610": "America/New_York",
    "617": "America/New_York",
    "631": "America/New_York",
    "646": "America/New_York",
    "678": "America/New_York",
    "704": "America/New_York",
    "718": "America/New_York",
    "732": "America/New_York",
    "757": "America/New_York",
    "770": "America/New_York",
    "786": "America/New_York",
    "803": "America/New_York",
    "804": "America/New_York",
    "813": "America/New_York",
    "845": "America/New_York",
    "856": "America/New_York",
    "860": "America/New_York",
    "862": "America/New_York",
    "904": "America/New_York",
    "908": "America/New_York",
    "910": "America/New_York",
    "914": "America/New_York",
    "917": "America/New_York",
    "919": "America/New_York",
    "929": "America/New_York",
    "941": "America/New_York",
    "954": "America/New_York",
    "973": "America/New_York",
    # Central
    "210": "America/Chicago",
    "214": "America/Chicago",
    "217": "America/Chicago",
    "224": "America/Chicago",
    "225": "America/Chicago",
    "251": "America/Chicago",
    "254": "America/Chicago",
    "256": "America/Chicago",
    "262": "America/Chicago",
    "281": "America/Chicago",
    "312": "America/Chicago",
    "314": "America/Chicago",
    "316": "America/Chicago",
    "317": "America/Chicago",
    "318": "America/Chicago",
    "319": "America/Chicago",
    "320": "America/Chicago",
    "331": "America/Chicago",
    "334": "America/Chicago",
    "337": "America/Chicago",
    "361": "America/Chicago",
    "402": "America/Chicago",
    "405": "America/Chicago",
    "409": "America/Chicago",
    "414": "America/Chicago",
    "417": "America/Chicago",
    "469": "America/Chicago",
    "501": "America/Chicago",
    "502": "America/Chicago",
    "504": "America/Chicago",
    "507": "America/Chicago",
    "512": "America/Chicago",
    "515": "America/Chicago",
    "563": "America/Chicago",
    "573": "America/Chicago",
    "601": "America/Chicago",
    "608": "America/Chicago",
    "612": "America/Chicago",
    "614": "America/Chicago",
    "615": "America/Chicago",
    "618": "America/Chicago",
    "630": "America/Chicago",
    "636": "America/Chicago",
    "651": "America/Chicago",
    "662": "America/Chicago",
    "682": "America/Chicago",
    "708": "America/Chicago",
    "713": "America/Chicago",
    "715": "America/Chicago",
    "731": "America/Chicago",
    "763": "America/Chicago",
    "769": "America/Chicago",
    "773": "America/Chicago",
    "779": "America/Chicago",
    "806": "America/Chicago",
    "812": "America/Chicago",
    "815": "America/Chicago",
    "816": "America/Chicago",
    "817": "America/Chicago",
    "830": "America/Chicago",
    "832": "America/Chicago",
    "847": "America/Chicago",
    "850": "America/Chicago",
    "870": "America/Chicago",
    "901": "America/Chicago",
    "903": "America/Chicago",
    "913": "America/Chicago",
    "918": "America/Chicago",
    "920": "America/Chicago",
    "936": "America/Chicago",
    "940": "America/Chicago",
    "952": "America/Chicago",
    "956": "America/Chicago",
    "972": "America/Chicago",
    "979": "America/Chicago",
    # Mountain
    "303": "America/Denver",
    "307": "America/Denver",
    "385": "America/Denver",
    "406": "America/Denver",
    "480": "America/Denver",
    "505": "America/Denver",
    "520": "America/Denver",
    "575": "America/Denver",
    "602": "America/Denver",
    "623": "America/Denver",
    "719": "America/Denver",
    "720": "America/Denver",
    "801": "America/Denver",
    "928": "America/Denver",
    # Pacific
    "206": "America/Los_Angeles",
    "208": "America/Los_Angeles",
    "209": "America/Los_Angeles",
    "213": "America/Los_Angeles",
    "253": "America/Los_Angeles",
    "310": "America/Los_Angeles",
    "323": "America/Los_Angeles",
    "360": "America/Los_Angeles",
    "408": "America/Los_Angeles",
    "415": "America/Los_Angeles",
    "424": "America/Los_Angeles",
    "425": "America/Los_Angeles",
    "442": "America/Los_Angeles",
    "503": "America/Los_Angeles",
    "509": "America/Los_Angeles",
    "510": "America/Los_Angeles",
    "530": "America/Los_Angeles",
    "541": "America/Los_Angeles",
    "559": "America/Los_Angeles",
    "562": "America/Los_Angeles",
    "619": "America/Los_Angeles",
    "626": "America/Los_Angeles",
    "650": "America/Los_Angeles",
    "657": "America/Los_Angeles",
    "661": "America/Los_Angeles",
    "669": "America/Los_Angeles",
    "702": "America/Los_Angeles",
    "707": "America/Los_Angeles",
    "714": "America/Los_Angeles",
    "725": "America/Los_Angeles",
    "747": "America/Los_Angeles",
    "760": "America/Los_Angeles",
    "775": "America/Los_Angeles",
    "805": "America/Los_Angeles",
    "818": "America/Los_Angeles",
    "831": "America/Los_Angeles",
    "858": "America/Los_Angeles",
    "909": "America/Los_Angeles",
    "916": "America/Los_Angeles",
    "925": "America/Los_Angeles",
    "949": "America/Los_Angeles",
    "951": "America/Los_Angeles",
    "971": "America/Los_Angeles",
    # Hawaii / Alaska
    "808": "Pacific/Honolulu",
    "907": "America/Anchorage",
}

DEFAULT_TIMEZONE = "America/New_York"


# ---------------------------------------------------------------------------
# Keyword handling
# ---------------------------------------------------------------------------


async def handle_mandatory_keywords(message: str, phone: str) -> str | None:
    """Process mandatory SMS keywords (STOP/HELP/START).

    This function MUST be called BEFORE any LLM invocation. It returns a
    canned response string for recognized keywords or ``None`` if the
    message is not a keyword and should continue to the agent graph.

    Args:
        message: Raw inbound SMS text.
        phone: Sender phone number (E.164).

    Returns:
        Response text for the keyword, or ``None`` to continue processing.
    """
    normalized = message.strip().lower()

    if normalized in STOP_KEYWORDS:
        logger.info("STOP keyword from %s -- opting out", phone[-4:])
        return "You have been unsubscribed. Reply START to resubscribe."

    if normalized in HELP_KEYWORDS:
        logger.info("HELP keyword from %s", phone[-4:])
        return (
            "Reply with your question or text STOP to opt out. "
            "For support call your casino host directly. "
            "Msg & data rates may apply."
        )

    if normalized in START_KEYWORDS:
        logger.info("START keyword from %s -- opting in", phone[-4:])
        return "You have been resubscribed. Reply STOP to unsubscribe."

    return None


# ---------------------------------------------------------------------------
# Quiet hours
# ---------------------------------------------------------------------------


def is_quiet_hours(
    tz_name: str,
    *,
    quiet_start: int = 21,
    quiet_end: int = 8,
    now: datetime | None = None,
) -> bool:
    """Check whether the current time falls within quiet hours.

    Quiet hours default to 9 PM - 8 AM in the specified timezone, per
    TCPA / FCC guidance and state laws (e.g. Florida 8am-8pm).

    Args:
        tz_name: IANA timezone string (e.g. ``America/New_York``).
        quiet_start: Hour (0-23) when quiet period begins. Default 21 (9 PM).
        quiet_end: Hour (0-23) when quiet period ends. Default 8 (8 AM).
        now: Override current time for testing.

    Returns:
        ``True`` if outbound messages should be blocked.
    """
    try:
        tz = ZoneInfo(tz_name)
    except (KeyError, ValueError):
        logger.warning("Unknown timezone '%s', defaulting to %s", tz_name, DEFAULT_TIMEZONE)
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    local_now = (now or datetime.now(timezone.utc)).astimezone(tz)
    hour = local_now.hour

    if quiet_start > quiet_end:
        # Wraps midnight: e.g. 21:00 - 08:00
        return hour >= quiet_start or hour < quiet_end
    # Same-day range (unusual but handle it)
    return quiet_start <= hour < quiet_end


def get_timezone_from_area_code(phone: str) -> str:
    """Map a US phone number's area code to an IANA timezone.

    Args:
        phone: Phone number, optionally prefixed with ``+1``.

    Returns:
        IANA timezone string. Defaults to ``America/New_York`` for
        unrecognized area codes.
    """
    digits = "".join(c for c in phone if c.isdigit())
    # Strip country code prefix
    if digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) >= 3:
        area_code = digits[:3]
        return _AREA_CODE_TZ.get(area_code, DEFAULT_TIMEZONE)
    return DEFAULT_TIMEZONE


# ---------------------------------------------------------------------------
# Consent hash chain (tamper-evident TCPA consent audit log)
# ---------------------------------------------------------------------------


class ConsentHashChain:
    """SHA-256 hash chain for tamper-evident TCPA consent events.

    Each event's hash incorporates the previous event's hash, making any
    retroactive modification detectable. This provides a legally defensible
    audit trail for consent opt-in / opt-out events.

    Usage::

        chain = ConsentHashChain()
        h1 = chain.add_event("opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form")
        h2 = chain.add_event("opt_out", "+12125551234", "2026-03-16T10:00:00Z", "STOP keyword")
        assert chain.verify_chain()
    """

    def __init__(self) -> None:
        self._events: list[dict[str, str]] = []
        self._hashes: list[str] = []

    @property
    def events(self) -> list[dict[str, str]]:
        """Return a copy of the event list."""
        return list(self._events)

    @property
    def hashes(self) -> list[str]:
        """Return a copy of the hash list."""
        return list(self._hashes)

    def add_event(
        self,
        event_type: str,
        phone: str,
        timestamp: str,
        evidence: str,
    ) -> str:
        """Append a consent event and return its SHA-256 hash.

        Args:
            event_type: One of ``opt_in``, ``opt_out``, ``scope_change``.
            phone: Guest phone number (E.164).
            timestamp: ISO-8601 timestamp of the event.
            evidence: Description of how consent was obtained/revoked.

        Returns:
            Hex-encoded SHA-256 hash of the event.
        """
        previous_hash = self._hashes[-1] if self._hashes else "0" * 64

        payload = f"{previous_hash}|{event_type}|{phone}|{timestamp}|{evidence}"
        event_hash = hashlib.sha256(payload.encode()).hexdigest()

        self._events.append(
            {
                "event_type": event_type,
                "phone": phone,
                "timestamp": timestamp,
                "evidence": evidence,
                "previous_hash": previous_hash,
                "hash": event_hash,
            }
        )
        self._hashes.append(event_hash)
        logger.debug("Consent event %s for %s -> %s", event_type, phone[-4:], event_hash[:12])
        return event_hash

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire hash chain.

        Returns:
            ``True`` if every event's hash matches a fresh recomputation
            from its inputs plus the preceding hash. ``False`` if any
            event has been tampered with.
        """
        if not self._events:
            return True

        for i, event in enumerate(self._events):
            previous_hash = self._hashes[i - 1] if i > 0 else "0" * 64
            payload = (
                f"{previous_hash}|{event['event_type']}|{event['phone']}"
                f"|{event['timestamp']}|{event['evidence']}"
            )
            expected = hashlib.sha256(payload.encode()).hexdigest()
            if expected != event["hash"]:
                logger.warning(
                    "Chain integrity failure at index %d: expected %s, got %s",
                    i,
                    expected[:12],
                    event["hash"][:12],
                )
                return False
        return True


# ---------------------------------------------------------------------------
# Consent level checking
# ---------------------------------------------------------------------------

# TCPA consent hierarchy (ascending strictness)
_CONSENT_HIERARCHY: list[str] = [
    "none",
    "prior_express_consent",
    "prior_express_written_consent",
]

# Minimum consent required per message type
_MESSAGE_TYPE_CONSENT: dict[str, str] = {
    "transactional": "none",
    "informational": "prior_express_consent",
    "marketing": "prior_express_written_consent",
}


def check_consent(profile: dict[str, Any], message_type: str) -> bool:
    """Check whether a guest profile has sufficient consent for a message type.

    Args:
        profile: Guest profile dict. Expected to have a ``consent`` sub-dict
            with fields ``sms_opt_in`` (bool), ``sms_opt_in_method`` (str),
            and optionally ``sms_opt_in_timestamp`` (str).
        message_type: One of ``transactional``, ``informational``, ``marketing``.

    Returns:
        ``True`` if the guest has sufficient consent to receive this message type.
    """
    required = _MESSAGE_TYPE_CONSENT.get(message_type, "prior_express_written_consent")

    # Transactional always allowed (response to guest inquiry)
    if required == "none":
        return True

    consent = profile.get("consent", {})

    # Opted out = no messages at all
    if consent.get("sms_opt_in") is False:
        return False

    # Determine the guest's current consent tier
    if not consent.get("sms_opt_in"):
        guest_consent = "none"
    elif consent.get("sms_opt_in_method") in ("web_form", "paper_form"):
        # Written consent methods satisfy the highest tier
        guest_consent = "prior_express_written_consent"
    elif consent.get("sms_opt_in_method") == "text_keyword":
        # Keyword opt-in satisfies express consent but not written
        guest_consent = "prior_express_consent"
    else:
        # Unknown method -- treat as basic consent
        guest_consent = "prior_express_consent"

    required_idx = _CONSENT_HIERARCHY.index(required)
    guest_idx = _CONSENT_HIERARCHY.index(guest_consent)

    return guest_idx >= required_idx
