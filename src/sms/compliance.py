"""TCPA compliance module for SMS communications.

Handles mandatory STOP/HELP/START keyword processing, quiet-hours enforcement
per guest timezone, US area-code-to-timezone mapping, consent level checking,
and a tamper-evident HMAC-SHA256 consent hash chain.

All keyword handling is deterministic -- no LLM calls. This module executes
BEFORE the agent graph to ensure regulatory compliance at zero cost and
zero latency.

Production deployments should pass ``settings.CONSENT_HMAC_SECRET`` to
``ConsentHashChain(hmac_secret=...)`` to authenticate the hash chain
(from Key Vault / environment variable).
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
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

# Area-code to IANA timezone mapping (major US area codes).
#
# IMPORTANT: Mobile Number Portability (MNP) means area codes are NOT
# a reliable indicator of current location or timezone. A guest may have
# ported their number from another state. This mapping provides a
# best-effort default; production systems should:
# 1. Prefer guest-profile timezone when available (from CRM/PMS)
# 2. Fall back to area code mapping only when profile timezone is unknown
# 3. Default to property timezone (America/New_York for Mohegan Sun)
#    when area code is unrecognized

_AREA_CODE_TZ: dict[str, str] = {
    # -----------------------------------------------------------------------
    # Covers 280+ of ~335 active US area codes (NANPA, as of Feb 2026).
    # Missing codes default to America/New_York (property timezone).
    #
    # Note: Mobile Number Portability (MNP) means area codes are NOT a
    # reliable indicator of current physical location or timezone. A guest
    # may have ported their number from another state. For production use,
    # consider libphonenumber or a guest timezone preference override from
    # CRM/PMS profile data.
    # -----------------------------------------------------------------------
    #
    # Eastern Time (America/New_York)
    "201": "America/New_York",   # NJ
    "202": "America/New_York",   # DC
    "203": "America/New_York",   # CT
    "207": "America/New_York",   # ME
    "212": "America/New_York",   # NY (Manhattan)
    "215": "America/New_York",   # PA (Philadelphia)
    "216": "America/New_York",   # OH (Cleveland)
    "220": "America/New_York",   # OH overlay
    "223": "America/New_York",   # PA overlay
    "229": "America/New_York",   # GA (Albany)
    "234": "America/New_York",   # OH overlay
    "239": "America/New_York",   # FL (Fort Myers)
    "240": "America/New_York",   # MD overlay
    "248": "America/New_York",   # MI (Troy)
    "267": "America/New_York",   # PA overlay
    "272": "America/New_York",   # PA (Scranton overlay)
    "276": "America/New_York",   # VA
    "301": "America/New_York",   # MD
    "302": "America/New_York",   # DE
    "304": "America/New_York",   # WV
    "305": "America/New_York",   # FL (Miami)
    "315": "America/New_York",   # NY (Syracuse)
    "321": "America/New_York",   # FL overlay
    "326": "America/New_York",   # OH overlay
    "330": "America/New_York",   # OH (Akron)
    "332": "America/New_York",   # NY (NYC overlay)
    "336": "America/New_York",   # NC
    "339": "America/New_York",   # MA overlay
    "340": "America/New_York",   # US Virgin Islands
    "347": "America/New_York",   # NY overlay
    "351": "America/New_York",   # MA overlay
    "352": "America/New_York",   # FL (Gainesville)
    "380": "America/New_York",   # OH overlay
    "386": "America/New_York",   # FL (Daytona)
    "401": "America/New_York",   # RI
    "404": "America/New_York",   # GA (Atlanta)
    "407": "America/New_York",   # FL (Orlando)
    "410": "America/New_York",   # MD (Baltimore)
    "412": "America/New_York",   # PA (Pittsburgh)
    "413": "America/New_York",   # MA (Springfield)
    "434": "America/New_York",   # VA
    "440": "America/New_York",   # OH
    "443": "America/New_York",   # MD overlay
    "445": "America/New_York",   # PA overlay
    "448": "America/New_York",   # PA overlay
    "470": "America/New_York",   # GA overlay
    "475": "America/New_York",   # CT overlay
    "478": "America/New_York",   # GA (Macon)
    "484": "America/New_York",   # PA overlay
    "508": "America/New_York",   # MA (Worcester)
    "513": "America/New_York",   # OH (Cincinnati)
    "516": "America/New_York",   # NY (Nassau)
    "517": "America/New_York",   # MI (Lansing)
    "518": "America/New_York",   # NY (Albany)
    "540": "America/New_York",   # VA (Roanoke)
    "551": "America/New_York",   # NJ overlay
    "561": "America/New_York",   # FL (West Palm Beach)
    "567": "America/New_York",   # OH overlay
    "570": "America/New_York",   # PA (Scranton)
    "571": "America/New_York",   # VA overlay
    "585": "America/New_York",   # NY (Rochester)
    "586": "America/New_York",   # MI
    "603": "America/New_York",   # NH
    "607": "America/New_York",   # NY (Binghamton)
    "609": "America/New_York",   # NJ (Trenton)
    "610": "America/New_York",   # PA
    "614": "America/New_York",   # OH (Columbus)
    "616": "America/New_York",   # MI (Grand Rapids)
    "617": "America/New_York",   # MA (Boston)
    "631": "America/New_York",   # NY (Suffolk)
    "640": "America/New_York",   # NJ overlay
    "646": "America/New_York",   # NY overlay
    "667": "America/New_York",   # MD overlay
    "678": "America/New_York",   # GA overlay
    "680": "America/New_York",   # NY overlay
    "681": "America/New_York",   # WV overlay
    "689": "America/New_York",   # FL overlay
    "703": "America/New_York",   # VA (Northern)
    "704": "America/New_York",   # NC (Charlotte)
    "706": "America/New_York",   # GA
    "716": "America/New_York",   # NY (Buffalo)
    "717": "America/New_York",   # PA
    "718": "America/New_York",   # NY (outer boroughs)
    "724": "America/New_York",   # PA
    "727": "America/New_York",   # FL (St. Petersburg)
    "732": "America/New_York",   # NJ
    "740": "America/New_York",   # OH
    "743": "America/New_York",   # NC overlay
    "754": "America/New_York",   # FL overlay
    "757": "America/New_York",   # VA (Norfolk)
    "762": "America/New_York",   # GA overlay
    "770": "America/New_York",   # GA (Atlanta suburbs)
    "772": "America/New_York",   # FL
    "774": "America/New_York",   # MA overlay
    "781": "America/New_York",   # MA
    "786": "America/New_York",   # FL (Miami overlay)
    "802": "America/New_York",   # VT
    "803": "America/New_York",   # SC
    "804": "America/New_York",   # VA (Richmond)
    "810": "America/New_York",   # MI (Flint)
    "813": "America/New_York",   # FL (Tampa)
    "828": "America/New_York",   # NC (Asheville)
    "838": "America/New_York",   # NY overlay
    "843": "America/New_York",   # SC (Charleston)
    "845": "America/New_York",   # NY (Hudson Valley)
    "848": "America/New_York",   # NJ overlay
    "854": "America/New_York",   # SC overlay
    "856": "America/New_York",   # NJ
    "857": "America/New_York",   # MA overlay
    "860": "America/New_York",   # CT (Mohegan Sun area)
    "862": "America/New_York",   # NJ overlay
    "863": "America/New_York",   # FL
    "878": "America/New_York",   # PA overlay
    "904": "America/New_York",   # FL (Jacksonville)
    "908": "America/New_York",   # NJ
    "910": "America/New_York",   # NC
    "912": "America/New_York",   # GA (Savannah)
    "914": "America/New_York",   # NY (Westchester)
    "917": "America/New_York",   # NY overlay
    "919": "America/New_York",   # NC (Raleigh)
    "929": "America/New_York",   # NY overlay
    "934": "America/New_York",   # NY overlay
    "937": "America/New_York",   # OH (Dayton)
    "941": "America/New_York",   # FL (Sarasota)
    "943": "America/New_York",   # GA overlay
    "947": "America/New_York",   # MI overlay
    "954": "America/New_York",   # FL (Ft. Lauderdale)
    "959": "America/New_York",   # CT overlay
    "973": "America/New_York",   # NJ
    "978": "America/New_York",   # MA
    "980": "America/New_York",   # NC overlay
    "984": "America/New_York",   # NC overlay
    # Central Time (America/Chicago)
    "205": "America/Chicago",    # AL (Birmingham)
    "210": "America/Chicago",    # TX (San Antonio)
    "214": "America/Chicago",    # TX (Dallas)
    "217": "America/Chicago",    # IL
    "218": "America/Chicago",    # MN
    "224": "America/Chicago",    # IL overlay
    "225": "America/Chicago",    # LA
    "228": "America/Chicago",    # MS
    "251": "America/Chicago",    # AL (Mobile)
    "254": "America/Chicago",    # TX
    "256": "America/Chicago",    # AL (Huntsville)
    "262": "America/Chicago",    # WI
    "270": "America/Chicago",    # KY
    "281": "America/Chicago",    # TX (Houston)
    "312": "America/Chicago",    # IL (Chicago)
    "314": "America/Chicago",    # MO (St. Louis)
    "316": "America/Chicago",    # KS
    "318": "America/Chicago",    # LA
    "319": "America/Chicago",    # IA
    "320": "America/Chicago",    # MN
    "325": "America/Chicago",    # TX
    "331": "America/Chicago",    # IL overlay
    "334": "America/Chicago",    # AL
    "337": "America/Chicago",    # LA
    "346": "America/Chicago",    # TX (Houston overlay)
    "361": "America/Chicago",    # TX
    "364": "America/Chicago",    # KY overlay
    "402": "America/Chicago",    # NE
    "405": "America/Chicago",    # OK
    "409": "America/Chicago",    # TX
    "414": "America/Chicago",    # WI (Milwaukee)
    "417": "America/Chicago",    # MO
    "430": "America/Chicago",    # TX overlay
    "432": "America/Chicago",    # TX
    "469": "America/Chicago",    # TX overlay
    "479": "America/Chicago",    # AR
    "501": "America/Chicago",    # AR
    "502": "America/Chicago",    # KY (Louisville)
    "504": "America/Chicago",    # LA (New Orleans)
    "507": "America/Chicago",    # MN
    "512": "America/Chicago",    # TX (Austin)
    "515": "America/Chicago",    # IA
    "531": "America/Chicago",    # NE overlay
    "539": "America/Chicago",    # OK overlay
    "563": "America/Chicago",    # IA
    "573": "America/Chicago",    # MO
    "580": "America/Chicago",    # OK
    "601": "America/Chicago",    # MS
    "608": "America/Chicago",    # WI
    "612": "America/Chicago",    # MN (Minneapolis)
    "615": "America/Chicago",    # TN (Nashville)
    "618": "America/Chicago",    # IL
    "620": "America/Chicago",    # KS
    "630": "America/Chicago",    # IL
    "636": "America/Chicago",    # MO
    "641": "America/Chicago",    # IA
    "651": "America/Chicago",    # MN
    "656": "America/Chicago",    # TX overlay
    "659": "America/Chicago",    # AL overlay
    "660": "America/Chicago",    # MO
    "662": "America/Chicago",    # MS
    "682": "America/Chicago",    # TX overlay
    "701": "America/Chicago",    # ND
    "708": "America/Chicago",    # IL
    "713": "America/Chicago",    # TX (Houston)
    "715": "America/Chicago",    # WI
    "726": "America/Chicago",    # TX overlay
    "731": "America/Chicago",    # TN
    "737": "America/Chicago",    # TX (Austin overlay)
    "763": "America/Chicago",    # MN overlay
    "769": "America/Chicago",    # MS overlay
    "773": "America/Chicago",    # IL (Chicago)
    "779": "America/Chicago",    # IL overlay
    "785": "America/Chicago",    # KS
    "806": "America/Chicago",    # TX
    "812": "America/Chicago",    # IN (Evansville)
    "815": "America/Chicago",    # IL
    "816": "America/Chicago",    # MO (Kansas City)
    "817": "America/Chicago",    # TX (Fort Worth)
    "830": "America/Chicago",    # TX
    "832": "America/Chicago",    # TX overlay
    "847": "America/Chicago",    # IL
    "850": "America/Chicago",    # FL (Panhandle)
    "870": "America/Chicago",    # AR
    "872": "America/Chicago",    # IL overlay
    "901": "America/Chicago",    # TN (Memphis)
    "903": "America/Chicago",    # TX
    "913": "America/Chicago",    # KS
    "918": "America/Chicago",    # OK (Tulsa)
    "920": "America/Chicago",    # WI
    "936": "America/Chicago",    # TX
    "940": "America/Chicago",    # TX
    "945": "America/Chicago",    # TX overlay
    "952": "America/Chicago",    # MN
    "956": "America/Chicago",    # TX
    "972": "America/Chicago",    # TX
    "979": "America/Chicago",    # TX
    # Indiana (Eastern — America/Indiana/Indianapolis)
    "260": "America/Indiana/Indianapolis",  # IN (Fort Wayne)
    "317": "America/Indiana/Indianapolis",  # IN (Indianapolis)
    "463": "America/Indiana/Indianapolis",  # IN (Indianapolis overlay)
    "574": "America/Indiana/Indianapolis",  # IN (South Bend)
    "765": "America/Indiana/Indianapolis",  # IN (Muncie)
    "930": "America/Indiana/Indianapolis",  # IN overlay
    # Mountain Time (America/Denver)
    "303": "America/Denver",     # CO (Denver)
    "307": "America/Denver",     # WY
    "385": "America/Denver",     # UT overlay
    "406": "America/Denver",     # MT
    "435": "America/Denver",     # UT
    "480": "America/Denver",     # AZ
    "505": "America/Denver",     # NM
    "520": "America/Denver",     # AZ (Tucson)
    "575": "America/Denver",     # NM
    "602": "America/Denver",     # AZ (Phoenix)
    "623": "America/Denver",     # AZ
    "719": "America/Denver",     # CO
    "720": "America/Denver",     # CO overlay
    "801": "America/Denver",     # UT
    "928": "America/Denver",     # AZ
    "970": "America/Denver",     # CO
    "983": "America/Denver",     # CO overlay
    # Pacific Time (America/Los_Angeles)
    "206": "America/Los_Angeles",  # WA (Seattle)
    "208": "America/Los_Angeles",  # ID
    "209": "America/Los_Angeles",  # CA
    "213": "America/Los_Angeles",  # CA (LA)
    "253": "America/Los_Angeles",  # WA (Tacoma)
    "279": "America/Los_Angeles",  # CA (Sacramento overlay)
    "310": "America/Los_Angeles",  # CA
    "323": "America/Los_Angeles",  # CA (LA)
    "341": "America/Los_Angeles",  # CA overlay
    "360": "America/Los_Angeles",  # WA
    "408": "America/Los_Angeles",  # CA (San Jose)
    "415": "America/Los_Angeles",  # CA (SF)
    "424": "America/Los_Angeles",  # CA overlay
    "425": "America/Los_Angeles",  # WA
    "442": "America/Los_Angeles",  # CA overlay
    "458": "America/Los_Angeles",  # OR overlay
    "503": "America/Los_Angeles",  # OR (Portland)
    "509": "America/Los_Angeles",  # WA (Spokane)
    "510": "America/Los_Angeles",  # CA (Oakland)
    "530": "America/Los_Angeles",  # CA
    "541": "America/Los_Angeles",  # OR
    "559": "America/Los_Angeles",  # CA (Fresno)
    "562": "America/Los_Angeles",  # CA (Long Beach)
    "564": "America/Los_Angeles",  # WA overlay
    "619": "America/Los_Angeles",  # CA (San Diego)
    "626": "America/Los_Angeles",  # CA
    "628": "America/Los_Angeles",  # CA (SF overlay)
    "650": "America/Los_Angeles",  # CA
    "657": "America/Los_Angeles",  # CA overlay
    "661": "America/Los_Angeles",  # CA
    "669": "America/Los_Angeles",  # CA overlay
    "702": "America/Los_Angeles",  # NV (Las Vegas)
    "707": "America/Los_Angeles",  # CA
    "714": "America/Los_Angeles",  # CA
    "725": "America/Los_Angeles",  # NV overlay
    "747": "America/Los_Angeles",  # CA overlay
    "760": "America/Los_Angeles",  # CA
    "775": "America/Los_Angeles",  # NV (Reno)
    "805": "America/Los_Angeles",  # CA
    "818": "America/Los_Angeles",  # CA
    "820": "America/Los_Angeles",  # OR overlay
    "831": "America/Los_Angeles",  # CA
    "858": "America/Los_Angeles",  # CA (San Diego)
    "909": "America/Los_Angeles",  # CA
    "916": "America/Los_Angeles",  # CA (Sacramento)
    "925": "America/Los_Angeles",  # CA
    "949": "America/Los_Angeles",  # CA (Irvine)
    "951": "America/Los_Angeles",  # CA
    "971": "America/Los_Angeles",  # OR overlay
    # Hawaii / Alaska
    "808": "Pacific/Honolulu",   # HI
    "907": "America/Anchorage",  # AK
    # US Territories
    "670": "Pacific/Guam",       # CNMI
    "671": "Pacific/Guam",       # Guam
    "684": "Pacific/Pago_Pago",  # American Samoa
    "787": "America/Puerto_Rico",  # PR
    "939": "America/Puerto_Rico",  # PR overlay
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


def _get_timezone_from_area_code(phone: str) -> str:
    """Map a US phone number's area code to an IANA timezone (best-effort).

    This is an internal helper used by ``get_guest_timezone()``.  External
    callers should use ``get_guest_timezone(phone, profile)`` which
    implements the full resolution order: profile timezone > area code > default.

    **MNP caveat**: Mobile Number Portability means area codes may not
    reflect current location. Prefer guest-profile timezone from CRM/PMS
    when available; this mapping is the fallback for unknown profiles.

    Args:
        phone: Phone number, optionally prefixed with ``+1``.

    Returns:
        IANA timezone string. Defaults to ``America/New_York`` (property
        timezone) for unrecognized area codes.
    """
    digits = "".join(c for c in phone if c.isdigit())
    # Strip country code prefix
    if digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) >= 3:
        area_code = digits[:3]
        return _AREA_CODE_TZ.get(area_code, DEFAULT_TIMEZONE)
    return DEFAULT_TIMEZONE


def get_guest_timezone(phone: str, profile: dict[str, Any] | None = None) -> str:
    """Resolve timezone for a guest -- the single public entry point for timezone resolution.

    Callers should use this function instead of accessing area code mapping
    directly.  The raw area code lookup is an internal implementation detail
    (``_get_timezone_from_area_code``).

    Timezone resolution order:
    1. Guest profile ``timezone`` field (most accurate -- from CRM/PMS)
    2. Area code mapping (best-effort fallback -- see MNP caveat)
    3. Property default timezone (America/New_York for Mohegan Sun)

    Args:
        phone: Guest phone number (E.164 format).
        profile: Optional guest profile dict with a ``timezone`` field.

    Returns:
        IANA timezone string.
    """
    # 1. Profile timezone (most reliable)
    if profile and profile.get("timezone"):
        tz_name = profile["timezone"]
        try:
            ZoneInfo(tz_name)  # Validate
            return tz_name
        except (KeyError, ValueError):
            logger.warning(
                "Invalid profile timezone '%s' for %s, falling back to area code",
                tz_name,
                phone[-4:],
            )

    # 2. Area code mapping (best-effort)
    return _get_timezone_from_area_code(phone)


# ---------------------------------------------------------------------------
# Consent hash chain (tamper-evident TCPA consent audit log)
# ---------------------------------------------------------------------------


class ConsentHashChain:
    """SHA-256 hash chain for tamper-evident TCPA consent events.

    Each event's hash incorporates the previous event's hash, making any
    retroactive modification detectable. This provides a legally defensible
    audit trail for consent opt-in / opt-out events.

    When ``hmac_secret`` is provided, events use HMAC-SHA256 instead of
    plain SHA-256, adding authentication to the tamper-evidence. This
    prevents an attacker who knows the hash algorithm from forging a
    valid chain. Production deployments should always provide a secret
    (from Key Vault / environment variable).

    Usage::

        chain = ConsentHashChain()
        h1 = chain.add_event("opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form")
        h2 = chain.add_event("opt_out", "+12125551234", "2026-03-16T10:00:00Z", "STOP keyword")
        assert chain.verify_chain()

        # With HMAC (production):
        chain = ConsentHashChain(hmac_secret="from-key-vault")
        h1 = chain.add_event("opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form")
        assert chain.verify_chain()
    """

    def __init__(self, hmac_secret: str | None = None) -> None:
        self._events: list[dict[str, str]] = []
        self._hashes: list[str] = []
        self._hmac_secret = hmac_secret

    @property
    def events(self) -> list[dict[str, str]]:
        """Return a copy of the event list."""
        return list(self._events)

    @property
    def hashes(self) -> list[str]:
        """Return a copy of the hash list."""
        return list(self._hashes)

    def _compute_hash(self, payload: str) -> str:
        """Compute SHA-256 or HMAC-SHA256 hash of a payload string."""
        if self._hmac_secret:
            return hmac_mod.new(
                self._hmac_secret.encode(), payload.encode(), hashlib.sha256
            ).hexdigest()
        return hashlib.sha256(payload.encode()).hexdigest()

    def add_event(
        self,
        event_type: str,
        phone: str,
        timestamp: str,
        evidence: str,
    ) -> str:
        """Append a consent event and return its SHA-256 (or HMAC-SHA256) hash.

        Args:
            event_type: One of ``opt_in``, ``opt_out``, ``scope_change``.
            phone: Guest phone number (E.164).
            timestamp: ISO-8601 timestamp of the event.
            evidence: Description of how consent was obtained/revoked.

        Returns:
            Hex-encoded hash of the event.
        """
        previous_hash = self._hashes[-1] if self._hashes else "0" * 64

        payload = f"{previous_hash}|{event_type}|{phone}|{timestamp}|{evidence}"
        event_hash = self._compute_hash(payload)

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
            expected = self._compute_hash(payload)
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
