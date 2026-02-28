"""Structured hours parsing and timezone-aware availability checking.

Converts string-based hours (e.g., "11:00 AM - 10:00 PM") into structured
dicts with open/close times and timezone. Provides is_open_now() for
time-aware responses.

Feature flag: hours_parsing_enabled (future -- currently always active).
"""

import logging
import re
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

__all__ = ["parse_hours", "is_open_now", "StructuredHours"]


class StructuredHours:
    """Parsed hours with timezone awareness."""

    def __init__(
        self,
        open_time: time | None,
        close_time: time | None,
        timezone: str = "America/New_York",
        raw: str = "",
        is_24h: bool = False,
        is_closed: bool = False,
    ):
        self.open_time = open_time
        self.close_time = close_time
        self.timezone = timezone
        self.raw = raw
        self.is_24h = is_24h
        self.is_closed = is_closed

    def is_open_at(self, dt: datetime) -> bool:
        """Check if venue is open at the given datetime."""
        if self.is_closed:
            return False
        if self.is_24h:
            return True
        if self.open_time is None or self.close_time is None:
            return False  # Can't determine -- fail-safe

        # Convert to venue timezone
        tz = ZoneInfo(self.timezone)
        local_dt = dt.astimezone(tz)
        current_time = local_dt.time()

        # Handle overnight hours (e.g., 6 PM - 2 AM)
        if self.close_time < self.open_time:
            return current_time >= self.open_time or current_time < self.close_time
        return self.open_time <= current_time < self.close_time


# Time parsing patterns
_TIME_PATTERN = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm|a\.m\.|p\.m\.)?",
    re.IGNORECASE,
)

_HOURS_RANGE_PATTERN = re.compile(
    r"(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)?)"
    r"\s*[-\u2013\u2014to]+"
    r"\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)?)",
    re.IGNORECASE,
)


def _parse_time(time_str: str) -> time | None:
    """Parse a time string like '11:00 AM' or '6pm' into a time object."""
    match = _TIME_PATTERN.search(time_str.strip())
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    period = match.group(3)

    if period:
        period_upper = period.upper().replace(".", "")
        if period_upper == "PM" and hour != 12:
            hour += 12
        elif period_upper == "AM" and hour == 12:
            hour = 0

    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return time(hour, minute)
    return None


def parse_hours(
    hours_str: str,
    timezone: str = "America/New_York",
) -> StructuredHours:
    """Parse a human-readable hours string into StructuredHours.

    Handles formats:
    - "11:00 AM - 10:00 PM"
    - "6pm-2am" (overnight)
    - "24 hours" / "Open 24 Hours"
    - "Closed" / "Temporarily Closed"

    Args:
        hours_str: Human-readable hours string.
        timezone: IANA timezone (default America/New_York for CT casinos).

    Returns:
        StructuredHours with parsed open/close times.
    """
    if not hours_str:
        return StructuredHours(None, None, timezone, hours_str)

    raw = hours_str.strip()
    lower = raw.lower()

    # Check for 24-hour operation
    if "24 hour" in lower or "24/7" in lower or "always open" in lower:
        return StructuredHours(None, None, timezone, raw, is_24h=True)

    # Check for closed
    if lower in ("closed", "temporarily closed", "seasonal"):
        return StructuredHours(None, None, timezone, raw, is_closed=True)

    # Try to parse range
    range_match = _HOURS_RANGE_PATTERN.search(raw)
    if range_match:
        open_t = _parse_time(range_match.group(1))
        close_t = _parse_time(range_match.group(2))
        return StructuredHours(open_t, close_t, timezone, raw)

    return StructuredHours(None, None, timezone, raw)


def is_open_now(
    hours_str: str,
    timezone: str = "America/New_York",
) -> bool | None:
    """Check if a venue is currently open based on hours string.

    Args:
        hours_str: Human-readable hours string.
        timezone: IANA timezone for the property.

    Returns:
        True if open, False if closed, None if can't determine.
    """
    parsed = parse_hours(hours_str, timezone)
    if parsed.is_closed:
        return False
    if parsed.is_24h:
        return True
    if parsed.open_time is None or parsed.close_time is None:
        return None  # Can't determine

    now = datetime.now(ZoneInfo(timezone))
    return parsed.is_open_at(now)
