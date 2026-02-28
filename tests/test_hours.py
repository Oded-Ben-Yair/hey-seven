"""Tests for structured hours parsing and timezone-aware availability."""

import pytest
from datetime import datetime, time
from zoneinfo import ZoneInfo

from src.agent.hours import parse_hours, is_open_now, StructuredHours


class TestParseHours:
    def test_standard_range(self):
        h = parse_hours("11:00 AM - 10:00 PM")
        assert h.open_time == time(11, 0)
        assert h.close_time == time(22, 0)

    def test_compact_format(self):
        h = parse_hours("6pm-2am")
        assert h.open_time == time(18, 0)
        assert h.close_time == time(2, 0)

    def test_24_hours(self):
        h = parse_hours("Open 24 Hours")
        assert h.is_24h is True

    def test_24_7(self):
        h = parse_hours("24/7")
        assert h.is_24h is True

    def test_always_open(self):
        h = parse_hours("Always Open")
        assert h.is_24h is True

    def test_closed(self):
        h = parse_hours("Closed")
        assert h.is_closed is True

    def test_temporarily_closed(self):
        h = parse_hours("Temporarily Closed")
        assert h.is_closed is True

    def test_seasonal(self):
        h = parse_hours("Seasonal")
        assert h.is_closed is True

    def test_empty_string(self):
        h = parse_hours("")
        assert h.open_time is None
        assert h.close_time is None
        assert h.is_24h is False
        assert h.is_closed is False

    def test_unparseable_string(self):
        h = parse_hours("Call for hours")
        assert h.open_time is None
        assert h.close_time is None

    def test_noon_format(self):
        h = parse_hours("12:00 PM - 11:00 PM")
        assert h.open_time == time(12, 0)
        assert h.close_time == time(23, 0)

    def test_midnight_format(self):
        h = parse_hours("12:00 AM - 6:00 AM")
        assert h.open_time == time(0, 0)
        assert h.close_time == time(6, 0)

    def test_with_timezone(self):
        h = parse_hours("9:00 AM - 5:00 PM", "America/Los_Angeles")
        assert h.timezone == "America/Los_Angeles"
        assert h.open_time == time(9, 0)
        assert h.close_time == time(17, 0)

    def test_raw_preserved(self):
        h = parse_hours("  11:00 AM - 10:00 PM  ")
        assert h.raw == "11:00 AM - 10:00 PM"


class TestIsOpenAt:
    def test_overnight_is_open(self):
        h = parse_hours("6:00 PM - 2:00 AM", "America/New_York")
        # 9 PM should be open
        dt = datetime(2026, 2, 28, 21, 0, tzinfo=ZoneInfo("America/New_York"))
        assert h.is_open_at(dt) is True

    def test_overnight_is_closed(self):
        h = parse_hours("6:00 PM - 2:00 AM", "America/New_York")
        # 10 AM should be closed
        dt = datetime(2026, 2, 28, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        assert h.is_open_at(dt) is False

    def test_overnight_after_midnight_is_open(self):
        h = parse_hours("6:00 PM - 2:00 AM", "America/New_York")
        # 1 AM should be open (after midnight, before close)
        dt = datetime(2026, 3, 1, 1, 0, tzinfo=ZoneInfo("America/New_York"))
        assert h.is_open_at(dt) is True

    def test_24h_always_open(self):
        h = parse_hours("24/7")
        dt = datetime(2026, 2, 28, 3, 30, tzinfo=ZoneInfo("America/New_York"))
        assert h.is_open_at(dt) is True

    def test_closed_never_open(self):
        h = parse_hours("Temporarily Closed")
        dt = datetime(2026, 2, 28, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert h.is_open_at(dt) is False

    def test_standard_hours_inside(self):
        h = parse_hours("9:00 AM - 5:00 PM", "America/New_York")
        dt = datetime(2026, 2, 28, 14, 0, tzinfo=ZoneInfo("America/New_York"))
        assert h.is_open_at(dt) is True

    def test_standard_hours_outside(self):
        h = parse_hours("9:00 AM - 5:00 PM", "America/New_York")
        dt = datetime(2026, 2, 28, 20, 0, tzinfo=ZoneInfo("America/New_York"))
        assert h.is_open_at(dt) is False

    def test_unparseable_returns_false(self):
        h = parse_hours("Call for hours")
        dt = datetime(2026, 2, 28, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert h.is_open_at(dt) is False  # fail-safe

    def test_cross_timezone(self):
        """A venue in LA at 9 AM-5 PM: 3 PM ET should be open (12 PM PT)."""
        h = parse_hours("9:00 AM - 5:00 PM", "America/Los_Angeles")
        dt = datetime(2026, 2, 28, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        assert h.is_open_at(dt) is True  # 15:00 ET = 12:00 PT


class TestIsOpenNow:
    def test_closed_returns_false(self):
        assert is_open_now("Closed") is False

    def test_24h_returns_true(self):
        assert is_open_now("Open 24 Hours") is True

    def test_unparseable_returns_none(self):
        assert is_open_now("Call for hours") is None
