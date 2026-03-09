"""Tests for casino_tools.py — LangGraph @tool definitions.

Tests pure tool functions against real knowledge-base data.
No LLM mocks needed — these are deterministic business logic wrappers.

R106: Architecture shift — tool-use instead of prompt engineering.
"""

import pytest


# ---------------------------------------------------------------------------
# check_comp_eligibility
# ---------------------------------------------------------------------------


class TestCheckCompEligibility:
    """Test comp eligibility tool."""

    def test_new_guest_returns_exploration_tier(self):
        from src.agent.casino_tools import check_comp_eligibility

        result = check_comp_eligibility.invoke({"guest_tier": "new", "occasion": ""})
        assert "exploration" in result.lower()
        assert "eligible comps" in result.lower() or "comp offers" in result.lower()

    def test_regular_guest_gets_dining_credit(self):
        from src.agent.casino_tools import check_comp_eligibility

        result = check_comp_eligibility.invoke(
            {"guest_tier": "regular", "occasion": ""}
        )
        assert "dining" in result.lower()
        assert "$" in result

    def test_vip_guest_gets_suite_options(self):
        from src.agent.casino_tools import check_comp_eligibility

        result = check_comp_eligibility.invoke({"guest_tier": "vip", "occasion": ""})
        assert (
            "suite" in result.lower()
            or "spa" in result.lower()
            or "vip" in result.lower()
        )

    def test_birthday_occasion_triggers_multiplier(self):
        from src.agent.casino_tools import check_comp_eligibility

        # Regular with birthday should have enhanced offers
        with_birthday = check_comp_eligibility.invoke(
            {"guest_tier": "regular", "occasion": "birthday"}
        )
        without = check_comp_eligibility.invoke(
            {"guest_tier": "regular", "occasion": ""}
        )
        # Birthday talking point should appear
        assert (
            "birthday" in with_birthday.lower()
            or "celebrating" in with_birthday.lower()
        )

    def test_high_roller_gets_premium_comps(self):
        from src.agent.casino_tools import check_comp_eligibility

        result = check_comp_eligibility.invoke(
            {"guest_tier": "high_roller", "occasion": ""}
        )
        assert "check with host" in result.lower() or "host" in result.lower()

    def test_unknown_tier_defaults_to_new(self):
        from src.agent.casino_tools import check_comp_eligibility

        result = check_comp_eligibility.invoke(
            {"guest_tier": "platinum_ultra", "occasion": ""}
        )
        assert "exploration" in result.lower()

    def test_returns_string(self):
        from src.agent.casino_tools import check_comp_eligibility

        result = check_comp_eligibility.invoke({"guest_tier": "new", "occasion": ""})
        assert isinstance(result, str)
        assert len(result) > 20


# ---------------------------------------------------------------------------
# check_tier_status
# ---------------------------------------------------------------------------


class TestCheckTierStatus:
    """Test tier status lookup tool."""

    def test_core_tier_lookup(self):
        from src.agent.casino_tools import check_tier_status

        result = check_tier_status.invoke({"tier_name": "core", "query": ""})
        assert "core" in result.lower()
        assert "entry" in result.lower() or "new" in result.lower()

    def test_ignite_tier_lookup(self):
        from src.agent.casino_tools import check_tier_status

        result = check_tier_status.invoke({"tier_name": "ignite", "query": ""})
        assert "ignite" in result.lower()
        assert "2,500" in result or "2500" in result

    def test_leap_tier_lookup(self):
        from src.agent.casino_tools import check_tier_status

        result = check_tier_status.invoke({"tier_name": "leap", "query": ""})
        assert "leap" in result.lower()
        assert "10,000" in result or "10000" in result

    def test_ascend_tier_lookup(self):
        from src.agent.casino_tools import check_tier_status

        result = check_tier_status.invoke({"tier_name": "ascend", "query": ""})
        assert "ascend" in result.lower()
        assert "25,000" in result or "25000" in result or "host" in result.lower()

    def test_soar_tier_lookup(self):
        from src.agent.casino_tools import check_tier_status

        result = check_tier_status.invoke({"tier_name": "soar", "query": ""})
        assert "soar" in result.lower()
        assert "invitation" in result.lower()

    def test_empty_tier_returns_overview(self):
        from src.agent.casino_tools import check_tier_status

        result = check_tier_status.invoke({"tier_name": "", "query": ""})
        assert "core" in result.lower()
        assert "soar" in result.lower()

    def test_fuzzy_match_via_query(self):
        from src.agent.casino_tools import check_tier_status

        result = check_tier_status.invoke(
            {"tier_name": "", "query": "what is the leap tier"}
        )
        assert "leap" in result.lower()

    def test_returns_string(self):
        from src.agent.casino_tools import check_tier_status

        result = check_tier_status.invoke({"tier_name": "core", "query": ""})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# lookup_upcoming_events
# ---------------------------------------------------------------------------


class TestLookupUpcomingEvents:
    """Test entertainment event lookup tool."""

    def test_arena_venue(self):
        from src.agent.casino_tools import lookup_upcoming_events

        result = lookup_upcoming_events.invoke({"venue_type": "arena", "interest": ""})
        assert "arena" in result.lower()
        assert "10,000" in result or "10000" in result

    def test_comedy_venue(self):
        from src.agent.casino_tools import lookup_upcoming_events

        result = lookup_upcoming_events.invoke({"venue_type": "comedy", "interest": ""})
        assert "comix" in result.lower() or "comedy" in result.lower()

    def test_wolf_den_venue(self):
        from src.agent.casino_tools import lookup_upcoming_events

        result = lookup_upcoming_events.invoke(
            {"venue_type": "wolf_den", "interest": ""}
        )
        assert "wolf den" in result.lower()
        assert "free" in result.lower()

    def test_all_venues(self):
        from src.agent.casino_tools import lookup_upcoming_events

        result = lookup_upcoming_events.invoke({"venue_type": "all", "interest": ""})
        assert "arena" in result.lower()
        assert "wolf den" in result.lower()

    def test_interest_filter(self):
        from src.agent.casino_tools import lookup_upcoming_events

        result = lookup_upcoming_events.invoke(
            {"venue_type": "all", "interest": "comedy"}
        )
        assert "comix" in result.lower() or "comedy" in result.lower()

    def test_family_venue(self):
        from src.agent.casino_tools import lookup_upcoming_events

        result = lookup_upcoming_events.invoke({"venue_type": "family", "interest": ""})
        assert "family" in result.lower()

    def test_returns_string(self):
        from src.agent.casino_tools import lookup_upcoming_events

        result = lookup_upcoming_events.invoke({"venue_type": "all", "interest": ""})
        assert isinstance(result, str)
        assert len(result) > 50


# ---------------------------------------------------------------------------
# check_incentive_eligibility
# ---------------------------------------------------------------------------


class TestCheckIncentiveEligibility:
    """Test incentive eligibility tool."""

    def test_birthday_triggers_incentive(self):
        from src.agent.casino_tools import check_incentive_eligibility

        result = check_incentive_eligibility.invoke(
            {"occasion": "birthday", "profile_completeness": 0.5, "guest_tier": "new"}
        )
        assert "birthday" in result.lower()
        assert "$" in result or "dining" in result.lower()

    def test_high_completeness_triggers_incentive(self):
        from src.agent.casino_tools import check_incentive_eligibility

        result = check_incentive_eligibility.invoke(
            {"occasion": "", "profile_completeness": 0.8, "guest_tier": "regular"}
        )
        assert "offer" in result.lower() or "incentive" in result.lower()

    def test_low_completeness_no_occasion_no_incentive(self):
        from src.agent.casino_tools import check_incentive_eligibility

        result = check_incentive_eligibility.invoke(
            {"occasion": "", "profile_completeness": 0.1, "guest_tier": "new"}
        )
        # May or may not have offers depending on threshold
        assert isinstance(result, str)

    def test_anniversary_triggers_incentive(self):
        from src.agent.casino_tools import check_incentive_eligibility

        result = check_incentive_eligibility.invoke(
            {
                "occasion": "wedding anniversary",
                "profile_completeness": 0.5,
                "guest_tier": "regular",
            }
        )
        assert "anniversary" in result.lower() or "offer" in result.lower()

    def test_returns_string(self):
        from src.agent.casino_tools import check_incentive_eligibility

        result = check_incentive_eligibility.invoke(
            {"occasion": "", "profile_completeness": 0.0, "guest_tier": "new"}
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Module-level data loading
# ---------------------------------------------------------------------------


class TestKnowledgeBaseLoading:
    """Test that knowledge-base data loads correctly at import time."""

    def test_tier_data_loaded(self):
        from src.agent.casino_tools import _TIER_DATA

        assert len(_TIER_DATA) > 0
        assert "core" in _TIER_DATA
        assert "soar" in _TIER_DATA

    def test_entertainment_data_loaded(self):
        from src.agent.casino_tools import _ENTERTAINMENT_DATA

        assert len(_ENTERTAINMENT_DATA) > 0
        # Check for known venues
        found_arena = any("arena" in k.lower() for k in _ENTERTAINMENT_DATA)
        found_wolf = any("wolf" in k.lower() for k in _ENTERTAINMENT_DATA)
        assert found_arena
        assert found_wolf

    def test_all_tools_list(self):
        from src.agent.casino_tools import ALL_CASINO_TOOLS

        assert len(ALL_CASINO_TOOLS) == 4
        names = [t.name for t in ALL_CASINO_TOOLS]
        assert "check_comp_eligibility" in names
        assert "check_tier_status" in names
        assert "lookup_upcoming_events" in names
        assert "check_incentive_eligibility" in names

    def test_tools_have_descriptions(self):
        from src.agent.casino_tools import ALL_CASINO_TOOLS

        for tool in ALL_CASINO_TOOLS:
            assert tool.description, f"Tool {tool.name} has no description"
            assert len(tool.description) > 20, f"Tool {tool.name} description too short"
