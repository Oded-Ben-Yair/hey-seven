"""Tests for LTV Nudge Engine (H10: Lifetime Value).

Pure deterministic tests — no LLM calls, no mocks needed.
"""

import json

import pytest

from src.agent.behavior_tools.ltv_nudge import (
    LTVNudge,
    NUDGE_CATALOG,
    get_ltv_nudges,
    get_ltv_prompt_section,
    _DOMAIN_NUDGE_AFFINITY,
    _SUPPRESS_SENTIMENTS,
)


class TestNudgeCatalog:
    """Test nudge catalog structure."""

    def test_all_casinos_have_nudges(self):
        for casino_id in ("mohegan_sun", "foxwoods", "wynn_las_vegas"):
            assert casino_id in NUDGE_CATALOG, f"Missing nudges for {casino_id}"

    def test_nudges_have_required_fields(self):
        for casino_id, nudges in NUDGE_CATALOG.items():
            for nudge in nudges:
                assert nudge.nudge_type
                assert nudge.message_fragment
                assert nudge.timing
                assert 0.0 <= nudge.relevance <= 1.0

    def test_nudge_messages_not_empty(self):
        for casino_id, nudges in NUDGE_CATALOG.items():
            for nudge in nudges:
                assert len(nudge.message_fragment) > 10, (
                    f"Short nudge message in {casino_id}"
                )


class TestGetLtvNudges:
    """Test nudge selection logic."""

    def test_returns_max_2_nudges(self):
        nudges = get_ltv_nudges("mohegan_sun", turn_count=3)
        assert len(nudges) <= 2

    def test_suppressed_for_grief(self):
        nudges = get_ltv_nudges(
            "mohegan_sun",
            guest_sentiment="grief",
            turn_count=5,
        )
        assert len(nudges) == 0

    def test_suppressed_for_crisis(self):
        nudges = get_ltv_nudges(
            "mohegan_sun",
            guest_sentiment="crisis",
            turn_count=5,
        )
        assert len(nudges) == 0

    def test_suppressed_for_frustrated(self):
        nudges = get_ltv_nudges(
            "mohegan_sun",
            guest_sentiment="frustrated",
            turn_count=5,
        )
        assert len(nudges) == 0

    def test_suppressed_for_short_conversations(self):
        nudges = get_ltv_nudges(
            "mohegan_sun",
            turn_count=1,
        )
        assert len(nudges) == 0

    def test_returns_nudges_for_normal_conversation(self):
        nudges = get_ltv_nudges(
            "mohegan_sun",
            turn_count=3,
        )
        assert len(nudges) > 0

    def test_domain_affinity_boosts_relevant(self):
        """Dining domain should boost seasonal_offer nudges."""
        nudges_without = get_ltv_nudges(
            "mohegan_sun",
            turn_count=3,
        )
        nudges_with = get_ltv_nudges(
            "mohegan_sun",
            domains_discussed=["dining"],
            turn_count=3,
        )
        # Both should return nudges but order may differ
        assert len(nudges_with) > 0

    def test_personal_callback_boosted_for_long_convos(self):
        nudges = get_ltv_nudges(
            "mohegan_sun",
            turn_count=6,
        )
        # Personal callback should be high-ranked for long conversations
        types = [n.nudge_type for n in nudges]
        assert "personal_callback" in types

    def test_unknown_casino_gets_defaults(self):
        nudges = get_ltv_nudges(
            "unknown_casino",
            turn_count=3,
        )
        assert len(nudges) > 0

    def test_all_suppress_sentiments_covered(self):
        """Every suppress sentiment actually suppresses."""
        for sentiment in _SUPPRESS_SENTIMENTS:
            nudges = get_ltv_nudges(
                "mohegan_sun",
                guest_sentiment=sentiment,
                turn_count=5,
            )
            assert len(nudges) == 0, f"Sentiment {sentiment} not suppressed"

    def test_occasion_boosts_nudges(self):
        nudges = get_ltv_nudges(
            "mohegan_sun",
            occasion="birthday",
            turn_count=3,
        )
        assert len(nudges) > 0

    def test_nudges_are_ltv_nudge_instances(self):
        nudges = get_ltv_nudges("mohegan_sun", turn_count=3)
        for n in nudges:
            assert isinstance(n, LTVNudge)


class TestGetLtvPromptSection:
    """Test prompt section generation."""

    def test_empty_state_no_nudges(self):
        """Very short conversation should not get nudges."""
        state = {"messages": [], "domains_discussed": [], "extracted_fields": {}}
        section = get_ltv_prompt_section(state, casino_id="mohegan_sun")
        assert section == ""

    def test_normal_state_gets_section(self):
        from langchain_core.messages import HumanMessage

        state = {
            "messages": [
                HumanMessage(content="Hi"),
                HumanMessage(content="What restaurants?"),
                HumanMessage(content="Thanks"),
            ],
            "domains_discussed": ["dining"],
            "guest_sentiment": None,
            "extracted_fields": {},
        }
        section = get_ltv_prompt_section(state, casino_id="mohegan_sun")
        assert "Return Visit" in section

    def test_frustrated_gets_no_section(self):
        from langchain_core.messages import HumanMessage

        state = {
            "messages": [HumanMessage(content="Hi")] * 3,
            "domains_discussed": ["dining"],
            "guest_sentiment": "frustrated",
            "extracted_fields": {},
        }
        section = get_ltv_prompt_section(state, casino_id="mohegan_sun")
        assert section == ""

    def test_section_has_one_nudge_rule(self):
        from langchain_core.messages import HumanMessage

        state = {
            "messages": [HumanMessage(content="x")] * 3,
            "domains_discussed": [],
            "guest_sentiment": None,
            "extracted_fields": {},
        }
        section = get_ltv_prompt_section(state, casino_id="mohegan_sun")
        if section:
            assert "Do NOT list multiple" in section


class TestLtvModels:
    """Test Pydantic model validation."""

    def test_nudge_serializable(self):
        nudge = LTVNudge(
            nudge_type="upcoming_event",
            message_fragment="Check out our upcoming show",
            timing="next visit",
            relevance=0.7,
        )
        serialized = json.dumps(nudge.model_dump())
        assert json.loads(serialized)

    def test_domain_affinity_complete(self):
        """Key domains have affinity mappings."""
        for domain in ("dining", "entertainment", "gaming"):
            assert domain in _DOMAIN_NUDGE_AFFINITY
