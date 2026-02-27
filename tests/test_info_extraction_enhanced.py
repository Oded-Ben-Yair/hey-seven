"""Tests for enhanced info extraction and guest profile summary.

R72 Phase C4: Tests for get_guest_profile_summary() and verified
existing extraction patterns for loyalty, urgency, fatigue, budget signals.
"""

import pytest

from src.agent.extraction import extract_fields, get_guest_profile_summary


class TestGuestProfileSummary:
    """get_guest_profile_summary() formatting for human host handoff."""

    def test_empty_fields_returns_empty(self):
        assert get_guest_profile_summary({}) == ""

    def test_none_fields_returns_empty(self):
        assert get_guest_profile_summary(None) == ""

    def test_name_only(self):
        result = get_guest_profile_summary({"name": "Sarah"})
        assert "Sarah" in result
        assert "Guest Profile Summary" in result

    def test_full_profile(self):
        result = get_guest_profile_summary({
            "name": "Sarah",
            "party_size": 5,
            "occasion": "birthday",
            "visit_date": "next Friday",
            "preferences": "vegetarian",
            "loyalty_signal": "Platinum member",
            "urgency": True,
            "fatigue": True,
            "budget_conscious": True,
        })
        assert "Sarah" in result
        assert "5" in result
        assert "birthday" in result
        assert "next Friday" in result
        assert "vegetarian" in result
        assert "Platinum" in result
        assert "time-constrained" in result
        assert "fatigued" in result
        assert "budget-conscious" in result

    def test_partial_profile(self):
        result = get_guest_profile_summary({
            "name": "Mike",
            "party_size": 2,
        })
        assert "Mike" in result
        assert "2" in result
        assert "occasion" not in result.lower()

    def test_behavioral_signals_only(self):
        result = get_guest_profile_summary({
            "urgency": True,
            "budget_conscious": True,
        })
        assert "time-constrained" in result
        assert "budget-conscious" in result


class TestLoyaltyExtraction:
    """Verify loyalty signal extraction patterns."""

    def test_momentum_member(self):
        fields = extract_fields("I'm a Momentum member")
        assert "loyalty_signal" in fields
        assert "Momentum" in fields["loyalty_signal"]

    def test_gold_tier(self):
        fields = extract_fields("I'm Gold tier status")
        assert "loyalty_signal" in fields

    def test_years_of_membership(self):
        fields = extract_fields("I've been a member for 20 years")
        assert "loyalty_signal" in fields

    def test_high_roller_claim(self):
        fields = extract_fields("I'm a high roller here")
        assert "loyalty_signal" in fields

    def test_used_to_be_gold(self):
        fields = extract_fields("I used to be Gold tier")
        assert "loyalty_signal" in fields


class TestUrgencyExtraction:
    """Verify urgency signal extraction patterns."""

    def test_checking_out_in_hour(self):
        fields = extract_fields("Checking out in an hour")
        assert fields.get("urgency") is True

    def test_quick_keyword(self):
        fields = extract_fields("Need something quick for lunch")
        assert fields.get("urgency") is True

    def test_leaving_soon(self):
        fields = extract_fields("We're leaving soon")
        assert fields.get("urgency") is True

    def test_asap(self):
        fields = extract_fields("Need a table asap")
        assert fields.get("urgency") is True


class TestFatigueExtraction:
    """Verify fatigue signal extraction patterns."""

    def test_exhausted(self):
        fields = extract_fields("I'm exhausted from the drive")
        assert fields.get("fatigue") is True

    def test_long_drive(self):
        fields = extract_fields("We had a long drive to get here")
        assert fields.get("fatigue") is True

    def test_need_to_unwind(self):
        fields = extract_fields("I really need to unwind after today")
        assert fields.get("fatigue") is True


class TestBudgetExtraction:
    """Verify budget consciousness extraction patterns."""

    def test_nothing_expensive(self):
        fields = extract_fields("Nothing too expensive please")
        assert fields.get("budget_conscious") is True

    def test_on_a_budget(self):
        fields = extract_fields("We're on a budget this trip")
        assert fields.get("budget_conscious") is True

    def test_affordable(self):
        fields = extract_fields("Looking for something affordable")
        assert fields.get("budget_conscious") is True

    def test_free_activities(self):
        fields = extract_fields("Are there free activities for the kids?")
        assert fields.get("budget_conscious") is True
