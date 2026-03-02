"""Tests for warm handoff protocol (Phase 4, B5) and R78 handoff summaries."""

from src.agent.extraction import format_handoff_summary
from src.agent.handoff import HandoffRequest, build_handoff_request


class TestHandoffRequest:
    def test_basic_handoff(self):
        req = build_handoff_request("responsible_gaming", "Guest showing crisis signs")
        assert req.type == "handoff"
        assert req.department == "responsible_gaming"
        assert req.consent_given is False

    def test_handoff_with_guest_context(self):
        fields = {"name": "Sarah", "party_size": 4, "occasion": "birthday"}
        req = build_handoff_request("vip_services", "VIP request", fields)
        assert "Sarah" in req.guest_summary
        assert "birthday" in req.guest_summary

    def test_handoff_urgency_levels(self):
        req = build_handoff_request("front_desk", "General request", urgency="critical")
        assert req.urgency == "critical"

    def test_handoff_with_consent(self):
        req = build_handoff_request("general", "Guest requested transfer", consent_given=True)
        assert req.consent_given is True

    def test_handoff_empty_fields(self):
        req = build_handoff_request("general", "Test", extracted_fields={})
        assert req.guest_summary == ""

    def test_handoff_model_serialization(self):
        req = build_handoff_request("responsible_gaming", "Crisis", urgency="high")
        data = req.model_dump()
        assert data["type"] == "handoff"
        assert data["urgency"] == "high"

    def test_handoff_uses_format_handoff_summary(self):
        """R78: build_handoff_request now uses format_handoff_summary."""
        fields = {
            "name": "Sarah",
            "party_size": 4,
            "occasion": "birthday",
            "preferences": "Italian",
            "loyalty_signal": "Gold member",
        }
        req = build_handoff_request("vip_services", "VIP request", fields)
        assert "**Guest Handoff Summary**" in req.guest_summary
        assert "Sarah" in req.guest_summary
        assert "birthday" in req.guest_summary
        assert "Italian" in req.guest_summary

    def test_handoff_fallback_when_summary_empty(self):
        """R78: Falls back to inline extraction for unrecognized field patterns."""
        # format_handoff_summary returns empty for fields it doesn't recognize,
        # but inline fallback should still work for known fields
        fields = {"name": "Mike"}
        req = build_handoff_request("general", "Test", fields)
        assert "Mike" in req.guest_summary


# ---------------------------------------------------------------------------
# R78 P9: format_handoff_summary tests
# ---------------------------------------------------------------------------


class TestFormatHandoffSummary:
    """R78 P9: Structured handoff summary for human casino hosts."""

    def test_empty_fields_returns_empty(self):
        assert format_handoff_summary({}) == ""
        assert format_handoff_summary(None) == ""

    def test_identity_fields(self):
        result = format_handoff_summary({"name": "Sarah", "party_size": 4})
        assert "**Guest Handoff Summary**" in result
        assert "Sarah" in result
        assert "4" in result

    def test_visit_context(self):
        result = format_handoff_summary({
            "visit_purpose": "Anniversary trip",
            "occasion": "anniversary",
            "visit_date": "next Saturday",
        })
        assert "anniversary" in result.lower()
        assert "Anniversary trip" in result
        assert "next Saturday" in result

    def test_preferences(self):
        result = format_handoff_summary({
            "preferences": "Italian",
            "dietary": "gluten-free",
            "gaming": "slots",
        })
        assert "Italian" in result
        assert "gluten-free" in result
        assert "slots" in result

    def test_loyalty_signals(self):
        result = format_handoff_summary({
            "loyalty_tier": "Platinum",
            "visit_frequency": "monthly",
        })
        assert "Platinum" in result
        assert "monthly" in result

    def test_behavioral_signals(self):
        result = format_handoff_summary({
            "urgency": True,
            "fatigue": True,
            "budget_conscious": True,
        })
        assert "time-constrained" in result
        assert "fatigued" in result
        assert "budget-conscious" in result

    def test_no_fields_populated_returns_empty(self):
        """Fields exist but are falsy."""
        result = format_handoff_summary({"name": "", "party_size": None})
        assert result == ""

    def test_full_profile(self):
        """Full rich profile produces comprehensive summary."""
        result = format_handoff_summary({
            "name": "Sarah",
            "party_size": 6,
            "party_composition": "couple with 4 friends",
            "occasion": "birthday",
            "preferences": "Italian, seafood",
            "dietary": "gluten-free",
            "gaming": "slots",
            "entertainment": "comedy shows",
            "spa": "massage",
            "loyalty_tier": "Gold",
            "loyalty_signal": "10 years member",
            "visit_frequency": "monthly",
            "urgency": True,
        })
        assert "Sarah" in result
        assert "6" in result
        assert "birthday" in result
        assert "Italian" in result
        assert "Gold" in result
        assert "time-constrained" in result
