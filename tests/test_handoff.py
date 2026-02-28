"""Tests for warm handoff protocol (Phase 4, B5)."""

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
