"""Tests for namespaced preference memory (Phase 4, B7)."""

from src.data.guest_profile import namespace_preferences


class TestNamespacedPreferences:
    def test_personal_namespace(self):
        fields = {"name": "Sarah", "party_size": 4, "visit_date": "next Friday"}
        ns = namespace_preferences(fields)
        assert ns["personal"]["name"] == "Sarah"
        assert ns["personal"]["party_size"] == 4

    def test_dining_namespace(self):
        fields = {"preferences": "vegetarian, gluten-free"}
        ns = namespace_preferences(fields)
        assert ns["dining"]["preferences"] == "vegetarian, gluten-free"

    def test_behavioral_namespace(self):
        fields = {"loyalty_signal": "Gold member 10 years", "urgency": True}
        ns = namespace_preferences(fields)
        assert ns["behavioral"]["loyalty_signal"] == "Gold member 10 years"
        assert ns["behavioral"]["urgency"] is True

    def test_empty_fields(self):
        assert namespace_preferences({}) == {}
        assert namespace_preferences(None) == {}

    def test_removes_empty_namespaces(self):
        fields = {"name": "John"}
        ns = namespace_preferences(fields)
        assert "personal" in ns
        assert "dining" not in ns
        assert "behavioral" not in ns

    def test_mixed_fields(self):
        fields = {
            "name": "Maria",
            "preferences": "seafood",
            "budget_conscious": True,
            "occasion": "anniversary",
        }
        ns = namespace_preferences(fields)
        assert ns["personal"]["name"] == "Maria"
        assert ns["personal"]["occasion"] == "anniversary"
        assert ns["dining"]["preferences"] == "seafood"
        assert ns["behavioral"]["budget_conscious"] is True
