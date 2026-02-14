"""Tests for prompt templates (src/agent/prompts.py)."""

from src.agent.prompts import (
    CONCIERGE_SYSTEM_PROMPT,
    RESPONSIBLE_GAMING_HELPLINES,
    ROUTER_PROMPT,
    VALIDATION_PROMPT,
)


class TestPromptTemplates:
    def test_concierge_renders_with_safe_substitute(self):
        """Concierge prompt renders property_name, current_time, and helplines."""
        result = CONCIERGE_SYSTEM_PROMPT.safe_substitute(
            property_name="Test Casino",
            current_time="Monday 3 PM",
            responsible_gaming_helplines=RESPONSIBLE_GAMING_HELPLINES,
        )
        assert "Test Casino" in result
        assert "Monday 3 PM" in result

    def test_brace_safety(self):
        """Templates don't crash on {curly braces} in user content."""
        result = ROUTER_PROMPT.safe_substitute(user_message="What about {this}?")
        assert "{this}" in result

    def test_helpline_numbers_present(self):
        """Concierge prompt includes all responsible gaming helplines via constant."""
        result = CONCIERGE_SYSTEM_PROMPT.safe_substitute(
            property_name="X",
            current_time="now",
            responsible_gaming_helplines=RESPONSIBLE_GAMING_HELPLINES,
        )
        assert "1-800-MY-RESET" in result or "1-800-699-7378" in result
        assert "1-888-789-7777" in result
        assert "ct.gov/selfexclusion" in result

    def test_helplines_constant_contains_all_resources(self):
        """RESPONSIBLE_GAMING_HELPLINES constant includes updated resources."""
        assert "1-800-MY-RESET" in RESPONSIBLE_GAMING_HELPLINES
        assert "1-800-699-7378" in RESPONSIBLE_GAMING_HELPLINES
        assert "1-888-789-7777" in RESPONSIBLE_GAMING_HELPLINES
        assert "ct.gov/selfexclusion" in RESPONSIBLE_GAMING_HELPLINES

    def test_router_prompt_includes_categories(self):
        """Router prompt lists all 7 query categories."""
        result = ROUTER_PROMPT.safe_substitute(user_message="test")
        for category in [
            "property_qa",
            "hours_schedule",
            "greeting",
            "off_topic",
            "gambling_advice",
            "action_request",
            "ambiguous",
        ]:
            assert category in result

    def test_validation_prompt_renders_all_variables(self):
        """Validation prompt renders user_question, retrieved_context, generated_response."""
        result = VALIDATION_PROMPT.safe_substitute(
            user_question="Where is the spa?",
            retrieved_context="[1] (amenities) Spa on level 2",
            generated_response="The spa is on level 2.",
        )
        assert "Where is the spa?" in result
        assert "Spa on level 2" in result
        assert "The spa is on level 2." in result

    def test_validation_prompt_includes_six_criteria(self):
        """Validation prompt checks 6 criteria."""
        result = VALIDATION_PROMPT.safe_substitute(
            user_question="x", retrieved_context="y", generated_response="z"
        )
        assert "Grounded" in result
        assert "On-topic" in result
        assert "No gambling advice" in result
        assert "Read-only" in result
        assert "Accurate" in result
        assert "Responsible gaming" in result
