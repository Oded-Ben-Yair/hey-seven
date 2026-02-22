"""Tests for system prompt parameterization (multi-property support).

Verifies that CONCIERGE_SYSTEM_PROMPT uses property_description from
casino profiles instead of hardcoded Mohegan Sun text.
"""

import pytest

from src.agent.prompts import CONCIERGE_SYSTEM_PROMPT, RESPONSIBLE_GAMING_HELPLINES
from src.casino.config import CASINO_PROFILES, DEFAULT_CONFIG, get_casino_profile


class TestPropertyDescriptionInConfig:
    """Verify property_description exists in all casino profiles."""

    def test_default_config_has_property_description(self):
        assert "property_description" in DEFAULT_CONFIG
        assert len(DEFAULT_CONFIG["property_description"]) > 0

    def test_mohegan_sun_has_property_description(self):
        profile = get_casino_profile("mohegan_sun")
        assert "property_description" in profile
        assert "Uncasville" in profile["property_description"]

    def test_foxwoods_has_property_description(self):
        profile = get_casino_profile("foxwoods")
        assert "property_description" in profile
        assert "Mashantucket" in profile["property_description"]

    def test_hard_rock_has_property_description(self):
        profile = get_casino_profile("hard_rock_ac")
        assert "property_description" in profile
        assert "Atlantic City" in profile["property_description"]

    def test_unknown_casino_uses_default(self):
        profile = get_casino_profile("unknown_casino")
        assert profile["property_description"] == DEFAULT_CONFIG["property_description"]


class TestPromptParameterization:
    """Verify system prompt renders property_description from each profile."""

    def _render_prompt(self, casino_id: str) -> str:
        profile = get_casino_profile(casino_id)
        return CONCIERGE_SYSTEM_PROMPT.safe_substitute(
            property_name=profile.get("prompts", {}).get("casino_name_display", "Test"),
            current_time="Monday 3 PM",
            responsible_gaming_helplines=RESPONSIBLE_GAMING_HELPLINES,
            property_description=profile.get("property_description", ""),
        )

    def test_mohegan_sun_prompt_contains_uncasville(self):
        result = self._render_prompt("mohegan_sun")
        assert "Uncasville, Connecticut" in result
        assert "Mohegan Tribe" in result

    def test_foxwoods_prompt_contains_mashantucket(self):
        result = self._render_prompt("foxwoods")
        assert "Mashantucket, Connecticut" in result
        assert "Mashantucket Pequot" in result

    def test_hard_rock_prompt_contains_atlantic_city(self):
        result = self._render_prompt("hard_rock_ac")
        assert "Atlantic City Boardwalk" in result
        assert "rock-and-roll" in result

    def test_unknown_casino_uses_default_description(self):
        result = self._render_prompt("unknown_casino")
        assert "premier casino resort" in result
        # Default description should NOT mention any specific property
        assert "Mohegan" not in result
        assert "Foxwoods" not in result
        assert "Hard Rock" not in result

    def test_prompt_no_mohegan_for_foxwoods(self):
        """Foxwoods prompt must not contain Mohegan Sun references."""
        result = self._render_prompt("foxwoods")
        assert "Mohegan Tribe" not in result
        assert "Uncasville" not in result

    def test_prompt_no_mohegan_for_hard_rock(self):
        """Hard Rock prompt must not contain Mohegan Sun references."""
        result = self._render_prompt("hard_rock_ac")
        assert "Mohegan Tribe" not in result
        assert "Uncasville" not in result

    def test_prompt_template_no_hardcoded_description(self):
        """The template itself must not contain hardcoded Mohegan Sun description."""
        raw_template = CONCIERGE_SYSTEM_PROMPT.template
        assert "Uncasville" not in raw_template
        assert "Mohegan Tribe" not in raw_template
        assert "10,000-seat arena" not in raw_template
        assert "$property_description" in raw_template
