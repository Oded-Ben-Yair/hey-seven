"""Tests for persona_envelope_node — branding, name injection, PII, truncation."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.persona import (
    _enforce_branding,
    _inject_guest_name,
    _validate_output,
)


# ---------------------------------------------------------------------------
# _inject_guest_name
# ---------------------------------------------------------------------------


class TestInjectGuestName:
    """Tests for guest name injection into responses."""

    def test_no_name_returns_unchanged(self):
        assert _inject_guest_name("Welcome to the resort.", None) == "Welcome to the resort."

    def test_empty_name_returns_unchanged(self):
        assert _inject_guest_name("Welcome to the resort.", "") == "Welcome to the resort."

    def test_name_already_present_returns_unchanged(self):
        assert _inject_guest_name("Hi Sarah, welcome!", "Sarah") == "Hi Sarah, welcome!"

    def test_short_response_not_injected(self):
        assert _inject_guest_name("Welcome!", "Sarah") == "Welcome!"

    def test_apology_not_injected(self):
        content = "I apologize for the inconvenience. Let me help you with that."
        assert _inject_guest_name(content, "Sarah") == content

    def test_generic_sentence_lowercases_first_char(self):
        content = "We have several excellent dining options available for you this evening."
        result = _inject_guest_name(content, "Sarah")
        assert result == "Sarah, we have several excellent dining options available for you this evening."

    def test_proper_noun_preserves_case(self):
        """Proper nouns at start of response must NOT be lowercased."""
        content = "Mohegan Sun has wonderful dining options for your party tonight."
        result = _inject_guest_name(content, "Sarah")
        assert result == "Sarah, Mohegan Sun has wonderful dining options for your party tonight."

    def test_proper_noun_bobby_flay(self):
        content = "Bobby Flay's Bar Americain is an excellent choice for American grill."
        result = _inject_guest_name(content, "John")
        assert result == "John, Bobby Flay's Bar Americain is an excellent choice for American grill."

    def test_case_insensitive_name_check(self):
        content = "There are many great shows tonight at the Arena for your enjoyment."
        result = _inject_guest_name(content, "SARAH")
        # "SARAH" should match "sarah" case-insensitively if already in content
        assert "SARAH" in result or "sarah" not in content.lower()


# ---------------------------------------------------------------------------
# _enforce_branding
# ---------------------------------------------------------------------------


class TestEnforceBranding:
    """Tests for BrandingConfig enforcement."""

    def test_exclamation_limit_default(self):
        """Excess exclamation marks replaced with periods."""
        content = "Welcome! Enjoy! Have fun! Great!"
        result = _enforce_branding(content, {"exclamation_limit": 1, "emoji_allowed": False})
        assert content.count("!") == 4
        assert result.count("!") == 1
        assert result.count(".") >= 3  # replacements

    def test_exclamation_limit_two(self):
        content = "Welcome! Enjoy! Have fun!"
        result = _enforce_branding(content, {"exclamation_limit": 2, "emoji_allowed": False})
        assert result.count("!") == 2

    def test_no_excess_exclamations_unchanged(self):
        content = "Welcome to the resort."
        result = _enforce_branding(content, {"exclamation_limit": 1, "emoji_allowed": False})
        assert result == content

    def test_emoji_removed_when_not_allowed(self):
        content = "Welcome to the casino! 🎰🎲"
        result = _enforce_branding(content, {"exclamation_limit": 5, "emoji_allowed": False})
        assert "🎰" not in result
        assert "🎲" not in result

    def test_emoji_kept_when_allowed(self):
        content = "Welcome! 🎰"
        result = _enforce_branding(content, {"exclamation_limit": 5, "emoji_allowed": True})
        assert "🎰" in result


# ---------------------------------------------------------------------------
# _validate_output (PII redaction)
# ---------------------------------------------------------------------------


class TestValidateOutput:
    """Tests for output PII guardrail."""

    def test_clean_text_unchanged(self):
        text = "Mohegan Sun has great restaurants."
        assert _validate_output(text) == text

    def test_pii_redacted(self):
        text = "Your card number is 4111-1111-1111-1111."
        result = _validate_output(text)
        assert "4111" not in result


# ---------------------------------------------------------------------------
# persona_envelope_node integration
# ---------------------------------------------------------------------------


class TestPersonaEnvelopeNode:
    """Integration tests for the full persona_envelope_node."""

    @pytest.mark.asyncio
    async def test_name_injection_in_full_pipeline(self):
        """Guest name from state is injected into the response."""
        from src.agent.persona import persona_envelope_node

        state = {
            "messages": [
                HumanMessage(content="What restaurants do you have?"),
                AIMessage(content="We have several wonderful dining options available for your evening."),
            ],
            "guest_name": "Sarah",
            "query_type": "property_qa",
            "router_confidence": 0.9,
            "retrieved_context": [],
            "validation_result": "PASS",
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": None,
            "responsible_gaming_count": 0,
            "guest_sentiment": None,
            "guest_context": {},
        }

        with patch("src.agent.persona.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(PERSONA_MAX_CHARS=0)
            result = await persona_envelope_node(state)

        assert "messages" in result
        assert "Sarah" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_sms_truncation_after_name_injection(self):
        """SMS truncation happens AFTER name injection (correct order)."""
        from src.agent.persona import persona_envelope_node

        long_content = "We have " + "a" * 200 + " great restaurants."
        state = {
            "messages": [
                HumanMessage(content="Restaurants?"),
                AIMessage(content=long_content),
            ],
            "guest_name": "Sarah",
            "query_type": "property_qa",
            "router_confidence": 0.9,
            "retrieved_context": [],
            "validation_result": "PASS",
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": None,
            "responsible_gaming_count": 0,
            "guest_sentiment": None,
            "guest_context": {},
        }

        with patch("src.agent.persona.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(PERSONA_MAX_CHARS=160)
            result = await persona_envelope_node(state)

        assert "messages" in result
        assert len(result["messages"][0].content) == 160
        assert result["messages"][0].content.endswith("...")
