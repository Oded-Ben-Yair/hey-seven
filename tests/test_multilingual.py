"""Tests for Phase 1 multilingual support (Spanish language detection + responses).

Mock purge R111: Retained only deterministic tests that do not depend on
MagicMock/AsyncMock/@patch for LLM calls. Covers: RouterOutput model
validation, crisis response localization, Spanish helplines.
"""

import pytest
from pydantic import ValidationError

from src.agent.state import RouterOutput


# ---------------------------------------------------------------------------
# Spanish Crisis Resources Tests (deterministic — pure function)
# ---------------------------------------------------------------------------


class TestSpanishCrisisResources:
    """Verify get_crisis_response_es() produces correct Spanish crisis text."""

    def test_crisis_response_es_contains_resources(self):
        """Spanish crisis response includes all required crisis resource numbers."""
        from src.agent.crisis import get_crisis_response_es

        content = get_crisis_response_es("Test Property", "555-0000")

        assert "988" in content
        assert "741741" in content
        assert "911" in content
        assert "1-888-628-9454" in content
        assert "AYUDA" in content
        assert "HOLA" in content

    def test_crisis_response_es_contains_property_info(self):
        """Spanish crisis response includes property name and phone."""
        from src.agent.crisis import get_crisis_response_es

        content = get_crisis_response_es("Mohegan Sun", "1-888-226-7711")

        assert "Mohegan Sun" in content
        assert "1-888-226-7711" in content


# ---------------------------------------------------------------------------
# Spanish Helplines Tests (deterministic — pure function)
# ---------------------------------------------------------------------------


class TestSpanishHelplines:
    """Verify get_responsible_gaming_helplines_es() returns localized helplines."""

    def test_helplines_es_mohegan_sun(self):
        """Mohegan Sun (CT) helplines include CT-specific info in Spanish."""
        from src.agent.prompts import get_responsible_gaming_helplines_es

        result = get_responsible_gaming_helplines_es("mohegan_sun")

        assert "1-800-GAMBLER" in result or "1-800-426-2537" in result
        assert any(
            word in result.lower()
            for word in [
                "servicio en espa",
                "ayuda",
                "autoexclu",
            ]
        ), f"Expected Spanish helplines, got: {result[:200]}"

    def test_helplines_es_default_fallback(self):
        """Unknown casino falls back to default Spanish helplines."""
        from src.agent.prompts import get_responsible_gaming_helplines_es

        result = get_responsible_gaming_helplines_es("unknown_casino_xyz")

        assert "1-800-GAMBLER" in result or "1-800-426-2537" in result
        assert any(
            word in result.lower()
            for word in [
                "servicio en espa",
                "ayuda",
                "autoexclu",
            ]
        ), f"Expected Spanish helplines fallback, got: {result[:200]}"


# ---------------------------------------------------------------------------
# RouterOutput Model Tests (deterministic — Pydantic model validation)
# ---------------------------------------------------------------------------


class TestRouterOutputLanguageField:
    """Verify RouterOutput Pydantic model language field behavior."""

    def test_detected_language_defaults_to_en(self):
        """RouterOutput.detected_language defaults to 'en' when not specified."""
        output = RouterOutput(query_type="greeting", confidence=0.9)
        assert output.detected_language == "en"

    def test_detected_language_accepts_es(self):
        """RouterOutput accepts 'es' for detected_language."""
        output = RouterOutput(
            query_type="property_qa", confidence=0.85, detected_language="es"
        )
        assert output.detected_language == "es"

    def test_detected_language_accepts_other(self):
        """RouterOutput accepts 'other' for unsupported languages."""
        output = RouterOutput(
            query_type="off_topic", confidence=0.7, detected_language="other"
        )
        assert output.detected_language == "other"

    def test_detected_language_rejects_invalid(self):
        """RouterOutput rejects invalid language codes."""
        with pytest.raises(ValidationError):
            RouterOutput(query_type="greeting", confidence=0.9, detected_language="fr")
