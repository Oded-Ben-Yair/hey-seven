"""Tests for Phase 1 multilingual support (Spanish language detection + responses).

Covers:
- Language detection in router_node (detected_language field)
- Feature flag gating (spanish_support_enabled)
- Spanish greeting, off-topic, and crisis responses
- Spanish helpline functions
- Crisis response localization
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.state import RouterOutput


def _state(**overrides):
    """Create a minimal PropertyQAState dict with defaults for multilingual tests."""
    base = {
        "messages": [],
        "query_type": None,
        "router_confidence": 0.0,
        "retrieved_context": [],
        "validation_result": None,
        "retry_count": 0,
        "skip_validation": False,
        "retry_feedback": None,
        "current_time": "Friday, February 28, 2026 03:00 PM UTC",
        "sources_used": [],
        "extracted_fields": {},
        "whisper_plan": None,
        "responsible_gaming_count": 0,
        "guest_sentiment": None,
        "guest_context": {},
        "guest_name": None,
        "specialist_name": None,
        "dispatch_method": None,
        "suggestion_offered": False,
        "domains_discussed": [],
        "crisis_active": False,
        "detected_language": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Language Detection Tests
# ---------------------------------------------------------------------------


class TestLanguageDetection:
    """Verify router_node populates detected_language from RouterOutput."""

    @patch("src.agent.nodes._get_llm", new_callable=AsyncMock)
    async def test_spanish_message_detected_as_es(self, mock_get_llm):
        """Router returns detected_language='es' for Spanish input."""
        from src.agent.nodes import router_node

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=RouterOutput(
            query_type="property_qa",
            confidence=0.9,
            detected_language="es",
        ))
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _state(messages=[HumanMessage(content="Hola, necesito informacion sobre los restaurantes")])
        result = await router_node(state)
        assert result["detected_language"] == "es"

    @patch("src.agent.nodes._get_llm", new_callable=AsyncMock)
    async def test_english_message_detected_as_en(self, mock_get_llm):
        """Router returns detected_language='en' for English input."""
        from src.agent.nodes import router_node

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=RouterOutput(
            query_type="property_qa",
            confidence=0.95,
            detected_language="en",
        ))
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _state(messages=[HumanMessage(content="What restaurants do you have?")])
        result = await router_node(state)
        assert result["detected_language"] == "en"

    @patch("src.agent.nodes._get_llm", new_callable=AsyncMock)
    async def test_router_value_error_defaults_to_en(self, mock_get_llm):
        """When router LLM raises ValueError, detected_language defaults to 'en'."""
        from src.agent.nodes import router_node

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=ValueError("parse error"))
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _state(messages=[HumanMessage(content="Hola")])
        result = await router_node(state)
        assert result["detected_language"] == "en"

    @patch("src.agent.nodes._get_llm", new_callable=AsyncMock)
    async def test_router_exception_defaults_to_en(self, mock_get_llm):
        """When router LLM raises general Exception, detected_language defaults to 'en'."""
        from src.agent.nodes import router_node

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("network error"))
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _state(messages=[HumanMessage(content="Hola amigos")])
        result = await router_node(state)
        assert result["detected_language"] == "en"

    async def test_empty_message_detected_language_none(self):
        """Empty message list returns detected_language=None (greeting fallback)."""
        from src.agent.nodes import router_node

        state = _state(messages=[])
        result = await router_node(state)
        assert result["detected_language"] is None
        assert result["query_type"] == "greeting"


# ---------------------------------------------------------------------------
# Feature Flag Gating Tests
# ---------------------------------------------------------------------------


class TestFeatureFlagGating:
    """Verify spanish_support_enabled flag gates Spanish response generation."""

    @patch("src.agent.nodes.is_feature_enabled", new_callable=AsyncMock)
    async def test_spanish_greeting_with_flag_enabled(self, mock_flag):
        """detected_language='es' + flag=True produces Spanish greeting."""
        from src.agent.nodes import greeting_node

        # Return True for all feature flag checks (ai_disclosure, spanish_support)
        mock_flag.return_value = True

        state = _state(detected_language="es")
        result = await greeting_node(state)
        content = result["messages"][0].content

        # Spanish greeting should contain Spanish words
        assert any(word in content.lower() for word in ["hola", "conserje", "ayudar"]), (
            f"Expected Spanish greeting, got: {content[:200]}"
        )

    @patch("src.agent.nodes.is_feature_enabled", new_callable=AsyncMock)
    async def test_spanish_greeting_with_flag_disabled(self, mock_flag):
        """detected_language='es' + flag=False falls back to English greeting."""
        from src.agent.nodes import greeting_node

        # spanish_support_enabled=False should fall through to English path
        mock_flag.return_value = False

        state = _state(detected_language="es")
        result = await greeting_node(state)
        content = result["messages"][0].content

        # English greeting should contain English words
        assert any(word in content for word in ["Hi!", "Seven", "concierge", "help"]), (
            f"Expected English greeting, got: {content[:200]}"
        )

    @patch("src.agent.nodes.is_feature_enabled", new_callable=AsyncMock)
    async def test_english_greeting_unaffected_by_flag(self, mock_flag):
        """detected_language='en' + flag=True still produces English greeting (regression)."""
        from src.agent.nodes import greeting_node

        mock_flag.return_value = True

        state = _state(detected_language="en")
        result = await greeting_node(state)
        content = result["messages"][0].content

        # English language detected — must stay English regardless of flag
        assert "Seven" in content
        assert any(word in content for word in ["Hi!", "concierge", "help"]), (
            f"Expected English greeting for detected_language='en', got: {content[:200]}"
        )


# ---------------------------------------------------------------------------
# Spanish Greeting Tests
# ---------------------------------------------------------------------------


class TestSpanishGreeting:
    """Verify Spanish greeting template renders correctly."""

    @patch("src.agent.nodes.is_feature_enabled", new_callable=AsyncMock, return_value=True)
    async def test_greeting_contains_property_name(self, mock_flag):
        """Spanish greeting includes the property name from settings."""
        from src.agent.nodes import greeting_node
        from src.config import get_settings

        settings = get_settings()
        state = _state(detected_language="es")
        result = await greeting_node(state)
        content = result["messages"][0].content

        assert settings.PROPERTY_NAME in content, (
            f"Spanish greeting missing property name '{settings.PROPERTY_NAME}': {content[:200]}"
        )

    @patch("src.agent.nodes.is_feature_enabled", new_callable=AsyncMock, return_value=True)
    async def test_greeting_contains_categories(self, mock_flag):
        """Spanish greeting includes category bullet points."""
        from src.agent.nodes import greeting_node

        state = _state(detected_language="es")
        result = await greeting_node(state)
        content = result["messages"][0].content

        # The greeting template uses $categories which is built from
        # _build_greeting_categories() — bullets with "- **label**" format
        assert "- **" in content, (
            f"Spanish greeting missing category bullets: {content[:200]}"
        )

    @patch("src.agent.nodes.is_feature_enabled", new_callable=AsyncMock, return_value=True)
    async def test_greeting_response_structure(self, mock_flag):
        """Spanish greeting returns correct dict keys."""
        from src.agent.nodes import greeting_node

        state = _state(detected_language="es")
        result = await greeting_node(state)

        assert "messages" in result
        assert "sources_used" in result
        assert "retrieved_context" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["sources_used"] == []
        assert result["retrieved_context"] == []


# ---------------------------------------------------------------------------
# Spanish Off-Topic Tests
# ---------------------------------------------------------------------------


class TestSpanishOffTopic:
    """Verify off-topic responses in Spanish when language detected."""

    @patch("src.agent.nodes.is_feature_enabled", new_callable=AsyncMock, return_value=True)
    async def test_off_topic_spanish_response(self, mock_flag):
        """detected_language='es' + general off-topic produces Spanish redirect."""
        from src.agent.nodes import off_topic_node

        state = _state(
            query_type="off_topic",
            detected_language="es",
        )
        result = await off_topic_node(state)
        content = result["messages"][0].content

        # Spanish off-topic response should contain Spanish text
        assert any(word in content.lower() for word in ["fuera", "ayudar", "restaurantes"]), (
            f"Expected Spanish off-topic response, got: {content[:200]}"
        )

    @patch("src.agent.nodes.is_feature_enabled", new_callable=AsyncMock, return_value=True)
    async def test_self_harm_spanish_crisis_response(self, mock_flag):
        """detected_language='es' + self_harm query produces Spanish crisis resources."""
        from src.agent.nodes import off_topic_node

        state = _state(
            query_type="self_harm",
            detected_language="es",
        )
        result = await off_topic_node(state)
        content = result["messages"][0].content

        # Must contain critical crisis resources
        assert "988" in content, f"Missing 988 in Spanish crisis response: {content[:200]}"
        assert "1-888-628-9454" in content, (
            f"Missing Spanish 988 line (1-888-628-9454): {content[:200]}"
        )
        assert "AYUDA" in content, f"Missing AYUDA keyword: {content[:200]}"


# ---------------------------------------------------------------------------
# Spanish Crisis Resources Tests
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
# Spanish Helplines Tests
# ---------------------------------------------------------------------------


class TestSpanishHelplines:
    """Verify get_responsible_gaming_helplines_es() returns localized helplines."""

    def test_helplines_es_mohegan_sun(self):
        """Mohegan Sun (CT) helplines include CT-specific info in Spanish."""
        from src.agent.prompts import get_responsible_gaming_helplines_es

        result = get_responsible_gaming_helplines_es("mohegan_sun")

        # Must contain primary helpline
        assert "1-800-GAMBLER" in result or "1-800-426-2537" in result
        # Must be in Spanish
        assert any(word in result.lower() for word in [
            "servicio en espa", "ayuda", "autoexclu",
        ]), f"Expected Spanish helplines, got: {result[:200]}"

    def test_helplines_es_default_fallback(self):
        """Unknown casino falls back to default Spanish helplines."""
        from src.agent.prompts import get_responsible_gaming_helplines_es

        result = get_responsible_gaming_helplines_es("unknown_casino_xyz")

        # Default fallback should still have Spanish text
        assert "1-800-GAMBLER" in result or "1-800-426-2537" in result
        assert any(word in result.lower() for word in [
            "servicio en espa", "ayuda", "autoexclu",
        ]), f"Expected Spanish helplines fallback, got: {result[:200]}"


# ---------------------------------------------------------------------------
# RouterOutput Model Tests
# ---------------------------------------------------------------------------


class TestRouterOutputLanguageField:
    """Verify RouterOutput Pydantic model language field behavior."""

    def test_detected_language_defaults_to_en(self):
        """RouterOutput.detected_language defaults to 'en' when not specified."""
        output = RouterOutput(query_type="greeting", confidence=0.9)
        assert output.detected_language == "en"

    def test_detected_language_accepts_es(self):
        """RouterOutput accepts 'es' for detected_language."""
        output = RouterOutput(query_type="property_qa", confidence=0.85, detected_language="es")
        assert output.detected_language == "es"

    def test_detected_language_accepts_other(self):
        """RouterOutput accepts 'other' for unsupported languages."""
        output = RouterOutput(query_type="off_topic", confidence=0.7, detected_language="other")
        assert output.detected_language == "other"

    def test_detected_language_rejects_invalid(self):
        """RouterOutput rejects invalid language codes."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RouterOutput(query_type="greeting", confidence=0.9, detected_language="fr")
