"""Tests verifying regulatory response content includes required information.

These tests ensure that safety-critical responses contain the legally required
information (helplines, crisis resources, jurisdictional references) -- not just
that they route to the correct node.
"""

import pytest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage


@pytest.fixture
def _mock_settings(monkeypatch):
    """Provide consistent settings for regulatory tests."""
    monkeypatch.setenv("PROPERTY_NAME", "Test Casino")
    monkeypatch.setenv("PROPERTY_PHONE", "1-555-TEST")
    monkeypatch.setenv("PROPERTY_STATE", "Connecticut")
    monkeypatch.setenv("CASINO_ID", "mohegan_sun")
    from src.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("_mock_settings")
class TestSelfHarmResponse:
    """Verify self-harm crisis response includes 988 Lifeline."""

    async def test_self_harm_response_includes_988_lifeline(self):
        from src.agent.nodes import off_topic_node
        state = {"query_type": "self_harm", "messages": [HumanMessage(content="test")]}
        result = await off_topic_node(state)
        content = result["messages"][0].content
        assert "988" in content, "Self-harm response must include 988 Lifeline number"

    async def test_self_harm_response_includes_crisis_text_line(self):
        from src.agent.nodes import off_topic_node
        state = {"query_type": "self_harm", "messages": [HumanMessage(content="test")]}
        result = await off_topic_node(state)
        content = result["messages"][0].content
        assert "741741" in content, "Self-harm response must include Crisis Text Line number"

    async def test_self_harm_response_includes_911(self):
        from src.agent.nodes import off_topic_node
        state = {"query_type": "self_harm", "messages": [HumanMessage(content="test")]}
        result = await off_topic_node(state)
        content = result["messages"][0].content
        assert "911" in content, "Self-harm response must include emergency 911"


@pytest.mark.usefixtures("_mock_settings")
class TestAgeVerificationResponse:
    """Verify age verification response includes required info."""

    async def test_age_verification_includes_21_requirement(self):
        from src.agent.nodes import off_topic_node
        state = {"query_type": "age_verification", "messages": [HumanMessage(content="test")]}
        result = await off_topic_node(state)
        content = result["messages"][0].content
        assert "21" in content, "Age verification must mention 21+ requirement"

    async def test_age_verification_includes_photo_id(self):
        from src.agent.nodes import off_topic_node
        state = {"query_type": "age_verification", "messages": [HumanMessage(content="test")]}
        result = await off_topic_node(state)
        content = result["messages"][0].content
        assert "photo ID" in content or "ID" in content, "Must mention ID requirement"


@pytest.mark.usefixtures("_mock_settings")
class TestPatronPrivacyResponse:
    """Verify patron privacy response blocks information sharing."""

    async def test_patron_privacy_blocks_correctly(self):
        from src.agent.nodes import off_topic_node
        state = {"query_type": "patron_privacy", "messages": [HumanMessage(content="test")]}
        result = await off_topic_node(state)
        content = result["messages"][0].content
        assert "not able to share" in content.lower() or "cannot share" in content.lower() or "i'm not able" in content.lower(), (
            "Patron privacy response must clearly refuse to share guest info"
        )


@pytest.mark.usefixtures("_mock_settings")
class TestResponsibleGamingHelplines:
    """Verify responsible gaming responses include correct helplines per jurisdiction."""

    async def test_gambling_advice_includes_helpline(self):
        from src.agent.nodes import off_topic_node
        state = {
            "query_type": "gambling_advice",
            "messages": [HumanMessage(content="test")],
            "responsible_gaming_count": 0,
        }
        result = await off_topic_node(state)
        content = result["messages"][0].content
        # Should include at least one helpline number
        assert "1-800" in content or "800" in content, (
            "Responsible gaming response must include a helpline number"
        )

    async def test_gambling_advice_escalation_after_threshold(self):
        from src.agent.nodes import off_topic_node
        state = {
            "query_type": "gambling_advice",
            "messages": [HumanMessage(content="test")],
            "responsible_gaming_count": 3,
        }
        result = await off_topic_node(state)
        content = result["messages"][0].content
        assert "several times" in content.lower() or "live" in content.lower(), (
            "Repeated RG triggers must escalate to live support"
        )


@pytest.mark.usefixtures("_mock_settings")
class TestBsaAmlResponse:
    """Verify BSA/AML response redirects to compliance team."""

    async def test_bsa_aml_redirects_to_compliance(self):
        from src.agent.nodes import off_topic_node
        state = {"query_type": "bsa_aml", "messages": [HumanMessage(content="test")]}
        result = await off_topic_node(state)
        content = result["messages"][0].content
        # R85: BSA response now redirects to "financial services team" (non-engaging)
        assert "financial" in content.lower(), (
            "BSA/AML response must redirect to financial services team"
        )
