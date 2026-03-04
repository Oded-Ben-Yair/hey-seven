"""Tests for pre_extract_node (R87 SRP refactor).

Validates that deterministic field extraction works correctly as a
standalone graph node, producing the same results as when it was
embedded in router_node.
"""

import pytest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage

from src.agent.pre_extract import pre_extract_node


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear settings cache between tests."""
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestPreExtractNode:
    """Unit tests for pre_extract_node."""

    @pytest.mark.asyncio
    async def test_extracts_name(self):
        """Extracts guest name from 'My name is Sarah'."""
        state = {"messages": [HumanMessage(content="My name is Sarah, party of 4")]}
        result = await pre_extract_node(state)
        assert result.get("guest_name") == "Sarah"
        assert result["extracted_fields"]["name"] == "Sarah"

    @pytest.mark.asyncio
    async def test_extracts_party_size(self):
        """Extracts party size from 'party of 6'."""
        state = {"messages": [HumanMessage(content="We need a table, party of 6")]}
        result = await pre_extract_node(state)
        assert result["extracted_fields"]["party_size"] == 6

    @pytest.mark.asyncio
    async def test_extracts_occasion(self):
        """Extracts occasion from 'celebrating anniversary'."""
        state = {
            "messages": [
                HumanMessage(content="We're celebrating our anniversary tonight")
            ]
        }
        result = await pre_extract_node(state)
        assert result["extracted_fields"]["occasion"] == "anniversary"

    @pytest.mark.asyncio
    async def test_empty_message_returns_empty(self):
        """No messages → empty dict."""
        state = {"messages": []}
        result = await pre_extract_node(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_extractable_content(self):
        """Generic question with no extractable fields → empty dict."""
        state = {"messages": [HumanMessage(content="What time does the pool close?")]}
        result = await pre_extract_node(state)
        assert result == {} or "extracted_fields" not in result

    @pytest.mark.asyncio
    async def test_extracts_preferences(self):
        """Extracts dietary preferences."""
        state = {
            "messages": [
                HumanMessage(content="I'm vegetarian, looking for Italian food")
            ]
        }
        result = await pre_extract_node(state)
        fields = result.get("extracted_fields", {})
        assert "preferences" in fields

    @pytest.mark.asyncio
    async def test_multiple_fields_extracted(self):
        """Extracts multiple fields from rich message."""
        state = {
            "messages": [
                HumanMessage(
                    content="My name is Mike, party of 4, celebrating a birthday"
                )
            ]
        }
        result = await pre_extract_node(state)
        fields = result["extracted_fields"]
        assert fields["name"] == "Mike"
        assert fields["party_size"] == 4
        assert fields["occasion"] == "birthday"
        assert result["guest_name"] == "Mike"

    @pytest.mark.asyncio
    async def test_feature_flag_disabled(self, monkeypatch):
        """When field_extraction_enabled=False, no extraction occurs."""
        monkeypatch.setenv("CASINO_ID", "test_casino")
        from src.config import get_settings

        get_settings.cache_clear()

        with patch(
            "src.agent.pre_extract.is_feature_enabled",
            new_callable=AsyncMock,
            return_value=False,
        ):
            state = {"messages": [HumanMessage(content="My name is Sarah, party of 4")]}
            result = await pre_extract_node(state)
            assert "extracted_fields" not in result

    @pytest.mark.asyncio
    async def test_gemini_list_content_format(self):
        """Handles Gemini 3.x list[dict] content format."""
        msg = HumanMessage(content=[{"type": "text", "text": "I'm Sarah, party of 2"}])
        state = {"messages": [msg]}
        result = await pre_extract_node(state)
        assert result.get("guest_name") == "Sarah"

    @pytest.mark.asyncio
    async def test_loyalty_signal_extraction(self):
        """Extracts loyalty signals from 'Platinum member'."""
        state = {
            "messages": [
                HumanMessage(content="I'm a Platinum member visiting next Friday")
            ]
        }
        result = await pre_extract_node(state)
        fields = result.get("extracted_fields", {})
        assert "loyalty_signal" in fields

    @pytest.mark.asyncio
    async def test_urgency_signal_extraction(self):
        """Extracts urgency signals from 'checking out in an hour'."""
        state = {
            "messages": [
                HumanMessage(
                    content="We're checking out in an hour, need quick breakfast"
                )
            ]
        }
        result = await pre_extract_node(state)
        fields = result.get("extracted_fields", {})
        assert "urgency" in fields
