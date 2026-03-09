"""Tests for tool_binding.py — per-agent tool mapping and LLM binding.

Tests verify:
1. Each agent gets the correct set of tools
2. Unknown agents get empty tool lists
3. bind_tools_to_llm respects feature flag
4. bind_tools_to_llm handles LLM without bind_tools gracefully

R106: Architecture shift — tool-use instead of prompt engineering.
"""

import pytest
from unittest.mock import MagicMock


class TestGetToolsForAgent:
    """Test per-agent tool mapping."""

    def test_comp_agent_gets_all_4_tools(self):
        from src.agent.agents.tool_binding import get_tools_for_agent

        tools = get_tools_for_agent("comp")
        names = [t.name for t in tools]
        assert len(tools) == 4
        assert "check_comp_eligibility" in names
        assert "check_tier_status" in names
        assert "lookup_upcoming_events" in names
        assert "check_incentive_eligibility" in names

    def test_host_agent_gets_all_4_tools(self):
        from src.agent.agents.tool_binding import get_tools_for_agent

        tools = get_tools_for_agent("host")
        assert len(tools) == 4

    def test_dining_agent_gets_2_tools(self):
        from src.agent.agents.tool_binding import get_tools_for_agent

        tools = get_tools_for_agent("dining")
        names = [t.name for t in tools]
        assert len(tools) == 2
        assert "lookup_upcoming_events" in names
        assert "check_tier_status" in names
        # Should NOT have comp or incentive tools
        assert "check_comp_eligibility" not in names
        assert "check_incentive_eligibility" not in names

    def test_entertainment_agent_gets_2_tools(self):
        from src.agent.agents.tool_binding import get_tools_for_agent

        tools = get_tools_for_agent("entertainment")
        assert len(tools) == 2

    def test_hotel_agent_gets_2_tools(self):
        from src.agent.agents.tool_binding import get_tools_for_agent

        tools = get_tools_for_agent("hotel")
        assert len(tools) == 2

    def test_unknown_agent_gets_empty_list(self):
        from src.agent.agents.tool_binding import get_tools_for_agent

        tools = get_tools_for_agent("unknown_agent")
        assert tools == []

    def test_empty_agent_name_gets_empty_list(self):
        from src.agent.agents.tool_binding import get_tools_for_agent

        tools = get_tools_for_agent("")
        assert tools == []

    def test_tools_are_callable(self):
        from src.agent.agents.tool_binding import get_tools_for_agent

        tools = get_tools_for_agent("comp")
        for tool in tools:
            assert hasattr(tool, "invoke"), f"Tool {tool.name} is not invocable"
            assert hasattr(tool, "name"), f"Tool missing name attribute"

    def test_all_tools_have_descriptions(self):
        from src.agent.agents.tool_binding import get_tools_for_agent

        tools = get_tools_for_agent("host")
        for tool in tools:
            assert tool.description, f"Tool {tool.name} has empty description"


class TestBindToolsToLlm:
    """Test LLM tool binding."""

    def test_feature_flag_off_returns_original_llm(self):
        from src.agent.agents.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        llm, tools, is_bound = bind_tools_to_llm(
            mock_llm, "comp", tool_use_enabled=False
        )
        assert llm is mock_llm
        assert tools == []
        assert is_bound is False
        mock_llm.bind_tools.assert_not_called()

    def test_feature_flag_on_binds_tools(self):
        from src.agent.agents.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound

        llm, tools, is_bound = bind_tools_to_llm(
            mock_llm, "comp", tool_use_enabled=True
        )
        assert llm is mock_bound
        assert len(tools) == 4
        assert is_bound is True
        mock_llm.bind_tools.assert_called_once()

    def test_unknown_agent_not_bound(self):
        from src.agent.agents.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        llm, tools, is_bound = bind_tools_to_llm(
            mock_llm, "unknown", tool_use_enabled=True
        )
        assert llm is mock_llm
        assert tools == []
        assert is_bound is False

    def test_bind_failure_returns_original_llm(self):
        from src.agent.agents.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        mock_llm.bind_tools.side_effect = AttributeError("No bind_tools method")

        llm, tools, is_bound = bind_tools_to_llm(
            mock_llm, "comp", tool_use_enabled=True
        )
        assert llm is mock_llm
        assert tools == []
        assert is_bound is False

    def test_returns_new_instance_not_mutated(self):
        from src.agent.agents.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound

        llm, _, _ = bind_tools_to_llm(mock_llm, "comp", tool_use_enabled=True)
        # The returned LLM should be the bound version, not the original
        assert llm is not mock_llm
        assert llm is mock_bound

    def test_default_tool_use_enabled_is_false(self):
        from src.agent.agents.tool_binding import bind_tools_to_llm

        mock_llm = MagicMock()
        llm, tools, is_bound = bind_tools_to_llm(mock_llm, "comp")
        assert is_bound is False
