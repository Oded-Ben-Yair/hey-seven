"""Tests for tool_binding.py -- per-agent tool mapping.

Tests verify:
1. Each agent gets the correct set of tools
2. Unknown agents get empty tool lists
3. Tools are callable and have descriptions

Mock purge R111: Removed TestBindToolsToLlm (used MagicMock for LLM binding).
Retained TestGetToolsForAgent (deterministic, imports real tool objects).

R106: Architecture shift -- tool-use instead of prompt engineering.
"""

import pytest


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
