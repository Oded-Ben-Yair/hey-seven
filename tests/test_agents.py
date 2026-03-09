"""Tests for the v2 specialist agents (src/agent/agents/).

Mock purge R111: Retained only deterministic tests that do not depend on
MagicMock/AsyncMock/@patch. All behavioral validation uses live eval.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Host Agent — deterministic (empty context, fallback, property name)
# ---------------------------------------------------------------------------


class TestHostAgent:
    async def test_empty_context_returns_no_info_fallback(self):
        """Empty retrieved_context returns 'no info' fallback with skip_validation."""
        from src.agent.agents.host_agent import host_agent

        state = {
            "messages": [HumanMessage(content="What about the moon?")],
            "query_type": None,
            "router_confidence": 0.0,
            "retrieved_context": [],
            "validation_result": None,
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": None,
        }
        result = await host_agent(state)
        assert result["skip_validation"] is True
        assert len(result["messages"]) == 1
        assert "don't have that specific info" in result["messages"][0].content

    async def test_fallback_includes_property_name(self):
        """Fallback messages include property name from config."""
        from src.agent.agents.host_agent import host_agent

        state = {
            "messages": [HumanMessage(content="Tell me about X")],
            "query_type": None,
            "router_confidence": 0.0,
            "retrieved_context": [],
            "validation_result": None,
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": None,
        }
        result = await host_agent(state)
        content = result["messages"][0].content
        assert "Mohegan Sun" in content


# ---------------------------------------------------------------------------
# Dining Agent — deterministic
# ---------------------------------------------------------------------------


class TestDiningAgent:
    async def test_empty_context_returns_dining_fallback(self):
        """Empty context returns dining-specific fallback."""
        from src.agent.agents.dining_agent import dining_agent

        state = {
            "messages": [HumanMessage(content="Do you have a sushi bar?")],
            "query_type": None,
            "router_confidence": 0.0,
            "retrieved_context": [],
            "validation_result": None,
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": None,
        }
        result = await dining_agent(state)
        assert result["skip_validation"] is True
        assert "dining" in result["messages"][0].content.lower()

    def test_dining_prompt_mentions_cuisine(self):
        """Dining system prompt includes cuisine and dietary guidance."""
        from src.agent.agents.dining_agent import DINING_SYSTEM_PROMPT

        prompt_text = DINING_SYSTEM_PROMPT.safe_substitute(
            property_name="Test Casino",
            current_time="Monday 3 PM",
            responsible_gaming_helplines="1-800-TEST",
        )
        assert "cuisine" in prompt_text.lower()
        assert "dietary" in prompt_text.lower()
        assert "dress code" in prompt_text.lower()


# ---------------------------------------------------------------------------
# Entertainment Agent — deterministic
# ---------------------------------------------------------------------------


class TestEntertainmentAgent:
    async def test_empty_context_returns_entertainment_fallback(self):
        """Empty context returns entertainment-specific fallback."""
        from src.agent.agents.entertainment_agent import entertainment_agent

        state = {
            "messages": [HumanMessage(content="Any opera performances?")],
            "query_type": None,
            "router_confidence": 0.0,
            "retrieved_context": [],
            "validation_result": None,
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": None,
        }
        result = await entertainment_agent(state)
        assert result["skip_validation"] is True
        assert (
            "entertainment" in result["messages"][0].content.lower()
            or "show" in result["messages"][0].content.lower()
        )

    def test_entertainment_prompt_mentions_shows_and_spa(self):
        """Entertainment system prompt includes shows and spa guidance."""
        from src.agent.agents.entertainment_agent import ENTERTAINMENT_SYSTEM_PROMPT

        prompt_text = ENTERTAINMENT_SYSTEM_PROMPT.safe_substitute(
            property_name="Test Casino",
            current_time="Monday 3 PM",
            responsible_gaming_helplines="1-800-TEST",
        )
        assert "show" in prompt_text.lower()
        assert "spa" in prompt_text.lower()
        assert "ticket" in prompt_text.lower()


# ---------------------------------------------------------------------------
# Comp Agent — deterministic
# ---------------------------------------------------------------------------


class TestCompAgent:
    async def test_empty_context_returns_comp_fallback(self):
        """Empty context returns comp-specific fallback."""
        from src.agent.agents.comp_agent import comp_agent

        state = {
            "messages": [HumanMessage(content="What comps can I get?")],
            "query_type": None,
            "router_confidence": 0.0,
            "retrieved_context": [],
            "validation_result": None,
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": None,
        }
        result = await comp_agent(state)
        assert result["skip_validation"] is True
        assert (
            "loyalty" in result["messages"][0].content.lower()
            or "promotions" in result["messages"][0].content.lower()
            or "rewards" in result["messages"][0].content.lower()
        )

    def test_comp_prompt_uses_cautious_language(self):
        """Comp system prompt includes cautious language guidance."""
        from src.agent.agents.comp_agent import COMP_SYSTEM_PROMPT

        prompt_text = COMP_SYSTEM_PROMPT.safe_substitute(
            property_name="Test Casino",
            current_time="Monday 3 PM",
            responsible_gaming_helplines="1-800-TEST",
        )
        assert "may be eligible" in prompt_text.lower()
        assert "never promise" in prompt_text.lower()
        assert "never guarantee" in prompt_text.lower()

    def test_comp_prompt_includes_momentum_tiers(self):
        """R77: Comp system prompt includes Momentum tier information."""
        from src.agent.agents.comp_agent import COMP_SYSTEM_PROMPT

        prompt_text = COMP_SYSTEM_PROMPT.safe_substitute(
            property_name="Mohegan Sun",
            current_time="Monday 3 PM",
            responsible_gaming_helplines="1-800-TEST",
        )
        assert "momentum" in prompt_text.lower()
        assert "core" in prompt_text.lower()
        assert "ignite" in prompt_text.lower()
        assert "soar" in prompt_text.lower()


# ---------------------------------------------------------------------------
# Hotel Agent — deterministic
# ---------------------------------------------------------------------------


class TestHotelAgent:
    async def test_empty_context_returns_hotel_fallback(self):
        """Empty context returns hotel-specific fallback."""
        from src.agent.agents.hotel_agent import hotel_agent

        state = {
            "messages": [HumanMessage(content="Do you have a presidential suite?")],
            "query_type": None,
            "router_confidence": 0.0,
            "retrieved_context": [],
            "validation_result": None,
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": None,
        }
        result = await hotel_agent(state)
        assert result["skip_validation"] is True
        assert (
            "room" in result["messages"][0].content.lower()
            or "suite" in result["messages"][0].content.lower()
        )

    def test_hotel_prompt_mentions_rooms_and_checkin(self):
        """Hotel system prompt includes rooms, suites, and check-in guidance."""
        from src.agent.agents.hotel_agent import HOTEL_SYSTEM_PROMPT

        prompt_text = HOTEL_SYSTEM_PROMPT.safe_substitute(
            property_name="Test Casino",
            current_time="Monday 3 PM",
            responsible_gaming_helplines="1-800-TEST",
        )
        assert "room" in prompt_text.lower()
        assert "suite" in prompt_text.lower()
        assert "check-in" in prompt_text.lower()

    async def test_hotel_agent_registered(self):
        """Hotel agent is registered in the agent registry."""
        from src.agent.agents.registry import list_agents

        assert "hotel" in list_agents()


# ---------------------------------------------------------------------------
# Registry — fully deterministic
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_get_agent_host(self):
        """get_agent('host') returns the host_agent function."""
        from src.agent.agents.host_agent import host_agent
        from src.agent.agents.registry import get_agent

        assert get_agent("host") is host_agent

    def test_get_agent_dining(self):
        """get_agent('dining') returns the dining_agent function."""
        from src.agent.agents.dining_agent import dining_agent
        from src.agent.agents.registry import get_agent

        assert get_agent("dining") is dining_agent

    def test_get_agent_entertainment(self):
        """get_agent('entertainment') returns the entertainment_agent function."""
        from src.agent.agents.entertainment_agent import entertainment_agent
        from src.agent.agents.registry import get_agent

        assert get_agent("entertainment") is entertainment_agent

    def test_get_agent_comp(self):
        """get_agent('comp') returns the comp_agent function."""
        from src.agent.agents.comp_agent import comp_agent
        from src.agent.agents.registry import get_agent

        assert get_agent("comp") is comp_agent

    def test_get_agent_hotel(self):
        """get_agent('hotel') returns the hotel_agent function."""
        from src.agent.agents.hotel_agent import hotel_agent
        from src.agent.agents.registry import get_agent

        assert get_agent("hotel") is hotel_agent

    def test_get_agent_nonexistent_raises_key_error(self):
        """get_agent('nonexistent') raises KeyError with helpful message."""
        from src.agent.agents.registry import get_agent

        with pytest.raises(KeyError, match="Unknown agent: nonexistent"):
            get_agent("nonexistent")

    def test_list_agents_returns_all_five(self):
        """list_agents() returns all 5 registered agent names."""
        from src.agent.agents.registry import list_agents

        agents = list_agents()
        assert len(agents) == 5
        assert "host" in agents
        assert "dining" in agents
        assert "entertainment" in agents
        assert "comp" in agents
        assert "hotel" in agents

    def test_list_agents_returns_sorted(self):
        """list_agents() returns names in sorted order."""
        from src.agent.agents.registry import list_agents

        agents = list_agents()
        assert agents == sorted(agents)


# ---------------------------------------------------------------------------
# Package Import — fully deterministic
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_all_agents_importable_from_package(self):
        """All agents and get_agent are importable from the agents package."""
        from src.agent.agents import (
            comp_agent,
            dining_agent,
            entertainment_agent,
            get_agent,
            host_agent,
            hotel_agent,
        )

        assert callable(host_agent)
        assert callable(dining_agent)
        assert callable(entertainment_agent)
        assert callable(comp_agent)
        assert callable(hotel_agent)
        assert callable(get_agent)

    def test_all_exports_match_dunder_all(self):
        """Package __all__ matches actual exports."""
        import src.agent.agents as agents_pkg

        for name in agents_pkg.__all__:
            assert hasattr(agents_pkg, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Parametrized Specialist Contract — empty context only (deterministic)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_fn,agent_module",
    [
        ("host_agent", "src.agent.agents.host_agent"),
        ("dining_agent", "src.agent.agents.dining_agent"),
        ("entertainment_agent", "src.agent.agents.entertainment_agent"),
        ("comp_agent", "src.agent.agents.comp_agent"),
        ("hotel_agent", "src.agent.agents.hotel_agent"),
    ],
)
class TestSpecialistContractDeterministic:
    """Verify all 5 specialist agents follow the same empty-context contract."""

    @pytest.mark.asyncio
    async def test_empty_context_returns_fallback(self, agent_fn, agent_module):
        """All specialists return domain-specific fallback on empty context."""
        import importlib

        mod = importlib.import_module(agent_module)
        fn = getattr(mod, agent_fn)

        state = {
            "messages": [HumanMessage(content="Tell me about the moon")],
            "query_type": None,
            "router_confidence": 0.0,
            "retrieved_context": [],
            "validation_result": None,
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": None,
        }
        result = await fn(state)
        assert result.get("skip_validation") is True
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
