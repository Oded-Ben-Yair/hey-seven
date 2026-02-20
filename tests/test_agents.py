"""Tests for the v2 specialist agents (src/agent/agents/)."""

import asyncio

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage


def _state(**overrides):
    """Create a minimal PropertyQAState dict with defaults."""
    base = {
        "messages": [],
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
    base.update(overrides)
    return base


def _high_completeness_fields():
    """Return flat extracted_fields dict with >=60% profile completeness.

    Uses the flat whisper-planner schema (name, visit_date, party_size, etc.)
    with 5/8 fields filled = 62.5% > 60% threshold.
    """
    return {
        "name": "John Doe",
        "visit_date": "2026-03-01",
        "party_size": 4,
        "dining": "steakhouse",
        "entertainment": "comedy show",
        "gaming": None,
        "occasions": None,
        "companions": None,
    }


# ---------------------------------------------------------------------------
# Host Agent
# ---------------------------------------------------------------------------


class TestHostAgent:
    @patch("src.agent.agents.host_agent._get_llm", new_callable=AsyncMock)
    async def test_generates_with_context(self, mock_get_llm):
        """Host agent produces an AIMessage when context is available."""
        from src.agent.agents.host_agent import host_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="The steakhouse opens at 5 PM."))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="When does the steakhouse open?")],
            retrieved_context=[
                {"content": "Steakhouse hours: 5-10 PM", "metadata": {"category": "restaurants"}, "score": 0.9}
            ],
        )
        result = await host_agent(state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "steakhouse" in result["messages"][0].content.lower()

    async def test_empty_context_returns_no_info_fallback(self):
        """Empty retrieved_context returns 'no info' fallback with skip_validation."""
        from src.agent.agents.host_agent import host_agent

        state = _state(
            messages=[HumanMessage(content="What about the moon?")],
            retrieved_context=[],
        )
        result = await host_agent(state)
        assert result["skip_validation"] is True
        assert len(result["messages"]) == 1
        assert "don't have specific information" in result["messages"][0].content

    @patch("src.agent.agents.host_agent._get_circuit_breaker")
    async def test_circuit_breaker_open_returns_fallback(self, mock_get_cb):
        """Host agent returns fallback when circuit breaker is open."""
        from src.agent.agents.host_agent import host_agent

        mock_cb = MagicMock()
        mock_cb.is_open = True
        mock_cb.allow_request = AsyncMock(return_value=False)
        mock_get_cb.return_value = mock_cb

        state = _state(
            messages=[HumanMessage(content="What restaurants?")],
            retrieved_context=[{"content": "data", "metadata": {}, "score": 1.0}],
        )
        result = await host_agent(state)
        assert result["skip_validation"] is True
        assert "technical difficulties" in result["messages"][0].content

    @patch("src.agent.agents.host_agent._get_llm", new_callable=AsyncMock)
    async def test_llm_error_returns_fallback(self, mock_get_llm):
        """LLM error produces a fallback message with skip_validation=True."""
        from src.agent.agents.host_agent import host_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ConnectionError("API error"))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="Question")],
            retrieved_context=[
                {"content": "data", "metadata": {"category": "faq"}, "score": 1.0}
            ],
        )
        result = await host_agent(state)
        assert result["skip_validation"] is True
        assert "trouble generating" in result["messages"][0].content.lower()

    @patch("src.agent.agents.host_agent._get_llm", new_callable=AsyncMock)
    async def test_value_error_returns_fallback(self, mock_get_llm):
        """ValueError from LLM parsing returns fallback."""
        from src.agent.agents.host_agent import host_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ValueError("bad response"))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="Question")],
            retrieved_context=[
                {"content": "data", "metadata": {"category": "faq"}, "score": 1.0}
            ],
        )
        result = await host_agent(state)
        assert result["skip_validation"] is False
        assert result["retry_count"] == 1
        assert "trouble processing" in result["messages"][0].content.lower()

    @patch("src.agent.agents.host_agent._get_llm", new_callable=AsyncMock)
    async def test_retry_injects_feedback(self, mock_get_llm):
        """On retry, host agent injects validation feedback into LLM messages."""
        from src.agent.agents.host_agent import host_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Corrected response."))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="What restaurants?")],
            retrieved_context=[
                {"content": "Steakhouse info", "metadata": {"category": "restaurants"}, "score": 0.9}
            ],
            retry_count=1,
            retry_feedback="Response was not grounded",
        )
        result = await host_agent(state)
        assert len(result["messages"]) == 1
        # Verify feedback was injected by checking LLM was called with it
        call_args = mock_llm.ainvoke.call_args[0][0]
        feedback_msgs = [m for m in call_args if hasattr(m, "content") and "failed validation" in m.content]
        assert len(feedback_msgs) == 1

    async def test_fallback_includes_property_info(self):
        """Fallback messages include property name and phone from config."""
        from src.agent.agents.host_agent import host_agent

        state = _state(
            messages=[HumanMessage(content="Tell me about X")],
            retrieved_context=[],
        )
        result = await host_agent(state)
        content = result["messages"][0].content
        assert "Mohegan Sun" in content
        assert "888" in content


# ---------------------------------------------------------------------------
# Dining Agent
# ---------------------------------------------------------------------------


class TestDiningAgent:
    @patch("src.agent.agents.dining_agent._get_llm", new_callable=AsyncMock)
    async def test_generates_dining_response(self, mock_get_llm):
        """Dining agent produces a dining-specific AIMessage."""
        from src.agent.agents.dining_agent import dining_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Todd English's Tuscany serves authentic Italian cuisine."
        ))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="What Italian restaurants do you have?")],
            retrieved_context=[
                {"content": "Todd English's Tuscany: Italian fine dining", "metadata": {"category": "restaurants"}, "score": 0.9}
            ],
        )
        result = await dining_agent(state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    async def test_empty_context_returns_dining_fallback(self):
        """Empty context returns dining-specific fallback."""
        from src.agent.agents.dining_agent import dining_agent

        state = _state(
            messages=[HumanMessage(content="Do you have a sushi bar?")],
            retrieved_context=[],
        )
        result = await dining_agent(state)
        assert result["skip_validation"] is True
        assert "dining" in result["messages"][0].content.lower()

    @patch("src.agent.agents.dining_agent._get_circuit_breaker")
    async def test_circuit_breaker_open_returns_fallback(self, mock_get_cb):
        """Dining agent returns fallback when circuit breaker is open."""
        from src.agent.agents.dining_agent import dining_agent

        mock_cb = MagicMock()
        mock_cb.is_open = True
        mock_cb.allow_request = AsyncMock(return_value=False)
        mock_get_cb.return_value = mock_cb

        state = _state(
            messages=[HumanMessage(content="What restaurants?")],
            retrieved_context=[{"content": "data", "metadata": {}, "score": 1.0}],
        )
        result = await dining_agent(state)
        assert result["skip_validation"] is True
        assert "technical difficulties" in result["messages"][0].content

    @patch("src.agent.agents.dining_agent._get_llm", new_callable=AsyncMock)
    async def test_llm_error_returns_fallback(self, mock_get_llm):
        """LLM error in dining agent produces fallback."""
        from src.agent.agents.dining_agent import dining_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ConnectionError("API error"))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="Question")],
            retrieved_context=[
                {"content": "data", "metadata": {"category": "restaurants"}, "score": 1.0}
            ],
        )
        result = await dining_agent(state)
        assert result["skip_validation"] is True
        assert "trouble generating" in result["messages"][0].content.lower()

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
# Entertainment Agent
# ---------------------------------------------------------------------------


class TestEntertainmentAgent:
    @patch("src.agent.agents.entertainment_agent._get_llm", new_callable=AsyncMock)
    async def test_generates_entertainment_response(self, mock_get_llm):
        """Entertainment agent produces an entertainment-specific AIMessage."""
        from src.agent.agents.entertainment_agent import entertainment_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="The arena hosts concerts every weekend with major artists."
        ))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="What shows are coming up?")],
            retrieved_context=[
                {"content": "Arena concerts: weekends", "metadata": {"category": "entertainment"}, "score": 0.9}
            ],
        )
        result = await entertainment_agent(state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    async def test_empty_context_returns_entertainment_fallback(self):
        """Empty context returns entertainment-specific fallback."""
        from src.agent.agents.entertainment_agent import entertainment_agent

        state = _state(
            messages=[HumanMessage(content="Any opera performances?")],
            retrieved_context=[],
        )
        result = await entertainment_agent(state)
        assert result["skip_validation"] is True
        assert "entertainment" in result["messages"][0].content.lower() or "show" in result["messages"][0].content.lower()

    @patch("src.agent.agents.entertainment_agent._get_circuit_breaker")
    async def test_circuit_breaker_open_returns_fallback(self, mock_get_cb):
        """Entertainment agent returns fallback when circuit breaker is open."""
        from src.agent.agents.entertainment_agent import entertainment_agent

        mock_cb = MagicMock()
        mock_cb.is_open = True
        mock_cb.allow_request = AsyncMock(return_value=False)
        mock_get_cb.return_value = mock_cb

        state = _state(
            messages=[HumanMessage(content="Any shows?")],
            retrieved_context=[{"content": "data", "metadata": {}, "score": 1.0}],
        )
        result = await entertainment_agent(state)
        assert result["skip_validation"] is True
        assert "technical difficulties" in result["messages"][0].content

    @patch("src.agent.agents.entertainment_agent._get_llm", new_callable=AsyncMock)
    async def test_llm_error_returns_fallback(self, mock_get_llm):
        """LLM error in entertainment agent produces fallback."""
        from src.agent.agents.entertainment_agent import entertainment_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ConnectionError("API error"))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="Question")],
            retrieved_context=[
                {"content": "data", "metadata": {"category": "entertainment"}, "score": 1.0}
            ],
        )
        result = await entertainment_agent(state)
        assert result["skip_validation"] is True
        assert "trouble generating" in result["messages"][0].content.lower()

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
# Comp Agent
# ---------------------------------------------------------------------------


class TestCompAgent:
    @patch("src.agent.agents.comp_agent._get_llm", new_callable=AsyncMock)
    async def test_generates_comp_response(self, mock_get_llm):
        """Comp agent produces a comp-specific AIMessage."""
        from src.agent.agents.comp_agent import comp_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="Based on available information, you may be eligible for loyalty rewards."
        ))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="What loyalty programs do you have?")],
            retrieved_context=[
                {"content": "Momentum rewards program: 3 tiers", "metadata": {"category": "promotions"}, "score": 0.9}
            ],
            extracted_fields=_high_completeness_fields(),
        )
        result = await comp_agent(state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    async def test_empty_context_returns_comp_fallback(self):
        """Empty context returns comp-specific fallback (after passing profile gate)."""
        from src.agent.agents.comp_agent import comp_agent

        state = _state(
            messages=[HumanMessage(content="What comps can I get?")],
            retrieved_context=[],
            extracted_fields=_high_completeness_fields(),
        )
        result = await comp_agent(state)
        assert result["skip_validation"] is True
        assert "loyalty" in result["messages"][0].content.lower() or "promotions" in result["messages"][0].content.lower()

    @patch("src.agent.agents.comp_agent._get_circuit_breaker")
    async def test_circuit_breaker_open_returns_fallback(self, mock_get_cb):
        """Comp agent returns fallback when circuit breaker is open."""
        from src.agent.agents.comp_agent import comp_agent

        mock_cb = MagicMock()
        mock_cb.is_open = True
        mock_cb.allow_request = AsyncMock(return_value=False)
        mock_get_cb.return_value = mock_cb

        state = _state(
            messages=[HumanMessage(content="What offers?")],
            retrieved_context=[{"content": "data", "metadata": {}, "score": 1.0}],
            extracted_fields=_high_completeness_fields(),
        )
        result = await comp_agent(state)
        assert result["skip_validation"] is True
        assert "technical difficulties" in result["messages"][0].content

    @patch("src.agent.agents.comp_agent._get_llm", new_callable=AsyncMock)
    async def test_llm_error_returns_fallback(self, mock_get_llm):
        """LLM error in comp agent produces fallback."""
        from src.agent.agents.comp_agent import comp_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ConnectionError("API error"))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="Question")],
            retrieved_context=[
                {"content": "data", "metadata": {"category": "promotions"}, "score": 1.0}
            ],
            extracted_fields=_high_completeness_fields(),
        )
        result = await comp_agent(state)
        assert result["skip_validation"] is True
        assert "trouble generating" in result["messages"][0].content.lower()

    async def test_flat_fields_completeness_below_threshold(self):
        """Flat extracted_fields with few filled values triggers deflection."""
        from src.agent.agents.comp_agent import comp_agent

        state = _state(
            messages=[HumanMessage(content="What comps?")],
            retrieved_context=[{"content": "data", "metadata": {}, "score": 0.9}],
            extracted_fields={
                "name": "John",
                "visit_date": None,
                "party_size": None,
                "dining": None,
                "entertainment": None,
                "gaming": None,
                "occasions": None,
                "companions": None,
            },  # 1/8 = 12.5% < 60%
        )
        result = await comp_agent(state)
        assert result["skip_validation"] is True
        assert "rewards and promotions" in result["messages"][0].content

    async def test_flat_fields_completeness_above_threshold(self):
        """Flat extracted_fields with enough values passes gate."""
        from src.agent.agents.comp_agent import comp_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Here are your rewards!"))

        state = _state(
            messages=[HumanMessage(content="What comps?")],
            retrieved_context=[{"content": "Loyalty info", "metadata": {"category": "promotions"}, "score": 0.9}],
            extracted_fields={
                "name": "John",
                "visit_date": "2026-03-01",
                "party_size": 4,
                "dining": "steakhouse",
                "entertainment": "comedy",
                "gaming": None,
                "occasions": None,
                "companions": None,
            },  # 5/8 = 62.5% > 60%
        )
        with patch("src.agent.agents.comp_agent._get_llm", new_callable=AsyncMock, return_value=mock_llm):
            result = await comp_agent(state)
        assert isinstance(result["messages"][0], AIMessage)
        mock_llm.ainvoke.assert_called_once()

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


# ---------------------------------------------------------------------------
# Hotel Agent
# ---------------------------------------------------------------------------


class TestHotelAgent:
    @patch("src.agent.agents.hotel_agent._get_llm", new_callable=AsyncMock)
    async def test_generates_hotel_response(self, mock_get_llm):
        """Hotel agent produces a hotel-specific AIMessage."""
        from src.agent.agents.hotel_agent import hotel_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="The Sky Tower King Suite offers stunning mountain views across 34 floors."
        ))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="What hotel rooms do you have?")],
            retrieved_context=[
                {"content": "Sky Tower: luxury tower, 34 floors, mountain views", "metadata": {"category": "hotel"}, "score": 0.9}
            ],
        )
        result = await hotel_agent(state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    async def test_empty_context_returns_hotel_fallback(self):
        """Empty context returns hotel-specific fallback."""
        from src.agent.agents.hotel_agent import hotel_agent

        state = _state(
            messages=[HumanMessage(content="Do you have a presidential suite?")],
            retrieved_context=[],
        )
        result = await hotel_agent(state)
        assert result["skip_validation"] is True
        assert "hotel" in result["messages"][0].content.lower()

    @patch("src.agent.agents.hotel_agent._get_circuit_breaker")
    async def test_circuit_breaker_open_returns_fallback(self, mock_get_cb):
        """Hotel agent returns fallback when circuit breaker is open."""
        from src.agent.agents.hotel_agent import hotel_agent

        mock_cb = MagicMock()
        mock_cb.is_open = True
        mock_cb.allow_request = AsyncMock(return_value=False)
        mock_get_cb.return_value = mock_cb

        state = _state(
            messages=[HumanMessage(content="What rooms are available?")],
            retrieved_context=[{"content": "data", "metadata": {}, "score": 1.0}],
        )
        result = await hotel_agent(state)
        assert result["skip_validation"] is True
        assert "technical difficulties" in result["messages"][0].content

    @patch("src.agent.agents.hotel_agent._get_llm", new_callable=AsyncMock)
    async def test_llm_error_returns_fallback(self, mock_get_llm):
        """LLM error in hotel agent produces fallback."""
        from src.agent.agents.hotel_agent import hotel_agent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ConnectionError("API error"))
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="Question")],
            retrieved_context=[
                {"content": "data", "metadata": {"category": "hotel"}, "score": 1.0}
            ],
        )
        result = await hotel_agent(state)
        assert result["skip_validation"] is True
        assert "trouble generating" in result["messages"][0].content.lower()

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
# Registry
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
# Package Import
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
# Parametrized Specialist Contract Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent_fn,agent_module", [
    ("host_agent", "src.agent.agents.host_agent"),
    ("dining_agent", "src.agent.agents.dining_agent"),
    ("entertainment_agent", "src.agent.agents.entertainment_agent"),
    ("comp_agent", "src.agent.agents.comp_agent"),
    ("hotel_agent", "src.agent.agents.hotel_agent"),
])
class TestSpecialistContract:
    """Verify all 5 specialist agents follow the same contract via _base.py."""

    @pytest.mark.asyncio
    async def test_returns_messages_key(self, agent_fn, agent_module):
        """All specialists must return dict with 'messages' key."""
        import importlib

        mod = importlib.import_module(agent_module)
        fn = getattr(mod, agent_fn)

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Test response from agent.")
        )

        # Comp agent needs high profile completeness to pass its gate
        extra = {}
        if agent_fn == "comp_agent":
            extra["extracted_fields"] = _high_completeness_fields()

        state = _state(
            messages=[HumanMessage(content="Tell me about the property")],
            retrieved_context=[
                {"content": "Property info here", "metadata": {"category": "property"}, "score": 0.9}
            ],
            **extra,
        )

        with patch(f"{agent_module}._get_llm", new_callable=AsyncMock, return_value=mock_llm):
            result = await fn(state)

        assert "messages" in result
        assert len(result["messages"]) >= 1
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_cb_open_returns_fallback(self, agent_fn, agent_module):
        """All specialists return fallback when circuit breaker is open."""
        import importlib

        mod = importlib.import_module(agent_module)
        fn = getattr(mod, agent_fn)

        mock_cb = MagicMock()
        mock_cb.is_open = True
        mock_cb.allow_request = AsyncMock(return_value=False)

        # Comp agent profile gate runs BEFORE CB check in execute_specialist,
        # so we need high completeness to reach the CB path
        extra = {}
        if agent_fn == "comp_agent":
            extra["extracted_fields"] = _high_completeness_fields()

        state = _state(
            messages=[HumanMessage(content="What offers?")],
            retrieved_context=[{"content": "data", "metadata": {}, "score": 1.0}],
            **extra,
        )

        with patch(f"{agent_module}._get_circuit_breaker", return_value=mock_cb):
            result = await fn(state)

        assert result.get("skip_validation") is True
        assert "technical difficulties" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_empty_context_returns_fallback(self, agent_fn, agent_module):
        """All specialists return domain-specific fallback on empty context."""
        import importlib

        mod = importlib.import_module(agent_module)
        fn = getattr(mod, agent_fn)

        extra = {}
        if agent_fn == "comp_agent":
            extra["extracted_fields"] = _high_completeness_fields()

        state = _state(
            messages=[HumanMessage(content="Tell me about the moon")],
            retrieved_context=[],
            **extra,
        )
        result = await fn(state)
        assert result.get("skip_validation") is True
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_llm_value_error_returns_retry_fallback(self, agent_fn, agent_module):
        """All specialists return retry fallback on LLM ValueError (parse failure)."""
        import importlib

        mod = importlib.import_module(agent_module)
        fn = getattr(mod, agent_fn)

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ValueError("bad structured output"))

        extra = {}
        if agent_fn == "comp_agent":
            extra["extracted_fields"] = _high_completeness_fields()

        state = _state(
            messages=[HumanMessage(content="Question about property")],
            retrieved_context=[
                {"content": "Some data", "metadata": {"category": "general"}, "score": 0.9}
            ],
            **extra,
        )

        with patch(f"{agent_module}._get_llm", new_callable=AsyncMock, return_value=mock_llm):
            result = await fn(state)

        assert result.get("skip_validation") is False
        assert result.get("retry_count") == 1
        assert "trouble processing" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_llm_network_error_returns_skip_fallback(self, agent_fn, agent_module):
        """All specialists return skip_validation fallback on network error."""
        import importlib

        mod = importlib.import_module(agent_module)
        fn = getattr(mod, agent_fn)

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        extra = {}
        if agent_fn == "comp_agent":
            extra["extracted_fields"] = _high_completeness_fields()

        state = _state(
            messages=[HumanMessage(content="Question about property")],
            retrieved_context=[
                {"content": "Some data", "metadata": {"category": "general"}, "score": 0.9}
            ],
            **extra,
        )

        with patch(f"{agent_module}._get_llm", new_callable=AsyncMock, return_value=mock_llm):
            result = await fn(state)

        assert result.get("skip_validation") is True
        assert "trouble generating" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_llm_timeout_returns_skip_fallback(self, agent_fn, agent_module):
        """All specialists return skip_validation fallback on asyncio.TimeoutError."""
        import importlib

        mod = importlib.import_module(agent_module)
        fn = getattr(mod, agent_fn)

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())

        extra = {}
        if agent_fn == "comp_agent":
            extra["extracted_fields"] = _high_completeness_fields()

        state = _state(
            messages=[HumanMessage(content="Question about property")],
            retrieved_context=[
                {"content": "Some data", "metadata": {"category": "general"}, "score": 0.9}
            ],
            **extra,
        )

        with patch(f"{agent_module}._get_llm", new_callable=AsyncMock, return_value=mock_llm):
            result = await fn(state)

        assert result.get("skip_validation") is True
        assert "trouble generating" in result["messages"][0].content.lower()
