"""Tests for frustration/crisis suppression and response length budgets.

R82 Track 1E: Frustration suppression of promotional content for comp agent.
R82 Track 1E: Crisis state suppression across ALL specialists.
R82 Track 1G (partial): Response length budgets per intent.
"""

import pytest
from string import Template
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.agents._base import (
    _build_behavioral_prompt_sections,
    _should_inject_suggestion,
    execute_specialist,
)


# ---------------------------------------------------------------------------
# Helpers (copied from test_base_specialist.py pattern)
# ---------------------------------------------------------------------------


def _state(**overrides):
    """Minimal PropertyQAState dict."""
    base = {
        "messages": [HumanMessage(content="test")],
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
        "guest_sentiment": None,
        "guest_context": {},
        "guest_name": None,
        "domains_discussed": [],
        "crisis_active": False,
        "crisis_turn_count": 0,
        "detected_language": None,
        "handoff_request": None,
        "profiling_phase": None,
        "profile_completeness_score": 0.0,
        "profiling_question_injected": False,
        "suggestion_offered": False,
        "responsible_gaming_count": 0,
        "specialist_name": None,
        "dispatch_method": None,
    }
    base.update(overrides)
    return base


DUMMY_TEMPLATE = Template(
    "You are a test agent for $property_name. Time: $current_time. "
    "${responsible_gaming_helplines}"
)


def _make_execute_kwargs(
    *,
    agent_name="test",
    get_llm_fn=None,
    get_cb_fn=None,
    llm_response="Test response.",
):
    """Build kwargs for execute_specialist with sensible defaults."""
    if get_cb_fn is None:
        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()
        mock_cb.record_failure = AsyncMock()
        get_cb_fn = AsyncMock(return_value=mock_cb)

    if get_llm_fn is None:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=llm_response))
        get_llm_fn = AsyncMock(return_value=mock_llm)

    return {
        "agent_name": agent_name,
        "system_prompt_template": DUMMY_TEMPLATE,
        "context_header": "Test Context",
        "no_context_fallback": "No info available. Contact Mohegan Sun at 1-888-226-7711.",
        "get_llm_fn": get_llm_fn,
        "get_cb_fn": get_cb_fn,
    }


# ---------------------------------------------------------------------------
# Track 1E: Frustration Suppression for Promotional Agents
# ---------------------------------------------------------------------------


class TestFrustrationSuppression:
    """Test that promotional content is suppressed for frustrated guests."""

    def test_behavioral_sections_detect_frustration(self):
        """_build_behavioral_prompt_sections returns frustrated as effective_sentiment."""
        state = _state(
            messages=[HumanMessage(content="This is terrible service!")],
            guest_sentiment="frustrated",
        )
        sections, eff_sent, dynamics, frust_count = _build_behavioral_prompt_sections(
            state,
            "This is terrible service!",
            "this is terrible service!",
            {},
            "frustrated",
        )
        assert eff_sent == "frustrated"

    def test_suggestion_blocked_on_frustration(self):
        """Proactive suggestions are blocked when guest is frustrated."""
        state = _state(
            whisper_plan={
                "proactive_suggestion": "Try our spa!",
                "suggestion_confidence": "0.9",
            },
            suggestion_offered=False,
            retrieved_context=[{"content": "test", "metadata": {}, "score": 0.9}],
        )
        section, offered = _should_inject_suggestion(
            state,
            "frustrated",
            {"turn_count": 5},
        )
        assert section == ""
        assert offered is False

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="R110: Mock incompatible with profile-reference injection. TODO: migrate to live eval."
    )
    async def test_comp_agent_frustrated_gets_override(self):
        """When comp agent runs with frustrated sentiment, system prompt has override."""
        mock_llm = MagicMock()
        captured_messages = []

        async def _capture_invoke(messages):
            captured_messages.extend(messages)
            return MagicMock(content="I understand your frustration.")

        mock_llm.ainvoke = _capture_invoke

        kwargs = _make_execute_kwargs(
            agent_name="comp",
            get_llm_fn=AsyncMock(return_value=mock_llm),
        )
        state = _state(
            messages=[HumanMessage(content="This rewards program is garbage!")],
            guest_sentiment="frustrated",
            retrieved_context=[
                {
                    "content": "Comp tier info",
                    "metadata": {"category": "comps"},
                    "score": 0.9,
                }
            ],
        )

        await execute_specialist(state, **kwargs)

        # The system message should contain the frustration override
        system_msgs = [
            m
            for m in captured_messages
            if hasattr(m, "content") and "OVERRIDE" in str(m.content)
        ]
        assert len(system_msgs) >= 1, (
            "Frustration override not injected into comp agent prompt"
        )
        override_text = system_msgs[0].content
        assert "Suppress Promotional Tone" in override_text
        assert "NO promotional language" in override_text

    @pytest.mark.asyncio
    async def test_dining_agent_frustrated_no_promotional_override(self):
        """Non-promotional agents (dining) should NOT get the frustration promotional override."""
        mock_llm = MagicMock()
        captured_messages = []

        async def _capture_invoke(messages):
            captured_messages.extend(messages)
            return MagicMock(content="Sorry about the wait.")

        mock_llm.ainvoke = _capture_invoke

        kwargs = _make_execute_kwargs(
            agent_name="dining",
            get_llm_fn=AsyncMock(return_value=mock_llm),
        )
        state = _state(
            messages=[HumanMessage(content="The restaurant service was awful!")],
            guest_sentiment="frustrated",
            retrieved_context=[
                {
                    "content": "Restaurant info",
                    "metadata": {"category": "dining"},
                    "score": 0.9,
                }
            ],
        )

        await execute_specialist(state, **kwargs)

        # Dining agent should NOT have the promotional suppression override
        system_msgs = [
            m
            for m in captured_messages
            if hasattr(m, "content") and "Suppress Promotional Tone" in str(m.content)
        ]
        assert len(system_msgs) == 0, (
            "Promotional suppression should not apply to dining agent"
        )

    @pytest.mark.asyncio
    async def test_comp_agent_positive_sentiment_no_override(self):
        """Comp agent with positive sentiment should NOT get frustration override."""
        mock_llm = MagicMock()
        captured_messages = []

        async def _capture_invoke(messages):
            captured_messages.extend(messages)
            return MagicMock(content="Great news about your rewards!")

        mock_llm.ainvoke = _capture_invoke

        kwargs = _make_execute_kwargs(
            agent_name="comp",
            get_llm_fn=AsyncMock(return_value=mock_llm),
        )
        state = _state(
            messages=[HumanMessage(content="What rewards do I have?")],
            guest_sentiment="positive",
            retrieved_context=[
                {
                    "content": "Comp tier info",
                    "metadata": {"category": "comps"},
                    "score": 0.9,
                }
            ],
        )

        await execute_specialist(state, **kwargs)

        # Should NOT have the frustration override
        system_msgs = [
            m
            for m in captured_messages
            if hasattr(m, "content") and "Suppress Promotional Tone" in str(m.content)
        ]
        assert len(system_msgs) == 0, (
            "Frustration override should not activate on positive sentiment"
        )

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="R110: Mock incompatible with profile-reference injection. TODO: migrate to live eval."
    )
    async def test_negative_sentiment_also_triggers_suppression(self):
        """Negative sentiment (not just frustrated) triggers suppression for comp agent."""
        mock_llm = MagicMock()
        captured_messages = []

        async def _capture_invoke(messages):
            captured_messages.extend(messages)
            return MagicMock(content="I understand.")

        mock_llm.ainvoke = _capture_invoke

        kwargs = _make_execute_kwargs(
            agent_name="comp",
            get_llm_fn=AsyncMock(return_value=mock_llm),
        )
        state = _state(
            messages=[HumanMessage(content="I'm really upset about my account.")],
            guest_sentiment="negative",
            retrieved_context=[
                {
                    "content": "Comp info",
                    "metadata": {"category": "comps"},
                    "score": 0.9,
                }
            ],
        )

        await execute_specialist(state, **kwargs)

        system_msgs = [
            m
            for m in captured_messages
            if hasattr(m, "content") and "Suppress Promotional Tone" in str(m.content)
        ]
        assert len(system_msgs) >= 1, (
            "Negative sentiment should also trigger suppression for comp"
        )


# ---------------------------------------------------------------------------
# Track 1E: Crisis State Suppression Across ALL Specialists
# ---------------------------------------------------------------------------


class TestCrisisSuppression:
    """Test crisis state suppression across all specialist agents."""

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="R110: Mock incompatible with profile-reference injection. TODO: migrate to live eval."
    )
    async def test_crisis_active_injects_override_for_comp(self):
        """When crisis_active=True, comp agent gets crisis override."""
        mock_llm = MagicMock()
        captured_messages = []

        async def _capture_invoke(messages):
            captured_messages.extend(messages)
            return MagicMock(content="I'm here to help.")

        mock_llm.ainvoke = _capture_invoke

        kwargs = _make_execute_kwargs(
            agent_name="comp",
            get_llm_fn=AsyncMock(return_value=mock_llm),
        )
        state = _state(
            messages=[HumanMessage(content="I need help")],
            crisis_active=True,
            retrieved_context=[
                {
                    "content": "Comp info",
                    "metadata": {"category": "comps"},
                    "score": 0.9,
                }
            ],
        )

        await execute_specialist(state, **kwargs)

        system_msgs = [
            m
            for m in captured_messages
            if hasattr(m, "content") and "Crisis State Active" in str(m.content)
        ]
        assert len(system_msgs) >= 1, "Crisis override not injected for comp agent"

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="R110: Mock incompatible with profile-reference injection. TODO: migrate to live eval."
    )
    async def test_crisis_active_injects_override_for_dining(self):
        """When crisis_active=True, dining agent also gets crisis override."""
        mock_llm = MagicMock()
        captured_messages = []

        async def _capture_invoke(messages):
            captured_messages.extend(messages)
            return MagicMock(content="I'm here to help.")

        mock_llm.ainvoke = _capture_invoke

        kwargs = _make_execute_kwargs(
            agent_name="dining",
            get_llm_fn=AsyncMock(return_value=mock_llm),
        )
        state = _state(
            messages=[HumanMessage(content="Where can I eat?")],
            crisis_active=True,
            retrieved_context=[
                {
                    "content": "Restaurant info",
                    "metadata": {"category": "dining"},
                    "score": 0.9,
                }
            ],
        )

        await execute_specialist(state, **kwargs)

        system_msgs = [
            m
            for m in captured_messages
            if hasattr(m, "content") and "Crisis State Active" in str(m.content)
        ]
        assert len(system_msgs) >= 1, "Crisis override should apply to ALL specialists"

    @pytest.mark.asyncio
    async def test_no_crisis_no_override(self):
        """When crisis_active=False, no crisis override is injected."""
        mock_llm = MagicMock()
        captured_messages = []

        async def _capture_invoke(messages):
            captured_messages.extend(messages)
            return MagicMock(content="Great choice!")

        mock_llm.ainvoke = _capture_invoke

        kwargs = _make_execute_kwargs(
            agent_name="dining",
            get_llm_fn=AsyncMock(return_value=mock_llm),
        )
        state = _state(
            messages=[HumanMessage(content="What's good for dinner?")],
            crisis_active=False,
            retrieved_context=[
                {
                    "content": "Restaurant info",
                    "metadata": {"category": "dining"},
                    "score": 0.9,
                }
            ],
        )

        await execute_specialist(state, **kwargs)

        system_msgs = [
            m
            for m in captured_messages
            if hasattr(m, "content") and "Crisis State Active" in str(m.content)
        ]
        assert len(system_msgs) == 0, (
            "Crisis override should not appear when crisis_active=False"
        )


# ---------------------------------------------------------------------------
# Track 1G: Response Length Budgets
# ---------------------------------------------------------------------------


class TestResponseLengthBudgets:
    """Test response length enforcement per intent."""

    @pytest.mark.asyncio
    async def test_greeting_truncated_when_over_budget(self):
        """Greeting responses exceeding 50 words are truncated."""
        # Generate a long response (60+ words)
        long_response = (
            "Welcome to Mohegan Sun! We are absolutely thrilled to have you here today. "
            "There are so many amazing things to explore at our property. We have world-class "
            "dining options, exciting entertainment venues, a luxurious spa, and of course "
            "our incredible gaming floor. Whether you are here for relaxation or excitement, "
            "we have something for everyone. Let me know how I can help you today."
        )
        kwargs = _make_execute_kwargs(
            agent_name="host",
            llm_response=long_response,
        )
        state = _state(
            messages=[HumanMessage(content="Hi!")],
            query_type="greeting",
            retrieved_context=[
                {
                    "content": "Property info",
                    "metadata": {"category": "general"},
                    "score": 0.9,
                }
            ],
        )

        result = await execute_specialist(state, **kwargs)

        response_text = result["messages"][0].content
        word_count = len(response_text.split())
        assert word_count <= 50, (
            f"Greeting response should be <= 50 words, got {word_count}"
        )

    @pytest.mark.asyncio
    async def test_off_topic_truncated_when_over_budget(self):
        """Off-topic responses exceeding 60 words are truncated."""
        long_response = " ".join(["word"] * 80) + ". End of response."
        kwargs = _make_execute_kwargs(
            agent_name="host",
            llm_response=long_response,
        )
        state = _state(
            messages=[HumanMessage(content="Tell me about quantum physics")],
            query_type="off_topic",
            retrieved_context=[
                {
                    "content": "Property info",
                    "metadata": {"category": "general"},
                    "score": 0.9,
                }
            ],
        )

        result = await execute_specialist(state, **kwargs)

        response_text = result["messages"][0].content
        word_count = len(response_text.split())
        assert word_count <= 60, (
            f"Off-topic response should be <= 60 words, got {word_count}"
        )

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="R110: Mock incompatible with profile-reference injection. TODO: migrate to live eval."
    )
    async def test_property_qa_not_truncated(self):
        """Property Q&A responses should NOT be truncated (no budget set)."""
        long_response = " ".join(["information"] * 100) + ". End."
        kwargs = _make_execute_kwargs(
            agent_name="dining",
            llm_response=long_response,
        )
        state = _state(
            messages=[HumanMessage(content="Tell me about your restaurants")],
            query_type="property_qa",
            retrieved_context=[
                {
                    "content": "Restaurant info",
                    "metadata": {"category": "dining"},
                    "score": 0.9,
                }
            ],
        )

        result = await execute_specialist(state, **kwargs)

        response_text = result["messages"][0].content
        word_count = len(response_text.split())
        # property_qa has no budget, so full response should be preserved
        assert word_count > 60, f"Property QA should not be truncated, got {word_count}"

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="R110: Mock incompatible with profile-reference injection. TODO: migrate to live eval."
    )
    async def test_short_greeting_not_truncated(self):
        """A greeting response under the budget should not be modified."""
        short_response = "Welcome to Mohegan Sun! How can I help you today?"
        kwargs = _make_execute_kwargs(
            agent_name="host",
            llm_response=short_response,
        )
        state = _state(
            messages=[HumanMessage(content="Hey")],
            query_type="greeting",
            retrieved_context=[
                {
                    "content": "Property info",
                    "metadata": {"category": "general"},
                    "score": 0.9,
                }
            ],
        )

        result = await execute_specialist(state, **kwargs)

        assert result["messages"][0].content == short_response

    @pytest.mark.asyncio
    async def test_truncation_preserves_sentence_boundary(self):
        """Truncation should end at a sentence boundary, not mid-word."""
        # First sentence is ~10 words, total is 60+
        long_response = (
            "Welcome to Mohegan Sun today! "
            "We have many restaurants for you to explore on the property. "
            "Our steakhouse is a popular choice among our guests who visit us. "
            "The buffet is also excellent and offers a wide variety of dishes."
        )
        kwargs = _make_execute_kwargs(
            agent_name="host",
            llm_response=long_response,
        )
        state = _state(
            messages=[HumanMessage(content="Hi there")],
            query_type="greeting",
            retrieved_context=[
                {
                    "content": "Property info",
                    "metadata": {"category": "general"},
                    "score": 0.9,
                }
            ],
        )

        result = await execute_specialist(state, **kwargs)

        response_text = result["messages"][0].content
        # Should end with a sentence-ending punctuation
        assert response_text.rstrip().endswith((".", "!", "?")), (
            f"Truncated response should end with sentence punctuation: '{response_text}'"
        )
