"""Tests for the shared specialist execution logic (_base.py)."""

import asyncio

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock
from string import Template

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


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
    }
    base.update(overrides)
    return base


DUMMY_TEMPLATE = Template(
    "You are a test agent for $property_name. Time: $current_time. "
    "${responsible_gaming_helplines}"
)


def _make_execute_kwargs(
    *,
    get_llm_fn=None,
    get_cb_fn=None,
    include_whisper=False,
):
    """Build kwargs for execute_specialist with sensible defaults."""
    if get_cb_fn is None:
        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.record_success = AsyncMock()
        mock_cb.record_failure = AsyncMock()
        get_cb_fn = MagicMock(return_value=mock_cb)

    if get_llm_fn is None:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Test response.")
        )
        get_llm_fn = MagicMock(return_value=mock_llm)

    return {
        "agent_name": "test",
        "system_prompt_template": DUMMY_TEMPLATE,
        "context_header": "Test Context",
        "no_context_fallback": "No info available. Contact Mohegan Sun at 1-888-226-7711.",
        "get_llm_fn": get_llm_fn,
        "get_cb_fn": get_cb_fn,
        "include_whisper": include_whisper,
    }


class TestCircuitBreakerOpen:
    """CB open returns fallback with skip_validation=True."""

    @pytest.mark.asyncio
    async def test_cb_open_returns_fallback(self):
        """When circuit breaker is open, return technical difficulties fallback."""
        from src.agent.agents._base import execute_specialist

        mock_cb = MagicMock()
        mock_cb.is_open = True
        get_cb_fn = MagicMock(return_value=mock_cb)

        kwargs = _make_execute_kwargs(get_cb_fn=get_cb_fn)
        state = _state(
            messages=[HumanMessage(content="What restaurants?")],
            retrieved_context=[{"content": "data", "metadata": {}, "score": 1.0}],
        )

        result = await execute_specialist(state, **kwargs)

        assert result["skip_validation"] is True
        assert "technical difficulties" in result["messages"][0].content
        assert isinstance(result["messages"][0], AIMessage)


class TestNoRetrievedContext:
    """Empty retrieved context returns no_context_fallback with skip_validation=True."""

    @pytest.mark.asyncio
    async def test_no_context_returns_fallback(self):
        """Empty retrieved_context returns the no_context_fallback message."""
        from src.agent.agents._base import execute_specialist

        kwargs = _make_execute_kwargs()
        state = _state(
            messages=[HumanMessage(content="Tell me about the moon")],
            retrieved_context=[],
        )

        result = await execute_specialist(state, **kwargs)

        assert result["skip_validation"] is True
        assert "No info available" in result["messages"][0].content


class TestHappyPath:
    """Happy path: LLM responds, no skip_validation."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_llm_response(self):
        """With context and working LLM, returns AIMessage without skip_validation."""
        from src.agent.agents._base import execute_specialist

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="The steakhouse opens at 5 PM.")
        )
        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.record_success = AsyncMock()

        kwargs = _make_execute_kwargs(
            get_llm_fn=MagicMock(return_value=mock_llm),
            get_cb_fn=MagicMock(return_value=mock_cb),
        )
        state = _state(
            messages=[HumanMessage(content="When does the steakhouse open?")],
            retrieved_context=[
                {"content": "Steakhouse hours: 5-10 PM", "metadata": {"category": "restaurants"}, "score": 0.9}
            ],
        )

        result = await execute_specialist(state, **kwargs)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "steakhouse" in result["messages"][0].content.lower()
        assert "skip_validation" not in result


class TestRetryFeedbackInjection:
    """On retry (retry_count>0, retry_feedback set), feedback is injected as SystemMessage."""

    @pytest.mark.asyncio
    async def test_retry_injects_feedback(self):
        """Retry feedback is injected as a SystemMessage before conversation history."""
        from src.agent.agents._base import execute_specialist

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Corrected response.")
        )
        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.record_success = AsyncMock()

        kwargs = _make_execute_kwargs(
            get_llm_fn=MagicMock(return_value=mock_llm),
            get_cb_fn=MagicMock(return_value=mock_cb),
        )
        state = _state(
            messages=[HumanMessage(content="What restaurants?")],
            retrieved_context=[
                {"content": "Steakhouse info", "metadata": {"category": "restaurants"}, "score": 0.9}
            ],
            retry_count=1,
            retry_feedback="Response was not grounded in context",
        )

        await execute_specialist(state, **kwargs)

        # Verify feedback was injected by inspecting LLM call args
        call_args = mock_llm.ainvoke.call_args[0][0]
        feedback_msgs = [
            m for m in call_args
            if isinstance(m, SystemMessage) and "failed validation" in m.content
        ]
        assert len(feedback_msgs) == 1
        assert "not grounded" in feedback_msgs[0].content


class TestWhisperPlanInjection:
    """include_whisper=True with whisper_plan in state injects guidance."""

    @pytest.mark.asyncio
    async def test_whisper_plan_injected(self):
        """Whisper plan guidance is appended to the system prompt when include_whisper=True."""
        from src.agent.agents._base import execute_specialist

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Response with whisper guidance.")
        )
        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.record_success = AsyncMock()

        kwargs = _make_execute_kwargs(
            get_llm_fn=MagicMock(return_value=mock_llm),
            get_cb_fn=MagicMock(return_value=mock_cb),
            include_whisper=True,
        )
        state = _state(
            messages=[HumanMessage(content="What restaurants?")],
            retrieved_context=[
                {"content": "Steakhouse info", "metadata": {"category": "restaurants"}, "score": 0.9}
            ],
            whisper_plan={
                "next_topic": "dining",
                "extraction_targets": ["dietary_restrictions"],
                "offer_readiness": 0.3,
                "conversation_note": "Guest seems interested in dining",
            },
        )

        await execute_specialist(state, **kwargs)

        # Verify whisper guidance was injected into system prompt
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msg = call_args[0]
        assert isinstance(system_msg, SystemMessage)
        assert "Whisper Track Guidance" in system_msg.content
        assert "dining" in system_msg.content


class TestValueTypeErrorFallback:
    """ValueError/TypeError from LLM returns fallback with validation enabled.

    Unlike circuit-breaker-open or network-error paths, parsing errors
    still go through validation (skip_validation=False) with retry_count=1
    so that the validator runs but does not trigger a second generate.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc_cls", [ValueError, TypeError])
    async def test_value_type_error_returns_fallback(self, exc_cls):
        """ValueError and TypeError from LLM produce fallback without skip_validation."""
        from src.agent.agents._base import execute_specialist

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=exc_cls("bad response"))
        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.record_failure = AsyncMock()

        kwargs = _make_execute_kwargs(
            get_llm_fn=MagicMock(return_value=mock_llm),
            get_cb_fn=MagicMock(return_value=mock_cb),
        )
        state = _state(
            messages=[HumanMessage(content="Question")],
            retrieved_context=[
                {"content": "data", "metadata": {"category": "faq"}, "score": 1.0}
            ],
        )

        result = await execute_specialist(state, **kwargs)

        assert result["skip_validation"] is False
        assert result["retry_count"] == 1
        assert "trouble processing" in result["messages"][0].content.lower()
        mock_cb.record_failure.assert_awaited_once()


class TestCancelledErrorReRaised:
    """CancelledError from LLM is re-raised, not swallowed."""

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """asyncio.CancelledError must propagate, not be caught as fallback."""
        from src.agent.agents._base import execute_specialist

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=asyncio.CancelledError())
        mock_cb = MagicMock()
        mock_cb.is_open = False

        kwargs = _make_execute_kwargs(
            get_llm_fn=MagicMock(return_value=mock_llm),
            get_cb_fn=MagicMock(return_value=mock_cb),
        )
        state = _state(
            messages=[HumanMessage(content="Question")],
            retrieved_context=[
                {"content": "data", "metadata": {"category": "faq"}, "score": 1.0}
            ],
        )

        with pytest.raises(asyncio.CancelledError):
            await execute_specialist(state, **kwargs)


class TestHttpErrorFallback:
    """httpx.HTTPError from LLM returns fallback with skip_validation=True."""

    @pytest.mark.asyncio
    async def test_http_error_returns_fallback(self):
        """Network/HTTP errors from LLM produce fallback with skip_validation."""
        from src.agent.agents._base import execute_specialist

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            side_effect=httpx.HTTPError("Connection refused")
        )
        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.record_failure = AsyncMock()

        kwargs = _make_execute_kwargs(
            get_llm_fn=MagicMock(return_value=mock_llm),
            get_cb_fn=MagicMock(return_value=mock_cb),
        )
        state = _state(
            messages=[HumanMessage(content="Question")],
            retrieved_context=[
                {"content": "data", "metadata": {"category": "faq"}, "score": 1.0}
            ],
        )

        result = await execute_specialist(state, **kwargs)

        assert result["skip_validation"] is True
        assert "trouble generating" in result["messages"][0].content.lower()
        mock_cb.record_failure.assert_awaited_once()
