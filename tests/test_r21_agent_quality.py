"""R21 agent quality tests: frustration escalation, proactive suggestions, persona drift.

Tests the three Phase 4 R21 features:
1. Frustration escalation: consecutive negative sentiment triggers soft escalation
2. Proactive suggestions: WhisperPlan suggestion fields + injection logic
3. Persona drift prevention: system prompt re-injection after threshold
"""

import asyncio
from string import Template
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.agents._base import (
    _PERSONA_REINJECT_THRESHOLD,
    _count_consecutive_frustrated,
    execute_specialist,
)
from src.agent.whisper_planner import WhisperPlan, format_whisper_plan


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _disable_features(monkeypatch):
    """Disable features that require external services."""
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    try:
        from src.config import get_settings
        get_settings.cache_clear()
    except (ImportError, AttributeError):
        pass
    yield


@pytest.fixture()
def base_state():
    """Build a minimal PropertyQAState for testing."""
    return {
        "messages": [HumanMessage(content="What restaurants do you have?")],
        "query_type": "property_qa",
        "router_confidence": 0.9,
        "retrieved_context": [
            {
                "content": "Todd English's Tuscany offers Italian cuisine",
                "metadata": {"category": "dining"},
                "score": 0.85,
            }
        ],
        "validation_result": None,
        "retry_count": 0,
        "skip_validation": False,
        "retry_feedback": None,
        "current_time": "Saturday, February 22, 2026 10:00 AM UTC",
        "sources_used": [],
        "extracted_fields": {},
        "whisper_plan": None,
        "responsible_gaming_count": 0,
        "guest_sentiment": None,
        "guest_context": {},
        "guest_name": None,
        "suggestion_offered": 0,
    }


# ---------------------------------------------------------------------------
# 1. Frustration Escalation Tests
# ---------------------------------------------------------------------------


class TestFrustrationEscalation:
    """Tests for consecutive negative sentiment detection and escalation."""

    def test_no_frustrated_messages(self):
        """Zero frustrated messages returns 0."""
        messages = [
            HumanMessage(content="Hello!"),
            AIMessage(content="Welcome!"),
            HumanMessage(content="What restaurants do you have?"),
        ]
        assert _count_consecutive_frustrated(messages) == 0

    def test_single_frustrated_message(self):
        """One frustrated message at the end returns 1."""
        messages = [
            HumanMessage(content="Hello!"),
            AIMessage(content="Welcome!"),
            HumanMessage(content="This is ridiculous, I can't find anything!"),
        ]
        assert _count_consecutive_frustrated(messages) == 1

    def test_two_consecutive_frustrated(self):
        """Two consecutive frustrated messages returns 2."""
        messages = [
            HumanMessage(content="Hello!"),
            AIMessage(content="Welcome!"),
            HumanMessage(content="This is terrible service"),
            AIMessage(content="I'm sorry to hear that"),
            HumanMessage(content="I'm frustrated with this whole experience"),
        ]
        assert _count_consecutive_frustrated(messages) == 2

    def test_frustrated_then_positive_resets(self):
        """Positive message after frustrated resets count."""
        messages = [
            HumanMessage(content="This is ridiculous"),
            AIMessage(content="I apologize"),
            HumanMessage(content="Thanks, that helps!"),
        ]
        assert _count_consecutive_frustrated(messages) == 0

    def test_three_consecutive_frustrated(self):
        """Three consecutive frustrated messages returns 3."""
        messages = [
            HumanMessage(content="This is unacceptable"),
            AIMessage(content="I understand"),
            HumanMessage(content="I can't believe this"),
            AIMessage(content="Let me help"),
            HumanMessage(content="What a disaster"),
        ]
        assert _count_consecutive_frustrated(messages) == 3

    def test_empty_messages(self):
        """Empty message list returns 0."""
        assert _count_consecutive_frustrated([]) == 0

    def test_only_ai_messages(self):
        """Only AI messages returns 0."""
        messages = [
            AIMessage(content="Welcome!"),
            AIMessage(content="How can I help?"),
        ]
        assert _count_consecutive_frustrated(messages) == 0

    def test_sarcasm_detected_as_frustrated(self):
        """Sarcastic messages should be detected as frustrated."""
        messages = [
            HumanMessage(content="Oh great, another thing that doesn't work"),
            AIMessage(content="I apologize"),
            HumanMessage(content="Thanks for nothing"),
        ]
        assert _count_consecutive_frustrated(messages) == 2

    @pytest.mark.asyncio()
    async def test_escalation_injected_in_system_prompt(self, base_state, _disable_features):
        """When frustrated_count >= 2 and sentiment is frustrated, escalation is injected."""
        # Build state with frustrated history
        base_state["messages"] = [
            HumanMessage(content="This is terrible"),
            AIMessage(content="I apologize"),
            HumanMessage(content="I can't believe this service"),
            AIMessage(content="Let me help"),
            HumanMessage(content="What a ridiculous experience"),
        ]
        base_state["guest_sentiment"] = "frustrated"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="I'm so sorry"))

        mock_cb = AsyncMock()
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()

        with patch("src.agent.agents._base.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                PROPERTY_NAME="Mohegan Sun",
                PROPERTY_PHONE="1-888-226-7711",
                MAX_HISTORY_MESSAGES=20,
            )

            result = await execute_specialist(
                base_state,
                agent_name="host",
                system_prompt_template=Template("You are a concierge for $property_name."),
                context_header="Context",
                no_context_fallback="No info available.",
                get_llm_fn=AsyncMock(return_value=mock_llm),
                get_cb_fn=AsyncMock(return_value=mock_cb),
            )

        # Verify escalation was injected by checking the system prompt
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
        system_text = " ".join(m.content for m in system_msgs)
        assert "Escalation Guidance" in system_text
        assert "human host" in system_text.lower()

    @pytest.mark.asyncio()
    async def test_no_escalation_when_not_frustrated(self, base_state, _disable_features):
        """When sentiment is positive, no escalation is injected."""
        base_state["guest_sentiment"] = "positive"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Great choice!"))

        mock_cb = AsyncMock()
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()

        with patch("src.agent.agents._base.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                PROPERTY_NAME="Mohegan Sun",
                PROPERTY_PHONE="1-888-226-7711",
                MAX_HISTORY_MESSAGES=20,
            )

            await execute_specialist(
                base_state,
                agent_name="host",
                system_prompt_template=Template("You are a concierge for $property_name."),
                context_header="Context",
                no_context_fallback="No info available.",
                get_llm_fn=AsyncMock(return_value=mock_llm),
                get_cb_fn=AsyncMock(return_value=mock_cb),
            )

        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
        system_text = " ".join(m.content for m in system_msgs)
        assert "Escalation Guidance" not in system_text


# ---------------------------------------------------------------------------
# 2. Proactive Suggestion Tests
# ---------------------------------------------------------------------------


class TestProactiveSuggestions:
    """Tests for WhisperPlan proactive suggestion fields and injection."""

    def test_whisper_plan_with_suggestion(self):
        """WhisperPlan accepts proactive suggestion fields."""
        plan = WhisperPlan(
            next_topic="dining",
            extraction_targets=["cuisine_preference"],
            offer_readiness=0.3,
            conversation_note="Guest mentioned dinner",
            proactive_suggestion="Try Todd English's Tuscany for Italian cuisine",
            suggestion_confidence=0.9,
        )
        assert plan.proactive_suggestion == "Try Todd English's Tuscany for Italian cuisine"
        assert plan.suggestion_confidence == 0.9

    def test_whisper_plan_without_suggestion(self):
        """WhisperPlan defaults to no suggestion."""
        plan = WhisperPlan(
            next_topic="none",
            extraction_targets=[],
            offer_readiness=0.0,
            conversation_note="No suggestion needed",
        )
        assert plan.proactive_suggestion is None
        assert plan.suggestion_confidence == 0.0

    def test_format_whisper_plan_excludes_suggestion(self):
        """format_whisper_plan does NOT include suggestion (R23 fix C-001: deduplicated).

        Proactive suggestions are now injected by execute_specialist() in _base.py
        with proper sentiment gating and max-1 enforcement. format_whisper_plan()
        only formats planning data, not suggestions.
        """
        plan_dict = {
            "next_topic": "dining",
            "extraction_targets": [],
            "offer_readiness": 0.5,
            "conversation_note": "Guest interested in dinner",
            "proactive_suggestion": "Check out Beauty & Essex for a special evening",
            "suggestion_confidence": 0.9,
        }
        result = format_whisper_plan(plan_dict)
        # R23 fix C-001: suggestion removed from format_whisper_plan
        assert "Proactive suggestion" not in result
        # But planning data is still present
        assert "dining" in result
        assert "Whisper Track Guidance" in result

    def test_format_whisper_plan_excludes_low_confidence(self):
        """format_whisper_plan excludes suggestion when confidence < 0.8."""
        plan_dict = {
            "next_topic": "dining",
            "extraction_targets": [],
            "offer_readiness": 0.5,
            "conversation_note": "Maybe dining",
            "proactive_suggestion": "Try the buffet",
            "suggestion_confidence": 0.5,
        }
        result = format_whisper_plan(plan_dict)
        assert "Proactive suggestion" not in result
        assert "buffet" not in result

    def test_format_whisper_plan_excludes_none_suggestion(self):
        """format_whisper_plan excludes when suggestion is None."""
        plan_dict = {
            "next_topic": "none",
            "extraction_targets": [],
            "offer_readiness": 0.0,
            "conversation_note": "Nothing to suggest",
            "proactive_suggestion": None,
            "suggestion_confidence": 0.0,
        }
        result = format_whisper_plan(plan_dict)
        assert "Proactive suggestion" not in result

    def test_format_whisper_plan_none_input(self):
        """format_whisper_plan returns empty string for None input."""
        assert format_whisper_plan(None) == ""

    def test_suggestion_confidence_boundary_08(self):
        """Confidence at 0.8 does NOT show in format_whisper_plan (R23: deduplicated)."""
        plan_dict = {
            "next_topic": "none",
            "extraction_targets": [],
            "offer_readiness": 0.0,
            "conversation_note": "Boundary test",
            "proactive_suggestion": "Try the spa",
            "suggestion_confidence": 0.8,
        }
        result = format_whisper_plan(plan_dict)
        # R23 fix C-001: suggestions no longer in format_whisper_plan
        assert "Proactive suggestion" not in result

    def test_suggestion_confidence_boundary_079(self):
        """Confidence at 0.79 should NOT be included."""
        plan_dict = {
            "next_topic": "none",
            "extraction_targets": [],
            "offer_readiness": 0.0,
            "conversation_note": "Below threshold",
            "proactive_suggestion": "Try the spa",
            "suggestion_confidence": 0.79,
        }
        result = format_whisper_plan(plan_dict)
        assert "Proactive suggestion" not in result

    @pytest.mark.asyncio()
    async def test_suggestion_not_injected_when_sentiment_none(self, base_state, _disable_features):
        """Proactive suggestion is NOT injected when sentiment is None (R23 fix C-002)."""
        base_state["guest_sentiment"] = None  # Detection disabled or failed
        base_state["whisper_plan"] = {
            "next_topic": "dining",
            "extraction_targets": [],
            "offer_readiness": 0.5,
            "conversation_note": "Guest asked about dinner",
            "proactive_suggestion": "Try Todd English's",
            "suggestion_confidence": 0.95,
        }

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Here are our restaurants"))

        mock_cb = AsyncMock()
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()

        with patch("src.agent.agents._base.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                PROPERTY_NAME="Mohegan Sun",
                PROPERTY_PHONE="1-888-226-7711",
                MAX_HISTORY_MESSAGES=20,
            )

            await execute_specialist(
                base_state,
                agent_name="host",
                system_prompt_template=Template("You are a concierge for $property_name."),
                context_header="Context",
                no_context_fallback="No info.",
                get_llm_fn=AsyncMock(return_value=mock_llm),
                get_cb_fn=AsyncMock(return_value=mock_cb),
            )

        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
        system_text = " ".join(m.content for m in system_msgs)
        assert "Proactive Suggestion" not in system_text

    @pytest.mark.asyncio()
    async def test_suggestion_not_injected_when_frustrated(self, base_state, _disable_features):
        """Proactive suggestion is NOT injected when guest sentiment is frustrated."""
        base_state["guest_sentiment"] = "frustrated"
        base_state["whisper_plan"] = {
            "next_topic": "dining",
            "extraction_targets": [],
            "offer_readiness": 0.5,
            "conversation_note": "Guest asked about dinner",
            "proactive_suggestion": "Try Todd English's",
            "suggestion_confidence": 0.95,
        }

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Here are our restaurants"))

        mock_cb = AsyncMock()
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()

        with patch("src.agent.agents._base.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                PROPERTY_NAME="Mohegan Sun",
                PROPERTY_PHONE="1-888-226-7711",
                MAX_HISTORY_MESSAGES=20,
            )

            await execute_specialist(
                base_state,
                agent_name="host",
                system_prompt_template=Template("You are a concierge for $property_name."),
                context_header="Context",
                no_context_fallback="No info.",
                get_llm_fn=AsyncMock(return_value=mock_llm),
                get_cb_fn=AsyncMock(return_value=mock_cb),
            )

        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
        system_text = " ".join(m.content for m in system_msgs)
        assert "Proactive Suggestion" not in system_text

    @pytest.mark.asyncio()
    async def test_suggestion_injected_when_positive(self, base_state, _disable_features):
        """Proactive suggestion IS injected when sentiment is positive and confidence high."""
        base_state["guest_sentiment"] = "positive"
        base_state["whisper_plan"] = {
            "next_topic": "dining",
            "extraction_targets": [],
            "offer_readiness": 0.5,
            "conversation_note": "Guest asked about dinner",
            "proactive_suggestion": "Try Todd English's Tuscany for a celebratory dinner",
            "suggestion_confidence": 0.92,
        }

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Great choice!"))

        mock_cb = AsyncMock()
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()

        with patch("src.agent.agents._base.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                PROPERTY_NAME="Mohegan Sun",
                PROPERTY_PHONE="1-888-226-7711",
                MAX_HISTORY_MESSAGES=20,
            )

            await execute_specialist(
                base_state,
                agent_name="host",
                system_prompt_template=Template("You are a concierge for $property_name."),
                context_header="Context",
                no_context_fallback="No info.",
                get_llm_fn=AsyncMock(return_value=mock_llm),
                get_cb_fn=AsyncMock(return_value=mock_cb),
            )

        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
        system_text = " ".join(m.content for m in system_msgs)
        assert "Proactive Suggestion" in system_text
        assert "Todd English" in system_text


# ---------------------------------------------------------------------------
# 3. Persona Drift Prevention Tests
# ---------------------------------------------------------------------------


class TestPersonaDriftPrevention:
    """Tests for periodic system prompt re-injection."""

    def test_threshold_constant_exists(self):
        """The persona re-injection threshold constant is defined."""
        assert _PERSONA_REINJECT_THRESHOLD == 10

    @pytest.mark.asyncio()
    async def test_persona_reminder_injected_for_long_conversations(
        self, base_state, _disable_features
    ):
        """Persona reminder is injected when history exceeds threshold."""
        # Build a long conversation history (> 10 messages)
        long_history = []
        for i in range(7):
            long_history.append(HumanMessage(content=f"Question {i}"))
            long_history.append(AIMessage(content=f"Answer {i}"))
        base_state["messages"] = long_history

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Here you go"))

        mock_cb = AsyncMock()
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()

        with patch("src.agent.agents._base.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                PROPERTY_NAME="Mohegan Sun",
                PROPERTY_PHONE="1-888-226-7711",
                MAX_HISTORY_MESSAGES=20,
            )

            await execute_specialist(
                base_state,
                agent_name="host",
                system_prompt_template=Template("You are a concierge for $property_name."),
                context_header="Context",
                no_context_fallback="No info.",
                get_llm_fn=AsyncMock(return_value=mock_llm),
                get_cb_fn=AsyncMock(return_value=mock_cb),
            )

        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
        system_text = " ".join(m.content for m in system_msgs)
        assert "PERSONA REMINDER" in system_text
        assert "Mohegan Sun" in system_text

    @pytest.mark.asyncio()
    async def test_no_persona_reminder_for_short_conversations(
        self, base_state, _disable_features
    ):
        """Persona reminder is NOT injected for short conversations."""
        # Short history (< 10 messages)
        base_state["messages"] = [
            HumanMessage(content="Hello"),
            AIMessage(content="Welcome!"),
            HumanMessage(content="What restaurants?"),
        ]

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="We have many"))

        mock_cb = AsyncMock()
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()

        with patch("src.agent.agents._base.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                PROPERTY_NAME="Mohegan Sun",
                PROPERTY_PHONE="1-888-226-7711",
                MAX_HISTORY_MESSAGES=20,
            )

            await execute_specialist(
                base_state,
                agent_name="host",
                system_prompt_template=Template("You are a concierge for $property_name."),
                context_header="Context",
                no_context_fallback="No info.",
                get_llm_fn=AsyncMock(return_value=mock_llm),
                get_cb_fn=AsyncMock(return_value=mock_cb),
            )

        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
        system_text = " ".join(m.content for m in system_msgs)
        assert "PERSONA REMINDER" not in system_text


# ---------------------------------------------------------------------------
# 4. WhisperPlan Validation Tests
# ---------------------------------------------------------------------------


class TestWhisperPlanValidation:
    """Tests for WhisperPlan Pydantic validation of new fields."""

    def test_suggestion_confidence_must_be_0_to_1(self):
        """suggestion_confidence must be between 0.0 and 1.0."""
        with pytest.raises(Exception):  # ValidationError
            WhisperPlan(
                next_topic="none",
                extraction_targets=[],
                offer_readiness=0.0,
                conversation_note="Test",
                suggestion_confidence=1.5,
            )

    def test_suggestion_confidence_negative_fails(self):
        """Negative suggestion_confidence fails validation."""
        with pytest.raises(Exception):  # ValidationError
            WhisperPlan(
                next_topic="none",
                extraction_targets=[],
                offer_readiness=0.0,
                conversation_note="Test",
                suggestion_confidence=-0.1,
            )

    def test_model_dump_includes_new_fields(self):
        """model_dump() includes proactive_suggestion and suggestion_confidence."""
        plan = WhisperPlan(
            next_topic="none",
            extraction_targets=[],
            offer_readiness=0.0,
            conversation_note="Test",
            proactive_suggestion="Try the spa",
            suggestion_confidence=0.85,
        )
        dumped = plan.model_dump()
        assert "proactive_suggestion" in dumped
        assert "suggestion_confidence" in dumped
        assert dumped["proactive_suggestion"] == "Try the spa"
        assert dumped["suggestion_confidence"] == 0.85

    def test_model_dump_defaults(self):
        """model_dump() defaults for new fields are None/0.0."""
        plan = WhisperPlan(
            next_topic="none",
            extraction_targets=[],
            offer_readiness=0.0,
            conversation_note="Test",
        )
        dumped = plan.model_dump()
        assert dumped["proactive_suggestion"] is None
        assert dumped["suggestion_confidence"] == 0.0
