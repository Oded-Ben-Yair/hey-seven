"""R21 agent quality tests: frustration escalation, proactive suggestions, persona drift.

Mock purge R111: Retained only deterministic tests that do not depend on
MagicMock/AsyncMock/@patch. All behavioral validation uses live eval.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.agents._base import (
    _PERSONA_REINJECT_THRESHOLD,
    _count_consecutive_frustrated,
)
from src.agent.whisper_planner import WhisperPlan, format_whisper_plan


# ---------------------------------------------------------------------------
# 1. Frustration Escalation Tests (deterministic — pure function)
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


# ---------------------------------------------------------------------------
# 2. Proactive Suggestion Tests (deterministic — Pydantic model + format fn)
# ---------------------------------------------------------------------------


class TestProactiveSuggestions:
    """Tests for WhisperPlan proactive suggestion fields and injection."""

    def test_whisper_plan_with_suggestion(self):
        """WhisperPlan accepts proactive suggestion fields."""
        plan = WhisperPlan(
            next_topic="dining",
            conversation_note="Guest mentioned dinner",
            proactive_suggestion="Try Todd English's Tuscany for Italian cuisine",
            suggestion_confidence="0.9",
        )
        assert (
            plan.proactive_suggestion
            == "Try Todd English's Tuscany for Italian cuisine"
        )
        assert plan.suggestion_confidence == "0.9"

    def test_whisper_plan_without_suggestion(self):
        """WhisperPlan defaults to no suggestion."""
        plan = WhisperPlan(
            next_topic="none",
            conversation_note="No suggestion needed",
        )
        assert plan.proactive_suggestion is None
        assert plan.suggestion_confidence == "0.0"

    def test_format_whisper_plan_excludes_suggestion(self):
        """format_whisper_plan does NOT include suggestion (R23 fix C-001: deduplicated)."""
        plan_dict = {
            "next_topic": "dining",
            "conversation_note": "Guest interested in dinner",
            "proactive_suggestion": "Check out Beauty & Essex for a special evening",
            "suggestion_confidence": "0.9",
        }
        result = format_whisper_plan(plan_dict)
        assert "Proactive suggestion" not in result
        assert "dining" in result
        assert "Whisper Track Guidance" in result

    def test_format_whisper_plan_excludes_low_confidence(self):
        """format_whisper_plan excludes suggestion when confidence < 0.8."""
        plan_dict = {
            "next_topic": "dining",
            "conversation_note": "Maybe dining",
            "proactive_suggestion": "Try the buffet",
            "suggestion_confidence": "0.5",
        }
        result = format_whisper_plan(plan_dict)
        assert "Proactive suggestion" not in result
        assert "buffet" not in result

    def test_format_whisper_plan_excludes_none_suggestion(self):
        """format_whisper_plan excludes when suggestion is None."""
        plan_dict = {
            "next_topic": "none",
            "conversation_note": "Nothing to suggest",
            "proactive_suggestion": None,
            "suggestion_confidence": "0.0",
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
            "conversation_note": "Boundary test",
            "proactive_suggestion": "Try the spa",
            "suggestion_confidence": "0.8",
        }
        result = format_whisper_plan(plan_dict)
        assert "Proactive suggestion" not in result

    def test_suggestion_confidence_boundary_079(self):
        """Confidence at 0.79 should NOT be included."""
        plan_dict = {
            "next_topic": "none",
            "conversation_note": "Below threshold",
            "proactive_suggestion": "Try the spa",
            "suggestion_confidence": "0.79",
        }
        result = format_whisper_plan(plan_dict)
        assert "Proactive suggestion" not in result


# ---------------------------------------------------------------------------
# 3. Persona Drift Prevention Tests (deterministic — constant value)
# ---------------------------------------------------------------------------


class TestPersonaDriftPrevention:
    """Tests for periodic system prompt re-injection."""

    def test_threshold_constant_exists(self):
        """The persona re-injection threshold constant is defined."""
        assert _PERSONA_REINJECT_THRESHOLD == 10


# ---------------------------------------------------------------------------
# 4. WhisperPlan Validation Tests (deterministic — Pydantic model)
# ---------------------------------------------------------------------------


class TestWhisperPlanValidation:
    """Tests for WhisperPlan Pydantic validation of new fields."""

    def test_suggestion_confidence_accepts_any_string(self):
        """R76 simplification: suggestion_confidence is plain str, no range validation."""
        plan = WhisperPlan(
            next_topic="none",
            conversation_note="Test",
            suggestion_confidence="1.5",
        )
        assert plan.suggestion_confidence == "1.5"

    def test_suggestion_confidence_accepts_negative_string(self):
        """R76 simplification: suggestion_confidence is plain str, no range validation."""
        plan = WhisperPlan(
            next_topic="none",
            conversation_note="Test",
            suggestion_confidence="-0.1",
        )
        assert plan.suggestion_confidence == "-0.1"

    def test_model_dump_includes_new_fields(self):
        """model_dump() includes proactive_suggestion and suggestion_confidence."""
        plan = WhisperPlan(
            next_topic="none",
            conversation_note="Test",
            proactive_suggestion="Try the spa",
            suggestion_confidence="0.85",
        )
        dumped = plan.model_dump()
        assert "proactive_suggestion" in dumped
        assert "suggestion_confidence" in dumped
        assert dumped["proactive_suggestion"] == "Try the spa"
        assert dumped["suggestion_confidence"] == "0.85"

    def test_model_dump_defaults(self):
        """model_dump() defaults for new fields are None/0.0."""
        plan = WhisperPlan(
            next_topic="none",
            conversation_note="Test",
        )
        dumped = plan.model_dump()
        assert dumped["proactive_suggestion"] is None
        assert dumped["suggestion_confidence"] == "0.0"
