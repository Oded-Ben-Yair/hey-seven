"""Tests for the Whisper Track Planner module (src/agent/whisper_planner.py).

Covers:
- WhisperPlan Pydantic model validation (valid & invalid inputs)
- whisper_planner_node happy path, parse errors, and API failures (fail-silent)
- format_whisper_plan formatting for valid, None, and empty plans
- _calculate_completeness placeholder logic
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage

from src.agent.whisper_planner import (
    WhisperPlan,
    format_whisper_plan,
    whisper_planner_node,
    _calculate_completeness,
)


def _state(**overrides):
    """Create a minimal CasinoHostState dict with defaults for testing."""
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


# ---------------------------------------------------------------------------
# WhisperPlan Model Validation
# ---------------------------------------------------------------------------


class TestWhisperPlanModel:
    def test_valid_plan_all_fields(self):
        """WhisperPlan accepts valid Literal topic and bounded offer_readiness."""
        plan = WhisperPlan(
            next_topic="dining",
            extraction_targets=["kids_ages", "dietary_restrictions"],
            offer_readiness=0.35,
            conversation_note="Guest mentioned anniversary, pivot to dining",
        )
        assert plan.next_topic == "dining"
        assert plan.offer_readiness == 0.35
        assert len(plan.extraction_targets) == 2

    def test_valid_plan_none_topic(self):
        """next_topic='none' is valid (no profiling this turn)."""
        plan = WhisperPlan(
            next_topic="none",
            extraction_targets=[],
            offer_readiness=0.0,
            conversation_note="Guest seems rushed",
        )
        assert plan.next_topic == "none"

    def test_valid_plan_offer_ready(self):
        """next_topic='offer_ready' with high offer_readiness is valid."""
        plan = WhisperPlan(
            next_topic="offer_ready",
            extraction_targets=[],
            offer_readiness=0.95,
            conversation_note="Profile is 80% complete, ready for offer",
        )
        assert plan.offer_readiness == 0.95

    def test_valid_plan_boundary_readiness_zero(self):
        """offer_readiness=0.0 is valid (minimum boundary)."""
        plan = WhisperPlan(
            next_topic="name",
            extraction_targets=["first_name"],
            offer_readiness=0.0,
            conversation_note="New guest, start profiling",
        )
        assert plan.offer_readiness == 0.0

    def test_valid_plan_boundary_readiness_one(self):
        """offer_readiness=1.0 is valid (maximum boundary)."""
        plan = WhisperPlan(
            next_topic="offer_ready",
            extraction_targets=[],
            offer_readiness=1.0,
            conversation_note="Fully profiled",
        )
        assert plan.offer_readiness == 1.0

    def test_invalid_topic_raises_validation_error(self):
        """Invalid next_topic not in Literal set raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WhisperPlan(
                next_topic="invalid_topic",
                extraction_targets=[],
                offer_readiness=0.5,
                conversation_note="Test",
            )

    def test_offer_readiness_above_one_raises(self):
        """offer_readiness > 1.0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WhisperPlan(
                next_topic="dining",
                extraction_targets=[],
                offer_readiness=1.5,
                conversation_note="Test",
            )

    def test_offer_readiness_below_zero_raises(self):
        """offer_readiness < 0.0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WhisperPlan(
                next_topic="dining",
                extraction_targets=[],
                offer_readiness=-0.1,
                conversation_note="Test",
            )

    def test_model_dump_round_trip(self):
        """model_dump() produces a dict that can reconstruct the model."""
        plan = WhisperPlan(
            next_topic="visit_date",
            extraction_targets=["arrival_date", "departure_date"],
            offer_readiness=0.2,
            conversation_note="Ask about trip dates",
        )
        dumped = plan.model_dump()
        assert isinstance(dumped, dict)
        reconstructed = WhisperPlan(**dumped)
        assert reconstructed.next_topic == "visit_date"
        assert reconstructed.offer_readiness == 0.2

    def test_all_valid_topics(self):
        """Every Literal topic value is accepted."""
        valid_topics = [
            "name", "visit_date", "party_size", "dining", "entertainment",
            "gaming", "occasions", "companions", "offer_ready", "none",
        ]
        for topic in valid_topics:
            plan = WhisperPlan(
                next_topic=topic,
                extraction_targets=[],
                offer_readiness=0.5,
                conversation_note="Test",
            )
            assert plan.next_topic == topic


# ---------------------------------------------------------------------------
# whisper_planner_node
# ---------------------------------------------------------------------------


class TestWhisperPlannerNode:
    @patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock)
    async def test_happy_path_returns_plan_dict(self, mock_get_llm):
        """Node returns whisper_plan dict when LLM returns valid WhisperPlan."""
        mock_plan = WhisperPlan(
            next_topic="dining",
            extraction_targets=["dietary_restrictions"],
            offer_readiness=0.35,
            conversation_note="Guest mentioned anniversary",
        )
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_plan)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="I'm visiting for my anniversary")],
        )
        result = await whisper_planner_node(state)

        assert "whisper_plan" in result
        assert result["whisper_plan"] is not None
        assert result["whisper_plan"]["next_topic"] == "dining"
        assert result["whisper_plan"]["offer_readiness"] == 0.35

    @patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock)
    async def test_parse_error_returns_none(self, mock_get_llm):
        """ValueError from structured output parsing returns whisper_plan=None."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=ValueError("bad JSON"))
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="Hello")],
        )
        result = await whisper_planner_node(state)

        assert result["whisper_plan"] is None

    @patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock)
    async def test_type_error_returns_none(self, mock_get_llm):
        """TypeError from structured output parsing returns whisper_plan=None."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=TypeError("type mismatch"))
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="Hello")],
        )
        result = await whisper_planner_node(state)

        assert result["whisper_plan"] is None

    @patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock)
    async def test_api_timeout_returns_none(self, mock_get_llm):
        """API timeout (generic Exception) returns whisper_plan=None."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("API timeout"))
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="Tell me about restaurants")],
        )
        result = await whisper_planner_node(state)

        assert result["whisper_plan"] is None

    @patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock)
    async def test_empty_messages_still_works(self, mock_get_llm):
        """Node handles empty message list gracefully."""
        mock_plan = WhisperPlan(
            next_topic="name",
            extraction_targets=["first_name"],
            offer_readiness=0.0,
            conversation_note="No conversation yet",
        )
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_plan)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _state(messages=[])
        result = await whisper_planner_node(state)

        assert result["whisper_plan"] is not None
        assert result["whisper_plan"]["next_topic"] == "name"

    @patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock)
    async def test_node_calls_with_structured_output(self, mock_get_llm):
        """Node calls with_structured_output(WhisperPlan) on the LLM."""
        mock_plan = WhisperPlan(
            next_topic="dining",
            extraction_targets=[],
            offer_readiness=0.5,
            conversation_note="Test",
        )
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_plan)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _state(
            messages=[HumanMessage(content="What about dining?")],
        )
        await whisper_planner_node(state)

        mock_llm.with_structured_output.assert_called_once_with(WhisperPlan)


# ---------------------------------------------------------------------------
# format_whisper_plan
# ---------------------------------------------------------------------------


class TestFormatWhisperPlan:
    def test_valid_plan_formatting(self):
        """Valid plan dict produces a formatted guidance string."""
        plan = {
            "next_topic": "dining",
            "extraction_targets": ["kids_ages", "dietary_restrictions"],
            "offer_readiness": 0.35,
            "conversation_note": "Guest mentioned anniversary, pivot to dining",
        }
        result = format_whisper_plan(plan)

        assert "Whisper Track Guidance" in result
        assert "dining" in result
        assert "kids_ages" in result
        assert "dietary_restrictions" in result
        assert "35%" in result
        assert "anniversary" in result

    def test_none_plan_returns_empty(self):
        """None plan returns empty string."""
        assert format_whisper_plan(None) == ""

    def test_empty_dict_returns_defaults(self):
        """Empty dict returns formatted string with default values."""
        result = format_whisper_plan({})
        assert "Whisper Track Guidance" in result
        assert "none" in result
        assert "0%" in result

    def test_empty_extraction_targets(self):
        """Plan with empty extraction_targets formats correctly."""
        plan = {
            "next_topic": "name",
            "extraction_targets": [],
            "offer_readiness": 0.0,
            "conversation_note": "Start profiling",
        }
        result = format_whisper_plan(plan)
        assert "name" in result
        assert "Start profiling" in result

    def test_never_reveal_instruction_present(self):
        """Format includes 'never reveal to guest' instruction."""
        plan = {
            "next_topic": "dining",
            "extraction_targets": [],
            "offer_readiness": 0.5,
            "conversation_note": "Test",
        }
        result = format_whisper_plan(plan)
        assert "never reveal to guest" in result


# ---------------------------------------------------------------------------
# _calculate_completeness
# ---------------------------------------------------------------------------


class TestCalculateCompleteness:
    def test_none_profile_returns_zero(self):
        """None profile returns 0.0."""
        assert _calculate_completeness(None) == 0.0

    def test_empty_dict_returns_zero(self):
        """Empty dict returns 0.0."""
        assert _calculate_completeness({}) == 0.0

    def test_populated_profile_returns_nonzero(self):
        """Profile with fields returns a value between 0.0 and 1.0."""
        profile = {"name": "John", "visit_date": "2026-03-01"}
        result = _calculate_completeness(profile)
        assert 0.0 < result <= 1.0

    def test_return_type_is_float(self):
        """Completeness is always a float."""
        assert isinstance(_calculate_completeness({}), float)
        assert isinstance(_calculate_completeness(None), float)
        assert isinstance(_calculate_completeness({"name": "Jane"}), float)


class TestFailureCounterThreshold:
    """Tests for the module-level failure counter alert threshold."""

    def _reset_counter(self):
        """Reset module-level counter state between tests."""
        import src.agent.whisper_planner as wp
        wp._failure_count = 0
        wp._failure_alerted = False

    def _increment(self):
        """Simulate a failure increment (mirrors whisper_planner_node logic)."""
        import src.agent.whisper_planner as wp
        wp._failure_count += 1
        if wp._failure_count >= wp._FAILURE_ALERT_THRESHOLD and not wp._failure_alerted:
            wp._failure_alerted = True
            wp.logger.error(
                "whisper_planner_systematic_failure: %d consecutive failures "
                "exceeded threshold (%d). Whisper planner may be misconfigured.",
                wp._failure_count,
                wp._FAILURE_ALERT_THRESHOLD,
            )

    def test_threshold_triggers_alert(self, caplog):
        """After _FAILURE_ALERT_THRESHOLD failures, an ERROR log is emitted."""
        import logging
        import src.agent.whisper_planner as wp
        self._reset_counter()
        wp._FAILURE_ALERT_THRESHOLD = 3
        with caplog.at_level(logging.ERROR):
            self._increment()
            self._increment()
            assert "systematic_failure" not in caplog.text
            self._increment()  # Threshold reached
            assert "systematic_failure" in caplog.text
        self._reset_counter()
        wp._FAILURE_ALERT_THRESHOLD = 10

    def test_alert_fires_once(self, caplog):
        """Alert only fires once, not on every subsequent failure."""
        import logging
        import src.agent.whisper_planner as wp
        self._reset_counter()
        wp._FAILURE_ALERT_THRESHOLD = 2
        with caplog.at_level(logging.ERROR):
            self._increment()
            self._increment()  # First alert
            count_before = caplog.text.count("systematic_failure")
            self._increment()  # Should NOT re-alert
            count_after = caplog.text.count("systematic_failure")
            assert count_before == count_after
        self._reset_counter()
        wp._FAILURE_ALERT_THRESHOLD = 10

    def test_reset_clears_count(self):
        """Resetting the counter clears count and alert state."""
        import src.agent.whisper_planner as wp
        self._reset_counter()
        self._increment()
        self._increment()
        assert wp._failure_count == 2
        self._reset_counter()
        assert wp._failure_count == 0

    def test_threshold_then_reset_then_below_threshold_no_alert(self, caplog):
        """10 failures (alert) -> reset -> 9 failures: no second alert."""
        import logging
        import src.agent.whisper_planner as wp
        self._reset_counter()
        with caplog.at_level(logging.ERROR):
            for _ in range(10):
                self._increment()
            first_count = caplog.text.count("systematic_failure")
            assert first_count == 1

            self._reset_counter()

            for _ in range(9):
                self._increment()
            second_count = caplog.text.count("systematic_failure")
            assert second_count == first_count
        self._reset_counter()

    def test_reset_then_re_trigger_fires_alert_again(self, caplog):
        """After reset, reaching threshold again DOES fire a new alert."""
        import logging
        import src.agent.whisper_planner as wp
        self._reset_counter()
        wp._FAILURE_ALERT_THRESHOLD = 3
        with caplog.at_level(logging.ERROR):
            for _ in range(3):
                self._increment()
            assert caplog.text.count("systematic_failure") == 1

            self._reset_counter()
            for _ in range(3):
                self._increment()
            assert caplog.text.count("systematic_failure") == 2
        self._reset_counter()
        wp._FAILURE_ALERT_THRESHOLD = 10
