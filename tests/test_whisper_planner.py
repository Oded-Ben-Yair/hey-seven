"""Tests for the Whisper Track Planner module (src/agent/whisper_planner.py).

Covers:
- WhisperPlan Pydantic model validation (valid & invalid inputs)
- format_whisper_plan formatting for valid, None, and empty plans
- _calculate_completeness placeholder logic
- Failure counter threshold alerting

Mock purge R111: Removed TestWhisperPlannerNode (all mock-based LLM tests).
Validate whisper_planner_node via live evaluation framework.
"""

import pytest

from src.agent.whisper_planner import (
    WhisperPlan,
    format_whisper_plan,
    _calculate_completeness,
)


# ---------------------------------------------------------------------------
# WhisperPlan Model Validation
# ---------------------------------------------------------------------------


class TestWhisperPlanModel:
    def test_valid_plan_all_fields(self):
        """WhisperPlan accepts all fields with valid values."""
        plan = WhisperPlan(
            next_topic="dining",
            conversation_note="Guest mentioned anniversary, pivot to dining",
            proactive_suggestion="Try the steakhouse",
            suggestion_confidence="0.85",
            next_profiling_question="How many in your party?",
            question_technique="give_to_get",
        )
        assert plan.next_topic == "dining"
        assert plan.suggestion_confidence == "0.85"
        assert plan.proactive_suggestion == "Try the steakhouse"
        assert plan.question_technique == "give_to_get"

    def test_valid_plan_none_topic(self):
        """next_topic='none' is valid (no profiling this turn)."""
        plan = WhisperPlan(
            next_topic="none",
            conversation_note="Guest seems rushed",
        )
        assert plan.next_topic == "none"

    def test_valid_plan_offer_ready_topic(self):
        """next_topic='offer_ready' is valid as a plain string."""
        plan = WhisperPlan(
            next_topic="offer_ready",
            conversation_note="Profile is 80% complete, ready for offer",
        )
        assert plan.next_topic == "offer_ready"

    def test_suggestion_confidence_is_str(self):
        """suggestion_confidence is a str field, not a bounded float."""
        plan = WhisperPlan(
            next_topic="name",
            suggestion_confidence="0.0",
            conversation_note="New guest, start profiling",
        )
        assert plan.suggestion_confidence == "0.0"

    def test_profiling_question_defaults_to_none(self):
        """next_profiling_question defaults to None."""
        plan = WhisperPlan(conversation_note="Fully profiled")
        assert plan.next_profiling_question is None

    def test_any_string_topic_is_valid(self):
        """next_topic is a plain str -- any value is accepted (no Literal constraint)."""
        plan = WhisperPlan(
            next_topic="custom_topic",
            conversation_note="Test",
        )
        assert plan.next_topic == "custom_topic"

    def test_question_technique_accepts_any_str(self):
        """question_technique is a plain str -- any value is accepted."""
        plan = WhisperPlan(
            next_topic="dining",
            question_technique="assumptive_bridge",
            conversation_note="Test",
        )
        assert plan.question_technique == "assumptive_bridge"

    def test_defaults_are_applied(self):
        """All fields have defaults -- an empty WhisperPlan is valid."""
        plan = WhisperPlan()
        assert plan.next_topic == "none"
        assert plan.conversation_note == ""
        assert plan.proactive_suggestion is None
        assert plan.suggestion_confidence == "0.0"
        assert plan.next_profiling_question is None
        assert plan.question_technique == "none"

    def test_model_dump_round_trip(self):
        """model_dump() produces a dict that can reconstruct the model."""
        plan = WhisperPlan(
            next_topic="visit_date",
            conversation_note="Ask about trip dates",
            suggestion_confidence="0.2",
        )
        dumped = plan.model_dump()
        assert isinstance(dumped, dict)
        reconstructed = WhisperPlan(**dumped)
        assert reconstructed.next_topic == "visit_date"
        assert reconstructed.suggestion_confidence == "0.2"

    def test_all_expected_topics(self):
        """All expected topic values are accepted (next_topic is plain str)."""
        # R103 fix P8: Updated to match corrected _PROFILE_FIELDS
        expected_topics = [
            "name",
            "party_size",
            "visit_purpose",
            "preferences",
            "entertainment",
            "gaming",
            "spa",
            "occasion",
            "party_composition",
            "visit_duration",
            "offer_ready",
            "none",
        ]
        for topic in expected_topics:
            plan = WhisperPlan(
                next_topic=topic,
                conversation_note="Test",
            )
            assert plan.next_topic == topic


# ---------------------------------------------------------------------------
# format_whisper_plan
# ---------------------------------------------------------------------------


class TestFormatWhisperPlan:
    def test_valid_plan_formatting(self):
        """Valid plan dict produces a formatted guidance string."""
        plan = {
            "next_topic": "dining",
            "conversation_note": "Guest mentioned anniversary, pivot to dining",
            "suggestion_confidence": "0.35",
            "next_profiling_question": "How many in your party?",
            "question_technique": "give_to_get",
        }
        result = format_whisper_plan(plan)

        assert "Whisper Track Guidance" in result
        assert "dining" in result
        assert "anniversary" in result
        assert "give_to_get" in result
        assert "How many in your party?" in result

    def test_none_plan_returns_empty(self):
        """None plan returns empty string."""
        assert format_whisper_plan(None) == ""

    def test_empty_dict_returns_defaults(self):
        """Empty dict returns formatted string with default values."""
        result = format_whisper_plan({})
        assert "Whisper Track Guidance" in result
        assert "none" in result

    def test_plan_without_profiling_question(self):
        """Plan with no profiling question formats correctly (no profiling line)."""
        plan = {
            "next_topic": "name",
            "conversation_note": "Start profiling",
            "question_technique": "none",
        }
        result = format_whisper_plan(plan)
        assert "name" in result
        assert "Start profiling" in result
        assert "Profiling question" not in result

    def test_never_reveal_instruction_present(self):
        """Format includes 'never reveal to guest' instruction."""
        plan = {
            "next_topic": "dining",
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
        # R103 fix P8: use corrected field names (preferences, not dining)
        profile = {"name": "John", "preferences": "Italian"}
        result = _calculate_completeness(profile)
        assert 0.0 < result <= 1.0

    def test_return_type_is_float(self):
        """Completeness is always a float."""
        assert isinstance(_calculate_completeness({}), float)
        assert isinstance(_calculate_completeness(None), float)
        assert isinstance(_calculate_completeness({"name": "Jane"}), float)

    def test_corrected_field_names_recognized(self):
        """R107: completeness delegates to profiling weighted calculation."""
        from src.agent.profiling import _calculate_profile_completeness_weighted

        profile = {
            "name": "Sarah",
            "party_size": 4,
            "visit_purpose": "celebration",
            "preferences": "Italian",
            "occasion": "anniversary",
        }
        result = _calculate_completeness(profile)
        expected = _calculate_profile_completeness_weighted(profile)
        assert result == expected

    def test_stale_field_names_produce_zero(self):
        """R103 fix P8: old stale field names produce low/zero score."""
        profile = {
            "dining": "Italian",
            "occasions": "birthday",
            "companions": "2 adults",
        }
        result = _calculate_completeness(profile)
        assert result == 0.0


class TestWhisperFieldParity:
    """R107: Verify whisper _calculate_completeness delegates to profiling."""

    def test_whisper_delegates_to_profiling(self):
        """_calculate_completeness uses profiling._calculate_profile_completeness_weighted."""
        from src.agent.profiling import _calculate_profile_completeness_weighted

        profile = {"name": "Mike", "party_size": 2, "gaming": "slots"}
        whisper_result = _calculate_completeness(profile)
        profiling_result = _calculate_profile_completeness_weighted(profile)
        assert whisper_result == profiling_result


class TestFailureCounterThreshold:
    """Tests for the _WhisperTelemetry failure counter alert threshold."""

    def _reset_counter(self):
        """Reset telemetry singleton state between tests."""
        import src.agent.whisper_planner as wp

        wp._telemetry.count = 0
        wp._telemetry.alerted = False

    def _increment(self):
        """Simulate a failure increment (mirrors whisper_planner_node logic)."""
        import src.agent.whisper_planner as wp

        wp._telemetry.count += 1
        if (
            wp._telemetry.count >= wp._telemetry.ALERT_THRESHOLD
            and not wp._telemetry.alerted
        ):
            wp._telemetry.alerted = True
            wp.logger.error(
                "whisper_planner_systematic_failure: %d consecutive failures "
                "exceeded threshold (%d). Whisper planner may be misconfigured.",
                wp._telemetry.count,
                wp._telemetry.ALERT_THRESHOLD,
            )

    def test_threshold_triggers_alert(self, caplog):
        """After ALERT_THRESHOLD failures, an ERROR log is emitted."""
        import logging
        import src.agent.whisper_planner as wp

        self._reset_counter()
        original_threshold = wp._telemetry.ALERT_THRESHOLD
        wp._telemetry.ALERT_THRESHOLD = 3
        with caplog.at_level(logging.ERROR):
            self._increment()
            self._increment()
            assert "systematic_failure" not in caplog.text
            self._increment()
            assert "systematic_failure" in caplog.text
        self._reset_counter()
        wp._telemetry.ALERT_THRESHOLD = original_threshold

    def test_alert_fires_once(self, caplog):
        """Alert only fires once, not on every subsequent failure."""
        import logging
        import src.agent.whisper_planner as wp

        self._reset_counter()
        original_threshold = wp._telemetry.ALERT_THRESHOLD
        wp._telemetry.ALERT_THRESHOLD = 2
        with caplog.at_level(logging.ERROR):
            self._increment()
            self._increment()
            count_before = caplog.text.count("systematic_failure")
            self._increment()
            count_after = caplog.text.count("systematic_failure")
            assert count_before == count_after
        self._reset_counter()
        wp._telemetry.ALERT_THRESHOLD = original_threshold

    def test_reset_clears_count(self):
        """Resetting the counter clears count and alert state."""
        import src.agent.whisper_planner as wp

        self._reset_counter()
        self._increment()
        self._increment()
        assert wp._telemetry.count == 2
        self._reset_counter()
        assert wp._telemetry.count == 0

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
        original_threshold = wp._telemetry.ALERT_THRESHOLD
        wp._telemetry.ALERT_THRESHOLD = 3
        with caplog.at_level(logging.ERROR):
            for _ in range(3):
                self._increment()
            assert caplog.text.count("systematic_failure") == 1

            self._reset_counter()
            for _ in range(3):
                self._increment()
            assert caplog.text.count("systematic_failure") == 2
        self._reset_counter()
        wp._telemetry.ALERT_THRESHOLD = original_threshold
