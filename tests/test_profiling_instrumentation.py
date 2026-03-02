"""Tests for profiling instrumentation logging (R82 Track 2E).

Verifies that profiling_enrichment_node logs:
1. Extraction results (filled fields, skipped fields)
2. Profile state (phase, completeness, field counts)
3. Question injection decisions (candidate, technique, reason)
4. Skip conditions (no user message)

These tests exercise the logging infrastructure, not the LLM extraction quality.
"""

import logging
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.profiling import (
    ProfileExtractionOutput,
    PROFILING_TECHNIQUE_PROMPTS,
    profiling_enrichment_node,
    _calculate_profile_completeness_weighted,
    _determine_profiling_phase,
)

# The LLM is imported inside profiling_enrichment_node via:
#   from src.agent.whisper_planner import _get_whisper_llm
# So the correct patch target is always src.agent.whisper_planner._get_whisper_llm
_WHISPER_LLM_PATCH = "src.agent.whisper_planner._get_whisper_llm"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profiling_state(**overrides):
    """Create a minimal state dict for profiling_enrichment_node tests."""
    base = {
        "messages": [
            HumanMessage(content="Hi, I'm Mike, party of 4"),
            AIMessage(content="Welcome Mike! How can I help?"),
        ],
        "extracted_fields": {},
        "query_type": "property_qa",
        "whisper_plan": None,
        "guest_sentiment": "neutral",
    }
    base.update(overrides)
    return base


def _mock_extraction_result(**field_values):
    """Create a mock ProfileExtractionOutput with specified fields."""
    defaults = {f: None for f in ProfileExtractionOutput.model_fields}
    defaults.update(field_values)
    return ProfileExtractionOutput(**defaults)


def _patch_whisper_llm(extraction_result):
    """Create a patch context manager for the whisper LLM returning extraction_result.

    The call chain in profiling_enrichment_node is:
        llm = await _get_whisper_llm()              # async, returns LLM object
        extraction_llm = llm.with_structured_output(ProfileExtractionOutput)  # sync
        result = await extraction_llm.ainvoke(prompt_text)  # async
    """
    # extraction_llm.ainvoke() must be async and return the extraction_result
    mock_extraction_llm = MagicMock()
    mock_extraction_llm.ainvoke = AsyncMock(return_value=extraction_result)

    # llm.with_structured_output() is sync and returns mock_extraction_llm
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_extraction_llm

    # _get_whisper_llm() is async and returns mock_llm_instance
    async def _mock_get_whisper_llm():
        return mock_llm_instance

    return patch(_WHISPER_LLM_PATCH, new=_mock_get_whisper_llm)


# ---------------------------------------------------------------------------
# Module-level attribute tests
# ---------------------------------------------------------------------------


class TestProfilingModuleAttributes:
    """Verify profiling module exports and structure."""

    def test_profiling_node_is_callable(self):
        """profiling_enrichment_node should be importable and callable."""
        assert callable(profiling_enrichment_node)

    def test_technique_prompts_exist(self):
        """PROFILING_TECHNIQUE_PROMPTS should have entries for each technique."""
        assert isinstance(PROFILING_TECHNIQUE_PROMPTS, dict)
        assert len(PROFILING_TECHNIQUE_PROMPTS) >= 3

    def test_technique_prompts_include_core_techniques(self):
        """Core techniques should be present."""
        assert "give_to_get" in PROFILING_TECHNIQUE_PROMPTS
        assert "assumptive_bridge" in PROFILING_TECHNIQUE_PROMPTS
        assert "none" in PROFILING_TECHNIQUE_PROMPTS

    def test_none_technique_is_noop(self):
        """'none' technique should indicate no question."""
        assert "not" in PROFILING_TECHNIQUE_PROMPTS["none"].lower() or \
               "do not" in PROFILING_TECHNIQUE_PROMPTS["none"].lower()

    def test_profile_extraction_output_has_fields(self):
        """ProfileExtractionOutput should have expected fields."""
        fields = set(ProfileExtractionOutput.model_fields.keys())
        assert "guest_name" in fields
        assert "party_size" in fields
        assert "visit_purpose" in fields
        assert "occasion" in fields
        assert "dining_preferences" in fields


# ---------------------------------------------------------------------------
# Completeness and phase calculation
# ---------------------------------------------------------------------------


class TestProfileCompleteness:
    """Test _calculate_profile_completeness_weighted."""

    def test_empty_fields_zero_completeness(self):
        """Empty extracted_fields should give 0.0 completeness."""
        assert _calculate_profile_completeness_weighted({}) == 0.0

    def test_name_only_gives_partial(self):
        """Name alone should give 0.15 completeness (name weight)."""
        result = _calculate_profile_completeness_weighted({"name": "Mike"})
        assert 0.1 <= result <= 0.2

    def test_multiple_fields_higher_completeness(self):
        """Multiple fields should give higher completeness."""
        fields = {"name": "Mike", "party_size": "4", "visit_purpose": "celebration"}
        result = _calculate_profile_completeness_weighted(fields)
        assert result > 0.3

    def test_none_values_not_counted(self):
        """None values should not contribute to completeness."""
        result = _calculate_profile_completeness_weighted({"name": None, "party_size": None})
        assert result == 0.0

    def test_max_completeness_is_1(self):
        """Completeness should never exceed 1.0."""
        all_fields = {
            "name": "Mike", "party_size": "4", "visit_purpose": "celebration",
            "preferences": "steak", "occasion": "anniversary", "visit_duration": "3 nights",
            "party_composition": "2 adults", "dietary": "gluten-free", "gaming": "blackjack",
            "entertainment": "comedy shows", "spa": "massage", "occasion_details": "10th",
            "home_market": "NYC", "budget_signal": "high", "loyalty_tier": "gold",
            "visit_frequency": "monthly",
        }
        result = _calculate_profile_completeness_weighted(all_fields)
        assert result <= 1.0


class TestProfilingPhase:
    """Test _determine_profiling_phase golden path."""

    def test_empty_fields_is_foundation(self):
        """No fields = foundation phase."""
        assert _determine_profiling_phase({}, 0.0) == "foundation"

    def test_name_and_purpose_is_preference(self):
        """Name + party_size (2+ foundation fields) = preference phase."""
        fields = {"name": "Mike", "party_size": "4", "visit_purpose": "leisure"}
        result = _determine_profiling_phase(fields, 0.37)
        assert result == "preference"

    def test_with_dining_pref_is_relationship(self):
        """Foundation + preference field = relationship phase."""
        fields = {
            "name": "Mike", "party_size": "4", "visit_purpose": "leisure",
            "preferences": "steak",
        }
        result = _determine_profiling_phase(fields, 0.47)
        assert result == "relationship"

    def test_full_profile_is_behavioral(self):
        """Foundation + preference + relationship = behavioral phase."""
        fields = {
            "name": "Mike", "party_size": "4", "visit_purpose": "leisure",
            "preferences": "steak", "occasion": "anniversary",
        }
        result = _determine_profiling_phase(fields, 0.57)
        assert result == "behavioral"


# ---------------------------------------------------------------------------
# Instrumentation logging tests
# ---------------------------------------------------------------------------


class TestProfilingExtractionLogging:
    """Test that extraction results are logged."""

    @pytest.mark.asyncio
    async def test_logs_extraction_results(self, caplog):
        """profiling_enrichment_node should log filled/total field counts."""
        state = _profiling_state()
        mock_result = _mock_extraction_result(
            guest_name="Mike",
            party_size="4",
            visit_purpose="celebration",
        )

        with caplog.at_level(logging.DEBUG, logger="src.agent.profiling"):
            with _patch_whisper_llm(mock_result):
                result = await profiling_enrichment_node(state)

        extraction_logs = [r for r in caplog.records if "Profiling extraction" in r.message]
        assert len(extraction_logs) >= 1, (
            f"Expected 'Profiling extraction' log, got: {[r.message for r in caplog.records]}"
        )
        log_msg = extraction_logs[0].message
        assert "filled=3/" in log_msg  # 3 fields filled

    @pytest.mark.asyncio
    async def test_logs_profile_state(self, caplog):
        """profiling_enrichment_node should log phase and completeness."""
        state = _profiling_state()
        mock_result = _mock_extraction_result(guest_name="Mike")

        with caplog.at_level(logging.DEBUG, logger="src.agent.profiling"):
            with _patch_whisper_llm(mock_result):
                result = await profiling_enrichment_node(state)

        state_logs = [r for r in caplog.records if "Profiling state" in r.message]
        assert len(state_logs) >= 1, (
            f"Expected 'Profiling state' log, got: {[r.message for r in caplog.records]}"
        )
        log_msg = state_logs[0].message
        assert "phase=" in log_msg
        assert "completeness=" in log_msg

    @pytest.mark.asyncio
    async def test_logs_zero_extraction(self, caplog):
        """When no fields extracted, should log filled=0."""
        state = _profiling_state()
        mock_result = _mock_extraction_result()  # All None

        with caplog.at_level(logging.DEBUG, logger="src.agent.profiling"):
            with _patch_whisper_llm(mock_result):
                result = await profiling_enrichment_node(state)

        extraction_logs = [r for r in caplog.records if "Profiling extraction" in r.message]
        assert len(extraction_logs) >= 1
        assert "filled=0/" in extraction_logs[0].message


class TestProfilingQuestionLogging:
    """Test that question injection decisions are logged."""

    @pytest.mark.asyncio
    async def test_logs_question_candidate(self, caplog):
        """When whisper_plan has a question, it should be logged."""
        state = _profiling_state(
            whisper_plan={
                "next_profiling_question": "How many will be joining you for dinner?",
                "question_technique": "give_to_get",
            },
        )
        mock_result = _mock_extraction_result(guest_name="Mike")

        with caplog.at_level(logging.DEBUG, logger="src.agent.profiling"):
            with _patch_whisper_llm(mock_result):
                result = await profiling_enrichment_node(state)

        question_logs = [r for r in caplog.records if "Profiling question" in r.message]
        assert len(question_logs) >= 1, (
            f"Expected 'Profiling question' log, got: {[r.message for r in caplog.records]}"
        )
        log_msg = question_logs[0].message
        assert "technique=give_to_get" in log_msg

    @pytest.mark.asyncio
    async def test_logs_sentiment_in_question_decision(self, caplog):
        """Question log should include guest sentiment."""
        state = _profiling_state(
            guest_sentiment="positive",
            whisper_plan={
                "next_profiling_question": "Are you celebrating something special?",
                "question_technique": "assumptive_bridge",
            },
        )
        mock_result = _mock_extraction_result(guest_name="Mike")

        with caplog.at_level(logging.DEBUG, logger="src.agent.profiling"):
            with _patch_whisper_llm(mock_result):
                result = await profiling_enrichment_node(state)

        question_logs = [r for r in caplog.records if "Profiling question" in r.message]
        assert len(question_logs) >= 1
        log_msg = question_logs[0].message
        assert "sentiment=positive" in log_msg

    @pytest.mark.asyncio
    async def test_logs_no_whisper_plan(self, caplog):
        """When no whisper_plan, should log reason=no_whisper_plan."""
        state = _profiling_state(whisper_plan=None)
        mock_result = _mock_extraction_result()

        with caplog.at_level(logging.DEBUG, logger="src.agent.profiling"):
            with _patch_whisper_llm(mock_result):
                result = await profiling_enrichment_node(state)

        skip_logs = [r for r in caplog.records if "injected=False" in r.message]
        assert len(skip_logs) >= 1, (
            f"Expected 'injected=False' log, got: {[r.message for r in caplog.records]}"
        )
        assert "no_whisper_plan" in skip_logs[0].message

    @pytest.mark.asyncio
    async def test_logs_technique_none_reason(self, caplog):
        """When technique is 'none', should log reason=technique_none."""
        state = _profiling_state(
            whisper_plan={
                "next_profiling_question": "Some question",
                "question_technique": "none",
            },
        )
        mock_result = _mock_extraction_result()

        with caplog.at_level(logging.DEBUG, logger="src.agent.profiling"):
            with _patch_whisper_llm(mock_result):
                result = await profiling_enrichment_node(state)

        skip_logs = [r for r in caplog.records if "injected=False" in r.message]
        assert len(skip_logs) >= 1
        assert "technique_none" in skip_logs[0].message


class TestProfilingSkipLogging:
    """Test that skip conditions are logged."""

    @pytest.mark.asyncio
    async def test_logs_skip_no_user_message(self, caplog):
        """When no user message, should log skip with reason."""
        state = _profiling_state(
            messages=[AIMessage(content="Hello!")],  # No HumanMessage
        )

        with caplog.at_level(logging.DEBUG, logger="src.agent.profiling"):
            result = await profiling_enrichment_node(state)

        skip_logs = [r for r in caplog.records if "Profiling skip" in r.message]
        assert len(skip_logs) >= 1, (
            f"Expected 'Profiling skip' log, got: {[r.message for r in caplog.records]}"
        )
        assert result["profiling_question_injected"] is False


class TestProfilingFailSilent:
    """Test that profiling node fails silently with logging."""

    @pytest.mark.asyncio
    async def test_llm_error_returns_safe_defaults(self, caplog):
        """When LLM fails, should return safe defaults and log warning."""
        state = _profiling_state()

        async def _raise_llm():
            raise RuntimeError("LLM unavailable")

        with caplog.at_level(logging.DEBUG, logger="src.agent.profiling"):
            with patch(_WHISPER_LLM_PATCH, new=_raise_llm):
                result = await profiling_enrichment_node(state)

        # Should return safe defaults
        assert result["profiling_question_injected"] is False
        assert "profiling_phase" in result
        assert "profile_completeness_score" in result

        # Should log the failure
        warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_logs) >= 1

    @pytest.mark.asyncio
    async def test_extraction_error_doesnt_crash(self):
        """Exception during extraction should not propagate."""
        state = _profiling_state()

        async def _broken_llm():
            mock = MagicMock()
            mock.with_structured_output.side_effect = ValueError("Schema error")
            return mock

        with patch(_WHISPER_LLM_PATCH, new=_broken_llm):
            result = await profiling_enrichment_node(state)

        # Should not raise, should return safe defaults
        assert isinstance(result, dict)
        assert result["profiling_question_injected"] is False


class TestProfilingQuestionInjection:
    """Test that profiling question injection works end-to-end."""

    @pytest.mark.asyncio
    async def test_question_injected_when_not_in_response(self):
        """When LLM response does not contain the question, it should be injected."""
        ai_msg = AIMessage(content="Welcome Mike!", id="msg-1")
        state = _profiling_state(
            messages=[
                HumanMessage(content="Hi, I'm Mike"),
                ai_msg,
            ],
            whisper_plan={
                "next_profiling_question": "How many will be joining you tonight?",
                "question_technique": "give_to_get",
            },
        )
        mock_result = _mock_extraction_result(guest_name="Mike")

        with _patch_whisper_llm(mock_result):
            result = await profiling_enrichment_node(state)

        assert result["profiling_question_injected"] is True
        # Replacement message should contain the question
        if "messages" in result:
            enriched = result["messages"][0].content
            assert "How many will be joining you tonight?" in enriched

    @pytest.mark.asyncio
    async def test_question_not_injected_when_already_present(self):
        """When response already contains the question, injection is skipped."""
        ai_msg = AIMessage(
            content="Welcome Mike! How many will be joining you tonight?",
            id="msg-1",
        )
        state = _profiling_state(
            messages=[
                HumanMessage(content="Hi, I'm Mike"),
                ai_msg,
            ],
            whisper_plan={
                "next_profiling_question": "How many will be joining you tonight?",
                "question_technique": "give_to_get",
            },
        )
        mock_result = _mock_extraction_result(guest_name="Mike")

        with _patch_whisper_llm(mock_result):
            result = await profiling_enrichment_node(state)

        # Should be marked as injected (from system prompt path)
        assert result["profiling_question_injected"] is True
        # But no message replacement needed
        assert "messages" not in result or result.get("messages") is None

    @pytest.mark.asyncio
    async def test_technique_none_skips_injection(self):
        """When technique is 'none', no question should be injected."""
        state = _profiling_state(
            whisper_plan={
                "next_profiling_question": "Some question",
                "question_technique": "none",
            },
        )
        mock_result = _mock_extraction_result(guest_name="Mike")

        with _patch_whisper_llm(mock_result):
            result = await profiling_enrichment_node(state)

        assert result["profiling_question_injected"] is False
