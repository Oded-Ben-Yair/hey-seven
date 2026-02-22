"""LLM-as-judge live evaluation tests.

These tests require GOOGLE_API_KEY and are skipped in CI without it.
Pydantic model tests run without any API key.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from src.observability.llm_judge import (
    ConversationEvalScore,
    GOLDEN_CONVERSATIONS,
    LLMJudgeDimension,
    LLMJudgeOutput,
    evaluate_conversation,
    evaluate_conversation_llm,
)


# ---------------------------------------------------------------------------
# 1. Pydantic Model Tests (no API key required)
# ---------------------------------------------------------------------------


class TestLLMJudgeModels:
    """Validate Pydantic models for LLM judge structured output."""

    def test_dimension_valid_scores(self):
        """LLMJudgeDimension accepts valid 1-10 score."""
        dim = LLMJudgeDimension(score=7, justification="Good response")
        assert dim.score == 7
        assert dim.justification == "Good response"

    def test_dimension_boundary_low(self):
        """LLMJudgeDimension accepts minimum score of 1."""
        dim = LLMJudgeDimension(score=1, justification="Poor")
        assert dim.score == 1

    def test_dimension_boundary_high(self):
        """LLMJudgeDimension accepts maximum score of 10."""
        dim = LLMJudgeDimension(score=10, justification="Perfect")
        assert dim.score == 10

    def test_dimension_rejects_score_too_high(self):
        """LLMJudgeDimension rejects score > 10."""
        with pytest.raises(Exception):
            LLMJudgeDimension(score=11, justification="Too high")

    def test_dimension_rejects_score_too_low(self):
        """LLMJudgeDimension rejects score < 1."""
        with pytest.raises(Exception):
            LLMJudgeDimension(score=0, justification="Too low")

    def test_full_output_model(self):
        """LLMJudgeOutput accepts all five dimensions."""
        output = LLMJudgeOutput(
            groundedness=LLMJudgeDimension(score=8, justification="Well grounded"),
            persona_fidelity=LLMJudgeDimension(score=9, justification="Strong persona"),
            safety=LLMJudgeDimension(score=10, justification="No violations"),
            contextual_relevance=LLMJudgeDimension(score=7, justification="Mostly relevant"),
            proactive_value=LLMJudgeDimension(score=6, justification="Some suggestions"),
        )
        assert output.groundedness.score == 8
        assert output.safety.score == 10
        assert output.proactive_value.justification == "Some suggestions"

    def test_output_model_json_serializable(self):
        """LLMJudgeOutput serializes to JSON without error."""
        output = LLMJudgeOutput(
            groundedness=LLMJudgeDimension(score=5, justification="Mid"),
            persona_fidelity=LLMJudgeDimension(score=5, justification="Mid"),
            safety=LLMJudgeDimension(score=5, justification="Mid"),
            contextual_relevance=LLMJudgeDimension(score=5, justification="Mid"),
            proactive_value=LLMJudgeDimension(score=5, justification="Mid"),
        )
        json_str = output.model_dump_json()
        assert "groundedness" in json_str
        assert "persona_fidelity" in json_str


# ---------------------------------------------------------------------------
# 2. Offline Fallback Tests (no API key required)
# ---------------------------------------------------------------------------


class TestOfflineFallbackStillWorks:
    """Verify offline scoring is unchanged by LLM judge additions."""

    def test_offline_scoring_unchanged(self):
        """Offline scoring returns valid ConversationEvalScore."""
        messages = [{"role": "user", "content": "What restaurants are open?"}]
        response = (
            "We have several wonderful dining options. "
            "I'd be happy to help you find the perfect restaurant."
        )
        score = evaluate_conversation(messages, response)
        assert isinstance(score, ConversationEvalScore)
        assert 0.0 <= score.empathy <= 1.0
        assert 0.0 <= score.cultural_sensitivity <= 1.0
        assert 0.0 <= score.conversation_flow <= 1.0
        assert 0.0 <= score.persona_consistency <= 1.0
        assert 0.0 <= score.guest_experience <= 1.0
        assert score.details.get("mode") in ("offline", "offline_fallback")

    def test_offline_mode_is_default(self):
        """Default mode (no EVAL_LLM_ENABLED) is offline."""
        messages = [{"role": "user", "content": "Hello"}]
        response = "Welcome to Mohegan Sun!"
        score = evaluate_conversation(messages, response)
        assert score.details.get("mode") == "offline"

    @pytest.mark.asyncio
    async def test_concurrent_evaluations_complete(self):
        """Multiple concurrent evaluate_conversation_llm calls complete without deadlock."""
        messages = [{"role": "user", "content": "What restaurants are open?"}]
        response = "We have several wonderful dining options at Mohegan Sun."

        # Launch 10 concurrent evaluations (semaphore limit is 5)
        tasks = [
            evaluate_conversation_llm(messages, response)
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for score in results:
            assert isinstance(score, ConversationEvalScore)
            assert 0.0 <= score.empathy <= 1.0


# ---------------------------------------------------------------------------
# 3. LLM Judge Fallback Test (no API key -- verifies graceful degradation)
# ---------------------------------------------------------------------------


class TestLLMJudgeFallback:
    """Verify LLM judge falls back to offline when API key is missing."""

    @pytest.mark.asyncio
    async def test_llm_judge_falls_back_without_api_key(self, monkeypatch):
        """evaluate_conversation_llm falls back to offline without GOOGLE_API_KEY."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        messages = [{"role": "user", "content": "What restaurants do you have?"}]
        response = (
            "We have several wonderful dining options at Mohegan Sun! "
            "Todd English's Tuscany offers upscale Italian."
        )
        score = await evaluate_conversation_llm(messages, response)
        assert isinstance(score, ConversationEvalScore)
        assert 0.0 <= score.empathy <= 1.0
        # Falls back to offline mode
        assert score.details.get("mode") in ("offline", "offline_fallback")


# ---------------------------------------------------------------------------
# 4. Live LLM Judge Tests (require GOOGLE_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="Requires GOOGLE_API_KEY for live LLM evaluation",
)
class TestLLMJudgeLive:
    """Live tests against actual Gemini API."""

    @pytest.mark.asyncio
    async def test_llm_judge_returns_valid_scores(self):
        """LLM judge returns scores in 0.0-1.0 range for all metrics."""
        messages = [
            {"role": "user", "content": "What restaurants do you have?"},
        ]
        response = (
            "We have several wonderful dining options at Mohegan Sun! "
            "Todd English's Tuscany offers upscale Italian, Bobby's Burgers "
            "is great for casual dining. What cuisine are you in the mood for?"
        )
        score = await evaluate_conversation_llm(messages, response)
        assert isinstance(score, ConversationEvalScore)
        assert 0.0 <= score.empathy <= 1.0
        assert 0.0 <= score.cultural_sensitivity <= 1.0
        assert 0.0 <= score.conversation_flow <= 1.0
        assert 0.0 <= score.persona_consistency <= 1.0
        assert 0.0 <= score.guest_experience <= 1.0
        assert score.details.get("mode") == "llm_judge"

    @pytest.mark.asyncio
    async def test_llm_judge_returns_justifications(self):
        """LLM judge returns non-empty justifications for each dimension."""
        messages = [
            {"role": "user", "content": "I can't stop gambling, I need help"},
        ]
        response = (
            "I appreciate you sharing that with me. Your wellbeing matters most. "
            "Please reach out to the National Problem Gambling Helpline at "
            "1-800-MY-RESET for free, confidential support 24/7."
        )
        score = await evaluate_conversation_llm(messages, response)
        assert score.details.get("mode") == "llm_judge"
        for dimension in ("groundedness", "persona_fidelity", "safety",
                          "contextual_relevance", "proactive_value"):
            dim_data = score.details.get(dimension, {})
            assert isinstance(dim_data.get("score"), int), f"Missing score for {dimension}"
            assert 1 <= dim_data["score"] <= 10, f"Score out of range for {dimension}"
            assert dim_data.get("justification"), f"Empty justification for {dimension}"
