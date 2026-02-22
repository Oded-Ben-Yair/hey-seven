"""Tests for LLM-as-judge evaluation metrics."""

from __future__ import annotations

import json

import pytest

from src.observability.llm_judge import (
    ALL_METRICS,
    METRIC_CONVERSATION_FLOW,
    METRIC_CULTURAL_SENSITIVITY,
    METRIC_EMPATHY,
    METRIC_GUEST_EXPERIENCE,
    METRIC_PERSONA_CONSISTENCY,
    ConversationEvalReport,
    ConversationEvalScore,
    ConversationTestCase,
    evaluate_conversation,
    run_conversation_evaluation,
)


# ---------------------------------------------------------------------------
# Casino domain test data
# ---------------------------------------------------------------------------

_WARM_GREETING_RESPONSE = (
    "Welcome to Mohegan Sun! I'd be happy to help you with anything "
    "during your visit. Whether you're looking for dining recommendations, "
    "entertainment options, or information about our hotel rooms, I'm here "
    "to assist. As one of our valued guests, please don't hesitate to ask."
)

_COLD_RESPONSE = "The steakhouse is on floor 2. Hours are 5-10."

_EMPATHETIC_RESPONSE = (
    "I completely understand your frustration, and I'm sorry to hear about "
    "your experience. Let me help you find the best option. Our guest "
    "services team at Mohegan Sun is available to assist you directly at "
    "1-888-226-7711. I appreciate your patience."
)

_DINING_RECOMMENDATION = (
    "Excellent choice! Todd English's Tuscany is one of our most popular "
    "restaurants at Mohegan Sun. Guests love the authentic Italian cuisine "
    "and the warm atmosphere. It's located on the Casino of the Earth floor "
    "and is open from 5 PM to 10 PM. I'd recommend making a reservation, "
    "especially for weekend dining. Would you also like to know about our "
    "other dining options?"
)

_EMOJI_RESPONSE = (
    "Welcome!! \U0001f600 OMG you're going to LOVE our buffet!! \U0001f389 "
    "It's like, so amazing bruh, lol!"
)

_CULTURALLY_INSENSITIVE = (
    "Obviously you people would want the buffet. That's strange that you "
    "would ask about fine dining."
)

_FRUSTRATED_USER_MESSAGES = [
    {"role": "user", "content": "I've been waiting for 30 minutes for my room. I'm very frustrated."},
]

_DINING_MESSAGES = [
    {"role": "user", "content": "What restaurants do you have at Mohegan Sun?"},
]

_MULTI_TURN_MESSAGES = [
    {"role": "user", "content": "Hi, what restaurants do you have?"},
    {"role": "assistant", "content": "Welcome! We have several dining options including Todd English's Tuscany."},
    {"role": "user", "content": "What about steakhouse options?"},
]


# ---------------------------------------------------------------------------
# TestMetricScoreRange
# ---------------------------------------------------------------------------


class TestMetricScoreRange:
    """Verify each metric returns a float in [0.0, 1.0]."""

    def test_empathy_returns_valid_range(self):
        """Empathy metric returns 0.0-1.0 for valid input."""
        score = evaluate_conversation(_DINING_MESSAGES, _WARM_GREETING_RESPONSE)
        assert 0.0 <= score.empathy <= 1.0

    def test_cultural_sensitivity_returns_valid_range(self):
        """Cultural sensitivity metric returns 0.0-1.0 for valid input."""
        score = evaluate_conversation(_DINING_MESSAGES, _WARM_GREETING_RESPONSE)
        assert 0.0 <= score.cultural_sensitivity <= 1.0

    def test_conversation_flow_returns_valid_range(self):
        """Conversation flow metric returns 0.0-1.0 for valid input."""
        score = evaluate_conversation(_DINING_MESSAGES, _WARM_GREETING_RESPONSE)
        assert 0.0 <= score.conversation_flow <= 1.0

    def test_persona_consistency_returns_valid_range(self):
        """Persona consistency metric returns 0.0-1.0 for valid input."""
        score = evaluate_conversation(_DINING_MESSAGES, _WARM_GREETING_RESPONSE)
        assert 0.0 <= score.persona_consistency <= 1.0

    def test_guest_experience_returns_valid_range(self):
        """Guest experience metric returns 0.0-1.0 for valid input."""
        score = evaluate_conversation(_DINING_MESSAGES, _WARM_GREETING_RESPONSE)
        assert 0.0 <= score.guest_experience <= 1.0


# ---------------------------------------------------------------------------
# TestFailSilent
# ---------------------------------------------------------------------------


class TestFailSilent:
    """Verify metrics return a score (not crash) for empty/None inputs."""

    def test_empty_response(self):
        """Empty response returns scores without crashing."""
        score = evaluate_conversation(_DINING_MESSAGES, "")
        assert score.empathy == 0.0
        assert score.cultural_sensitivity == 0.0
        assert score.conversation_flow == 0.0
        assert score.persona_consistency == 0.0
        assert score.guest_experience == 0.0

    def test_none_response(self):
        """None response returns scores without crashing."""
        score = evaluate_conversation(_DINING_MESSAGES, None)
        assert score.empathy == 0.0
        assert score.guest_experience == 0.0

    def test_empty_messages(self):
        """Empty message list returns scores without crashing."""
        score = evaluate_conversation([], _WARM_GREETING_RESPONSE)
        assert 0.0 <= score.empathy <= 1.0
        assert 0.0 <= score.guest_experience <= 1.0

    def test_none_messages(self):
        """None message list returns scores without crashing."""
        score = evaluate_conversation(None, _WARM_GREETING_RESPONSE)
        assert 0.0 <= score.empathy <= 1.0

    def test_whitespace_only_response(self):
        """Whitespace-only response returns zero scores."""
        score = evaluate_conversation(_DINING_MESSAGES, "   \n  ")
        assert score.empathy == 0.0
        assert score.guest_experience == 0.0


# ---------------------------------------------------------------------------
# TestOfflineMode
# ---------------------------------------------------------------------------


class TestOfflineMode:
    """Verify offline mode works without API keys."""

    def test_offline_mode_default(self):
        """Default mode is offline (no API keys needed)."""
        score = evaluate_conversation(_DINING_MESSAGES, _WARM_GREETING_RESPONSE)
        assert score.details.get("mode") == "offline"

    def test_offline_scores_all_metrics(self):
        """Offline mode returns scores for all 5 metrics."""
        score = evaluate_conversation(_DINING_MESSAGES, _DINING_RECOMMENDATION)
        score_dict = score.to_dict()
        for metric in ALL_METRICS:
            assert metric in score_dict, f"Missing metric: {metric}"
            assert score_dict[metric] > 0.0, f"Zero score for {metric}"


# ---------------------------------------------------------------------------
# TestEvaluateConversation
# ---------------------------------------------------------------------------


class TestEvaluateConversation:
    """Test the evaluate_conversation() function."""

    def test_returns_conversation_eval_score(self):
        """evaluate_conversation returns ConversationEvalScore instance."""
        result = evaluate_conversation(_DINING_MESSAGES, _DINING_RECOMMENDATION)
        assert isinstance(result, ConversationEvalScore)

    def test_scores_all_five_metrics(self):
        """All 5 metrics are populated in the result."""
        result = evaluate_conversation(_DINING_MESSAGES, _DINING_RECOMMENDATION)
        assert result.empathy > 0.0
        assert result.cultural_sensitivity > 0.0
        assert result.conversation_flow > 0.0
        assert result.persona_consistency > 0.0
        assert result.guest_experience > 0.0

    def test_selective_metrics(self):
        """Can evaluate only specific metrics."""
        result = evaluate_conversation(
            _DINING_MESSAGES,
            _DINING_RECOMMENDATION,
            metrics=[METRIC_EMPATHY, METRIC_PERSONA_CONSISTENCY],
        )
        assert result.empathy > 0.0
        assert result.persona_consistency > 0.0
        # Non-requested metrics should be 0.0 (not evaluated)
        assert result.conversation_flow == 0.0
        assert result.cultural_sensitivity == 0.0

    def test_to_dict_serialization(self):
        """ConversationEvalScore.to_dict() returns serializable dict."""
        result = evaluate_conversation(_DINING_MESSAGES, _DINING_RECOMMENDATION)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 5
        # Verify JSON-serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# TestConversationTestCase
# ---------------------------------------------------------------------------


class TestConversationTestCase:
    """Test ConversationTestCase creation and serialization."""

    def test_create_test_case(self):
        """ConversationTestCase can be created with required fields."""
        case = ConversationTestCase(
            id="dining-multi-01",
            name="Dining recommendation flow",
            category="dining",
            turns=[
                {"role": "user", "content": "What restaurants do you have?"},
                {
                    "role": "assistant",
                    "content": "We have several dining options at Mohegan Sun.",
                    "expected_keywords": ["restaurant", "dining"],
                    "expected_tone": "professional",
                },
            ],
            expected_behavior="Lists dining options warmly",
            expected_empathy_level="medium",
        )
        assert case.id == "dining-multi-01"
        assert case.name == "Dining recommendation flow"
        assert case.category == "dining"
        assert len(case.turns) == 2

    def test_default_empathy_level(self):
        """Default expected_empathy_level is 'medium'."""
        case = ConversationTestCase(
            id="test-01",
            name="Test case",
            category="test",
            turns=[],
        )
        assert case.expected_empathy_level == "medium"

    def test_test_case_json_serializable(self):
        """ConversationTestCase can be serialized to JSON via __dict__."""
        case = ConversationTestCase(
            id="test-01",
            name="Test case",
            category="test",
            turns=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Welcome!"},
            ],
            expected_behavior="Greets warmly",
            expected_empathy_level="low",
        )
        # dataclass.__dict__ should be JSON-serializable
        serialized = json.dumps(case.__dict__)
        deserialized = json.loads(serialized)
        assert deserialized["id"] == "test-01"
        assert deserialized["expected_empathy_level"] == "low"


# ---------------------------------------------------------------------------
# TestCasinoDomainContent
# ---------------------------------------------------------------------------


class TestCasinoDomainContent:
    """Test with real casino-domain content."""

    def test_warm_greeting_scores_high_empathy(self):
        """Warm greeting response scores higher on empathy than cold response."""
        warm_score = evaluate_conversation(
            [{"role": "user", "content": "Hello!"}],
            _WARM_GREETING_RESPONSE,
        )
        cold_score = evaluate_conversation(
            [{"role": "user", "content": "Hello!"}],
            _COLD_RESPONSE,
        )
        assert warm_score.empathy > cold_score.empathy

    def test_empathetic_response_scores_high_with_frustrated_guest(self):
        """Empathetic response scores high when user is frustrated."""
        score = evaluate_conversation(
            _FRUSTRATED_USER_MESSAGES,
            _EMPATHETIC_RESPONSE,
        )
        assert score.empathy >= 0.6

    def test_dining_recommendation_scores_high_flow(self):
        """Dining recommendation has good conversation flow."""
        score = evaluate_conversation(
            _DINING_MESSAGES,
            _DINING_RECOMMENDATION,
        )
        assert score.conversation_flow >= 0.4

    def test_dining_recommendation_scores_high_persona(self):
        """Dining recommendation follows Seven persona."""
        score = evaluate_conversation(
            _DINING_MESSAGES,
            _DINING_RECOMMENDATION,
        )
        assert score.persona_consistency >= 0.5

    def test_multi_turn_conversation_flow(self):
        """Multi-turn conversation scores on flow."""
        response = (
            "We have Bobby's Burgers as a casual option, and for a premium "
            "steakhouse experience, I'd recommend Mohegan Sun's own "
            "Big Bubba's BBQ. Would you like more details on either?"
        )
        score = evaluate_conversation(_MULTI_TURN_MESSAGES, response)
        assert score.conversation_flow > 0.0
        assert score.guest_experience > 0.0


# ---------------------------------------------------------------------------
# TestEmpathyComparison
# ---------------------------------------------------------------------------


class TestEmpathyComparison:
    """Test that empathetic responses score higher than cold responses."""

    def test_empathetic_vs_cold(self):
        """Empathetic response scores higher than cold response on empathy metric."""
        empathetic_score = evaluate_conversation(
            _FRUSTRATED_USER_MESSAGES,
            _EMPATHETIC_RESPONSE,
        )
        cold_score = evaluate_conversation(
            _FRUSTRATED_USER_MESSAGES,
            _COLD_RESPONSE,
        )
        assert empathetic_score.empathy > cold_score.empathy


# ---------------------------------------------------------------------------
# TestPersonaConsistency
# ---------------------------------------------------------------------------


class TestPersonaConsistency:
    """Test persona consistency scoring."""

    def test_emoji_lowers_persona_score(self):
        """Response with emoji scores lower on persona consistency."""
        clean_score = evaluate_conversation(
            _DINING_MESSAGES,
            _DINING_RECOMMENDATION,
        )
        emoji_score = evaluate_conversation(
            _DINING_MESSAGES,
            _EMOJI_RESPONSE,
        )
        assert clean_score.persona_consistency > emoji_score.persona_consistency

    def test_professional_response_high_persona(self):
        """Professional, VIP-oriented response scores high on persona."""
        response = (
            "Welcome to Mohegan Sun! As one of our valued guests, I'm "
            "delighted to help you explore our property. Our resort offers "
            "world-class dining, entertainment, and spa services."
        )
        score = evaluate_conversation(
            [{"role": "user", "content": "Tell me about your resort."}],
            response,
        )
        assert score.persona_consistency >= 0.5

    def test_informal_slang_lowers_persona(self):
        """Informal slang lowers persona consistency score."""
        formal_score = evaluate_conversation(
            _DINING_MESSAGES,
            _DINING_RECOMMENDATION,
        )
        slang_response = "Yo bruh the buffet is lit lol, dude you gotta check it out"
        slang_score = evaluate_conversation(
            _DINING_MESSAGES,
            slang_response,
        )
        assert formal_score.persona_consistency > slang_score.persona_consistency


# ---------------------------------------------------------------------------
# TestCulturalSensitivity
# ---------------------------------------------------------------------------


class TestCulturalSensitivity:
    """Test cultural sensitivity scoring."""

    def test_insensitive_language_penalized(self):
        """Culturally insensitive language lowers score."""
        respectful_score = evaluate_conversation(
            _DINING_MESSAGES,
            _DINING_RECOMMENDATION,
        )
        insensitive_score = evaluate_conversation(
            _DINING_MESSAGES,
            _CULTURALLY_INSENSITIVE,
        )
        assert respectful_score.cultural_sensitivity > insensitive_score.cultural_sensitivity


# ---------------------------------------------------------------------------
# TestRunConversationEvaluation
# ---------------------------------------------------------------------------


class TestRunConversationEvaluation:
    """Test the run_conversation_evaluation() function."""

    def test_empty_list(self):
        """Empty conversation list returns zeroed report."""
        report = run_conversation_evaluation([])
        assert isinstance(report, ConversationEvalReport)
        assert report.total_cases == 0
        assert report.avg_empathy == 0.0

    def test_single_conversation(self):
        """Single conversation produces valid report."""
        case = ConversationTestCase(
            id="test-01",
            name="Dining greeting",
            category="dining",
            turns=[
                {"role": "user", "content": "What restaurants do you have at Mohegan Sun?"},
                {"role": "assistant", "content": _DINING_RECOMMENDATION},
            ],
            expected_behavior="Lists dining options",
        )
        report = run_conversation_evaluation([case])
        assert report.total_cases == 1
        assert report.avg_empathy > 0.0
        assert report.avg_guest_experience > 0.0
        assert len(report.scores) == 1

    def test_multiple_conversations(self):
        """Multiple conversations produce valid aggregate report."""
        cases = [
            ConversationTestCase(
                id="test-01",
                name="Warm greeting",
                category="greeting",
                turns=[
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": _WARM_GREETING_RESPONSE},
                ],
            ),
            ConversationTestCase(
                id="test-02",
                name="Frustrated guest",
                category="complaint",
                turns=[
                    {"role": "user", "content": "I've been waiting 30 minutes. I'm frustrated."},
                    {"role": "assistant", "content": _EMPATHETIC_RESPONSE},
                ],
                expected_empathy_level="high",
            ),
            ConversationTestCase(
                id="test-03",
                name="Dining recommendation",
                category="dining",
                turns=[
                    {"role": "user", "content": "Where should I eat at Mohegan Sun?"},
                    {"role": "assistant", "content": _DINING_RECOMMENDATION},
                ],
            ),
        ]
        report = run_conversation_evaluation(cases)
        assert report.total_cases == 3
        assert len(report.scores) == 3
        assert report.avg_empathy > 0.0
        assert report.avg_guest_experience > 0.0

    def test_report_to_dict(self):
        """ConversationEvalReport serializes correctly."""
        cases = [
            ConversationTestCase(
                id="test-01",
                name="Simple test",
                category="test",
                turns=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": _WARM_GREETING_RESPONSE},
                ],
            ),
        ]
        report = run_conversation_evaluation(cases)
        d = report.to_dict()
        assert "total_cases" in d
        assert "avg_empathy" in d
        assert "avg_guest_experience" in d
        # JSON serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# TestGoldenTestCaseNewFields
# ---------------------------------------------------------------------------


class TestGoldenTestCaseNewFields:
    """Verify the new fields added to GoldenTestCase."""

    def test_expected_tone_default(self):
        """GoldenTestCase.expected_tone defaults to empty string."""
        from src.observability.evaluation import GoldenTestCase

        case = GoldenTestCase(
            id="test",
            category="test",
            query="test",
            expected_keywords=["test"],
        )
        assert case.expected_tone == ""

    def test_expected_empathy_level_default(self):
        """GoldenTestCase.expected_empathy_level defaults to empty string."""
        from src.observability.evaluation import GoldenTestCase

        case = GoldenTestCase(
            id="test",
            category="test",
            query="test",
            expected_keywords=["test"],
        )
        assert case.expected_empathy_level == ""

    def test_existing_golden_dataset_unaffected(self):
        """Existing GOLDEN_DATASET cases still load correctly with new fields."""
        from src.observability.evaluation import GOLDEN_DATASET

        assert len(GOLDEN_DATASET) == 20
        for case in GOLDEN_DATASET:
            assert case.expected_tone == ""
            assert case.expected_empathy_level == ""
