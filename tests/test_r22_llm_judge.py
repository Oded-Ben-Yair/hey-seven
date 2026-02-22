"""R22 LLM-as-Judge evaluation tests: multi-turn conversations, regression detection.

Tests the R22 evaluation enhancements:
1. Multi-turn golden conversation evaluation
2. Regression detection against quality baselines
3. Individual scoring metric validation
4. R21 feature coverage (frustration handling, proactive suggestions, persona drift)
"""

import pytest

from src.observability.llm_judge import (
    ALL_METRICS,
    GOLDEN_CONVERSATIONS,
    METRIC_CONVERSATION_FLOW,
    METRIC_CULTURAL_SENSITIVITY,
    METRIC_EMPATHY,
    METRIC_GUEST_EXPERIENCE,
    METRIC_PERSONA_CONSISTENCY,
    QUALITY_BASELINE,
    ConversationEvalReport,
    ConversationEvalScore,
    ConversationTestCase,
    detect_regression,
    evaluate_conversation,
    run_conversation_evaluation,
)


# ---------------------------------------------------------------------------
# 1. Golden Conversation Dataset Tests
# ---------------------------------------------------------------------------


class TestGoldenConversations:
    """Validate the golden conversation dataset is well-formed."""

    def test_golden_dataset_exists(self):
        """Dataset has at least 5 conversations."""
        assert len(GOLDEN_CONVERSATIONS) >= 5

    def test_all_cases_have_required_fields(self):
        """Every test case has id, name, category, and turns."""
        for case in GOLDEN_CONVERSATIONS:
            assert case.id, f"Missing id for case: {case.name}"
            assert case.name, f"Missing name for case: {case.id}"
            assert case.category, f"Missing category for case: {case.id}"
            assert len(case.turns) >= 2, f"Need at least 2 turns: {case.id}"

    def test_all_cases_have_assistant_response(self):
        """Every conversation ends with or contains an assistant turn."""
        for case in GOLDEN_CONVERSATIONS:
            has_assistant = any(
                t.get("role") == "assistant" for t in case.turns
            )
            assert has_assistant, f"No assistant turn in: {case.id}"

    def test_unique_ids(self):
        """All case IDs are unique."""
        ids = [case.id for case in GOLDEN_CONVERSATIONS]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"

    def test_category_coverage(self):
        """Dataset covers key categories."""
        categories = {case.category for case in GOLDEN_CONVERSATIONS}
        # Must cover at least these core categories
        assert "dining" in categories
        assert "complaint" in categories or "safety" in categories
        assert "persona" in categories or "retention" in categories

    def test_frustrated_scenario_exists(self):
        """At least one conversation tests frustrated guest handling."""
        frustrated = [c for c in GOLDEN_CONVERSATIONS if c.category == "complaint"]
        assert len(frustrated) >= 1

    def test_proactive_scenario_exists(self):
        """At least one conversation tests proactive suggestion."""
        proactive = [c for c in GOLDEN_CONVERSATIONS if c.category == "proactive"]
        assert len(proactive) >= 1

    def test_safety_scenario_exists(self):
        """At least one conversation tests responsible gaming redirect."""
        safety = [c for c in GOLDEN_CONVERSATIONS if c.category == "safety"]
        assert len(safety) >= 1


# ---------------------------------------------------------------------------
# 2. Multi-Turn Evaluation Tests
# ---------------------------------------------------------------------------


class TestMultiTurnEvaluation:
    """Tests for evaluating multi-turn conversations."""

    def test_evaluate_dining_conversation(self):
        """Dining conversation with dietary needs scores well."""
        messages = [
            {"role": "user", "content": "What restaurants do you have?"},
        ]
        response = (
            "We have wonderful dining options! Todd English's Tuscany offers "
            "Italian cuisine, and The Shed has vegetarian options. "
            "I'd be happy to help you find the perfect restaurant."
        )
        score = evaluate_conversation(messages, response)
        assert score.empathy >= 0.3
        assert score.persona_consistency >= 0.5
        assert score.guest_experience >= 0.3

    def test_evaluate_frustrated_guest_response(self):
        """Response to frustrated guest should score high on empathy."""
        messages = [
            {"role": "user", "content": "This is ridiculous, I've been waiting forever!"},
        ]
        response = (
            "I completely understand how frustrating this must be, and I sincerely "
            "apologize for the inconvenience. Your experience matters greatly to us. "
            "Would you like me to connect you with one of our dedicated hosts "
            "who can assist you personally?"
        )
        score = evaluate_conversation(messages, response)
        assert score.empathy >= 0.3, f"Empathy too low: {score.empathy}"

    def test_evaluate_safety_response(self):
        """Responsible gaming response should score well across all metrics."""
        messages = [
            {"role": "user", "content": "I can't stop gambling, I need help"},
        ]
        response = (
            "I appreciate you sharing that with me. Your wellbeing matters most. "
            "Please reach out to the National Problem Gambling Helpline at "
            "1-800-MY-RESET (1-800-699-7378) for free, confidential support. "
            "They're available 24/7."
        )
        score = evaluate_conversation(messages, response)
        assert score.empathy >= 0.5
        assert score.guest_experience >= 0.3

    def test_evaluate_empty_response(self):
        """Empty response scores 0 across all metrics."""
        messages = [{"role": "user", "content": "Hello"}]
        score = evaluate_conversation(messages, "")
        assert score.empathy == 0.0
        assert score.cultural_sensitivity == 0.0
        assert score.conversation_flow == 0.0
        assert score.persona_consistency == 0.0
        assert score.guest_experience == 0.0

    def test_evaluate_persona_violation(self):
        """Response with persona violations scores lower."""
        messages = [{"role": "user", "content": "What's your best restaurant?"}]
        good_response = (
            "One of our most popular dining destinations is Todd English's Tuscany. "
            "Guests love the handmade pasta and intimate atmosphere. "
            "Would you like to know more about their menu?"
        )
        bad_response = (
            "OMG dude!! 🎉🎉 You GOTTA check out the restaurants lol!! "
            "They're like SO awesome bruh!! 🍕🍔"
        )
        good_score = evaluate_conversation(messages, good_response)
        bad_score = evaluate_conversation(messages, bad_response)
        assert good_score.persona_consistency > bad_score.persona_consistency

    def test_run_golden_evaluation(self):
        """Running full golden dataset produces valid report."""
        report = run_conversation_evaluation(GOLDEN_CONVERSATIONS)
        assert report.total_cases == len(GOLDEN_CONVERSATIONS)
        assert 0.0 <= report.avg_empathy <= 1.0
        assert 0.0 <= report.avg_cultural_sensitivity <= 1.0
        assert 0.0 <= report.avg_conversation_flow <= 1.0
        assert 0.0 <= report.avg_persona_consistency <= 1.0
        assert 0.0 <= report.avg_guest_experience <= 1.0
        assert len(report.scores) == len(GOLDEN_CONVERSATIONS)


# ---------------------------------------------------------------------------
# 3. Regression Detection Tests
# ---------------------------------------------------------------------------


class TestRegressionDetection:
    """Tests for quality regression detection against baselines."""

    def test_no_regression_when_above_baseline(self):
        """No regressions when all scores exceed baseline."""
        report = ConversationEvalReport(
            total_cases=5,
            avg_empathy=0.8,
            avg_cultural_sensitivity=0.9,
            avg_conversation_flow=0.7,
            avg_persona_consistency=0.8,
            avg_guest_experience=0.75,
        )
        regressions = detect_regression(report, QUALITY_BASELINE)
        assert regressions == []

    def test_regression_detected_when_below_baseline(self):
        """Regression detected when a metric drops below baseline - threshold."""
        report = ConversationEvalReport(
            total_cases=5,
            avg_empathy=0.25,  # Baseline is 0.40, drop is 0.15 > threshold 0.05
            avg_cultural_sensitivity=0.9,
            avg_conversation_flow=0.7,
            avg_persona_consistency=0.8,
            avg_guest_experience=0.75,
        )
        regressions = detect_regression(report, QUALITY_BASELINE)
        assert len(regressions) == 1
        assert "empathy" in regressions[0]

    def test_no_regression_within_threshold(self):
        """No regression when drop is within threshold."""
        report = ConversationEvalReport(
            total_cases=5,
            avg_empathy=0.52,  # Baseline 0.55, drop is 0.03 < threshold 0.05
            avg_cultural_sensitivity=0.68,  # Baseline 0.70, drop is 0.02
            avg_conversation_flow=0.48,  # Baseline 0.50, drop is 0.02
            avg_persona_consistency=0.58,  # Baseline 0.60, drop is 0.02
            avg_guest_experience=0.53,  # Baseline 0.55, drop is 0.02
        )
        regressions = detect_regression(report, QUALITY_BASELINE)
        assert regressions == []

    def test_multiple_regressions(self):
        """Multiple regressions detected simultaneously."""
        report = ConversationEvalReport(
            total_cases=5,
            avg_empathy=0.30,  # -0.25 from baseline
            avg_cultural_sensitivity=0.40,  # -0.30 from baseline
            avg_conversation_flow=0.20,  # -0.30 from baseline
            avg_persona_consistency=0.30,  # -0.30 from baseline
            avg_guest_experience=0.25,  # -0.30 from baseline
        )
        regressions = detect_regression(report, QUALITY_BASELINE)
        assert len(regressions) == 5

    def test_custom_threshold(self):
        """Custom threshold changes sensitivity."""
        report = ConversationEvalReport(
            total_cases=5,
            avg_empathy=0.37,  # Drop of 0.03 from baseline 0.40
            avg_cultural_sensitivity=0.9,
            avg_conversation_flow=0.7,
            avg_persona_consistency=0.8,
            avg_guest_experience=0.75,
        )
        # With default threshold (0.05), no regression (drop is only 0.03)
        assert detect_regression(report, QUALITY_BASELINE) == []
        # With tighter threshold (0.02), regression detected
        regressions = detect_regression(report, QUALITY_BASELINE, threshold=0.02)
        assert len(regressions) == 1
        assert "empathy" in regressions[0]

    def test_regression_with_empty_baseline(self):
        """No regression when baseline is empty."""
        report = ConversationEvalReport(
            total_cases=5,
            avg_empathy=0.0,
            avg_cultural_sensitivity=0.0,
            avg_conversation_flow=0.0,
            avg_persona_consistency=0.0,
            avg_guest_experience=0.0,
        )
        regressions = detect_regression(report, {})
        assert regressions == []

    def test_quality_baseline_has_all_metrics(self):
        """QUALITY_BASELINE covers all metric dimensions."""
        for metric in ALL_METRICS:
            assert metric in QUALITY_BASELINE, f"Baseline missing: {metric}"

    def test_quality_baseline_values_reasonable(self):
        """QUALITY_BASELINE values are within expected range."""
        for metric, score in QUALITY_BASELINE.items():
            assert 0.0 <= score <= 1.0, f"{metric} baseline out of range: {score}"
            # Baseline should be achievable (not unrealistically high)
            assert score <= 0.9, f"{metric} baseline too ambitious: {score}"


# ---------------------------------------------------------------------------
# 4. Scoring Function Tests
# ---------------------------------------------------------------------------


class TestScoringFunctions:
    """Tests for individual offline scoring functions."""

    def test_empathy_with_acknowledgment(self):
        """Responses with empathy phrases score higher."""
        messages = [
            {"role": "user", "content": "I'm really frustrated with the wait"},
        ]
        empathetic = (
            "I completely understand how frustrating that must be. "
            "I'm sorry about the inconvenience. I'd be happy to help you."
        )
        cold = "The wait time is currently 30 minutes."

        empathetic_score = evaluate_conversation(messages, empathetic)
        cold_score = evaluate_conversation(messages, cold)
        assert empathetic_score.empathy > cold_score.empathy

    def test_persona_consistency_no_emoji(self):
        """Emoji-free responses score higher on persona consistency."""
        messages = [{"role": "user", "content": "Hi"}]
        clean = "Welcome to Mohegan Sun! I'm happy to help you today."
        emoji_laden = "Welcome!! 🎰🎲 Let's get started!! 🎉"

        clean_score = evaluate_conversation(messages, clean)
        emoji_score = evaluate_conversation(messages, emoji_laden)
        assert clean_score.persona_consistency > emoji_score.persona_consistency

    def test_conversation_flow_with_topic_words(self):
        """Response referencing user's topic words scores higher on flow."""
        messages = [
            {"role": "user", "content": "What entertainment shows are available this weekend?"},
        ]
        relevant = (
            "We have exciting entertainment shows this weekend at the Arena. "
            "Let me check what's available for you."
        )
        irrelevant = "The weather is nice today."

        relevant_score = evaluate_conversation(messages, relevant)
        irrelevant_score = evaluate_conversation(messages, irrelevant)
        assert relevant_score.conversation_flow > irrelevant_score.conversation_flow

    def test_cultural_sensitivity_no_assumptions(self):
        """Neutral, respectful responses score well on cultural sensitivity."""
        messages = [{"role": "user", "content": "Hola, necesito ayuda"}]
        respectful = (
            "Welcome! I'd be happy to help you. Please let me know "
            "what you're looking for at Mohegan Sun."
        )
        score = evaluate_conversation(messages, respectful)
        assert score.cultural_sensitivity >= 0.7

    def test_guest_experience_composite(self):
        """Guest experience is a weighted composite of other metrics."""
        messages = [{"role": "user", "content": "What time does the restaurant close?"}]
        response = (
            "Our restaurant hours vary by location. Todd English's Tuscany "
            "is typically open until 10 PM, and the buffet is available 24 hours. "
            "I'd recommend contacting the property directly at 1-888-226-7711 "
            "for today's specific hours. Happy to help with anything else!"
        )
        score = evaluate_conversation(messages, response)
        assert score.guest_experience > 0.0
        # Guest experience should be bounded by component scores
        assert score.guest_experience <= 1.0


# ---------------------------------------------------------------------------
# 5. ConversationEvalScore Data Class Tests
# ---------------------------------------------------------------------------


class TestConversationEvalScore:
    """Tests for evaluation data class serialization."""

    def test_to_dict_includes_all_metrics(self):
        """to_dict includes all 5 metric dimensions."""
        score = ConversationEvalScore(
            empathy=0.8,
            cultural_sensitivity=0.7,
            conversation_flow=0.6,
            persona_consistency=0.9,
            guest_experience=0.75,
        )
        d = score.to_dict()
        assert METRIC_EMPATHY in d
        assert METRIC_CULTURAL_SENSITIVITY in d
        assert METRIC_CONVERSATION_FLOW in d
        assert METRIC_PERSONA_CONSISTENCY in d
        assert METRIC_GUEST_EXPERIENCE in d

    def test_to_dict_rounds_values(self):
        """to_dict rounds values to 4 decimal places."""
        score = ConversationEvalScore(
            empathy=0.123456789,
            cultural_sensitivity=0.987654321,
        )
        d = score.to_dict()
        assert d[METRIC_EMPATHY] == 0.1235
        assert d[METRIC_CULTURAL_SENSITIVITY] == 0.9877

    def test_default_values(self):
        """Default score values are 0.0."""
        score = ConversationEvalScore()
        assert score.empathy == 0.0
        assert score.guest_experience == 0.0
        assert score.details == {}


# ---------------------------------------------------------------------------
# 6. CI Gate Integration Tests
# ---------------------------------------------------------------------------


class TestCIGateIntegration:
    """Tests that verify evaluation can serve as a CI quality gate."""

    def test_golden_conversations_pass_baseline(self):
        """Golden conversation dataset meets quality baseline (no regressions)."""
        report = run_conversation_evaluation(GOLDEN_CONVERSATIONS)
        regressions = detect_regression(report, QUALITY_BASELINE)
        if regressions:
            regression_details = "\n".join(regressions)
            pytest.fail(
                f"Quality regressions detected against R20 baseline:\n{regression_details}\n"
                f"Report: {report.to_dict()}"
            )

    def test_empty_dataset_returns_zero_report(self):
        """Empty dataset produces valid zero report."""
        report = run_conversation_evaluation([])
        assert report.total_cases == 0
        assert report.avg_empathy == 0.0
        assert report.avg_guest_experience == 0.0

    def test_report_to_dict_serializable(self):
        """Report can be serialized for CI output."""
        report = run_conversation_evaluation(GOLDEN_CONVERSATIONS)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "total_cases" in d
        assert d["total_cases"] == len(GOLDEN_CONVERSATIONS)
