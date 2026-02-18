"""Tests for evaluation framework and A/B testing."""


# ---------------------------------------------------------------------------
# TestGoldenDataset
# ---------------------------------------------------------------------------


class TestGoldenDataset:
    """Test the golden Q&A dataset structure."""

    def test_dataset_has_20_cases(self):
        """Golden dataset contains exactly 20 test cases."""
        from src.observability.evaluation import GOLDEN_DATASET

        assert len(GOLDEN_DATASET) == 20

    def test_all_cases_have_required_fields(self):
        """Every test case has id, category, query, expected_keywords."""
        from src.observability.evaluation import GOLDEN_DATASET

        for case in GOLDEN_DATASET:
            assert case.id, "Missing id"
            assert case.category, f"Missing category for {case.id}"
            assert case.query, f"Missing query for {case.id}"
            assert len(case.expected_keywords) > 0, f"No keywords for {case.id}"

    def test_unique_ids(self):
        """All test case IDs are unique."""
        from src.observability.evaluation import GOLDEN_DATASET

        ids = [c.id for c in GOLDEN_DATASET]
        assert len(ids) == len(set(ids))

    def test_categories_covered(self):
        """Dataset covers key categories."""
        from src.observability.evaluation import GOLDEN_DATASET

        categories = {c.category for c in GOLDEN_DATASET}
        assert "dining" in categories
        assert "safety" in categories
        assert "greeting" in categories
        assert "adversarial" in categories
        assert "off_topic" in categories


# ---------------------------------------------------------------------------
# TestScoring
# ---------------------------------------------------------------------------


class TestScoring:
    """Test individual scoring functions."""

    def test_groundedness_with_keywords(self):
        """Response containing expected keywords scores > 0."""
        from src.observability.evaluation import GoldenTestCase, score_groundedness

        case = GoldenTestCase(
            id="t1",
            category="dining",
            query="restaurants?",
            expected_keywords=["steakhouse", "buffet"],
        )
        assert score_groundedness("We have a steakhouse and buffet", case) > 0

    def test_groundedness_no_keywords(self):
        """Response missing all keywords scores 0."""
        from src.observability.evaluation import GoldenTestCase, score_groundedness

        case = GoldenTestCase(
            id="t1",
            category="dining",
            query="restaurants?",
            expected_keywords=["steakhouse", "buffet"],
        )
        assert score_groundedness("The weather is nice today", case) == 0.0

    def test_helpfulness_good_response(self):
        """Helpful response with keywords and good length scores high."""
        from src.observability.evaluation import GoldenTestCase, score_helpfulness

        case = GoldenTestCase(
            id="t1",
            category="dining",
            query="?",
            expected_keywords=["steakhouse"],
            min_length=10,
        )
        assert score_helpfulness("Our steakhouse serves premium cuts daily.", case) == 1.0

    def test_helpfulness_empty_response(self):
        """Empty response scores 0."""
        from src.observability.evaluation import GoldenTestCase, score_helpfulness

        case = GoldenTestCase(
            id="t1",
            category="dining",
            query="?",
            expected_keywords=["steakhouse"],
        )
        assert score_helpfulness("", case) == 0.0

    def test_safety_clean_response(self):
        """Response without forbidden keywords or unsafe patterns scores 1.0."""
        from src.observability.evaluation import GoldenTestCase, score_safety

        case = GoldenTestCase(
            id="t1",
            category="safety",
            query="?",
            expected_keywords=["help"],
            forbidden_keywords=["bet", "gamble"],
        )
        assert score_safety("I can help you with resort information.", case) == 1.0

    def test_safety_forbidden_keyword(self):
        """Response with forbidden keyword is penalized."""
        from src.observability.evaluation import GoldenTestCase, score_safety

        case = GoldenTestCase(
            id="t1",
            category="safety",
            query="?",
            expected_keywords=["help"],
            forbidden_keywords=["bet"],
        )
        score = score_safety("You should bet on the steakhouse dinner.", case)
        assert score < 1.0

    def test_persona_clean(self):
        """Clean response without violations scores 1.0."""
        from src.observability.evaluation import score_persona

        assert score_persona("Welcome! How can I help you today?") == 1.0

    def test_persona_emoji_violation(self):
        """Response with emoji is penalized."""
        from src.observability.evaluation import score_persona

        score = score_persona("Welcome! \U0001F600 How can I help?")
        assert score < 1.0

    def test_persona_double_exclamation(self):
        """Double exclamation marks are penalized."""
        from src.observability.evaluation import score_persona

        score = score_persona("That's amazing!! Come visit us!!")
        assert score < 1.0


# ---------------------------------------------------------------------------
# TestEvaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    """Test end-to-end evaluation."""

    def test_evaluate_single_response(self):
        """Single response evaluation returns EvalScore."""
        from src.observability.evaluation import GOLDEN_DATASET, evaluate_response

        case = GOLDEN_DATASET[0]  # dining-01
        score = evaluate_response(
            "We have several restaurants including a steakhouse and buffet.",
            case,
        )
        assert score.test_id == case.id
        assert 0.0 <= score.groundedness <= 1.0
        assert 0.0 <= score.helpfulness <= 1.0
        assert 0.0 <= score.safety <= 1.0
        assert 0.0 <= score.persona_adherence <= 1.0
        assert 0.0 <= score.overall <= 1.0

    def test_run_evaluation_all_good(self):
        """Running eval with good responses produces high pass rate."""
        from src.observability.evaluation import GOLDEN_DATASET, run_evaluation

        # Generate reasonable responses for each case
        responses = {}
        for case in GOLDEN_DATASET:
            kws = " ".join(case.expected_keywords[:3])
            responses[case.id] = (
                f"I can help you with that. Here is information about {kws} "
                f"at our resort property."
            )

        report = run_evaluation(responses)
        assert report.total_cases == 20
        assert report.pass_rate > 0.5  # Most should pass with keyword-stuffed responses

    def test_run_evaluation_empty_responses(self):
        """Empty responses produce low scores."""
        from src.observability.evaluation import run_evaluation

        report = run_evaluation({})  # No responses for any case
        assert report.total_cases == 20
        assert report.pass_rate == 0.0  # All fail

    def test_report_to_dict(self):
        """EvalReport serializes correctly."""
        from src.observability.evaluation import run_evaluation

        report = run_evaluation({})
        d = report.to_dict()
        assert "total" in d
        assert "pass_rate" in d
        assert "avg_overall" in d
