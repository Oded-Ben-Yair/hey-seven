"""Tests for evaluation framework and A/B testing."""

import pytest


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


# ---------------------------------------------------------------------------
# TestABTesting
# ---------------------------------------------------------------------------


class TestABTesting:
    """Test A/B testing framework."""

    def _make_experiment(self):
        """Create a simple 50/50 experiment."""
        from src.observability.ab_testing import ABExperiment, ABVariant

        return ABExperiment(
            experiment_id="prompt-v2",
            description="Test new system prompt",
            variants=[
                ABVariant(name="control", description="Current prompt"),
                ABVariant(
                    name="treatment",
                    description="New prompt",
                    config_overrides={"system_prompt_suffix": "Be extra friendly."},
                ),
            ],
            traffic_split=[50, 50],
        )

    def test_deterministic_assignment(self):
        """Same thread_id always gets same variant."""
        from src.observability.ab_testing import assign_variant

        exp = self._make_experiment()
        v1 = assign_variant("thread-123", exp)
        v2 = assign_variant("thread-123", exp)
        assert v1.variant_name == v2.variant_name
        assert v1.bucket == v2.bucket

    def test_different_threads_different_variants(self):
        """Different thread_ids get distributed across variants."""
        from src.observability.ab_testing import assign_variant

        exp = self._make_experiment()
        variants = set()
        for i in range(100):
            v = assign_variant(f"thread-{i}", exp)
            variants.add(v.variant_name)
        # With 100 threads and 50/50 split, both variants should appear
        assert len(variants) == 2

    def test_disabled_experiment_returns_control(self):
        """Disabled experiment always returns first variant."""
        from src.observability.ab_testing import assign_variant

        exp = self._make_experiment()
        exp.enabled = False
        for i in range(20):
            v = assign_variant(f"thread-{i}", exp)
            assert v.variant_name == "control"

    def test_traffic_split_must_sum_100(self):
        """Traffic split not summing to 100 raises ValueError."""
        from src.observability.ab_testing import ABExperiment, ABVariant

        with pytest.raises(ValueError, match="must sum to 100"):
            ABExperiment(
                experiment_id="bad",
                description="Bad split",
                variants=[
                    ABVariant(name="a", description="A"),
                    ABVariant(name="b", description="B"),
                ],
                traffic_split=[60, 60],  # Sums to 120
            )

    def test_variants_traffic_split_length_mismatch(self):
        """Mismatched lengths raise ValueError."""
        from src.observability.ab_testing import ABExperiment, ABVariant

        with pytest.raises(ValueError, match="must have same length"):
            ABExperiment(
                experiment_id="bad",
                description="Bad",
                variants=[ABVariant(name="a", description="A")],
                traffic_split=[50, 50],
            )

    def test_get_trace_tags(self):
        """Trace tags include experiment, variant, and bucket."""
        from src.observability.ab_testing import ABAssignment, get_trace_tags

        assignment = ABAssignment(
            experiment_id="prompt-v2",
            variant_name="treatment",
            config_overrides={},
            bucket=42,
        )
        tags = get_trace_tags(assignment)
        assert "ab:prompt-v2" in tags
        assert "variant:treatment" in tags
        assert "bucket:42" in tags

    def test_three_way_split(self):
        """Three-way split distributes correctly."""
        from src.observability.ab_testing import ABExperiment, ABVariant, assign_variant

        exp = ABExperiment(
            experiment_id="3way",
            description="Three variants",
            variants=[
                ABVariant(name="a", description="A"),
                ABVariant(name="b", description="B"),
                ABVariant(name="c", description="C"),
            ],
            traffic_split=[33, 34, 33],
        )
        counts = {"a": 0, "b": 0, "c": 0}
        for i in range(300):
            v = assign_variant(f"thread-{i}", exp)
            counts[v.variant_name] += 1
        # All three variants should have some assignments
        assert counts["a"] > 0
        assert counts["b"] > 0
        assert counts["c"] > 0

    def test_assignment_includes_config_overrides(self):
        """Assignment carries the variant's config overrides."""
        from src.observability.ab_testing import assign_variant

        exp = self._make_experiment()
        # Find a thread that gets treatment
        for i in range(100):
            v = assign_variant(f"thread-{i}", exp)
            if v.variant_name == "treatment":
                assert "system_prompt_suffix" in v.config_overrides
                break
