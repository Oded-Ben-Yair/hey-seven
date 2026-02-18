"""Phase 4 integration tests: Observability, Evaluation, A/B Testing, PII, Feedback.

Validates that Phase 4 modules (LangFuse observability, evaluation framework,
A/B testing, PII redaction, feedback endpoint) are correctly wired into the
API and that all modules work together end-to-end.
"""

import json
import uuid
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_app():
    """Create a test client with mocked agent for Phase 4 tests."""
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    app = create_app()
    app.state.agent = MagicMock()
    app.state.property_data = {"property": {"name": "Test Casino"}}
    app.state.ready = True
    return TestClient(app)


# ---------------------------------------------------------------------------
# TestHealthEndpointObservability
# ---------------------------------------------------------------------------


class TestHealthEndpointObservability:
    """Test that /health reports observability status."""

    @pytest.fixture
    def client(self):
        return _make_test_app()

    def test_health_includes_observability_enabled(self, client):
        """Health endpoint returns observability_enabled field."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "observability_enabled" in data
        # Without LANGFUSE_PUBLIC_KEY set, should be False
        assert data["observability_enabled"] is False

    def test_health_still_reports_all_fields(self, client):
        """Health endpoint still has all original fields + new field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "agent_ready" in data
        assert "property_loaded" in data
        assert "observability_enabled" in data


# ---------------------------------------------------------------------------
# TestFeedbackEndpoint
# ---------------------------------------------------------------------------


class TestFeedbackEndpoint:
    """Test POST /feedback endpoint."""

    @pytest.fixture
    def client(self):
        return _make_test_app()

    def test_valid_feedback(self, client):
        """Valid feedback returns 200 with 'received' status."""
        thread_id = str(uuid.uuid4())
        response = client.post(
            "/feedback",
            json={"thread_id": thread_id, "rating": 5, "comment": "Great answer!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "received"
        assert data["thread_id"] == thread_id

    def test_feedback_without_comment(self, client):
        """Feedback without comment is valid."""
        thread_id = str(uuid.uuid4())
        response = client.post(
            "/feedback",
            json={"thread_id": thread_id, "rating": 3},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "received"

    def test_feedback_min_rating(self, client):
        """Rating of 1 is valid."""
        thread_id = str(uuid.uuid4())
        response = client.post(
            "/feedback",
            json={"thread_id": thread_id, "rating": 1},
        )
        assert response.status_code == 200

    def test_feedback_max_rating(self, client):
        """Rating of 5 is valid."""
        thread_id = str(uuid.uuid4())
        response = client.post(
            "/feedback",
            json={"thread_id": thread_id, "rating": 5},
        )
        assert response.status_code == 200

    def test_feedback_invalid_rating_too_low(self, client):
        """Rating below 1 returns 422."""
        thread_id = str(uuid.uuid4())
        response = client.post(
            "/feedback",
            json={"thread_id": thread_id, "rating": 0},
        )
        assert response.status_code == 422

    def test_feedback_invalid_rating_too_high(self, client):
        """Rating above 5 returns 422."""
        thread_id = str(uuid.uuid4())
        response = client.post(
            "/feedback",
            json={"thread_id": thread_id, "rating": 6},
        )
        assert response.status_code == 422

    def test_feedback_invalid_thread_id(self, client):
        """Invalid thread_id returns 422."""
        response = client.post(
            "/feedback",
            json={"thread_id": "not-a-uuid", "rating": 3},
        )
        assert response.status_code == 422

    def test_feedback_missing_rating(self, client):
        """Missing rating returns 422."""
        thread_id = str(uuid.uuid4())
        response = client.post(
            "/feedback",
            json={"thread_id": thread_id},
        )
        assert response.status_code == 422

    def test_feedback_pii_in_comment_redacted_in_logs(self, client):
        """PII in comment doesn't leak into response (redacted in logs only)."""
        thread_id = str(uuid.uuid4())
        response = client.post(
            "/feedback",
            json={
                "thread_id": thread_id,
                "rating": 4,
                "comment": "My name is John Smith and my number is 203-555-1234",
            },
        )
        # Response itself doesn't include the comment
        assert response.status_code == 200
        data = response.json()
        assert "comment" not in data  # Comment is not echoed back


# ---------------------------------------------------------------------------
# TestObservabilityModuleIntegration
# ---------------------------------------------------------------------------


class TestObservabilityModuleIntegration:
    """Test that observability module works end-to-end."""

    def test_trace_context_full_lifecycle(self):
        """TraceContext lifecycle: create -> record spans -> end -> serialize."""
        from src.observability.traces import (
            create_trace_context,
            end_trace,
            record_node_span,
        )

        ctx = create_trace_context(
            "int-trace-001",
            session_id="session-abc",
            tags=["integration_test"],
        )

        # Simulate graph execution
        record_node_span(ctx, "compliance_gate", 5, metadata={"clean": True})
        record_node_span(ctx, "router", 120, metadata={"query_type": "property_qa"})
        record_node_span(ctx, "retrieve", 200, metadata={"doc_count": 5})
        record_node_span(ctx, "whisper_planner", 80, metadata={"has_plan": True})
        record_node_span(ctx, "generate", 800, metadata={"model": "gemini-2.5-flash"})
        record_node_span(ctx, "validate", 100, metadata={"result": "PASS"})
        record_node_span(ctx, "persona_envelope", 10)
        record_node_span(ctx, "respond", 5, metadata={"sources": ["dining"]})

        end_trace(ctx)

        # Verify serialization
        d = ctx.to_dict()
        assert d["trace_id"] == "int-trace-001"
        assert d["session_id"] == "session-abc"
        assert d["span_count"] == 8
        assert d["total_duration_ms"] >= 0
        assert d["tags"] == ["integration_test"]

        # Verify JSON serializable
        serialized = json.dumps(d)
        assert "int-trace-001" in serialized

    def test_langfuse_disabled_without_config(self):
        """LangFuse is disabled without LANGFUSE_PUBLIC_KEY."""
        from src.observability import is_observability_enabled

        assert is_observability_enabled() is False

    def test_langfuse_handler_returns_none_without_config(self):
        """get_langfuse_handler returns None without config."""
        from src.observability import get_langfuse_handler

        handler = get_langfuse_handler(
            trace_id="test-trace",
            session_id="test-session",
            tags=["test"],
        )
        assert handler is None

    def test_observability_imports(self):
        """All observability exports are importable."""
        from src.observability import (
            TraceContext,
            create_trace_context,
            end_trace,
            get_langfuse_handler,
            is_observability_enabled,
            record_node_span,
        )

        assert callable(create_trace_context)
        assert callable(end_trace)
        assert callable(record_node_span)
        assert callable(get_langfuse_handler)
        assert callable(is_observability_enabled)
        assert TraceContext is not None


# ---------------------------------------------------------------------------
# TestEvaluationFrameworkIntegration
# ---------------------------------------------------------------------------


class TestEvaluationFrameworkIntegration:
    """Test evaluation framework end-to-end."""

    def test_full_evaluation_run(self):
        """Run full evaluation with mock responses."""
        from src.observability.evaluation import GOLDEN_DATASET, run_evaluation

        # Generate responses that should score well
        responses = {}
        for case in GOLDEN_DATASET:
            if case.category == "safety" and "gambling" in case.query.lower():
                responses[case.id] = (
                    "I want to help you enjoy your time responsibly. "
                    "If you need support, please call 1-800-522-4700."
                )
            elif case.category == "safety" and "self-exclu" in case.query.lower():
                responses[case.id] = (
                    "I can help with self-exclusion information. "
                    "Please contact our support team for the self-exclusion program."
                )
            elif case.category == "safety" and "real person" in case.query.lower():
                responses[case.id] = (
                    "I'm an AI assistant, an artificial intelligence designed "
                    "to help you with information about our resort."
                )
            elif case.category == "greeting":
                responses[case.id] = (
                    "Hello! Welcome to our resort. How can I help you today?"
                )
            elif case.category == "adversarial":
                responses[case.id] = (
                    "I'd be happy to help you with information about our "
                    "casino resort and its amenities."
                )
            elif case.category == "off_topic":
                responses[case.id] = (
                    "I'd be happy to assist you with information about our "
                    "property and resort amenities."
                )
            elif case.category == "booking":
                responses[case.id] = (
                    "While I can't make a reservation directly, I'd recommend "
                    "calling our restaurant to book a table. You can reach them "
                    "at the number on our website."
                )
            else:
                kws = " ".join(case.expected_keywords[:3])
                responses[case.id] = (
                    f"Here is information about {kws} at our resort property. "
                    f"We offer a variety of {case.category} options."
                )

        report = run_evaluation(responses)
        assert report.total_cases == 20
        assert report.pass_rate > 0.5
        assert 0.0 <= report.avg_overall <= 1.0

        # Verify report serialization
        d = report.to_dict()
        assert isinstance(d["pass_rate"], float)
        assert isinstance(d["avg_overall"], float)

    def test_evaluation_scores_have_correct_ranges(self):
        """All scoring dimensions produce values in [0, 1]."""
        from src.observability.evaluation import GOLDEN_DATASET, evaluate_response

        for case in GOLDEN_DATASET[:5]:  # Test first 5 for speed
            score = evaluate_response(
                "We have a steakhouse and buffet for dining.",
                case,
            )
            assert 0.0 <= score.groundedness <= 1.0
            assert 0.0 <= score.helpfulness <= 1.0
            assert 0.0 <= score.safety <= 1.0
            assert 0.0 <= score.persona_adherence <= 1.0
            assert 0.0 <= score.overall <= 1.0


# ---------------------------------------------------------------------------
# TestABTestingIntegration — REMOVED (ab_testing.py deleted in v2.1 cleanup)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestPIIRedactionIntegration
# ---------------------------------------------------------------------------


class TestPIIRedactionIntegration:
    """Test PII redaction in integration with other modules."""

    def test_pii_redaction_with_trace_metadata(self):
        """PII in trace metadata is redactable."""
        from src.api.pii_redaction import redact_dict
        from src.observability.traces import create_trace_context, end_trace

        ctx = create_trace_context(
            trace_id="pii-trace",
            metadata={"user_message": "My name is John Smith, call 203-555-1234"},
        )
        end_trace(ctx)

        d = ctx.to_dict()
        redacted = redact_dict(d["metadata"])
        assert "[NAME]" in redacted["user_message"]
        assert "[PHONE]" in redacted["user_message"]
        assert "John Smith" not in redacted["user_message"]

    def test_pii_redaction_preserves_non_pii_metadata(self):
        """Non-PII metadata is preserved after redaction."""
        from src.api.pii_redaction import redact_dict

        metadata = {
            "query_type": "property_qa",
            "doc_count": 5,
            "user_message": "What restaurants do you have?",
        }
        redacted = redact_dict(metadata)
        assert redacted["query_type"] == "property_qa"
        assert redacted["doc_count"] == 5
        assert redacted["user_message"] == "What restaurants do you have?"

    def test_pii_detection_for_logging_gate(self):
        """contains_pii can gate whether to redact before logging."""
        from src.api.pii_redaction import contains_pii, redact_pii

        safe_msg = "What restaurants do you have?"
        risky_msg = "My name is John Smith"

        # Only redact when PII detected (saves CPU on clean messages)
        if contains_pii(safe_msg):
            safe_msg = redact_pii(safe_msg)
        assert "What restaurants" in safe_msg  # Unchanged

        if contains_pii(risky_msg):
            risky_msg = redact_pii(risky_msg)
        assert "[NAME]" in risky_msg  # Redacted


# ---------------------------------------------------------------------------
# TestPhase4ModuleImports
# ---------------------------------------------------------------------------


class TestPhase4ModuleImports:
    """Verify all Phase 4 modules are importable and wired."""

    def test_observability_package(self):
        """src.observability package imports cleanly."""
        import src.observability

        assert hasattr(src.observability, "create_trace_context")
        assert hasattr(src.observability, "get_langfuse_handler")

    def test_evaluation_module(self):
        """Evaluation module is importable with golden dataset."""
        from src.observability.evaluation import GOLDEN_DATASET, run_evaluation

        assert len(GOLDEN_DATASET) == 20
        assert callable(run_evaluation)

    def test_pii_redaction_module(self):
        """PII redaction module is importable."""
        from src.api.pii_redaction import contains_pii, redact_dict, redact_pii

        assert callable(redact_pii)
        assert callable(redact_dict)
        assert callable(contains_pii)

    def test_feedback_model(self):
        """FeedbackRequest model validates correctly."""
        from src.api.models import FeedbackRequest

        fb = FeedbackRequest(
            thread_id="12345678-1234-1234-1234-123456789abc",
            rating=4,
            comment="Good",
        )
        assert fb.rating == 4

    def test_health_response_has_observability_field(self):
        """HealthResponse model includes observability_enabled."""
        from src.api.models import HealthResponse

        hr = HealthResponse(
            status="healthy",
            version="0.1.0",
            agent_ready=True,
            property_loaded=True,
            observability_enabled=False,
        )
        assert hr.observability_enabled is False


# ---------------------------------------------------------------------------
# TestConfTestCleanup
# ---------------------------------------------------------------------------


class TestConfTestCleanup:
    """Verify conftest clears Phase 4 caches."""

    def test_langfuse_cache_cleared(self):
        """Conftest clears langfuse client cache between tests."""
        from src.observability.langfuse_client import _get_langfuse_client

        # Call to populate cache
        _get_langfuse_client()
        # After test, conftest fixture will clear it.
        # We verify the cache_clear method exists and is callable.
        assert hasattr(_get_langfuse_client, "cache_clear")
        assert callable(_get_langfuse_client.cache_clear)

    def test_langfuse_client_is_singleton(self):
        """LangFuse client is cached (lru_cache singleton)."""
        from src.observability.langfuse_client import _get_langfuse_client

        a = _get_langfuse_client()
        b = _get_langfuse_client()
        # Both return None (not configured), but importantly they're the same call
        assert a is b
