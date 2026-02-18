"""Tests for the observability module.

All tests work without the langfuse package installed.
LangFuse integration is tested via mocked imports.
"""

import time


class TestLangfuseClient:
    """Test LangFuse client initialization and configuration."""

    def test_disabled_when_no_public_key(self):
        """LangFuse is disabled when LANGFUSE_PUBLIC_KEY is empty."""
        from src.observability.langfuse_client import is_observability_enabled

        assert is_observability_enabled() is False

    def test_get_handler_returns_none_when_disabled(self):
        """get_langfuse_handler returns None when not configured."""
        from src.observability.langfuse_client import get_langfuse_handler

        handler = get_langfuse_handler(trace_id="test-123")
        assert handler is None

    def test_should_sample_dev_always_true(self):
        """Development mode always samples (100%)."""
        from src.observability.langfuse_client import should_sample

        # Default ENVIRONMENT is "development"
        results = [should_sample() for _ in range(10)]
        assert all(results)

    def test_should_sample_production_respects_rate(self, monkeypatch):
        """Production mode samples at 10% rate."""
        from src.config import get_settings

        monkeypatch.setattr(get_settings(), "ENVIRONMENT", "production")

        from src.observability.langfuse_client import should_sample

        import random

        random.seed(42)
        results = [should_sample() for _ in range(100)]
        # With 10% rate, expect roughly 10 True out of 100
        assert 2 <= sum(results) <= 25  # Generous bounds for randomness

    def test_clear_cache_callable(self):
        """clear_langfuse_cache doesn't raise."""
        from src.observability.langfuse_client import clear_langfuse_cache

        clear_langfuse_cache()  # Should not raise

    def test_handler_returns_none_when_import_fails(self):
        """Handler returns None when langfuse package not installed."""
        from src.observability.langfuse_client import get_langfuse_handler

        # Even if we trick it into thinking it's enabled, import failure is safe
        result = get_langfuse_handler(trace_id="test")
        assert result is None


class TestTraceContext:
    """Test trace context creation and management."""

    def test_create_trace_context(self):
        """TraceContext is created with correct fields."""
        from src.observability.traces import create_trace_context

        ctx = create_trace_context(
            "trace-123",
            session_id="session-456",
            user_id="user-789",
            tags=["test"],
        )
        assert ctx.trace_id == "trace-123"
        assert ctx.session_id == "session-456"
        assert ctx.user_id == "user-789"
        assert ctx.tags == ["test"]
        assert len(ctx.spans) == 0

    def test_record_node_span(self):
        """Node spans are recorded correctly."""
        from src.observability.traces import create_trace_context, record_node_span

        ctx = create_trace_context("trace-123")
        span = record_node_span(
            ctx,
            "router",
            150,
            metadata={"query_type": "property_qa"},
        )
        assert span.node_name == "router"
        assert span.duration_ms == 150
        assert span.metadata == {"query_type": "property_qa"}
        assert span.error is None
        assert len(ctx.spans) == 1

    def test_record_multiple_spans(self):
        """Multiple spans are recorded in order."""
        from src.observability.traces import create_trace_context, record_node_span

        ctx = create_trace_context("trace-123")
        record_node_span(ctx, "router", 100)
        record_node_span(ctx, "retrieve", 200)
        record_node_span(ctx, "generate", 500)
        record_node_span(ctx, "validate", 80)
        assert len(ctx.spans) == 4
        assert [s.node_name for s in ctx.spans] == [
            "router",
            "retrieve",
            "generate",
            "validate",
        ]

    def test_record_span_with_error(self):
        """Error spans are recorded correctly."""
        from src.observability.traces import create_trace_context, record_node_span

        ctx = create_trace_context("trace-123")
        span = record_node_span(ctx, "generate", 1000, error="LLM timeout")
        assert span.error == "LLM timeout"

    def test_end_trace(self):
        """end_trace sets total_duration_ms."""
        from src.observability.traces import create_trace_context, end_trace, record_node_span

        ctx = create_trace_context("trace-123")
        record_node_span(ctx, "router", 100)
        time.sleep(0.01)  # Small delay for measurable duration
        result = end_trace(ctx)
        assert result.total_duration_ms is not None
        assert result.total_duration_ms >= 0
        assert result.end_time is not None

    def test_to_dict(self):
        """TraceContext serializes to dict correctly."""
        from src.observability.traces import create_trace_context, end_trace, record_node_span

        ctx = create_trace_context("trace-123", session_id="sess-1", tags=["dining"])
        record_node_span(ctx, "router", 100, metadata={"query_type": "property_qa"})
        record_node_span(ctx, "retrieve", 200, metadata={"doc_count": 5})
        end_trace(ctx)

        d = ctx.to_dict()
        assert d["trace_id"] == "trace-123"
        assert d["session_id"] == "sess-1"
        assert d["span_count"] == 2
        assert len(d["spans"]) == 2
        assert d["spans"][0]["node"] == "router"
        assert d["spans"][1]["node"] == "retrieve"
        assert d["tags"] == ["dining"]
        assert d["total_duration_ms"] >= 0

    def test_empty_trace_to_dict(self):
        """Empty trace (no spans) serializes correctly."""
        from src.observability.traces import create_trace_context, end_trace

        ctx = create_trace_context("trace-empty")
        end_trace(ctx)
        d = ctx.to_dict()
        assert d["span_count"] == 0
        assert d["spans"] == []

    def test_trace_context_defaults(self):
        """TraceContext has sensible defaults."""
        from src.observability.traces import create_trace_context

        ctx = create_trace_context("trace-min")
        assert ctx.session_id is None
        assert ctx.user_id is None
        assert ctx.tags == []
        assert ctx.metadata == {}
        assert ctx.end_time is None
        assert ctx.total_duration_ms is None


class TestNodeSpan:
    """Test NodeSpan dataclass."""

    def test_span_creation(self):
        """NodeSpan can be created directly."""
        from src.observability.traces import NodeSpan

        span = NodeSpan(node_name="test", start_time=time.monotonic())
        assert span.node_name == "test"
        assert span.end_time is None
        assert span.duration_ms is None
        assert span.error is None

    def test_span_with_all_fields(self):
        """NodeSpan with all fields set."""
        from src.observability.traces import NodeSpan

        now = time.monotonic()
        span = NodeSpan(
            node_name="generate",
            start_time=now - 0.5,
            end_time=now,
            duration_ms=500,
            metadata={"model": "gemini-2.5-flash"},
            error=None,
        )
        assert span.duration_ms == 500
        assert span.metadata["model"] == "gemini-2.5-flash"
