"""Trace context management for graph node execution.

Provides lightweight trace tracking without requiring the full LangFuse
SDK. Records node entry/exit times, durations, and metadata for both
LangFuse integration and structured logging.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NodeSpan:
    """A recorded span for a single graph node execution."""

    node_name: str
    start_time: float  # monotonic
    end_time: float | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class TraceContext:
    """Context for a single graph execution trace.

    Tracks all node spans within a single user message -> response cycle.
    Can be serialized for logging and sent to LangFuse.
    """

    trace_id: str
    session_id: str | None = None
    user_id: str | None = None
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    total_duration_ms: int | None = None
    spans: list[NodeSpan] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize trace context for logging and LangFuse."""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "total_duration_ms": self.total_duration_ms,
            "span_count": len(self.spans),
            "spans": [
                {
                    "node": s.node_name,
                    "duration_ms": s.duration_ms,
                    "error": s.error,
                    "metadata": s.metadata,
                }
                for s in self.spans
            ],
            "tags": self.tags,
            "metadata": self.metadata,
        }


def create_trace_context(
    trace_id: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TraceContext:
    """Create a new trace context for a graph execution.

    Args:
        trace_id: Unique ID for this trace (e.g., request_id).
        session_id: Session/thread ID for grouping traces.
        user_id: User identifier (hashed phone, etc.).
        tags: Categorization tags.
        metadata: Additional key-value metadata.

    Returns:
        A new TraceContext instance.
    """
    return TraceContext(
        trace_id=trace_id,
        session_id=session_id,
        user_id=user_id,
        tags=tags or [],
        metadata=metadata or {},
    )


def record_node_span(
    ctx: TraceContext,
    node_name: str,
    duration_ms: int,
    *,
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
) -> NodeSpan:
    """Record a completed node span in the trace context.

    Args:
        ctx: The active trace context.
        node_name: Name of the graph node (e.g., "router", "generate").
        duration_ms: Execution time in milliseconds.
        metadata: Node-specific metadata (e.g., query_type, doc_count).
        error: Error message if the node failed.

    Returns:
        The recorded NodeSpan.
    """
    now = time.monotonic()
    span = NodeSpan(
        node_name=node_name,
        start_time=now - (duration_ms / 1000.0),
        end_time=now,
        duration_ms=duration_ms,
        metadata=metadata or {},
        error=error,
    )
    ctx.spans.append(span)
    return span


def end_trace(ctx: TraceContext) -> TraceContext:
    """Finalize a trace context by recording the total duration.

    Args:
        ctx: The trace context to finalize.

    Returns:
        The finalized TraceContext with total_duration_ms set.
    """
    ctx.end_time = time.monotonic()
    ctx.total_duration_ms = int((ctx.end_time - ctx.start_time) * 1000)

    logger.info(
        "Trace completed: trace_id=%s spans=%d duration=%dms",
        ctx.trace_id,
        len(ctx.spans),
        ctx.total_duration_ms,
    )
    return ctx
