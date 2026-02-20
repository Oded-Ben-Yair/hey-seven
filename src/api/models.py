"""Pydantic v2 request/response models for the Hey Seven Property Q&A API."""

import re

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    thread_id: str | None = None

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            v,
            re.IGNORECASE,
        ):
            raise ValueError("thread_id must be a valid UUID")
        return v


# --- SSE Event Schemas (Wire Format Documentation) ---
# These models document the SSE event contract between backend and frontend.
# Events are serialized as dicts in graph.py for streaming performance;
# these schemas serve as the canonical API reference for each event type.


class SSEMetadataEvent(BaseModel):
    """``event: metadata`` — sent first with the thread ID."""

    thread_id: str


class SSETokenEvent(BaseModel):
    """``event: token`` — incremental text chunk from the LLM."""

    content: str


class SSESourcesEvent(BaseModel):
    """``event: sources`` — knowledge-base categories cited."""

    sources: list[str]


class SSEDoneEvent(BaseModel):
    """``event: done`` — signals end of stream."""

    done: bool = True


class SSEPingEvent(BaseModel):
    """``event: ping`` — heartbeat sent every 15s during long generations.

    Prevents client-side EventSource timeouts. Clients should ignore this event.
    """

    pass


class SSEErrorEvent(BaseModel):
    """``event: error`` — sent on failure."""

    error: str


class HealthResponse(BaseModel):
    status: str
    version: str
    agent_ready: bool
    property_loaded: bool
    rag_ready: bool = False
    observability_enabled: bool = False
    circuit_breaker_state: str = "unknown"


class FeedbackRequest(BaseModel):
    """User feedback on agent responses."""

    thread_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = Field(None, max_length=2000)

    @field_validator("thread_id")
    @classmethod
    def validate_feedback_thread_id(cls, v: str) -> str:
        if not re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            v,
            re.IGNORECASE,
        ):
            raise ValueError("thread_id must be a valid UUID")
        return v


class FeedbackResponse(BaseModel):
    """Response to feedback submission."""

    status: str
    thread_id: str


class SSEGraphNodeEvent(BaseModel):
    """``event: graph_node`` — node lifecycle event for graph trace panel."""

    node: str
    status: str  # "start" or "complete"
    duration_ms: int | None = None
    metadata: dict | None = None


class SSEReplaceEvent(BaseModel):
    """``event: replace`` — full response from non-streaming nodes."""

    content: str


class SmsWebhookResponse(BaseModel):
    """Response model for POST /sms/webhook."""

    status: str  # "ignored", "keyword_handled", "received"
    response: str | None = None  # Keyword response text (when status=keyword_handled)
    from_: str | None = Field(None, alias="from")  # Sender number (when status=received)


class CmsWebhookResponse(BaseModel):
    """Response model for POST /cms/webhook."""

    status: str  # "success", "rejected", "error"
    updated_categories: list[str] | None = None
    error: str | None = None


class GraphStructureResponse(BaseModel):
    """Response model for GET /graph endpoint."""

    nodes: list[str]
    edges: list[dict]


class PropertyInfoResponse(BaseModel):
    name: str
    location: str
    categories: list[str]
    document_count: int
