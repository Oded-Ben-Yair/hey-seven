"""Pydantic v2 request/response models for the Hey Seven Property Q&A API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    """SSE payload shape for /chat responses.

    Not used as a FastAPI ``response_model`` because the endpoint
    returns ``EventSourceResponse``. This model documents the JSON
    structure inside each SSE ``data:`` field.
    """

    response: str
    thread_id: str
    sources: list[str] = []
    done: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
    agent_ready: bool
    property_loaded: bool


class PropertyInfoResponse(BaseModel):
    name: str
    location: str
    categories: list[str]
    document_count: int
