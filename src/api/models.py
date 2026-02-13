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


class SSEErrorEvent(BaseModel):
    """``event: error`` — sent on failure."""

    error: str


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
