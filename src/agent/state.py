"""State schema for the Property Q&A custom StateGraph.

Defines PropertyQAState as a TypedDict with 9 fields for the 8-node graph,
plus Pydantic models for structured LLM outputs (router + validation).
"""

from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class PropertyQAState(TypedDict):
    """Typed state flowing through the 8-node property Q&A graph."""
    messages: Annotated[list, add_messages]
    query_type: str | None          # router classification (7 categories)
    router_confidence: float        # 0.0-1.0 from router LLM
    retrieved_context: list[dict]   # [{content, metadata, score}]
    validation_result: str | None   # PASS / FAIL / RETRY
    retry_count: int                # max 1 retry before fallback
    retry_feedback: str | None      # why validation failed
    current_time: str               # injected at graph entry
    sources_used: list[str]         # knowledge-base categories cited


class RouterOutput(BaseModel):
    """Structured output from the router node."""
    query_type: Literal[
        "property_qa", "hours_schedule", "greeting", "off_topic",
        "gambling_advice", "action_request", "ambiguous",
    ] = Field(
        description="One of: property_qa, hours_schedule, greeting, off_topic, gambling_advice, action_request, ambiguous"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the classification (0.0 to 1.0)"
    )


class ValidationResult(BaseModel):
    """Structured output from the validation node."""
    status: Literal["PASS", "FAIL"] = Field(
        description="PASS if the response meets all 6 criteria, FAIL otherwise"
    )
    reason: str = Field(
        description="Why the response passed or failed validation"
    )
