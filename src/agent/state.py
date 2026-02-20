"""State schema for the Property Q&A custom StateGraph.

Defines PropertyQAState as a TypedDict with fields for the graph nodes,
plus Pydantic models for structured LLM outputs (router + validation).

v2 additions: ``extracted_fields``, ``whisper_plan`` for specialist agent
routing and background planning.
"""

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


def _keep_max(a: int, b: int) -> int:
    """Reducer that preserves the maximum value across state updates.

    Used for session-level counters that must accumulate across turns.
    When _initial_state() passes 0, max(existing, 0) preserves the count.
    When a node increments, max(existing, new) updates the count.
    """
    return max(a, b)


class RetrievedChunk(TypedDict):
    """A single chunk returned by the RAG retriever.

    Explicit schema prevents implicit dict contracts from drifting
    between pipeline.py and nodes.py.
    """

    content: str
    metadata: dict[str, Any]
    score: float


class PropertyQAState(TypedDict):
    """Typed state flowing through the property Q&A graph.

    ``messages`` is persisted across turns via the checkpointer's ``add_messages``
    reducer. ``responsible_gaming_count`` also persists via ``_keep_max`` reducer
    for session-level escalation tracking. All other fields are **per-turn** —
    they are reset by ``_initial_state()`` at the start of each ``chat()`` /
    ``chat_stream()`` call.

    ``skip_validation`` is set to ``True`` by the generate node (host_agent)
    when the response is a deterministic fallback (empty context, LLM error,
    circuit breaker open) that does not need adversarial validation.

    v2 fields (all optional with defaults in ``_initial_state()``):
    - ``extracted_fields``: structured fields extracted from the guest message
    - ``whisper_plan``: background planner output (dict from WhisperPlan.model_dump())
    """

    messages: Annotated[list, add_messages]
    query_type: str | None          # router classification (7 categories)
    router_confidence: float        # 0.0-1.0 from router LLM
    retrieved_context: list[RetrievedChunk]  # chunks from RAG retriever
    validation_result: str | None   # PASS / FAIL / RETRY
    retry_count: int                # max 1 retry before fallback
    skip_validation: bool           # True to bypass validator (safe fallback paths)
    retry_feedback: str | None      # why validation failed
    current_time: str               # injected at graph entry
    sources_used: list[str]         # knowledge-base categories cited
    # v2 fields
    extracted_fields: dict[str, Any]  # structured fields from guest message
    whisper_plan: dict[str, Any] | None  # background planner output (WhisperPlan.model_dump())
    # _keep_max reducer: preserves the maximum value across state updates.
    # When _initial_state() resets this to 0, max(existing, 0) preserves
    # the accumulated count. When compliance_gate increments, the new value
    # is preserved. This prevents accidental reset of the escalation counter
    # when per-turn fields are reset via _initial_state().
    responsible_gaming_count: Annotated[int, _keep_max]


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
    status: Literal["PASS", "FAIL", "RETRY"] = Field(
        description="PASS if the response meets all 6 criteria, RETRY for minor issues worth correcting, FAIL for serious violations"
    )
    reason: str = Field(
        description="Why the response passed or failed validation"
    )


