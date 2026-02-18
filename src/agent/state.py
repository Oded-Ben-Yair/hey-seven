"""State schema for the Property Q&A custom StateGraph.

Defines PropertyQAState as a TypedDict with fields for the graph nodes,
plus Pydantic models for structured LLM outputs (router + validation).

v2 additions: ``extracted_fields``, ``whisper_plan`` for specialist agent
routing and background planning.  ``CasinoHostState`` is a
backward-compatible alias for v2 code.
"""

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


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

    ``messages`` is the only field persisted across turns via the checkpointer's
    ``add_messages`` reducer. All other fields are **per-turn** — they are reset
    by ``_initial_state()`` at the start of each ``chat()`` / ``chat_stream()``
    call. This ensures clean routing and validation state per turn while the
    checkpointer maintains conversation history.

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
    responsible_gaming_count: int  # session-level escalation counter for responsible gaming triggers


# Deprecated alias — use PropertyQAState directly. Kept for backward
# compatibility with any external code referencing the v1 name.
# TODO(v3): Remove this alias.
CasinoHostState = PropertyQAState


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


class ExtractedFields(BaseModel):
    """Schema for fields extracted from guest messages by specialist agents.

    Validates that extracted fields conform to expected types. Unknown
    fields are preserved (forward-compatible) via ``model_config``.
    """
    guest_name: str | None = None
    party_size: int | None = Field(None, ge=1, le=100)
    date_preference: str | None = None
    cuisine_preference: str | None = None
    budget_range: str | None = None
    special_requests: str | None = None

    model_config = {"extra": "allow"}  # Forward-compatible: unknown fields preserved


class WhisperPlan(BaseModel):
    """Schema for whisper planner output.

    Validates the background planner's structured output before it flows
    through the graph state.
    """
    engagement_hooks: list[str] = Field(default_factory=list)
    upsell_opportunities: list[str] = Field(default_factory=list)
    personalization_notes: str = ""
    profile_completeness: float = Field(0.0, ge=0.0, le=1.0)

    model_config = {"extra": "allow"}


def validate_state_transition(state: dict) -> list[str]:
    """Validate state field constraints for debugging and monitoring.

    Returns a list of warning messages for any constraint violations.
    Does NOT raise — callers decide whether to log or abort.

    Checked constraints:
    - retry_count in [0, 2] (max 1 retry + initial attempt)
    - router_confidence in [0.0, 1.0]
    - validation_result in {None, "PASS", "FAIL", "RETRY"}
    - query_type is a known category or None
    """
    warnings: list[str] = []

    retry = state.get("retry_count", 0)
    if not (0 <= retry <= 2):
        warnings.append(f"retry_count={retry} outside valid range [0, 2]")

    confidence = state.get("router_confidence", 0.0)
    if not (0.0 <= confidence <= 1.0):
        warnings.append(f"router_confidence={confidence} outside valid range [0.0, 1.0]")

    vr = state.get("validation_result")
    if vr is not None and vr not in ("PASS", "FAIL", "RETRY"):
        warnings.append(f"validation_result='{vr}' not in {{PASS, FAIL, RETRY}}")

    qt = state.get("query_type")
    _VALID_QUERY_TYPES = {
        None, "property_qa", "hours_schedule", "greeting", "off_topic",
        "gambling_advice", "action_request", "ambiguous",
        "responsible_gaming", "age_verification", "bsa_aml",
        "patron_privacy", "injection",
    }
    if qt not in _VALID_QUERY_TYPES:
        warnings.append(f"query_type='{qt}' not a recognized category")

    return warnings
