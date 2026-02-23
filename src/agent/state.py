"""State schema for the Property Q&A custom StateGraph.

Defines PropertyQAState as a TypedDict with fields for the graph nodes,
plus Pydantic models for structured LLM outputs (router + validation).

v2 additions: ``extracted_fields``, ``whisper_plan`` for specialist agent
routing and background planning.
"""

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

__all__ = [
    "PropertyQAState",
    "RouterOutput",
    "DispatchOutput",
    "ValidationResult",
    "RetrievedChunk",
    "GuestContext",
]


def _merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Reducer that merges extracted fields across turns (latest value wins).

    Design decision: "latest wins" is intentional for guest profiling.
    When a guest corrects their name ("Actually, I'm Sarah not Sara"),
    the new value should overwrite the old one. This is a conscious
    choice for guest data where the most recent extraction is the most
    accurate.

    When ``_initial_state()`` passes ``{}``, ``{**existing, **{}} == existing``
    so accumulated fields persist across turns. When extraction produces
    new fields, they merge into the existing dict. When the same key is
    re-extracted, the newer value wins.

    If conflict detection is needed in the future, add a
    ``_merge_dicts_with_conflicts`` variant that logs when an existing
    non-None value is overwritten.
    """
    return {**a, **b}


def _keep_max(a: int, b: int) -> int:
    """Reducer that preserves the maximum value across state updates.

    Used for session-level counters (e.g., responsible_gaming_count) that
    must accumulate across turns.  When _initial_state() passes 0,
    max(existing, 0) preserves the count.  When a node increments,
    max(existing, new) updates the count.
    """
    return max(a, b)


def _keep_truthy(a: bool, b: bool) -> bool:
    """Reducer that preserves True once set (sticky flag).

    Once a suggestion has been offered, it stays offered for the session.
    When _initial_state() passes False, ``False or existing_True`` = True
    (preserved).  When a node sets True, ``existing_False or True`` = True
    (updated).
    """
    return a or b


class RetrievedChunk(TypedDict):
    """A single chunk returned by the RAG retriever.

    Explicit schema prevents implicit dict contracts from drifting
    between pipeline.py and nodes.py.
    """

    content: str
    metadata: dict[str, Any]
    score: float


class GuestContext(TypedDict, total=False):
    """Structured guest context passed to specialist agents.

    Populated by _dispatch_to_specialist via get_agent_context(extracted_fields).
    Using TypedDict instead of dict[str, Any] for IDE autocompletion and
    documentation of expected keys.

    total=False: all fields optional — not every turn extracts every field.
    """

    name: str | None
    party_size: int | None
    occasion: str | None
    visit_date: str | None
    dietary: str | None
    preferences: list[str]


class PropertyQAState(TypedDict):
    """Typed state flowing through the property Q&A graph.

    ``messages`` is persisted across turns via the checkpointer's ``add_messages``
    reducer. ``responsible_gaming_count`` persists via ``_keep_max`` reducer
    for session-level escalation tracking. ``extracted_fields`` persists via
    ``_merge_dicts`` reducer for multi-turn guest profiling. All other fields
    are **per-turn** — they are reset by ``_initial_state()`` at the start of
    each ``chat()`` / ``chat_stream()`` call.

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
    extracted_fields: Annotated[dict[str, Any], _merge_dicts]  # structured fields from guest message (persists across turns via reducer)
    whisper_plan: dict[str, Any] | None  # background planner output (WhisperPlan.model_dump())
    # v3 fields (Phase 3: Agent Quality Revolution)
    guest_sentiment: str | None  # positive/negative/neutral/frustrated (from VADER)
    # Populated by _dispatch_to_specialist via get_agent_context(extracted_fields).
    guest_context: GuestContext
    # Denormalized from extracted_fields["name"] for O(1) access in
    # persona_envelope_node (which runs on every response turn).
    # Updated by _dispatch_to_specialist when guest_profile_enabled.
    guest_name: str | None  # extracted guest name for personalization
    # _keep_max reducer: preserves the maximum value across state updates.
    # When _initial_state() resets this to 0, max(existing, 0) preserves
    # the accumulated count. When compliance_gate increments, the new value
    # is preserved. This prevents accidental reset of the escalation counter
    # when per-turn fields are reset via _initial_state().
    responsible_gaming_count: Annotated[int, _keep_max]
    # v4 fields (Phase 4: R21-R23)
    # _keep_truthy reducer: once True, stays True for the session.
    # _initial_state() passes False; ``False or existing_True`` = True (preserved).
    # R23 fix C-003: enforces max-1-suggestion-per-conversation.
    suggestion_offered: Annotated[bool, _keep_truthy]


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


class DispatchOutput(BaseModel):
    """Structured output for specialist agent dispatch.

    Used by ``_dispatch_to_specialist()`` to route queries to the
    appropriate specialist agent via LLM classification instead of
    keyword counting.  Falls back to keyword counting when the LLM
    is unavailable (circuit breaker open, parsing failure, network error).
    """
    specialist: Literal["dining", "entertainment", "comp", "hotel", "host"] = Field(
        description="Which specialist agent should handle this query"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the routing decision (0.0 to 1.0)"
    )
    reasoning: str = Field(
        max_length=200,
        description="Brief explanation of why this specialist was chosen"
    )


class ValidationResult(BaseModel):
    """Structured output from the validation node."""
    status: Literal["PASS", "FAIL", "RETRY"] = Field(
        description="PASS if the response meets all 6 criteria, RETRY for minor issues worth correcting, FAIL for serious violations"
    )
    reason: str = Field(
        description="Why the response passed or failed validation",
        max_length=500,
    )


