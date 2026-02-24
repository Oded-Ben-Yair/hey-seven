"""State schema for the Property Q&A custom StateGraph.

Defines PropertyQAState as a TypedDict with fields for the graph nodes,
plus Pydantic models for structured LLM outputs (router + validation).

v2 additions: ``extracted_fields``, ``whisper_plan`` for specialist agent
routing and background planning.
"""

from typing import Annotated, Any, Literal, NotRequired, TypedDict

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
    """Reducer that merges extracted fields across turns (latest non-None wins).

    Design decision: "latest wins" is intentional for guest profiling.
    When a guest corrects their name ("Actually, I'm Sarah not Sara"),
    the new value should overwrite the old one. This is a conscious
    choice for guest data where the most recent extraction is the most
    accurate.

    R37 fix M-008: Filter None values from ``b`` before merging. Without
    this, an extraction that finds no name returns ``{"name": None}``
    which overwrites a previously-extracted name. Only non-None values
    should overwrite existing data.

    When ``_initial_state()`` passes ``{}``, ``{**existing, **{}} == existing``
    so accumulated fields persist across turns. When extraction produces
    new fields, they merge into the existing dict. When the same key is
    re-extracted with a non-None value, the newer value wins.
    """
    # R38 fix C-003: Also filter empty strings from b. An extraction or CRM
    # import returning {"name": ""} would overwrite a previously-extracted
    # valid name. Empty string is not a valid guest data value.
    return {**a, **{k: v for k, v in b.items() if v is not None and v != ""}}


def _keep_max(a: int, b: int) -> int:
    """Reducer that preserves the maximum value across state updates.

    Used for session-level counters (e.g., responsible_gaming_count) that
    must accumulate across turns.  When _initial_state() passes 0,
    max(existing, 0) preserves the count.  When a node increments,
    max(existing, new) updates the count.
    """
    # R38 fix M-007: Guard against None input from buggy nodes.
    # max(5, None) raises TypeError in Python. Defensive: treat None as 0.
    # R39 fix M-006: Use explicit None check instead of `or 0`. The `or`
    # idiom conflates False, 0, None, and "" — `False or 0` evaluates to 0,
    # silently resetting the counter if a node returns bool instead of int.
    return max(0 if a is None else a, 0 if b is None else b)


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

    R44 fix D2-M001: Added ``rrf_score`` field alongside ``score``.
    ``score`` is the raw cosine similarity (quality gate); ``rrf_score``
    is the RRF fusion score (ranking metric).
    """

    content: str
    metadata: dict[str, Any]
    score: float
    rrf_score: NotRequired[float]


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
    # R37 fix C-001: Persist specialist name so RETRY re-uses the same
    # specialist instead of re-dispatching (which wastes tokens and risks
    # non-deterministic specialist switching).
    specialist_name: str | None
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


