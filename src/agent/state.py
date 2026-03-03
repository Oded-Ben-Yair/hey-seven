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
    "UNSET_SENTINEL",
]

# R47 fix C7: Tombstone sentinel for explicit field deletion in _merge_dicts.
# Guest says "remove the peanut allergy" → LLM returns {"dietary": UNSET_SENTINEL}
# → _merge_dicts pops "dietary" from accumulated state.
#
# R49 fix: Changed from object() back to string, but with UUID-namespaced prefix
# that cannot appear in natural language input. R48's object() sentinel was correct
# for collision prevention but fails JSON serialization through FirestoreSaver
# (Grok MAJOR-D3-001, GPT-5.2 CRITICAL, DeepSeek CRITICAL — 3/4 models agreed).
# The UUID prefix "$$UNSET:7a3f..." makes accidental collision astronomically unlikely
# while surviving JSON roundtrip.
UNSET_SENTINEL = "$$UNSET:7a3f9c2e-b1d4-4e8a-9f5c-3a7d2e1b0c8f$$"


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
    #
    # R47 fix C7: Tombstone pattern for explicit deletion. If v == UNSET_SENTINEL,
    # pop the key from merged dict. This allows LLM to explicitly clear a field
    # (e.g., "remove the peanut allergy" → {"dietary": "__UNSET__"}).
    # Without this, _merge_dicts filters None, making fields "sticky" — once set,
    # they can never be unset.
    # R52 fix D3: Guard against None input from buggy nodes.
    # A node returning {"extracted_fields": None} would crash on .items().
    if not b:
        return dict(a) if a else {}
    if not a:
        a = {}
    merged = dict(a)
    for k, v in b.items():
        if v == UNSET_SENTINEL:
            merged.pop(k, None)
        elif v is not None and v != "":
            merged[k] = v
    return merged


def _append_unique(a: list[str] | None, b: list[str] | None) -> list[str]:
    """Reducer that accumulates unique strings across turns.

    Used for ``domains_discussed`` to track which specialist domains
    (dining, entertainment, hotel, etc.) have been covered in this session.
    Deduplicates: if "dining" is already in the list, adding it again is a no-op.

    R72 B4: Enables cross-domain suggestion — when guest asks "what else?",
    the agent suggests categories NOT yet discussed.
    """
    existing = list(a) if a else []
    if not b:
        return existing
    seen = set(existing)
    for item in b:
        if item and item not in seen:
            seen.add(item)
            existing.append(item)
    return existing


def _keep_latest_str(a: str | None, b: str | None) -> str | None:
    """Reducer that keeps the latest non-None string value.

    Used for guest_name: when extraction updates the name, the new value
    should win. When _initial_state() passes None, the existing name persists.
    """
    if b is not None and b != "":
        return b
    return a


def _keep_max(a: int | None, b: int | None) -> int:
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

    R48 fix: Wrap in bool() to enforce type contract. Without this,
    ``None or None`` returns ``None`` (not ``False``), violating the
    ``Annotated[bool, _keep_truthy]`` type annotation. A buggy node
    returning None would corrupt the bool field.
    """
    return bool(a or b)


class RetrievedChunk(TypedDict):
    """A single chunk returned by the RAG retriever.

    Explicit schema prevents implicit dict contracts from drifting
    between pipeline.py and nodes.py.

    R44 fix D2-M001: Added ``rrf_score`` field alongside ``score``.
    ``score`` is the raw cosine similarity (quality gate); ``rrf_score``
    is the RRF fusion score (ranking metric).

    Wave 2 fix D2: Added ``source_name`` for human-readable provenance
    in citation metadata emitted via the SSE ``sources`` event.
    """

    content: str
    metadata: dict[str, Any]
    score: float
    rrf_score: NotRequired[float]
    source_name: NotRequired[str]


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

    Field categories:
        **Persistent** (survive across turns via custom reducers):
        - messages: Conversation history (add_messages reducer)
        - extracted_fields: Guest profile data (_merge_dicts reducer)
        - responsible_gaming_count: Session escalation counter (_keep_max reducer)
        - suggestion_offered: Sticky flag for max-1 suggestion (_keep_truthy reducer)

        **Ephemeral** (reset by _initial_state() each turn):
        - query_type, router_confidence: Router classification
        - retrieved_context, sources_used: RAG results
        - validation_result, retry_count, skip_validation, retry_feedback: Validate loop
        - current_time: Injected timestamp
        - whisper_plan: Background planner output
        - guest_sentiment, guest_context, guest_name: Guest profiling
        - specialist_name, dispatch_method: Dispatch observability

    ``skip_validation`` is set to ``True`` by the generate node (host_agent)
    when the response is a deterministic fallback (empty context, LLM error,
    circuit breaker open) that does not need adversarial validation.
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
    sources_used: list[dict[str, Any] | str]  # knowledge-base sources (dict provenance or str category)
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
    guest_name: Annotated[str | None, _keep_latest_str]  # extracted guest name for personalization
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
    # R52 fix D1: Persist dispatch method for observability and debugging.
    # Values: structured_output / keyword_fallback / retry_reuse / feature_flag_override
    dispatch_method: str | None
    # v4 fields (Phase 4: R21-R23)
    # _keep_truthy reducer: once True, stays True for the session.
    # _initial_state() passes False; ``False or existing_True`` = True (preserved).
    # R23 fix C-003: enforces max-1-suggestion-per-conversation.
    suggestion_offered: Annotated[bool, _keep_truthy]
    # v5 fields (R72: Behavioral Excellence)
    # _append_unique reducer: accumulates specialist domain names across turns.
    # When _initial_state() passes [], existing domains persist across turns.
    # Used for cross-domain suggestion: when guest asks "what else?", suggest
    # categories NOT yet discussed.
    domains_discussed: Annotated[list[str], _append_unique]
    # R73: Crisis context persistence. Once crisis is detected (self_harm or
    # crisis_immediate/urgent), this flag stays True for the session.
    # _keep_truthy reducer: once True, stays True — ensures follow-up turns
    # maintain crisis awareness even if the specific message doesn't re-trigger
    # crisis regex patterns. Reset only on new session (new thread_id).
    crisis_active: Annotated[bool, _keep_truthy]
    # R81 fix: Track consecutive crisis turns for response adaptation.
    # _keep_max reducer: incremented by off_topic_node self_harm branch.
    # Reset only on new session. Allows empathetic variation on turns 2+.
    crisis_turn_count: Annotated[int, _keep_max]
    # Phase 1: Language detection for multilingual support.
    # Set by router_node from RouterOutput.detected_language.
    # Used by greeting_node, off_topic_node, and execute_specialist to select
    # language-appropriate responses. Feature-gated by spanish_support_enabled.
    detected_language: str | None
    # Phase 5: Structured handoff request for human host transfer.
    # Set by off_topic_node (self_harm) and execute_specialist (frustration >= 3).
    # dict | None because HandoffRequest is serialized via .model_dump().
    handoff_request: dict[str, Any] | None
    # Profiling Intelligence System fields.
    # profiling_phase: Golden path stage (foundation → preference → relationship → behavioral).
    # Set by profiling_enrichment_node. Reset per-turn via _initial_state().
    profiling_phase: Annotated[str | None, _keep_latest_str]
    # profile_completeness_score: Weighted 0.0-1.0 completeness of guest profile.
    # Computed by profiling_enrichment_node each turn.
    profile_completeness_score: float
    # profiling_question_injected: Ephemeral per-turn flag indicating a profiling
    # question was appended to the AI response in this turn.
    profiling_question_injected: bool
    # R83: Model routing observability. Records which model was used for this turn
    # (e.g., "gemini-3-flash-preview" or "gemini-3.1-pro-preview").
    # Ephemeral per-turn — reset by _initial_state().
    model_used: str | None


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
    detected_language: Literal["en", "es", "other"] = Field(
        default="en",
        description="Detected language of the user message (en=English, es=Spanish, other=unsupported)"
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
        max_length=500,
        description="Brief explanation of why this specialist was chosen"
    )


class ValidationResult(BaseModel):
    """Structured output from the validation node."""
    status: Literal["PASS", "FAIL", "RETRY"] = Field(
        description="PASS if the response meets all 6 criteria, RETRY for minor issues worth correcting, FAIL for serious violations"
    )
    reason: str = Field(
        default="",
        description="Why the response passed or failed validation",
        max_length=500,
    )


