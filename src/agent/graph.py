"""Custom 13-node StateGraph for property Q&A.

Topology:
    START -> compliance_gate -> {greeting, off_topic, router}
    router -> {greeting, off_topic, retrieve}
    retrieve -> whisper_planner -> pre_extract -> generate -> validate -> {persona_envelope -> respond, generate (RETRY), fallback}

The ``generate`` node runs ``_dispatch_to_specialist()`` which orchestrates
three extracted helpers (see ``dispatch.py``):

- ``_route_to_specialist()``: Structured LLM dispatch + keyword fallback routing.
- ``_inject_guest_context()``: Guest profile injection (fail-silent).
- ``_execute_specialist()``: Agent execution with timeout + result sanitization.

The agent is resolved via ``get_agent()`` from the registry.  The node name
``"generate"`` is preserved for SSE streaming compatibility.

Concurrency Model
-----------------
- **Max concurrent streams**: Bounded by Cloud Run ``--concurrency=50`` (container-level).
  Each ``chat_stream()`` call runs as an independent async generator tracked by
  ``_active_streams`` in ``app.py``.  No graph-level concurrency limit; backpressure
  is at the LLM call layer via ``asyncio.Semaphore(20)`` in ``_base.py``.
- **Checkpointer thread safety**: ``MemorySaver`` (local dev) is thread-safe but
  serializes writes.  ``FirestoreSaver`` (production) supports concurrent read/write
  across instances.  Concurrent writes to the *same* ``thread_id`` are last-write-wins
  — acceptable because each thread_id maps to a single user session.
- **Singleton safety**: LLM clients, circuit breaker, retriever, and settings are
  TTL-cached with ``asyncio.Lock`` (async) or ``threading.Lock`` (sync) guards.
  Separate locks per client type (main LLM, validator LLM) prevent cascading stalls
  during credential refresh.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.api.pii_redaction import redact_pii
from src.agent.streaming_pii import StreamingPIIRedactor
from src.casino.feature_flags import DEFAULT_FEATURES
from src.config import get_settings
from src.observability.langfuse_client import get_langfuse_handler

from .compliance_gate import compliance_gate_node
from .dispatch import (
    _dispatch_to_specialist,
    _extract_node_metadata,
    # Backward-compat re-exports for tests
    _CATEGORY_TO_AGENT,
    _CATEGORY_PRIORITY,
    _DISPATCH_OWNED_KEYS,
    _DISPATCH_PROMPT,
    _VALID_STATE_KEYS,
    _execute_specialist,
    _inject_guest_context,
    _keyword_dispatch,
    _route_to_specialist,
)
from .pre_extract import pre_extract_node
from .profiling import profiling_enrichment_node
from .whisper_planner import whisper_planner_node
from .nodes import (
    _normalize_content,
    fallback_node,
    greeting_node,
    off_topic_node,
    respond_node,
    retrieve_node,
    route_from_router,
    router_node,
    validate_node,
)
from .constants import (
    NODE_COMPLIANCE_GATE,
    NODE_FALLBACK,
    NODE_GENERATE,
    NODE_GREETING,
    NODE_OFF_TOPIC,
    NODE_PERSONA,
    NODE_PRE_EXTRACT,
    NODE_PROFILING,
    NODE_RESPOND,
    NODE_RETRIEVE,
    NODE_ROUTER,
    NODE_VALIDATE,
    NODE_WHISPER,
    _KNOWN_NODES,
    _NON_STREAM_NODES,
)
from .persona import persona_envelope_node
from .state import PropertyQAState

logger = logging.getLogger(__name__)

__all__ = ["build_graph", "chat", "chat_stream"]


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_from_compliance(state: PropertyQAState) -> str:
    """Route after compliance gate based on whether guardrails triggered.

    Two-layer routing rationale (compliance_gate -> router):
      - **compliance_gate** (Layer 1): Deterministic regex-based guardrails.
        Zero-cost, zero-latency safety net that catches safety-critical
        queries (prompt injection, responsible gaming, BSA/AML, patron
        privacy, age verification) without an LLM call.
      - **router** (Layer 2): LLM-based classification via structured
        output. Required for nuanced intent classification (property_qa
        vs hours_schedule vs ambiguous) that regex cannot handle.

    These layers are NOT redundant — they implement defense-in-depth.
    Compliance gate provides a hard deterministic floor; the router
    adds intelligent classification on top.  Removing either layer
    creates a gap: no compliance gate = safety depends on prompt
    engineering alone; no router = every query hits RAG retrieval.

    If ``query_type`` is ``None``, all guardrails passed and LLM
    classification is needed (route to router).  Otherwise, route
    directly to the appropriate terminal node.
    """
    query_type = state.get("query_type")
    if query_type is None:
        return NODE_ROUTER
    if query_type == "greeting":
        return NODE_GREETING
    # All other guardrail-triggered types go to off_topic
    return NODE_OFF_TOPIC


def _route_after_validate_v2(state: PropertyQAState) -> str:
    """Route after validate node — v2 sends PASS to persona_envelope.

    Returns the name of the next node to execute.
    """
    result = state.get("validation_result", "PASS")
    if result == "PASS":
        return NODE_PERSONA
    if result == "RETRY":
        return NODE_GENERATE
    if result == "FAIL":
        return NODE_FALLBACK
    # Defensive: unexpected validation_result value — log and fail safe.
    logger.warning(
        "Unexpected validation_result=%r, routing to fallback (fail-safe)",
        result,
    )
    return NODE_FALLBACK


def build_graph(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Build the custom 13-node property Q&A graph (v2.4).

    v2.3 topology:
        START → compliance_gate → {greeting, off_topic, router}
        router → {greeting, off_topic, retrieve}
        retrieve → whisper_planner → generate (specialist dispatch) → profiling_enrichment → validate → {persona_envelope → respond, generate (RETRY), fallback}

    Args:
        checkpointer: Optional checkpointer for conversation persistence.
            Defaults to MemorySaver for local development.

    Returns:
        A compiled StateGraph ready for invoke/astream.
    """
    settings = get_settings()
    graph = StateGraph(PropertyQAState)

    # Add all 13 nodes
    graph.add_node(NODE_COMPLIANCE_GATE, compliance_gate_node)
    graph.add_node(NODE_ROUTER, router_node)
    graph.add_node(NODE_RETRIEVE, retrieve_node)
    graph.add_node(NODE_WHISPER, whisper_planner_node)
    graph.add_node(NODE_PRE_EXTRACT, pre_extract_node)
    graph.add_node(
        NODE_GENERATE, _dispatch_to_specialist
    )  # v2.2: dispatches to specialist via registry
    graph.add_node(NODE_VALIDATE, validate_node)
    graph.add_node(NODE_PERSONA, persona_envelope_node)
    graph.add_node(NODE_RESPOND, respond_node)
    graph.add_node(NODE_FALLBACK, fallback_node)
    graph.add_node(NODE_GREETING, greeting_node)
    graph.add_node(NODE_OFF_TOPIC, off_topic_node)
    graph.add_node(NODE_PROFILING, profiling_enrichment_node)

    # Edge map — v2 topology
    # START → compliance_gate
    graph.add_edge(START, NODE_COMPLIANCE_GATE)

    # compliance_gate → {greeting, off_topic, router}
    graph.add_conditional_edges(
        NODE_COMPLIANCE_GATE,
        route_from_compliance,
        {
            NODE_ROUTER: NODE_ROUTER,
            NODE_GREETING: NODE_GREETING,
            NODE_OFF_TOPIC: NODE_OFF_TOPIC,
        },
    )

    # router → {retrieve, greeting, off_topic} (defense-in-depth: router still has guardrails)
    graph.add_conditional_edges(
        NODE_ROUTER,
        route_from_router,
        {
            NODE_RETRIEVE: NODE_RETRIEVE,
            NODE_GREETING: NODE_GREETING,
            NODE_OFF_TOPIC: NODE_OFF_TOPIC,
        },
    )

    # -------------------------------------------------------------------
    # Feature Flag Architecture (Dual-Layer Design)
    # -------------------------------------------------------------------
    # LAYER 1 — BUILD TIME (graph topology):
    #   Feature flags that control GRAPH TOPOLOGY (which nodes exist,
    #   which edges connect them) are evaluated once at startup via
    #   DEFAULT_FEATURES (sync).  This is mandatory because LangGraph
    #   compiles the graph once — per-request graph compilation would
    #   be expensive (~40ms+ per request) and fragile.
    #   Example: whisper_planner_enabled removes the whisper_planner
    #   node entirely.
    #
    # LAYER 2 — RUNTIME (per-request behavior):
    #   Feature flags that control BEHAVIOR WITHIN NODES are evaluated
    #   per-request via the async is_feature_enabled(casino_id, flag)
    #   API, supporting per-casino overrides stored in Firestore.
    #   Examples: ai_disclosure_enabled, specialist_agents_enabled,
    #   comp_agent_enabled.
    #
    # WHY NOT ALL RUNTIME?
    #   Topology flags cannot be runtime without per-request graph
    #   compilation.  Per-request compilation adds ~40ms+ latency and
    #   breaks LangGraph's checkpointer assumptions (checkpoint
    #   references specific node names).
    #
    # EMERGENCY DISABLE:
    #   To disable whisper_planner during an incident: restart the
    #   container with FEATURE_FLAGS='{"whisper_planner_enabled": false}'
    #   env var.  Cloud Run supports rolling restarts with zero downtime.
    #
    # See also:
    #   - src/casino/feature_flags.py  (DEFAULT_FEATURES, async API)
    #   - src/agent/whisper_planner.py (runtime flag guard)
    #   - src/agent/agents/_base.py    (runtime specialist dispatch)
    # -------------------------------------------------------------------
    whisper_enabled = DEFAULT_FEATURES.get("whisper_planner_enabled", True)
    profiling_enabled = DEFAULT_FEATURES.get("profiling_enabled", True)
    logger.info(
        "Feature flags at graph build: %s",
        {k: v for k, v in DEFAULT_FEATURES.items() if v is not False},
    )

    if whisper_enabled:
        graph.add_edge(NODE_RETRIEVE, NODE_WHISPER)
        graph.add_edge(NODE_WHISPER, NODE_PRE_EXTRACT)
    else:
        graph.add_edge(NODE_RETRIEVE, NODE_PRE_EXTRACT)
    graph.add_edge(NODE_PRE_EXTRACT, NODE_GENERATE)

    if profiling_enabled:
        graph.add_edge(NODE_GENERATE, NODE_PROFILING)
        graph.add_edge(NODE_PROFILING, NODE_VALIDATE)
    else:
        graph.add_edge(NODE_GENERATE, NODE_VALIDATE)

    # validate → {persona_envelope (PASS), generate (RETRY), fallback (FAIL)}
    graph.add_conditional_edges(
        NODE_VALIDATE,
        _route_after_validate_v2,
        {
            NODE_PERSONA: NODE_PERSONA,
            NODE_GENERATE: NODE_GENERATE,
            NODE_FALLBACK: NODE_FALLBACK,
        },
    )

    # persona_envelope → respond → END
    graph.add_edge(NODE_PERSONA, NODE_RESPOND)
    graph.add_edge(NODE_RESPOND, END)

    # Terminal nodes
    graph.add_edge(NODE_FALLBACK, END)
    graph.add_edge(NODE_GREETING, END)
    graph.add_edge(NODE_OFF_TOPIC, END)

    if checkpointer is None:
        # Default to MemorySaver for local development and testing.
        # Production deployment uses FirestoreSaver for cross-request persistence:
        #   from langgraph.checkpoint.firestore import FirestoreSaver
        #   checkpointer = FirestoreSaver(project=settings.FIRESTORE_PROJECT)
        # Cloud Run single-container: MemorySaver is sufficient (conversation
        # state lives for the container lifetime). Multi-container: requires
        # FirestoreSaver for consistent conversation history across instances.
        checkpointer = MemorySaver()

    # HITL interrupt: when enabled, the graph pauses before the generate node
    # so a human operator can review/approve the retrieved context before
    # the LLM generates a response.
    interrupt_before = [NODE_GENERATE] if settings.ENABLE_HITL_INTERRUPT else None

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )
    compiled.recursion_limit = settings.GRAPH_RECURSION_LIMIT  # type: ignore[attr-defined]
    logger.info("Custom 12-node StateGraph compiled successfully.")
    return compiled


def _initial_state(message: str) -> dict[str, Any]:
    """Build a fresh per-turn state dict (DRY helper for chat/chat_stream).

    Non-message fields are reset every turn. Only ``messages`` persists across
    turns via the checkpointer's ``add_messages`` reducer — see
    ``PropertyQAState`` docstring for details.
    """
    now = datetime.now(tz=timezone.utc).strftime("%A, %B %d, %Y %I:%M %p UTC")
    return {
        "messages": [HumanMessage(content=message)],
        "current_time": now,
        "query_type": None,
        "router_confidence": 0.0,
        "retrieved_context": [],
        "validation_result": None,
        "retry_count": 0,
        "skip_validation": False,
        "retry_feedback": None,
        "sources_used": [],
        # v2 fields
        "extracted_fields": {},
        "whisper_plan": None,
        "responsible_gaming_count": 0,
        # v3 fields (Phase 3)
        "guest_sentiment": None,
        "guest_context": {},
        "guest_name": None,
        # R37 fix C-001: specialist name persisted for retry path
        "specialist_name": None,
        # R52 fix D1: dispatch method for observability
        "dispatch_method": None,
        # v4 fields (Phase 4: R23 fix C-003)
        "suggestion_offered": False,  # _keep_truthy: once True, stays True for the session
        # v5 fields (R72: Behavioral Excellence)
        "domains_discussed": [],  # _append_unique: accumulates across turns
        # R73: Crisis context persistence
        "crisis_active": False,  # _keep_truthy: once True, stays True for session
        # R81 fix: crisis turn counter for response variation
        "crisis_turn_count": 0,
        # Phase 1: Language detection for multilingual support
        "detected_language": None,
        # Phase 5: Structured handoff request for human host transfer
        "handoff_request": None,
        # Profiling Intelligence System
        "profiling_phase": None,  # _keep_latest_str: persists across turns
        "profile_completeness_score": 0.0,
        "profiling_question_injected": False,
        # R83: Model routing observability
        "model_used": None,
        # R92: Booking intent signal for specialist pipeline
        "booking_intent": None,
    }


# Parity check: ensure _initial_state covers all non-messages fields.
# R10 fix (Gemini F10, GPT P2-F5): converted from `assert` (vanishes with
# `python -O`) to a runtime ValueError that fires regardless of optimization
# mode. This catches state schema drift at import time in ALL environments.
_EXPECTED_FIELDS = frozenset(PropertyQAState.__annotations__) - frozenset({"messages"})
_INITIAL_FIELDS = frozenset(_initial_state("test").keys()) - frozenset({"messages"})
if _EXPECTED_FIELDS != _INITIAL_FIELDS:
    raise ValueError(
        f"_initial_state parity mismatch: "
        f"missing={_EXPECTED_FIELDS - _INITIAL_FIELDS}, "
        f"extra={_INITIAL_FIELDS - _EXPECTED_FIELDS}"
    )


async def chat(
    graph: CompiledStateGraph,
    message: str,
    thread_id: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Send a message to the graph and get a response.

    Args:
        graph: A compiled StateGraph (from build_graph).
        message: The user's message text.
        thread_id: Optional conversation thread ID for persistence.
        request_id: Optional X-Request-ID from HTTP middleware for
            end-to-end observability correlation (appears in LangFuse traces).

    Returns:
        A dict with response, thread_id, and sources.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if request_id:
        config["configurable"]["request_id"] = request_id
    handler = get_langfuse_handler(session_id=thread_id, request_id=request_id)
    if handler:
        config["callbacks"] = [handler]

    try:
        result = await graph.ainvoke(
            _initial_state(message),
            config=config,
        )
    except GraphRecursionError:
        # R42 fix D1-M001: Catch GraphRecursionError when the validate->generate
        # retry loop exceeds GRAPH_RECURSION_LIMIT. The retry_count bounds (max 1)
        # should prevent this, but a bug in retry tracking could trigger it.
        logger.error(
            "GraphRecursionError: recursion limit reached (thread_id=%s). "
            "This indicates a bug in retry_count tracking.",
            thread_id,
        )
        return {
            "response": (
                "I apologize, but I encountered an issue processing your request. "
                "Please try again, or contact us directly for assistance."
            ),
            "thread_id": thread_id,
            "sources": [],
        }

    # Extract final AI response
    messages = result.get("messages", [])
    response_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            response_text = _normalize_content(msg.content)
            break

    sources = result.get("sources_used", [])

    return {
        "response": response_text,
        "thread_id": thread_id,
        "sources": sources,
    }


def _source_key(s: dict | str) -> str:
    """Return a dedup key for a source entry (supports str and dict formats)."""
    if isinstance(s, dict):
        return f"{s.get('category', '')}:{s.get('source', '')}"
    return str(s)


def _merge_sources(
    target: list[dict | str],
    new_sources: list[dict | str],
) -> None:
    """Merge new source entries into target list, deduplicating by key."""
    existing_keys = {_source_key(x) for x in target}
    for s in new_sources:
        key = _source_key(s)
        if key not in existing_keys:
            existing_keys.add(key)
            target.append(s)


async def chat_stream(
    graph: CompiledStateGraph,
    message: str,
    thread_id: str | None = None,
    request_id: str | None = None,
) -> AsyncGenerator[dict[str, str], None]:
    """Stream a response from the graph as typed SSE events.

    Yields dicts with ``event`` and ``data`` keys suitable for
    ``EventSourceResponse``.

    Event types:
        metadata  – thread_id (sent first)
        token     – incremental text chunk (from generate node only)
        replace   – full response from non-streaming nodes (greeting, off_topic, fallback)
        sources   – cited knowledge-base categories
        handoff   – handoff request (self-harm, frustrated x3, incentive approval)
        done      – signals end of stream
        error     – on failure

    Args:
        graph: A compiled StateGraph (from build_graph).
        message: The user's message text.
        thread_id: Optional conversation thread ID for persistence.
        request_id: Optional X-Request-ID from HTTP middleware for
            end-to-end observability correlation (appears in LangFuse traces).
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if request_id:
        config["configurable"]["request_id"] = request_id
    handler = get_langfuse_handler(session_id=thread_id, request_id=request_id)
    if handler:
        config["callbacks"] = [handler]

    # R42 fix D4-M001: Emit retry:0 to disable browser EventSource auto-reconnect.
    # Without this, network blips cause the browser to reconnect and resend the
    # message, creating duplicate LLM calls and duplicate conversation messages.
    # Full Last-Event-ID reconnection support (caching + replay) is deferred to
    # post-MVP; for now, disabling auto-reconnect prevents the duplicate problem.
    yield {
        "event": "metadata",
        "data": json.dumps({"thread_id": thread_id, "retry": 0}),
    }

    sources: list[dict | str] = []  # Support both old str and new dict format
    _handoff_request: dict | None = (
        None  # R87: Capture handoff_request from terminal nodes
    )
    node_start_times: dict[str, float] = {}
    errored = False
    # Streaming PII redactor: buffers incoming tokens and applies regex-based
    # PII redaction before emitting to the client. Created fresh per request
    # to avoid cross-request state leakage.
    _pii_redactor = StreamingPIIRedactor()

    try:
        async for event in graph.astream_events(
            _initial_state(message),
            config=config,
            version="v2",
        ):
            kind = event.get("event")
            langgraph_node = event.get("metadata", {}).get("langgraph_node", "")

            # --- Graph node lifecycle: start ---
            if (
                kind == "on_chain_start"
                and langgraph_node in _KNOWN_NODES
                and langgraph_node not in node_start_times
            ):
                node_start_times[langgraph_node] = time.monotonic()
                yield {
                    "event": "graph_node",
                    "data": json.dumps({"node": langgraph_node, "status": "start"}),
                }

            # Stream tokens from generate node only — with inline PII redaction.
            # Tokens are fed to StreamingPIIRedactor which buffers and applies
            # regex-based PII redaction before releasing safe text. The redactor
            # retains a trailing lookahead window to catch PII patterns (phone
            # numbers, SSNs, card numbers) that span multiple tokens.
            if kind == "on_chat_model_stream" and langgraph_node == NODE_GENERATE:
                chunk = event.get("data", {}).get("chunk")
                if (
                    isinstance(chunk, AIMessageChunk)
                    and chunk.content
                    and not getattr(chunk, "tool_call_chunks", None)
                ):
                    content = _normalize_content(chunk.content)
                    for safe_chunk in _pii_redactor.feed(content):
                        yield {
                            "event": "token",
                            "data": json.dumps({"content": safe_chunk}),
                        }

            # Capture non-streaming node outputs (greeting, off_topic, fallback).
            # Apply PII redaction to replace events — these bypass the streaming
            # PII buffer (which only protects token events from the generate node).
            elif kind == "on_chain_end" and langgraph_node in _NON_STREAM_NODES:
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    msgs = output.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, AIMessage) and msg.content:
                            content = _normalize_content(msg.content)
                            # R39 fix M-005: Single-pass PII redaction. Previously
                            # called contains_pii() then redact_pii() — two full
                            # regex passes. redact_pii() returns the original text
                            # when no matches found, so the check is redundant.
                            content = redact_pii(content)
                            yield {
                                "event": "replace",
                                "data": json.dumps({"content": content}),
                            }

                    # Capture sources from non-streaming nodes.
                    # Wave 2: supports both old str format and new dict format
                    # with dedup by category:source key.
                    node_sources = output.get("sources_used", [])
                    _merge_sources(sources, node_sources)

            # Capture sources from respond node
            elif kind == "on_chain_end" and langgraph_node == NODE_RESPOND:
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    node_sources = output.get("sources_used", [])
                    _merge_sources(sources, node_sources)

            # --- Graph node lifecycle: complete ---
            if kind == "on_chain_end" and langgraph_node in node_start_times:
                duration_ms = int(
                    (time.monotonic() - node_start_times.pop(langgraph_node)) * 1000
                )
                output = event.get("data", {}).get("output", {})
                meta = _extract_node_metadata(
                    langgraph_node, output, duration_ms=duration_ms
                )
                yield {
                    "event": "graph_node",
                    "data": json.dumps(
                        {
                            "node": langgraph_node,
                            "status": "complete",
                            "duration_ms": duration_ms,
                            "metadata": meta,
                        }
                    ),
                }

                # R87: Capture handoff_request from any node output.
                # Set by off_topic_node (self-harm), _base.py (frustrated x3),
                # or incentive approval path (above-threshold comps).
                if (
                    isinstance(output, dict)
                    and "handoff_request" in output
                    and output["handoff_request"]
                ):
                    _handoff_request = output["handoff_request"]

    except asyncio.CancelledError:
        # Intentionally NOT flushing _pii_redactor on cancel: dropping
        # buffered tokens is safer than emitting potentially unredacted
        # PII to a disconnecting client. The partial tokens are lost
        # (fail-safe). R10 documentation fix (DeepSeek F6).
        logger.info(
            "SSE stream cancelled (client disconnect), dropping %d buffered chars",
            _pii_redactor.buffer_size,
        )
        raise
    except GraphRecursionError:
        # R42 fix D1-M001: Explicit handling for recursion limit exceeded.
        logger.error(
            "GraphRecursionError during SSE stream — retry_count tracking bug suspected"
        )
        errored = True
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "error": "I encountered an issue processing your request. Please try again."
                }
            ),
        }
    except Exception:
        logger.exception("Error during SSE stream")
        errored = True
        yield {
            "event": "error",
            "data": json.dumps(
                {"error": "An error occurred while generating the response."}
            ),
        }

    # Flush remaining PII redactor buffer (tokens accumulated but not yet emitted)
    if not errored:
        for safe_chunk in _pii_redactor.flush():
            yield {
                "event": "token",
                "data": json.dumps({"content": safe_chunk}),
            }

    # After an error, sources may contain stale/partial data from a partially-
    # completed graph execution — suppress to avoid confusing the client.
    # Wave 2: Emit richer provenance data alongside sources for citation support.
    if sources and not errored:
        yield {
            "event": "sources",
            "data": json.dumps({"sources": sources, "citations": sources}),
        }

    # R87: Emit handoff event when handoff_request is set (self-harm, frustrated x3,
    # incentive approval). Emitted after sources, before done — client can show
    # handoff UI while agent response is still visible.
    if _handoff_request and not errored:
        yield {
            "event": "handoff",
            "data": json.dumps(_handoff_request),
        }

    yield {
        "event": "done",
        "data": json.dumps({"done": True}),
    }
