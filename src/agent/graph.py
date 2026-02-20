"""Custom 11-node StateGraph for property Q&A (v2.2).

v1 (8 nodes): START → router → {greeting, off_topic, retrieve → generate → validate → respond/fallback}
v2 (10 nodes): START → compliance_gate → {greeting, off_topic, router → {greeting, off_topic, retrieve → generate → validate → persona_envelope → respond/fallback}}
v2.1 (11 nodes): v2 + whisper_planner between retrieve and generate
v2.2: generate node dispatches to specialist agents (host, dining, entertainment, comp)
      via the agent registry based on dominant category in retrieved context.

The ``generate`` node runs ``_dispatch_to_specialist()`` which examines the
dominant category in ``retrieved_context`` metadata and routes to the
appropriate specialist agent via ``get_agent()`` from the registry.  The
node name ``"generate"`` is preserved for SSE streaming compatibility.
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
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.casino.feature_flags import DEFAULT_FEATURES, is_feature_enabled
from src.config import get_settings
from src.observability.langfuse_client import get_langfuse_handler

from .agents.registry import get_agent
from .compliance_gate import compliance_gate_node
from .whisper_planner import whisper_planner_node
from .nodes import (
    fallback_node,
    greeting_node,
    off_topic_node,
    respond_node,
    retrieve_node,
    route_from_router,
    router_node,
    validate_node,
)
from .persona import persona_envelope_node
from .state import PropertyQAState

logger = logging.getLogger(__name__)

__all__ = ["build_graph", "chat", "chat_stream"]

# ---------------------------------------------------------------------------
# Node name constants — shared between build_graph() and chat_stream()
# to prevent silent breakage if a node is renamed.
# ---------------------------------------------------------------------------
NODE_ROUTER = "router"
NODE_RETRIEVE = "retrieve"
NODE_GENERATE = "generate"
NODE_VALIDATE = "validate"
NODE_RESPOND = "respond"
NODE_FALLBACK = "fallback"
NODE_GREETING = "greeting"
NODE_OFF_TOPIC = "off_topic"
NODE_COMPLIANCE_GATE = "compliance_gate"
NODE_PERSONA = "persona_envelope"
NODE_WHISPER = "whisper_planner"

_NON_STREAM_NODES = frozenset({
    NODE_GREETING, NODE_OFF_TOPIC, NODE_FALLBACK,
    NODE_COMPLIANCE_GATE, NODE_PERSONA, NODE_WHISPER,
})

_KNOWN_NODES = frozenset({
    NODE_ROUTER, NODE_RETRIEVE, NODE_GENERATE, NODE_VALIDATE,
    NODE_RESPOND, NODE_FALLBACK, NODE_GREETING, NODE_OFF_TOPIC,
    NODE_COMPLIANCE_GATE, NODE_PERSONA, NODE_WHISPER,
})


def _extract_node_metadata(node: str, output: Any) -> dict:
    """Extract per-node metadata for graph trace SSE events."""
    if not isinstance(output, dict):
        return {}
    if node == NODE_COMPLIANCE_GATE:
        return {
            "query_type": output.get("query_type"),
            "confidence": output.get("router_confidence"),
        }
    if node == NODE_ROUTER:
        return {
            "query_type": output.get("query_type"),
            "confidence": output.get("router_confidence"),
        }
    if node == NODE_RETRIEVE:
        ctx = output.get("retrieved_context", [])
        return {"doc_count": len(ctx)}
    if node == NODE_VALIDATE:
        return {"result": output.get("validation_result")}
    if node == NODE_RESPOND:
        return {"sources": output.get("sources_used", [])}
    if node == NODE_WHISPER:
        return {"has_plan": bool(output.get("whisper_plan"))}
    return {}


# ---------------------------------------------------------------------------
# Specialist agent dispatch
# ---------------------------------------------------------------------------

# Maps retrieved-context metadata categories to specialist agent names.
# Categories not listed here route to the "host" (general concierge) agent.
# "spa" → "entertainment": spa services are managed by the entertainment/amenities
# team at most casino properties; a separate spa agent would duplicate 90% of
# entertainment agent logic for minimal retrieval benefit.
_CATEGORY_TO_AGENT: dict[str, str] = {
    "restaurants": "dining",
    "entertainment": "entertainment",
    "spa": "entertainment",
    "gaming": "comp",
    "promotions": "comp",
    "hotel": "hotel",
}


async def _dispatch_to_specialist(state: PropertyQAState) -> dict[str, Any]:
    """Dispatch to the appropriate specialist agent based on retrieved context.

    Examines the dominant category in ``retrieved_context`` metadata and
    routes to the specialist with domain-specific prompts via the agent
    registry.  Falls back to ``host_agent`` for general or mixed queries.

    Dispatch logic:
    - Count category occurrences across all retrieved chunks.
    - If a dominant category maps to a specialist, dispatch to it.
    - Otherwise, use the general ``host`` concierge agent.
    - ``host_agent`` includes whisper planner guidance; specialists do not.
    """
    retrieved = state.get("retrieved_context", [])

    # Count categories in retrieved context
    category_counts: dict[str, int] = {}
    for chunk in retrieved:
        cat = chunk.get("metadata", {}).get("category", "")
        if cat:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    # Determine specialist from dominant category
    agent_name = "host"
    dominant = "none"
    if category_counts:
        dominant = max(category_counts, key=lambda k: (category_counts[k], k))
        candidate = _CATEGORY_TO_AGENT.get(dominant, "host")
        # Feature flag: specialist_agents_enabled controls dispatch to non-host agents.
        # When disabled, all queries route to the general host concierge.
        # Uses async is_feature_enabled() for multi-tenant support (per-casino overrides).
        if candidate != "host" and not await is_feature_enabled(get_settings().CASINO_ID, "specialist_agents_enabled"):
            logger.info("Specialist agents disabled; routing %s to host", candidate)
            candidate = "host"
        agent_name = candidate

    agent_fn = get_agent(agent_name)
    logger.info(
        "Dispatching to %s agent (dominant_category=%s, categories=%s)",
        agent_name,
        dominant,
        category_counts,
    )
    return await agent_fn(state)


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
    """Build the custom 11-node property Q&A graph (v2.2).

    v2.2 topology:
        START → compliance_gate → {greeting, off_topic, router}
        router → {greeting, off_topic, retrieve}
        retrieve → whisper_planner → generate (specialist dispatch) → validate → {persona_envelope → respond, generate (RETRY), fallback}

    Args:
        checkpointer: Optional checkpointer for conversation persistence.
            Defaults to MemorySaver for local development.

    Returns:
        A compiled StateGraph ready for invoke/astream.
    """
    settings = get_settings()
    graph = StateGraph(PropertyQAState)

    # Add all 11 nodes
    graph.add_node(NODE_COMPLIANCE_GATE, compliance_gate_node)
    graph.add_node(NODE_ROUTER, router_node)
    graph.add_node(NODE_RETRIEVE, retrieve_node)
    graph.add_node(NODE_WHISPER, whisper_planner_node)
    graph.add_node(NODE_GENERATE, _dispatch_to_specialist)  # v2.2: dispatches to specialist via registry
    graph.add_node(NODE_VALIDATE, validate_node)
    graph.add_node(NODE_PERSONA, persona_envelope_node)
    graph.add_node(NODE_RESPOND, respond_node)
    graph.add_node(NODE_FALLBACK, fallback_node)
    graph.add_node(NODE_GREETING, greeting_node)
    graph.add_node(NODE_OFF_TOPIC, off_topic_node)

    # Edge map — v2 topology
    # START → compliance_gate
    graph.add_edge(START, NODE_COMPLIANCE_GATE)

    # compliance_gate → {greeting, off_topic, router}
    graph.add_conditional_edges(NODE_COMPLIANCE_GATE, route_from_compliance, {
        NODE_ROUTER: NODE_ROUTER,
        NODE_GREETING: NODE_GREETING,
        NODE_OFF_TOPIC: NODE_OFF_TOPIC,
    })

    # router → {retrieve, greeting, off_topic} (defense-in-depth: router still has guardrails)
    graph.add_conditional_edges(NODE_ROUTER, route_from_router, {
        NODE_RETRIEVE: NODE_RETRIEVE,
        NODE_GREETING: NODE_GREETING,
        NODE_OFF_TOPIC: NODE_OFF_TOPIC,
    })

    # Feature flags consumed here (graph topology) and in nodes
    # (greeting_node, off_topic_node) and in agents/_base.py (execute_specialist).
    # Topology-altering flags must be checked at build time; runtime flags
    # are checked per-invocation inside node functions.
    # NOTE: whisper_planner_enabled uses DEFAULT_FEATURES (static) because it
    # controls GRAPH TOPOLOGY which is built once at startup (not in an async
    # context).  Runtime flags (ai_disclosure_enabled, specialist_agents_enabled)
    # use the async is_feature_enabled() API for per-casino overrides.
    whisper_enabled = DEFAULT_FEATURES.get("whisper_planner_enabled", True)
    logger.info(
        "Feature flags at graph build: %s",
        {k: v for k, v in DEFAULT_FEATURES.items() if v != False},  # noqa: E712 — intentional identity check for readability
    )

    if whisper_enabled:
        graph.add_edge(NODE_RETRIEVE, NODE_WHISPER)
        graph.add_edge(NODE_WHISPER, NODE_GENERATE)
    else:
        graph.add_edge(NODE_RETRIEVE, NODE_GENERATE)

    graph.add_edge(NODE_GENERATE, NODE_VALIDATE)

    # validate → {persona_envelope (PASS), generate (RETRY), fallback (FAIL)}
    graph.add_conditional_edges(NODE_VALIDATE, _route_after_validate_v2, {
        NODE_PERSONA: NODE_PERSONA,
        NODE_GENERATE: NODE_GENERATE,
        NODE_FALLBACK: NODE_FALLBACK,
    })

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
    logger.info("Custom 11-node StateGraph compiled successfully.")
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
    }


# Parity check: ensure _initial_state covers all non-messages fields.
# Guarded behind __debug__ so it is stripped in optimized mode (python -O),
# avoiding a production crash if state schema drifts between deploys.
if __debug__:
    _PARITY_FIELDS = set(PropertyQAState.__annotations__) - {"messages"}
    _INITIAL_FIELDS = set(_initial_state("test").keys()) - {"messages"}
    assert _PARITY_FIELDS == _INITIAL_FIELDS, (
        f"_initial_state parity mismatch: "
        f"missing={_PARITY_FIELDS - _INITIAL_FIELDS}, "
        f"extra={_INITIAL_FIELDS - _PARITY_FIELDS}"
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

    result = await graph.ainvoke(
        _initial_state(message),
        config=config,
    )

    # Extract final AI response
    messages = result.get("messages", [])
    response_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            response_text = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    sources = result.get("sources_used", [])

    return {
        "response": response_text,
        "thread_id": thread_id,
        "sources": sources,
    }


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

    yield {
        "event": "metadata",
        "data": json.dumps({"thread_id": thread_id}),
    }

    sources: list[str] = []
    node_start_times: dict[str, float] = {}
    errored = False

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

            # Stream tokens from generate node only
            if kind == "on_chat_model_stream" and langgraph_node == NODE_GENERATE:
                chunk = event.get("data", {}).get("chunk")
                if (
                    isinstance(chunk, AIMessageChunk)
                    and chunk.content
                    and not getattr(chunk, "tool_call_chunks", None)
                ):
                    content = (
                        chunk.content
                        if isinstance(chunk.content, str)
                        else str(chunk.content)
                    )
                    yield {
                        "event": "token",
                        "data": json.dumps({"content": content}),
                    }

            # Capture non-streaming node outputs (greeting, off_topic, fallback)
            elif kind == "on_chain_end" and langgraph_node in _NON_STREAM_NODES:
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    msgs = output.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, AIMessage) and msg.content:
                            content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            yield {
                                "event": "replace",
                                "data": json.dumps({"content": content}),
                            }

                    # Capture sources from non-streaming nodes
                    node_sources = output.get("sources_used", [])
                    for s in node_sources:
                        if s not in sources:
                            sources.append(s)

            # Capture sources from respond node
            elif kind == "on_chain_end" and langgraph_node == NODE_RESPOND:
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    node_sources = output.get("sources_used", [])
                    for s in node_sources:
                        if s not in sources:
                            sources.append(s)

            # --- Graph node lifecycle: complete ---
            if kind == "on_chain_end" and langgraph_node in node_start_times:
                duration_ms = int(
                    (time.monotonic() - node_start_times.pop(langgraph_node)) * 1000
                )
                output = event.get("data", {}).get("output", {})
                meta = _extract_node_metadata(langgraph_node, output)
                yield {
                    "event": "graph_node",
                    "data": json.dumps({
                        "node": langgraph_node,
                        "status": "complete",
                        "duration_ms": duration_ms,
                        "metadata": meta,
                    }),
                }

    except asyncio.CancelledError:
        # Client disconnect during SSE stream — normal, not an error.
        # Re-raise to let ASGI server handle cleanup.
        logger.info("SSE stream cancelled (client disconnect)")
        raise
    except Exception:
        logger.exception("Error during SSE stream")
        errored = True
        yield {
            "event": "error",
            "data": json.dumps({"error": "An error occurred while generating the response."}),
        }

    # After an error, sources may contain stale/partial data from a partially-
    # completed graph execution — suppress to avoid confusing the client.
    if sources and not errored:
        yield {
            "event": "sources",
            "data": json.dumps({"sources": sources}),
        }

    yield {
        "event": "done",
        "data": json.dumps({"done": True}),
    }
