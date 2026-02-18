"""Custom 11-node StateGraph for property Q&A (v2.1).

v1 (8 nodes): START → router → {greeting, off_topic, retrieve → generate → validate → respond/fallback}
v2 (10 nodes): START → compliance_gate → {greeting, off_topic, router → {greeting, off_topic, retrieve → generate → validate → persona_envelope → respond/fallback}}
v2.1 (11 nodes): v2 + whisper_planner between retrieve and generate

The ``generate`` node now runs ``host_agent`` (from ``agents.host_agent``)
instead of the v1 ``generate_node``.  The node name ``"generate"`` is
preserved for SSE streaming compatibility and test backward compat.
"""

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

from src.config import get_settings

from .agents.host_agent import host_agent
from .compliance_gate import compliance_gate_node
from .whisper_planner import whisper_planner_node
from .nodes import (
    fallback_node,
    greeting_node,
    off_topic_node,
    respond_node,
    retrieve_node,
    route_after_validate,  # noqa: F401 — kept for backward compat (tests import from nodes)
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
# Routing functions
# ---------------------------------------------------------------------------


def route_from_compliance(state: PropertyQAState) -> str:
    """Route after compliance gate based on whether guardrails triggered.

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
    # FAIL
    return NODE_FALLBACK


def build_graph(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Build the custom 11-node property Q&A graph (v2.1).

    v2.1 topology:
        START → compliance_gate → {greeting, off_topic, router}
        router → {greeting, off_topic, retrieve}
        retrieve → whisper_planner → generate (host_agent) → validate → {persona_envelope → respond, generate (RETRY), fallback}

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
    graph.add_node(NODE_GENERATE, host_agent)  # v2: host_agent replaces generate_node
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

    # retrieve → whisper_planner → generate (host_agent) → validate
    graph.add_edge(NODE_RETRIEVE, NODE_WHISPER)
    graph.add_edge(NODE_WHISPER, NODE_GENERATE)
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
    }


async def chat(
    graph: CompiledStateGraph,
    message: str,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Send a message to the graph and get a response.

    Args:
        graph: A compiled StateGraph (from build_graph).
        message: The user's message text.
        thread_id: Optional conversation thread ID for persistence.

    Returns:
        A dict with response, thread_id, and sources.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

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
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    yield {
        "event": "metadata",
        "data": json.dumps({"thread_id": thread_id}),
    }

    sources: list[str] = []
    node_start_times: dict[str, float] = {}

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

    except Exception:
        logger.exception("Error during SSE stream")
        yield {
            "event": "error",
            "data": json.dumps({"error": "An error occurred while generating the response."}),
        }

    if sources:
        yield {
            "event": "sources",
            "data": json.dumps({"sources": sources}),
        }

    yield {
        "event": "done",
        "data": json.dumps({"done": True}),
    }
