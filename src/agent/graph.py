"""Custom 8-node StateGraph for property Q&A.

Nodes: router → retrieve → generate → validate → respond
Branches: greeting, off_topic, fallback
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

from .nodes import (
    fallback_node,
    generate_node,
    greeting_node,
    off_topic_node,
    respond_node,
    retrieve_node,
    route_after_validate,
    route_from_router,
    router_node,
    validate_node,
)
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

_NON_STREAM_NODES = frozenset({NODE_GREETING, NODE_OFF_TOPIC, NODE_FALLBACK})

_KNOWN_NODES = frozenset({
    NODE_ROUTER, NODE_RETRIEVE, NODE_GENERATE, NODE_VALIDATE,
    NODE_RESPOND, NODE_FALLBACK, NODE_GREETING, NODE_OFF_TOPIC,
})


def _extract_node_metadata(node: str, output: Any) -> dict:
    """Extract per-node metadata for graph trace SSE events."""
    if not isinstance(output, dict):
        return {}
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
    return {}


def build_graph(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Build the custom 8-node property Q&A graph.

    Args:
        checkpointer: Optional checkpointer for conversation persistence.
            Defaults to MemorySaver for local development.

    Returns:
        A compiled StateGraph ready for invoke/astream.
    """
    settings = get_settings()
    graph = StateGraph(PropertyQAState)

    # Add all 8 nodes
    graph.add_node(NODE_ROUTER, router_node)
    graph.add_node(NODE_RETRIEVE, retrieve_node)
    graph.add_node(NODE_GENERATE, generate_node)
    graph.add_node(NODE_VALIDATE, validate_node)
    graph.add_node(NODE_RESPOND, respond_node)
    graph.add_node(NODE_FALLBACK, fallback_node)
    graph.add_node(NODE_GREETING, greeting_node)
    graph.add_node(NODE_OFF_TOPIC, off_topic_node)

    # Edge map
    graph.add_edge(START, NODE_ROUTER)
    graph.add_conditional_edges(NODE_ROUTER, route_from_router, {
        NODE_RETRIEVE: NODE_RETRIEVE,
        NODE_GREETING: NODE_GREETING,
        NODE_OFF_TOPIC: NODE_OFF_TOPIC,
    })
    graph.add_edge(NODE_RETRIEVE, NODE_GENERATE)
    graph.add_edge(NODE_GENERATE, NODE_VALIDATE)
    graph.add_conditional_edges(NODE_VALIDATE, route_after_validate, {
        NODE_RESPOND: NODE_RESPOND,
        NODE_GENERATE: NODE_GENERATE,
        NODE_FALLBACK: NODE_FALLBACK,
    })
    graph.add_edge(NODE_RESPOND, END)
    graph.add_edge(NODE_FALLBACK, END)
    graph.add_edge(NODE_GREETING, END)
    graph.add_edge(NODE_OFF_TOPIC, END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    # HITL interrupt: when enabled, the graph pauses before generate_node
    # so a human operator can review/approve the retrieved context before
    # the LLM generates a response.  This is the LangGraph-native pattern
    # for regulated environments where certain responses need human oversight.
    interrupt_before = [NODE_GENERATE] if settings.ENABLE_HITL_INTERRUPT else None

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )
    compiled.recursion_limit = settings.GRAPH_RECURSION_LIMIT  # type: ignore[attr-defined]
    logger.info("Custom 8-node StateGraph compiled successfully.")
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
