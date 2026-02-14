"""Custom 8-node StateGraph for property Q&A.

Nodes: router → retrieve → generate → validate → respond
Branches: greeting, off_topic, fallback
"""

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

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

#: Categories in the property knowledge base (used for source extraction).
KNOWN_CATEGORIES = frozenset({
    "restaurants", "entertainment", "hotel", "amenities",
    "gaming", "promotions", "faq", "property",
})


def build_graph(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Build the custom 8-node property Q&A graph.

    Args:
        checkpointer: Optional checkpointer for conversation persistence.
            Defaults to MemorySaver for local development.

    Returns:
        A compiled StateGraph ready for invoke/astream.
    """
    graph = StateGraph(PropertyQAState)

    # Add all 8 nodes
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("respond", respond_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("off_topic", off_topic_node)

    # Edge map
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_from_router, {
        "retrieve": "retrieve",
        "greeting": "greeting",
        "off_topic": "off_topic",
    })
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", route_after_validate, {
        "respond": "respond",
        "generate": "generate",
        "fallback": "fallback",
    })
    graph.add_edge("respond", END)
    graph.add_edge("fallback", END)
    graph.add_edge("greeting", END)
    graph.add_edge("off_topic", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Custom 8-node StateGraph compiled successfully.")
    return compiled


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

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}

    now = datetime.now(tz=timezone.utc).strftime("%A, %B %d, %Y %I:%M %p UTC")

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=message)],
            "current_time": now,
            "query_type": None,
            "router_confidence": 0.0,
            "retrieved_context": [],
            "validation_result": None,
            "retry_count": 0,
            "retry_feedback": None,
            "sources_used": [],
        },
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

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}

    now = datetime.now(tz=timezone.utc).strftime("%A, %B %d, %Y %I:%M %p UTC")

    yield {
        "event": "metadata",
        "data": json.dumps({"thread_id": thread_id}),
    }

    sources: list[str] = []
    non_stream_nodes = {"greeting", "off_topic", "fallback"}

    try:
        async for event in graph.astream_events(
            {
                "messages": [HumanMessage(content=message)],
                "current_time": now,
                "query_type": None,
                "router_confidence": 0.0,
                "retrieved_context": [],
                "validation_result": None,
                "retry_count": 0,
                "retry_feedback": None,
                "sources_used": [],
            },
            config=config,
            version="v2",
        ):
            kind = event.get("event")
            langgraph_node = event.get("metadata", {}).get("langgraph_node", "")

            # Stream tokens from generate node only
            if kind == "on_chat_model_stream" and langgraph_node == "generate":
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
            elif kind == "on_chain_end" and langgraph_node in non_stream_nodes:
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
            elif kind == "on_chain_end" and langgraph_node == "respond":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    node_sources = output.get("sources_used", [])
                    for s in node_sources:
                        if s not in sources:
                            sources.append(s)

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
