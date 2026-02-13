"""LangGraph agent for property Q&A.

Uses ``create_react_agent`` from ``langgraph.prebuilt`` with two RAG
search tools. The prebuilt agent handles the LLM -> tools -> LLM
loop automatically.
"""

import json
import logging
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from src.config import get_settings

from .prompts import PROPERTY_CONCIERGE_PROMPT
from .tools import get_property_hours, search_property

logger = logging.getLogger(__name__)

#: Categories in the property knowledge base (used for source extraction).
KNOWN_CATEGORIES = frozenset({
    "restaurants", "entertainment", "hotel", "amenities",
    "gaming", "promotions", "faq", "property",
})


def create_agent() -> CompiledStateGraph:
    """Create a property Q&A agent.

    Returns:
        A compiled LangGraph agent.
    """
    settings = get_settings()

    llm = ChatGoogleGenerativeAI(
        model=settings.MODEL_NAME,
        temperature=settings.MODEL_TEMPERATURE,
    )

    system_prompt = PROPERTY_CONCIERGE_PROMPT.format(
        property_name=settings.PROPERTY_NAME,
    )

    checkpointer = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=[search_property, get_property_hours],
        prompt=system_prompt,
        checkpointer=checkpointer,
    )

    logger.info("Property Q&A agent compiled successfully.")
    return agent


async def chat(
    agent: CompiledStateGraph,
    message: str,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Send a message to the agent and get a response.

    Args:
        agent: A compiled LangGraph agent (from create_agent).
        message: The user's message text.
        thread_id: Optional conversation thread ID for persistence.
            If None, a new thread is created.

    Returns:
        A dict with:
            - response: The agent's text response.
            - thread_id: The conversation thread ID.
            - sources: List of source categories cited.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )

    # Extract final AI response
    messages = result.get("messages", [])
    response_text = ""
    sources: list[str] = []

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            response_text = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    # Extract sources from tool call results
    # Format: "[1] (category) content" as produced by search_property tool
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool" and msg.content:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            for match in re.findall(r"\[\d+\]\s+\((\w+)\)", content):
                if match in KNOWN_CATEGORIES and match not in sources:
                    sources.append(match)

    return {
        "response": response_text,
        "thread_id": thread_id,
        "sources": sources,
    }


async def chat_stream(
    agent: CompiledStateGraph,
    message: str,
    thread_id: str | None = None,
) -> AsyncGenerator[dict[str, str], None]:
    """Stream a response from the agent as typed SSE events.

    Yields dicts with ``event`` and ``data`` keys suitable for
    ``EventSourceResponse``.

    Event types:
        metadata  – thread_id (sent first)
        token     – incremental text chunk
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

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        version="v2",
    ):
        kind = event.get("event")

        # Stream LLM tokens
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if (
                isinstance(chunk, AIMessageChunk)
                and chunk.content
                and not chunk.tool_call_chunks
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

        # Collect sources from tool outputs
        elif kind == "on_tool_end":
            output = event.get("data", {}).get("output", "")
            if isinstance(output, str):
                for match in re.findall(r"\[\d+\]\s+\((\w+)\)", output):
                    if match in KNOWN_CATEGORIES and match not in sources:
                        sources.append(match)

    if sources:
        yield {
            "event": "sources",
            "data": json.dumps({"sources": sources}),
        }

    yield {
        "event": "done",
        "data": json.dumps({"done": True}),
    }
