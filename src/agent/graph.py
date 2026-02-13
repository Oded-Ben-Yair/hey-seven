"""LangGraph agent for property Q&A.

Uses ``create_react_agent`` from ``langgraph.prebuilt`` with a single
RAG search tool. The prebuilt agent handles the LLM -> tools -> LLM
loop automatically.
"""

import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from .prompts import PROPERTY_CONCIERGE_PROMPT
from .tools import search_property

logger = logging.getLogger(__name__)


def create_agent(
    property_data_path: str = "data/mohegan_sun.json",
) -> CompiledStateGraph:
    """Create a property Q&A agent.

    Args:
        property_data_path: Path to the property JSON data file.
            Reserved for future use (property name is currently hardcoded).

    Returns:
        A compiled LangGraph agent.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    system_prompt = PROPERTY_CONCIERGE_PROMPT.format(
        property_name="Mohegan Sun",
    )

    checkpointer = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=[search_property],
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
    known_categories = {"restaurants", "entertainment", "hotel", "amenities", "gaming", "promotions", "faq", "property"}
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool" and msg.content:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            for match in re.findall(r"\[\d+\]\s+\((\w+)\)", content):
                if match in known_categories and match not in sources:
                    sources.append(match)

    return {
        "response": response_text,
        "thread_id": thread_id,
        "sources": sources,
    }
