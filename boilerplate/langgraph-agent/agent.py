"""Main agent assembly for the Casino Host.

Builds the LangGraph StateGraph, connects all nodes with conditional
routing, compiles with checkpointing, and exposes the entry-point function
for the API layer.

Graph structure:

    START
      |
      v
    agent_node  <-----+
      |               |
      |-- has tool_calls? --> tool_executor --> agent_node
      |
      |-- escalation_needed? --> escalation_handler --> END
      |
      |-- compliance flags? --> compliance_checker --> agent_node
      |
      +-- otherwise --> END
"""

import logging
import os
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .memory import FirestoreCheckpointSaver
from .nodes import (
    agent_node,
    compliance_checker,
    escalation_handler,
    response_formatter,
    tool_executor,
)
from .state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------------------------


def route_after_agent(state: AgentState) -> Literal[
    "tool_executor", "compliance_checker", "escalation_handler", "__end__"
]:
    """Determine the next node after the agent_node produces a response.

    Routing priority:
    1. If the LLM response contains tool_calls -> tool_executor
    2. If escalation_needed flag is set -> escalation_handler
    3. If compliance flags require review -> compliance_checker
    4. Otherwise -> END (return response to caller)

    Args:
        state: Current agent state.

    Returns:
        The name of the next node, or "__end__" to terminate.
    """
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]

    # Priority 1: Tool calls from LLM
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_executor"

    # Priority 2: Escalation flag
    if state.get("escalation_needed", False):
        return "escalation_handler"

    # Priority 3: Compliance review needed
    compliance_flags = state.get("compliance_flags", [])
    if "COMPLIANCE_REVIEW_NEEDED" in compliance_flags:
        return "compliance_checker"

    # Default: end the turn
    return "__end__"


def route_after_compliance(state: AgentState) -> Literal["agent_node", "__end__"]:
    """Route after compliance check completes.

    If compliance blocked the action, return to the agent to inform the
    user. Otherwise, end the turn.

    Args:
        state: Current agent state.

    Returns:
        Next node name.
    """
    compliance_flags = state.get("compliance_flags", [])
    if "COMPLIANCE_BLOCK" in compliance_flags:
        # Let the agent explain the compliance block to the user
        return "agent_node"
    return "__end__"


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------


def build_graph(
    checkpointer: Any = None,
    use_firestore: bool = False,
) -> Any:
    """Build and compile the Casino Host agent graph.

    Args:
        checkpointer: Optional pre-configured checkpointer. If None, uses
            MemorySaver for local dev or FirestoreCheckpointSaver if
            use_firestore is True.
        use_firestore: If True and no checkpointer provided, creates a
            FirestoreCheckpointSaver.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tool_executor", tool_executor)
    workflow.add_node("compliance_checker", compliance_checker)
    workflow.add_node("escalation_handler", escalation_handler)
    workflow.add_node("response_formatter", response_formatter)

    # Entry edge: START -> agent_node
    workflow.add_edge(START, "agent_node")

    # Conditional edges after agent_node
    workflow.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {
            "tool_executor": "tool_executor",
            "compliance_checker": "compliance_checker",
            "escalation_handler": "escalation_handler",
            "__end__": "response_formatter",
        },
    )

    # After tool execution, return to agent for next decision
    workflow.add_edge("tool_executor", "agent_node")

    # After compliance check, route based on result
    workflow.add_conditional_edges(
        "compliance_checker",
        route_after_compliance,
        {
            "agent_node": "agent_node",
            "__end__": "response_formatter",
        },
    )

    # Escalation handler ends the turn
    workflow.add_edge("escalation_handler", "response_formatter")

    # Response formatter is the terminal node
    workflow.add_edge("response_formatter", END)

    # Select checkpointer
    if checkpointer is None:
        if use_firestore:
            checkpointer = FirestoreCheckpointSaver()
        else:
            checkpointer = MemorySaver()

    # Compile
    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info("Casino Host agent graph compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def create_agent() -> Any:
    """Create a production-ready Casino Host agent.

    Reads configuration from environment variables:
        - GOOGLE_API_KEY: Required for Gemini LLM.
        - USE_FIRESTORE: Set to "true" for Firestore checkpointing.
        - GCP_PROJECT_ID: Required if using Firestore.

    Returns:
        A compiled LangGraph agent ready for invocation.
    """
    use_firestore = os.environ.get("USE_FIRESTORE", "false").lower() == "true"
    return build_graph(use_firestore=use_firestore)


async def chat(
    agent: Any,
    message: str,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Send a message to the Casino Host agent and get a response.

    This is the primary entry point for the API layer. It handles thread
    management and extracts the final response from the agent state.

    Args:
        agent: A compiled LangGraph agent (from create_agent or build_graph).
        message: The user's message text.
        thread_id: Optional conversation thread ID for persistence. If None,
            a new thread is created.

    Returns:
        A dict with:
            - response: The agent's text response.
            - thread_id: The conversation thread ID (for continuation).
            - player_id: Currently discussed player, if any.
            - escalation: Whether the case was escalated.
            - compliance_flags: Active compliance flags.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    input_state = {
        "messages": [HumanMessage(content=message)],
    }

    # Invoke the graph
    result = await agent.ainvoke(input_state, config=config)

    # Extract final response
    messages = result.get("messages", [])
    response_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            response_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return {
        "response": response_text,
        "thread_id": thread_id,
        "player_id": result.get("player_id"),
        "escalation": result.get("escalation_needed", False),
        "compliance_flags": result.get("compliance_flags", []),
    }


# ---------------------------------------------------------------------------
# Demo / Test
# ---------------------------------------------------------------------------


async def _demo() -> None:
    """Run a quick demo of the Casino Host agent.

    Requires GOOGLE_API_KEY environment variable to be set.
    """
    print("=" * 60)
    print("Casino Host Agent — Demo")
    print("=" * 60)

    agent = build_graph()

    # Demo conversation
    demo_messages = [
        "Can you look up player PLY-482910? I need to know their tier and preferences.",
        "What comp would they qualify for on a dining reservation?",
        "Great, go ahead and book SW Steakhouse for them on 2025-03-15, party of 4.",
    ]

    thread_id = str(uuid.uuid4())

    for msg in demo_messages:
        print(f"\nHost: {msg}")
        result = await chat(agent, msg, thread_id=thread_id)
        print(f"\nSeven: {result['response'][:500]}")
        if result.get("compliance_flags"):
            print(f"  [Compliance Flags: {result['compliance_flags']}]")
        print("-" * 40)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo())
