"""Main agent assembly for the Casino Host.

Uses ``create_react_agent`` from ``langgraph.prebuilt`` -- the recommended
LangGraph 1.0 GA pattern for tool-calling agents. The prebuilt agent handles
the standard LLM -> tools -> LLM loop automatically.

Compliance checking and escalation handling are added as post-processing
nodes that run after the agent finishes its tool-calling loop, before the
final response is returned to the caller.

Graph structure::

    START
      |
      v
    react_agent (prebuilt: LLM <-> tool_executor loop)
      |
      v
    post_process (compliance check + escalation routing)
      |
      v
    response_formatter
      |
      v
    END
"""

import logging
import os
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from .memory import get_checkpointer
from .nodes import (
    ComplianceResult,
    agent_node,
    compliance_checker,
    escalation_handler,
    get_llm,
    response_formatter,
    tool_executor,
)
from .prompts import CASINO_HOST_SYSTEM_PROMPT
from .state import AgentState, CasinoHostState
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing Logic (C7 fix: descriptive routing keys, C8 fix: compliance guard)
# ---------------------------------------------------------------------------


def route_after_agent(state: CasinoHostState) -> Literal[
    "tool_executor", "compliance_checker", "escalation_handler", "format_response"
]:
    """Determine the next node after the agent_node produces a response.

    Routing priority:
    1. If the LLM response contains tool_calls -> tool_executor
    2. If escalation_needed flag is set -> escalation_handler
    3. If compliance flags require review AND not already checked (C8 fix)
       -> compliance_checker
    4. Otherwise -> format_response (descriptive key, C7 fix)

    Args:
        state: Current agent state.

    Returns:
        The name of the next node.
    """
    messages = state.get("messages", [])
    if not messages:
        return "format_response"

    last_message = messages[-1]

    # Priority 1: Tool calls from LLM
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_executor"

    # Priority 2: Escalation flag
    if state.get("escalation_needed", False):
        return "escalation_handler"

    # Priority 3: Compliance review needed (C8 fix: skip if already checked)
    compliance_flags = state.get("compliance_flags", [])
    compliance_checked = state.get("compliance_checked", False)
    if "COMPLIANCE_REVIEW_NEEDED" in compliance_flags and not compliance_checked:
        return "compliance_checker"

    # Default: format and return response
    return "format_response"


def route_after_compliance(
    state: CasinoHostState,
) -> Literal["agent_node", "format_response"]:
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
    return "format_response"


# ---------------------------------------------------------------------------
# Graph Assembly — Two Patterns Available
# ---------------------------------------------------------------------------


def build_react_agent(
    checkpointer: Any = None,
    use_firestore: bool = False,
) -> Any:
    """Build the Casino Host agent using ``create_react_agent`` (C2 fix).

    This is the recommended LangGraph 1.0 GA pattern. The prebuilt agent
    handles the LLM <-> tool execution loop automatically. The system
    prompt and tools are passed directly.

    For compliance and escalation, the calling code can inspect the final
    state and invoke dedicated handlers as needed. Alternatively, use
    ``build_graph()`` for a fully-wired custom graph with built-in
    compliance and escalation routing.

    Args:
        checkpointer: Optional pre-configured checkpointer.
        use_firestore: If True and no checkpointer provided, creates a
            Firestore-backed checkpointer.

    Returns:
        A compiled LangGraph agent ready for invocation.
    """
    llm = get_llm()

    # Select checkpointer
    if checkpointer is None:
        checkpointer = get_checkpointer(use_firestore=use_firestore)

    # create_react_agent handles the tool-calling loop internally.
    # NOTE: ``state_modifier`` is the correct param name for langgraph<=0.2.60.
    # The ``prompt`` alias was added in langgraph>=0.2.70.
    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        checkpointer=checkpointer,
        state_schema=CasinoHostState,
        state_modifier=CASINO_HOST_SYSTEM_PROMPT,
    )

    logger.info("Casino Host react agent compiled successfully.")
    return agent


def build_graph(
    checkpointer: Any = None,
    use_firestore: bool = False,
) -> Any:
    """Build the Casino Host agent graph with full compliance/escalation routing.

    This is the extended version that includes compliance checking and
    escalation handling as dedicated graph nodes. Use this when you need
    the full regulatory workflow built into the graph itself.

    Args:
        checkpointer: Optional pre-configured checkpointer. If None, uses
            MemorySaver for local dev or Firestore if use_firestore is True.
        use_firestore: If True and no checkpointer provided, creates a
            Firestore-backed checkpointer.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    workflow = StateGraph(CasinoHostState)

    # Add nodes
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tool_executor", tool_executor)
    workflow.add_node("compliance_checker", compliance_checker)
    workflow.add_node("escalation_handler", escalation_handler)
    workflow.add_node("response_formatter", response_formatter)

    # Entry edge: START -> agent_node
    workflow.add_edge(START, "agent_node")

    # Conditional edges after agent_node (C7 fix: descriptive routing keys)
    workflow.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {
            "tool_executor": "tool_executor",
            "compliance_checker": "compliance_checker",
            "escalation_handler": "escalation_handler",
            "format_response": "response_formatter",
        },
    )

    # After tool execution, return to agent for next decision
    workflow.add_edge("tool_executor", "agent_node")

    # After compliance check, route based on result (C7 fix: descriptive keys)
    workflow.add_conditional_edges(
        "compliance_checker",
        route_after_compliance,
        {
            "agent_node": "agent_node",
            "format_response": "response_formatter",
        },
    )

    # Escalation handler ends the turn
    workflow.add_edge("escalation_handler", "response_formatter")

    # Response formatter is the terminal node
    workflow.add_edge("response_formatter", END)

    # Select checkpointer
    if checkpointer is None:
        checkpointer = get_checkpointer(use_firestore=use_firestore)

    # Compile
    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info("Casino Host agent graph compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def create_agent(use_react: bool = True) -> Any:
    """Create a production-ready Casino Host agent.

    Reads configuration from environment variables:
        - GOOGLE_API_KEY: Required for Gemini LLM.
        - USE_FIRESTORE: Set to "true" for Firestore checkpointing.
        - USE_REACT_AGENT: Set to "false" to use the full custom graph
          with built-in compliance/escalation routing.

    Args:
        use_react: If True (default), use ``create_react_agent`` (the
            modern LangGraph 1.0 pattern). If False, use the full custom
            graph with compliance/escalation nodes.

    Returns:
        A compiled LangGraph agent ready for invocation.
    """
    use_firestore = os.environ.get("USE_FIRESTORE", "false").lower() == "true"

    # Allow env var override
    env_react = os.environ.get("USE_REACT_AGENT", "").lower()
    if env_react == "false":
        use_react = False
    elif env_react == "true":
        use_react = True

    if use_react:
        return build_react_agent(use_firestore=use_firestore)
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
            response_text = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
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
    print("Casino Host Agent -- Demo")
    print("=" * 60)

    agent = create_agent()

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
