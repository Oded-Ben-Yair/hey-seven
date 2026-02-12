"""Processing nodes for the Casino Host agent graph.

Each function is a node in the LangGraph StateGraph. Nodes receive the full
AgentState and return a partial state dict with only the keys they modify.
The StateGraph's reducers handle merging updates into the canonical state.
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

from .prompts import (
    CASINO_HOST_SYSTEM_PROMPT,
    COMPLIANCE_CHECK_PROMPT,
    ESCALATION_ASSESSMENT_PROMPT,
)
from .state import AgentState
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)


def _get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Create a configured Gemini 2.5 Flash instance with tools bound.

    Args:
        temperature: Sampling temperature. Lower for operational tasks,
            higher for conversational responses.

    Returns:
        A ChatGoogleGenerativeAI instance. Requires GOOGLE_API_KEY env var.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_output_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Core Agent Node
# ---------------------------------------------------------------------------


def agent_node(state: AgentState) -> dict[str, Any]:
    """Main LLM reasoning node. Decides what to do next.

    Invokes Gemini 2.5 Flash with the full conversation history and the
    casino host system prompt. The LLM may:
    - Call one or more tools (player lookup, comp calc, etc.)
    - Provide a direct text response
    - Request escalation

    The LLM has all casino tools bound, so it can autonomously decide which
    tools to call based on the conversation context.

    Args:
        state: Current agent state.

    Returns:
        Partial state with updated messages (containing the LLM response,
        which may include tool_calls).
    """
    llm = _get_llm(temperature=0.3)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Build the message list: system prompt + conversation history
    system_msg = SystemMessage(content=CASINO_HOST_SYSTEM_PROMPT)
    messages = [system_msg] + list(state["messages"])

    # Inject player context if available
    if state.get("player_context"):
        context_note = (
            f"\n[System Note: Current player context loaded: "
            f"{json.dumps(state['player_context'], default=str)}]"
        )
        messages.append(SystemMessage(content=context_note))

    # Inject compliance flags if any
    if state.get("compliance_flags"):
        flags_note = (
            f"\n[System Note: Active compliance flags: "
            f"{', '.join(state['compliance_flags'])}. "
            f"Check compliance before any player action.]"
        )
        messages.append(SystemMessage(content=flags_note))

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Tool Executor Node
# ---------------------------------------------------------------------------

# ToolNode from langgraph.prebuilt handles tool execution automatically.
# It reads tool_calls from the last AIMessage, executes them, and returns
# ToolMessage results.
tool_executor = ToolNode(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Compliance Checker Node
# ---------------------------------------------------------------------------


def compliance_checker(state: AgentState) -> dict[str, Any]:
    """Validates pending actions against regulatory requirements.

    This node is invoked when the routing logic detects that a compliance-
    sensitive action is being taken (comp issuance, freeplay, marker,
    messaging). It reviews the action context and updates compliance_flags.

    Args:
        state: Current agent state.

    Returns:
        Partial state with updated compliance_flags and potentially
        escalation_needed.
    """
    llm = _get_llm(temperature=0.1)

    # Build compliance review context
    messages = [
        SystemMessage(content=COMPLIANCE_CHECK_PROMPT),
        HumanMessage(
            content=(
                f"Review the following interaction for compliance:\n\n"
                f"Player ID: {state.get('player_id', 'Unknown')}\n"
                f"Player Context: {json.dumps(state.get('player_context', {}), default=str)}\n"
                f"Pending Actions: {json.dumps(state.get('pending_actions', []), default=str)}\n"
                f"Existing Flags: {state.get('compliance_flags', [])}\n\n"
                f"Recent messages:\n"
                f"{_format_recent_messages(state['messages'], count=5)}\n\n"
                f"Provide your compliance assessment."
            )
        ),
    ]

    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)

    # Parse compliance result
    new_flags = list(state.get("compliance_flags", []))
    escalation = state.get("escalation_needed", False)

    if "NON-COMPLIANT" in content.upper() or "NON_COMPLIANT" in content.upper():
        new_flags.append("COMPLIANCE_BLOCK")
        escalation = True
        logger.warning(
            "Compliance check FAILED for player %s: %s",
            state.get("player_id"),
            content[:200],
        )
    elif "NEEDS_REVIEW" in content.upper():
        new_flags.append("COMPLIANCE_REVIEW_NEEDED")
        logger.info(
            "Compliance check needs review for player %s",
            state.get("player_id"),
        )

    return {
        "compliance_flags": new_flags,
        "escalation_needed": escalation,
        "messages": [
            AIMessage(
                content=f"[Compliance Check Result]\n{content}",
                name="compliance_checker",
            )
        ],
    }


# ---------------------------------------------------------------------------
# Escalation Handler Node
# ---------------------------------------------------------------------------


def escalation_handler(state: AgentState) -> dict[str, Any]:
    """Routes complex cases to a human casino host.

    Generates an escalation summary with full context, creates a handoff
    message, and resets the escalation_needed flag.

    Args:
        state: Current agent state.

    Returns:
        Partial state with escalation message and reset flag.
    """
    llm = _get_llm(temperature=0.2)

    messages = [
        SystemMessage(content=ESCALATION_ASSESSMENT_PROMPT),
        HumanMessage(
            content=(
                f"Prepare an escalation handoff for the following case:\n\n"
                f"Player ID: {state.get('player_id', 'Unknown')}\n"
                f"Player Context: {json.dumps(state.get('player_context', {}), default=str)}\n"
                f"Compliance Flags: {state.get('compliance_flags', [])}\n"
                f"Pending Actions: {json.dumps(state.get('pending_actions', []), default=str)}\n\n"
                f"Conversation summary:\n"
                f"{_format_recent_messages(state['messages'], count=10)}\n\n"
                f"Generate the escalation ticket content."
            )
        ),
    ]

    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "escalation_needed": False,
        "messages": [
            AIMessage(
                content=(
                    f"I've prepared an escalation to a human host. "
                    f"Here is the handoff summary:\n\n{content}"
                ),
                name="escalation_handler",
            )
        ],
    }


# ---------------------------------------------------------------------------
# Response Formatter Node
# ---------------------------------------------------------------------------


def response_formatter(state: AgentState) -> dict[str, Any]:
    """Formats the final response for the host interface.

    Ensures the last message is clean, well-structured, and appropriate
    for the audience (human host or player-facing). This node runs before
    the graph returns to the caller.

    Args:
        state: Current agent state.

    Returns:
        Partial state — typically a pass-through since formatting happens
        in the agent_node. Can add metadata annotations.
    """
    # Extract the last AI message for any final cleanup
    last_messages = state.get("messages", [])
    if not last_messages:
        return {}

    last_msg = last_messages[-1]

    # If the last message is from a sub-node (compliance/escalation), the
    # agent_node will process it on the next turn. No formatting needed here.
    if hasattr(last_msg, "name") and last_msg.name in (
        "compliance_checker",
        "escalation_handler",
    ):
        return {}

    # Update conversation summary for long conversations
    if len(last_messages) > 20 and not state.get("conversation_summary"):
        return {
            "conversation_summary": _generate_summary_stub(last_messages),
        }

    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_recent_messages(messages: list, count: int = 5) -> str:
    """Format recent messages for context injection into prompts.

    Args:
        messages: Full message list from state.
        count: Number of recent messages to include.

    Returns:
        A formatted string of recent messages.
    """
    recent = messages[-count:] if len(messages) > count else messages
    lines = []
    for msg in recent:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = str(content)
        lines.append(f"[{role}]: {content[:500]}")
    return "\n".join(lines)


def _generate_summary_stub(messages: list) -> str:
    """Generate a brief conversation summary (stub).

    In production, this would call the LLM with a summarization prompt.
    For now, returns a simple message count summary.

    Args:
        messages: Full message list.

    Returns:
        A brief summary string.
    """
    human_count = sum(1 for m in messages if hasattr(m, "type") and m.type == "human")
    ai_count = sum(1 for m in messages if hasattr(m, "type") and m.type == "ai")
    return (
        f"Conversation with {human_count} user messages and {ai_count} "
        f"assistant responses. Total turns: {len(messages)}."
    )
