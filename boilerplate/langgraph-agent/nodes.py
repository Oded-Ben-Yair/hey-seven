"""Processing nodes for the Casino Host agent graph.

Each function is a node in the LangGraph StateGraph. Nodes receive the full
CasinoHostState and return a partial state dict with only the keys they modify.
The StateGraph's reducers handle merging updates into the canonical state.

The LLM is instantiated ONCE at module level (lazy singleton) and shared
across all nodes. This avoids re-creating the client on every invocation.
"""

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from .prompts import (
    CASINO_HOST_SYSTEM_PROMPT,
    COMPLIANCE_CHECK_PROMPT,
    ESCALATION_ASSESSMENT_PROMPT,
)
from .state import CasinoHostState
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Singleton (C1 fix: instantiate ONCE, not per-node invocation)
# ---------------------------------------------------------------------------

_llm_instance: ChatGoogleGenerativeAI | None = None
_llm_with_tools: Any = None


def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Return the module-level LLM singleton, creating it on first call.

    The LLM is created once and reused across all node invocations within
    the same process. This avoids the overhead of instantiating the client
    and setting up connections on every graph step.

    Args:
        temperature: Sampling temperature for the first initialization only.
            Subsequent calls return the existing instance regardless of this
            parameter. Use ``llm.with_config()`` for per-call overrides.

    Returns:
        A ChatGoogleGenerativeAI instance configured for Gemini 2.5 Flash.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            max_output_tokens=4096,
        )
    return _llm_instance


def get_llm_with_tools() -> Any:
    """Return the LLM with casino tools bound (cached singleton).

    Avoids calling ``bind_tools(ALL_TOOLS)`` on every agent_node invocation.
    The bound model is created once and reused.
    """
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm().bind_tools(ALL_TOOLS)
    return _llm_with_tools


# ---------------------------------------------------------------------------
# Structured Output Models (C3 fix: replace substring parsing)
# ---------------------------------------------------------------------------


class ComplianceResult(BaseModel):
    """Structured compliance assessment result.

    Used with ``.with_structured_output()`` to get reliable, parseable
    compliance decisions instead of substring matching on free-text.
    """

    status: Literal["COMPLIANT", "NON_COMPLIANT", "NEEDS_REVIEW"] = Field(
        description=(
            "The compliance determination. COMPLIANT means the action may "
            "proceed. NON_COMPLIANT means the action is blocked. "
            "NEEDS_REVIEW means a human compliance officer should review."
        )
    )
    flags: list[str] = Field(
        default_factory=list,
        description=(
            "Specific compliance flags raised. Examples: "
            "'SELF_EXCLUSION_ACTIVE', 'EXCESSIVE_COMP_VALUE', "
            "'RESPONSIBLE_GAMING_CONCERN', 'MARKETING_OPT_OUT'."
        ),
    )
    reasoning: str = Field(
        description=(
            "Brief explanation of the compliance determination, including "
            "which regulations or policies apply."
        )
    )


# ---------------------------------------------------------------------------
# Core Agent Node
# ---------------------------------------------------------------------------


def agent_node(state: CasinoHostState) -> dict[str, Any]:
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
        which may include tool_calls). Resets compliance_checked to False
        for each new agent turn.
    """
    llm_with_tools = get_llm_with_tools()

    # Build the message list: system prompt + conversation history
    system_msg = SystemMessage(content=CASINO_HOST_SYSTEM_PROMPT)
    messages: list = [system_msg] + list(state["messages"])

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

    # Reset compliance_checked on each new agent turn so compliance can
    # run again if needed (C8 fix: prevents stale flag from prior turns).
    return {
        "messages": [response],
        "compliance_checked": False,
    }


# ---------------------------------------------------------------------------
# Tool Executor Node
# ---------------------------------------------------------------------------

# ToolNode from langgraph.prebuilt handles tool execution automatically.
# It reads tool_calls from the last AIMessage, executes them, and returns
# ToolMessage results.
tool_executor = ToolNode(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Compliance Checker Node (C3 + C8 fix)
# ---------------------------------------------------------------------------


def compliance_checker(state: CasinoHostState) -> dict[str, Any]:
    """Validates pending actions against regulatory requirements.

    This node is invoked when the routing logic detects that a compliance-
    sensitive action is being taken. It uses structured output to get a
    reliable ComplianceResult instead of substring matching on free-text.

    After evaluation, it REPLACES the COMPLIANCE_REVIEW_NEEDED flag (rather
    than appending to it) and sets compliance_checked=True to prevent
    infinite re-entry.

    Args:
        state: Current agent state.

    Returns:
        Partial state with updated compliance_flags, escalation_needed,
        compliance_checked, and a compliance result message.
    """
    llm = get_llm()
    compliance_llm = llm.with_structured_output(ComplianceResult)

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

    result: ComplianceResult = compliance_llm.invoke(messages)

    # Build new flags list: remove COMPLIANCE_REVIEW_NEEDED (C8 fix),
    # add any new flags from the structured result.
    new_flags = [
        f for f in state.get("compliance_flags", [])
        if f != "COMPLIANCE_REVIEW_NEEDED"
    ]
    escalation = state.get("escalation_needed", False)

    if result.status == "NON_COMPLIANT":
        new_flags.append("COMPLIANCE_BLOCK")
        new_flags.extend(result.flags)
        escalation = True
        logger.warning(
            "Compliance check FAILED for player %s: %s",
            state.get("player_id"),
            result.reasoning[:200],
        )
    elif result.status == "NEEDS_REVIEW":
        # Do NOT re-add COMPLIANCE_REVIEW_NEEDED (prevents infinite loop).
        # Instead, mark for human review via escalation.
        new_flags.extend(result.flags)
        escalation = True
        logger.info(
            "Compliance check needs human review for player %s: %s",
            state.get("player_id"),
            result.reasoning[:200],
        )
    else:
        # COMPLIANT: add any informational flags
        new_flags.extend(result.flags)

    # Deduplicate flags
    new_flags = list(dict.fromkeys(new_flags))

    return {
        "compliance_flags": new_flags,
        "escalation_needed": escalation,
        "compliance_checked": True,
        "messages": [
            AIMessage(
                content=(
                    f"[Compliance Check Result: {result.status}]\n"
                    f"Reasoning: {result.reasoning}\n"
                    f"Flags: {', '.join(result.flags) if result.flags else 'None'}"
                ),
                name="compliance_checker",
            )
        ],
    }


# ---------------------------------------------------------------------------
# Escalation Handler Node
# ---------------------------------------------------------------------------


def escalation_handler(state: CasinoHostState) -> dict[str, Any]:
    """Routes complex cases to a human casino host.

    Generates an escalation summary with full context, creates a handoff
    message, and resets the escalation_needed flag.

    Args:
        state: Current agent state.

    Returns:
        Partial state with escalation message and reset flag.
    """
    llm = get_llm()

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


def response_formatter(state: CasinoHostState) -> dict[str, Any]:
    """Formats the final response for the host interface.

    Ensures the last message is clean, well-structured, and appropriate
    for the audience (human host or player-facing). This node runs before
    the graph returns to the caller.

    Args:
        state: Current agent state.

    Returns:
        Partial state -- typically a pass-through since formatting happens
        in the agent_node. Can add metadata annotations.
    """
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
    lines: list[str] = []
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
