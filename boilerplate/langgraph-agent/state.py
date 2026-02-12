"""Agent state schema for the Casino Host agent.

Defines the TypedDict-based state that flows through the LangGraph StateGraph.
Each field represents a distinct aspect of the conversation context that nodes
can read from and write to.
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Full state schema for the Casino Host agent graph.

    Attributes:
        messages: Conversation history with the add_messages reducer, which
            handles appending and deduplication of messages automatically.
        player_id: The casino player ID currently being discussed. Set when
            the agent identifies which player the conversation is about.
        player_context: Cached player profile data (tier, ADT, visit history,
            preferences). Populated by the check_player_status tool.
        comp_calculation: Current comp computation results. Populated by the
            calculate_comp tool, includes theoretical win, reinvestment
            percentage, and eligible comp value.
        pending_actions: Actions that require human host confirmation before
            execution (high-value comps, VIP suite upgrades, etc.).
        escalation_needed: Flag indicating the current interaction should be
            routed to a human casino host.
        compliance_flags: Regulatory flags raised during the conversation
            (self-exclusion match, responsible gaming trigger, underage
            indicator, etc.).
        conversation_summary: Rolling summary of the conversation for long
            interactions, used when context window approaches limits.
    """

    messages: Annotated[list, add_messages]
    player_id: str | None
    player_context: dict[str, Any]
    comp_calculation: dict[str, Any]
    pending_actions: list[dict[str, Any]]
    escalation_needed: bool
    compliance_flags: list[str]
    conversation_summary: str


def create_initial_state() -> AgentState:
    """Create a fresh initial state for a new conversation.

    Returns:
        An AgentState with all fields set to their empty/default values.
    """
    return AgentState(
        messages=[],
        player_id=None,
        player_context={},
        comp_calculation={},
        pending_actions=[],
        escalation_needed=False,
        compliance_flags=[],
        conversation_summary="",
    )
