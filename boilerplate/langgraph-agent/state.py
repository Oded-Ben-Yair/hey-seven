"""Agent state schema for the Casino Host agent.

Extends LangGraph's MessagesState with casino-domain fields. Uses the
modern LangGraph 1.0 pattern where MessagesState provides the `messages`
field with the add_messages reducer pre-configured.
"""

from typing import Annotated, Any

from langgraph.graph import MessagesState


class CasinoHostState(MessagesState):
    """Full state schema for the Casino Host agent graph.

    Inherits ``messages: Annotated[list[AnyMessage], add_messages]`` from
    MessagesState, which handles message appending and deduplication.

    Attributes:
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
        compliance_checked: Whether the current turn has already been through
            compliance review. Prevents infinite re-entry into the compliance
            checker node.
        conversation_summary: Rolling summary of the conversation for long
            interactions, used when context window approaches limits.
    """

    # NOTE: Mutable defaults are safe here because CasinoHostState is a TypedDict.
    # LangGraph creates a fresh state dict per invocation — these defaults are
    # never shared across threads or invocations.
    player_id: str | None = None
    player_context: dict[str, Any] = {}
    comp_calculation: dict[str, Any] = {}
    pending_actions: list[dict[str, Any]] = []
    escalation_needed: bool = False
    compliance_flags: list[str] = []
    compliance_checked: bool = False
    conversation_summary: str = ""


# Keep backward-compatible alias for any external imports
AgentState = CasinoHostState
