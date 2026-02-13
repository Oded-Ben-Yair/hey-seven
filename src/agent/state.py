"""Agent state schema for the Property Q&A agent.

Extends LangGraph's MessagesState with property-specific fields.

Note: ``create_react_agent`` uses its own internal ``AgentState``.
This class documents the expected state shape and is available for
custom graph builds (e.g., adding compliance or escalation nodes).
"""

from langgraph.graph import MessagesState


class PropertyQAState(MessagesState):
    """State for property Q&A conversations.

    Inherits ``messages: Annotated[list[AnyMessage], add_messages]`` from
    MessagesState, which handles message appending and deduplication.

    Currently used for type documentation. For a custom ``StateGraph``
    (e.g., with compliance or escalation nodes), pass this class as
    ``StateGraph(PropertyQAState)`` instead of using ``create_react_agent``.
    """

    property_name: str = "Mohegan Sun"
