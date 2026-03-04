"""Host agent — primary concierge for general property Q&A.

Functionally equivalent to v1's ``generate_node`` in ``nodes.py``.
Handles all property questions that don't fall into a specialist domain.
"""

from src.agent.circuit_breaker import _get_circuit_breaker
from src.agent.nodes import _get_llm
from src.agent.prompts import CONCIERGE_SYSTEM_PROMPT
from src.agent.state import PropertyQAState
from src.config import get_settings

from ._base import execute_specialist


async def host_agent(state: PropertyQAState) -> dict:
    """Generate a concierge response using retrieved context.

    Delegates to ``execute_specialist()`` with the general concierge prompt
    and whisper planner integration enabled.
    """
    settings = get_settings()
    fallback = (
        f"I don't have that specific info right now. What else can I help you with "
        f"at {settings.PROPERTY_NAME}?"
    )

    return await execute_specialist(
        state,
        agent_name="host",
        system_prompt_template=CONCIERGE_SYSTEM_PROMPT,
        context_header="Retrieved Knowledge Base Context",
        no_context_fallback=fallback,
        get_llm_fn=_get_llm,
        get_cb_fn=_get_circuit_breaker,
        include_whisper=True,
    )
