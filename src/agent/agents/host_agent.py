"""Host agent — primary concierge for general property Q&A.

Functionally equivalent to v1's ``generate_node`` in ``nodes.py``.
Handles all property questions that don't fall into a specialist domain.
"""

from string import Template

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
    fallback = Template(
        "I appreciate your question! Unfortunately, I don't have specific information "
        "about that in my knowledge base. For the most accurate and up-to-date details, "
        "I'd recommend contacting $property_name directly at $property_phone or visiting "
        "$property_website."
    ).safe_substitute(
        property_name=settings.PROPERTY_NAME,
        property_phone=settings.PROPERTY_PHONE,
        property_website=settings.PROPERTY_WEBSITE,
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
