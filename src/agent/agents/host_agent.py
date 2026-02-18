"""Host agent — primary concierge for general property Q&A.

Functionally equivalent to v1's ``generate_node`` in ``nodes.py``.
Handles all property questions that don't fall into a specialist domain.
"""

import logging
from string import Template

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.circuit_breaker import _get_circuit_breaker
from src.agent.nodes import _format_context_block, _get_llm
from src.agent.prompts import CONCIERGE_SYSTEM_PROMPT, RESPONSIBLE_GAMING_HELPLINES
from src.agent.state import PropertyQAState
from src.agent.whisper_planner import format_whisper_plan
from src.config import get_settings

logger = logging.getLogger(__name__)


async def host_agent(state: PropertyQAState) -> dict:
    """Generate a concierge response using retrieved context.

    Mirrors the v1 ``generate_node`` logic:
    1. Check circuit breaker — if open, return fallback.
    2. Build system prompt with property name, time, helplines.
    3. Append retrieved context — if empty, return "no info" fallback.
    4. Handle retry feedback injection.
    5. Sliding window on message history.
    6. Call LLM, record success/failure on circuit breaker.
    7. Return AIMessage with content.
    """
    settings = get_settings()
    retrieved = state.get("retrieved_context", [])
    current_time = state.get("current_time", "unknown")
    retry_count = state.get("retry_count", 0)
    retry_feedback = state.get("retry_feedback")

    # Circuit breaker check — early exit before building prompts
    if _get_circuit_breaker().is_open:
        logger.warning("Circuit breaker open — returning fallback without LLM call")
        return {
            "messages": [AIMessage(content=(
                "I'm experiencing temporary technical difficulties. "
                "Please try again in a minute, or contact "
                "$property_name directly at $property_phone."
            ).replace("$property_name", settings.PROPERTY_NAME)
             .replace("$property_phone", settings.PROPERTY_PHONE))],
            "skip_validation": True,
        }

    system_prompt = CONCIERGE_SYSTEM_PROMPT.safe_substitute(
        property_name=settings.PROPERTY_NAME,
        current_time=current_time,
        responsible_gaming_helplines=RESPONSIBLE_GAMING_HELPLINES,
    )

    # Format retrieved context as numbered sources
    if retrieved:
        context_block = _format_context_block(retrieved)
        system_prompt += Template(
            "\n\n## Retrieved Knowledge Base Context\n$context"
        ).safe_substitute(context=context_block)
    else:
        # No context found — signal to skip validation
        return {
            "messages": [AIMessage(content=(
                "I appreciate your question! Unfortunately, I don't have specific information "
                "about that in my knowledge base. For the most accurate and up-to-date details, "
                "I'd recommend contacting $property_name directly at $property_phone or visiting "
                "$property_website."
            ).replace("$property_name", settings.PROPERTY_NAME)
             .replace("$property_phone", settings.PROPERTY_PHONE)
             .replace("$property_website", settings.PROPERTY_WEBSITE))],
            "skip_validation": True,
        }

    # Inject whisper planner guidance (if available from upstream node)
    whisper_guidance = format_whisper_plan(state.get("whisper_plan"))
    if whisper_guidance:
        system_prompt += whisper_guidance

    # Build message list
    llm_messages = [SystemMessage(content=system_prompt)]

    # On retry, inject feedback
    if retry_count > 0 and retry_feedback:
        llm_messages.append(SystemMessage(
            content=Template(
                "IMPORTANT: Your previous response failed validation. Reason: $feedback. "
                "Please generate a corrected response that addresses this issue."
            ).safe_substitute(feedback=retry_feedback)
        ))

    # Add conversation history (only HumanMessage and AIMessage, skip tool messages).
    # Sliding window: keep only the last MAX_HISTORY_MESSAGES to bound context size.
    history = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
    window = history[-settings.MAX_HISTORY_MESSAGES:]
    llm_messages.extend(window)

    llm = _get_llm()

    try:
        response = await llm.ainvoke(llm_messages)
        await _get_circuit_breaker().record_success()
        content = response.content if isinstance(response.content, str) else str(response.content)
        return {"messages": [AIMessage(content=content)]}
    except (ValueError, TypeError) as exc:
        await _get_circuit_breaker().record_failure()
        logger.warning("Host agent LLM response parsing failed: %s", exc)
        return {
            "messages": [AIMessage(content=(
                "I apologize, but I had trouble processing that response. "
                "Please try again, or contact $property_name directly at $property_phone."
            ).replace("$property_name", settings.PROPERTY_NAME)
             .replace("$property_phone", settings.PROPERTY_PHONE))],
            "skip_validation": True,
        }
    except Exception:
        await _get_circuit_breaker().record_failure()
        logger.exception("Host agent LLM call failed")
        return {
            "messages": [AIMessage(content=(
                "I apologize, but I'm having trouble generating a response right now. "
                "Please try again, or contact $property_name directly at $property_phone."
            ).replace("$property_name", settings.PROPERTY_NAME)
             .replace("$property_phone", settings.PROPERTY_PHONE))],
            "skip_validation": True,
        }
