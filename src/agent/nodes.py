"""Node functions for the 8-node Property Q&A StateGraph.

Each node takes PropertyQAState and returns a partial dict update.
Two routing functions determine conditional edges.
Includes audit_input for deterministic prompt-injection detection.
"""

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings

from .prompts import CONCIERGE_SYSTEM_PROMPT, ROUTER_PROMPT, VALIDATION_PROMPT
from .state import PropertyQAState, RouterOutput, ValidationResult
from .tools import search_hours, search_knowledge_base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic input guardrails (pre-LLM)
# ---------------------------------------------------------------------------

#: Regex patterns for prompt injection detection.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\b", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"\bDAN\b.*\bmode\b", re.I),
    re.compile(r"pretend\s+(?:you(?:'re|\s+are)\s+)?(?:a|an|the)\b", re.I),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|your)\b", re.I),
    re.compile(r"act\s+as\s+(?:if\s+)?(?:you(?:'re|\s+are)\s+)?(?:a|an|the)\b", re.I),
]

#: Regex patterns for responsible gaming detection (pre-LLM safety net).
_RESPONSIBLE_GAMING_PATTERNS = [
    re.compile(r"gambling\s+problem", re.I),
    re.compile(r"problem\s+gambl", re.I),
    re.compile(r"addict(?:ed|ion)?\s+(?:to\s+)?gambl", re.I),
    re.compile(r"self[- ]?exclu", re.I),
    re.compile(r"can'?t\s+stop\s+gambl", re.I),
    re.compile(r"help\s+(?:with|for)\s+gambl", re.I),
]


def detect_responsible_gaming(message: str) -> bool:
    """Check if user message indicates a gambling problem or self-exclusion need.

    Deterministic regex-based safety net that ensures responsible gaming
    helplines are always provided, regardless of LLM routing.

    Args:
        message: The raw user input message.

    Returns:
        True if responsible gaming support is needed.
    """
    for pattern in _RESPONSIBLE_GAMING_PATTERNS:
        if pattern.search(message):
            logger.info("Responsible gaming query detected: %r", message[:200])
            return True
    return False


def audit_input(message: str) -> bool:
    """Check user input for prompt injection patterns.

    Deterministic regex-based guardrail that runs before any LLM call.
    Logs a warning if injection patterns are detected.

    Args:
        message: The raw user input message.

    Returns:
        True if the input looks safe, False if injection detected.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(message):
            logger.warning("Prompt injection detected: %r", message[:200])
            return False
    return True

# ---------------------------------------------------------------------------
# LLM singleton
# ---------------------------------------------------------------------------

_llm_instance: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    """Get or create the shared LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        settings = get_settings()
        _llm_instance = ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            temperature=settings.MODEL_TEMPERATURE,
            timeout=settings.MODEL_TIMEOUT,
            max_retries=settings.MODEL_MAX_RETRIES,
            max_output_tokens=settings.MODEL_MAX_OUTPUT_TOKENS,
        )
    return _llm_instance


# ---------------------------------------------------------------------------
# 1. Router Node
# ---------------------------------------------------------------------------


def router_node(state: PropertyQAState) -> dict:
    """Classify user intent into one of 7 categories.

    Turn-limit check: if >40 messages, forces off_topic to end conversation.
    Uses structured output for reliable JSON parsing.
    """
    messages = state.get("messages", [])

    # Turn-limit guard
    if len(messages) > 40:
        logger.warning("Turn limit exceeded (%d messages), forcing off_topic", len(messages))
        return {
            "query_type": "off_topic",
            "router_confidence": 1.0,
        }

    # Get the last human message
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not user_message:
        return {
            "query_type": "greeting",
            "router_confidence": 1.0,
        }

    # Deterministic prompt injection check (pre-LLM)
    if not audit_input(user_message):
        return {
            "query_type": "off_topic",
            "router_confidence": 1.0,
        }

    # Deterministic responsible gaming detection (pre-LLM safety net)
    if detect_responsible_gaming(user_message):
        return {
            "query_type": "gambling_advice",
            "router_confidence": 1.0,
        }

    llm = _get_llm()
    router_llm = llm.with_structured_output(RouterOutput)

    prompt_text = ROUTER_PROMPT.safe_substitute(user_message=user_message)

    try:
        result: RouterOutput = router_llm.invoke(prompt_text)
        return {
            "query_type": result.query_type,
            "router_confidence": result.confidence,
        }
    except Exception:
        logger.exception("Router LLM call failed, defaulting to property_qa")
        return {
            "query_type": "property_qa",
            "router_confidence": 0.5,
        }


# ---------------------------------------------------------------------------
# 2. Retrieve Node
# ---------------------------------------------------------------------------


def retrieve_node(state: PropertyQAState) -> dict:
    """Retrieve relevant documents from the knowledge base.

    Extracts the latest user message and searches for matching content.
    Uses schedule-focused search for hours_schedule queries.
    """
    messages = state.get("messages", [])
    query_type = state.get("query_type", "property_qa")

    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not query:
        return {"retrieved_context": []}

    # Use schedule-focused search for hours/schedule queries
    if query_type == "hours_schedule":
        results = search_hours(query)
    else:
        results = search_knowledge_base(query)

    return {"retrieved_context": results}


# ---------------------------------------------------------------------------
# 3. Generate Node
# ---------------------------------------------------------------------------


def generate_node(state: PropertyQAState) -> dict:
    """Generate a response using the concierge system prompt and retrieved context.

    If no context was retrieved, sets retry_count=99 to skip validation.
    On retry, prepends validation feedback as a SystemMessage.
    """
    settings = get_settings()
    retrieved = state.get("retrieved_context", [])
    current_time = state.get("current_time", "unknown")
    retry_count = state.get("retry_count", 0)
    retry_feedback = state.get("retry_feedback")

    system_prompt = CONCIERGE_SYSTEM_PROMPT.safe_substitute(
        property_name=settings.PROPERTY_NAME,
        current_time=current_time,
    )

    # Format retrieved context as numbered sources
    if retrieved:
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            category = doc.get("metadata", {}).get("category", "general")
            content = doc.get("content", "")
            context_parts.append(f"[{i}] ({category}) {content}")
        context_block = "\n---\n".join(context_parts)
        system_prompt += f"\n\n## Retrieved Knowledge Base Context\n{context_block}"
    else:
        # No context found — signal to skip validation
        system_prompt += (
            "\n\n## No relevant context found in the knowledge base."
            "\nBe honest and let the guest know you don't have specific information about their question."
        )
        return {
            "messages": [AIMessage(content=(
                "I appreciate your question! Unfortunately, I don't have specific information "
                "about that in my knowledge base. For the most accurate and up-to-date details, "
                "I'd recommend contacting Mohegan Sun directly at 1-888-226-7711 or visiting "
                "mohegansun.com."
            ))],
            "retry_count": 99,  # Skip validation
        }

    # Build message list
    llm_messages = [SystemMessage(content=system_prompt)]

    # On retry, inject feedback
    if retry_count > 0 and retry_feedback:
        llm_messages.append(SystemMessage(
            content=f"IMPORTANT: Your previous response failed validation. Reason: {retry_feedback}. "
            "Please generate a corrected response that addresses this issue."
        ))

    # Add conversation history (only HumanMessage and AIMessage, skip tool messages)
    for msg in state.get("messages", []):
        if isinstance(msg, (HumanMessage, AIMessage)):
            llm_messages.append(msg)

    llm = _get_llm()

    try:
        response = llm.invoke(llm_messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        return {"messages": [AIMessage(content=content)]}
    except Exception:
        logger.exception("Generate LLM call failed")
        return {
            "messages": [AIMessage(content=(
                "I apologize, but I'm having trouble generating a response right now. "
                "Please try again, or contact Mohegan Sun directly at 1-888-226-7711."
            ))],
            "retry_count": 99,  # Skip validation on error
        }


# ---------------------------------------------------------------------------
# 4. Validate Node
# ---------------------------------------------------------------------------


def validate_node(state: PropertyQAState) -> dict:
    """Adversarial review of the generated response against 6 criteria.

    If retry_count >= 99 (empty context or error), auto-PASS.
    If validation fails and retry_count < 1, returns RETRY.
    If retry_count >= 1, returns FAIL (max 1 retry).
    """
    retry_count = state.get("retry_count", 0)

    # Skip validation for empty-context responses
    if retry_count >= 99:
        return {"validation_result": "PASS"}

    # Get the user question
    user_question = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_question = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Get the generated response (last AI message)
    generated_response = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            generated_response = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Format retrieved context
    retrieved = state.get("retrieved_context", [])
    context_parts = []
    for i, doc in enumerate(retrieved, 1):
        category = doc.get("metadata", {}).get("category", "general")
        content = doc.get("content", "")
        context_parts.append(f"[{i}] ({category}) {content}")
    context_text = "\n".join(context_parts) if context_parts else "No context retrieved."

    prompt_text = VALIDATION_PROMPT.safe_substitute(
        user_question=user_question,
        retrieved_context=context_text,
        generated_response=generated_response,
    )

    llm = _get_llm()
    validator_llm = llm.with_structured_output(ValidationResult)

    try:
        result: ValidationResult = validator_llm.invoke(prompt_text)

        if result.status == "PASS":
            return {"validation_result": "PASS"}

        # RETRY or FAIL
        if retry_count < 1:
            return {
                "validation_result": "RETRY",
                "retry_count": retry_count + 1,
                "retry_feedback": result.reason,
            }

        # Already retried once — FAIL
        return {
            "validation_result": "FAIL",
            "retry_feedback": result.reason,
        }

    except Exception:
        logger.exception("Validation LLM call failed, auto-passing")
        return {"validation_result": "PASS"}


# ---------------------------------------------------------------------------
# 5. Respond Node
# ---------------------------------------------------------------------------


def respond_node(state: PropertyQAState) -> dict:
    """Extract sources from retrieved context and prepare final response.

    Clears retry_feedback. Sets sources_used from context metadata.
    """
    retrieved = state.get("retrieved_context", [])
    sources = []
    for doc in retrieved:
        category = doc.get("metadata", {}).get("category", "")
        if category and category not in sources:
            sources.append(category)

    return {
        "sources_used": sources,
        "retry_feedback": None,
    }


# ---------------------------------------------------------------------------
# 6. Fallback Node
# ---------------------------------------------------------------------------


def fallback_node(state: PropertyQAState) -> dict:
    """Safe fallback response when validation fails.

    Provides contact information and logs the failure reason.
    """
    retry_feedback = state.get("retry_feedback", "Unknown validation failure")
    logger.warning("Fallback triggered. Reason: %s", retry_feedback)

    return {
        "messages": [AIMessage(content=(
            "I want to make sure I give you the most accurate information. "
            "For this question, I'd recommend reaching out directly to Mohegan Sun:\n\n"
            "- Phone: 1-888-226-7711\n"
            "- Website: mohegansun.com\n\n"
            "They'll be able to help you with the most up-to-date details!"
        ))],
        "sources_used": [],
        "retry_feedback": None,
    }


# ---------------------------------------------------------------------------
# 7. Greeting Node
# ---------------------------------------------------------------------------


def greeting_node(state: PropertyQAState) -> dict:
    """Template welcome listing available knowledge categories."""
    settings = get_settings()
    return {
        "messages": [AIMessage(content=(
            f"Welcome to {settings.PROPERTY_NAME}! I'm your AI concierge, "
            "here to help you explore everything the resort has to offer.\n\n"
            "I can help with:\n"
            "- **Restaurants & Dining** — from casual to fine dining\n"
            "- **Entertainment & Shows** — concerts, comedy, and events\n"
            "- **Hotel & Accommodations** — rooms, suites, and towers\n"
            "- **Gaming** — casino floor, table games, and poker\n"
            "- **Amenities** — spa, pool, shopping, and more\n"
            "- **Promotions** — current offers and loyalty programs\n\n"
            "What would you like to know about?"
        ))],
        "sources_used": [],
    }


# ---------------------------------------------------------------------------
# 8. Off-Topic Node
# ---------------------------------------------------------------------------


def off_topic_node(state: PropertyQAState) -> dict:
    """Handle off-topic, gambling advice, and action requests.

    Three sub-cases based on query_type:
    - off_topic: General redirect to property topics
    - gambling_advice: Redirect with responsible gaming helplines
    - action_request: Explain read-only limitations
    """
    query_type = state.get("query_type", "off_topic")
    settings = get_settings()

    if query_type == "gambling_advice":
        content = (
            "I appreciate your interest, but I'm not able to provide gambling advice, "
            "betting strategies, or information about odds. I can share general information "
            f"about the gaming areas at {settings.PROPERTY_NAME}.\n\n"
            "If you or someone you know needs help with problem gambling, "
            "please reach out to these resources:\n"
            "- National Council on Problem Gambling: 1-800-522-4700\n"
            "- Connecticut Council on Problem Gambling: 1-888-789-7777\n"
            "- CT DMHAS Self-Exclusion Program: 1-860-418-7000\n\n"
            "Is there anything else about the resort I can help with?"
        )
    elif query_type == "action_request":
        content = (
            "I appreciate you asking! While I can't make reservations, bookings, "
            "or take any actions on your behalf, I can provide all the information "
            "you need to do so yourself.\n\n"
            f"For reservations and bookings, please contact {settings.PROPERTY_NAME} "
            "directly at 1-888-226-7711 or visit mohegansun.com.\n\n"
            "Is there any information I can help you with?"
        )
    else:
        # General off-topic
        content = (
            f"I'm your concierge for {settings.PROPERTY_NAME}, so I'm best equipped "
            "to answer questions about the resort — restaurants, entertainment, "
            "hotel rooms, gaming, amenities, and promotions.\n\n"
            "What would you like to know about the property?"
        )

    return {
        "messages": [AIMessage(content=content)],
        "sources_used": [],
    }


# ---------------------------------------------------------------------------
# Routing Functions (used as conditional edges)
# ---------------------------------------------------------------------------


def route_from_router(state: PropertyQAState) -> str:
    """Route after the router node based on query_type and confidence.

    Returns the name of the next node to execute.
    """
    query_type = state.get("query_type", "property_qa")
    confidence = state.get("router_confidence", 0.5)

    if query_type == "greeting":
        return "greeting"

    if query_type in ("off_topic", "gambling_advice", "action_request"):
        return "off_topic"

    if confidence < 0.3:
        return "off_topic"

    # property_qa, hours_schedule, ambiguous → retrieve
    return "retrieve"


def route_after_validate(state: PropertyQAState) -> str:
    """Route after the validate node based on validation result.

    Returns the name of the next node to execute.
    """
    result = state.get("validation_result", "PASS")

    if result == "PASS":
        return "respond"
    if result == "RETRY":
        return "generate"
    # FAIL
    return "fallback"
