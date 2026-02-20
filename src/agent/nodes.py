"""Node functions for the 11-node Property Q&A StateGraph (v2).

Each node takes PropertyQAState and returns a partial dict update.
Two routing functions determine conditional edges.

The ``generate_node`` from v1 has been removed — ``host_agent``
(from ``agents.host_agent``) is the generate node in v2.

Guardrail functions (``audit_input``, ``detect_responsible_gaming``, etc.)
live in ``guardrails.py``.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from cachetools import TTLCache

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.casino.feature_flags import is_feature_enabled
from src.config import get_settings

from .circuit_breaker import CircuitBreaker, _get_circuit_breaker  # noqa: F401 (CircuitBreaker re-exported for tests)
from .prompts import (
    RESPONSIBLE_GAMING_HELPLINES,
    ROUTER_PROMPT,
    VALIDATION_PROMPT,
)
from .state import PropertyQAState, RouterOutput, ValidationResult
from .tools import search_hours, search_knowledge_base

logger = logging.getLogger(__name__)

__all__ = [
    "router_node",
    "retrieve_node",
    "validate_node",
    "respond_node",
    "fallback_node",
    "greeting_node",
    "off_topic_node",
    "route_from_router",
]


def _format_context_block(retrieved: list[dict], separator: str = "\n---\n") -> str:
    """Format retrieved context as numbered sources for LLM consumption.

    Used by specialist agents (appended to system prompt) and
    ``validate_node`` (included in validation prompt) to ensure consistent
    context presentation across the pipeline.

    Args:
        retrieved: List of RetrievedChunk dicts with content and metadata.
        separator: String to join numbered source entries.

    Returns:
        Formatted context string, or "No context retrieved." if empty.
    """
    if not retrieved:
        return "No context retrieved."
    parts = []
    for i, doc in enumerate(retrieved, 1):
        category = doc.get("metadata", {}).get("category", "general")
        content = doc.get("content", "")
        parts.append(f"[{i}] ({category}) {content}")
    return separator.join(parts)


def _get_last_human_message(messages: list) -> str:
    """Extract the content of the last HumanMessage from a message list.

    Iterates in reverse to find the most recent user message.
    Returns an empty string if no HumanMessage is found.
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""



# ---------------------------------------------------------------------------
# LLM singleton (TTL-cached for credential rotation)
# ---------------------------------------------------------------------------

# TTL-cached LLM singletons: process-scoped with automatic refresh.
# TTL=3600s (1 hour) allows credential rotation (e.g., GCP Workload Identity
# Federation) without process restart.  Coroutine-safe via asyncio.Lock
# (non-blocking under contention, unlike threading.Lock which blocks the
# event loop).
#
# Separate locks per client type (main LLM vs validator LLM): if validator
# construction stalls (e.g., network hiccup during lazy init), it must NOT
# block main LLM acquisition.  This prevents cascading latency spikes and
# request pileups during partial outages.
_LLM_CACHE_TTL = 3600
_llm_cache: TTLCache = TTLCache(maxsize=1, ttl=_LLM_CACHE_TTL)
_llm_lock = asyncio.Lock()
_validator_cache: TTLCache = TTLCache(maxsize=1, ttl=_LLM_CACHE_TTL)
_validator_lock = asyncio.Lock()


async def _get_llm() -> ChatGoogleGenerativeAI:
    """Get or create the shared LLM instance (TTL-cached singleton).

    Cache refreshes every hour to pick up rotated credentials.
    Coroutine-safe via ``asyncio.Lock`` (non-blocking under contention,
    unlike the previous ``threading.Lock`` which blocked the event loop).
    Tests can ``patch("src.agent.nodes._get_llm")`` without interacting
    with the cache.
    """
    async with _llm_lock:
        cached = _llm_cache.get("llm")
        if cached is not None:
            return cached
        settings = get_settings()
        llm = ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            temperature=settings.MODEL_TEMPERATURE,
            timeout=settings.MODEL_TIMEOUT,
            max_retries=settings.MODEL_MAX_RETRIES,
            max_output_tokens=settings.MODEL_MAX_OUTPUT_TOKENS,
        )
        _llm_cache["llm"] = llm
        return llm


async def _get_validator_llm() -> ChatGoogleGenerativeAI:
    """Get or create the validation LLM instance (TTL-cached singleton).

    Uses temperature=0.0 for deterministic binary classification
    (PASS/RETRY/FAIL). Separate from ``_get_llm()`` which uses the
    configured temperature for creative response generation.

    Uses a **separate lock and cache** from ``_get_llm()`` so that
    validator construction stalls do not block main LLM acquisition
    (prevents cascading latency spikes during partial outages).

    Cache refreshes every hour (same TTL as ``_get_llm``).
    Coroutine-safe via ``asyncio.Lock``.
    """
    async with _validator_lock:
        cached = _validator_cache.get("validator")
        if cached is not None:
            return cached
        settings = get_settings()
        llm = ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            temperature=0.0,
            timeout=settings.MODEL_TIMEOUT,
            max_retries=settings.MODEL_MAX_RETRIES,
            max_output_tokens=512,  # Validation produces short structured output
        )
        _validator_cache["validator"] = llm
        return llm


# ---------------------------------------------------------------------------
# 1. Router Node
# ---------------------------------------------------------------------------


async def router_node(state: PropertyQAState) -> dict[str, Any]:
    """Classify user intent into one of 7 categories.

    Uses structured output for reliable JSON parsing.
    Turn-limit is enforced upstream by compliance_gate_node.
    """
    messages = state.get("messages", [])

    # Turn-limit check removed: compliance_gate_node runs upstream and
    # already enforces MAX_MESSAGE_LIMIT before messages reach the router.

    # Get the last human message
    user_message = _get_last_human_message(messages)

    if not user_message:
        return {
            "query_type": "greeting",
            "router_confidence": 1.0,
        }

    # Note: All 5 deterministic guardrail checks (prompt injection, responsible
    # gaming, age verification, BSA/AML, patron privacy) are handled by the
    # upstream compliance_gate_node.  Guardrail-triggered messages never reach
    # the router, so duplicate checks here are unnecessary.

    llm = await _get_llm()
    router_llm = llm.with_structured_output(RouterOutput)

    prompt_text = ROUTER_PROMPT.safe_substitute(user_message=user_message)

    try:
        result: RouterOutput = await router_llm.ainvoke(prompt_text)
        return {
            "query_type": result.query_type,
            "router_confidence": result.confidence,
        }
    except (ValueError, TypeError) as exc:
        # Structured output parsing failure (invalid JSON from LLM)
        logger.warning("Router structured output parsing failed: %s", exc)
        return {
            "query_type": "property_qa",
            "router_confidence": 0.5,
        }
    except Exception:
        # Network errors, API failures, timeouts — broad catch is intentional
        # because google-genai raises various exception types across versions.
        # KeyboardInterrupt/SystemExit propagate (not subclasses of Exception).
        # Fail-safe: route to off_topic since we cannot classify the query.
        # This is safer than defaulting to property_qa which would send
        # unclassified queries through the full RAG + LLM pipeline.
        logger.exception("Router LLM call failed, defaulting to off_topic (fail-safe)")
        return {
            "query_type": "off_topic",
            "router_confidence": 0.0,
        }


# ---------------------------------------------------------------------------
# 2. Retrieve Node
# ---------------------------------------------------------------------------


async def retrieve_node(state: PropertyQAState) -> dict[str, Any]:
    """Retrieve relevant documents from the knowledge base.

    Extracts the latest user message and searches for matching content.
    Uses schedule-focused search for hours_schedule queries.
    """
    messages = state.get("messages", [])
    query_type = state.get("query_type", "property_qa")

    query = _get_last_human_message(messages)

    if not query:
        return {"retrieved_context": []}

    # Use schedule-focused search for hours/schedule queries.
    # ChromaDB is sync-only in LangChain; wrap in asyncio.to_thread to avoid
    # blocking the event loop.  Production path (Vertex AI) has native async.
    # Timeout guard: prevents a hung ChromaDB query (e.g., corrupted SQLite)
    # from permanently blocking the event loop thread pool.
    _RETRIEVAL_TIMEOUT = 10  # seconds
    try:
        if query_type == "hours_schedule":
            results = await asyncio.wait_for(
                asyncio.to_thread(search_hours, query),
                timeout=_RETRIEVAL_TIMEOUT,
            )
        else:
            results = await asyncio.wait_for(
                asyncio.to_thread(search_knowledge_base, query),
                timeout=_RETRIEVAL_TIMEOUT,
            )
    except TimeoutError:
        logger.warning("Retrieval timed out after %ds for query: %s", _RETRIEVAL_TIMEOUT, query[:80])
        results = []
    except Exception:
        # Broad catch: ChromaDB can raise ChromaError, sqlite3.OperationalError;
        # Firestore can raise google.api_core.exceptions.*; embedding model can
        # raise ValueError on dimension mismatch.  Without this, non-timeout
        # retrieval errors propagate to graph.py's top-level except, aborting
        # the entire graph execution with no fallback response.  The empty
        # results path gracefully degrades to the no-context specialist fallback.
        logger.exception("Retrieval failed for query: %s", query[:80])
        results = []

    return {"retrieved_context": results}


# ---------------------------------------------------------------------------
# 3. Validate Node (generate_node removed in v2 — host_agent is the generate node)
# ---------------------------------------------------------------------------


async def validate_node(state: PropertyQAState) -> dict[str, Any]:
    """Adversarial review of the generated response against 6 criteria.

    If ``skip_validation`` is True (empty context or error), auto-PASS.
    If validation fails and retry_count < 1, returns RETRY.
    If retry_count >= 1, returns FAIL (max 1 retry).
    """
    # Skip validation for deterministic fallback responses
    if state.get("skip_validation", False):
        return {"validation_result": "PASS"}

    retry_count = state.get("retry_count", 0)

    # Get the user question
    user_question = _get_last_human_message(state.get("messages", []))

    # Get the generated response (last AI message)
    generated_response = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            generated_response = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Format retrieved context (shared helper with generate_node for consistency)
    retrieved = state.get("retrieved_context", [])
    context_text = _format_context_block(retrieved, separator="\n")

    prompt_text = VALIDATION_PROMPT.safe_substitute(
        user_question=user_question,
        retrieved_context=context_text,
        generated_response=generated_response,
    )

    validator_llm = (await _get_validator_llm()).with_structured_output(ValidationResult)

    try:
        result: ValidationResult = await validator_llm.ainvoke(prompt_text)

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

    except (ValueError, TypeError) as exc:
        logger.warning("Validation structured output parsing failed: %s", exc)
        # Degraded-pass strategy: balance availability vs safety.
        if retry_count == 0:
            # First attempt: generate succeeded, only validator failed.
            # Deterministic guardrails already passed. Serve unvalidated.
            logger.warning(
                "Degraded-pass: serving unvalidated response (first attempt, "
                "validator unavailable)"
            )
            return {"validation_result": "PASS"}
        # Retry attempt: prior validation issues + validator failure = suspect.
        # Fail-closed to protect guest safety.
        return {
            "validation_result": "FAIL",
            "retry_feedback": "Validation unavailable — returning safe fallback for guest safety.",
        }
    except Exception:
        logger.exception("Validation LLM call failed")
        if retry_count == 0:
            logger.warning(
                "Degraded-pass: serving unvalidated response (first attempt, "
                "validator unavailable)"
            )
            return {"validation_result": "PASS"}
        return {
            "validation_result": "FAIL",
            "retry_feedback": "Validation unavailable — returning safe fallback for guest safety.",
        }


# ---------------------------------------------------------------------------
# 5. Respond Node
# ---------------------------------------------------------------------------


async def respond_node(state: PropertyQAState) -> dict[str, Any]:
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


async def fallback_node(state: PropertyQAState) -> dict[str, Any]:
    """Safe fallback response when validation fails.

    Provides contact information and logs the failure reason.
    """
    settings = get_settings()
    retry_feedback = state.get("retry_feedback", "Unknown validation failure")
    logger.warning("Fallback triggered. Reason: %s", retry_feedback)

    return {
        "messages": [AIMessage(content=(
            "I want to make sure I give you the most accurate information. "
            f"For this question, I'd recommend reaching out directly to {settings.PROPERTY_NAME}:\n\n"
            f"- Phone: {settings.PROPERTY_PHONE}\n"
            f"- Website: {settings.PROPERTY_WEBSITE}\n\n"
            "They'll be able to help you with the most up-to-date details!"
        ))],
        "sources_used": [],
        "retry_feedback": None,
    }


# ---------------------------------------------------------------------------
# 7. Greeting Node
# ---------------------------------------------------------------------------


_KNOWN_CATEGORY_LABELS: dict[str, str] = {
    "restaurants": "Restaurants & Dining — from casual to fine dining",
    "entertainment": "Entertainment & Shows — concerts, comedy, and events",
    "hotel": "Hotel & Accommodations — rooms, suites, and towers",
    "gaming": "Gaming — casino floor, table games, and poker",
    "amenities": "Amenities — spa, pool, shopping, and more",
    "promotions": "Promotions — current offers and loyalty programs",
    "spa": "Spa & Wellness — treatments and relaxation",
    "shopping": "Shopping — retail and boutiques",
}
"""Curated labels for known property-data categories."""


_greeting_cache: TTLCache = TTLCache(maxsize=8, ttl=3600)
"""Per-casino greeting categories cache (multi-tenant safe, 1-hour TTL)."""


def _build_greeting_categories(casino_id: str | None = None) -> dict[str, str]:
    """Derive greeting categories from the actual property data file.

    Reads the property JSON at startup (cached per casino_id), so adding a
    new category to the data file automatically surfaces it in the greeting.
    Falls back to ``_KNOWN_CATEGORY_LABELS`` if the file cannot be read.
    """
    cache_key = casino_id or get_settings().CASINO_ID
    if cache_key in _greeting_cache:
        return _greeting_cache[cache_key]

    settings = get_settings()
    try:
        path = Path(settings.PROPERTY_DATA_PATH)
        if not path.exists():
            result = dict(_KNOWN_CATEGORY_LABELS)
            _greeting_cache[cache_key] = result
            return result
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            result = dict(_KNOWN_CATEGORY_LABELS)
            _greeting_cache[cache_key] = result
            return result
        categories: dict[str, str] = {}
        for key in data:
            if key == "property":
                continue
            if key in _KNOWN_CATEGORY_LABELS:
                categories[key] = _KNOWN_CATEGORY_LABELS[key]
            else:
                categories[key] = f"{key.replace('_', ' ').title()} — information and details"
        result = categories if categories else dict(_KNOWN_CATEGORY_LABELS)
        _greeting_cache[cache_key] = result
        return result
    except Exception:
        logger.warning("Could not load property data for greeting categories, using defaults")
        result = dict(_KNOWN_CATEGORY_LABELS)
        _greeting_cache[cache_key] = result
        return result


async def greeting_node(state: PropertyQAState) -> dict[str, Any]:
    """Template welcome listing available knowledge categories.

    Categories are derived from the actual property data file (cached)
    to stay in sync with available knowledge-base content.

    Feature flag: ``ai_disclosure_enabled`` — when True, includes an
    explicit AI disclosure line ("I'm an AI assistant") per regulatory
    best practice for automated guest interactions.
    """
    settings = get_settings()
    categories = _build_greeting_categories(casino_id=settings.CASINO_ID)
    bullets = "\n".join(f"- **{label}**" for label in categories.values())

    # Feature flag: AI disclosure for regulatory transparency (async API for multi-tenant)
    ai_disclosure = await is_feature_enabled(get_settings().CASINO_ID, "ai_disclosure_enabled")
    disclosure_line = " I'm an AI assistant, " if ai_disclosure else " "

    return {
        "messages": [AIMessage(content=(
            f"Hi! I'm **Seven**, your AI concierge for {settings.PROPERTY_NAME}."
            f"{disclosure_line}"
            "I'm here to help you explore everything the resort has to offer.\n\n"
            f"I can help with:\n{bullets}\n\n"
            "What would you like to know?"
        ))],
        "sources_used": [],
    }


# ---------------------------------------------------------------------------
# 8. Off-Topic Node
# ---------------------------------------------------------------------------


async def off_topic_node(state: PropertyQAState) -> dict[str, Any]:
    """Handle off-topic, gambling advice, and action requests.

    Feature flags consulted:
    - ``ai_disclosure_enabled``: Appends AI disclosure to off-topic responses.

    Three sub-cases based on query_type:
    - off_topic: General redirect to property topics
    - gambling_advice: Redirect with responsible gaming helplines
    - action_request: Explain read-only limitations
    """
    query_type = state.get("query_type", "off_topic")
    settings = get_settings()

    if query_type == "patron_privacy":
        content = (
            "I'm not able to share information about other guests, including whether "
            "someone is a member, their presence at the property, or any personal details. "
            "Guest privacy is a top priority.\n\n"
            f"If you need to reach someone, I'd suggest contacting them directly or calling "
            f"{settings.PROPERTY_NAME} at {settings.PROPERTY_PHONE}.\n\n"
            "Is there anything else about the resort I can help with?"
        )
    elif query_type == "gambling_advice":
        # Feature flag: ai_disclosure_enabled adds transparency to gambling-advice responses
        ai_disclosure = await is_feature_enabled(get_settings().CASINO_ID, "ai_disclosure_enabled")
        disclosure_suffix = (
            "\n\n*As an AI assistant, I'm required to direct you to these "
            "professional resources rather than provide gambling guidance.*"
            if ai_disclosure else ""
        )
        # Escalation: after 3+ responsible gaming triggers in a session,
        # add a stronger message encouraging live support contact.
        rg_count = state.get("responsible_gaming_count", 0)
        escalation_msg = ""
        if rg_count >= 3:
            escalation_msg = (
                f"\n\n**I've noticed you've raised this topic several times.** "
                f"I strongly encourage you to speak with a live team member who "
                f"can provide confidential, personalized support. You can reach "
                f"{settings.PROPERTY_NAME}'s Responsible Gaming team directly "
                f"at {settings.PROPERTY_PHONE}, or visit the Responsible Gaming "
                f"desk on-property."
            )
        content = (
            "I appreciate your interest, but I'm not able to provide gambling advice, "
            "betting strategies, or information about odds. I can share general information "
            f"about the gaming areas at {settings.PROPERTY_NAME}.\n\n"
            "If you or someone you know needs help with problem gambling, "
            "please reach out to these resources:\n"
            f"{RESPONSIBLE_GAMING_HELPLINES}"
            f"{escalation_msg}\n\n"
            f"Is there anything else about the resort I can help with?{disclosure_suffix}"
        )
    elif query_type == "age_verification":
        content = (
            f"Great question! At {settings.PROPERTY_NAME}, guests must be **21 years of age or older** "
            "to access the gaming floor, purchase or consume alcohol, and enter most entertainment venues.\n\n"
            "**What minors CAN do:**\n"
            "- Walk through designated non-gaming areas\n"
            "- Dine at select restaurants (with an adult)\n"
            f"- Visit the shops at {settings.PROPERTY_NAME}\n\n"
            "**What requires 21+:**\n"
            "- Casino gaming floor\n"
            "- Table games, slots, and poker\n"
            "- Bars and lounges\n"
            "- Most entertainment venues\n\n"
            f"Valid government-issued photo ID is required. {settings.PROPERTY_STATE} state law strictly "
            "enforces the 21+ age requirement for gaming.\n\n"
            "Is there anything else I can help you with?"
        )
    elif query_type == "action_request":
        content = (
            "I appreciate you asking! While I can't make reservations, bookings, "
            "or take any actions on your behalf, I can provide all the information "
            "you need to do so yourself.\n\n"
            f"For reservations and bookings, please contact {settings.PROPERTY_NAME} "
            f"directly at {settings.PROPERTY_PHONE} or visit {settings.PROPERTY_WEBSITE}.\n\n"
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

    Design decision: ``ambiguous`` queries route to ``retrieve`` (same as
    ``property_qa`` and ``hours_schedule``).  The RAG pipeline + validation
    loop handles ambiguity safely: if relevant context exists, the response
    is grounded and validated; if not, the empty-context fallback provides
    a safe "contact the property" message.  This is preferable to routing
    ambiguous queries to ``off_topic``, which would refuse to help with
    legitimate-but-unclear property questions.
    """
    query_type = state.get("query_type", "property_qa")
    confidence = state.get("router_confidence", 0.5)

    if query_type == "greeting":
        return "greeting"

    if query_type in ("off_topic", "gambling_advice", "action_request", "age_verification", "patron_privacy"):
        return "off_topic"

    if confidence < 0.3:
        return "off_topic"

    # property_qa, hours_schedule, and ambiguous all route to retrieve.
    # See docstring for ambiguous rationale.
    return "retrieve"


