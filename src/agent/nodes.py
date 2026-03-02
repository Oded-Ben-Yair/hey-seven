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
    ROUTER_PROMPT,
    VALIDATION_PROMPT,
    get_responsible_gaming_helplines,
)
from .constants import NODE_GREETING, NODE_OFF_TOPIC, NODE_RETRIEVE
from .extraction import extract_fields
from .sentiment import detect_sentiment
from .slang import normalize_for_search
from .state import PropertyQAState, RouterOutput, ValidationResult
from .tools import search_hours, search_knowledge_base
from src.data.validators import validate_retrieved_chunk

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


# ---------------------------------------------------------------------------
# i18n Architecture Decision Record (ADR)
# ---------------------------------------------------------------------------
# DECISION: Response language is English-only for MVP
# CONTEXT: Guardrails detect 4 languages (EN, ES, PT, ZH) for safety-critical
#   patterns (responsible gaming, BSA/AML). However, the concierge response
#   is always English because:
#   1. System prompt is English-only
#   2. RAG knowledge base is English-only
#   3. LLM validation prompt is English-only
#   4. Multi-language response requires per-language prompt templates,
#      localized helplines, and localized validation — significant scope
# PLANNED APPROACH (post-MVP):
#   1. Detect language in router (add to RouterOutput)
#   2. Select localized prompt template (CONCIERGE_SYSTEM_PROMPT_ES, etc.)
#   3. Select localized helplines from casino profile
#   4. Validate in the response language
# STATUS: Guardrails multilingual (EN+ES+PT+ZH), responses English-only
# ---------------------------------------------------------------------------


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
# ADR (R40 fix D8-C001): Stagger TTLs with random jitter (0-300s) to prevent
# thundering herd when all singletons expire simultaneously after startup.
# Why 0-300s: spreads re-creation across a 5-minute window instead of a single
# instant. With 50 concurrent SSE streams, simultaneous expiry causes 50 parallel
# GCP credential lookups. Additive jitter (not multiplicative) chosen for
# simplicity — the absolute spread matters more than percentage for this use case.
# RNG: random.randint() is non-cryptographic, appropriate for timing jitter.
import random as _random

_LLM_CACHE_TTL = 3600
_llm_cache: TTLCache = TTLCache(maxsize=1, ttl=_LLM_CACHE_TTL + _random.randint(0, 300))
_llm_lock = asyncio.Lock()
_validator_cache: TTLCache = TTLCache(maxsize=1, ttl=_LLM_CACHE_TTL + _random.randint(0, 300))
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
            temperature=0.3,  # R77: Relaxed from 0.0 to reduce over-strict grounding rejections
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
            "detected_language": None,
        }

    # R36 fix A1: Hoist settings once to eliminate TOCTOU race. Same pattern
    # as _dispatch_to_specialist (R35 fix A2). If TTL expires between calls,
    # the function would operate with mixed config values.
    settings = get_settings()

    # Phase 3: Detect guest sentiment before LLM routing (sub-1ms VADER).
    # Feature flag: sentiment_detection_enabled (default True).
    # R76 fix: Do NOT overwrite guest_sentiment if compliance_gate already
    # set it to a high-priority value like "grief". Compliance gate grief
    # detection (position 7.6) runs before the router and sets guest_sentiment
    # to "grief" — VADER would overwrite this with "neutral" or "negative",
    # causing the specialist to lose the grief context entirely.
    sentiment_update: dict[str, Any] = {}
    _existing_sentiment = state.get("guest_sentiment")
    _PRIORITY_SENTIMENTS = ("grief", "celebration")  # Set by compliance_gate, never overwrite
    if _existing_sentiment not in _PRIORITY_SENTIMENTS:
        if await is_feature_enabled(settings.CASINO_ID, "sentiment_detection_enabled"):
            sentiment = detect_sentiment(user_message)
            sentiment_update["guest_sentiment"] = sentiment
            # Phase 5: LLM augmentation for ambiguous VADER band (neutral, 10+ words).
            # Feature flag: sentiment_llm_augmented (default True).
            if sentiment == "neutral" and await is_feature_enabled(
                settings.CASINO_ID, "sentiment_llm_augmented"
            ):
                from src.agent.sentiment import detect_sentiment_augmented

                augmented = await detect_sentiment_augmented(
                    user_message, _get_llm, vader_result=sentiment
                )
                sentiment_update["guest_sentiment"] = augmented

    # Phase 4: Deterministic field extraction (sub-1ms regex, no LLM).
    # Populates extracted_fields for whisper planner + guest profile.
    # Feature flag: field_extraction_enabled (default True).
    extraction_update: dict[str, Any] = {}
    if await is_feature_enabled(settings.CASINO_ID, "field_extraction_enabled"):
        extracted = extract_fields(user_message)
        if extracted:
            # Merge with existing fields (accumulate across turns)
            existing = dict(state.get("extracted_fields", {}) or {})
            existing.update(extracted)
            extraction_update["extracted_fields"] = existing
            # Also set guest_name if extracted
            if extracted.get("name"):
                extraction_update["guest_name"] = extracted["name"]

    # Phase 5: LLM fallback for regex extraction misses (15+ word messages).
    # Feature flag: extraction_llm_augmented (default True).
    if (
        not extraction_update.get("extracted_fields")
        and len(user_message.split()) >= 15
        and await is_feature_enabled(settings.CASINO_ID, "extraction_llm_augmented")
    ):
        from src.agent.extraction import extract_fields_augmented

        augmented = await extract_fields_augmented(
            user_message, state.get("extracted_fields", {}), _get_llm
        )
        if augmented:
            existing = dict(state.get("extracted_fields", {}) or {})
            existing.update(augmented)
            extraction_update["extracted_fields"] = existing
            if augmented.get("name"):
                extraction_update["guest_name"] = augmented["name"]

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
            "detected_language": result.detected_language,
            **sentiment_update,
            **extraction_update,
        }
    except (ValueError, TypeError) as exc:
        # Structured output parsing failure (invalid JSON from LLM)
        logger.warning("Router structured output parsing failed: %s", exc)
        return {
            "query_type": "property_qa",
            "router_confidence": 0.5,
            "detected_language": "en",
            **sentiment_update,
            **extraction_update,
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
            "detected_language": "en",
            **sentiment_update,
            **extraction_update,
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

    # R72 C3: Normalize gambling slang and drunk-typing for better RAG retrieval.
    # Normalization is for SEARCH ONLY — the original message is preserved in state.
    # Example: "cn u get me a rm upgrde" → "can you get me a room upgrade"
    # Example: "I'm on tilt, need somewhere to eat" → "I'm frustrated after losing, need somewhere to eat"
    query = normalize_for_search(query)

    # R60 fix D2: Removed redundant outer asyncio.wait_for wrapper.
    # search_knowledge_base and search_hours have internal per-strategy
    # timeouts (asyncio.wait_for per strategy with settings.RETRIEVAL_TIMEOUT).
    # The outer timeout created a race condition: it could cancel before
    # internal timeouts fired, losing partial results from a strategy that
    # completed successfully. Both functions return [] on total failure.
    try:
        if query_type == "hours_schedule":
            results = await search_hours(query)
        else:
            results = await search_knowledge_base(query)
    except Exception:
        # Broad catch: ChromaDB can raise ChromaError, sqlite3.OperationalError;
        # Firestore can raise google.api_core.exceptions.*; embedding model can
        # raise ValueError on dimension mismatch.  Without this, non-timeout
        # retrieval errors propagate to graph.py's top-level except, aborting
        # the entire graph execution with no fallback response.  The empty
        # results path gracefully degrades to the no-context specialist fallback.
        logger.exception("Retrieval failed for query: %s", query[:80])
        results = []

    # R69 fix D3: Validate each retrieved chunk before passing to graph state.
    # Invalid chunks (missing fields, wrong types) are logged and skipped.
    # Validation failure is non-fatal — degraded retrieval is better than crash.
    validated: list = []
    for chunk in results:
        try:
            if validate_retrieved_chunk(chunk):
                validated.append(chunk)
            else:
                logger.warning("Skipping invalid retrieved chunk: %s", str(chunk)[:100])
        except Exception:
            logger.warning("Chunk validation raised exception, skipping", exc_info=True)
    return {"retrieved_context": validated}


# ---------------------------------------------------------------------------
# 3. Validate Node (generate_node removed in v2 — host_agent is the generate node)
# ---------------------------------------------------------------------------


def _degraded_pass_result(retry_count: int, has_grounding: bool = True) -> dict[str, Any]:
    """Return degraded-pass or fail-closed result based on attempt number and grounding.

    R38 fix M-001: Extracted from duplicate try/except blocks in validate_node.
    R74 fix: Added grounding check. Degrade-pass on first attempt is only safe
    when retrieval context exists. Without grounding, an unvalidated response
    is likely hallucinated — route to fallback instead.
    """
    if retry_count == 0 and has_grounding:
        logger.warning(
            "Degraded-pass: serving unvalidated response (first attempt, "
            "validator unavailable, grounding present)"
        )
        return {"validation_result": "PASS"}
    return {
        "validation_result": "FAIL",
        "retry_feedback": "Validation unavailable — returning safe fallback for guest safety.",
    }


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
    has_grounding = bool(retrieved)
    context_text = _format_context_block(retrieved, separator="\n")

    prompt_text = VALIDATION_PROMPT.safe_substitute(
        user_question=user_question,
        retrieved_context=context_text,
        generated_response=generated_response,
    )

    validator_llm = (await _get_validator_llm()).with_structured_output(ValidationResult)

    try:
        result: ValidationResult = await validator_llm.ainvoke(prompt_text)

        # R77 fix: Gemini sometimes returns empty or truncated status.
        # If status is missing/empty and grounding exists, default to PASS.
        if not result.status or result.status not in ("PASS", "FAIL", "RETRY"):
            if has_grounding:
                logger.warning(
                    "Validator returned invalid status '%s' with grounding — defaulting to PASS",
                    result.status,
                )
                return {"validation_result": "PASS"}
            else:
                logger.warning(
                    "Validator returned invalid status '%s' without grounding — defaulting to FAIL",
                    result.status,
                )
                return {
                    "validation_result": "FAIL",
                    "retry_feedback": "Validation returned invalid status",
                }

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
        return _degraded_pass_result(retry_count, has_grounding=has_grounding)
    except Exception:
        logger.exception("Validation LLM call failed")
        return _degraded_pass_result(retry_count, has_grounding=has_grounding)


# ---------------------------------------------------------------------------
# 5. Respond Node
# ---------------------------------------------------------------------------


async def respond_node(state: PropertyQAState) -> dict[str, Any]:
    """Extract sources from retrieved context and prepare final response.

    Clears retry_feedback. Sets sources_used from context metadata with
    richer provenance (category, source file, retrieval score).

    Wave 2 fix D2: Changed sources_used from list[str] (category names only)
    to list[dict] with category, source, and score for citation provenance.
    """
    retrieved = state.get("retrieved_context", [])
    sources: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for doc in retrieved:
        meta = doc.get("metadata", {})
        category = meta.get("category", "")
        source = meta.get("source", "")
        source_key = f"{category}:{source}"
        if source_key and source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "category": category,
                "source": source,
                "score": round(doc.get("score", 0.0), 3),
            })

    return {
        "sources_used": sources,
        "retry_feedback": None,
        # R41 fix D1-M001: Clear retrieved_context before checkpoint write to
        # prevent stale chunk data from accumulating in Firestore checkpoints.
        "retrieved_context": [],
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

    # R76 fix: Improved fallback to be less corporate and more helpful.
    # The old "I want to make sure I give you the most accurate information"
    # was the single most common UX failure — guests got this redirect for
    # follow-up questions, emotional inputs, and anything the validator rejected.
    return {
        "messages": [AIMessage(content=(
            f"I don't have enough specific details to answer that fully, but "
            f"the team at {settings.PROPERTY_NAME} can help — "
            f"call {settings.PROPERTY_PHONE} or visit {settings.PROPERTY_WEBSITE}.\n\n"
            "Is there anything else about the property I can help with?"
        ))],
        "sources_used": [],
        "retry_feedback": None,
        # R41 fix D1-M001: Clear before checkpoint write (same as respond_node).
        "retrieved_context": [],
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


_greeting_cache: TTLCache = TTLCache(maxsize=8, ttl=3600 + _random.randint(0, 300))
"""Per-casino greeting categories cache (multi-tenant safe, 1-hour TTL + jitter)."""


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

    # Phase 1: Spanish greeting when language detected and feature enabled
    detected_lang = state.get("detected_language")
    if detected_lang == "es" and await is_feature_enabled(settings.CASINO_ID, "spanish_support_enabled"):
        from src.agent.prompts import GREETING_TEMPLATE_ES
        return {
            "messages": [AIMessage(content=GREETING_TEMPLATE_ES.safe_substitute(
                property_name=settings.PROPERTY_NAME,
                categories=bullets,
            ))],
            "sources_used": [],
            "retrieved_context": [],
        }

    # Feature flag: AI disclosure for regulatory transparency (async API for multi-tenant)
    # R17 fix: Gemini M-002 — reuse already-fetched settings variable.
    ai_disclosure = await is_feature_enabled(settings.CASINO_ID, "ai_disclosure_enabled")
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
        # R41 fix D1-M001: Defensive clear (greeting never follows retrieve,
        # but ensures clean checkpoint state).
        "retrieved_context": [],
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

    # Phase 1: Detect language once for all off-topic branches
    detected_lang = state.get("detected_language")
    _is_spanish = (
        detected_lang == "es"
        and await is_feature_enabled(settings.CASINO_ID, "spanish_support_enabled")
    )

    if query_type == "bsa_aml":
        if _is_spanish:
            content = (
                "Gracias por tu pregunta. Los asuntos relacionados con el cumplimiento "
                "financiero y los requisitos de informes son manejados por nuestro equipo "
                "dedicado de cumplimiento. Para asistencia, por favor habla con un anfitrión "
                "del casino o contacta a nuestro departamento de cumplimiento directamente. "
                "¿Hay algo más sobre las amenidades del resort en lo que pueda ayudarte?"
            )
        else:
            content = (
                "Thank you for your question. Matters related to financial compliance "
                "and reporting requirements are handled by our dedicated compliance team. "
                "For assistance, please speak with a casino host or contact our compliance "
                "department directly. Is there anything else about our resort amenities "
                "I can help you with?"
            )
    elif query_type == "patron_privacy":
        if _is_spanish:
            content = (
                "No puedo compartir información sobre otros huéspedes, incluyendo si "
                "alguien es miembro, su presencia en la propiedad, o cualquier detalle personal. "
                "La privacidad de los huéspedes es una prioridad.\n\n"
                f"Si necesitas contactar a alguien, te sugiero comunicarte directamente o llamar "
                f"a {settings.PROPERTY_NAME} al {settings.PROPERTY_PHONE}.\n\n"
                "¿Hay algo más sobre el resort en lo que pueda ayudarte?"
            )
        else:
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
        # R17 fix: Gemini M-002 — reuse already-fetched settings variable.
        ai_disclosure = await is_feature_enabled(settings.CASINO_ID, "ai_disclosure_enabled")
        # Escalation: after 3+ responsible gaming triggers in a session,
        # add a stronger message encouraging live support contact.
        rg_count = state.get("responsible_gaming_count", 0)
        if _is_spanish:
            from src.agent.prompts import get_responsible_gaming_helplines_es
            disclosure_suffix_es = (
                "\n\n*Como asistente de IA, estoy obligado a dirigirte a estos "
                "recursos profesionales en lugar de proporcionar orientación sobre el juego.*"
                if ai_disclosure else ""
            )
            escalation_msg = ""
            if rg_count >= 3:
                escalation_msg = (
                    f"\n\n**He notado que has mencionado este tema varias veces.** "
                    f"Te recomiendo fuertemente hablar con un miembro del equipo en vivo "
                    f"que pueda brindarte apoyo confidencial y personalizado. Puedes contactar "
                    f"al equipo de Juego Responsable de {settings.PROPERTY_NAME} directamente "
                    f"al {settings.PROPERTY_PHONE}."
                )
            content = (
                "Agradezco tu interés, pero no puedo proporcionar consejos de juego, "
                "estrategias de apuestas, o información sobre probabilidades. Puedo compartir "
                f"información general sobre las áreas de juego de {settings.PROPERTY_NAME}.\n\n"
                "Si tú o alguien que conoces necesita ayuda con problemas de juego, "
                "por favor comunícate con estos recursos:\n"
                f"{get_responsible_gaming_helplines_es(casino_id=settings.CASINO_ID)}"
                f"{escalation_msg}\n\n"
                f"¿Hay algo más sobre el resort en lo que pueda ayudarte?{disclosure_suffix_es}"
            )
        else:
            disclosure_suffix = (
                "\n\n*As an AI assistant, I'm required to direct you to these "
                "professional resources rather than provide gambling guidance.*"
                if ai_disclosure else ""
            )
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
                f"{get_responsible_gaming_helplines(casino_id=settings.CASINO_ID)}"
                f"{escalation_msg}\n\n"
                f"Is there anything else about the resort I can help with?{disclosure_suffix}"
            )
    elif query_type == "age_verification":
        if _is_spanish:
            content = (
                f"En {settings.PROPERTY_NAME}, los huéspedes deben tener **21 años o más** "
                "para acceder al piso de juegos, comprar o consumir alcohol, y entrar a la mayoría "
                "de los lugares de entretenimiento.\n\n"
                "**Lo que los menores PUEDEN hacer:**\n"
                "- Caminar por las áreas designadas sin juegos\n"
                "- Cenar en restaurantes selectos (con un adulto)\n"
                f"- Visitar las tiendas de {settings.PROPERTY_NAME}\n\n"
                "**Lo que requiere 21+:**\n"
                "- Piso de juegos del casino\n"
                "- Juegos de mesa, tragamonedas y póker\n"
                "- Bares y lounges\n"
                "- La mayoría de los lugares de entretenimiento\n\n"
                "Se requiere identificación gubernamental con foto válida. "
                "La ley estatal exige estrictamente el requisito de 21+ para el juego.\n\n"
                "¿Hay algo más en lo que pueda ayudarte?"
            )
        else:
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
        if _is_spanish:
            content = (
                "Agradezco que preguntes. Aunque no puedo hacer reservaciones, reservas, "
                "o tomar acciones en tu nombre, puedo darte toda la información "
                "que necesitas para hacerlo tú mismo.\n\n"
                f"Para reservaciones, por favor contacta a {settings.PROPERTY_NAME} "
                f"directamente al {settings.PROPERTY_PHONE} o visita {settings.PROPERTY_WEBSITE}.\n\n"
                "¿Hay alguna información en la que pueda ayudarte?"
            )
        else:
            content = (
                "I appreciate you asking! While I can't make reservations, bookings, "
                "or take any actions on your behalf, I can provide all the information "
                "you need to do so yourself.\n\n"
                f"For reservations and bookings, please contact {settings.PROPERTY_NAME} "
                f"directly at {settings.PROPERTY_PHONE} or visit {settings.PROPERTY_WEBSITE}.\n\n"
                "Is there any information I can help you with?"
            )
    elif query_type == "self_harm":
        # R50 fix (Grok CRITICAL-D1-001): Self-harm crisis response with 988 Lifeline.
        # compliance_gate detects crisis language and routes here. The response
        # prioritizes safety resources over property information. Never dismissive,
        # never minimizing, always empathetic with concrete action steps.
        from src.agent.handoff import build_handoff_request

        if _is_spanish:
            from src.agent.crisis import get_crisis_response_es
            content = get_crisis_response_es(settings.PROPERTY_NAME, settings.PROPERTY_PHONE)
        else:
            content = (
                "I can hear that you're going through a really difficult time, and I want "
                "you to know that help is available right now.\n\n"
                "**Please reach out to these confidential resources:**\n\n"
                "- **988 Suicide & Crisis Lifeline**: Call or text **988** (24/7, free, confidential)\n"
                "- **Crisis Text Line**: Text **HOME** to **741741**\n"
                "- **Emergency**: Call **911** if you or someone is in immediate danger\n\n"
                "You don't have to face this alone. Trained counselors are available "
                "right now who understand what you're going through and can help.\n\n"
                f"If you'd like to speak with someone at {settings.PROPERTY_NAME} in person, "
                f"any team member can connect you with support services. You can also call us "
                f"at {settings.PROPERTY_PHONE}."
            )
        # Phase 5: Early return with structured handoff for crisis situations.
        return {
            "messages": [AIMessage(content=content)],
            "sources_used": [],
            "retrieved_context": [],
            "handoff_request": build_handoff_request(
                department="responsible_gaming",
                reason="Guest expressing self-harm or crisis indicators",
                extracted_fields=state.get("extracted_fields"),
                urgency="critical",
            ).model_dump(),
        }
    else:
        # General off-topic — genuinely unrelated to the property.
        # Kept brief and non-robotic. Does NOT fire for emotional or
        # conversational messages (those now route through ambiguous → retrieve).
        if _is_spanish:
            from src.agent.prompts import OFF_TOPIC_RESPONSE_ES
            content = OFF_TOPIC_RESPONSE_ES.safe_substitute(property_name=settings.PROPERTY_NAME)
        else:
            content = (
                "That's outside what I can help with, but I'm happy to assist with "
                f"anything about {settings.PROPERTY_NAME} — dining, entertainment, "
                "hotel, spa, or gaming. What can I help you find?"
            )

    return {
        "messages": [AIMessage(content=content)],
        "sources_used": [],
        # R41 fix D1-M001: Defensive clear (off_topic never follows retrieve,
        # but ensures clean checkpoint state).
        "retrieved_context": [],
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
        return NODE_GREETING

    if query_type in ("off_topic", "gambling_advice", "action_request", "age_verification", "patron_privacy", "bsa_aml"):
        return NODE_OFF_TOPIC

    if confidence < 0.3:
        return NODE_OFF_TOPIC

    # property_qa, hours_schedule, and ambiguous all route to retrieve.
    # See docstring for ambiguous rationale.
    return NODE_RETRIEVE


