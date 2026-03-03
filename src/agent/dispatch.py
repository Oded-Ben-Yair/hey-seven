"""Specialist agent dispatch logic for the Property Q&A StateGraph.

Extracted from graph.py (R52 D1 decomposition) to keep graph.py focused
on graph topology and streaming. This module handles:
- Routing queries to specialist agents (LLM dispatch + keyword fallback)
- Guest context injection from profile data
- Specialist execution with timeout and result sanitization
"""

import asyncio
import logging
from string import Template as _StrTemplate
from types import MappingProxyType as _MappingProxy
from typing import Any

from langchain_core.messages import AIMessage

from src.casino.feature_flags import is_feature_enabled
from src.config import get_settings

from .agents.registry import _AGENT_REGISTRY, get_agent
from .circuit_breaker import _get_circuit_breaker
from .constants import (
    NODE_COMPLIANCE_GATE,
    NODE_GENERATE,
    NODE_RESPOND,
    NODE_RETRIEVE,
    NODE_ROUTER,
    NODE_VALIDATE,
    NODE_WHISPER,
)
from .nodes import _get_last_human_message, _get_llm, _select_model
from .state import DispatchOutput, PropertyQAState

logger = logging.getLogger(__name__)

# Keys that only _dispatch_to_specialist may set -- specialist agents should
# not return these.  Module-level for zero per-call allocation (R33 fix).
_DISPATCH_OWNED_KEYS = frozenset({"guest_context", "guest_name"})

# R45 fix D1-M002: Hoist valid state keys to module level. Previously computed
# inside _execute_specialist on every call via frozenset(PropertyQAState.__annotations__).
# Since PropertyQAState annotations are fixed at import time, this is safe and
# eliminates per-call frozenset construction (~50 concurrent streams * 1 alloc each).
_VALID_STATE_KEYS = frozenset(PropertyQAState.__annotations__)


def _extract_node_metadata(
    node: str,
    output: Any,
    duration_ms: int | None = None,
) -> dict:
    """Extract per-node metadata for graph trace SSE events.

    Args:
        node: The LangGraph node name.
        output: The node's output dict.
        duration_ms: Optional per-node execution time in milliseconds
            (provided by the streaming layer's monotonic timing).

    Returns:
        Metadata dict with node-specific fields and optional ``node_latency_ms``.
    """
    if not isinstance(output, dict):
        meta: dict[str, Any] = {}
    elif node in (NODE_COMPLIANCE_GATE, NODE_ROUTER):
        meta = {
            "query_type": output.get("query_type"),
            "confidence": output.get("router_confidence"),
        }
    elif node == NODE_RETRIEVE:
        ctx = output.get("retrieved_context", [])
        meta = {"doc_count": len(ctx)}
    elif node == NODE_GENERATE:
        # R41 fix D1-M002: Surface specialist name in SSE metadata for observability.
        meta = {"specialist": output.get("specialist_name")}
    elif node == NODE_VALIDATE:
        meta = {"result": output.get("validation_result")}
    elif node == NODE_RESPOND:
        meta = {"sources": output.get("sources_used", [])}
    elif node == NODE_WHISPER:
        meta = {"has_plan": bool(output.get("whisper_plan"))}
    else:
        meta = {}

    # R75: Include per-node latency for observability dashboards and
    # performance alerting. Populated by the chat_stream timing layer.
    if duration_ms is not None:
        meta["node_latency_ms"] = duration_ms

    return meta


# ---------------------------------------------------------------------------
# Specialist agent dispatch
# ---------------------------------------------------------------------------

# Maps retrieved-context metadata categories to specialist agent names.
# Categories not listed here route to the "host" (general concierge) agent.
# "spa" -> "entertainment": spa services are managed by the entertainment/amenities
# team at most casino properties; a separate spa agent would duplicate 90% of
# entertainment agent logic for minimal retrieval benefit.
# R50 fix (Grok MAJOR-D1-002): MappingProxyType prevents accidental mutation.
# Plain dict allows `_CATEGORY_TO_AGENT["typo"] = "value"` to corrupt routing
# for all concurrent requests in the same process.
_CATEGORY_TO_AGENT: dict[str, str] = _MappingProxy({
    "restaurants": "dining",
    "entertainment": "entertainment",
    "spa": "entertainment",
    "gaming": "comp",
    "promotions": "comp",
    "hotel": "hotel",
})

# Business-priority tie-break order for specialist dispatch.
# When two categories have equal chunk counts, the higher-priority category
# wins. Dining > hotel > entertainment > comp because dining queries are the
# most common guest request type and have the most actionable content.
_CATEGORY_PRIORITY: dict[str, int] = _MappingProxy({
    "restaurants": 4,
    "hotel": 3,
    "entertainment": 2,
    "spa": 2,
    "gaming": 1,
    "promotions": 1,
})


def _keyword_dispatch(retrieved: list[dict]) -> str:
    """Determine specialist agent name from retrieved context via keyword counting.

    This is the deterministic fallback used when structured LLM dispatch
    is unavailable (circuit breaker open, LLM failure, parsing error).

    Returns:
        Agent name from ``_CATEGORY_TO_AGENT`` or ``"host"`` for unmapped categories.
    """
    category_counts: dict[str, int] = {}
    for chunk in retrieved:
        cat = chunk.get("metadata", {}).get("category", "")
        if cat:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    if not category_counts:
        return "host"

    # Tie-break: business priority (dining > hotel > entertainment > comp),
    # then alphabetical for categories not in _CATEGORY_PRIORITY.
    dominant = max(
        category_counts,
        key=lambda k: (category_counts[k], _CATEGORY_PRIORITY.get(k, 0), k),
    )
    return _CATEGORY_TO_AGENT.get(dominant, "host")


# Dispatch prompt template -- minimal for routing efficiency.
# Uses string.Template for safe substitution (no crash on user braces).
_DISPATCH_PROMPT = _StrTemplate("""\
Route the guest query to the best specialist agent.

Guest query: $query
Retrieved context categories: $categories

Available specialists:
- dining: restaurants, food, bars, dining reservations
- entertainment: shows, events, concerts, comedy, spa, amenities
- comp: player rewards, comps, loyalty programs, gaming promotions
- hotel: rooms, reservations, towers, accommodations
- host: general concierge, mixed topics, or unclear domain""")


async def _route_to_specialist(
    state: PropertyQAState,
    settings: Any,
) -> tuple[str, str]:
    """Determine which specialist agent should handle this query.

    Uses structured LLM dispatch (primary) with deterministic keyword
    fallback. Respects retry reuse and specialist_agents_enabled flag.

    Args:
        state: Current graph state.
        settings: Hoisted settings (avoids TOCTOU from multiple get_settings calls).

    Returns:
        Tuple of (agent_name, dispatch_method).
    """
    retrieved = state.get("retrieved_context", [])
    dispatch_method = "keyword_fallback"

    # R37 fix C-001: On RETRY, reuse the previously-dispatched specialist.
    # This avoids (1) a wasted LLM dispatch call, and (2) non-deterministic
    # specialist switching where the dispatch LLM could route to a different
    # specialist on retry.
    existing_specialist = state.get("specialist_name")
    if existing_specialist and existing_specialist in _AGENT_REGISTRY:
        logger.info(
            "Retry path -- reusing specialist %s (skipping dispatch)",
            existing_specialist,
        )
        return existing_specialist, "retry_reuse"

    agent_name: str | None = None

    # --- Try structured LLM dispatch first ---
    # R15 fix (DeepSeek F-001, GPT F1): acquire CB before try block to prevent
    # UnboundLocalError in except handlers if _get_circuit_breaker() itself raises.
    cb = await _get_circuit_breaker()
    try:
        if await cb.allow_request():
            llm = await _get_llm()
            dispatch_llm = llm.with_structured_output(DispatchOutput)

            query = _get_last_human_message(state.get("messages", []))
            # Extract category names from retrieved context for the prompt
            categories = [
                c.get("metadata", {}).get("category", "")
                for c in retrieved
                if c.get("metadata", {}).get("category", "")
            ]

            # R62 fix D1: Clean comma-separated string instead of str(list)
            prompt = _DISPATCH_PROMPT.safe_substitute(
                query=query,
                categories=", ".join(categories) if categories else "none",
            )

            async with asyncio.timeout(settings.MODEL_TIMEOUT):
                result: DispatchOutput = await dispatch_llm.ainvoke(prompt)

            # R36 fix A1: Record CB success AFTER validating the specialist
            # name is in the registry. Previously, success was recorded before
            # validation -- a valid DispatchOutput with an unknown specialist
            # (e.g., "spa") was counted as healthy even though it fell through
            # to keyword fallback, inflating CB health metrics.
            if result.specialist in _AGENT_REGISTRY:
                await cb.record_success()
                agent_name = result.specialist
                dispatch_method = "structured_output"
                logger.info(
                    "Structured dispatch -> %s (confidence=%.2f, reason=%s)",
                    agent_name,
                    result.confidence,
                    (result.reasoning or "")[:80],
                )
            else:
                logger.warning(
                    "Structured dispatch returned unknown specialist %r, "
                    "falling back to keyword routing",
                    result.specialist,
                )
        else:
            logger.info("Circuit breaker not allowing requests; using keyword fallback")
    except (ValueError, TypeError) as exc:
        # Structured output parsing failed -- LLM returned unparseable JSON.
        # R15 fix (Gemini F7): do NOT record CB failure for parse errors.
        # Parse failure means the LLM IS reachable (network healthy) but
        # returned bad JSON (prompt engineering issue). Recording failures
        # here would conflate parse quality with LLM availability, potentially
        # tripping the CB on prompt issues rather than actual outages.
        # NOTE: google-genai SDK may also raise ValueError for quota/auth
        # errors (e.g., "429 Resource has been exhausted"). These are
        # conflated with parse errors here — a known limitation. The broad
        # `except Exception` block below catches most network/API failures,
        # but SDK-level ValueError wrapping means some availability issues
        # are misclassified as parse errors and do not trip the CB.
        logger.warning("Structured dispatch parsing failed: %s", exc)
    except Exception:
        # Network errors, API failures, timeouts -- broad catch is intentional
        # because google-genai raises various exception types across versions.
        # Record failure so the circuit breaker tracks dispatch LLM health.
        await cb.record_failure()
        logger.warning("Structured dispatch LLM call failed, falling back to keyword counting", exc_info=True)

    # --- Fallback: keyword counting ---
    if agent_name is None:
        agent_name = _keyword_dispatch(retrieved)
        logger.info(
            "Keyword fallback dispatch -> %s (categories=%s)",
            agent_name,
            {c.get("metadata", {}).get("category", "") for c in retrieved if c.get("metadata", {}).get("category")},
        )

    # Feature flag: specialist_agents_enabled controls dispatch to non-host agents.
    # When disabled, all queries route to the general host concierge.
    specialist_enabled = await is_feature_enabled(settings.CASINO_ID, "specialist_agents_enabled")
    if agent_name != "host" and not specialist_enabled:
        logger.info("Specialist agents disabled; routing %s to host", agent_name)
        agent_name = "host"
        # R48 fix: Update dispatch_method when feature flag overrides routing.
        # Previously retained "structured_output" or "keyword_fallback" from
        # original routing, making logs misleading about why host was selected.
        dispatch_method = "feature_flag_override"

    return agent_name, dispatch_method


def _inject_guest_context(
    state: PropertyQAState,
    profile_enabled: bool,
) -> dict[str, Any]:
    """Build guest context update dict from extracted fields.

    Fail-silent: profile lookup failure = empty context, not crash.

    Args:
        state: Current graph state (reads extracted_fields).
        profile_enabled: Whether guest_profile_enabled feature flag is on.

    Returns:
        Dict with guest_context and/or guest_name keys, or empty dict.
    """
    if not profile_enabled:
        return {}
    try:
        from src.data.guest_profile import get_agent_context
        extracted = state.get("extracted_fields", {})
        if not extracted:
            return {}
        agent_ctx = get_agent_context(extracted)
        if not agent_ctx:
            return {}
        # Phase 5: Namespace extracted preferences for structured access.
        from src.data.guest_profile import namespace_preferences

        namespaced = namespace_preferences(extracted)
        if namespaced:
            agent_ctx["namespaced_preferences"] = namespaced
        update: dict[str, Any] = {"guest_context": agent_ctx}
        if agent_ctx.get("name"):
            update["guest_name"] = agent_ctx["name"]
        return update
    except Exception:
        logger.warning("Guest profile lookup failed, continuing without context", exc_info=True)
        return {}


async def _execute_specialist(
    state: PropertyQAState,
    agent_name: str,
    guest_context_update: dict[str, Any],
    settings: Any,
    dispatch_method: str,
) -> dict[str, Any]:
    """Execute the specialist agent with timeout and result sanitization.

    Handles: timeout fallback, dispatch-owned key collision logging,
    unknown state key filtering, specialist name persistence, and
    guest context merging.

    Args:
        state: Current graph state.
        agent_name: Resolved specialist agent name.
        guest_context_update: Guest context dict to merge into agent input and result.
        settings: Hoisted settings.
        dispatch_method: How the specialist was selected (for observability).

    Returns:
        Sanitized result dict ready to be returned as node output.
    """
    try:
        agent_fn = get_agent(agent_name)
    except KeyError:
        # R45 fix D1-M001: Defensive catch for registry lookup failure.
        # _route_to_specialist validates against _AGENT_REGISTRY, but a race
        # between routing and execution (hot reload, registry mutation) could
        # cause a KeyError. Fall back to host agent instead of crashing the graph.
        logger.error(
            "Agent %r not found in registry -- falling back to host agent",
            agent_name,
        )
        agent_fn = get_agent("host")
        agent_name = "host"

    # R34 fix A1: Wrap agent execution in timeout to prevent unbounded execution.
    # Uses 2x MODEL_TIMEOUT because agents may make multiple LLM calls internally
    # (prompt assembly + LLM invoke + retries). Without this, a hung specialist
    # blocks the entire graph indefinitely.
    try:
        async with asyncio.timeout(settings.MODEL_TIMEOUT * 2):
            result = await agent_fn({**state, **guest_context_update})
    except TimeoutError:
        logger.error(
            "Specialist %s timed out after %ds -- returning fallback",
            agent_name,
            settings.MODEL_TIMEOUT * 2,
        )
        result = {
            "messages": [AIMessage(content=(
                "I apologize, but I'm having trouble generating a response right now. "
                "Please try again, or contact us directly for assistance."
            ))],
            "skip_validation": True,
        }
    except Exception:
        # R63 fix D1: Catch unexpected agent errors locally for better diagnostics.
        # Without this, errors propagate to the graph-level handler which returns
        # a generic "error during SSE stream" without specialist context.
        logger.exception(
            "Specialist %s raised unexpected error -- returning fallback",
            agent_name,
        )
        result = {
            "messages": [AIMessage(content=(
                "I apologize, but I'm having trouble generating a response right now. "
                "Please try again, or contact us directly for assistance."
            ))],
            "skip_validation": True,
        }

    # Guard: warn if specialist result contains keys that should only be
    # set by _dispatch_to_specialist (not the specialist agent itself).
    # R33 fix: _DISPATCH_OWNED_KEYS moved to module level (see below call site).
    collisions = _DISPATCH_OWNED_KEYS & set(result.keys())
    if collisions:
        logger.warning(
            "Specialist %s returned dispatch-owned keys %s -- stripping "
            "(specialist should not set these)",
            agent_name, collisions,
        )
        # R47 fix C1: Strip dispatch-owned keys from specialist result.
        # Previously only warned -- specialist values persisted in state,
        # creating TOCTOU where specialist overwrites dispatch-layer context.
        for k in collisions:
            result.pop(k, None)

    # R37 fix M-001: Filter specialist result to known PropertyQAState keys.
    # Prevents unknown keys from polluting state and type mismatches from
    # crashing reducers (e.g., "messages": "not a list" crashes add_messages).
    unknown = set(result.keys()) - _VALID_STATE_KEYS
    if unknown:
        logger.warning(
            "Specialist %s returned unknown state keys %s -- filtering out",
            agent_name, unknown,
        )
        result = {k: v for k, v in result.items() if k in _VALID_STATE_KEYS}

    # R37 fix C-001: Persist specialist name in state for retry path.
    result["specialist_name"] = agent_name

    # R52 fix D1: Persist dispatch method for observability and debugging.
    result["dispatch_method"] = dispatch_method

    # R72 B4: Track which specialist domains have been discussed.
    # _append_unique reducer deduplicates across turns.
    result["domains_discussed"] = [agent_name]

    # Merge guest context updates into the result so they persist in state
    if guest_context_update:
        result.update(guest_context_update)
    return result


async def _dispatch_to_specialist(state: PropertyQAState) -> dict[str, Any]:
    """Route to the best specialist agent and execute it.

    Orchestrates three phases:
    1. **Routing** (_route_to_specialist): Determine which specialist handles
       the query via structured LLM dispatch or keyword fallback.
    2. **Guest context** (_inject_guest_context): Inject guest profile data
       when the feature flag is enabled.
    3. **Execution** (_execute_specialist): Run the specialist with timeout,
       sanitize the result, and merge guest context.

    Feature flag: ``specialist_agents_enabled`` controls whether non-host
    specialists are used. When disabled, all queries route to the general
    host concierge regardless of dispatch method.
    """
    # R35 fix A2: Hoist settings once to eliminate TOCTOU from 5 separate calls.
    # Settings are TTL-cached (1 hour), but if the TTL expires between call 1
    # and call 5, the function operates with mixed configuration values.
    settings = get_settings()

    # Phase 1: Route to specialist
    agent_name, dispatch_method = await _route_to_specialist(state, settings)

    # R38 fix M-002: Cache profile feature flag alongside routing to avoid
    # TOCTOU issues from calling is_feature_enabled() at different times.
    profile_enabled = await is_feature_enabled(settings.CASINO_ID, "guest_profile_enabled")

    # Phase 2: Inject guest profile context
    guest_context_update = _inject_guest_context(state, profile_enabled)

    # R83: Compute model routing decision BEFORE specialist execution.
    # The model_route is recorded in state for observability and passed through
    # to execute_specialist which uses it to select Flash vs Pro LLM.
    model_route = _select_model(state)
    model_name = settings.COMPLEX_MODEL_NAME if model_route == "complex" else settings.MODEL_NAME

    # Phase 3: Execute specialist agent
    logger.info(
        "Dispatching to %s agent (method=%s, model=%s)",
        agent_name,
        dispatch_method,
        model_name,
    )
    result = await _execute_specialist(state, agent_name, guest_context_update, settings, dispatch_method)
    # R83: Record model routing decision in state for observability
    result["model_used"] = model_name
    return result
