"""Pre-extraction node for deterministic field extraction.

Runs regex-based extraction (sub-1ms) and optional LLM augmentation on the
latest user message BEFORE specialist dispatch.  Results merge into
``extracted_fields`` via the ``_merge_dicts`` reducer, making them available
to the specialist on the SAME turn (eliminates one-turn profiling lag).

Extracted from ``router_node`` (R87 SRP refactor) so the router focuses
solely on LLM classification.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.casino.feature_flags import is_feature_enabled
from src.config import get_settings

from .extraction import extract_fields
from .state import PropertyQAState

logger = logging.getLogger(__name__)


def _get_last_human_message(messages: list) -> str:
    """Return content of the last HumanMessage, or empty string."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, list):
                # Gemini 3.x returns list[dict] content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
                return ""
            return str(content) if content else ""
    return ""


async def pre_extract_node(state: PropertyQAState) -> dict[str, Any]:
    """Deterministic field extraction before specialist dispatch.

    Runs regex extraction (sub-1ms) and optional LLM augmentation
    on the latest user message.  Results merge into extracted_fields
    via _merge_dicts reducer, available to the specialist in generate.

    Feature flags:
        - field_extraction_enabled: enables regex extraction
        - extraction_llm_augmented: enables LLM fallback for 15+ word messages
    """
    messages = state.get("messages", [])
    user_message = _get_last_human_message(messages)

    if not user_message:
        return {}

    settings = get_settings()
    extraction_update: dict[str, Any] = {}

    # Phase 1: Deterministic field extraction (sub-1ms regex, no LLM).
    if await is_feature_enabled(settings.CASINO_ID, "field_extraction_enabled"):
        extracted = extract_fields(user_message)
        if extracted:
            extraction_update["extracted_fields"] = extracted
            if extracted.get("name"):
                extraction_update["guest_name"] = extracted["name"]

    # Phase 2: LLM fallback for regex extraction misses (15+ word messages).
    if (
        not extraction_update.get("extracted_fields")
        and len(user_message.split()) >= 15
        and await is_feature_enabled(settings.CASINO_ID, "extraction_llm_augmented")
    ):
        from .extraction import extract_fields_augmented
        from .nodes import _get_llm

        augmented = await extract_fields_augmented(
            user_message, state.get("extracted_fields", {}), _get_llm
        )
        if augmented:
            extraction_update["extracted_fields"] = augmented
            if augmented.get("name"):
                extraction_update["guest_name"] = augmented["name"]

    return extraction_update
