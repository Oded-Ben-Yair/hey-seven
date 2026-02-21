"""Per-casino configuration with Firestore and in-memory cache.

Config documents live at Firestore path: ``config/{casino_id}``.
Cache has 5-minute TTL per casino_id for hot-reload without redeployment.

When Firestore is unavailable (local dev, tests), ``DEFAULT_CONFIG`` is
returned for every casino_id, ensuring the same async API works
everywhere without mocking infrastructure.

Cache is protected by ``asyncio.Lock`` to prevent thundering herd on
TTL expiry. R5 fix per DeepSeek F4 analysis.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TypedDict

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache: casino_id -> config_dict (TTLCache handles expiry and maxsize)
# ---------------------------------------------------------------------------

_CONFIG_TTL_SECONDS = 300  # 5 minutes
_config_cache: TTLCache = TTLCache(maxsize=100, ttl=_CONFIG_TTL_SECONDS)
_config_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# TypedDict config sections
# ---------------------------------------------------------------------------


class BrandingConfig(TypedDict, total=False):
    """Branding configuration per casino."""

    persona_name: str  # default "Seven"
    tone: str  # default "warm_professional"
    formality_level: str  # default "casual_respectful"
    emoji_allowed: bool  # default False
    exclamation_limit: int  # default 1


class RegulationConfig(TypedDict, total=False):
    """Regulatory configuration per casino."""

    state: str  # e.g., "CT"
    gaming_age_minimum: int  # default 21
    ai_disclosure_required: bool  # default True
    ai_disclosure_law: str
    quiet_hours_start: str  # "21:00"
    quiet_hours_end: str  # "08:00"
    responsible_gaming_helpline: str
    state_helpline: str


class OperationalConfig(TypedDict, total=False):
    """Operational configuration per casino."""

    timezone: str  # default "America/New_York"
    telnyx_phone_number: str
    escalation_slack_channel: str
    escalation_sms_number: str
    contact_phone: str
    max_messages_per_guest_per_day: int  # default 20
    session_timeout_hours: int  # default 48


class RagConfig(TypedDict, total=False):
    """RAG configuration per casino."""

    min_relevance_score: float  # default 0.35
    top_k: int  # default 5
    embedding_model: str  # default "gemini-embedding-001"
    embedding_dimensions: int  # default 768


class PromptsConfig(TypedDict, total=False):
    """Prompt overrides per casino."""

    system_prompt_override: str | None
    greeting_template: str
    greeting_template_es: str
    fallback_message: str
    casino_name_display: str


class CasinoConfig(TypedDict, total=False):
    """Full per-casino configuration document."""

    _id: str
    _version: int
    _updated_at: str
    features: dict[str, bool]
    prompts: PromptsConfig
    branding: BrandingConfig
    regulations: RegulationConfig
    operational: OperationalConfig
    rag: RagConfig


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "_id": "default",
    "_version": 1,
    "_updated_at": "",
    "features": {
        "ai_disclosure_enabled": True,
        "whisper_planner_enabled": True,
        "specialist_agents_enabled": True,
        "comp_agent_enabled": True,
        "spanish_support_enabled": True,
        "outbound_campaigns_enabled": False,
        "hitl_interrupt_enabled": False,
        "human_like_delay_enabled": True,
        "sms_enabled": False,
    },
    "prompts": {
        "system_prompt_override": None,
        "greeting_template": "Hi $guest_name, this is $persona_name from $casino_name! How can I help you today?",
        "greeting_template_es": "Hola $guest_name, soy $persona_name de $casino_name. Como puedo ayudarte hoy?",
        "fallback_message": "I want to make sure I get you the right answer. Let me connect you with someone who can help.",
        "casino_name_display": "the resort",
    },
    "branding": {
        "persona_name": "Seven",
        "tone": "warm_professional",
        "formality_level": "casual_respectful",
        "emoji_allowed": False,
        "exclamation_limit": 1,
    },
    "regulations": {
        "state": "CT",
        "gaming_age_minimum": 21,
        "ai_disclosure_required": True,
        "ai_disclosure_law": "",
        "quiet_hours_start": "21:00",
        "quiet_hours_end": "08:00",
        "responsible_gaming_helpline": "1-800-522-4700",
        "state_helpline": "",
    },
    "operational": {
        "timezone": "America/New_York",
        "telnyx_phone_number": "",
        "escalation_slack_channel": "",
        "escalation_sms_number": "",
        "contact_phone": "",
        "max_messages_per_guest_per_day": 20,
        "session_timeout_hours": 48,
    },
    "rag": {
        "min_relevance_score": 0.35,
        "top_k": 5,
        "embedding_model": "gemini-embedding-001",
        "embedding_dimensions": 768,
    },
}


# ---------------------------------------------------------------------------
# Firestore client accessor
# ---------------------------------------------------------------------------


_fs_config_client_cache: dict[str, Any] = {}
_fs_config_client_lock = asyncio.Lock()


async def _get_firestore_client() -> Any | None:
    """Return the cached Firestore AsyncClient if available, else None.

    Lazy-imports ``google.cloud.firestore`` to avoid import failures when
    the dependency is not installed (local dev without GCP SDK).

    Protected by ``asyncio.Lock`` to prevent race conditions during
    concurrent cold-start requests.  R16: converted from ``threading.Lock``
    (which blocks the event loop) to ``asyncio.Lock`` (non-blocking).

    Note: A near-identical helper exists in ``src.data.guest_profile``.
    Both are intentionally kept separate: this one uses ``CASINO_ID`` as
    the database parameter (per-casino config isolation), while the guest
    profile variant uses the same pattern for guest document storage.
    Extracting a shared utility would save ~10 LOC but add an import
    dependency between unrelated modules.
    """
    # Fast path outside lock (safe: dict.get is atomic under GIL,
    # and we only ever SET the value under the lock).
    cached = _fs_config_client_cache.get("client")
    if cached is not None:
        return cached

    async with _fs_config_client_lock:
        # Double-check after acquiring lock
        cached = _fs_config_client_cache.get("client")
        if cached is not None:
            return cached

        try:
            from google.cloud.firestore import AsyncClient  # noqa: F401

            from src.config import get_settings

            settings = get_settings()
            if settings.VECTOR_DB == "firestore" and settings.FIRESTORE_PROJECT:
                client = AsyncClient(
                    project=settings.FIRESTORE_PROJECT,
                    database=settings.CASINO_ID,
                )
                _fs_config_client_cache["client"] = client
                return client
        except ImportError:
            logger.debug("google-cloud-firestore not installed; using defaults")
        except Exception:
            logger.warning(
                "Firestore client init failed; falling back to defaults",
                exc_info=True,
            )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into base, returning a new dict.

    Only merges nested dicts; non-dict values in overrides replace base values.
    """
    merged = dict(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


async def get_casino_config(casino_id: str) -> dict:
    """Load per-casino configuration with 5-minute TTL cache.

    Resolution order:
    1. Return from cache if TTL has not expired.
    2. Attempt Firestore read from ``config/{casino_id}``.
    3. Fall back to ``DEFAULT_CONFIG`` if Firestore unavailable.

    The returned config is always a deep merge of ``DEFAULT_CONFIG`` with
    any casino-specific overrides from Firestore, so callers can rely on
    every key being present.

    Cache access is protected by ``asyncio.Lock`` to prevent thundering
    herd on TTL expiry (R5 fix per DeepSeek F4).

    Args:
        casino_id: Casino identifier (e.g., ``"mohegan_sun"``).

    Returns:
        A config dict with all sections populated.
    """
    # Fast path: check cache without lock (TTLCache handles expiry)
    cached = _config_cache.get(casino_id)
    if cached is not None:
        return cached

    # Slow path: lock to prevent thundering herd on cache miss
    async with _config_lock:
        # Double-check after acquiring lock
        cached = _config_cache.get(casino_id)
        if cached is not None:
            return cached

        # Try Firestore
        db = await _get_firestore_client()
        if db is not None:
            try:
                doc_ref = db.collection("config").document(casino_id)
                doc = await doc_ref.get()
                if doc.exists:
                    overrides = doc.to_dict()
                    config = _deep_merge(DEFAULT_CONFIG, overrides)
                    config["_id"] = casino_id
                    _config_cache[casino_id] = config
                    logger.info("Loaded config for casino %s from Firestore", casino_id)
                    return config
                logger.debug(
                    "No config document for casino %s; using defaults", casino_id
                )
            except Exception:
                logger.warning(
                    "Firestore config read failed for %s; using defaults",
                    casino_id,
                    exc_info=True,
                )

        # Fallback to defaults
        import copy

        config = copy.deepcopy(DEFAULT_CONFIG)
        config["_id"] = casino_id
        _config_cache[casino_id] = config
        return config


def clear_config_cache() -> None:
    """Clear the config cache and Firestore client cache (for testing)."""
    _config_cache.clear()
    _fs_config_client_cache.clear()
