"""Per-casino feature flag checking.

Wraps ``get_casino_config()`` to provide clean boolean flag queries.
Default flags ensure safe behavior: new/experimental features are off,
core features are on.

Cache is protected by ``asyncio.Lock`` to prevent thundering herd on
TTL expiry (multiple concurrent coroutines all miss the cache and
issue redundant Firestore reads). R5 fix per DeepSeek F3 analysis.
"""

from __future__ import annotations

import asyncio
import logging
import types
from typing import TypedDict

from cachetools import TTLCache

from src.casino.config import get_casino_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TypedDict for feature flags
# ---------------------------------------------------------------------------


class FeatureFlags(TypedDict, total=False):
    """Known feature flags with their expected types."""

    ai_disclosure_enabled: bool
    whisper_planner_enabled: bool
    specialist_agents_enabled: bool
    comp_agent_enabled: bool
    spanish_support_enabled: bool
    outbound_campaigns_enabled: bool
    hitl_interrupt_enabled: bool
    human_like_delay_enabled: bool
    sms_enabled: bool


# ---------------------------------------------------------------------------
# Default feature flags
# ---------------------------------------------------------------------------

# Trade-off: Graph nodes read DEFAULT_FEATURES directly (synchronous) rather than
# calling the async get_feature_flags(casino_id) API.  This is intentional for
# Phase 1 single-tenant deployment — avoids async overhead in every node.
# The async API (get_feature_flags / is_feature_enabled) is ready for Phase 2
# multi-tenant support when per-casino overrides are needed from Firestore.
DEFAULT_FEATURES: types.MappingProxyType[str, bool] = types.MappingProxyType({
    "ai_disclosure_enabled": True,
    "whisper_planner_enabled": True,
    "specialist_agents_enabled": True,
    "comp_agent_enabled": True,
    "spanish_support_enabled": True,
    "outbound_campaigns_enabled": False,
    "hitl_interrupt_enabled": False,
    "human_like_delay_enabled": True,
    "sms_enabled": False,  # Requires Telnyx setup
})

# Parity assertion: FeatureFlags TypedDict must declare every key in DEFAULT_FEATURES.
# Catches schema drift at import time (same pattern as _initial_state parity in graph.py).
assert set(FeatureFlags.__annotations__) == set(DEFAULT_FEATURES.keys()), (
    f"FeatureFlags TypedDict drift: "
    f"missing={set(DEFAULT_FEATURES.keys()) - set(FeatureFlags.__annotations__)}, "
    f"extra={set(FeatureFlags.__annotations__) - set(DEFAULT_FEATURES.keys())}"
)

# Cross-module parity: DEFAULT_CONFIG["features"] (config.py) must match DEFAULT_FEATURES.
# Prevents drift between the two sources of truth for feature flags.
from src.casino.config import DEFAULT_CONFIG as _DEFAULT_CONFIG  # noqa: E402

assert set(_DEFAULT_CONFIG["features"].keys()) == set(DEFAULT_FEATURES.keys()), (
    f"DEFAULT_CONFIG['features'] drift from DEFAULT_FEATURES: "
    f"missing={set(DEFAULT_FEATURES.keys()) - set(_DEFAULT_CONFIG['features'].keys())}, "
    f"extra={set(_DEFAULT_CONFIG['features'].keys()) - set(DEFAULT_FEATURES.keys())}"
)


# ---------------------------------------------------------------------------
# TTL cache for feature flags (avoids repeated Firestore reads)
# ---------------------------------------------------------------------------

# TTLCache with maxsize: prevents unbounded growth under multi-tenant use.
# asyncio.Lock prevents thundering herd on TTL expiry — only one coroutine
# fetches from Firestore while others wait on the lock. R5 fix.
_FLAG_CACHE_TTL = 300  # 5 minutes
_flag_cache: TTLCache = TTLCache(maxsize=100, ttl=_FLAG_CACHE_TTL)
_flag_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_feature_flags(casino_id: str) -> dict[str, bool]:
    """Get merged feature flags for a casino.

    Merges ``DEFAULT_FEATURES`` with any casino-specific overrides from
    the config document's ``features`` section. Unknown flags in the
    config document are preserved (forward-compatible).

    Results are cached per casino_id with a 5-minute TTL to avoid
    repeated Firestore reads on every graph invocation. Cache access is
    protected by ``asyncio.Lock`` to prevent thundering herd on expiry.

    Args:
        casino_id: Casino identifier (e.g., ``"mohegan_sun"``).

    Returns:
        A dict of flag_name -> bool with all known flags present.
    """
    # Fast path: check cache without lock (TTLCache handles expiry)
    cached = _flag_cache.get(casino_id)
    if cached is not None:
        return cached

    # Slow path: lock to prevent thundering herd on cache miss
    async with _flag_lock:
        # Double-check after acquiring lock (another coroutine may have filled it)
        cached = _flag_cache.get(casino_id)
        if cached is not None:
            return cached

        config = await get_casino_config(casino_id)
        casino_features = config.get("features", {})

        # Defaults first, then casino overrides
        merged = dict(DEFAULT_FEATURES)
        merged.update(casino_features)

        _flag_cache[casino_id] = merged
        return merged


async def is_feature_enabled(casino_id: str, flag_name: str) -> bool:
    """Check whether a single feature flag is enabled.

    Returns ``False`` for unknown flag names (safe default).

    Args:
        casino_id: Casino identifier.
        flag_name: The feature flag to check.

    Returns:
        True if the flag is enabled, False otherwise.
    """
    flags = await get_feature_flags(casino_id)
    return flags.get(flag_name, False)


def get_default_features() -> dict[str, bool]:
    """Return a mutable copy of the default feature flags.

    Returns:
        A new dict with all default feature flag values.
    """
    return dict(DEFAULT_FEATURES)
