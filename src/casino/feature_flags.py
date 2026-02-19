"""Per-casino feature flag checking.

Wraps ``get_casino_config()`` to provide clean boolean flag queries.
Default flags ensure safe behavior: new/experimental features are off,
core features are on.
"""

from __future__ import annotations

import logging
import time
import types
from typing import TypedDict

from src.casino.config import get_casino_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TypedDict for feature flags
# ---------------------------------------------------------------------------


class FeatureFlags(TypedDict, total=False):
    """Known feature flags with their expected types."""

    ai_disclosure_enabled: bool
    whisper_planner_enabled: bool
    comp_agent_enabled: bool
    spanish_support_enabled: bool
    outbound_campaigns_enabled: bool
    hitl_interrupt_enabled: bool
    human_like_delay_enabled: bool
    sms_enabled: bool


# ---------------------------------------------------------------------------
# Default feature flags
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TTL cache for feature flags (avoids repeated Firestore reads)
# ---------------------------------------------------------------------------

# Simple TTL cache for feature flags (avoids repeated Firestore reads).
# 5-minute TTL matches the casino config cache in config.py.
_flag_cache: dict[str, tuple[dict[str, bool], float]] = {}
_FLAG_CACHE_TTL = 300.0  # 5 minutes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_feature_flags(casino_id: str) -> dict[str, bool]:
    """Get merged feature flags for a casino.

    Merges ``DEFAULT_FEATURES`` with any casino-specific overrides from
    the config document's ``features`` section. Unknown flags in the
    config document are preserved (forward-compatible).

    Results are cached per casino_id with a 5-minute TTL to avoid
    repeated Firestore reads on every graph invocation.

    Args:
        casino_id: Casino identifier (e.g., ``"mohegan_sun"``).

    Returns:
        A dict of flag_name -> bool with all known flags present.
    """
    now = time.monotonic()
    cached = _flag_cache.get(casino_id)
    if cached is not None:
        flags, expires_at = cached
        if now < expires_at:
            return flags

    config = await get_casino_config(casino_id)
    casino_features = config.get("features", {})

    # Defaults first, then casino overrides
    merged = dict(DEFAULT_FEATURES)
    merged.update(casino_features)

    _flag_cache[casino_id] = (merged, now + _FLAG_CACHE_TTL)
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
