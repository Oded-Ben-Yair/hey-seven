"""LangFuse observability client for Hey Seven.

Provides a LangChain callback handler that produces trace hierarchies:
- Trace = one graph execution (user message -> response)
- Span = individual node (router, retrieve, generate, validate, etc.)
- Generation = LLM call details (tokens, cost, latency)

Disabled in test/dev environments (ENVIRONMENT != 'production') or when
LANGFUSE_PUBLIC_KEY is not set.
"""

import logging
import random
import threading
from typing import Any

from cachetools import TTLCache

from src.config import get_settings

logger = logging.getLogger(__name__)

# Sampling rate: 10% in production, 100% in development
_PRODUCTION_SAMPLE_RATE = 0.10
_DEV_SAMPLE_RATE = 1.0

# R35 CRITICAL fix: Migrate from @lru_cache to TTLCache for credential rotation.
# LangFuse credentials may rotate (GCP Workload Identity, secret manager rotation).
# @lru_cache never expires — requires container restart. TTLCache gives 1-hour refresh.
# R46 fix D8: Add TTL jitter to prevent thundering herd (parity with other singletons).
_langfuse_cache: TTLCache = TTLCache(maxsize=1, ttl=3600 + random.randint(0, 300))
_langfuse_lock = threading.Lock()


def _get_langfuse_client() -> Any | None:
    """Initialize LangFuse client if configured.

    Lazy-imports langfuse to avoid import failures when
    the package is not installed. Uses TTLCache (1-hour TTL)
    for credential rotation support.

    Returns:
        Langfuse client instance or None if not configured.
    """
    cached = _langfuse_cache.get("client")
    if cached is not None:
        return cached

    with _langfuse_lock:
        # Double-check after acquiring lock
        cached = _langfuse_cache.get("client")
        if cached is not None:
            return cached

        settings = get_settings()

        if not settings.LANGFUSE_PUBLIC_KEY:
            logger.debug("LangFuse not configured (no LANGFUSE_PUBLIC_KEY)")
            return None

        try:
            from langfuse import Langfuse  # noqa: E402

            client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY.get_secret_value(),
                host=settings.LANGFUSE_HOST,
            )
            logger.info("LangFuse client initialized (host=%s)", settings.LANGFUSE_HOST)
            _langfuse_cache["client"] = client
            return client
        except ImportError:
            logger.debug("langfuse package not installed; observability disabled")
            return None
        except Exception:
            logger.warning("LangFuse client init failed; observability disabled", exc_info=True)
            return None


def is_observability_enabled() -> bool:
    """Check if observability is enabled and configured."""
    return _get_langfuse_client() is not None


def should_sample() -> bool:
    """Determine if this request should be traced based on sampling rate.

    In production: 10% sampling rate.
    In development: 100% (trace everything).
    Always trace error paths (caller's responsibility).
    """
    settings = get_settings()
    rate = _PRODUCTION_SAMPLE_RATE if settings.ENVIRONMENT == "production" else _DEV_SAMPLE_RATE
    return random.random() < rate  # noqa: S311


def get_langfuse_handler(
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    """Get a LangChain callback handler for LangFuse tracing.

    Returns None if LangFuse is not configured or sampling says skip.
    The caller can safely pass None to LangChain's callbacks list.

    Args:
        trace_id: Unique trace ID (defaults to request_id).
        session_id: Session ID for grouping traces (e.g., thread_id).
        user_id: User identifier (e.g., guest phone hash).
        request_id: HTTP X-Request-ID for correlating graph traces with
            HTTP request logs.  Stored in trace metadata.
        tags: List of tags (e.g., ["property_qa", "dining"]).
        metadata: Additional metadata dict.

    Returns:
        CallbackHandler instance or None.
    """
    client = _get_langfuse_client()
    if client is None:
        return None

    if not should_sample():
        return None

    # Merge request_id into metadata for end-to-end correlation
    merged_metadata = dict(metadata or {})
    if request_id:
        merged_metadata["request_id"] = request_id

    try:
        from langfuse.callback import CallbackHandler  # noqa: E402

        handler = CallbackHandler(
            public_key=get_settings().LANGFUSE_PUBLIC_KEY,
            secret_key=get_settings().LANGFUSE_SECRET_KEY.get_secret_value(),
            host=get_settings().LANGFUSE_HOST,
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
            metadata=merged_metadata,
        )
        return handler
    except ImportError:
        return None
    except Exception:
        logger.warning("Failed to create LangFuse callback handler", exc_info=True)
        return None


def clear_langfuse_cache() -> None:
    """Clear the LangFuse client cache (for testing)."""
    _langfuse_cache.clear()


# Backward-compat shim: conftest.py and other callers use
# _get_langfuse_client.cache_clear() — provide the same interface.
_get_langfuse_client.cache_clear = _langfuse_cache.clear
