"""Checkpointer factory for conversation state persistence.

Returns ``MemorySaver`` (wrapped in ``BoundedMemorySaver``) for local
development or ``FirestoreSaver`` for production (``VECTOR_DB=firestore``).

TTL-cached with ``asyncio.Lock`` consistent with ``_get_llm()`` and
``_get_validator_llm()`` patterns for GCP credential rotation support.

Production note: ``MemorySaver`` is process-scoped. In single-container
Cloud Run, conversation state lives for the container lifetime. Multi-
container or scale-to-zero deployments MUST use ``FirestoreSaver`` for
durable cross-instance persistence.

ADR: Checkpointer Choice (R39 fix D9-M002)
-------------------------------------------
Decision: MemorySaver (dev) / FirestoreSaver (prod)
Rationale:
- **MemorySaver**: Zero external dependencies for local dev. Bounded via
  BoundedMemorySaver (LRU eviction at MAX_ACTIVE_THREADS=1000) to prevent
  OOM. ~50 MB max at 50 KB/thread.
- **FirestoreSaver**: Community package (langgraph-checkpoint-firestore).
  Firestore max document size = 1 MB, adequate for ~40-message conversations
  (~25 KB typical). Cost: ~$0.006/100 reads + $0.018/100 writes (Firestore
  Native pricing). At 1000 conversations/day with 10 turns avg = $0.24/day.
- **Why not Redis?**: Firestore is GCP-native (no extra infra), has built-in
  TTL for auto-cleanup, and integrates with GCP IAM. Redis (Cloud Memorystore)
  adds ~$30/mo for basic tier with no persistence guarantees on eviction.
- **Latency**: FirestoreSaver adds ~10-20ms per checkpoint read/write.
  Acceptable for conversational latency where LLM calls dominate (2-15s).
"""

import asyncio
import collections
import logging
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Max threads for BoundedMemorySaver (development guard against OOM).
# Each thread stores ~10-50 KB of state (messages, context, metadata).
# At 1000 threads x 50 KB = ~50 MB — safe for Cloud Run 512 MB containers.
MAX_ACTIVE_THREADS = 1000

# TTL-cached checkpointer: refreshes every hour for credential rotation
# (FirestoreSaver uses GCP credentials that rotate under Workload Identity
# Federation). Consistent with _get_llm() / _get_validator_llm() in nodes.py.
# R40 fix D8-C001: TTL jitter to prevent thundering herd on synchronized expiry.
import random as _random

_checkpointer_cache: TTLCache = TTLCache(maxsize=1, ttl=3600 + _random.randint(0, 300))
_checkpointer_lock = asyncio.Lock()


class BoundedMemorySaver:
    """MemorySaver wrapper with LRU eviction for development use.

    Wraps ``MemorySaver`` to prevent unbounded memory growth. When the
    number of tracked threads exceeds ``max_threads``, the oldest thread
    is evicted (LRU order). This prevents OOM kills during extended
    development sessions or demo deployments.

    NOT for production — use ``FirestoreSaver`` for durable persistence.
    """

    def __init__(self, max_threads: int = MAX_ACTIVE_THREADS) -> None:
        from langgraph.checkpoint.memory import MemorySaver

        self._inner = MemorySaver()
        self._max_threads = max_threads
        # Track thread access order for LRU eviction
        self._thread_order: collections.OrderedDict[str, bool] = (
            collections.OrderedDict()
        )

    def _track_thread(self, config: dict) -> None:
        """Track thread_id access for LRU eviction."""
        thread_id = config.get("configurable", {}).get("thread_id", "")
        if not thread_id:
            return
        # Move to end (most recently used)
        if thread_id in self._thread_order:
            self._thread_order.move_to_end(thread_id)
        else:
            self._thread_order[thread_id] = True
        # Evict LRU if over capacity
        while len(self._thread_order) > self._max_threads:
            evicted_id, _ = self._thread_order.popitem(last=False)
            # Remove from inner MemorySaver storage if accessible
            if hasattr(self._inner, "storage"):
                keys_to_remove = [
                    k
                    for k in self._inner.storage
                    if isinstance(k, tuple) and len(k) > 0 and k[0] == evicted_id
                ]
                for k in keys_to_remove:
                    del self._inner.storage[k]
            logger.debug(
                "Evicted thread %s (LRU, capacity=%d)", evicted_id, self._max_threads
            )

    # Delegate all checkpointer protocol methods to inner MemorySaver
    async def aget(self, config: dict) -> Any:
        self._track_thread(config)
        return await self._inner.aget(config)

    async def aput(
        self, config: dict, checkpoint: Any, metadata: Any, new_versions: Any
    ) -> Any:
        self._track_thread(config)
        return await self._inner.aput(config, checkpoint, metadata, new_versions)

    async def alist(self, config: dict, **kwargs: Any) -> Any:
        return await self._inner.alist(config, **kwargs)

    async def aput_writes(self, config: dict, writes: list, task_id: str) -> None:
        self._track_thread(config)
        return await self._inner.aput_writes(config, writes, task_id)

    def get(self, config: dict) -> Any:
        self._track_thread(config)
        return self._inner.get(config)

    def put(
        self, config: dict, checkpoint: Any, metadata: Any, new_versions: Any
    ) -> Any:
        self._track_thread(config)
        return self._inner.put(config, checkpoint, metadata, new_versions)

    def list(self, config: dict, **kwargs: Any) -> Any:
        return self._inner.list(config, **kwargs)

    def put_writes(self, config: dict, writes: list, task_id: str) -> None:
        self._track_thread(config)
        return self._inner.put_writes(config, writes, task_id)

    def get_tuple(self, config: dict) -> Any:
        self._track_thread(config)
        return self._inner.get_tuple(config)

    async def aget_tuple(self, config: dict) -> Any:
        self._track_thread(config)
        return await self._inner.aget_tuple(config)

    @property
    def active_threads(self) -> int:
        """Number of tracked threads (for monitoring/health checks)."""
        return len(self._thread_order)


async def get_checkpointer() -> Any:
    """Get or create the checkpointer singleton (TTL-cached, async).

    Returns ``FirestoreSaver`` when ``VECTOR_DB=firestore`` (production),
    otherwise ``BoundedMemorySaver`` (local development).

    TTL cache refreshes every hour to support GCP credential rotation
    (Workload Identity Federation). Consistent with ``_get_llm()`` /
    ``_get_validator_llm()`` patterns in ``nodes.py``.

    Returns:
        A LangGraph checkpointer instance.
    """
    async with _checkpointer_lock:
        cached = _checkpointer_cache.get("cp")
        if cached is not None:
            return cached

        from src.config import get_settings

        settings = get_settings()

        if settings.VECTOR_DB == "firestore":
            try:
                from langgraph.checkpoint.firestore import FirestoreSaver

                checkpointer = FirestoreSaver(project=settings.FIRESTORE_PROJECT)
                logger.info(
                    "Using FirestoreSaver checkpointer (project=%s)",
                    settings.FIRESTORE_PROJECT,
                )
                _checkpointer_cache["cp"] = checkpointer
                return checkpointer
            except ImportError:
                # R107: Fail-hard in production — MemorySaver loses state on restart.
                if settings.ENVIRONMENT == "production":
                    raise RuntimeError(
                        "VECTOR_DB=firestore but langgraph-checkpoint-firestore not installed. "
                        "Install it or change VECTOR_DB for production."
                    )
                logger.warning(
                    "langgraph-checkpoint-firestore not installed. "
                    "Falling back to BoundedMemorySaver."
                )
            except Exception:
                if settings.ENVIRONMENT == "production":
                    raise
                logger.exception(
                    "Failed to create FirestoreSaver. Falling back to BoundedMemorySaver."
                )

        if settings.ENVIRONMENT == "production":
            logger.warning(
                "ENVIRONMENT=production but using BoundedMemorySaver (in-memory). "
                "Conversation state will be lost on container recycle and is not "
                "shared across Cloud Run instances. Set VECTOR_DB=firestore for "
                "durable checkpointing."
            )

        checkpointer = BoundedMemorySaver(max_threads=MAX_ACTIVE_THREADS)
        logger.info(
            "Using BoundedMemorySaver checkpointer (in-memory, max_threads=%d).",
            MAX_ACTIVE_THREADS,
        )
        _checkpointer_cache["cp"] = checkpointer
        return checkpointer


def clear_checkpointer_cache() -> None:
    """Clear the checkpointer cache (for testing and credential rotation)."""
    _checkpointer_cache.clear()
