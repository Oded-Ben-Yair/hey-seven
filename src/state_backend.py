"""Pluggable state backend for distributed state (rate limiter, CB, idempotency).

Default: InMemoryBackend (single-container, zero-dependency).
Production: RedisBackend (multi-container, requires REDIS_URL).

Switch via STATE_BACKEND env var: "memory" (default) | "redis".
"""

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)


class StateBackend(ABC):
    """Abstract state backend for distributed counters and flags."""

    @abstractmethod
    def increment(self, key: str, ttl: int = 60) -> int: ...

    @abstractmethod
    def get_count(self, key: str) -> int: ...

    @abstractmethod
    def set(self, key: str, value: str, ttl: int = 300) -> None: ...

    @abstractmethod
    def get(self, key: str) -> str | None: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class InMemoryBackend(StateBackend):
    """In-memory state backend. Per-container, suitable for single-instance deployment.

    Includes probabilistic sweep on writes to prevent unbounded memory growth
    from expired-but-unreaped entries (R11 fix: 3/3 reviewer consensus).
    """

    # Max store size to prevent OOM under high key cardinality.
    # When exceeded, a full sweep runs to evict expired entries.
    _MAX_STORE_SIZE = 50_000

    # Probability of running a full sweep on each set()/increment() call.
    # 1/100 means ~1% of writes trigger a sweep — amortized O(1) per write.
    _SWEEP_PROBABILITY = 0.01

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._sweep_counter: int = 0
        # R36 fix B5: threading.Lock protects all read-modify-write ops.
        # InMemoryBackend is called from async coroutines but its operations
        # are synchronous and sub-microsecond, so threading.Lock is appropriate
        # (no awaits inside critical sections). Prevents TOCTOU races in
        # increment() where concurrent coroutines read the same count.
        self._lock = threading.Lock()

    def _cleanup_expired(self, key: str) -> None:
        if key in self._store and self._store[key][1] < time.monotonic():
            del self._store[key]

    # R37 fix beta-C2: Cap sweep batch size to prevent event loop blocking.
    # At 50K entries, iterating all entries under threading.Lock takes 10-50ms,
    # blocking the asyncio event loop (no SSE heartbeats, no health checks).
    # Batching to 1000 entries per sweep keeps lock hold time under 1ms.
    _SWEEP_BATCH_SIZE = 1000

    def _maybe_sweep(self) -> None:
        """Probabilistic sweep: evict expired entries with ~1% probability.

        Also triggers unconditionally when store exceeds _MAX_STORE_SIZE.
        Prevents unbounded memory growth from write-once-never-read keys
        (e.g., transient IP rate-limit windows).

        R37 fix beta-C2: Sweep is batched to _SWEEP_BATCH_SIZE entries per
        tick to prevent the threading.Lock from blocking the asyncio event
        loop. Multiple sweeps across successive writes will eventually
        evict all expired entries (amortized full cleanup).

        R11 fix: DeepSeek F-007, Gemini F-002, GPT F-001 (3/3 consensus).
        """
        self._sweep_counter += 1
        force = len(self._store) >= self._MAX_STORE_SIZE
        if not force and random.random() > self._SWEEP_PROBABILITY:
            return

        now = time.monotonic()
        is_full = len(self._store) >= self._MAX_STORE_SIZE
        expired_keys = []
        for k, (_, expiry) in self._store.items():
            if expiry < now:
                expired_keys.append(k)
            if len(expired_keys) >= self._SWEEP_BATCH_SIZE:
                break

        for k in expired_keys:
            del self._store[k]

        # R44 fix D8-M001: If the store is at capacity and no expired keys
        # were found, evict the oldest entry (FIFO) to prevent a death spiral
        # where every write triggers a full sweep that finds nothing to evict.
        # This converts the worst case from O(BATCH_SIZE)-per-write overhead
        # to bounded LRU behavior.
        if is_full and not expired_keys and self._store:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
            logger.debug(
                "InMemoryBackend sweep: FIFO eviction of oldest entry (store at capacity, 0 expired)"
            )

        if expired_keys:
            logger.debug(
                "InMemoryBackend sweep: evicted %d expired entries (store size: %d, batch limit: %d)",
                len(expired_keys),
                len(self._store),
                self._SWEEP_BATCH_SIZE,
            )

    def increment(self, key: str, ttl: int = 60) -> int:
        with self._lock:
            self._cleanup_expired(key)
            expiry = time.monotonic() + ttl
            current = int(self._store.get(key, (0, 0))[0])
            self._store[key] = (current + 1, expiry)
            self._maybe_sweep()
            return current + 1

    def get_count(self, key: str) -> int:
        with self._lock:
            self._cleanup_expired(key)
            return int(self._store.get(key, (0, 0))[0])

    def set(self, key: str, value: str, ttl: int = 300) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl)
            self._maybe_sweep()

    def get(self, key: str) -> str | None:
        with self._lock:
            self._cleanup_expired(key)
            entry = self._store.get(key)
            return entry[0] if entry else None

    def exists(self, key: str) -> bool:
        with self._lock:
            self._cleanup_expired(key)
            return key in self._store

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)


class RedisBackend(StateBackend):
    """Redis state backend for multi-container deployments.

    Requires REDIS_URL in settings (e.g., redis://10.0.0.1:6379/0).
    """

    def __init__(self, redis_url: str) -> None:
        try:
            import redis

            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            # Log only host info, never full URL (may contain credentials
            # in query params or non-standard formats). R11 fix: GPT F-003.
            conn = self._client.connection_pool.connection_kwargs
            logger.info(
                "Redis state backend connected: host=%s port=%s db=%s ssl=%s",
                conn.get("host", "?"),
                conn.get("port", "?"),
                conn.get("db", "?"),
                conn.get("ssl", False),
            )
        except Exception:
            logger.error("Redis connection failed", exc_info=True)
            raise

    def increment(self, key: str, ttl: int = 60) -> int:
        pipe = self._client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl)
        results = pipe.execute()
        return results[0]

    def get_count(self, key: str) -> int:
        val = self._client.get(key)
        return int(val) if val else 0

    def set(self, key: str, value: str, ttl: int = 300) -> None:
        self._client.setex(key, ttl, value)

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(key))

    def delete(self, key: str) -> None:
        self._client.delete(key)


# R35 CRITICAL fix: Migrate from @lru_cache to TTLCache for credential rotation.
# Redis connection may drop; TTLCache allows periodic reconnection attempt.
# R40 fix D8-C001: TTL jitter to prevent thundering herd on synchronized expiry.
_state_backend_cache: TTLCache = TTLCache(maxsize=1, ttl=3600 + random.randint(0, 300))
_state_backend_lock = threading.Lock()


def get_state_backend() -> StateBackend:
    """Return the configured state backend singleton."""
    cached = _state_backend_cache.get("backend")
    if cached is not None:
        return cached

    with _state_backend_lock:
        # Double-check after acquiring lock
        cached = _state_backend_cache.get("backend")
        if cached is not None:
            return cached

        from src.config import get_settings

        settings = get_settings()
        # R45 fix D8-M002: Use direct attribute access instead of getattr().
        # STATE_BACKEND and REDIS_URL are defined Pydantic fields with defaults.
        # getattr() with fallback masks typos and misconfiguration by silently
        # returning the fallback value instead of raising AttributeError.
        backend_type = settings.STATE_BACKEND
        if backend_type == "redis":
            redis_url = settings.REDIS_URL
            if not redis_url:
                logger.warning("STATE_BACKEND=redis but REDIS_URL not set, falling back to memory")
                backend = InMemoryBackend()
            else:
                backend = RedisBackend(redis_url)
        else:
            backend = InMemoryBackend()
        _state_backend_cache["backend"] = backend
        return backend


# Backward-compat shim: conftest.py uses get_state_backend.cache_clear()
get_state_backend.cache_clear = _state_backend_cache.clear
