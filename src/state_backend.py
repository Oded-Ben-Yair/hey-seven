"""Pluggable state backend for distributed state (rate limiter, CB, idempotency).

Default: InMemoryBackend (single-container, zero-dependency).
Production: RedisBackend (multi-container, requires REDIS_URL).

Switch via STATE_BACKEND env var: "memory" (default) | "redis".
"""

import logging
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

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
    """In-memory state backend. Per-container, suitable for single-instance deployment."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}

    def _cleanup_expired(self, key: str) -> None:
        if key in self._store and self._store[key][1] < time.monotonic():
            del self._store[key]

    def increment(self, key: str, ttl: int = 60) -> int:
        self._cleanup_expired(key)
        expiry = time.monotonic() + ttl
        current = int(self._store.get(key, (0, 0))[0])
        self._store[key] = (current + 1, expiry)
        return current + 1

    def get_count(self, key: str) -> int:
        self._cleanup_expired(key)
        return int(self._store.get(key, (0, 0))[0])

    def set(self, key: str, value: str, ttl: int = 300) -> None:
        self._store[key] = (value, time.monotonic() + ttl)

    def get(self, key: str) -> str | None:
        self._cleanup_expired(key)
        entry = self._store.get(key)
        return entry[0] if entry else None

    def exists(self, key: str) -> bool:
        self._cleanup_expired(key)
        return key in self._store

    def delete(self, key: str) -> None:
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
            logger.info("Redis state backend connected: %s", redis_url.split("@")[-1])
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


@lru_cache(maxsize=1)
def get_state_backend() -> StateBackend:
    """Return the configured state backend singleton."""
    from src.config import get_settings

    settings = get_settings()
    backend_type = getattr(settings, "STATE_BACKEND", "memory")
    if backend_type == "redis":
        redis_url = getattr(settings, "REDIS_URL", "")
        if not redis_url:
            logger.warning("STATE_BACKEND=redis but REDIS_URL not set, falling back to memory")
            return InMemoryBackend()
        return RedisBackend(redis_url)
    return InMemoryBackend()
