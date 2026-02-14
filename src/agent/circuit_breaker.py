"""Circuit breaker for LLM calls.

Protects against cascading failures by tracking consecutive LLM errors
and temporarily blocking requests when a threshold is exceeded.

States: closed (normal) → open (blocking) → half_open (probe one request).
Thread-safe via ``asyncio.Lock`` for concurrent coroutine access.
"""

import asyncio
import logging
import time
from functools import lru_cache

from src.config import get_settings

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """In-memory circuit breaker to protect against cascading LLM failures.

    States: closed (normal) -> open (blocking) -> half_open (probe one request).
    Thread-safe via ``asyncio.Lock`` for concurrent coroutine access.

    Note:
        The ``state`` property reads and may mutate ``_state`` (open → half_open
        transition) **outside** the async lock. This is a documented trade-off:
        acquiring an async lock from a sync property is not possible, and the
        transition is idempotent (worst case: two coroutines both see "half_open"
        and both proceed as probe requests). For the single-worker deployment
        (``--workers 1``) used in this demo, no concurrent mutation is possible.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._last_failure_time: float | None = None
        self._state = "closed"
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        if self._state == "open" and self._last_failure_time is not None:
            if (time.monotonic() - self._last_failure_time) >= self._cooldown_seconds:
                self._state = "half_open"
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    async def record_success(self) -> None:
        async with self._lock:
            self._failure_count = 0
            self._state = "closed"

    async def record_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold:
                self._state = "open"
                logger.warning(
                    "Circuit breaker OPEN after %d consecutive failures (cooldown: %ds)",
                    self._failure_count,
                    self._cooldown_seconds,
                )


@lru_cache(maxsize=1)
def _get_circuit_breaker() -> CircuitBreaker:
    """Get or create the circuit breaker singleton (lazy, cached).

    Uses ``@lru_cache`` consistent with ``_get_llm()`` / ``get_settings()``
    pattern. Lazy initialization avoids reading settings at import time.
    """
    settings = get_settings()
    return CircuitBreaker(
        failure_threshold=settings.CB_FAILURE_THRESHOLD,
        cooldown_seconds=settings.CB_COOLDOWN_SECONDS,
    )
