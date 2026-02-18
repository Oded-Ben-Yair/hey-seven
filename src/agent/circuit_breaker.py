"""Circuit breaker for LLM calls.

Protects against cascading failures by tracking LLM errors within a
rolling time window and temporarily blocking requests when a threshold
is exceeded.

States: closed (normal) -> open (blocking) -> half_open (probe one request).
Thread-safe via ``asyncio.Lock`` for concurrent coroutine access.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import lru_cache

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for the circuit breaker.

    Frozen dataclass so it can be used as a dict key or set member if needed.

    Attributes:
        failure_threshold: Number of failures within the rolling window to trip.
        cooldown_seconds: Seconds to wait in open state before probing.
        rolling_window_seconds: Time window for counting failures (seconds).
    """

    failure_threshold: int = 5
    cooldown_seconds: float = 60.0
    rolling_window_seconds: float = 300.0  # 5 min window


class CircuitBreaker:
    """In-memory circuit breaker to protect against cascading LLM failures.

    States: closed (normal) -> open (blocking) -> half_open (probe one request).
    Thread-safe via ``asyncio.Lock`` for concurrent coroutine access.

    Uses a rolling time window for failure counting: only failures within the
    last ``rolling_window_seconds`` count toward the threshold. Old failures
    are pruned automatically.

    In ``half_open`` state, exactly one probe request is allowed through.
    If it succeeds, the breaker closes. If it fails, the breaker re-opens.

    Note:
        The ``state`` property reads and may mutate ``_state`` (open -> half_open
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
        rolling_window_seconds: float = 300.0,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        if config is not None:
            self._failure_threshold = config.failure_threshold
            self._cooldown_seconds = config.cooldown_seconds
            self._rolling_window_seconds = config.rolling_window_seconds
        else:
            self._failure_threshold = failure_threshold
            self._cooldown_seconds = cooldown_seconds
            self._rolling_window_seconds = rolling_window_seconds

        self._failure_timestamps: list[float] = []
        self._last_failure_time: float | None = None
        self._state = "closed"
        self._half_open_in_progress = False
        self._lock = asyncio.Lock()

    def _prune_old_failures(self) -> None:
        """Remove failure timestamps outside the rolling window."""
        cutoff = time.monotonic() - self._rolling_window_seconds
        self._failure_timestamps = [
            ts for ts in self._failure_timestamps if ts > cutoff
        ]

    @property
    def failure_count(self) -> int:
        """Number of failures within the rolling window."""
        self._prune_old_failures()
        return len(self._failure_timestamps)

    @property
    def _failure_count(self) -> int:
        """Alias for failure_count (backward compatibility with existing tests)."""
        return self.failure_count

    @property
    def state(self) -> str:
        if self._state == "open" and self._last_failure_time is not None:
            if (time.monotonic() - self._last_failure_time) >= self._cooldown_seconds:
                self._state = "half_open"
                self._half_open_in_progress = False
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    @property
    def is_half_open(self) -> bool:
        return self.state == "half_open"

    async def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        In ``half_open`` state, only one probe request is allowed at a time.

        Returns:
            True if the request is allowed, False if blocked.
        """
        async with self._lock:
            current_state = self.state
            if current_state == "closed":
                return True
            if current_state == "open":
                return False
            # half_open: allow exactly one probe
            if current_state == "half_open" and not self._half_open_in_progress:
                self._half_open_in_progress = True
                return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            self._failure_timestamps.clear()
            self._state = "closed"
            self._half_open_in_progress = False

    async def record_failure(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._failure_timestamps.append(now)
            self._last_failure_time = now
            self._prune_old_failures()

            if self._state == "half_open":
                # Probe failed: re-open
                self._state = "open"
                self._half_open_in_progress = False
                logger.warning(
                    "Circuit breaker re-OPENED after failed half_open probe (cooldown: %ds)",
                    self._cooldown_seconds,
                )
            elif len(self._failure_timestamps) >= self._failure_threshold:
                self._state = "open"
                logger.warning(
                    "Circuit breaker OPEN after %d failures in %.0fs window (cooldown: %ds)",
                    len(self._failure_timestamps),
                    self._rolling_window_seconds,
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
