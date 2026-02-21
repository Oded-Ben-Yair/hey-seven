"""Circuit breaker for LLM calls.

Protects against cascading failures by tracking LLM errors within a
rolling time window and temporarily blocking requests when a threshold
is exceeded.

States: closed (normal) -> open (blocking) -> half_open (probe one request).
Thread-safe via ``asyncio.Lock`` for concurrent coroutine access.
"""

import asyncio
import collections
import logging
import time

from cachetools import TTLCache

from src.config import get_settings

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """In-memory circuit breaker to protect against cascading LLM failures.

    States: closed (normal) -> open (blocking) -> half_open (probe one request).
    Thread-safe via ``asyncio.Lock`` for concurrent coroutine access.
    Failure timestamps stored in a bounded ``collections.deque`` to prevent
    unbounded memory growth under sustained failure conditions.

    Uses a rolling time window for failure counting: only failures within the
    last ``rolling_window_seconds`` count toward the threshold. Old failures
    are pruned automatically.

    In ``half_open`` state, exactly one probe request is allowed through.
    If it succeeds, the breaker closes. If it fails, the breaker re-opens.

    The ``state`` property is **read-only** — it never mutates ``_state``.
    The open -> half_open transition happens inside ``allow_request()`` under
    the async lock, ensuring no concurrent race on state transitions.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        rolling_window_seconds: float = 300.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._rolling_window_seconds = rolling_window_seconds

        # No maxlen: memory is bounded by _prune_old_failures() which removes
        # timestamps outside the rolling window. The natural bound is
        # failure_rate * rolling_window_seconds. Previous maxlen caused silent
        # undercounting when failures arrived faster than maxlen / window_seconds,
        # preventing the breaker from tripping under sustained moderate load.
        # R5 fix: removed maxlen per DeepSeek F1 analysis.
        self._failure_timestamps: collections.deque[float] = collections.deque()
        self._last_failure_time: float | None = None
        self._state = "closed"
        self._half_open_in_progress = False
        self._lock = asyncio.Lock()

    def _prune_old_failures(self) -> None:
        """Remove failure timestamps outside the rolling window."""
        cutoff = time.monotonic() - self._rolling_window_seconds
        while self._failure_timestamps and self._failure_timestamps[0] <= cutoff:
            self._failure_timestamps.popleft()

    def _cooldown_expired(self) -> bool:
        """Check if the cooldown period has elapsed since the last failure."""
        if self._last_failure_time is None:
            return False
        return (time.monotonic() - self._last_failure_time) >= self._cooldown_seconds

    @property
    def failure_count(self) -> int:
        """Failure count within the rolling window (read-only, no mutation).

        For monitoring, health checks, and observability. Uses read-only
        iteration without calling _prune_old_failures() to avoid mutating
        the deque without the asyncio.Lock.

        R10 fix (DeepSeek F5): replaced mutation-based count with read-only
        iteration for concurrency safety.

        Note: This count filters by cutoff on read, so it is always accurate
        even if the deque contains stale entries. Stale entries are pruned
        on the next record_failure() call under the lock.
        """
        cutoff = time.monotonic() - self._rolling_window_seconds
        return sum(1 for t in self._failure_timestamps if t > cutoff)

    @property
    def state(self) -> str:
        """Current circuit breaker state (read-only, no mutation).

        To check whether to allow a request, use ``allow_request()``
        which performs the open -> half_open transition under the lock.
        """
        if self._state == "open" and self._cooldown_expired():
            return "half_open"
        return self._state

    @property
    def is_open(self) -> bool:
        """Approximate breaker state for monitoring/health checks only.

        **WARNING**: This is a non-atomic read (no lock).  The sequence
        ``read _state -> call _cooldown_expired() -> compare`` is NOT atomic
        under async concurrency: another coroutine can call ``record_success()``
        between the reads, causing ``state`` to return ``"half_open"`` when the
        breaker is actually ``"closed"``.

        For **control flow** (deciding whether to allow a request), always use
        ``allow_request()`` which performs the state check under ``self._lock``.

        For **monitoring/health checks**, this property is acceptable: a
        briefly stale reading is harmless for dashboard display.

        Returns:
            True if the breaker is open and cooldown has NOT expired.
        """
        return self.state == "open"

    @property
    def is_half_open(self) -> bool:
        """Approximate half-open state for monitoring only.

        Same concurrency caveats as ``is_open`` — use ``allow_request()``
        for authoritative checks.

        Returns:
            True if the breaker is open and cooldown HAS expired.
        """
        return self.state == "half_open"

    async def get_state(self) -> str:
        """Lock-protected authoritative state read for accurate monitoring.

        Use this instead of ``is_open``/``is_half_open`` when accurate state
        is required (e.g., health check endpoints, structured metrics).

        Returns:
            Current state: "closed", "open", or "half_open".
        """
        async with self._lock:
            if self._state == "open" and self._cooldown_expired():
                return "half_open"
            return self._state

    async def get_failure_count(self) -> int:
        """Lock-protected authoritative failure count with pruning.

        Use this for accurate monitoring/health checks. Unlike the
        ``failure_count`` property (which is read-only and approximate),
        this method prunes stale entries under the lock before counting.

        R11 fix: GPT F-005 (failure_count drift without pruning).
        """
        async with self._lock:
            self._prune_old_failures()
            return len(self._failure_timestamps)

    async def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Performs the open -> half_open transition atomically under the lock.
        In ``half_open`` state, only one probe request is allowed at a time.

        Returns:
            True if the request is allowed, False if blocked.
        """
        async with self._lock:
            # Perform open -> half_open transition under the lock (race-safe)
            if self._state == "open" and self._cooldown_expired():
                self._state = "half_open"
                self._half_open_in_progress = False

            if self._state == "closed":
                return True
            if self._state == "open":
                return False
            # half_open: allow exactly one probe
            if self._state == "half_open" and not self._half_open_in_progress:
                self._half_open_in_progress = True
                return True
            return False

    async def record_success(self) -> None:
        """Record a successful LLM call and reset the breaker to closed state.

        Clears all failure timestamps and transitions from any state to closed.
        Thread-safe via ``asyncio.Lock``.
        """
        async with self._lock:
            prev_state = self._state
            self._failure_timestamps.clear()
            self._state = "closed"
            self._half_open_in_progress = False
            if prev_state != "closed":
                logger.info(
                    "Circuit breaker %s -> closed (recovered after successful probe)",
                    prev_state,
                )

    async def record_cancellation(self) -> None:
        """Handle a cancelled request (client disconnect) without counting as failure.

        Resets the ``_half_open_in_progress`` flag so a cancelled half-open
        probe does not permanently block subsequent requests. Does NOT
        record a failure or affect the failure threshold.

        R11 fix: DeepSeek F-005 — CancelledError (SSE client disconnect) is
        not an LLM failure and should not count toward the circuit breaker
        threshold. The previous approach (record_failure on CancelledError)
        caused artificially inflated failure counts under normal SSE traffic.
        """
        async with self._lock:
            if self._state == "half_open" and self._half_open_in_progress:
                # Probe was cancelled (client disconnected). Reset probe flag
                # so the next request can try another probe, but do NOT
                # re-open the breaker (cancellation is not a LLM failure).
                self._half_open_in_progress = False
                logger.info(
                    "Circuit breaker half_open probe cancelled (client disconnect) — "
                    "probe flag reset, state remains half_open"
                )

    async def record_failure(self) -> None:
        """Record a failed LLM call and potentially trip the breaker.

        Appends the current timestamp to the failure window, prunes old
        failures, and transitions to open if the threshold is exceeded.
        In half-open state, any failure immediately re-opens the breaker.
        Thread-safe via ``asyncio.Lock``.
        """
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


# R10 fix (DeepSeek F4): converted from @lru_cache to TTLCache to match
# the consistent TTL-cache pattern used by _get_llm(), _get_validator_llm(),
# and _get_whisper_llm(). TTL ensures settings changes (CB_FAILURE_THRESHOLD,
# CB_COOLDOWN_SECONDS, CB_ROLLING_WINDOW_SECONDS) are picked up within 1 hour.
#
# R11 fix: 3/3 reviewer consensus on CB TTL issues. The TTL serves two
# purposes: (1) config refresh and (2) implicit state reset. For incident
# response, use clear_circuit_breaker_cache() for immediate config reload
# rather than waiting for the 1-hour TTL to expire.
_cb_cache: TTLCache = TTLCache(maxsize=1, ttl=3600)
_cb_lock = asyncio.Lock()


async def _get_circuit_breaker() -> CircuitBreaker:
    """Get or create the circuit breaker singleton (TTL-cached, 1-hour refresh).

    Lazy initialization avoids reading settings at import time.
    TTL cache refreshes hourly, consistent with LLM singleton caching pattern.
    Coroutine-safe via ``asyncio.Lock`` (matches ``_get_llm()`` and
    ``_get_validator_llm()`` patterns).

    R15 fix (3/3 consensus: DeepSeek F-002, Gemini M2, GPT F1): added
    asyncio.Lock for pattern consistency with other singleton caches.
    Without the lock, two concurrent coroutines hitting a TTL expiry could
    both create separate CircuitBreaker instances, splitting failure state.
    While the function body is fully synchronous (no yield points in asyncio),
    the lock ensures correctness if future changes add async operations.

    For immediate config changes during incidents, call
    ``clear_circuit_breaker_cache()`` which forces the next call to
    create a fresh CircuitBreaker with current settings.

    Production note: Circuit breaker state is process-scoped (in-memory).
    In multi-container deployments (Cloud Run), each container maintains
    independent failure tracking. This is acceptable for per-container
    health detection. For global circuit breaking across containers,
    use Redis-backed state with ``CB_BACKEND=redis`` config (not yet
    implemented -- current single-container deployment uses in-memory).
    """
    async with _cb_lock:
        cached = _cb_cache.get("cb")
        if cached is not None:
            return cached
        settings = get_settings()
        cb = CircuitBreaker(
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            cooldown_seconds=settings.CB_COOLDOWN_SECONDS,
            rolling_window_seconds=settings.CB_ROLLING_WINDOW_SECONDS,
        )
        _cb_cache["cb"] = cb
        return cb


def clear_circuit_breaker_cache() -> None:
    """Force-clear the circuit breaker cache for immediate config reload.

    Use during incidents when CB thresholds/cooldowns need immediate tuning
    without waiting for the 1-hour TTL to expire. The next call to
    ``_get_circuit_breaker()`` will create a fresh instance with current
    settings.

    R11 fix: DeepSeek F-010, Gemini F-003, GPT F-004 (3/3 consensus).
    """
    _cb_cache.clear()
    logger.info("Circuit breaker cache cleared — next call creates fresh instance")
