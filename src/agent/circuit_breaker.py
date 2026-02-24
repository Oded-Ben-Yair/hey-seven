"""Circuit breaker for LLM calls.

Protects against cascading failures by tracking LLM errors within a
rolling time window and temporarily blocking requests when a threshold
is exceeded.

States: closed (normal) -> open (blocking) -> half_open (probe one request).
Thread-safe via ``asyncio.Lock`` for concurrent coroutine access.

**Multi-instance state sharing (v1.1)**: When ``STATE_BACKEND=redis``, the
circuit breaker syncs state to Redis (Cloud Memorystore) after each mutation.
Local in-memory deque serves as L1 cache for sub-millisecond reads; Redis
serves as L2 for cross-instance state propagation. State promotion is
one-directional: if ANY instance detects an LLM outage (opens its CB),
other instances adopt the open state on their next sync. Demotion (open ->
closed) happens locally per-instance via the standard half-open probe.

Fallback: If Redis is unavailable, the CB operates in local-only mode
(identical to v1.0 behavior). Availability > consistency.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import TYPE_CHECKING

from cachetools import TTLCache

from src.config import get_settings

if TYPE_CHECKING:
    from src.state_backend import StateBackend

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
        state_backend: StateBackend | None = None,
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

        # R46 D8: Optional Redis backend for cross-instance state sharing.
        # Local deque = L1 (sub-ms reads), Redis = L2 (cross-instance sync).
        # Fallback to local-only if backend is None or Redis fails.
        self._backend = state_backend
        self._backend_sync_interval = 5.0  # seconds between backend reads
        self._last_backend_sync = 0.0

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

    def _backend_key(self, suffix: str) -> str:
        """Redis key for circuit breaker shared state."""
        return f"cb:{suffix}"

    async def _sync_to_backend(self) -> None:
        """Push local CB state to shared backend (best-effort, non-blocking).

        Called after each state mutation (record_failure, record_success).
        Writes state, failure count, and last failure time with TTL slightly
        longer than the rolling window to ensure stale entries expire.

        R47 fix C2: Uses native redis.asyncio client instead of
        asyncio.to_thread() wrapper. to_thread() exhausts the default
        ThreadPoolExecutor (~8-10 threads on Cloud Run) under 50 concurrent
        SSE streams. Native async client uses zero threads.
        """
        if not self._backend:
            return
        try:
            ttl = int(self._rolling_window_seconds) + 60
            state = self._state
            failure_count = str(len(self._failure_timestamps))
            last_failure = (
                str(time.time()) if self._last_failure_time is not None else None
            )

            await self._backend.async_set(self._backend_key("state"), state, ttl=ttl)
            await self._backend.async_set(
                self._backend_key("failure_count"),
                failure_count,
                ttl=ttl,
            )
            if last_failure is not None:
                await self._backend.async_set(
                    self._backend_key("last_failure_time"),
                    last_failure,
                    ttl=ttl,
                )
        except Exception:
            logger.debug("CB backend sync failed (non-critical)", exc_info=True)

    async def _sync_from_backend(self) -> None:
        """Pull shared CB state from backend — bidirectional sync.

        Promotes state from closed -> open AND recovers from open -> closed.
        This ensures that if ANY instance detects an LLM outage, all instances
        benefit from that detection. It also ensures that when one instance
        successfully probes and recovers (open -> closed), other instances
        adopt the closed state instead of remaining open until their own
        cooldown expires.

        R47 fix C3: Added bidirectional sync (recovery propagation).
        Previously only promoted closed->open. If Instance A recovered via
        half-open probe and synced "closed" to Redis, Instance B stayed open
        until its own cooldown expired — defeating the purpose of shared state.

        Rate-limited to one read per ``_backend_sync_interval`` seconds.

        R47 fix C2: Uses native redis.asyncio client instead of
        asyncio.to_thread(). See _sync_to_backend docstring for rationale.
        """
        if not self._backend:
            return
        now = time.monotonic()
        if now - self._last_backend_sync < self._backend_sync_interval:
            return
        self._last_backend_sync = now
        try:
            state_key = self._backend_key("state")
            count_key = self._backend_key("failure_count")

            remote_state = await self._backend.async_get(state_key)
            remote_count_str = await self._backend.async_get(count_key)
            if remote_state and remote_count_str:
                remote_count = int(remote_count_str)
                # Promote: if remote says open and local says closed,
                # adopt the remote state (another instance detected the outage).
                if remote_state == "open" and self._state == "closed":
                    if remote_count >= self._failure_threshold:
                        self._state = "open"
                        logger.info(
                            "CB synced from backend: closed -> open "
                            "(remote failure_count=%d, threshold=%d)",
                            remote_count,
                            self._failure_threshold,
                        )
                # R47 fix C3: Recovery propagation. If remote says closed and
                # local says open, another instance has successfully recovered
                # via half-open probe. Adopt closed state to avoid serving
                # fallback responses unnecessarily.
                elif remote_state == "closed" and self._state == "open":
                    if remote_count < self._failure_threshold:
                        prev = self._state
                        self._state = "closed"
                        self._half_open_in_progress = False
                        logger.info(
                            "CB synced from backend: %s -> closed "
                            "(remote recovered, failure_count=%d)",
                            prev,
                            remote_count,
                        )
        except Exception:
            logger.debug("CB backend read failed (non-critical)", exc_info=True)

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

    async def get_metrics(self) -> dict:
        """Lock-protected metrics snapshot for observability endpoints.

        Returns a structlog-compatible dict with current circuit breaker
        state, failure count (pruned), and configuration parameters.
        All reads happen under the async lock to ensure a consistent
        snapshot (state and failure_count are from the same instant).

        Returns:
            Dict with keys: state, failure_count, cooldown_seconds,
            rolling_window_seconds, last_failure_time_ago.
        """
        async with self._lock:
            self._prune_old_failures()
            failure_count = len(self._failure_timestamps)

            if self._state == "open" and self._cooldown_expired():
                state = "half_open"
            else:
                state = self._state

            last_failure_ago: float | None = None
            if self._last_failure_time is not None:
                last_failure_ago = round(time.monotonic() - self._last_failure_time, 1)

        return {
            "state": state,
            "failure_count": failure_count,
            "cooldown_seconds": self._cooldown_seconds,
            "rolling_window_seconds": self._rolling_window_seconds,
            "last_failure_time_ago": last_failure_ago,
        }

    async def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Performs the open -> half_open transition atomically under the lock.
        In ``half_open`` state, only one probe request is allowed at a time.

        R46 D8: Syncs from backend before checking local state, so that
        if another instance has tripped the CB, this instance also blocks.

        R47 fix C15: Redis I/O (_sync_from_backend) runs OUTSIDE the lock.
        Previously, sync was inside the lock, meaning a 100ms Redis stall
        serialized all 50 concurrent callers (head-of-line blocking).
        Now: multiple coroutines can read Redis simultaneously; only local
        state mutation happens under the lock.

        R47 fix C10: Prune old failures before state checks to prevent
        unbounded deque growth in long-running containers.

        Returns:
            True if the request is allowed, False if blocked.
        """
        # I/O outside lock — multiple coroutines can read Redis concurrently.
        await self._sync_from_backend()

        async with self._lock:
            # R47 fix C10: Prune stale failure timestamps to prevent unbounded
            # memory growth. Without this, a container running for days
            # accumulates expired timestamps that are never cleaned up.
            self._prune_old_failures()

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
        """Record a successful LLM call and transition the breaker.

        - **From closed**: clears all failure timestamps (healthy operation).
        - **From half_open**: halves the failure count instead of clearing all.
          This prevents rapid flapping under intermittent failure conditions
          (50% error rate). A single successful probe should reduce risk,
          not reset to a clean slate.

        R35 fix: industry-standard half-open recovery with decay instead of
        full history clear. GPT-5.2 and Gemini 3 Pro consensus.
        Thread-safe via ``asyncio.Lock``.
        """
        async with self._lock:
            prev_state = self._state
            if prev_state == "half_open":
                # Halve failure count: single success reduces risk but doesn't
                # erase evidence of prior instability.
                # R36 fix A3: Use max(..., 1) to always retain at least 1
                # failure timestamp. Without this, integer division with
                # small N (e.g., 1 // 2 = 0) clears ALL evidence, making
                # a single half-open success equivalent to a full reset.
                keep_count = max(len(self._failure_timestamps) // 2, 1)
                while len(self._failure_timestamps) > keep_count:
                    self._failure_timestamps.popleft()
            else:
                # In closed state, full clear is safe (healthy operation).
                self._failure_timestamps.clear()
            self._state = "closed"
            self._half_open_in_progress = False
            if prev_state != "closed":
                logger.info(
                    "Circuit breaker %s -> closed (recovered after successful probe, "
                    "%d failure timestamps retained)",
                    prev_state,
                    len(self._failure_timestamps),
                )
        await self._sync_to_backend()

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
        await self._sync_to_backend()


# R10 fix (DeepSeek F4): converted from @lru_cache to TTLCache to match
# the consistent TTL-cache pattern used by _get_llm(), _get_validator_llm(),
# and _get_whisper_llm(). TTL ensures settings changes (CB_FAILURE_THRESHOLD,
# CB_COOLDOWN_SECONDS, CB_ROLLING_WINDOW_SECONDS) are picked up within 1 hour.
#
# R11 fix: 3/3 reviewer consensus on CB TTL issues. The TTL serves two
# purposes: (1) config refresh and (2) implicit state reset. For incident
# response, use clear_circuit_breaker_cache() for immediate config reload
# rather than waiting for the 1-hour TTL to expire.
# R40 fix D8-C001: TTL jitter to prevent thundering herd on synchronized expiry.
import random as _random

_cb_cache: TTLCache = TTLCache(maxsize=1, ttl=3600 + _random.randint(0, 300))
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
        # R46 D8: Pass state backend for cross-instance CB state sharing.
        # Import here to avoid circular imports (state_backend imports config).
        backend: StateBackend | None = None
        if settings.STATE_BACKEND == "redis":
            try:
                from src.state_backend import get_state_backend

                backend = get_state_backend()
            except Exception:
                logger.warning(
                    "Redis backend unavailable for CB, using local-only",
                    exc_info=True,
                )
        cb = CircuitBreaker(
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            cooldown_seconds=settings.CB_COOLDOWN_SECONDS,
            rolling_window_seconds=settings.CB_ROLLING_WINDOW_SECONDS,
            state_backend=backend,
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
