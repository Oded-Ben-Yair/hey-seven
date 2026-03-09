"""Tests for R46 D8 scalability improvements.

Covers: State backend ping(), circuit breaker with InMemoryBackend,
langfuse cache jitter, semaphore timeout, config fields.

Mock-based tests (RedisBackend, rate limiter Redis integration,
rate limiter Redis distributed) removed per NO-MOCK ground rule.
"""

import asyncio

import pytest

from src.state_backend import InMemoryBackend, StateBackend


# ---------------------------------------------------------------------------
# StateBackend.ping()
# ---------------------------------------------------------------------------


class TestStateBackendPing:
    """R46 D8: StateBackend ping() method for health checks."""

    def test_in_memory_ping_always_true(self):
        """InMemoryBackend.ping() always returns True."""
        backend = InMemoryBackend()
        assert backend.ping() is True

    def test_state_backend_abc_requires_ping(self):
        """StateBackend ABC requires ping() implementation."""
        assert hasattr(StateBackend, "ping")


# ---------------------------------------------------------------------------
# CircuitBreaker with StateBackend (InMemoryBackend — no mocks)
# ---------------------------------------------------------------------------


class TestCircuitBreakerBackendSync:
    """R46 D8: Circuit breaker syncs state to/from InMemoryBackend."""

    @pytest.fixture
    def mock_backend(self):
        """Create an InMemoryBackend that stores data in a dict."""
        return InMemoryBackend()

    @pytest.fixture
    def cb_with_backend(self, mock_backend):
        """Circuit breaker with a state backend attached."""
        from src.agent.circuit_breaker import CircuitBreaker

        return CircuitBreaker(
            failure_threshold=3,
            cooldown_seconds=5.0,
            rolling_window_seconds=60.0,
            state_backend=mock_backend,
        )

    @pytest.fixture
    def cb_without_backend(self):
        """Circuit breaker without a state backend (local-only)."""
        from src.agent.circuit_breaker import CircuitBreaker

        return CircuitBreaker(
            failure_threshold=3,
            cooldown_seconds=5.0,
            rolling_window_seconds=60.0,
            state_backend=None,
        )

    @pytest.mark.asyncio
    async def test_sync_to_backend_writes_state(self, cb_with_backend, mock_backend):
        """record_failure() syncs state to backend."""
        await cb_with_backend.record_failure()
        state = mock_backend.get("cb:state")
        assert state == "closed"  # 1 failure, threshold=3
        count = mock_backend.get("cb:failure_count")
        assert count == "1"

    @pytest.mark.asyncio
    async def test_sync_to_backend_writes_open_state(
        self, cb_with_backend, mock_backend
    ):
        """After enough failures, backend shows open state."""
        for _ in range(3):
            await cb_with_backend.record_failure()
        state = mock_backend.get("cb:state")
        assert state == "open"
        count = mock_backend.get("cb:failure_count")
        assert count == "3"

    @pytest.mark.asyncio
    async def test_sync_from_backend_promotes_closed_to_open(
        self, cb_with_backend, mock_backend
    ):
        """If backend says open with enough failures, local CB adopts open state."""
        # Simulate another instance writing open state to backend
        mock_backend.set("cb:state", "open", ttl=300)
        mock_backend.set("cb:failure_count", "5", ttl=300)
        cb_with_backend._last_backend_sync = 0  # Force sync

        assert cb_with_backend._state == "closed"
        rs, rc = await cb_with_backend._read_backend_state()
        if rs is not None and rc is not None:
            cb_with_backend._apply_backend_state(rs, rc)
        assert cb_with_backend._state == "open"

    @pytest.mark.asyncio
    async def test_sync_from_backend_ignores_insufficient_failures(
        self, cb_with_backend, mock_backend
    ):
        """Backend open state is ignored if failure count < threshold."""
        mock_backend.set("cb:state", "open", ttl=300)
        mock_backend.set("cb:failure_count", "1", ttl=300)  # Below threshold of 3
        cb_with_backend._last_backend_sync = 0

        rs, rc = await cb_with_backend._read_backend_state()
        if rs is not None and rc is not None:
            cb_with_backend._apply_backend_state(rs, rc)
        assert cb_with_backend._state == "closed"  # Not promoted

    @pytest.mark.asyncio
    async def test_sync_rate_limited(self, cb_with_backend, mock_backend):
        """Backend reads are rate-limited to every 5 seconds."""
        mock_backend.set("cb:state", "open", ttl=300)
        mock_backend.set("cb:failure_count", "5", ttl=300)

        # First sync should work
        cb_with_backend._last_backend_sync = 0
        rs, rc = await cb_with_backend._read_backend_state()
        if rs is not None and rc is not None:
            cb_with_backend._apply_backend_state(rs, rc)
        assert cb_with_backend._state == "open"

        # Reset state manually
        cb_with_backend._state = "closed"

        # Second sync within 5s should be skipped
        rs, rc = await cb_with_backend._read_backend_state()
        if rs is not None and rc is not None:
            cb_with_backend._apply_backend_state(rs, rc)
        assert cb_with_backend._state == "closed"  # Not synced again

    @pytest.mark.asyncio
    async def test_no_backend_skips_sync(self, cb_without_backend):
        """Without backend, sync methods are no-ops."""
        await cb_without_backend._sync_to_backend()
        rs, rc = await cb_without_backend._read_backend_state()
        if rs is not None and rc is not None:
            cb_without_backend._apply_backend_state(rs, rc)
        # Should not raise

    @pytest.mark.asyncio
    async def test_record_success_syncs_to_backend(self, cb_with_backend, mock_backend):
        """record_success() also syncs state to backend."""
        await cb_with_backend.record_failure()
        await cb_with_backend.record_success()
        state = mock_backend.get("cb:state")
        assert state == "closed"

    @pytest.mark.asyncio
    async def test_allow_request_syncs_from_backend(
        self, cb_with_backend, mock_backend
    ):
        """allow_request() reads from backend before checking local state."""
        mock_backend.set("cb:state", "open", ttl=300)
        mock_backend.set("cb:failure_count", "5", ttl=300)
        cb_with_backend._last_backend_sync = 0

        allowed = await cb_with_backend.allow_request()
        assert allowed is False  # Backend promoted to open


# ---------------------------------------------------------------------------
# Langfuse Cache Jitter
# ---------------------------------------------------------------------------


class TestLangfuseCacheJitter:
    """R46 D8: Langfuse cache has TTL jitter for thundering herd prevention."""

    def test_langfuse_cache_ttl_has_jitter(self):
        """Langfuse cache TTL should be 3600 + 0-300 (not exactly 3600)."""
        from src.observability.langfuse_client import _langfuse_cache

        # TTL should be between 3600 and 3900
        assert 3600 <= _langfuse_cache.ttl <= 3900


# ---------------------------------------------------------------------------
# Semaphore Timeout
# ---------------------------------------------------------------------------


class TestSemaphoreTimeout:
    """R46 D8: LLM semaphore has timeout for backpressure."""

    @pytest.mark.asyncio
    async def test_semaphore_module_level_exists(self):
        """_LLM_SEMAPHORE exists at module level."""
        from src.agent.agents._base import _LLM_SEMAPHORE

        assert isinstance(_LLM_SEMAPHORE, asyncio.Semaphore)


# ---------------------------------------------------------------------------
# Config New Fields
# ---------------------------------------------------------------------------


class TestR46ConfigFields:
    """R46: New deployment config fields exist with correct defaults."""

    def test_kms_key_path_default(self):
        from src.config import Settings

        s = Settings()
        assert s.KMS_KEY_PATH == ""

    def test_canary_error_threshold_default(self):
        from src.config import Settings

        s = Settings()
        assert s.CANARY_ERROR_THRESHOLD == 5.0

    def test_canary_stage_wait_default(self):
        from src.config import Settings

        s = Settings()
        assert s.CANARY_STAGE_WAIT_SECONDS == 60

    def test_llm_semaphore_timeout_default(self):
        from src.config import Settings

        s = Settings()
        assert s.LLM_SEMAPHORE_TIMEOUT == 30
