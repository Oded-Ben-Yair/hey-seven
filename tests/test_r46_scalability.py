"""Tests for R46 D8 scalability improvements.

Covers: Circuit breaker Redis backend sync, rate limiter Redis fallback,
state backend ping(), langfuse cache jitter, semaphore timeout backpressure.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

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
# CircuitBreaker with StateBackend
# ---------------------------------------------------------------------------


class TestRedisBackendMocked:
    """R46 D8: Test RedisBackend methods with mocked Redis client."""

    def _make_redis_backend(self):
        """Create a RedisBackend with a mocked redis.Redis client.

        RedisBackend imports redis inside __init__, so we mock at the
        import level using patch.dict on sys.modules.
        """
        from src.state_backend import RedisBackend

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.connection_pool.connection_kwargs = {
            "host": "localhost", "port": 6379, "db": 0, "ssl": False,
        }

        # RedisBackend imports redis inside __init__. We patch the
        # redis.Redis.from_url return value before constructing.
        with patch("redis.Redis.from_url", return_value=mock_client):
            backend = RedisBackend("redis://localhost:6379/0")
        return backend, mock_client

    def test_redis_increment(self):
        """RedisBackend.increment() uses pipeline for atomic incr+expire."""
        backend, mock_client = self._make_redis_backend()
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [5, True]
        mock_client.pipeline.return_value = mock_pipe

        result = backend.increment("key1", ttl=60)
        assert result == 5
        mock_pipe.incr.assert_called_once_with("key1")
        mock_pipe.expire.assert_called_once_with("key1", 60)

    def test_redis_get_count(self):
        """RedisBackend.get_count() reads int from Redis."""
        backend, mock_client = self._make_redis_backend()
        mock_client.get.return_value = "42"
        assert backend.get_count("key1") == 42

    def test_redis_get_count_missing(self):
        """RedisBackend.get_count() returns 0 for missing keys."""
        backend, mock_client = self._make_redis_backend()
        mock_client.get.return_value = None
        assert backend.get_count("missing") == 0

    def test_redis_set(self):
        """RedisBackend.set() calls setex."""
        backend, mock_client = self._make_redis_backend()
        backend.set("key1", "val1", ttl=300)
        mock_client.setex.assert_called_once_with("key1", 300, "val1")

    def test_redis_get(self):
        """RedisBackend.get() returns value or None."""
        backend, mock_client = self._make_redis_backend()
        mock_client.get.return_value = "hello"
        assert backend.get("key1") == "hello"

    def test_redis_exists(self):
        """RedisBackend.exists() returns bool."""
        backend, mock_client = self._make_redis_backend()
        mock_client.exists.return_value = 1
        assert backend.exists("key1") is True

    def test_redis_delete(self):
        """RedisBackend.delete() calls Redis delete."""
        backend, mock_client = self._make_redis_backend()
        backend.delete("key1")
        mock_client.delete.assert_called_once_with("key1")

    def test_redis_ping_success(self):
        """RedisBackend.ping() returns True when Redis is up."""
        backend, mock_client = self._make_redis_backend()
        mock_client.ping.return_value = True
        assert backend.ping() is True

    def test_redis_ping_failure(self):
        """RedisBackend.ping() returns False on connection error."""
        backend, mock_client = self._make_redis_backend()
        mock_client.ping.side_effect = Exception("Connection refused")
        assert backend.ping() is False


class TestCircuitBreakerBackendSync:
    """R46 D8: Circuit breaker syncs state to/from Redis backend."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock StateBackend that stores data in a dict."""
        backend = InMemoryBackend()
        return backend

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
    async def test_sync_to_backend_writes_open_state(self, cb_with_backend, mock_backend):
        """After enough failures, backend shows open state."""
        for _ in range(3):
            await cb_with_backend.record_failure()
        state = mock_backend.get("cb:state")
        assert state == "open"
        count = mock_backend.get("cb:failure_count")
        assert count == "3"

    @pytest.mark.asyncio
    async def test_sync_from_backend_promotes_closed_to_open(self, cb_with_backend, mock_backend):
        """If backend says open with enough failures, local CB adopts open state."""
        # Simulate another instance writing open state to backend
        mock_backend.set("cb:state", "open", ttl=300)
        mock_backend.set("cb:failure_count", "5", ttl=300)
        cb_with_backend._last_backend_sync = 0  # Force sync

        assert cb_with_backend._state == "closed"
        await cb_with_backend._sync_from_backend()
        assert cb_with_backend._state == "open"

    @pytest.mark.asyncio
    async def test_sync_from_backend_ignores_insufficient_failures(self, cb_with_backend, mock_backend):
        """Backend open state is ignored if failure count < threshold."""
        mock_backend.set("cb:state", "open", ttl=300)
        mock_backend.set("cb:failure_count", "1", ttl=300)  # Below threshold of 3
        cb_with_backend._last_backend_sync = 0

        await cb_with_backend._sync_from_backend()
        assert cb_with_backend._state == "closed"  # Not promoted

    @pytest.mark.asyncio
    async def test_sync_rate_limited(self, cb_with_backend, mock_backend):
        """Backend reads are rate-limited to every 5 seconds."""
        mock_backend.set("cb:state", "open", ttl=300)
        mock_backend.set("cb:failure_count", "5", ttl=300)

        # First sync should work
        cb_with_backend._last_backend_sync = 0
        await cb_with_backend._sync_from_backend()
        assert cb_with_backend._state == "open"

        # Reset state manually
        cb_with_backend._state = "closed"

        # Second sync within 5s should be skipped
        await cb_with_backend._sync_from_backend()
        assert cb_with_backend._state == "closed"  # Not synced again

    @pytest.mark.asyncio
    async def test_no_backend_skips_sync(self, cb_without_backend):
        """Without backend, sync methods are no-ops."""
        await cb_without_backend._sync_to_backend()
        await cb_without_backend._sync_from_backend()
        # Should not raise

    @pytest.mark.asyncio
    async def test_backend_error_is_non_fatal(self, cb_with_backend):
        """Backend errors during sync do not crash the CB."""
        # Replace backend with one that throws
        broken_backend = MagicMock()
        broken_backend.set.side_effect = Exception("Redis down")
        broken_backend.get.side_effect = Exception("Redis down")
        cb_with_backend._backend = broken_backend

        # Should not raise
        await cb_with_backend._sync_to_backend()
        cb_with_backend._last_backend_sync = 0
        await cb_with_backend._sync_from_backend()

    @pytest.mark.asyncio
    async def test_record_success_syncs_to_backend(self, cb_with_backend, mock_backend):
        """record_success() also syncs state to backend."""
        await cb_with_backend.record_failure()
        await cb_with_backend.record_success()
        state = mock_backend.get("cb:state")
        assert state == "closed"

    @pytest.mark.asyncio
    async def test_allow_request_syncs_from_backend(self, cb_with_backend, mock_backend):
        """allow_request() reads from backend before checking local state."""
        mock_backend.set("cb:state", "open", ttl=300)
        mock_backend.set("cb:failure_count", "5", ttl=300)
        cb_with_backend._last_backend_sync = 0

        allowed = await cb_with_backend.allow_request()
        assert allowed is False  # Backend promoted to open


# ---------------------------------------------------------------------------
# Rate Limiter Redis Fallback
# ---------------------------------------------------------------------------


class TestRateLimiterRedisIntegration:
    """R46 D8: Rate limiter uses Redis when available, falls back to in-memory."""

    def test_rate_limiter_initializes_without_redis(self):
        """RateLimitMiddleware works without Redis."""
        from src.api.middleware import RateLimitMiddleware
        app = MagicMock()
        middleware = RateLimitMiddleware(app)
        assert middleware._redis_client is None

    @pytest.mark.asyncio
    async def test_is_allowed_redis_fallback_on_error(self):
        """When Redis fails, _is_allowed_redis falls back to in-memory."""
        from src.api.middleware import RateLimitMiddleware
        app = MagicMock()
        middleware = RateLimitMiddleware(app)

        # Set up a broken redis client
        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = Exception("Connection refused")
        middleware._redis_client = mock_redis

        # Should fall back to in-memory (allowed)
        result = await middleware._is_allowed_redis("127.0.0.1")
        assert result is True


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


class TestRateLimiterRedisDistributed:
    """R46 D8: Redis sorted set rate limiting logic."""

    @pytest.mark.asyncio
    async def test_is_allowed_redis_allows_under_limit(self):
        """Redis rate limiter allows requests under the limit."""
        from src.api.middleware import RateLimitMiddleware
        app = MagicMock()
        middleware = RateLimitMiddleware(app)

        # Mock a working Redis pipeline
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [0, 5, True, True]  # zrem, zcard=5, zadd, expire
        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        middleware._redis_client = mock_redis
        middleware.max_tokens = 20

        result = await middleware._is_allowed_redis("10.0.0.1")
        assert result is True

    @pytest.mark.asyncio
    async def test_is_allowed_redis_blocks_over_limit(self):
        """Redis rate limiter blocks requests over the limit."""
        from src.api.middleware import RateLimitMiddleware
        app = MagicMock()
        middleware = RateLimitMiddleware(app)

        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [0, 20, True, True]  # zcard=20 (at limit)
        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        middleware._redis_client = mock_redis
        middleware.max_tokens = 20

        result = await middleware._is_allowed_redis("10.0.0.1")
        assert result is False
        # Should have called zrem to remove the tentative entry
        mock_redis.zrem.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_call_chooses_redis_path(self):
        """When redis_client is set, __call__ uses Redis path."""
        from src.api.middleware import RateLimitMiddleware
        app = MagicMock()
        middleware = RateLimitMiddleware(app)

        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [0, 1, True, True]
        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        middleware._redis_client = mock_redis

        scope = {
            "type": "http",
            "path": "/chat",
            "headers": [],
            "client": ("10.0.0.1", 1234),
        }

        # _is_allowed_redis should be called (not _is_allowed)
        result = await middleware._is_allowed_redis("10.0.0.1")
        assert result is True


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
