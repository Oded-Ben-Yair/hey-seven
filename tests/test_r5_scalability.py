"""Tests for R5 scalability fixes.

Covers: BoundedMemorySaver eviction, circuit breaker maxlen removal,
PII buffer max-size guard, LLM semaphore, guest profile memory store bounds.

Mock-based tests (rate limiter deque, feature flag cache lock, casino config
cache lock, ApiKeyMiddleware atomic tuple) removed per NO-MOCK ground rule.
"""

import asyncio

import pytest


# ---------------------------------------------------------------------------
# BoundedMemorySaver
# ---------------------------------------------------------------------------


class TestBoundedMemorySaver:
    """R5 fix: MemorySaver with LRU eviction (Gemini F1, GPT F10)."""

    def test_tracks_active_threads(self):
        """Thread access is tracked for LRU ordering."""
        from src.agent.memory import BoundedMemorySaver

        saver = BoundedMemorySaver(max_threads=5)
        assert saver.active_threads == 0

        # Simulate tracking
        saver._track_thread({"configurable": {"thread_id": "t1"}})
        saver._track_thread({"configurable": {"thread_id": "t2"}})
        assert saver.active_threads == 2

    def test_evicts_lru_when_at_capacity(self):
        """Oldest thread is evicted when max_threads exceeded."""
        from src.agent.memory import BoundedMemorySaver

        saver = BoundedMemorySaver(max_threads=3)

        for i in range(3):
            saver._track_thread({"configurable": {"thread_id": f"t{i}"}})
        assert saver.active_threads == 3

        # Adding 4th should evict t0 (LRU)
        saver._track_thread({"configurable": {"thread_id": "t3"}})
        assert saver.active_threads == 3
        assert "t0" not in saver._thread_order
        assert "t3" in saver._thread_order

    def test_recently_used_survives_eviction(self):
        """Re-accessing a thread moves it to MRU position."""
        from src.agent.memory import BoundedMemorySaver

        saver = BoundedMemorySaver(max_threads=3)

        saver._track_thread({"configurable": {"thread_id": "t0"}})
        saver._track_thread({"configurable": {"thread_id": "t1"}})
        saver._track_thread({"configurable": {"thread_id": "t2"}})

        # Re-access t0 (move to MRU)
        saver._track_thread({"configurable": {"thread_id": "t0"}})

        # Adding t3 should evict t1 (now LRU), not t0
        saver._track_thread({"configurable": {"thread_id": "t3"}})
        assert "t0" in saver._thread_order
        assert "t1" not in saver._thread_order

    @pytest.mark.asyncio
    async def test_get_checkpointer_returns_bounded(self):
        """get_checkpointer returns BoundedMemorySaver for dev mode."""
        from src.agent.memory import (
            BoundedMemorySaver,
            clear_checkpointer_cache,
            get_checkpointer,
        )

        clear_checkpointer_cache()
        cp = await get_checkpointer()
        assert isinstance(cp, BoundedMemorySaver)
        clear_checkpointer_cache()

    @pytest.mark.asyncio
    async def test_get_checkpointer_caches(self):
        """get_checkpointer returns same instance on repeated calls."""
        from src.agent.memory import clear_checkpointer_cache, get_checkpointer

        clear_checkpointer_cache()
        cp1 = await get_checkpointer()
        cp2 = await get_checkpointer()
        assert cp1 is cp2
        clear_checkpointer_cache()


# ---------------------------------------------------------------------------
# Circuit Breaker Maxlen Removal
# ---------------------------------------------------------------------------


class TestCircuitBreakerNoMaxlen:
    """R5 fix: deque without maxlen prevents undercounting (DeepSeek F1)."""

    def test_deque_has_no_maxlen(self):
        """Failure deque has no maxlen after R5 fix."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5)
        assert cb._failure_timestamps.maxlen is None

    @pytest.mark.asyncio
    async def test_high_failure_rate_trips_breaker(self):
        """Sustained failures beyond threshold trip the breaker correctly."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=10, rolling_window_seconds=60)

        # Record 15 failures rapidly
        for _ in range(15):
            await cb.record_failure()

        assert cb._state == "open"
        assert cb.failure_count >= 10

    @pytest.mark.asyncio
    async def test_failures_naturally_bounded_by_pruning(self):
        """Old failures outside the window are pruned, bounding memory."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=100, rolling_window_seconds=0.1)

        # Record failures
        for _ in range(20):
            await cb.record_failure()

        # Wait for window to expire
        await asyncio.sleep(0.15)

        # After pruning, count should be 0
        assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# R10 Circuit Breaker Fixes
# ---------------------------------------------------------------------------


class TestCircuitBreakerFailureCountReadOnly:
    """R10 fix (DeepSeek F5): failure_count is read-only (no mutation)."""

    @pytest.mark.asyncio
    async def test_failure_count_no_mutation(self):
        """failure_count does not call _prune_old_failures (no deque mutation)."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=10, rolling_window_seconds=300)
        for _ in range(5):
            await cb.record_failure()

        # Reading failure_count should not modify the deque
        count_before = len(cb._failure_timestamps)
        _ = cb.failure_count
        count_after = len(cb._failure_timestamps)
        assert count_before == count_after

    @pytest.mark.asyncio
    async def test_failure_count_filters_by_window(self):
        """failure_count only counts failures within the rolling window."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=100, rolling_window_seconds=0.05)
        for _ in range(3):
            await cb.record_failure()

        assert cb.failure_count == 3
        await asyncio.sleep(0.1)
        # After window expires, count should be 0 (without pruning the deque)
        assert cb.failure_count == 0


class TestCircuitBreakerTTLCache:
    """R10 fix (DeepSeek F4): CB factory uses TTLCache instead of @lru_cache."""

    def test_cb_cache_is_ttl_cache(self):
        """_cb_cache is a TTLCache instance (not @lru_cache)."""
        from src.agent.circuit_breaker import _cb_cache
        from cachetools import TTLCache

        assert isinstance(_cb_cache, TTLCache)

    @pytest.mark.asyncio
    async def test_cb_factory_returns_circuit_breaker(self):
        """_get_circuit_breaker returns a CircuitBreaker instance."""
        from src.agent.circuit_breaker import (
            CircuitBreaker,
            _cb_cache,
            _get_circuit_breaker,
        )

        _cb_cache.clear()
        cb = await _get_circuit_breaker()
        assert isinstance(cb, CircuitBreaker)
        _cb_cache.clear()

    @pytest.mark.asyncio
    async def test_cb_factory_caches(self):
        """Repeated calls return the same instance."""
        from src.agent.circuit_breaker import _cb_cache, _get_circuit_breaker

        _cb_cache.clear()
        cb1 = await _get_circuit_breaker()
        cb2 = await _get_circuit_breaker()
        assert cb1 is cb2
        _cb_cache.clear()


class TestCBRollingWindowConfigurable:
    """R10 fix (DeepSeek F8): rolling_window_seconds is configurable via Settings."""

    def test_settings_has_cb_rolling_window(self):
        """Settings includes CB_ROLLING_WINDOW_SECONDS with default 300."""
        from src.config import Settings

        s = Settings()
        assert s.CB_ROLLING_WINDOW_SECONDS == 300.0

    @pytest.mark.asyncio
    async def test_cb_factory_passes_rolling_window(self):
        """_get_circuit_breaker passes rolling_window_seconds from settings."""
        from src.agent.circuit_breaker import _cb_cache, _get_circuit_breaker

        _cb_cache.clear()
        cb = await _get_circuit_breaker()
        assert cb._rolling_window_seconds == 300.0
        _cb_cache.clear()


# ---------------------------------------------------------------------------
# PII Buffer Max Size
# ---------------------------------------------------------------------------


class TestPIIBufferMaxSize:
    """R5 fix: PII buffer has hard cap at 500 chars (Gemini F10).

    Updated for StreamingPIIRedactor: the cap is now defined as
    ``StreamingPIIRedactor.MAX_BUFFER`` and chat_stream creates
    an instance per request.
    """

    def test_max_buffer_constant_exists(self):
        """Verify StreamingPIIRedactor has MAX_BUFFER cap and graph uses it."""
        from src.agent.streaming_pii import StreamingPIIRedactor

        assert hasattr(StreamingPIIRedactor, "MAX_BUFFER")
        assert StreamingPIIRedactor.MAX_BUFFER == 500

        # Verify chat_stream creates a StreamingPIIRedactor
        import inspect
        import src.agent.graph as graph_module

        source = inspect.getsource(graph_module.chat_stream)
        assert "StreamingPIIRedactor" in source


# ---------------------------------------------------------------------------
# LLM Semaphore
# ---------------------------------------------------------------------------


class TestLLMSemaphore:
    """R5 fix: LLM concurrency backpressure (Gemini F3)."""

    def test_semaphore_exists(self):
        """_LLM_SEMAPHORE is defined in _base.py."""
        from src.agent.agents._base import _LLM_SEMAPHORE

        assert isinstance(_LLM_SEMAPHORE, asyncio.Semaphore)

    def test_semaphore_has_reasonable_limit(self):
        """Semaphore limit is between 5 and 50 (reasonable for Gemini API)."""
        from src.agent.agents._base import _LLM_SEMAPHORE

        # asyncio.Semaphore stores the value in _value
        assert 5 <= _LLM_SEMAPHORE._value <= 50


# ---------------------------------------------------------------------------
# Guest Profile Memory Store Bounds
# ---------------------------------------------------------------------------


class TestGuestProfileMemoryBounds:
    """R5 fix: in-memory guest profile store is bounded (GPT F2)."""

    def test_max_constant_defined(self):
        """_MEMORY_STORE_MAX is defined and reasonable."""
        from src.data.guest_profile import _MEMORY_STORE_MAX

        assert _MEMORY_STORE_MAX == 10_000


# ---------------------------------------------------------------------------
# Firestore Client Lock (guest_profile)
# ---------------------------------------------------------------------------


class TestFirestoreClientLock:
    """R5 fix: Firestore client creation is lock-protected (GPT F5)."""

    def test_lock_exists(self):
        """_firestore_client_lock exists in guest_profile module."""
        from src.data.guest_profile import _firestore_client_lock

        # threading.Lock() returns a _thread.lock instance; verify it has acquire/release
        assert hasattr(_firestore_client_lock, "acquire")
        assert hasattr(_firestore_client_lock, "release")

    def test_config_client_lock_exists(self):
        """_fs_config_client_lock exists in casino config module."""
        from src.casino.config import _fs_config_client_lock

        assert hasattr(_fs_config_client_lock, "acquire")
        assert hasattr(_fs_config_client_lock, "release")


# ---------------------------------------------------------------------------
# CASINO_ID Cache Keying (P2 Task 1)
# ---------------------------------------------------------------------------


class TestCasinoIdCacheKeying:
    """P2-T1: All caches keyed by CASINO_ID for multi-tenant safety."""

    def test_greeting_cache_is_ttlcache(self):
        """Greeting categories cache uses TTLCache, not lru_cache."""
        from src.agent.nodes import _greeting_cache
        from cachetools import TTLCache

        assert isinstance(_greeting_cache, TTLCache)

    def test_greeting_cache_keyed_by_casino_id(self):
        """Different casino_id values produce separate cache entries."""
        from src.agent.nodes import _build_greeting_categories, _greeting_cache

        _greeting_cache.clear()
        _build_greeting_categories(casino_id="casino_a")
        _build_greeting_categories(casino_id="casino_b")
        assert len(_greeting_cache) == 2

    def test_retriever_cache_key_includes_casino_id(self):
        """Retriever cache key includes CASINO_ID, not just 'default'."""
        from src.rag.pipeline import _retriever_cache

        # After any retrieval, the cache key should contain casino_id
        for key in _retriever_cache:
            assert key != "default", (
                "Cache key should include casino_id, not be 'default'"
            )
