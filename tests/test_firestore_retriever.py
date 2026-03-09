"""Tests for checkpointer factory and circuit breaker.

Mock purge R111: Removed TestGetRetrieverFactory (used MagicMock for embeddings),
TestEmbeddingsTaskType (used @patch for GoogleGenerativeAIEmbeddings class).
Retained: TestGetCheckpointer (env var patching only), TestCircuitBreakerEnhancements
(real CircuitBreaker objects).
"""

import os

import pytest


# ---------------------------------------------------------------------------
# TestGetCheckpointer (uses real code with env var patching only)
# ---------------------------------------------------------------------------


class TestGetCheckpointer:
    """Tests for the checkpointer factory."""

    @pytest.mark.asyncio
    async def test_returns_bounded_memory_saver_for_chroma(self, monkeypatch):
        """get_checkpointer returns BoundedMemorySaver when VECTOR_DB=chroma."""
        from src.agent.memory import (
            BoundedMemorySaver,
            clear_checkpointer_cache,
            get_checkpointer,
        )

        clear_checkpointer_cache()
        monkeypatch.setenv("VECTOR_DB", "chroma")
        from src.config import get_settings

        get_settings.cache_clear()
        cp = await get_checkpointer()
        assert isinstance(cp, BoundedMemorySaver)
        clear_checkpointer_cache()
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_returns_bounded_memory_saver_by_default(self):
        """get_checkpointer returns BoundedMemorySaver when VECTOR_DB not set."""
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
    async def test_firestore_saver_import_attempted(self, monkeypatch):
        """get_checkpointer attempts FirestoreSaver when VECTOR_DB=firestore."""
        from src.agent.memory import (
            BoundedMemorySaver,
            clear_checkpointer_cache,
            get_checkpointer,
        )

        clear_checkpointer_cache()
        monkeypatch.setenv("VECTOR_DB", "firestore")
        monkeypatch.setenv("FIRESTORE_PROJECT", "test-proj")
        from src.config import get_settings

        get_settings.cache_clear()
        cp = await get_checkpointer()
        assert isinstance(cp, BoundedMemorySaver)  # Fallback
        clear_checkpointer_cache()
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# TestCircuitBreakerEnhancements (deterministic -- real CircuitBreaker objects)
# ---------------------------------------------------------------------------


class TestCircuitBreakerEnhancements:
    """Tests for enhanced circuit breaker (rolling window, half_open, config)."""

    def test_circuit_breaker_default_values(self):
        """CircuitBreaker has sensible defaults."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        assert cb._failure_threshold == 5
        assert cb._cooldown_seconds == 60.0
        assert cb._rolling_window_seconds == 300.0

    def test_circuit_breaker_custom_values(self):
        """CircuitBreaker accepts custom threshold and cooldown values."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(
            failure_threshold=3,
            cooldown_seconds=30.0,
            rolling_window_seconds=120.0,
        )
        assert cb._failure_threshold == 3
        assert cb._cooldown_seconds == 30.0
        assert cb._rolling_window_seconds == 120.0

    @pytest.mark.asyncio
    async def test_rolling_window_expiry(self):
        """Failures outside the rolling window are pruned and don't count."""
        import asyncio

        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(
            failure_threshold=3, cooldown_seconds=60.0, rolling_window_seconds=0.1
        )

        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == "closed"
        assert cb.failure_count == 2

        # Wait for failures to expire outside the rolling window
        await asyncio.sleep(0.15)
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_allows_one_probe(self):
        """In half_open state, allow_request permits exactly one probe."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.01)

        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == "open"

        import asyncio

        await asyncio.sleep(0.02)
        assert cb.state == "half_open"

        assert await cb.allow_request() is True
        assert await cb.allow_request() is False

    @pytest.mark.asyncio
    async def test_half_open_success_closes(self):
        """Successful probe in half_open state closes the breaker."""
        import asyncio

        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.01)

        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == "open"

        await asyncio.sleep(0.02)
        assert cb.state == "half_open"

        await cb.record_success()
        assert cb.state == "closed"
        assert await cb.allow_request() is True

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        """Failed probe in half_open state re-opens the breaker."""
        import asyncio

        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.01)

        await cb.record_failure()
        await cb.record_failure()

        await asyncio.sleep(0.02)
        assert cb.state == "half_open"

        await cb.record_failure()
        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_closed_state_allows_requests(self):
        """Closed state always allows requests."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5)
        assert cb.state == "closed"
        assert await cb.allow_request() is True
        assert await cb.allow_request() is True

    @pytest.mark.asyncio
    async def test_open_state_blocks_requests(self):
        """Open state blocks all requests."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        await cb.record_failure()
        assert cb.state == "open"
        assert await cb.allow_request() is False
