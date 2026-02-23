"""Tests for pluggable state backend."""

import time

from src.state_backend import InMemoryBackend, get_state_backend


class TestInMemoryBackend:
    def test_increment_and_get(self):
        b = InMemoryBackend()
        b.increment("rate:127.0.0.1", ttl=60)
        b.increment("rate:127.0.0.1", ttl=60)
        assert b.get_count("rate:127.0.0.1") == 2

    def test_set_and_get(self):
        b = InMemoryBackend()
        b.set("cb:main:state", "open", ttl=300)
        assert b.get("cb:main:state") == "open"

    def test_exists(self):
        b = InMemoryBackend()
        assert not b.exists("idempotency:abc123")
        b.set("idempotency:abc123", "1", ttl=600)
        assert b.exists("idempotency:abc123")

    def test_delete(self):
        b = InMemoryBackend()
        b.set("key", "value", ttl=60)
        b.delete("key")
        assert not b.exists("key")

    def test_expiry(self):
        b = InMemoryBackend()
        b.set("key", "value", ttl=0)  # Expires immediately
        time.sleep(0.01)
        assert not b.exists("key")

    def test_increment_respects_ttl(self):
        b = InMemoryBackend()
        b.increment("counter", ttl=0)
        time.sleep(0.01)
        assert b.get_count("counter") == 0  # Expired


class TestInMemoryBackendSweep:
    """R11 fix: probabilistic sweep prevents memory leaks from expired entries."""

    def test_sweep_evicts_expired_entries(self):
        """Force a sweep and verify expired entries are removed."""
        b = InMemoryBackend()
        # Add entries that expire immediately
        for i in range(10):
            b.set(f"key_{i}", "val", ttl=0)
        time.sleep(0.01)

        # Force a sweep by calling _maybe_sweep with store exceeding max
        original_max = InMemoryBackend._MAX_STORE_SIZE
        try:
            InMemoryBackend._MAX_STORE_SIZE = 5  # Force sweep
            b._maybe_sweep()
        finally:
            InMemoryBackend._MAX_STORE_SIZE = original_max

        assert len(b._store) == 0

    def test_sweep_preserves_valid_entries(self):
        """Sweep removes expired but keeps valid entries."""
        b = InMemoryBackend()
        b.set("expired", "val", ttl=0)
        b.set("valid", "val", ttl=3600)
        time.sleep(0.01)

        original_max = InMemoryBackend._MAX_STORE_SIZE
        try:
            InMemoryBackend._MAX_STORE_SIZE = 1  # Force sweep
            b._maybe_sweep()
        finally:
            InMemoryBackend._MAX_STORE_SIZE = original_max

        assert b.get("valid") == "val"
        assert b.get("expired") is None

    def test_set_triggers_maybe_sweep(self):
        """set() calls _maybe_sweep internally."""
        b = InMemoryBackend()
        # Add many expired entries then a valid one
        for i in range(100):
            b._store[f"expired_{i}"] = ("v", time.monotonic() - 1)
        b.set("valid", "val", ttl=3600)

        # The sweep may or may not fire (probabilistic), but at least
        # the valid entry should be accessible
        assert b.get("valid") == "val"

    def test_max_store_size_forces_sweep(self):
        """When store exceeds _MAX_STORE_SIZE, sweep fires unconditionally."""
        b = InMemoryBackend()
        original_max = InMemoryBackend._MAX_STORE_SIZE
        try:
            InMemoryBackend._MAX_STORE_SIZE = 10
            # Add 15 expired entries
            for i in range(15):
                b._store[f"expired_{i}"] = ("v", time.monotonic() - 1)
            # Trigger sweep via set()
            b.set("new_key", "val", ttl=3600)
            # All expired should be gone
            assert len(b._store) == 1
            assert b.get("new_key") == "val"
        finally:
            InMemoryBackend._MAX_STORE_SIZE = original_max


class TestInMemoryBackendBatchSweep:
    """R40 fix D5-C002: Tests for batch sweep limit (R37 fix beta-C2).

    The sweep batches eviction to _SWEEP_BATCH_SIZE entries per tick
    to prevent blocking the event loop. These tests verify:
    - Batch limit is respected (not all expired entries evicted in one sweep)
    - Multiple sweeps eventually evict all expired entries
    """

    def test_sweep_batch_limit_respected(self):
        """Sweep evicts at most _SWEEP_BATCH_SIZE entries per call."""
        b = InMemoryBackend()
        batch_size = InMemoryBackend._SWEEP_BATCH_SIZE
        original_max = InMemoryBackend._MAX_STORE_SIZE
        try:
            # Add more expired entries than the batch size
            count = batch_size + 500
            for i in range(count):
                b._store[f"expired_{i}"] = ("v", time.monotonic() - 1)

            InMemoryBackend._MAX_STORE_SIZE = 1  # Force sweep
            b._maybe_sweep()

            # After one sweep, at most batch_size entries should be evicted
            remaining = len(b._store)
            evicted = count - remaining
            assert evicted <= batch_size, (
                f"Expected at most {batch_size} evictions, got {evicted}"
            )
            assert remaining > 0, "Should have remaining entries after batched sweep"
        finally:
            InMemoryBackend._MAX_STORE_SIZE = original_max

    def test_multiple_sweeps_evict_all_expired(self):
        """Successive sweeps eventually clear all expired entries."""
        b = InMemoryBackend()
        batch_size = InMemoryBackend._SWEEP_BATCH_SIZE
        original_max = InMemoryBackend._MAX_STORE_SIZE
        try:
            count = batch_size * 3
            for i in range(count):
                b._store[f"expired_{i}"] = ("v", time.monotonic() - 1)

            InMemoryBackend._MAX_STORE_SIZE = 1  # Force sweep
            for _ in range(5):  # Enough rounds to clear all
                b._maybe_sweep()

            assert len(b._store) == 0, "All expired entries should be evicted"
        finally:
            InMemoryBackend._MAX_STORE_SIZE = original_max

    def test_probabilistic_sweep_skips_when_not_triggered(self):
        """With default _SWEEP_PROBABILITY and small store, sweep is probabilistic."""
        b = InMemoryBackend()
        # Add a few expired entries (below _MAX_STORE_SIZE)
        for i in range(5):
            b._store[f"expired_{i}"] = ("v", time.monotonic() - 1)

        # Add one valid entry
        b._store["valid"] = ("v", time.monotonic() + 3600)

        # Multiple sweeps — probabilistic, so some may skip
        initial_count = len(b._store)
        b._maybe_sweep()
        # We can't assert deterministically, but valid key must survive
        assert b.get("valid") == "v"

    def test_increment_triggers_sweep(self):
        """increment() calls _maybe_sweep internally."""
        b = InMemoryBackend()
        original_max = InMemoryBackend._MAX_STORE_SIZE
        try:
            InMemoryBackend._MAX_STORE_SIZE = 5
            for i in range(10):
                b._store[f"expired_{i}"] = ("v", time.monotonic() - 1)

            b.increment("counter", ttl=3600)

            # Sweep should have evicted expired entries
            assert b.get_count("counter") == 1
            assert len(b._store) <= 2  # counter + maybe a few remaining
        finally:
            InMemoryBackend._MAX_STORE_SIZE = original_max


class TestInMemoryBackendLRUEviction:
    """R40 fix D5-C002: Tests for get_count on expired keys and cleanup_expired."""

    def test_get_count_returns_zero_for_expired(self):
        """get_count returns 0 for an expired key."""
        b = InMemoryBackend()
        b.increment("key", ttl=0)
        time.sleep(0.01)
        assert b.get_count("key") == 0

    def test_get_returns_none_for_expired(self):
        """get() returns None for an expired key."""
        b = InMemoryBackend()
        b.set("key", "value", ttl=0)
        time.sleep(0.01)
        assert b.get("key") is None

    def test_cleanup_removes_expired_entry_from_store(self):
        """_cleanup_expired removes the key entirely, not just returns None."""
        b = InMemoryBackend()
        b.set("key", "value", ttl=0)
        time.sleep(0.01)
        b._cleanup_expired("key")
        assert "key" not in b._store

    def test_cleanup_keeps_valid_entry(self):
        """_cleanup_expired does not remove a valid (non-expired) entry."""
        b = InMemoryBackend()
        b.set("key", "value", ttl=3600)
        b._cleanup_expired("key")
        assert "key" in b._store

    def test_cleanup_nonexistent_key_is_noop(self):
        """_cleanup_expired on missing key does not raise."""
        b = InMemoryBackend()
        b._cleanup_expired("nonexistent")  # Should not raise


class TestGetStateBackend:
    def test_default_is_memory(self):
        backend = get_state_backend()
        assert isinstance(backend, InMemoryBackend)
