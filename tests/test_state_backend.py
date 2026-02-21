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


class TestGetStateBackend:
    def test_default_is_memory(self):
        backend = get_state_backend()
        assert isinstance(backend, InMemoryBackend)
