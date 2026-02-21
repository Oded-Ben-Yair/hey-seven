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


class TestGetStateBackend:
    def test_default_is_memory(self):
        backend = get_state_backend()
        assert isinstance(backend, InMemoryBackend)
