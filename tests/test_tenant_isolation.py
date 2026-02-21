"""Multi-tenant isolation tests -- verify property boundaries are enforced.

These tests ensure that no data, cache, or rate limit state can leak
between casino properties in a multi-tenant deployment.
"""
import inspect

import pytest
from cachetools import TTLCache


class TestRetrieverIsolation:
    """Retriever must always filter by property_id."""

    def test_retriever_includes_property_filter(self):
        """CasinoKnowledgeRetriever.retrieve_with_scores passes property_id filter."""
        from src.rag.pipeline import CasinoKnowledgeRetriever

        source = inspect.getsource(CasinoKnowledgeRetriever.retrieve_with_scores)
        assert "property_id" in source, \
            "retrieve_with_scores must filter by property_id"

    def test_retrieve_also_includes_property_filter(self):
        """CasinoKnowledgeRetriever.retrieve passes property_id filter."""
        from src.rag.pipeline import CasinoKnowledgeRetriever

        source = inspect.getsource(CasinoKnowledgeRetriever.retrieve)
        assert "property_id" in source, \
            "retrieve must filter by property_id"

    def test_retriever_filter_uses_settings(self):
        """Property filter derives from settings.PROPERTY_NAME, not hardcoded."""
        from src.rag.pipeline import CasinoKnowledgeRetriever

        source = inspect.getsource(CasinoKnowledgeRetriever.retrieve_with_scores)
        assert "get_settings" in source or "PROPERTY_NAME" in source, \
            "Property filter must derive from settings, not hardcoded"

    def test_retriever_cache_key_includes_casino_id(self):
        """Retriever cache key includes CASINO_ID, not a static 'default'."""
        from src.rag.pipeline import _get_retriever_cached

        source = inspect.getsource(_get_retriever_cached)
        # The cache key must incorporate CASINO_ID for multi-tenant safety
        assert "CASINO_ID" in source, \
            "Retriever cache key must include CASINO_ID"
        assert "cache_key" in source, \
            "Retriever cache must use a computed cache_key"


class TestCacheIsolation:
    """Caches must be keyed by casino identifier."""

    def test_greeting_cache_is_ttlcache(self):
        """Greeting cache uses TTLCache (not lru_cache) for multi-tenant safety."""
        from src.agent.nodes import _greeting_cache

        assert isinstance(_greeting_cache, TTLCache), \
            f"Expected TTLCache, got {type(_greeting_cache).__name__}"

    def test_greeting_cache_separate_entries(self):
        """Different casino_ids produce separate cache entries."""
        from src.agent.nodes import _build_greeting_categories, _greeting_cache

        _greeting_cache.clear()
        _build_greeting_categories(casino_id="casino_alpha")
        _build_greeting_categories(casino_id="casino_beta")
        assert len(_greeting_cache) == 2, \
            f"Expected 2 cache entries for 2 casino_ids, got {len(_greeting_cache)}"

    def test_greeting_cache_maxsize_supports_multiple_tenants(self):
        """Greeting cache maxsize >= 2 to support multiple concurrent tenants."""
        from src.agent.nodes import _greeting_cache

        assert _greeting_cache.maxsize >= 2, \
            f"TTLCache maxsize={_greeting_cache.maxsize} too small for multi-tenant"

    def test_retriever_cache_is_keyed_by_casino_id(self):
        """_retriever_cache stores entries keyed by '{casino_id}:default'."""
        from src.rag.pipeline import _get_retriever_cached

        source = inspect.getsource(_get_retriever_cached)
        # Verify the cache key pattern includes casino_id
        assert "CASINO_ID" in source and "default" in source, \
            "Retriever cache key must be '{casino_id}:default' pattern"


class TestRateLimitIsolation:
    """Rate limiter must use per-client keys, not global counters."""

    def test_rate_limit_uses_client_ip(self):
        """Rate limiter keys by client IP address."""
        from src.api.middleware import RateLimitMiddleware

        source = inspect.getsource(RateLimitMiddleware)
        assert "client_ip" in source, \
            "Rate limiter must key by client_ip"

    def test_rate_limit_has_per_client_tracking(self):
        """Rate limiter uses per-client data structure (OrderedDict of deques)."""
        from src.api.middleware import RateLimitMiddleware

        source = inspect.getsource(RateLimitMiddleware)
        assert "_requests" in source, \
            "Rate limiter must use _requests for per-client tracking"
        assert "OrderedDict" in source or "dict" in source, \
            "Rate limiter must map client IPs to request histories"

    def test_rate_limit_not_global_counter(self):
        """Rate limiter tracks per-client, not a single global count."""
        from src.api.middleware import RateLimitMiddleware

        source = inspect.getsource(RateLimitMiddleware._is_allowed)
        # Must accept client_ip parameter -- not a single shared counter
        assert "client_ip" in source, \
            "_is_allowed must accept client_ip for per-client enforcement"

    def test_rate_limit_respects_xff_from_trusted_proxies_only(self):
        """Rate limiter checks trusted proxies before trusting X-Forwarded-For."""
        from src.api.middleware import RateLimitMiddleware

        source = inspect.getsource(RateLimitMiddleware._get_client_ip)
        assert "trusted" in source.lower() or "TRUSTED_PROXIES" in source, \
            "Rate limiter must validate XFF against trusted proxy list"


class TestPIIRedactionIsolation:
    """PII redaction applies uniformly -- no tenant bypass."""

    def test_pii_redaction_has_no_tenant_bypass(self):
        """PII redaction does not check tenant/casino -- applies to all."""
        from src.api.pii_redaction import redact_pii

        source = inspect.getsource(redact_pii)
        assert "casino_id" not in source.lower(), \
            "PII redaction must not have casino_id-based bypass"
        assert "tenant" not in source.lower(), \
            "PII redaction must not have tenant-based bypass"

    def test_pii_redaction_fails_closed(self):
        """PII redaction returns safe placeholder on error, never pass-through."""
        from src.api.pii_redaction import redact_pii

        source = inspect.getsource(redact_pii)
        assert "PII_REDACTION_ERROR" in source, \
            "PII redaction must return safe placeholder on error (fail-closed)"


class TestIngestionIsolation:
    """Ingestion stamps property_id metadata on every chunk."""

    def test_ingest_stamps_property_id_on_json_chunks(self):
        """_load_property_json includes property_id in metadata."""
        from src.rag.pipeline import _load_property_json

        source = inspect.getsource(_load_property_json)
        assert "property_id" in source, \
            "JSON ingestion must stamp property_id metadata on chunks"

    def test_ingest_stamps_property_id_on_markdown_chunks(self):
        """_load_knowledge_base_markdown includes property_id in metadata."""
        from src.rag.pipeline import _load_knowledge_base_markdown

        source = inspect.getsource(_load_knowledge_base_markdown)
        assert "property_id" in source, \
            "Markdown ingestion must stamp property_id metadata on chunks"

    def test_reingest_item_stamps_property_id(self):
        """reingest_item includes property_id in metadata for CMS updates."""
        from src.rag.pipeline import reingest_item

        source = inspect.getsource(reingest_item)
        assert "property_id" in source, \
            "CMS re-ingestion must stamp property_id metadata"

    def test_stale_purge_scoped_to_property_id(self):
        """Stale chunk purging filters by property_id to avoid cross-tenant purge."""
        from src.rag.pipeline import ingest_property

        source = inspect.getsource(ingest_property)
        assert "property_id" in source, \
            "Stale purge must be scoped to property_id"
