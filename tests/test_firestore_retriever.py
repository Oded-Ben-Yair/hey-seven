"""Tests for Firestore vector search retriever and checkpointer factory."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# Pre-register mock modules for google.cloud.firestore so lazy imports work
_mock_vector_module = MagicMock()
_mock_base_vector_query_module = MagicMock()


@pytest.fixture(autouse=True)
def _mock_firestore_imports():
    """Mock google.cloud.firestore modules for all tests in this file."""
    with patch.dict(sys.modules, {
        "google.cloud.firestore": MagicMock(),
        "google.cloud.firestore_v1": MagicMock(),
        "google.cloud.firestore_v1.vector": _mock_vector_module,
        "google.cloud.firestore_v1.base_vector_query": _mock_base_vector_query_module,
    }):
        yield


class TestFirestoreRetriever:
    """Tests for FirestoreRetriever with mock Firestore client."""

    def _make_retriever(self, mock_client=None):
        """Create a FirestoreRetriever with optional mock client."""
        from src.rag.firestore_retriever import FirestoreRetriever

        embeddings = MagicMock()
        embeddings.embed_query.return_value = [0.1] * 768

        retriever = FirestoreRetriever(
            project="test-project",
            collection="test_collection",
            embeddings=embeddings,
        )
        if mock_client is not None:
            retriever._client = mock_client
        return retriever

    def _mock_doc_snapshot(self, data, doc_id="doc1"):
        """Create a mock Firestore document snapshot."""
        snap = MagicMock()
        snap.to_dict.return_value = data
        snap.id = doc_id
        return snap

    def test_retrieve_with_scores_returns_correct_format(self):
        """retrieve_with_scores returns list of (Document, float) tuples."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collection.return_value = mock_collection

        snapshot = self._mock_doc_snapshot({
            "content": "Steakhouse info",
            "metadata": {"category": "restaurants", "property_id": "mohegan_sun", "source": "data.json"},
            "distance": 0.2,
        })
        mock_vector_query = MagicMock()
        mock_vector_query.get.return_value = [snapshot]
        mock_collection.find_nearest.return_value = mock_vector_query

        retriever = self._make_retriever(mock_client)
        results = retriever.retrieve_with_scores("steakhouse")

        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert doc.page_content == "Steakhouse info"
        assert doc.metadata["category"] == "restaurants"
        assert score == pytest.approx(0.8)  # 1.0 - 0.2

    def test_retrieve_with_scores_filters_by_property_id(self):
        """retrieve_with_scores filters out docs with wrong property_id."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collection.return_value = mock_collection

        own_doc = self._mock_doc_snapshot({
            "content": "Our restaurant",
            "metadata": {"property_id": "mohegan_sun", "source": "data.json"},
            "distance": 0.1,
        })
        other_doc = self._mock_doc_snapshot({
            "content": "Other casino",
            "metadata": {"property_id": "other_casino", "source": "data.json"},
            "distance": 0.1,
        }, doc_id="doc2")
        mock_vector_query = MagicMock()
        mock_vector_query.get.return_value = [own_doc, other_doc]
        mock_collection.find_nearest.return_value = mock_vector_query

        retriever = self._make_retriever(mock_client)
        results = retriever.retrieve_with_scores("restaurant")

        assert len(results) == 1
        assert results[0][0].metadata["property_id"] == "mohegan_sun"

    def test_retrieve_with_scores_empty_results(self):
        """retrieve_with_scores returns empty list when no docs match."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collection.return_value = mock_collection

        mock_vector_query = MagicMock()
        mock_vector_query.get.return_value = []
        mock_collection.find_nearest.return_value = mock_vector_query

        retriever = self._make_retriever(mock_client)
        results = retriever.retrieve_with_scores("nonexistent")

        assert results == []

    def test_retrieve_convenience_method(self):
        """retrieve() returns list of Documents (no scores)."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collection.return_value = mock_collection

        snapshot = self._mock_doc_snapshot({
            "content": "Buffet info",
            "metadata": {"category": "restaurants", "property_id": "mohegan_sun"},
            "distance": 0.15,
        })
        mock_vector_query = MagicMock()
        mock_vector_query.get.return_value = [snapshot]
        mock_collection.find_nearest.return_value = mock_vector_query

        retriever = self._make_retriever(mock_client)
        docs = retriever.retrieve("buffet")

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "Buffet info"

    def test_retrieve_with_category_filter(self):
        """retrieve() filters by category when specified."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collection.return_value = mock_collection

        restaurant_doc = self._mock_doc_snapshot({
            "content": "Restaurant",
            "metadata": {"category": "restaurants", "property_id": "mohegan_sun"},
            "distance": 0.1,
        })
        entertainment_doc = self._mock_doc_snapshot({
            "content": "Show",
            "metadata": {"category": "entertainment", "property_id": "mohegan_sun"},
            "distance": 0.2,
        }, doc_id="doc2")
        mock_vector_query = MagicMock()
        mock_vector_query.get.return_value = [restaurant_doc, entertainment_doc]
        mock_collection.find_nearest.return_value = mock_vector_query

        retriever = self._make_retriever(mock_client)
        docs = retriever.retrieve("food", filter_category="restaurants")

        assert len(docs) == 1
        assert docs[0].metadata["category"] == "restaurants"

    def test_retrieve_with_scores_handles_embed_failure(self):
        """retrieve_with_scores returns empty on embedding failure."""
        from src.rag.firestore_retriever import FirestoreRetriever

        embeddings = MagicMock()
        embeddings.embed_query.side_effect = RuntimeError("API key invalid")

        retriever = FirestoreRetriever(
            project="test", collection="test", embeddings=embeddings,
        )
        results = retriever.retrieve_with_scores("test query")
        assert results == []

    def test_cosine_distance_to_similarity_conversion(self):
        """Cosine distance is correctly converted to similarity."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collection.return_value = mock_collection

        # distance=0 => similarity=1.0 (identical)
        snap_identical = self._mock_doc_snapshot({
            "content": "exact match",
            "metadata": {"property_id": "mohegan_sun"},
            "distance": 0.0,
        })
        # distance=1.0 => similarity=0.0 (orthogonal)
        snap_orthogonal = self._mock_doc_snapshot({
            "content": "orthogonal",
            "metadata": {"property_id": "mohegan_sun"},
            "distance": 1.0,
        }, doc_id="doc2")
        mock_vector_query = MagicMock()
        mock_vector_query.get.return_value = [snap_identical, snap_orthogonal]
        mock_collection.find_nearest.return_value = mock_vector_query

        retriever = self._make_retriever(mock_client)
        results = retriever.retrieve_with_scores("test")

        assert results[0][1] == pytest.approx(1.0)
        assert results[1][1] == pytest.approx(0.0)


class TestGetCheckpointer:
    """Tests for the checkpointer factory."""

    def test_returns_memory_saver_for_chroma(self):
        """get_checkpointer returns MemorySaver when VECTOR_DB=chroma."""
        from src.agent.memory import get_checkpointer

        get_checkpointer.cache_clear()
        with patch.dict(os.environ, {"VECTOR_DB": "chroma"}):
            from src.config import get_settings
            get_settings.cache_clear()
            cp = get_checkpointer()
            from langgraph.checkpoint.memory import MemorySaver
            assert isinstance(cp, MemorySaver)
            get_checkpointer.cache_clear()
            get_settings.cache_clear()

    def test_returns_memory_saver_by_default(self):
        """get_checkpointer returns MemorySaver when VECTOR_DB not set."""
        from src.agent.memory import get_checkpointer

        get_checkpointer.cache_clear()
        cp = get_checkpointer()
        from langgraph.checkpoint.memory import MemorySaver
        assert isinstance(cp, MemorySaver)
        get_checkpointer.cache_clear()

    def test_firestore_saver_import_attempted(self):
        """get_checkpointer attempts FirestoreSaver when VECTOR_DB=firestore."""
        from src.agent.memory import get_checkpointer

        get_checkpointer.cache_clear()
        with patch.dict(os.environ, {"VECTOR_DB": "firestore", "FIRESTORE_PROJECT": "test-proj"}):
            from src.config import get_settings
            get_settings.cache_clear()
            # FirestoreSaver import will fail (no real GCP) but should fall back
            cp = get_checkpointer()
            from langgraph.checkpoint.memory import MemorySaver
            assert isinstance(cp, MemorySaver)  # Fallback
            get_checkpointer.cache_clear()
            get_settings.cache_clear()


class TestGetRetrieverFactory:
    """Tests for get_retriever() factory with VECTOR_DB config."""

    def test_returns_chroma_retriever_by_default(self):
        """get_retriever returns CasinoKnowledgeRetriever when VECTOR_DB=chroma."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, get_retriever

        get_retriever.cache_clear()
        retriever = get_retriever()
        assert isinstance(retriever, CasinoKnowledgeRetriever)
        get_retriever.cache_clear()

    def test_returns_firestore_retriever_when_configured(self):
        """get_retriever returns FirestoreRetriever when VECTOR_DB=firestore."""
        from src.rag.pipeline import get_retriever

        get_retriever.cache_clear()
        with patch.dict(os.environ, {
            "VECTOR_DB": "firestore",
            "FIRESTORE_PROJECT": "test-proj",
            "FIRESTORE_COLLECTION": "kb",
        }):
            from src.config import get_settings
            get_settings.cache_clear()

            with patch("src.rag.pipeline.get_embeddings", return_value=MagicMock()):
                retriever = get_retriever()

            from src.rag.firestore_retriever import FirestoreRetriever
            assert isinstance(retriever, FirestoreRetriever)

            get_retriever.cache_clear()
            get_settings.cache_clear()


class TestEmbeddingsTaskType:
    """Tests for enhanced embeddings with task_type parameter."""

    def test_get_embeddings_without_task_type(self):
        """get_embeddings() without task_type uses default."""
        from src.rag.embeddings import get_embeddings

        with patch("src.rag.embeddings.GoogleGenerativeAIEmbeddings") as mock_cls:
            get_embeddings.cache_clear()
            mock_cls.return_value = "default_embeddings"
            result = get_embeddings()
            assert result == "default_embeddings"
            call_kwargs = mock_cls.call_args.kwargs
            assert "task_type" not in call_kwargs
            get_embeddings.cache_clear()

    def test_get_embeddings_with_retrieval_query(self):
        """get_embeddings(task_type=RETRIEVAL_QUERY) passes task_type."""
        from src.rag.embeddings import get_embeddings

        with patch("src.rag.embeddings.GoogleGenerativeAIEmbeddings") as mock_cls:
            get_embeddings.cache_clear()
            mock_cls.return_value = "query_embeddings"
            result = get_embeddings(task_type="RETRIEVAL_QUERY")
            assert result == "query_embeddings"
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["task_type"] == "RETRIEVAL_QUERY"
            get_embeddings.cache_clear()

    def test_get_embeddings_different_task_types_cached_separately(self):
        """Different task_type values produce separate cached instances."""
        from src.rag.embeddings import get_embeddings

        with patch("src.rag.embeddings.GoogleGenerativeAIEmbeddings") as mock_cls:
            get_embeddings.cache_clear()
            mock_cls.side_effect = lambda **kwargs: f"embed_{kwargs.get('task_type', 'default')}"
            r1 = get_embeddings()
            r2 = get_embeddings(task_type="RETRIEVAL_QUERY")
            r3 = get_embeddings()  # Should be cached
            assert r1 == "embed_default"
            assert r2 == "embed_RETRIEVAL_QUERY"
            assert r1 is r3  # Same cached instance
            assert mock_cls.call_count == 2  # Only 2 unique calls
            get_embeddings.cache_clear()


class TestCircuitBreakerEnhancements:
    """Tests for enhanced circuit breaker (rolling window, half_open, config)."""

    def test_circuit_breaker_config_dataclass(self):
        """CircuitBreakerConfig is a frozen dataclass with defaults."""
        from src.agent.circuit_breaker import CircuitBreakerConfig

        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.cooldown_seconds == 60.0
        assert config.rolling_window_seconds == 300.0

    def test_circuit_breaker_config_custom_values(self):
        """CircuitBreakerConfig accepts custom values."""
        from src.agent.circuit_breaker import CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=3,
            cooldown_seconds=30.0,
            rolling_window_seconds=120.0,
        )
        assert config.failure_threshold == 3
        assert config.cooldown_seconds == 30.0
        assert config.rolling_window_seconds == 120.0

    def test_circuit_breaker_config_is_frozen(self):
        """CircuitBreakerConfig is immutable (frozen)."""
        from src.agent.circuit_breaker import CircuitBreakerConfig

        config = CircuitBreakerConfig()
        with pytest.raises(AttributeError):
            config.failure_threshold = 10

    def test_circuit_breaker_from_config(self):
        """CircuitBreaker can be created from a CircuitBreakerConfig."""
        from src.agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=10.0)
        cb = CircuitBreaker(config=config)
        assert cb._failure_threshold == 3
        assert cb._cooldown_seconds == 10.0

    @pytest.mark.asyncio
    async def test_rolling_window_expiry(self):
        """Failures outside the rolling window are pruned and don't count."""
        import time
        from unittest.mock import patch as mock_patch

        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0, rolling_window_seconds=1.0)

        # Record 2 failures
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == "closed"
        assert cb.failure_count == 2

        # Wait for rolling window to expire
        with mock_patch.object(time, "monotonic", side_effect=[
            time.monotonic() + 2.0,  # _prune_old_failures check
            time.monotonic() + 2.0,  # failure_count property
        ]):
            assert cb.failure_count == 0  # Old failures pruned

    @pytest.mark.asyncio
    async def test_half_open_allows_one_probe(self):
        """In half_open state, allow_request permits exactly one probe."""
        from src.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.01)

        # Trip the breaker
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == "open"

        # Wait for cooldown
        import asyncio
        await asyncio.sleep(0.02)
        assert cb.state == "half_open"

        # First request in half_open: allowed (probe)
        assert await cb.allow_request() is True
        # Second request in half_open: blocked (probe in progress)
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

        # Probe succeeds
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

        # Probe fails
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
