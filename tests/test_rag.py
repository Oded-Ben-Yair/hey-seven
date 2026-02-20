"""Tests for the RAG pipeline (ingestion, retrieval, metadata, formatters, embeddings)."""

from unittest.mock import patch

import pytest

# Skip all tests if chromadb is not installed
chromadb = pytest.importorskip("chromadb")


class FakeEmbeddings:
    """Simple deterministic embeddings for testing (no API key needed)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate simple hash-based embeddings."""
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Generate a query embedding."""
        return self._hash_embed(text)

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        """Produce a 384-dim embedding from text hash."""
        import hashlib

        h = hashlib.sha384(text.encode()).digest()
        return [float(b) / 255.0 for b in h]


@pytest.fixture(autouse=True)
def _mock_embeddings():
    """Replace Google embeddings with fake deterministic embeddings."""
    with patch("src.rag.pipeline.get_embeddings", return_value=FakeEmbeddings()):
        yield


@pytest.fixture(autouse=True)
def _reset_retriever_singleton():
    """Reset the global retriever singleton between tests.

    Uses clear_retriever_cache() on the TTL-cache-based retriever singleton,
    consistent with conftest.py's _clear_singleton_caches fixture.
    """
    from src.rag.pipeline import clear_retriever_cache

    clear_retriever_cache()
    yield
    clear_retriever_cache()


@pytest.fixture
def no_kb_dir(tmp_path):
    """Return a path to a nonexistent knowledge-base directory.

    Prevents tests from accidentally ingesting the real knowledge-base/
    markdown files when only testing JSON ingestion.
    """
    return str(tmp_path / "no_kb")


class TestIngestion:
    def test_ingestion_creates_collection(self, test_property_file, tmp_path, no_kb_dir):
        """Data loads and indexes into ChromaDB without error."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_test")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        assert vectorstore is not None
        collection = vectorstore._collection
        assert collection.count() > 0

    def test_ingestion_indexes_all_categories(self, test_property_file, tmp_path, no_kb_dir):
        """All non-property categories are indexed."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_test2")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        results = vectorstore._collection.get(include=["metadatas"])
        categories = {m.get("category") for m in results["metadatas"] if m}
        # At minimum restaurants and entertainment should be present
        assert len(categories) >= 2


class TestRetrieval:
    def test_retrieval_returns_results(self, test_property_file, tmp_path, no_kb_dir):
        """Query for 'steakhouse' returns results from the vectorstore."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        persist_dir = str(tmp_path / "chroma_retrieval")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)
        docs = retriever.retrieve("steakhouse")
        assert len(docs) > 0

    def test_empty_query_returns_gracefully(self, test_property_file, tmp_path, no_kb_dir):
        """Empty query does not raise an exception."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        persist_dir = str(tmp_path / "chroma_empty")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)
        # Should not raise
        docs = retriever.retrieve("")
        assert isinstance(docs, list)


class TestCategoryMetadata:
    def test_documents_have_category_metadata(self, test_property_file, tmp_path, no_kb_dir):
        """Each indexed document carries its source category in metadata."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_meta")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        results = vectorstore._collection.get(include=["metadatas"])
        for metadata in results["metadatas"]:
            assert "category" in metadata, f"Document missing 'category' metadata: {metadata}"

    def test_category_values_match_source(self, test_property_file, tmp_path):
        """Category metadata values match the JSON keys."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_cat_values")
        # Use nonexistent kb_dir to test JSON categories only
        vectorstore = ingest_property(
            test_property_file,
            persist_dir=persist_dir,
            knowledge_base_dir=str(tmp_path / "no_kb"),
        )
        results = vectorstore._collection.get(include=["metadatas"])
        categories = {m["category"] for m in results["metadatas"] if "category" in m}
        expected = {"restaurants", "entertainment", "hotel", "gaming", "faq", "property", "amenities", "promotions"}
        # All discovered categories should be a subset of expected
        assert categories.issubset(expected), f"Unexpected categories: {categories - expected}"


class TestCategoryFilter:
    def test_filter_by_category(self, test_property_file, tmp_path, no_kb_dir):
        """Retrieval with filter_category only returns matching category."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        persist_dir = str(tmp_path / "chroma_filter")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)
        docs = retriever.retrieve("food", filter_category="restaurants")
        for doc in docs:
            assert doc.metadata.get("category") == "restaurants"


class TestNestedDictFlattening:
    def test_hotel_nested_dict_produces_documents(self, test_property_file, tmp_path, no_kb_dir):
        """Hotel data (nested dict with towers/room_types) produces indexed docs."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_nested")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        results = vectorstore._collection.get(include=["metadatas"])
        hotel_docs = [m for m in results["metadatas"] if m.get("category") == "hotel"]
        assert len(hotel_docs) >= 2  # at least towers + room_types


class TestFlattenNestedDictUnit:
    """Direct unit tests for _flatten_nested_dict edge cases."""

    def test_list_sub_items_become_separate_dicts(self):
        """List values are flattened into separate dicts."""
        from src.rag.pipeline import _flatten_nested_dict

        data = {
            "towers": [
                {"name": "Sky Tower", "floors": 34},
                {"name": "Earth Tower", "floors": 20},
            ]
        }
        result = _flatten_nested_dict(data, "hotel")
        assert len(result) == 2
        assert result[0]["name"] == "Sky Tower"
        assert result[1]["name"] == "Earth Tower"

    def test_sub_dict_becomes_flat_item(self):
        """Nested dicts get flattened with key as name."""
        from src.rag.pipeline import _flatten_nested_dict

        data = {
            "poker_room": {"tables": 20, "stakes": "$1/$2 to $100/$200"},
            "sportsbook": {"screens": 50, "hours": "24/7"},
        }
        result = _flatten_nested_dict(data, "gaming")
        assert len(result) == 2
        names = {item["name"] for item in result}
        assert "Poker Room" in names
        assert "Sportsbook" in names

    def test_scalar_values_produce_overview(self):
        """Top-level scalars are collected into an overview document."""
        from src.rag.pipeline import _flatten_nested_dict

        data = {"casino_size_sqft": 300000, "slot_machines": 5000, "table_games": 300}
        result = _flatten_nested_dict(data, "gaming")
        assert len(result) == 1
        assert "Gaming Overview" in result[0]["name"]
        assert "300000" in result[0]["description"]

    def test_mixed_structure(self):
        """Mix of lists, dicts, and scalars handles all branches."""
        from src.rag.pipeline import _flatten_nested_dict

        data = {
            "total_rooms": 1200,
            "towers": [{"name": "Sky Tower"}],
            "spa": {"open": "9 AM", "close": "9 PM"},
        }
        result = _flatten_nested_dict(data, "hotel")
        assert len(result) >= 3  # overview + tower + spa

    def test_empty_dict_returns_original(self):
        """Empty dict returns itself wrapped in a list."""
        from src.rag.pipeline import _flatten_nested_dict

        result = _flatten_nested_dict({}, "empty")
        assert len(result) == 1
        assert result[0] == {}

    def test_list_items_without_name_get_key_as_name(self):
        """Sub-list items without a 'name' key get the parent key as name."""
        from src.rag.pipeline import _flatten_nested_dict

        data = {"room_types": [{"size": "400 sqft", "rate": "$199/night"}]}
        result = _flatten_nested_dict(data, "hotel")
        assert len(result) == 1
        assert result[0]["name"] == "Room Types"

    def test_scalar_list_items(self):
        """Non-dict list items become scalar parts."""
        from src.rag.pipeline import _flatten_nested_dict

        data = {"features": ["pool", "gym", "spa"]}
        result = _flatten_nested_dict(data, "amenities")
        # Scalar list items go into scalar_parts overview
        assert len(result) >= 1


class TestMissingDataFile:
    def test_missing_file_returns_empty(self, tmp_path, no_kb_dir):
        """Ingest with non-existent JSON file AND no KB returns None (no crash)."""
        from src.rag.pipeline import ingest_property

        result = ingest_property(
            str(tmp_path / "nonexistent.json"),
            persist_dir=str(tmp_path / "chroma_missing"),
            knowledge_base_dir=no_kb_dir,
        )
        assert result is None


class TestFilterByRelevance:
    """Tests for the _filter_by_relevance score filtering in tools.py."""

    def test_keeps_above_threshold(self):
        """Documents with score >= threshold are kept."""
        from langchain_core.documents import Document

        from src.agent.tools import _filter_by_relevance

        results = [
            (Document(page_content="high"), 0.9),
            (Document(page_content="medium"), 0.5),
            (Document(page_content="low"), 0.1),
        ]
        filtered = _filter_by_relevance(results, min_score=0.3)
        assert len(filtered) == 2
        assert filtered[0][1] == 0.9
        assert filtered[1][1] == 0.5

    def test_removes_below_threshold(self):
        """Documents with score < threshold are removed."""
        from langchain_core.documents import Document

        from src.agent.tools import _filter_by_relevance

        results = [
            (Document(page_content="low1"), 0.1),
            (Document(page_content="low2"), 0.2),
        ]
        filtered = _filter_by_relevance(results, min_score=0.3)
        assert len(filtered) == 0

    def test_keeps_exact_threshold(self):
        """Document with score exactly at threshold is kept (>= not >)."""
        from langchain_core.documents import Document

        from src.agent.tools import _filter_by_relevance

        results = [(Document(page_content="exact"), 0.3)]
        filtered = _filter_by_relevance(results, min_score=0.3)
        assert len(filtered) == 1

    def test_empty_input(self):
        """Empty results list returns empty."""
        from src.agent.tools import _filter_by_relevance

        assert _filter_by_relevance([], min_score=0.3) == []

    def test_all_above_threshold(self):
        """All documents above threshold are preserved in order."""
        from langchain_core.documents import Document

        from src.agent.tools import _filter_by_relevance

        results = [
            (Document(page_content="a"), 0.95),
            (Document(page_content="b"), 0.80),
            (Document(page_content="c"), 0.60),
        ]
        filtered = _filter_by_relevance(results, min_score=0.3)
        assert len(filtered) == 3


class TestReciprocalRankFusion:
    """Tests for the RRF reranking function in tools.py."""

    def test_single_list_preserves_order(self):
        """Single result list is returned in original order (top_k)."""
        from langchain_core.documents import Document

        from src.rag.reranking import rerank_by_rrf

        results = [
            (Document(page_content="first"), 0.9),
            (Document(page_content="second"), 0.7),
            (Document(page_content="third"), 0.5),
        ]
        fused = rerank_by_rrf([results], top_k=3)
        assert len(fused) == 3
        assert fused[0][0].page_content == "first"

    def test_duplicate_across_lists_gets_boosted(self):
        """Document appearing in both lists gets higher RRF score."""
        from langchain_core.documents import Document

        from src.rag.reranking import rerank_by_rrf

        list_a = [
            (Document(page_content="shared"), 0.8),
            (Document(page_content="only_a"), 0.9),
        ]
        list_b = [
            (Document(page_content="only_b"), 0.85),
            (Document(page_content="shared"), 0.7),
        ]
        fused = rerank_by_rrf([list_a, list_b], top_k=3)
        # "shared" appears in both lists and should rank first
        assert fused[0][0].page_content == "shared"

    def test_respects_top_k(self):
        """RRF returns at most top_k results."""
        from langchain_core.documents import Document

        from src.rag.reranking import rerank_by_rrf

        results = [(Document(page_content=f"doc{i}"), 0.5) for i in range(10)]
        fused = rerank_by_rrf([results], top_k=3)
        assert len(fused) == 3

    def test_empty_input(self):
        """Empty result lists returns empty."""
        from src.rag.reranking import rerank_by_rrf

        assert rerank_by_rrf([], top_k=5) == []

    def test_keeps_best_score_for_duplicates(self):
        """When a doc appears in multiple lists, keeps the highest original score."""
        from langchain_core.documents import Document

        from src.rag.reranking import rerank_by_rrf

        list_a = [(Document(page_content="doc"), 0.6)]
        list_b = [(Document(page_content="doc"), 0.9)]
        fused = rerank_by_rrf([list_a, list_b], top_k=1)
        assert fused[0][1] == 0.9  # Higher score kept

    def test_different_source_not_merged(self):
        """Same content from different sources stays separate (hash includes source)."""
        from langchain_core.documents import Document

        from src.rag.reranking import rerank_by_rrf

        doc_a = Document(page_content="same text", metadata={"source": "restaurants"})
        doc_b = Document(page_content="same text", metadata={"source": "entertainment"})
        list_a = [(doc_a, 0.8)]
        list_b = [(doc_b, 0.7)]
        fused = rerank_by_rrf([list_a, list_b], top_k=5)
        assert len(fused) == 2, "Same content from different sources must not merge"


class TestChunkCount:
    def test_chunk_count_greater_or_equal_to_doc_count(self, test_property_file, tmp_path, no_kb_dir):
        """Total chunks >= total input documents (chunking splits, never removes)."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_chunks")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        collection = vectorstore._collection
        # We have at least restaurants(2) + entertainment(1) + hotel + gaming + faq = 5+
        assert collection.count() >= 5


class TestPropertyIdIsolation:
    """Tests for cross-property leakage prevention via property_id metadata filter."""

    def test_retrieve_with_scores_filters_by_property_id(self, test_property_file, tmp_path, no_kb_dir):
        """retrieve_with_scores only returns docs matching the configured PROPERTY_NAME."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        persist_dir = str(tmp_path / "chroma_prop_id")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)

        # Normal retrieval returns results for the configured property
        results = retriever.retrieve_with_scores("steakhouse")
        assert len(results) > 0

        # All results carry the correct property_id
        for doc, score in results:
            assert doc.metadata.get("property_id") == "mohegan_sun"

    def test_cross_property_returns_empty(self, test_property_file, tmp_path, no_kb_dir):
        """Queries with a different PROPERTY_NAME return no results (isolation)."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        persist_dir = str(tmp_path / "chroma_cross_prop")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)

        # Change property name — simulates a different tenant querying the same collection
        with patch("src.rag.pipeline.get_settings") as mock_settings:
            mock_settings.return_value.PROPERTY_NAME = "Rival Casino"
            results = retriever.retrieve_with_scores("steakhouse")
            assert len(results) == 0, "Cross-property query should return empty"


class TestSearchKnowledgeBaseUnit:
    """Unit tests for search_knowledge_base with mocked retriever."""

    def test_returns_results_with_rrf(self):
        """search_knowledge_base returns fused results from RRF."""
        from unittest.mock import MagicMock

        from langchain_core.documents import Document

        mock_retriever = MagicMock()
        doc = Document(page_content="Steakhouse info", metadata={"category": "restaurants"})
        mock_retriever.retrieve_with_scores.return_value = [(doc, 0.9)]

        with patch("src.agent.tools.get_retriever", return_value=mock_retriever):
            from src.agent.tools import search_knowledge_base

            results = search_knowledge_base("steakhouse")
            assert len(results) >= 1
            assert results[0]["content"] == "Steakhouse info"

    def test_returns_empty_on_value_error(self):
        """search_knowledge_base returns empty on ValueError."""
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_scores.side_effect = ValueError("bad value")

        with patch("src.agent.tools.get_retriever", return_value=mock_retriever):
            from src.agent.tools import search_knowledge_base

            results = search_knowledge_base("test")
            assert results == []

    def test_returns_empty_on_generic_exception(self):
        """search_knowledge_base returns empty on generic exception."""
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_scores.side_effect = RuntimeError("connection failed")

        with patch("src.agent.tools.get_retriever", return_value=mock_retriever):
            from src.agent.tools import search_knowledge_base

            results = search_knowledge_base("test")
            assert results == []


class TestSearchHoursUnit:
    """Unit tests for search_hours with mocked retriever."""

    def test_returns_results_with_rrf(self):
        """search_hours returns fused results from RRF."""
        from unittest.mock import MagicMock

        from langchain_core.documents import Document

        mock_retriever = MagicMock()
        doc = Document(page_content="Pool open 9am-9pm", metadata={"category": "amenities"})
        mock_retriever.retrieve_with_scores.return_value = [(doc, 0.85)]

        with patch("src.agent.tools.get_retriever", return_value=mock_retriever):
            from src.agent.tools import search_hours

            results = search_hours("pool hours")
            assert len(results) >= 1
            assert "9am-9pm" in results[0]["content"]

    def test_returns_empty_on_type_error(self):
        """search_hours returns empty on TypeError."""
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_scores.side_effect = TypeError("type issue")

        with patch("src.agent.tools.get_retriever", return_value=mock_retriever):
            from src.agent.tools import search_hours

            results = search_hours("pool hours")
            assert results == []

    def test_returns_empty_on_generic_exception(self):
        """search_hours returns empty on generic exception."""
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_scores.side_effect = RuntimeError("network error")

        with patch("src.agent.tools.get_retriever", return_value=mock_retriever):
            from src.agent.tools import search_hours

            results = search_hours("pool hours")
            assert results == []


class TestFormatters:
    """Direct unit tests for pipeline.py formatter functions."""

    def test_format_restaurant_full(self):
        """Restaurant formatter includes all fields when present."""
        from src.rag.pipeline import _format_restaurant

        item = {
            "name": "Bobby's Burgers",
            "cuisine": "American",
            "price_range": "$$",
            "location": "Casino of the Earth",
            "hours": "11am-11pm",
            "description": "Gourmet burgers and shakes.",
            "dress_code": "Casual",
            "reservations": "Walk-in only",
        }
        text = _format_restaurant(item)
        assert "Bobby's Burgers" in text
        assert "American cuisine" in text
        assert "$$" in text
        assert "Casino of the Earth" in text
        assert "11am-11pm" in text
        assert "Gourmet burgers" in text
        assert "Casual" in text
        assert "Walk-in only" in text

    def test_format_restaurant_minimal(self):
        """Restaurant formatter handles minimal data without error."""
        from src.rag.pipeline import _format_restaurant

        text = _format_restaurant({})
        assert "Unknown" in text

    def test_format_entertainment_full(self):
        """Entertainment formatter includes all fields."""
        from src.rag.pipeline import _format_entertainment

        item = {
            "name": "Mohegan Sun Arena",
            "type": "Concert Venue",
            "description": "10,000 seat arena.",
            "venue": "Main Complex",
            "capacity": "10000",
            "schedule": "Varies by event",
        }
        text = _format_entertainment(item)
        assert "Mohegan Sun Arena" in text
        assert "Concert Venue" in text
        assert "10,000 seat arena" in text

    def test_format_hotel_with_list_amenities(self):
        """Hotel formatter joins list amenities."""
        from src.rag.pipeline import _format_hotel

        item = {
            "name": "Sky Tower Suite",
            "description": "Luxury suite.",
            "size": "800 sqft",
            "rate": "$399/night",
            "amenities": ["Mini bar", "Jacuzzi", "Mountain view"],
        }
        text = _format_hotel(item)
        assert "Sky Tower Suite" in text
        assert "Luxury suite" in text
        assert "800 sqft" in text
        assert "Mini bar, Jacuzzi, Mountain view" in text

    def test_format_hotel_with_string_amenities(self):
        """Hotel formatter handles string amenities."""
        from src.rag.pipeline import _format_hotel

        item = {"name": "Standard Room", "amenities": "WiFi, TV, AC"}
        text = _format_hotel(item)
        assert "WiFi, TV, AC" in text

    def test_format_generic_covers_all_value_types(self):
        """Generic formatter includes string, int, float, and list values."""
        from src.rag.pipeline import _format_generic

        item = {
            "name": "Spa",
            "description": "Full-service spa.",
            "rooms": 12,
            "rating": 4.8,
            "services": ["Massage", "Facial", "Sauna"],
        }
        text = _format_generic(item)
        assert "Spa" in text
        assert "Full-service spa" in text
        assert "12" in text
        assert "4.8" in text
        assert "Massage, Facial, Sauna" in text


class TestChunking:
    """Direct unit tests for _chunk_documents."""

    def test_chunk_preserves_metadata(self):
        """Chunking preserves all metadata fields and adds chunk_index."""
        from src.rag.pipeline import _chunk_documents

        docs = [{"content": "A " * 500, "metadata": {"category": "test", "source": "file.json"}}]
        chunks = _chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 2  # Should split
        for chunk in chunks:
            assert chunk["metadata"]["category"] == "test"
            assert "chunk_index" in chunk["metadata"]

    def test_short_doc_stays_single_chunk(self):
        """Document shorter than chunk_size remains a single chunk."""
        from src.rag.pipeline import _chunk_documents

        docs = [{"content": "Short text", "metadata": {"category": "x"}}]
        chunks = _chunk_documents(docs, chunk_size=800, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["chunk_index"] == 0


class TestRetrieverNoVectorstore:
    """CasinoKnowledgeRetriever returns empty when vectorstore is None."""

    def test_retrieve_returns_empty(self):
        from src.rag.pipeline import CasinoKnowledgeRetriever

        retriever = CasinoKnowledgeRetriever()
        assert retriever.retrieve("test") == []

    def test_retrieve_with_scores_returns_empty(self):
        from src.rag.pipeline import CasinoKnowledgeRetriever

        retriever = CasinoKnowledgeRetriever()
        assert retriever.retrieve_with_scores("test") == []


class TestEmbeddings:
    """Tests for the embeddings module."""

    def test_get_embeddings_returns_instance(self):
        """get_embeddings returns a GoogleGenerativeAIEmbeddings instance."""
        from src.rag.embeddings import get_embeddings

        with patch("src.rag.embeddings.GoogleGenerativeAIEmbeddings") as mock_cls:
            get_embeddings.cache_clear()
            mock_cls.return_value = "mock_embeddings"
            result = get_embeddings()
            assert result == "mock_embeddings"
            mock_cls.assert_called_once()
            get_embeddings.cache_clear()

    def test_get_embeddings_uses_config_model(self):
        """get_embeddings passes EMBEDDING_MODEL from settings."""
        from src.rag.embeddings import get_embeddings

        with patch("src.rag.embeddings.GoogleGenerativeAIEmbeddings") as mock_cls:
            get_embeddings.cache_clear()
            get_embeddings()
            call_kwargs = mock_cls.call_args
            assert "model" in call_kwargs.kwargs or len(call_kwargs.args) > 0
            get_embeddings.cache_clear()

    def test_get_embeddings_is_cached(self):
        """get_embeddings returns the same instance on second call (lru_cache)."""
        from src.rag.embeddings import get_embeddings

        with patch("src.rag.embeddings.GoogleGenerativeAIEmbeddings") as mock_cls:
            get_embeddings.cache_clear()
            r1 = get_embeddings()
            r2 = get_embeddings()
            assert r1 is r2
            assert mock_cls.call_count == 1
            get_embeddings.cache_clear()


class TestLoadPropertyJsonEdgeCases:
    """Edge cases for _load_property_json."""

    def test_list_format_json(self, tmp_path):
        """JSON file with top-level list is handled."""
        import json as json_mod

        from src.rag.pipeline import _load_property_json

        data = [{"name": "Buffet", "description": "All-you-can-eat."}]
        path = tmp_path / "list_data.json"
        path.write_text(json_mod.dumps(data))

        docs = _load_property_json(str(path))
        assert len(docs) >= 1
        assert docs[0]["metadata"]["category"] == "general"

    def test_scalar_category_value(self, tmp_path):
        """Category with scalar (non-list, non-dict) value is handled."""
        import json as json_mod

        from src.rag.pipeline import _load_property_json

        data = {"version": "2.0"}
        path = tmp_path / "scalar.json"
        path.write_text(json_mod.dumps(data))

        docs = _load_property_json(str(path))
        assert len(docs) >= 1

    def test_string_list_item(self, tmp_path):
        """Category with list of strings (not dicts) is handled."""
        import json as json_mod

        from src.rag.pipeline import _load_property_json

        data = {"features": ["Pool", "Gym", "Spa"]}
        path = tmp_path / "strings.json"
        path.write_text(json_mod.dumps(data))

        docs = _load_property_json(str(path))
        assert len(docs) >= 1


class TestMarkdownIngestion:
    """Tests for _load_knowledge_base_markdown."""

    def test_loads_markdown_files(self, tmp_path):
        """Markdown files are loaded and split by heading."""
        from src.rag.pipeline import _load_knowledge_base_markdown

        kb_dir = tmp_path / "knowledge-base" / "regulations"
        kb_dir.mkdir(parents=True)
        md_file = kb_dir / "state-requirements.md"
        md_file.write_text(
            "# State Requirements\n\n## Nevada\nGaming rules for NV.\n\n"
            "## New Jersey\nGaming rules for NJ.\n"
        )

        docs = _load_knowledge_base_markdown(str(tmp_path / "knowledge-base"))
        assert len(docs) >= 2
        categories = {d["metadata"]["category"] for d in docs}
        assert "regulations" in categories
        # All docs should have doc_type=markdown
        for doc in docs:
            assert doc["metadata"]["doc_type"] == "markdown"

    def test_missing_directory_returns_empty(self, tmp_path):
        """Non-existent directory returns empty list."""
        from src.rag.pipeline import _load_knowledge_base_markdown

        docs = _load_knowledge_base_markdown(str(tmp_path / "nonexistent"))
        assert docs == []

    def test_empty_markdown_skipped(self, tmp_path):
        """Empty markdown files are skipped."""
        from src.rag.pipeline import _load_knowledge_base_markdown

        kb_dir = tmp_path / "knowledge-base" / "empty"
        kb_dir.mkdir(parents=True)
        (kb_dir / "empty.md").write_text("")

        docs = _load_knowledge_base_markdown(str(tmp_path / "knowledge-base"))
        assert docs == []

    def test_markdown_metadata_has_property_id(self, tmp_path):
        """Markdown chunks carry property_id for multi-tenant isolation."""
        from src.rag.pipeline import _load_knowledge_base_markdown

        kb_dir = tmp_path / "knowledge-base" / "casino-operations"
        kb_dir.mkdir(parents=True)
        (kb_dir / "comp-system.md").write_text("# Comp System\n\nADT formula details.")

        docs = _load_knowledge_base_markdown(str(tmp_path / "knowledge-base"))
        assert len(docs) >= 1
        assert docs[0]["metadata"]["property_id"] == "mohegan_sun"
        assert docs[0]["metadata"]["source"] == "comp-system.md"
        assert docs[0]["metadata"]["category"] == "casino_operations"

    def test_ingest_property_includes_markdown(self, tmp_path):
        """ingest_property ingests both JSON and markdown documents."""
        import json as json_mod

        from src.rag.pipeline import ingest_property

        # Create minimal JSON
        json_file = tmp_path / "test.json"
        json_file.write_text(json_mod.dumps({
            "restaurants": [{"name": "Test Grill", "cuisine": "American"}]
        }))

        # Create minimal markdown
        kb_dir = tmp_path / "knowledge-base" / "regulations"
        kb_dir.mkdir(parents=True)
        (kb_dir / "rules.md").write_text("# Rules\n\nSelf-exclusion rules apply.")

        persist_dir = str(tmp_path / "chroma_md")
        vectorstore = ingest_property(
            str(json_file),
            persist_dir=persist_dir,
            knowledge_base_dir=str(tmp_path / "knowledge-base"),
        )
        assert vectorstore is not None
        collection = vectorstore._collection
        results = collection.get(include=["metadatas"])
        doc_types = {m.get("doc_type") for m in results["metadatas"]}
        # Should have both JSON (no doc_type) and markdown docs
        assert "markdown" in doc_types
        assert None in doc_types or len(results["ids"]) > 1

    def test_schema_version_replaces_ingestion_version(self, tmp_path):
        """Metadata uses _schema_version instead of old ingestion_version."""
        import json as json_mod

        from src.rag.pipeline import _load_property_json

        data = {"faq": [{"question": "Test?", "answer": "Yes"}]}
        path = tmp_path / "test.json"
        path.write_text(json_mod.dumps(data))

        docs = _load_property_json(str(path))
        assert len(docs) >= 1
        meta = docs[0]["metadata"]
        assert "_schema_version" in meta
        assert "ingestion_version" not in meta


class TestNewFormatters:
    """Tests for category-specific formatters added in R7."""

    def test_format_faq_qa_pair(self):
        """FAQ formatter leads with question for better embedding alignment."""
        from src.rag.pipeline import _format_faq

        item = {"question": "Is there free parking?", "answer": "Yes, free self-parking available."}
        text = _format_faq(item)
        assert text.startswith("Is there free parking?")
        assert "free self-parking" in text

    def test_format_faq_fallback_to_generic(self):
        """FAQ formatter falls back to generic for non-Q&A items."""
        from src.rag.pipeline import _format_faq

        item = {"name": "General Info", "description": "Some info"}
        text = _format_faq(item)
        assert "General Info" in text

    def test_format_gaming_boolean_values(self):
        """Gaming formatter renders booleans as Available/Not Available."""
        from src.rag.pipeline import _format_gaming

        item = {"name": "Main Casino", "poker_room": True, "sportsbook": False}
        text = _format_gaming(item)
        assert "Poker Room: Available" in text
        assert "Sportsbook: Not Available" in text

    def test_format_gaming_numeric_values(self):
        """Gaming formatter comma-formats large numbers."""
        from src.rag.pipeline import _format_gaming

        item = {"name": "Floor Stats", "slot_machines": 5000, "casino_size_sqft": 364000}
        text = _format_gaming(item)
        assert "5,000" in text
        assert "364,000 sq ft" in text

    def test_format_amenity(self):
        """Amenity formatter includes type, hours, and location."""
        from src.rag.pipeline import _format_amenity

        item = {
            "name": "Elemis Spa",
            "type": "Full-service spa",
            "description": "Luxury treatments.",
            "hours": "9 AM - 9 PM",
            "location": "Level 2",
        }
        text = _format_amenity(item)
        assert "Elemis Spa" in text
        assert "Full-service spa" in text
        assert "9 AM - 9 PM" in text
        assert "Level 2" in text

    def test_format_promotion(self):
        """Promotion formatter includes benefits and how-to-join."""
        from src.rag.pipeline import _format_promotion

        item = {
            "name": "Momentum Rewards",
            "description": "Loyalty program.",
            "benefits": ["Free play", "Dining discounts"],
            "how_to_join": "Sign up at any desk.",
        }
        text = _format_promotion(item)
        assert "Momentum Rewards" in text
        assert "Free play" in text
        assert "Dining discounts" in text
        assert "Sign up" in text

    def test_format_generic_boolean_human_readable(self):
        """Generic formatter renders booleans as Available/Not Available."""
        from src.rag.pipeline import _format_generic

        item = {"name": "Poker Room", "poker_room": True, "high_stakes": False}
        text = _format_generic(item)
        assert "Available" in text
        assert "Not Available" in text

    def test_format_generic_large_numbers_comma(self):
        """Generic formatter comma-formats large numbers."""
        from src.rag.pipeline import _format_generic

        item = {"name": "Casino", "casino_size_sqft": 364000}
        text = _format_generic(item)
        assert "364,000" in text
        assert "sq ft" in text

    def test_formatters_registered_for_all_categories(self):
        """All 10+ categories have registered formatters."""
        from src.rag.pipeline import _FORMATTERS

        required = {"restaurants", "dining", "entertainment", "shows", "hotel",
                     "rooms", "hotel_rooms", "faq", "gaming", "amenities", "promotions"}
        assert required.issubset(set(_FORMATTERS.keys()))


class TestIngestionWithAllCategories:
    """Tests that ingestion handles all categories including amenities and promotions."""

    def test_all_categories_indexed(self, test_property_file, tmp_path):
        """Verify amenities and promotions categories are indexed."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_all_cats")
        vectorstore = ingest_property(
            test_property_file,
            persist_dir=persist_dir,
            knowledge_base_dir=str(tmp_path / "nonexistent_kb"),
        )
        results = vectorstore._collection.get(include=["metadatas"])
        categories = {m.get("category") for m in results["metadatas"] if m}
        assert "amenities" in categories, f"amenities missing. Found: {categories}"
        assert "promotions" in categories, f"promotions missing. Found: {categories}"


class TestProductionChromaGuard:
    """Test that VECTOR_DB=chroma in production raises RuntimeError."""

    def test_chroma_in_production_raises(self, monkeypatch):
        """Production + chroma must hard-fail, not silently log."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("VECTOR_DB", "chroma")
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("CMS_WEBHOOK_SECRET", "test-secret")

        from src.rag.pipeline import _get_retriever_cached, clear_retriever_cache

        clear_retriever_cache()

        with pytest.raises(RuntimeError, match="VECTOR_DB=chroma in production"):
            _get_retriever_cached()


class TestEmbeddingModelNormalization:
    """Test that Settings strips models/ prefix from EMBEDDING_MODEL."""

    def test_strips_models_prefix(self, monkeypatch):
        """EMBEDDING_MODEL with models/ prefix is normalized."""
        monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        from src.config import Settings

        settings = Settings()
        assert settings.EMBEDDING_MODEL == "gemini-embedding-001"

    def test_bare_name_unchanged(self, monkeypatch):
        """EMBEDDING_MODEL without models/ prefix stays unchanged."""
        monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-001")
        from src.config import Settings

        settings = Settings()
        assert settings.EMBEDDING_MODEL == "gemini-embedding-001"


class TestTaskTypeWiring:
    """Test that task_type is passed through to get_embeddings."""

    def test_ingestion_uses_retrieval_document(self, test_property_file, tmp_path):
        """ingest_property passes task_type=RETRIEVAL_DOCUMENT to embeddings."""
        from unittest.mock import MagicMock, call

        from src.rag.pipeline import ingest_property

        mock_embeddings = FakeEmbeddings()
        calls = []

        def tracking_get_embeddings(task_type=None):
            calls.append(task_type)
            return mock_embeddings

        with patch("src.rag.pipeline.get_embeddings", side_effect=tracking_get_embeddings):
            persist_dir = str(tmp_path / "chroma_task_type")
            ingest_property(
                test_property_file,
                persist_dir=persist_dir,
                knowledge_base_dir=str(tmp_path / "nonexistent_kb"),
            )

        assert "RETRIEVAL_DOCUMENT" in calls, f"Expected RETRIEVAL_DOCUMENT in calls: {calls}"
