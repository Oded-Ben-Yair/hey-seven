"""Tests for the RAG pipeline (ingestion, retrieval, metadata)."""

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

    Uses cache_clear() on the @lru_cache-based get_retriever singleton,
    consistent with conftest.py's _clear_singleton_caches fixture.
    """
    from src.rag.pipeline import get_retriever

    get_retriever.cache_clear()
    yield
    get_retriever.cache_clear()


class TestIngestion:
    def test_ingestion_creates_collection(self, test_property_file, tmp_path):
        """Data loads and indexes into ChromaDB without error."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_test")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
        assert vectorstore is not None
        collection = vectorstore._collection
        assert collection.count() > 0

    def test_ingestion_indexes_all_categories(self, test_property_file, tmp_path):
        """All non-property categories are indexed."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_test2")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
        results = vectorstore._collection.get(include=["metadatas"])
        categories = {m.get("category") for m in results["metadatas"] if m}
        # At minimum restaurants and entertainment should be present
        assert len(categories) >= 2


class TestRetrieval:
    def test_retrieval_returns_results(self, test_property_file, tmp_path):
        """Query for 'steakhouse' returns results from the vectorstore."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        persist_dir = str(tmp_path / "chroma_retrieval")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)
        docs = retriever.retrieve("steakhouse")
        assert len(docs) > 0

    def test_empty_query_returns_gracefully(self, test_property_file, tmp_path):
        """Empty query does not raise an exception."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        persist_dir = str(tmp_path / "chroma_empty")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)
        # Should not raise
        docs = retriever.retrieve("")
        assert isinstance(docs, list)


class TestCategoryMetadata:
    def test_documents_have_category_metadata(self, test_property_file, tmp_path):
        """Each indexed document carries its source category in metadata."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_meta")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
        results = vectorstore._collection.get(include=["metadatas"])
        for metadata in results["metadatas"]:
            assert "category" in metadata, f"Document missing 'category' metadata: {metadata}"

    def test_category_values_match_source(self, test_property_file, tmp_path):
        """Category metadata values match the JSON keys."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_cat_values")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
        results = vectorstore._collection.get(include=["metadatas"])
        categories = {m["category"] for m in results["metadatas"] if "category" in m}
        expected = {"restaurants", "entertainment", "hotel", "gaming", "faq", "property"}
        # All discovered categories should be a subset of expected
        assert categories.issubset(expected), f"Unexpected categories: {categories - expected}"


class TestCategoryFilter:
    def test_filter_by_category(self, test_property_file, tmp_path):
        """Retrieval with filter_category only returns matching category."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        persist_dir = str(tmp_path / "chroma_filter")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)
        docs = retriever.retrieve("food", filter_category="restaurants")
        for doc in docs:
            assert doc.metadata.get("category") == "restaurants"


class TestNestedDictFlattening:
    def test_hotel_nested_dict_produces_documents(self, test_property_file, tmp_path):
        """Hotel data (nested dict with towers/room_types) produces indexed docs."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_nested")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
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
    def test_missing_file_returns_empty(self, tmp_path):
        """Ingest with non-existent file returns None (no crash)."""
        from src.rag.pipeline import ingest_property

        result = ingest_property(
            str(tmp_path / "nonexistent.json"),
            persist_dir=str(tmp_path / "chroma_missing"),
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

        from src.agent.tools import _rerank_by_rrf

        results = [
            (Document(page_content="first"), 0.9),
            (Document(page_content="second"), 0.7),
            (Document(page_content="third"), 0.5),
        ]
        fused = _rerank_by_rrf([results], top_k=3)
        assert len(fused) == 3
        assert fused[0][0].page_content == "first"

    def test_duplicate_across_lists_gets_boosted(self):
        """Document appearing in both lists gets higher RRF score."""
        from langchain_core.documents import Document

        from src.agent.tools import _rerank_by_rrf

        list_a = [
            (Document(page_content="shared"), 0.8),
            (Document(page_content="only_a"), 0.9),
        ]
        list_b = [
            (Document(page_content="only_b"), 0.85),
            (Document(page_content="shared"), 0.7),
        ]
        fused = _rerank_by_rrf([list_a, list_b], top_k=3)
        # "shared" appears in both lists and should rank first
        assert fused[0][0].page_content == "shared"

    def test_respects_top_k(self):
        """RRF returns at most top_k results."""
        from langchain_core.documents import Document

        from src.agent.tools import _rerank_by_rrf

        results = [(Document(page_content=f"doc{i}"), 0.5) for i in range(10)]
        fused = _rerank_by_rrf([results], top_k=3)
        assert len(fused) == 3

    def test_empty_input(self):
        """Empty result lists returns empty."""
        from src.agent.tools import _rerank_by_rrf

        assert _rerank_by_rrf([], top_k=5) == []

    def test_keeps_best_score_for_duplicates(self):
        """When a doc appears in multiple lists, keeps the highest original score."""
        from langchain_core.documents import Document

        from src.agent.tools import _rerank_by_rrf

        list_a = [(Document(page_content="doc"), 0.6)]
        list_b = [(Document(page_content="doc"), 0.9)]
        fused = _rerank_by_rrf([list_a, list_b], top_k=1)
        assert fused[0][1] == 0.9  # Higher score kept

    def test_different_source_not_merged(self):
        """Same content from different sources stays separate (hash includes source)."""
        from langchain_core.documents import Document

        from src.agent.tools import _rerank_by_rrf

        doc_a = Document(page_content="same text", metadata={"source": "restaurants"})
        doc_b = Document(page_content="same text", metadata={"source": "entertainment"})
        list_a = [(doc_a, 0.8)]
        list_b = [(doc_b, 0.7)]
        fused = _rerank_by_rrf([list_a, list_b], top_k=5)
        assert len(fused) == 2, "Same content from different sources must not merge"


class TestChunkCount:
    def test_chunk_count_greater_or_equal_to_doc_count(self, test_property_file, tmp_path):
        """Total chunks >= total input documents (chunking splits, never removes)."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_chunks")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
        collection = vectorstore._collection
        # We have at least restaurants(2) + entertainment(1) + hotel + gaming + faq = 5+
        assert collection.count() >= 5


class TestPropertyIdIsolation:
    """Tests for cross-property leakage prevention via property_id metadata filter."""

    def test_retrieve_with_scores_filters_by_property_id(self, test_property_file, tmp_path):
        """retrieve_with_scores only returns docs matching the configured PROPERTY_NAME."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        persist_dir = str(tmp_path / "chroma_prop_id")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)

        # Normal retrieval returns results for the configured property
        results = retriever.retrieve_with_scores("steakhouse")
        assert len(results) > 0

        # All results carry the correct property_id
        for doc, score in results:
            assert doc.metadata.get("property_id") == "mohegan_sun"

    def test_cross_property_returns_empty(self, test_property_file, tmp_path):
        """Queries with a different PROPERTY_NAME return no results (isolation)."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        persist_dir = str(tmp_path / "chroma_cross_prop")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
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
