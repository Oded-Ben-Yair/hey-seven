"""Tests for the RAG pipeline (ingestion, retrieval, metadata)."""

import sys
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
    """Reset the global retriever singleton between tests."""
    rag_module = sys.modules.get("src.rag.pipeline")
    if rag_module:
        rag_module._retriever_instance = None
    yield
    if rag_module:
        rag_module._retriever_instance = None


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


class TestMissingDataFile:
    def test_missing_file_returns_empty(self, tmp_path):
        """Ingest with non-existent file returns None (no crash)."""
        from src.rag.pipeline import ingest_property

        result = ingest_property(
            str(tmp_path / "nonexistent.json"),
            persist_dir=str(tmp_path / "chroma_missing"),
        )
        assert result is None


class TestChunkCount:
    def test_chunk_count_greater_or_equal_to_doc_count(self, test_property_file, tmp_path):
        """Total chunks >= total input documents (chunking splits, never removes)."""
        from src.rag.pipeline import ingest_property

        persist_dir = str(tmp_path / "chroma_chunks")
        vectorstore = ingest_property(test_property_file, persist_dir=persist_dir)
        collection = vectorstore._collection
        # We have at least restaurants(2) + entertainment(1) + hotel + gaming + faq = 5+
        assert collection.count() >= 5
