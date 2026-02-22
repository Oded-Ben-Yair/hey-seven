"""RAG quality tests: purging, tenant isolation, retrieval benchmarks.

Uses FakeEmbeddings (SHA-384 hash) — no API keys needed.
RAG_MIN_RELEVANCE_SCORE=-100 because hash embeddings produce low cosine scores.
"""

import json
import os

import pytest

# Skip all tests if chromadb is not installed
chromadb = pytest.importorskip("chromadb")

# Set RAG_MIN_RELEVANCE_SCORE before importing src modules —
# hash embeddings produce low cosine scores that would be filtered out.
os.environ.setdefault("RAG_MIN_RELEVANCE_SCORE", "-100")


class FakeEmbeddings:
    """Deterministic SHA-384 hash-based embeddings for testing (no API key needed)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        import hashlib

        h = hashlib.sha384(text.encode()).digest()
        return [float(b) / 255.0 for b in h]


@pytest.fixture(autouse=True)
def _rag_quality_env(monkeypatch):
    """Set environment for all RAG quality tests."""
    monkeypatch.setenv("RAG_MIN_RELEVANCE_SCORE", "-100")
    monkeypatch.setenv("PROPERTY_NAME", "Mohegan Sun")
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _mock_embeddings(monkeypatch):
    """Replace Google embeddings with fake deterministic embeddings."""
    fake = FakeEmbeddings()
    monkeypatch.setattr("src.rag.pipeline.get_embeddings", lambda **kwargs: fake)


@pytest.fixture(autouse=True)
def _reset_retriever():
    """Reset the global retriever singleton between tests."""
    from src.rag.pipeline import clear_retriever_cache

    clear_retriever_cache()
    yield
    clear_retriever_cache()


@pytest.fixture
def no_kb_dir(tmp_path):
    """Path to a nonexistent knowledge-base dir (prevents accidental KB ingestion)."""
    return str(tmp_path / "no_kb")


def _write_property_json(tmp_path, data, filename="property.json"):
    """Write a property JSON file and return its path."""
    p = tmp_path / filename
    p.write_text(json.dumps(data))
    return str(p)


class TestStaleChunkPurging:
    """Test that _purge_stale_chunks removes old version chunks."""

    def test_reingest_purges_old_version_chunks(self, tmp_path, test_property_data, monkeypatch, no_kb_dir):
        """After content edit + re-ingest, old chunks are purged."""
        from src.rag.pipeline import ingest_property

        data_file = tmp_path / "property.json"
        persist_dir = str(tmp_path / "chroma_purge")

        # V1: Ingest original data
        data_file.write_text(json.dumps(test_property_data))
        vs1 = ingest_property(str(data_file), persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        count_v1 = vs1._collection.count()
        assert count_v1 > 0

        # V2: Modify a restaurant name (changes content -> different SHA-256 ID)
        modified_data = json.loads(json.dumps(test_property_data))
        modified_data["restaurants"][0]["name"] = "Updated Steakhouse"
        modified_data["restaurants"][0]["description"] = "A completely new description for the updated steakhouse"
        data_file.write_text(json.dumps(modified_data))

        vs2 = ingest_property(str(data_file), persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        count_v2 = vs2._collection.count()

        # V2 should have same count as V1 because old chunks were purged
        # and new chunks replaced them. The count should be equal (not doubled).
        assert count_v2 == count_v1

    def test_identical_reingest_is_idempotent(self, tmp_path, test_property_data, no_kb_dir):
        """Re-ingesting identical data does not change chunk count."""
        from src.rag.pipeline import ingest_property

        data_file = _write_property_json(tmp_path, test_property_data)
        persist_dir = str(tmp_path / "chroma_idem")

        # First ingestion
        vs1 = ingest_property(data_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        count1 = vs1._collection.count()

        # Second ingestion with identical data
        vs2 = ingest_property(data_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        count2 = vs2._collection.count()

        assert count1 == count2, (
            f"Identical re-ingestion changed chunk count: {count1} -> {count2}"
        )


class TestPropertyIsolation:
    """Test multi-tenant property_id filtering prevents cross-property leakage."""

    def _ingest_for_property(self, tmp_path, property_name, data, persist_dir, monkeypatch, no_kb_dir):
        """Helper: ingest data for a specific property."""
        from src.config import get_settings
        from src.rag.pipeline import ingest_property

        monkeypatch.setenv("PROPERTY_NAME", property_name)
        get_settings.cache_clear()

        data_file = _write_property_json(tmp_path, data, f"{property_name.lower().replace(' ', '_')}.json")
        return ingest_property(data_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)

    def test_mohegan_query_excludes_foxwoods(self, tmp_path, monkeypatch, no_kb_dir):
        """Querying as Mohegan Sun returns zero Foxwoods results."""
        from src.config import get_settings
        from src.rag.pipeline import CasinoKnowledgeRetriever

        persist_dir = str(tmp_path / "chroma_isolation")

        # Ingest Mohegan Sun data
        mohegan_data = {
            "restaurants": [
                {"name": "Mohegan Steakhouse", "cuisine": "Steakhouse", "description": "Premium steaks at Mohegan Sun."}
            ]
        }
        self._ingest_for_property(tmp_path, "Mohegan Sun", mohegan_data, persist_dir, monkeypatch, no_kb_dir)

        # Ingest Foxwoods data (same collection, different property_id)
        foxwoods_data = {
            "restaurants": [
                {"name": "Foxwoods Grill", "cuisine": "American", "description": "Classic grill at Foxwoods Resort."}
            ]
        }
        self._ingest_for_property(tmp_path, "Foxwoods", foxwoods_data, persist_dir, monkeypatch, no_kb_dir)

        # Query as Mohegan Sun
        monkeypatch.setenv("PROPERTY_NAME", "Mohegan Sun")
        get_settings.cache_clear()

        retriever = CasinoKnowledgeRetriever(
            vectorstore=self._open_chroma(persist_dir)
        )
        results = retriever.retrieve("steakhouse restaurant")
        for doc in results:
            assert doc.metadata.get("property_id") == "mohegan_sun", (
                f"Cross-tenant leak: got property_id={doc.metadata.get('property_id')}"
            )

    def test_foxwoods_query_excludes_mohegan(self, tmp_path, monkeypatch, no_kb_dir):
        """Querying as Foxwoods returns zero Mohegan Sun results."""
        from src.config import get_settings
        from src.rag.pipeline import CasinoKnowledgeRetriever

        persist_dir = str(tmp_path / "chroma_isolation2")

        # Ingest both properties
        mohegan_data = {
            "restaurants": [
                {"name": "Mohegan Steakhouse", "cuisine": "Steakhouse", "description": "Premium steaks at Mohegan."}
            ]
        }
        self._ingest_for_property(tmp_path, "Mohegan Sun", mohegan_data, persist_dir, monkeypatch, no_kb_dir)

        foxwoods_data = {
            "restaurants": [
                {"name": "Foxwoods Grill", "cuisine": "American", "description": "Classic grill at Foxwoods."}
            ]
        }
        self._ingest_for_property(tmp_path, "Foxwoods", foxwoods_data, persist_dir, monkeypatch, no_kb_dir)

        # Query as Foxwoods
        monkeypatch.setenv("PROPERTY_NAME", "Foxwoods")
        get_settings.cache_clear()

        retriever = CasinoKnowledgeRetriever(
            vectorstore=self._open_chroma(persist_dir)
        )
        results = retriever.retrieve("steakhouse restaurant")
        for doc in results:
            assert doc.metadata.get("property_id") == "foxwoods", (
                f"Cross-tenant leak: got property_id={doc.metadata.get('property_id')}"
            )

    @staticmethod
    def _open_chroma(persist_dir):
        """Open an existing Chroma collection."""
        from langchain_community.vectorstores import Chroma

        return Chroma(
            collection_name="property_knowledge",
            embedding_function=FakeEmbeddings(),
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )


class TestRetrievalQuality:
    """Golden retrieval tests: known queries return expected chunk categories."""

    def _ingest_and_retrieve(self, tmp_path, test_property_data, query, no_kb_dir):
        """Helper: ingest test data and retrieve results for a query."""
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        data_file = _write_property_json(tmp_path, test_property_data)
        persist_dir = str(tmp_path / "chroma_quality")
        vectorstore = ingest_property(data_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)
        return retriever.retrieve_with_scores(query, top_k=5)

    def test_restaurant_query_returns_dining_chunks(self, tmp_path, test_property_data, no_kb_dir):
        """Query about steakhouse returns chunks from restaurants category."""
        results = self._ingest_and_retrieve(tmp_path, test_property_data, "steakhouse dinner reservation", no_kb_dir)
        assert len(results) > 0, "Expected at least one result for restaurant query"
        categories = [doc.metadata.get("category") for doc, _score in results]
        assert "restaurants" in categories, f"Expected 'restaurants' in categories: {categories}"

    def test_spa_query_returns_amenity_chunks(self, tmp_path, test_property_data, no_kb_dir):
        """Query about spa returns chunks from amenities category."""
        results = self._ingest_and_retrieve(tmp_path, test_property_data, "spa massage facial treatment", no_kb_dir)
        assert len(results) > 0, "Expected at least one result for spa query"
        categories = [doc.metadata.get("category") for doc, _score in results]
        assert "amenities" in categories, f"Expected 'amenities' in categories: {categories}"

    def test_hotel_query_returns_room_chunks(self, tmp_path, test_property_data, no_kb_dir):
        """Query about hotel rooms returns chunks from hotel category."""
        results = self._ingest_and_retrieve(tmp_path, test_property_data, "hotel room king bed suite rate", no_kb_dir)
        assert len(results) > 0, "Expected at least one result for hotel query"
        categories = [doc.metadata.get("category") for doc, _score in results]
        assert "hotel" in categories, f"Expected 'hotel' in categories: {categories}"


class TestSHA256IdempotentIngestion:
    """Test that SHA-256 IDs prevent duplicate chunks."""

    def test_double_ingest_same_count(self, tmp_path, test_property_data, no_kb_dir):
        """Ingesting same data twice produces identical chunk count."""
        from src.rag.pipeline import ingest_property

        data_file = _write_property_json(tmp_path, test_property_data)
        persist_dir = str(tmp_path / "chroma_sha256")

        # First ingestion
        vs1 = ingest_property(data_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        count1 = vs1._collection.count()
        assert count1 > 0

        # Second ingestion (same data, same persist dir)
        vs2 = ingest_property(data_file, persist_dir=persist_dir, knowledge_base_dir=no_kb_dir)
        count2 = vs2._collection.count()

        assert count1 == count2, (
            f"SHA-256 idempotency broken: first={count1}, second={count2}"
        )

    def test_chunk_ids_are_deterministic(self, tmp_path, test_property_data, no_kb_dir):
        """Same content produces same chunk IDs across separate ingestions."""
        from src.rag.pipeline import ingest_property

        data_file = _write_property_json(tmp_path, test_property_data)

        # Ingest into two separate directories
        persist_a = str(tmp_path / "chroma_a")
        persist_b = str(tmp_path / "chroma_b")

        vs_a = ingest_property(data_file, persist_dir=persist_a, knowledge_base_dir=no_kb_dir)
        vs_b = ingest_property(data_file, persist_dir=persist_b, knowledge_base_dir=no_kb_dir)

        ids_a = set(vs_a._collection.get()["ids"])
        ids_b = set(vs_b._collection.get()["ids"])

        assert ids_a == ids_b, "Chunk IDs should be deterministic for identical content"
