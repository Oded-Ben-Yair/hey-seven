"""Firestore-native vector search retriever.

Uses Firestore's built-in ``find_nearest()`` API with COSINE distance
for vector similarity search.  Implements ``AbstractRetriever`` for
formal swappability in ``get_retriever()``.

Multi-strategy RRF reranking is handled by ``tools.py`` (the caller),
NOT inside the retriever.  This matches ``CasinoKnowledgeRetriever``
which also returns single-strategy results.  ``tools.py`` calls
``retrieve_with_scores()`` twice (semantic + augmented) and applies
RRF once — avoiding the double-RRF bug where the retriever did
internal RRF and the caller did external RRF on top.

The ``google.cloud.firestore`` import is lazy to avoid pulling in the
heavy GCP SDK (~200MB) at module level, consistent with ChromaDB lazy
imports in ``pipeline.py``.

R14 fix (DeepSeek F-003, Gemini F4, Grok F-002 — 3/3 consensus):
- Server-side property_id pre-filter via ``where()`` clause, with
  Python-side fallback when composite index is not yet created.
- Cosine distance normalization aligned with ChromaDB's LangChain formula:
  ``similarity = 1 - (distance / 2)`` for consistent behavior across backends.
"""

import logging
from typing import Any

from langchain_core.documents import Document

from src.config import get_settings
from src.rag.pipeline import AbstractRetriever

logger = logging.getLogger(__name__)

# Track whether the server-side property_id filter is working.
# Avoids spamming the warning log on every query after the first failure.
_server_filter_warned = False


class FirestoreRetriever(AbstractRetriever):
    """Firestore-native vector search retriever.

    Implements ``AbstractRetriever`` alongside ``CasinoKnowledgeRetriever``.
    Uses Firestore's built-in vector search (``find_nearest``) with
    COSINE distance.

    R14 fix: property_id filtering now uses server-side ``where()`` pre-filter
    when a composite index on ``(metadata.property_id, embedding)`` exists.
    Falls back to Python-side filtering with a logged warning if the index
    is missing. This eliminates cross-tenant data reaching the application
    layer and reduces Firestore document read costs.

    Args:
        project: GCP project ID for Firestore.
        collection: Firestore collection name containing embedded documents.
        embeddings: An embeddings instance with ``embed_query(text)`` method.
    """

    def __init__(
        self,
        project: str,
        collection: str,
        embeddings: Any,
    ) -> None:
        self._project = project
        self._collection_name = collection
        self._embeddings = embeddings
        self._client: Any | None = None
        self._use_server_filter = True  # Start optimistic; fall back on error

    def _get_client(self) -> Any:
        """Lazy-initialize Firestore client (heavy import)."""
        if self._client is None:
            from google.cloud.firestore import Client

            self._client = Client(project=self._project)
        return self._client

    def _single_vector_query(
        self,
        query_text: str,
        top_k: int,
        property_id: str,
    ) -> list[tuple[Document, float]]:
        """Execute a single Firestore vector search for one query string.

        R14 fix (3/3 consensus): Uses server-side ``where()`` pre-filter for
        property_id isolation. Falls back to Python-side filtering if the
        composite index is not yet created (logs warning on first fallback).

        Args:
            query_text: The text to embed and search for.
            top_k: Number of results to return.
            property_id: Property ID filter for multi-tenant isolation.

        Returns:
            List of (Document, relevance_score) tuples.
        """
        try:
            query_vector = self._embeddings.embed_query(query_text)
        except Exception:
            logger.exception("Failed to embed query for Firestore vector search.")
            return []

        try:
            from google.cloud.firestore_v1.vector import Vector
            from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
            from google.cloud.firestore_v1.field_path import FieldPath

            client = self._get_client()
            collection_ref = client.collection(self._collection_name)

            # R14 fix (DeepSeek F-003, Gemini F4, Grok F-002 — 3/3 consensus):
            # Attempt server-side property_id pre-filter via where() clause.
            # This requires a composite index on (metadata.property_id, embedding).
            # Benefits: cross-tenant data never reaches app layer, 50% less
            # Firestore document read costs, no result starvation from skewed data.
            # Falls back to Python-side filtering if composite index missing.
            use_server = self._use_server_filter
            if use_server:
                try:
                    filtered_ref = collection_ref.where(
                        "metadata.property_id", "==", property_id
                    )
                    vector_query = filtered_ref.find_nearest(
                        vector_field="embedding",
                        query_vector=Vector(query_vector),
                        distance_measure=DistanceMeasure.COSINE,
                        limit=top_k,
                        distance_result_field="distance",
                    )
                except Exception:
                    # Composite index not yet created — fall back to Python-side
                    global _server_filter_warned
                    if not _server_filter_warned:
                        logger.warning(
                            "Firestore server-side property_id filter failed "
                            "(composite index likely missing). Falling back to "
                            "Python-side filtering. Create a composite index on "
                            "(metadata.property_id, embedding) to enable server-side "
                            "filtering and reduce cross-tenant data transfer.",
                        )
                        _server_filter_warned = True
                    self._use_server_filter = False
                    use_server = False

            if not use_server:
                # Fallback: over-fetch 2x for Python-side filtering
                vector_query = collection_ref.find_nearest(
                    vector_field="embedding",
                    query_vector=Vector(query_vector),
                    distance_measure=DistanceMeasure.COSINE,
                    limit=top_k * 2,
                    distance_result_field="distance",
                )

            results: list[tuple[Document, float]] = []
            for doc_snapshot in vector_query.get():
                data = doc_snapshot.to_dict()

                # Python-side property_id filter (only needed in fallback mode)
                if not use_server:
                    doc_property_id = data.get("metadata", {}).get("property_id", "")
                    if doc_property_id != property_id:
                        continue

                page_content = data.get("content", "")
                metadata = data.get("metadata", {})
                metadata["source"] = metadata.get("source", doc_snapshot.id)

                # R14 fix (Grok F-007): Cosine distance normalization aligned
                # with ChromaDB's LangChain formula for cross-backend consistency.
                # Firestore COSINE distance is in [0, 2]; LangChain ChromaDB uses
                # similarity = 1 - (distance / 2), mapping [0,2] -> [1,0].
                # Previous formula (1 - distance) compressed the scale: distance=0.3
                # gave 0.70 here vs 0.85 in ChromaDB, causing the same
                # RAG_MIN_RELEVANCE_SCORE threshold to filter differently.
                distance = data.get("distance", 2.0)
                similarity = max(0.0, 1.0 - (distance / 2.0))

                results.append((
                    Document(page_content=page_content, metadata=metadata),
                    similarity,
                ))

            return results[:top_k]

        except Exception:
            logger.exception("Firestore vector search failed.")
            return []

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with relevance scores via single Firestore vector search.

        Returns single-strategy results. Multi-strategy RRF reranking is
        handled by the caller (``tools.search_knowledge_base()``) which
        calls this method twice with different query variants and applies
        RRF once.  This matches ``CasinoKnowledgeRetriever`` behavior and
        avoids the double-RRF bug (retriever-internal + caller-external).

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of (Document, relevance_score) tuples where 1.0 = exact match.
        """
        settings = get_settings()
        property_id = settings.PROPERTY_NAME.lower().replace(" ", "_")
        results = self._single_vector_query(query, top_k, property_id)
        logger.info(
            "Firestore search returned %d results (query: %.40s...)",
            len(results),
            query,
        )
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_category: str | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents for a query.

        Convenience method matching ``CasinoKnowledgeRetriever.retrieve()``
        interface.  Optionally filters by category.

        Args:
            query: The search query.
            top_k: Number of results to return.
            filter_category: Optional category filter.

        Returns:
            List of LangChain Document objects.
        """
        scored = self.retrieve_with_scores(query, top_k=top_k)
        docs = []
        for doc, _score in scored:
            if filter_category and doc.metadata.get("category") != filter_category:
                continue
            docs.append(doc)
        return docs
