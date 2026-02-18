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
"""

import logging
from typing import Any

from langchain_core.documents import Document

from src.config import get_settings
from src.rag.pipeline import AbstractRetriever

logger = logging.getLogger(__name__)


class FirestoreRetriever(AbstractRetriever):
    """Firestore-native vector search retriever.

    Implements ``AbstractRetriever`` alongside ``CasinoKnowledgeRetriever``.
    Uses Firestore's built-in vector search (``find_nearest``) with
    COSINE distance.

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

            client = self._get_client()
            collection_ref = client.collection(self._collection_name)

            vector_query = collection_ref.find_nearest(
                vector_field="embedding",
                query_vector=Vector(query_vector),
                distance_measure=DistanceMeasure.COSINE,
                limit=top_k,
                distance_result_field="distance",
            )

            results: list[tuple[Document, float]] = []
            for doc_snapshot in vector_query.get():
                data = doc_snapshot.to_dict()
                doc_property_id = data.get("metadata", {}).get("property_id", "")
                if doc_property_id != property_id:
                    continue

                page_content = data.get("content", "")
                metadata = data.get("metadata", {})
                metadata["source"] = metadata.get("source", doc_snapshot.id)

                # Firestore COSINE distance is in [0, 2]; convert to similarity [0, 1].
                # distance = 0 => similarity = 1 (identical).
                # Default to 2.0 (worst case) when field is missing to avoid
                # treating missing-distance docs as perfect matches.
                distance = data.get("distance", 2.0)
                similarity = max(0.0, 1.0 - distance)

                results.append((
                    Document(page_content=page_content, metadata=metadata),
                    similarity,
                ))

            return results

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
