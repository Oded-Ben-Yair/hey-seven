"""Firestore-native vector search retriever.

Uses Firestore's built-in ``find_nearest()`` API with COSINE distance
for vector similarity search.  Implements ``AbstractRetriever`` for
formal swappability in ``get_retriever()``.

Multi-strategy retrieval with RRF reranking matches the ChromaDB path
in ``tools.py``.  The ``_rerank_by_rrf`` function is intentionally
duplicated here (rather than imported from ``tools.py``) to avoid
circular imports: ``tools.py`` imports from ``rag.pipeline``.

The ``google.cloud.firestore`` import is lazy to avoid pulling in the
heavy GCP SDK (~200MB) at module level, consistent with ChromaDB lazy
imports in ``pipeline.py``.
"""

import hashlib
import logging
from typing import Any

from langchain_core.documents import Document

from src.config import get_settings
from src.rag.pipeline import AbstractRetriever

logger = logging.getLogger(__name__)


def _rerank_by_rrf(
    result_lists: list[list[tuple]],
    top_k: int = 5,
    k: int = 60,
) -> list[tuple]:
    """Reciprocal Rank Fusion: merge multiple ranked lists into one.

    RRF score = sum(1 / (k + rank_i)) across all rankings for each document.
    Standard k=60 per the original RRF paper.

    Note: intentionally duplicated from ``tools._rerank_by_rrf`` to avoid
    circular imports (tools -> rag.pipeline -> this module).

    Args:
        result_lists: List of ranked result lists, each containing
            (Document, score) tuples.
        top_k: Number of results to return after fusion.
        k: RRF constant (default 60).

    Returns:
        Top-k fused results as (Document, original_score) tuples,
        sorted by fused RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, tuple] = {}

    for results in result_lists:
        for rank, (doc, score) in enumerate(results):
            doc_id = hashlib.sha256(
                (doc.page_content + str(doc.metadata.get("source", ""))).encode()
            ).hexdigest()
            if doc_id not in doc_map or score > doc_map[doc_id][1]:
                doc_map[doc_id] = (doc, score)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids[:top_k]]


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
                distance = data.get("distance", 0.0)
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
        """Retrieve documents with relevance scores via multi-strategy Firestore search.

        Uses Reciprocal Rank Fusion (RRF) to combine two retrieval strategies:
        1. Direct semantic search (embedding cosine similarity)
        2. Entity-augmented search (query + domain terms for proper noun matching)

        This matches the dual-strategy RRF pattern used by the ChromaDB path
        in ``tools.search_knowledge_base()``.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of (Document, relevance_score) tuples where 1.0 = exact match.
        """
        settings = get_settings()
        property_id = settings.PROPERTY_NAME.lower().replace(" ", "_")

        # Strategy 1: Direct semantic search
        semantic_results = self._single_vector_query(query, top_k, property_id)

        # Strategy 2: Entity-augmented query for proper noun / name matching
        augmented_results = self._single_vector_query(
            f"{query} name location details", top_k, property_id
        )

        # Reciprocal Rank Fusion: merge rankings for better recall
        fused = _rerank_by_rrf(
            [semantic_results, augmented_results],
            top_k=top_k,
        )

        logger.info(
            "Firestore RRF search returned %d results from %d+%d strategies "
            "(query: %.40s...)",
            len(fused),
            len(semantic_results),
            len(augmented_results),
            query,
        )
        return fused

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
