"""Tool functions for the Property Q&A agent.

Plain functions (no @tool decorators) called directly by graph nodes.
Each returns list[RetrievedChunk] with keys: content, metadata, score.
Respects RAG_TOP_K and RAG_MIN_RELEVANCE_SCORE from settings.
"""

import logging

from src.agent.state import RetrievedChunk
from src.config import get_settings
from src.rag.pipeline import get_retriever

logger = logging.getLogger(__name__)


def _filter_by_relevance(results: list[tuple], min_score: float) -> list[tuple]:
    """Filter retrieval results by minimum relevance score.

    Scores are normalized to [0, 1] where 1.0 = exact match (identical
    embedding).  The retriever uses ``similarity_search_with_relevance_scores``
    which applies ``1 / (1 + distance)`` normalization, so ``>= threshold``
    correctly keeps the most relevant documents regardless of ChromaDB's
    underlying distance metric.

    Args:
        results: List of (Document, relevance_score) tuples from retriever.
        min_score: Minimum relevance score to keep (0-1, higher = more relevant).

    Returns:
        Filtered list of (Document, score) tuples above the threshold.
    """
    return [(doc, score) for doc, score in results if score >= min_score]


def search_knowledge_base(query: str) -> list[RetrievedChunk]:
    """Search the property knowledge base for information.

    Uses RAG_TOP_K and RAG_MIN_RELEVANCE_SCORE from application settings.

    Args:
        query: Natural language search query about the property.

    Returns:
        List of RetrievedChunk dicts with keys: content, metadata, score.
        Empty list on error or no results above the relevance threshold.
    """
    settings = get_settings()
    try:
        retriever = get_retriever()
        results = retriever.retrieve_with_scores(query, top_k=settings.RAG_TOP_K)
    except Exception:
        logger.exception("Error searching knowledge base for: %s", query)
        return []

    if not results:
        return []

    results = _filter_by_relevance(results, settings.RAG_MIN_RELEVANCE_SCORE)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        }
        for doc, score in results
    ]


def search_hours(query: str) -> list[RetrievedChunk]:
    """Search for hours and schedule information.

    Augments the user query with schedule-related keywords for better retrieval.
    Uses RAG_TOP_K and RAG_MIN_RELEVANCE_SCORE from application settings.

    Args:
        query: The user's original question about hours or schedules.

    Returns:
        List of RetrievedChunk dicts with keys: content, metadata, score.
        Empty list on error or no results above the relevance threshold.
    """
    settings = get_settings()
    try:
        retriever = get_retriever()
        results = retriever.retrieve_with_scores(
            f"{query} hours schedule open close",
            top_k=settings.RAG_TOP_K,
        )
    except Exception:
        logger.exception("Error looking up hours for: %s", query)
        return []

    if not results:
        return []

    results = _filter_by_relevance(results, settings.RAG_MIN_RELEVANCE_SCORE)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        }
        for doc, score in results
    ]
