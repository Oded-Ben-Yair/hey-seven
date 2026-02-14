"""Tool functions for the Property Q&A agent.

Plain functions (no @tool decorators) called directly by graph nodes.
Each returns list[dict] with keys: content, metadata, score.
"""

import logging

from src.rag.pipeline import get_retriever

logger = logging.getLogger(__name__)


def search_knowledge_base(query: str, top_k: int = 5) -> list[dict]:
    """Search the property knowledge base for information.

    Args:
        query: Natural language search query about the property.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: content, metadata, score.
        Empty list on error or no results.
    """
    try:
        retriever = get_retriever()
        results = retriever.retrieve_with_scores(query, top_k=top_k)
    except Exception:
        logger.exception("Error searching knowledge base for: %s", query)
        return []

    if not results:
        return []

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        }
        for doc, score in results
    ]


def search_hours(venue_name: str, top_k: int = 5) -> list[dict]:
    """Search for hours and schedule information for a specific venue.

    Args:
        venue_name: Name of the restaurant, show, or venue to look up.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: content, metadata, score.
        Empty list on error or no results.
    """
    try:
        retriever = get_retriever()
        results = retriever.retrieve_with_scores(
            f"{venue_name} hours schedule open close",
            top_k=top_k,
        )
    except Exception:
        logger.exception("Error looking up hours for: %s", venue_name)
        return []

    if not results:
        return []

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        }
        for doc, score in results
    ]
