"""Tool functions for the Property Q&A agent.

Plain functions (no @tool decorators) called directly by graph nodes.
Each returns list[RetrievedChunk] with keys: content, metadata, score.
Respects RAG_TOP_K and RAG_MIN_RELEVANCE_SCORE from settings.

Retrieval uses Reciprocal Rank Fusion (RRF) to combine multiple query
strategies for improved recall, especially for proper noun queries
(e.g., "Todd English's") where augmented search ranks differently.
"""

import logging

from src.agent.state import RetrievedChunk
from src.config import get_settings
from src.rag.pipeline import get_retriever
from src.rag.reranking import rerank_by_rrf

logger = logging.getLogger(__name__)


def _filter_by_relevance(results: list[tuple], min_score: float) -> list[tuple]:
    """Filter retrieval results by minimum relevance score.

    Scores are normalized to [0, 1] where 1.0 = exact match (cosine
    similarity).  The collection uses ``hnsw:space=cosine``, so scores
    are cosine similarities directly.

    Args:
        results: List of (Document, relevance_score) tuples from retriever.
        min_score: Minimum relevance score to keep (0-1, higher = more relevant).

    Returns:
        Filtered list of (Document, score) tuples above the threshold.
    """
    return [(doc, score) for doc, score in results if score >= min_score]



def search_knowledge_base(query: str) -> list[RetrievedChunk]:
    """Search the property knowledge base using multi-strategy retrieval with RRF.

    Combines two retrieval strategies via Reciprocal Rank Fusion:
    1. Direct semantic search (embedding cosine similarity)
    2. Entity-augmented search (query + domain terms for proper noun matching)

    RRF fusion improves recall for entity-heavy queries (e.g., "Todd English's")
    where augmented search may rank restaurant name matches higher than pure
    semantic search.

    Args:
        query: Natural language search query about the property.

    Returns:
        List of RetrievedChunk dicts with keys: content, metadata, score.
        Empty list on error or no results above the relevance threshold.
    """
    settings = get_settings()
    try:
        retriever = get_retriever()
        # Strategy 1: Direct semantic search
        semantic_results = retriever.retrieve_with_scores(query, top_k=settings.RAG_TOP_K)
        # Strategy 2: Entity-augmented query for proper noun / name matching
        augmented_results = retriever.retrieve_with_scores(
            f"{query} name location details",
            top_k=settings.RAG_TOP_K,
        )
        # Reciprocal Rank Fusion: merge rankings for better recall
        fused = rerank_by_rrf(
            [semantic_results, augmented_results],
            top_k=settings.RAG_TOP_K,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("Retrieval parsing error for query '%s': %s", query, exc)
        return []
    except Exception:
        logger.exception("Error searching knowledge base for: %s", query)
        return []

    if not fused:
        return []

    # Filter by original cosine similarity (not RRF rank score).
    # A doc may rank high via RRF fusion but still be below the
    # absolute relevance threshold — cosine score is the right
    # quality gate for grounding.
    fused = _filter_by_relevance(fused, settings.RAG_MIN_RELEVANCE_SCORE)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        }
        for doc, score in fused
    ]


def search_hours(query: str) -> list[RetrievedChunk]:
    """Search for hours and schedule information with RRF reranking.

    Uses two strategies via RRF:
    1. Schedule-augmented semantic search (query + schedule keywords)
    2. Direct semantic search (for broader context about the venue)

    Args:
        query: The user's original question about hours or schedules.

    Returns:
        List of RetrievedChunk dicts with keys: content, metadata, score.
        Empty list on error or no results above the relevance threshold.
    """
    settings = get_settings()
    try:
        retriever = get_retriever()
        # Strategy 1: Schedule-focused query
        schedule_results = retriever.retrieve_with_scores(
            f"{query} hours schedule open close",
            top_k=settings.RAG_TOP_K,
        )
        # Strategy 2: Direct semantic search for broader venue context
        semantic_results = retriever.retrieve_with_scores(query, top_k=settings.RAG_TOP_K)
        # Reciprocal Rank Fusion
        fused = rerank_by_rrf(
            [schedule_results, semantic_results],
            top_k=settings.RAG_TOP_K,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("Retrieval parsing error for hours query '%s': %s", query, exc)
        return []
    except Exception:
        logger.exception("Error looking up hours for: %s", query)
        return []

    if not fused:
        return []

    fused = _filter_by_relevance(fused, settings.RAG_MIN_RELEVANCE_SCORE)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        }
        for doc, score in fused
    ]
