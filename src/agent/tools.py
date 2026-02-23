"""Tool functions for the Property Q&A agent.

Plain functions (no @tool decorators) called directly by graph nodes.
Each returns list[RetrievedChunk] with keys: content, metadata, score.
Respects RAG_TOP_K and RAG_MIN_RELEVANCE_SCORE from settings.

RAG retrieval architecture (dual-strategy + RRF fusion):

    This module is the orchestration layer for retrieval. The pipeline is:

    1. **Strategy 1 — direct semantic search**: The raw user query is
       embedded and compared via cosine similarity against the vector store.
       Handles well-formed natural language questions.

    2. **Strategy 2 — entity-augmented search**: The query is augmented with
       domain-relevant terms (e.g., "name location details" or "hours schedule
       open close") to boost proper noun and schedule matches that pure
       semantic search may rank lower.

    3. **Fusion — Reciprocal Rank Fusion (RRF)**: Both result lists are
       merged using RRF with k=60 (per the original RRF paper). Documents
       appearing in both lists receive higher fused scores. The highest
       original cosine score is preserved per document for quality filtering.

    4. **Post-fusion filtering**: Results below ``RAG_MIN_RELEVANCE_SCORE``
       are removed using the original cosine similarity (not the RRF rank
       score). This ensures grounding quality even for documents ranked
       highly by fusion.

    The retriever (``CasinoKnowledgeRetriever.retrieve_with_scores()``) in
    ``src/rag/pipeline.py`` returns single-strategy results. Multi-strategy
    orchestration and RRF fusion happen here in the tools layer, keeping the
    retriever backend-agnostic and reusable.
"""

import logging

from src.agent.state import RetrievedChunk
from src.config import get_settings
from src.rag.pipeline import get_retriever
from src.rag.reranking import rerank_by_rrf

logger = logging.getLogger(__name__)

# Query-type keyword sets for smart augmentation (Strategy 2).
# Each set maps to domain-specific augmentation terms that boost
# retrieval accuracy for that query category.
_TIME_WORDS = frozenset({
    "hour", "hours", "time", "open", "close", "schedule", "when",
    "early", "late", "morning", "evening", "tonight", "today",
})
_PRICE_WORDS = frozenset({
    "price", "cost", "rate", "expensive", "cheap", "affordable",
    "menu", "fee", "charge", "dollar", "per",
})


def _get_augmentation_terms(query: str) -> str:
    """Select domain-specific augmentation terms based on query content.

    Improves Strategy 2 (entity-augmented search) by matching augmentation
    terms to the likely query intent:
    - Time/schedule queries: augmented with schedule keywords
    - Price/cost queries: augmented with pricing keywords
    - Default: augmented with name/location keywords for entity matching

    Args:
        query: The user's search query.

    Returns:
        Space-separated augmentation terms to append to the query.
    """
    query_lower = query.lower()
    words = set(query_lower.split())
    if words & _TIME_WORDS:
        return "hours schedule open close"
    if words & _PRICE_WORDS:
        return "price rate cost menu"
    return "name location details"


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
        # Strategy 2: Query-type-aware augmented search for improved recall
        augmentation = _get_augmentation_terms(query)
        augmented_results = retriever.retrieve_with_scores(
            f"{query} {augmentation}",
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
