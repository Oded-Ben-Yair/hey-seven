"""Tool functions for the Property Q&A agent.

Plain functions (no @tool decorators) called directly by graph nodes.
Each returns list[RetrievedChunk] with keys: content, metadata, score, rrf_score.
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

import concurrent.futures
import logging

# R57 fix D2: Module-level reusable thread pool for concurrent retrieval.
# Per-request ThreadPoolExecutor (R56) created/destroyed 50+ pools/minute
# under load. Module-level pool reuses 2 threads across all requests.
# max_workers=2: one per retrieval strategy (semantic + augmented).
# R57 fix D2: max_workers sized for Cloud Run --concurrency=50.
# Each /chat request submits 2 futures (semantic + augmented). With 50
# concurrent requests, worst case = 100 futures. Workers > 50 ensures
# no request blocks waiting for a thread. Threads are lightweight for
# I/O-bound work (no GIL contention on network waits).
_RETRIEVAL_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=50, thread_name_prefix="rag")

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
    # R52 fix D2: Strip punctuation before word matching. "hours?" must match "hours".
    words = {w.strip(".,;:!?\"'()[]{}") for w in query_lower.split()}
    if words & _TIME_WORDS:
        return "hours schedule open close"
    if words & _PRICE_WORDS:
        return "price rate cost menu"
    return "name location details"


def _filter_by_relevance(results: list[tuple], min_score: float) -> list[tuple]:
    """Filter retrieval results by minimum cosine relevance score.

    R44 fix D2-M001: RRF now returns 3-tuples (doc, cosine_score, rrf_score).
    Filtering uses the cosine score (index 1) for quality gating, since
    cosine similarity is the absolute relevance metric.  The RRF score
    is for ranking, not quality filtering.

    Args:
        results: List of (Document, cosine_score, rrf_score) tuples from RRF.
        min_score: Minimum cosine similarity to keep (0-1, higher = more relevant).

    Returns:
        Filtered list of tuples above the cosine threshold.
    """
    return [r for r in results if r[1] >= min_score]



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
        augmentation = _get_augmentation_terms(query)

        # R57 fix D2: Concurrent retrieval via module-level _RETRIEVAL_POOL.
        # Halves retrieval latency (both strategies are I/O-bound).
        # Reuses thread pool across requests (R56 per-request pool was wasteful).
        future_semantic = _RETRIEVAL_POOL.submit(
            retriever.retrieve_with_scores, query, settings.RAG_TOP_K,
        )
        future_augmented = _RETRIEVAL_POOL.submit(
            retriever.retrieve_with_scores,
            f"{query} {augmentation}",
            settings.RAG_TOP_K,
        )

        semantic_results: list = []
        augmented_results: list = []
        try:
            semantic_results = future_semantic.result(timeout=settings.RETRIEVAL_TIMEOUT)
        except Exception:
            logger.warning("Semantic retrieval failed", exc_info=True)

        try:
            augmented_results = future_augmented.result(timeout=settings.RETRIEVAL_TIMEOUT)
        except Exception:
            logger.warning("Augmented retrieval failed, using semantic-only", exc_info=True)

        # Both strategies failed -- return empty
        if not semantic_results and not augmented_results:
            logger.warning("All retrieval strategies failed for query: %s", query[:80])
            return []

        # Reciprocal Rank Fusion: merge rankings for better recall
        result_lists = []
        if semantic_results:
            result_lists.append(semantic_results)
        if augmented_results:
            result_lists.append(augmented_results)
        fused = rerank_by_rrf(
            result_lists,
            top_k=settings.RAG_TOP_K,
            k=settings.RRF_K,
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
            "score": cosine_score,
            "rrf_score": rrf_score,
        }
        for doc, cosine_score, rrf_score in fused
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

        # R57 fix D2: Concurrent retrieval via module-level _RETRIEVAL_POOL.
        future_schedule = _RETRIEVAL_POOL.submit(
            retriever.retrieve_with_scores,
            f"{query} hours schedule open close",
            settings.RAG_TOP_K,
        )
        future_semantic = _RETRIEVAL_POOL.submit(
            retriever.retrieve_with_scores, query, settings.RAG_TOP_K,
        )

        schedule_results: list = []
        semantic_results: list = []
        try:
            schedule_results = future_schedule.result(timeout=settings.RETRIEVAL_TIMEOUT)
        except Exception:
            logger.warning("Schedule retrieval failed", exc_info=True)
        try:
            semantic_results = future_semantic.result(timeout=settings.RETRIEVAL_TIMEOUT)
        except Exception:
            logger.warning("Semantic retrieval failed for hours query", exc_info=True)

        # Both failed
        if not schedule_results and not semantic_results:
            logger.warning("All retrieval strategies failed for hours query: %s", query[:80])
            return []

        # Reciprocal Rank Fusion
        result_lists = []
        if schedule_results:
            result_lists.append(schedule_results)
        if semantic_results:
            result_lists.append(semantic_results)
        fused = rerank_by_rrf(
            result_lists,
            top_k=settings.RAG_TOP_K,
            k=settings.RRF_K,
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
            "score": cosine_score,
            "rrf_score": rrf_score,
        }
        for doc, cosine_score, rrf_score in fused
    ]
