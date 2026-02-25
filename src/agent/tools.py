"""Tool functions for the Property Q&A agent.

Async functions called directly by graph nodes (no @tool decorators).
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

import asyncio
import concurrent.futures
import logging

# R59 fix D2: Module-level thread pool for bridging sync ChromaDB to async.
# ChromaDB (dev) and Firestore retriever are sync-only; run_in_executor
# bridges them to async without the nested threading anti-pattern
# (previously: to_thread spawned one thread, which submitted to this pool).
# Now: retrieve_node calls async search functions directly, which use
# run_in_executor with this pool for the sync retriever calls.
# max_workers=50: sized for Cloud Run --concurrency=50, 2 futures per request.
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



async def _safe_await(coro, name: str) -> list:
    """Await a coroutine with error isolation for asyncio.gather.

    Returns the result on success, empty list on any failure.
    Used by search_knowledge_base and search_hours to run strategies
    concurrently via asyncio.gather without one failure killing the other.

    Args:
        coro: The coroutine to await (typically asyncio.wait_for wrapped).
        name: Human-readable strategy name for logging.

    Returns:
        The coroutine result on success, or empty list on failure.
    """
    try:
        return await coro
    except Exception:
        logger.warning("%s retrieval failed", name, exc_info=True)
        return []


async def search_knowledge_base(query: str) -> list[RetrievedChunk]:
    """Search the property knowledge base using async multi-strategy retrieval with RRF.

    R59 fix D2: Converted from sync to async-native. Uses loop.run_in_executor
    with _RETRIEVAL_POOL for the sync ChromaDB calls. This eliminates the
    nested threading anti-pattern where retrieve_node used asyncio.to_thread
    to call a sync function that internally used ThreadPoolExecutor.submit.

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
        loop = asyncio.get_running_loop()

        # R59 fix D2: Native async with run_in_executor. No nested threading.
        # ChromaDB is sync; run_in_executor bridges to async cleanly.
        # Both strategies run concurrently in _RETRIEVAL_POOL threads.
        semantic_task = loop.run_in_executor(
            _RETRIEVAL_POOL,
            retriever.retrieve_with_scores, query, settings.RAG_TOP_K,
        )
        augmented_task = loop.run_in_executor(
            _RETRIEVAL_POOL,
            retriever.retrieve_with_scores,
            f"{query} {augmentation}",
            settings.RAG_TOP_K,
        )

        # R60 fix D2: asyncio.gather for true concurrent awaiting.
        # Previously sequential awaits meant if semantic took 5s, augmented
        # waited even if it finished in 1s. gather() returns when both complete.
        # _safe_await isolates errors per strategy (one can fail, other succeeds).
        semantic_results, augmented_results = await asyncio.gather(
            _safe_await(asyncio.wait_for(semantic_task, timeout=settings.RETRIEVAL_TIMEOUT), "Semantic"),
            _safe_await(asyncio.wait_for(augmented_task, timeout=settings.RETRIEVAL_TIMEOUT), "Augmented"),
        )

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


async def search_hours(query: str) -> list[RetrievedChunk]:
    """Search for hours and schedule information with async RRF reranking.

    R59 fix D2: Converted from sync to async-native. Same pattern as
    search_knowledge_base — uses loop.run_in_executor for sync retriever calls.

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
        loop = asyncio.get_running_loop()

        # R59 fix D2: Native async with run_in_executor (no nested threading).
        schedule_task = loop.run_in_executor(
            _RETRIEVAL_POOL,
            retriever.retrieve_with_scores,
            f"{query} hours schedule open close",
            settings.RAG_TOP_K,
        )
        semantic_task = loop.run_in_executor(
            _RETRIEVAL_POOL,
            retriever.retrieve_with_scores, query, settings.RAG_TOP_K,
        )

        # R60 fix D2: asyncio.gather for true concurrent awaiting (same as search_knowledge_base).
        schedule_results, semantic_results = await asyncio.gather(
            _safe_await(asyncio.wait_for(schedule_task, timeout=settings.RETRIEVAL_TIMEOUT), "Schedule"),
            _safe_await(asyncio.wait_for(semantic_task, timeout=settings.RETRIEVAL_TIMEOUT), "Semantic (hours)"),
        )

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
