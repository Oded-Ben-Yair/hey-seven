"""Reciprocal Rank Fusion (RRF) reranking.

Shared module to eliminate 3-way duplication across tools.py and
firestore_retriever.py.  Both ChromaDB and Firestore retrieval paths
use the same RRF algorithm.

Reference: Cormack, Clarke, Buettcher (2009) — k=60 per original paper.
"""

import hashlib
import logging

logger = logging.getLogger(__name__)


def rerank_by_rrf(
    result_lists: list[list[tuple]],
    top_k: int = 5,
    k: int = 60,
) -> list[tuple]:
    """Reciprocal Rank Fusion: merge multiple ranked lists into one.

    RRF score = sum(1 / (k + rank_i)) across all rankings for each document.
    Standard k=60 dampens the influence of high-ranking outliers while
    preserving the benefit of appearing in multiple result lists.

    Documents appearing in multiple lists get a boost, improving recall
    for queries where different strategies surface different relevant docs.

    Args:
        result_lists: List of ranked result lists, each containing
            (Document, score) tuples.
        top_k: Number of results to return after fusion.
        k: RRF constant (default 60, per original RRF paper).

    Returns:
        Top-k fused results as (Document, best_cosine_score, rrf_score) tuples,
        sorted by fused RRF score descending.

        **Score contract (R44 fix D2-M001)**:
        - ``result[1]`` = best raw cosine similarity across all lists.
          Use this for **quality filtering** (e.g., RAG_MIN_RELEVANCE_SCORE).
        - ``result[2]`` = RRF fusion score.  Use this for **ranking** and
          **monitoring**.  Higher means the doc appeared in more lists
          and/or ranked higher across strategies.

        Previously, only the cosine score was returned and the RRF score
        was discarded.  Downstream consumers could not distinguish "ranked
        high by RRF" from "high cosine in one strategy."  Now both are
        available.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, tuple] = {}

    for results in result_lists:
        for rank, (doc, score) in enumerate(results):
            # R36 fix A8: Use null byte delimiter to prevent ambiguous
            # concatenation (same fix as _compute_chunk_id in pipeline.py).
            doc_id = hashlib.sha256(
                f"{doc.page_content}\x00{doc.metadata.get('source', '')}".encode()
            ).hexdigest()
            if doc_id not in doc_map or score > doc_map[doc_id][1]:
                doc_map[doc_id] = (doc, score)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    # R44 fix D2-M001: Return both cosine and RRF scores.
    # Cosine score for quality filtering, RRF score for ranking/monitoring.
    fused = [
        (doc_map[doc_id][0], doc_map[doc_id][1], rrf_scores[doc_id])
        for doc_id in sorted_ids[:top_k]
    ]
    logger.debug(
        "RRF fusion: %d lists, %d unique docs, returning top %d",
        len(result_lists),
        len(doc_map),
        len(fused),
    )
    return fused
