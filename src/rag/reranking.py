"""Reciprocal Rank Fusion (RRF) reranking.

Shared module to eliminate 3-way duplication across tools.py and
firestore_retriever.py.  Both ChromaDB and Firestore retrieval paths
use the same RRF algorithm.

Reference: Cormack, Clarke, Buettcher (2009) — k=60 per original paper.
"""

import hashlib


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
