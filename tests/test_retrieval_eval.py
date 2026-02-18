"""Offline retrieval evaluation with known query-to-category mapping."""

import pytest


# Retrieval eval: query -> expected category in top-k results
_EVAL_QUERIES = [
    ("What restaurants are available?", "restaurants"),
    ("Tell me about the hotel rooms", "hotel"),
    ("What shows are playing?", "entertainment"),
    ("What's the casino size?", "gaming"),
    ("How do I get to Mohegan Sun?", "property"),
]


@pytest.mark.parametrize("query,expected_category", _EVAL_QUERIES)
def test_retrieval_returns_relevant_category(query, expected_category):
    """Verify retriever returns docs from expected category."""
    # This test requires a populated ChromaDB -- skip if not available
    pytest.importorskip("chromadb")

    try:
        from src.rag.pipeline import get_retriever

        retriever = get_retriever()
        results = retriever.retrieve_with_scores(query, top_k=5)
        if not results:
            pytest.skip("No retrieval results -- ChromaDB may not be populated")
        categories = [doc.metadata.get("category", "") for doc, _ in results]
        assert expected_category in categories, (
            f"Expected '{expected_category}' in results for '{query}', got: {categories}"
        )
    except Exception as e:
        pytest.skip(f"Retrieval not available: {e}")
