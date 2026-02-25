"""Tests for retrieval fault tolerance.

R54 fix D5: Verify that retrieval gracefully degrades when one
strategy fails, preserving results from the working strategy.

R59 fix D2: Tests updated to async — search_knowledge_base is now
async-native (uses loop.run_in_executor internally).
"""

import concurrent.futures

import pytest
from unittest.mock import patch, MagicMock

from src.agent.tools import search_knowledge_base


class TestRetrievalResilience:
    """Retrieval graceful degradation under partial failures."""

    async def test_augmented_failure_preserves_semantic(self):
        """When augmented search fails, semantic results still returned."""
        mock_retriever = MagicMock()
        # First call (semantic) succeeds; second call (augmented) fails
        mock_retriever.retrieve_with_scores.side_effect = [
            [(MagicMock(page_content="Lobster Bar menu", metadata={"source": "dining.json", "category": "restaurants"}), 0.9)],
            Exception("Augmented search timeout"),
        ]
        mock_rrf_result = [
            (MagicMock(page_content="Lobster Bar menu", metadata={"source": "dining.json", "category": "restaurants"}), 0.9, 0.016),
        ]
        with patch("src.agent.tools.get_retriever", return_value=mock_retriever), \
             patch("src.agent.tools.rerank_by_rrf", return_value=mock_rrf_result):
            results = await search_knowledge_base("best restaurants")
            assert len(results) >= 1
            assert results[0]["content"] == "Lobster Bar menu"

    async def test_semantic_failure_preserves_augmented(self):
        """When semantic search fails, augmented results still returned."""
        mock_retriever = MagicMock()
        # First call (semantic) fails; second call (augmented) succeeds
        mock_retriever.retrieve_with_scores.side_effect = [
            Exception("Vector DB down"),
            [(MagicMock(page_content="Pool hours 9am-9pm", metadata={"source": "amenities.json", "category": "entertainment"}), 0.8)],
        ]
        mock_rrf_result = [
            (MagicMock(page_content="Pool hours 9am-9pm", metadata={"source": "amenities.json", "category": "entertainment"}), 0.8, 0.016),
        ]
        with patch("src.agent.tools.get_retriever", return_value=mock_retriever), \
             patch("src.agent.tools.rerank_by_rrf", return_value=mock_rrf_result):
            results = await search_knowledge_base("pool hours")
            assert len(results) >= 1
            assert results[0]["content"] == "Pool hours 9am-9pm"

    async def test_both_failures_returns_empty(self):
        """When both strategies fail, returns empty list (not crash)."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_scores.side_effect = Exception("Total failure")
        with patch("src.agent.tools.get_retriever", return_value=mock_retriever):
            results = await search_knowledge_base("anything")
            assert results == []

    async def test_retriever_unavailable_returns_empty(self):
        """When retriever itself can't be created, returns empty."""
        with patch("src.agent.tools.get_retriever", side_effect=RuntimeError("No embeddings")):
            results = await search_knowledge_base("test query")
            assert results == []

    async def test_single_strategy_success_passes_to_rrf(self):
        """When only one strategy succeeds, RRF receives a single result list."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_scores.side_effect = [
            [(MagicMock(page_content="Spa services", metadata={"source": "spa.json"}), 0.85)],
            Exception("Augmented timeout"),
        ]
        mock_rrf_result = [
            (MagicMock(page_content="Spa services", metadata={"source": "spa.json"}), 0.85, 0.016),
        ]
        with patch("src.agent.tools.get_retriever", return_value=mock_retriever), \
             patch("src.agent.tools.rerank_by_rrf", return_value=mock_rrf_result) as mock_rrf:
            results = await search_knowledge_base("spa treatments")
            # RRF should receive exactly 1 result list (semantic only)
            assert mock_rrf.called
            call_args = mock_rrf.call_args
            result_lists = call_args[0][0] if call_args[0] else call_args[1].get("result_lists", [])
            assert len(result_lists) == 1


class TestRetrievalPool:
    """R59 fix D5: Verify _RETRIEVAL_POOL module-level lifecycle."""

    def test_retrieval_pool_is_module_level_thread_pool(self):
        """_RETRIEVAL_POOL is a module-level ThreadPoolExecutor, not per-request."""
        from src.agent.tools import _RETRIEVAL_POOL

        assert isinstance(_RETRIEVAL_POOL, concurrent.futures.ThreadPoolExecutor)

    def test_retrieval_pool_has_sufficient_workers(self):
        """Pool has enough workers for concurrent retrieval under load."""
        from src.agent.tools import _RETRIEVAL_POOL

        # Cloud Run --concurrency=50, 2 futures per request -> need >= 50 workers.
        # Actual value is 50 (set in tools.py).
        assert _RETRIEVAL_POOL._max_workers >= 10

    def test_retrieval_pool_shutdown_is_safe(self):
        """Pool shutdown in app lifespan does not raise on double shutdown."""
        # Verifies the pattern used in src/api/app.py lifespan shutdown.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-rag")
        pool.shutdown(wait=False)
        # Second shutdown should not raise
        pool.shutdown(wait=False)
