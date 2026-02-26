"""Concurrent graph execution tests.

Verifies that multiple simultaneous graph invocations with different
thread_ids do not interfere with each other (state isolation).
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear all singleton caches between tests."""
    yield
    from src.config import get_settings
    get_settings.cache_clear()
    try:
        from src.agent.circuit_breaker import _cb_cache
        _cb_cache.clear()
    except ImportError:
        pass


class _MockLLM:
    """Mock LLM that returns deterministic responses based on input."""

    def with_structured_output(self, schema, **kwargs):
        return self

    async def ainvoke(self, prompt, **kwargs):
        from src.agent.state import RouterOutput
        if isinstance(prompt, str) and "router" in prompt.lower():
            return RouterOutput(query_type="greeting", confidence=0.99)
        return AIMessage(content="Mock response")


async def test_concurrent_threads_no_interference():
    """Two concurrent chat() calls with different thread_ids should not interfere."""
    from src.agent.graph import build_graph, chat

    with patch("src.agent.nodes._get_llm", new_callable=AsyncMock, return_value=_MockLLM()), \
         patch("src.agent.nodes._get_validator_llm", new_callable=AsyncMock, return_value=_MockLLM()):

        graph = build_graph()

        results = await asyncio.gather(
            chat(graph, "Hello from thread A", thread_id="concurrent_A"),
            chat(graph, "Hello from thread B", thread_id="concurrent_B"),
        )

        assert len(results) == 2
        # Both should complete without error
        assert results[0]["thread_id"] == "concurrent_A"
        assert results[1]["thread_id"] == "concurrent_B"
        # Both should have non-empty responses
        assert results[0]["response"]
        assert results[1]["response"]
