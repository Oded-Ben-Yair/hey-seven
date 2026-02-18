"""Live LLM smoke test -- requires GOOGLE_API_KEY env var."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set -- skipping live LLM test",
)


@pytest.mark.asyncio
async def test_live_graph_response():
    """Send one message through the full graph and verify non-empty response."""
    from src.agent.graph import build_graph, chat

    graph = build_graph()
    result = await chat(graph, "What restaurants do you have?")

    assert result["response"], "Graph should return non-empty response"
    assert result["thread_id"], "Should return a thread_id"
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 20, "Response should be substantial"
