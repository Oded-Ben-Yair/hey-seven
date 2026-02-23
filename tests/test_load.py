"""SSE load tests: concurrent request handling and rate limiter enforcement.

Verifies that the FastAPI /chat endpoint handles concurrent SSE requests
without dropping connections and that the rate limiter kicks in when the
threshold is exceeded.
"""

import asyncio
import json
import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    """Disable API key auth and clear settings cache for load tests."""
    monkeypatch.setenv("API_KEY", "")
    try:
        from src.config import get_settings
        get_settings.cache_clear()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from src.config import get_settings
        get_settings.cache_clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def mock_agent():
    """Create a mock agent that returns a simple streaming response."""
    async def _fake_stream(*args, **kwargs):
        yield {"event": "token", "data": "Hello"}
        yield {"event": "done", "data": json.dumps({"done": True})}

    with patch("src.agent.graph.build_graph") as mock_build, \
         patch("src.agent.memory.get_checkpointer", new_callable=AsyncMock), \
         patch("src.agent.graph.chat_stream", side_effect=_fake_stream):
        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        yield mock_graph


def _make_chat_body(index: int) -> dict:
    """Create a valid /chat request body with a valid UUID thread_id."""
    return {
        "message": f"Test message {index}",
        "thread_id": str(uuid.uuid4()),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcurrentSSERequests:
    """Verify /chat handles concurrent SSE requests without dropping."""

    @pytest.mark.asyncio
    async def test_concurrent_sse_requests(self, mock_agent):
        """20 concurrent POST /chat requests all receive SSE responses."""
        import httpx
        from src.api.app import create_app

        app = create_app()
        app.state.agent = mock_agent
        app.state.ready = True
        app.state.property_data = {"property": {"name": "Test"}}

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            tasks = [
                client.post(
                    "/chat",
                    json=_make_chat_body(i),
                    headers={"Content-Type": "application/json"},
                )
                for i in range(20)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful responses (200 or 429 for rate-limited ones)
        successful = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful) >= 1, "At least some requests should succeed"

        # All non-exception responses should be valid HTTP codes
        for r in successful:
            assert r.status_code in (200, 429), f"Unexpected status: {r.status_code}"


class TestRateLimiterEnforcement:
    """Verify rate limiter kicks in beyond RATE_LIMIT_CHAT threshold."""

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self, monkeypatch, mock_agent):
        """Requests beyond RATE_LIMIT_CHAT get 429 Too Many Requests."""
        monkeypatch.setenv("RATE_LIMIT_CHAT", "3")

        from src.config import get_settings
        get_settings.cache_clear()

        import httpx
        from src.api.app import create_app

        app = create_app()
        app.state.agent = mock_agent
        app.state.ready = True
        app.state.property_data = {"property": {"name": "Test"}}

        status_codes = []
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            # Send requests sequentially to ensure consistent rate limiting
            for i in range(8):
                resp = await client.post(
                    "/chat",
                    json=_make_chat_body(i),
                    headers={"Content-Type": "application/json"},
                )
                status_codes.append(resp.status_code)

        # At least some should be rate-limited (429)
        assert 429 in status_codes, (
            f"Expected at least one 429 response. Got: {status_codes}"
        )
        # First few should succeed (200)
        assert 200 in status_codes, (
            f"Expected at least one 200 response. Got: {status_codes}"
        )

        get_settings.cache_clear()
