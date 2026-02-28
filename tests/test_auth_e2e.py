"""Auth-enabled E2E tests -- exercises production code paths with API key active.

D5 hardening: ensures the authentication middleware is tested end-to-end,
not just in isolation. These tests set API_KEY to a non-empty value so the
ApiKeyMiddleware is ACTIVE, then exercise /chat, /health, /graph, /property,
and /feedback endpoints with valid, invalid, and missing API keys.

This file complements tests/test_api.py (which disables auth via conftest)
by verifying the production auth path that all prior review rounds flagged
as undertested (R47: "90% coverage with auth disabled = fake coverage").
"""

import json
import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


_TEST_API_KEY = "test-auth-e2e-secret-key"


def _make_auth_app(api_key=_TEST_API_KEY):
    """Create a test app with API_KEY set (auth middleware active).

    Same pattern as test_api._make_test_app but with API_KEY configured
    so ApiKeyMiddleware enforces authentication on protected endpoints.
    """
    from src.config import get_settings

    get_settings.cache_clear()

    mock_agent = MagicMock()
    data = {
        "property": {"name": "Auth Test Casino", "location": "Test City"},
        "restaurants": [{"name": "Steakhouse"}],
        "entertainment": [{"name": "Arena"}],
    }

    @asynccontextmanager
    async def test_lifespan(app):
        app.state.agent = mock_agent
        app.state.property_data = data
        app.state.ready = True
        yield
        app.state.ready = False

    __import__("src.api.app")
    app_module = sys.modules["src.api.app"]

    original_lifespan = app_module.lifespan
    app_module.lifespan = test_lifespan
    try:
        with patch.dict("os.environ", {
            "API_KEY": api_key,
            "SEMANTIC_INJECTION_ENABLED": "false",
        }):
            get_settings.cache_clear()
            app = app_module.create_app()
    finally:
        app_module.lifespan = original_lifespan

    return app, mock_agent


def _mock_chat_stream(tokens=None):
    """Return a factory for a fake async generator mimicking chat_stream."""
    tokens = tokens or ["Hello", " there"]

    async def fake_stream(agent, message, thread_id, **kwargs):
        yield {"event": "metadata", "data": json.dumps({"thread_id": thread_id})}
        for tok in tokens:
            yield {"event": "token", "data": json.dumps({"content": tok})}
        yield {"event": "done", "data": json.dumps({"done": True})}

    return fake_stream


class TestAuthChatEndpoint:
    """E2E tests for /chat with auth middleware active."""

    @patch("src.agent.graph.chat_stream")
    def test_chat_with_valid_api_key(self, mock_stream):
        """POST /chat with valid API key returns 200 SSE stream."""
        mock_stream.side_effect = _mock_chat_stream()
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                json={"message": "hello", "thread_id": "00000000-0000-0000-0000-000000000001"},
                headers={"X-API-Key": _TEST_API_KEY},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

    def test_chat_without_api_key_returns_401(self):
        """POST /chat without API key header returns 401."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                json={"message": "hello"},
            )
            assert resp.status_code == 401

    def test_chat_with_wrong_api_key_returns_401(self):
        """POST /chat with incorrect API key returns 401."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                json={"message": "hello"},
                headers={"X-API-Key": "wrong-key-value"},
            )
            assert resp.status_code == 401


class TestAuthHealthEndpoint:
    """Health and liveness endpoints do NOT require auth."""

    def test_health_no_auth_required(self):
        """GET /health is accessible without API key."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code in (200, 503)  # 503 when degraded is OK

    def test_live_no_auth_required(self):
        """GET /live is accessible without API key."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.get("/live")
            assert resp.status_code == 200


class TestAuthProtectedEndpoints:
    """All protected endpoints enforce auth when API_KEY is set."""

    def test_graph_without_api_key_returns_401(self):
        """GET /graph without API key returns 401."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.get("/graph")
            assert resp.status_code == 401

    def test_graph_with_valid_api_key_returns_200(self):
        """GET /graph with valid API key returns 200."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.get(
                "/graph",
                headers={"X-API-Key": _TEST_API_KEY},
            )
            assert resp.status_code == 200

    def test_property_without_api_key_returns_401(self):
        """GET /property without API key returns 401."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.get("/property")
            assert resp.status_code == 401

    def test_property_with_valid_api_key_returns_200(self):
        """GET /property with valid API key returns 200."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.get(
                "/property",
                headers={"X-API-Key": _TEST_API_KEY},
            )
            assert resp.status_code == 200

    def test_feedback_without_api_key_returns_401(self):
        """POST /feedback without API key returns 401."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.post(
                "/feedback",
                json={"thread_id": "00000000-0000-0000-0000-000000000002", "rating": 5},
            )
            assert resp.status_code == 401

    def test_feedback_with_valid_api_key_returns_200(self):
        """POST /feedback with valid API key returns 200."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.post(
                "/feedback",
                json={"thread_id": "00000000-0000-0000-0000-000000000002", "rating": 5},
                headers={"X-API-Key": _TEST_API_KEY},
            )
            assert resp.status_code == 200

    def test_metrics_without_api_key_returns_401(self):
        """GET /metrics without API key returns 401."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.get("/metrics")
            assert resp.status_code == 401

    def test_metrics_with_valid_api_key_returns_200(self):
        """GET /metrics with valid API key returns 200."""
        app, _ = _make_auth_app()
        with TestClient(app) as client:
            resp = client.get(
                "/metrics",
                headers={"X-API-Key": _TEST_API_KEY},
            )
            assert resp.status_code == 200
