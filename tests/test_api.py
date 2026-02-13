"""Tests for the FastAPI application endpoints."""

import json
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_test_app(property_data=None):
    """Create a test app with mocked agent and property data.

    Replaces the real lifespan to avoid needing API keys or data files.
    """
    mock_agent = MagicMock()
    default_data = {
        "property": {"name": "Test Casino", "location": "Test City"},
        "restaurants": [{"name": "Steakhouse"}],
        "entertainment": [{"name": "Arena"}],
    }
    data = property_data or default_data

    @asynccontextmanager
    async def test_lifespan(app):
        app.state.agent = mock_agent
        app.state.property_data = data
        app.state.ready = True
        yield
        app.state.ready = False

    # Access the actual module via sys.modules (src.api.__init__ re-exports
    # 'app' which shadows the module name on attribute access).
    import src.api.app  # noqa: ensure module is loaded
    app_module = sys.modules["src.api.app"]

    original_lifespan = app_module.lifespan
    app_module.lifespan = test_lifespan
    try:
        app = app_module.create_app()
    finally:
        app_module.lifespan = original_lifespan

    return app, mock_agent


class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /health returns 200 with status fields."""
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["version"] == "0.1.0"
            assert data["agent_ready"] is True
            assert data["property_loaded"] is True


class TestPropertyEndpoint:
    def test_property_returns_metadata(self):
        """GET /property returns correct property metadata."""
        prop_data = {
            "property": {"name": "Mohegan Sun", "location": "Uncasville, CT"},
            "restaurants": [{"name": "R1"}, {"name": "R2"}],
            "entertainment": [{"name": "E1"}],
            "gaming": {"slots": 5000},
        }
        app, _ = _make_test_app(property_data=prop_data)
        with TestClient(app) as client:
            resp = client.get("/property")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Mohegan Sun"
            assert data["location"] == "Uncasville, CT"
            assert "restaurants" in data["categories"]
            assert "entertainment" in data["categories"]
            assert data["document_count"] == 4  # 2 restaurants + 1 entertainment + 1 gaming


class TestChatEndpoint:
    @patch("src.agent.graph.chat", new_callable=AsyncMock)
    def test_chat_returns_sse_response(self, mock_chat):
        """POST /chat returns SSE event with response."""
        mock_chat.return_value = {
            "response": "The steakhouse is on the main floor.",
            "thread_id": "test-thread-123",
            "sources": ["restaurants"],
        }
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                json={"message": "Where is the steakhouse?"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            # Parse SSE data line
            lines = resp.text.strip().split("\n")
            data_lines = [l for l in lines if l.startswith("data:")]
            assert len(data_lines) >= 1
            payload = json.loads(data_lines[0].removeprefix("data:").strip())
            assert payload["done"] is True
            assert "steakhouse" in payload["response"].lower()
            assert payload["thread_id"] == "test-thread-123"

    def test_chat_missing_message_returns_422(self):
        """POST /chat with empty body returns 422."""
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/chat", json={})
            assert resp.status_code == 422

    @patch("src.agent.graph.chat", new_callable=AsyncMock)
    def test_chat_response_has_correct_fields(self, mock_chat):
        """Response payload contains all expected fields."""
        mock_chat.return_value = {
            "response": "Hello!",
            "thread_id": "t1",
            "sources": [],
        }
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "Hi"})
            assert resp.status_code == 200
            lines = resp.text.strip().split("\n")
            data_lines = [l for l in lines if l.startswith("data:")]
            payload = json.loads(data_lines[0].removeprefix("data:").strip())
            assert "response" in payload
            assert "thread_id" in payload
            assert "sources" in payload
            assert "done" in payload


class TestChatValidation:
    def test_message_too_short(self):
        """POST /chat with empty string returns 422."""
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": ""})
            assert resp.status_code == 422

    @patch("src.agent.graph.chat", new_callable=AsyncMock)
    def test_message_with_thread_id(self, mock_chat):
        """POST /chat accepts optional thread_id."""
        mock_chat.return_value = {
            "response": "Follow-up answer.",
            "thread_id": "existing-thread",
            "sources": [],
        }
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                json={"message": "Follow up question", "thread_id": "existing-thread"},
            )
            assert resp.status_code == 200
            mock_chat.assert_called_once()
            call_args = mock_chat.call_args
            assert call_args[0][1] == "Follow up question"
            assert call_args[0][2] == "existing-thread"
