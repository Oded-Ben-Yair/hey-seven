"""Tests for the FastAPI application endpoints."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

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
    __import__("src.api.app")
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


def _mock_chat_stream(thread_id="test-thread-123", tokens=None, sources=None):
    """Create a mock async generator that simulates chat_stream output."""
    tokens = tokens or ["The steakhouse ", "is on the ", "main floor."]
    sources = sources or ["restaurants"]

    async def fake_stream(agent, message, tid=None):
        yield {"event": "metadata", "data": json.dumps({"thread_id": thread_id})}
        for tok in tokens:
            yield {"event": "token", "data": json.dumps({"content": tok})}
        if sources:
            yield {"event": "sources", "data": json.dumps({"sources": sources})}
        yield {"event": "done", "data": json.dumps({"done": True})}

    return fake_stream


def _parse_sse_events(text: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    events = []
    current_event = "message"
    for line in text.strip().split("\n"):
        if line.startswith("event:"):
            current_event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_str = line.removeprefix("data:").strip()
            events.append({"event": current_event, "data": json.loads(data_str)})
            current_event = "message"
    return events


class TestChatEndpoint:
    @patch("src.agent.graph.chat_stream")
    def test_chat_returns_sse_streaming(self, mock_stream):
        """POST /chat returns SSE events with metadata, tokens, sources, done."""
        mock_stream.side_effect = _mock_chat_stream()
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                json={"message": "Where is the steakhouse?"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            events = _parse_sse_events(resp.text)
            event_types = [e["event"] for e in events]
            assert "metadata" in event_types
            assert "token" in event_types
            assert "done" in event_types
            # Metadata has thread_id
            meta = next(e for e in events if e["event"] == "metadata")
            assert meta["data"]["thread_id"] == "test-thread-123"

    def test_chat_missing_message_returns_422(self):
        """POST /chat with empty body returns 422."""
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/chat", json={})
            assert resp.status_code == 422

    @patch("src.agent.graph.chat_stream")
    def test_chat_streams_all_event_types(self, mock_stream):
        """Response includes metadata, token, sources, and done events."""
        mock_stream.side_effect = _mock_chat_stream(
            tokens=["Hello!"], sources=["restaurants"]
        )
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "Hi"})
            assert resp.status_code == 200
            events = _parse_sse_events(resp.text)
            event_types = [e["event"] for e in events]
            assert "metadata" in event_types
            assert "token" in event_types
            assert "sources" in event_types
            assert "done" in event_types


class TestChatValidation:
    def test_message_too_short(self):
        """POST /chat with empty string returns 422."""
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": ""})
            assert resp.status_code == 422

    @patch("src.agent.graph.chat_stream")
    def test_message_with_thread_id(self, mock_stream):
        """POST /chat accepts optional thread_id (valid UUID)."""
        tid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        mock_stream.side_effect = _mock_chat_stream(thread_id=tid)
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                json={"message": "Follow up question", "thread_id": tid},
            )
            assert resp.status_code == 200
            mock_stream.assert_called_once()
            events = _parse_sse_events(resp.text)
            meta = next(e for e in events if e["event"] == "metadata")
            assert meta["data"]["thread_id"] == tid

    def test_invalid_thread_id_returns_422(self):
        """POST /chat with non-UUID thread_id returns 422."""
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                json={"message": "test", "thread_id": "not-a-uuid"},
            )
            assert resp.status_code == 422


def _make_test_app_no_agent():
    """Create a test app with agent=None (for 503/degraded tests)."""
    @asynccontextmanager
    async def no_agent_lifespan(app):
        app.state.agent = None
        app.state.property_data = {"property": {"name": "Test"}}
        app.state.ready = True
        yield
        app.state.ready = False

    import src.api.app  # noqa: F401
    app_module = sys.modules["src.api.app"]
    original = app_module.lifespan
    app_module.lifespan = no_agent_lifespan
    try:
        return app_module.create_app()
    finally:
        app_module.lifespan = original


class TestChatAgent503:
    def test_503_when_agent_none(self):
        """POST /chat returns 503 with Retry-After when agent is None."""
        test_app = _make_test_app_no_agent()
        with TestClient(test_app) as client:
            resp = client.post("/chat", json={"message": "hello"})
            assert resp.status_code == 503
            assert "Retry-After" in resp.headers
            assert resp.json()["error"] == "Agent not initialized. Try again later."


class TestHealthDegraded:
    def test_health_degraded_when_agent_not_ready(self):
        """GET /health returns 503 'degraded' when agent is None."""
        test_app = _make_test_app_no_agent()
        with TestClient(test_app) as client:
            resp = client.get("/health")
            assert resp.status_code == 503
            assert resp.json()["status"] == "degraded"
            assert resp.json()["agent_ready"] is False


def _mock_chat_stream_error():
    """Create a mock that raises an exception mid-stream."""
    async def failing_stream(agent, message, tid=None):
        yield {"event": "metadata", "data": json.dumps({"thread_id": "err-thread"})}
        yield {"event": "token", "data": json.dumps({"content": "partial "})}
        raise RuntimeError("LLM API failure")

    return failing_stream


def _mock_chat_stream_slow():
    """Create a mock that hangs (simulates timeout)."""
    async def slow_stream(agent, message, tid=None):
        yield {"event": "metadata", "data": json.dumps({"thread_id": "slow-thread"})}
        await asyncio.sleep(999)  # Will be interrupted by timeout

    return slow_stream


class TestChatSSEErrors:
    @patch("src.agent.graph.chat_stream")
    def test_mid_stream_error_yields_error_event(self, mock_stream):
        """Exception during streaming emits an error SSE event."""
        mock_stream.side_effect = _mock_chat_stream_error()
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "test"})
            assert resp.status_code == 200
            events = _parse_sse_events(resp.text)
            event_types = [e["event"] for e in events]
            assert "error" in event_types
            assert "done" in event_types
            error_event = next(e for e in events if e["event"] == "error")
            assert "error" in error_event["data"]

    @patch("src.agent.graph.chat_stream")
    def test_timeout_yields_error_event(self, mock_stream):
        """SSE timeout emits an error event instead of hanging."""
        mock_stream.side_effect = _mock_chat_stream_slow()
        app, _ = _make_test_app()

        # Patch the module-level settings to use a 1-second timeout
        import src.api.app  # noqa: F401
        app_mod = sys.modules["src.api.app"]

        original_timeout = app_mod.settings.SSE_TIMEOUT_SECONDS
        app_mod.settings.SSE_TIMEOUT_SECONDS = 1
        try:
            with TestClient(app) as client:
                resp = client.post("/chat", json={"message": "test"})
                assert resp.status_code == 200
                events = _parse_sse_events(resp.text)
                event_types = [e["event"] for e in events]
                assert "error" in event_types
                assert "done" in event_types
                error_event = next(e for e in events if e["event"] == "error")
                assert "timed out" in error_event["data"]["error"].lower()
        finally:
            app_mod.settings.SSE_TIMEOUT_SECONDS = original_timeout


class TestChatReplaceEvent:
    @patch("src.agent.graph.chat_stream")
    def test_replace_event_in_sse_stream(self, mock_stream):
        """POST /chat can include 'replace' SSE events (from greeting/off_topic/fallback)."""

        async def replace_stream(agent, message, tid=None):
            yield {"event": "metadata", "data": json.dumps({"thread_id": "replace-test"})}
            yield {"event": "replace", "data": json.dumps({"content": "Welcome to Mohegan Sun!"})}
            yield {"event": "done", "data": json.dumps({"done": True})}

        mock_stream.side_effect = replace_stream
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "Hello"})
            assert resp.status_code == 200
            events = _parse_sse_events(resp.text)
            event_types = [e["event"] for e in events]
            assert "replace" in event_types
            replace_event = next(e for e in events if e["event"] == "replace")
            assert replace_event["data"]["content"] == "Welcome to Mohegan Sun!"


class TestSecurityHeaders:
    def test_security_headers_on_health(self):
        """GET /health includes security headers."""
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.headers.get("x-content-type-options") == "nosniff"
            assert resp.headers.get("x-frame-options") == "DENY"

    @patch("src.agent.graph.chat_stream")
    def test_security_headers_on_chat(self, mock_stream):
        """POST /chat includes security headers."""
        mock_stream.side_effect = _mock_chat_stream(tokens=["Hi"])
        app, _ = _make_test_app()
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "test"})
            assert resp.headers.get("x-content-type-options") == "nosniff"
