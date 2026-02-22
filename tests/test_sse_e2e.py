"""End-to-end SSE streaming test.

Verifies the full SSE stream lifecycle through the /chat endpoint:
metadata -> tokens/replace -> sources -> done.

Uses mocked LLM and agent to avoid API key requirements while
testing the SSE infrastructure, event ordering, and error handling.
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _disable_api_key_auth(monkeypatch):
    """Ensure API key auth is disabled for all SSE tests."""
    monkeypatch.setenv("API_KEY", "")
    # Clear settings cache so middleware sees the empty API_KEY
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sse_test_app():
    """Create a test app with a mock agent that yields SSE events."""
    __import__("src.api.app")
    app_module = sys.modules["src.api.app"]

    @asynccontextmanager
    async def test_lifespan(app):
        app.state.agent = MagicMock()
        app.state.property_data = {"property": {"name": "Test Casino"}}
        app.state.ready = True
        yield
        app.state.ready = False

    original_lifespan = app_module.lifespan
    app_module.lifespan = test_lifespan
    try:
        app = app_module.create_app()
    finally:
        app_module.lifespan = original_lifespan

    return app


async def _mock_chat_stream(graph, message, thread_id=None, request_id=None):
    """Mock chat_stream that yields a realistic event sequence."""
    yield {"event": "metadata", "data": json.dumps({"thread_id": "test-thread-123"})}
    yield {"event": "graph_node", "data": json.dumps({"node": "router", "status": "start"})}
    yield {"event": "graph_node", "data": json.dumps({"node": "router", "status": "complete"})}
    yield {"event": "graph_node", "data": json.dumps({"node": "generate", "status": "start"})}
    yield {"event": "token", "data": "Welcome "}
    yield {"event": "token", "data": "to "}
    yield {"event": "token", "data": "the resort."}
    yield {"event": "graph_node", "data": json.dumps({"node": "generate", "status": "complete"})}
    yield {"event": "sources", "data": json.dumps(["dining", "general"])}
    yield {"event": "done", "data": json.dumps({"done": True})}


async def _mock_replace_stream(graph, message, thread_id=None, request_id=None):
    """Mock chat_stream that uses replace event (greeting/off_topic)."""
    yield {"event": "metadata", "data": json.dumps({"thread_id": "test-thread-greet"})}
    yield {"event": "replace", "data": "Hello! Welcome to the resort. How can I help you today?"}
    yield {"event": "done", "data": json.dumps({"done": True})}


def _parse_sse_events(response_text: str) -> list[dict]:
    """Parse SSE response text into a list of event dicts."""
    events = []
    current_event = {}
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            if current_event:
                events.append(current_event)
                current_event = {}
            continue
        if line.startswith("event:"):
            current_event["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_event["data"] = line[len("data:"):].strip()
    if current_event:
        events.append(current_event)
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSSEStreamLifecycle:
    """Full SSE streaming lifecycle tests."""

    def test_sse_stream_event_ordering(self):
        """Verify metadata -> tokens -> sources -> done event order."""
        app = _make_sse_test_app()
        with patch("src.agent.graph.chat_stream", side_effect=_mock_chat_stream):
            with TestClient(app) as client:
                response = client.post(
                    "/chat",
                    json={"message": "Tell me about restaurants"},
                )
                assert response.status_code == 200
                assert "text/event-stream" in response.headers.get("content-type", "")

                events = _parse_sse_events(response.text)
                event_types = [e.get("event") for e in events]

                # First event must be metadata
                assert event_types[0] == "metadata"
                # Last event must be done
                assert event_types[-1] == "done"
                # Must contain tokens
                assert "token" in event_types

    def test_sse_metadata_contains_thread_id(self):
        """Metadata event must contain thread_id."""
        app = _make_sse_test_app()
        with patch("src.agent.graph.chat_stream", side_effect=_mock_chat_stream):
            with TestClient(app) as client:
                response = client.post(
                    "/chat",
                    json={"message": "Hello"},
                )
                events = _parse_sse_events(response.text)
                metadata = next(e for e in events if e.get("event") == "metadata")
                data = json.loads(metadata["data"])
                assert "thread_id" in data
                assert data["thread_id"] == "test-thread-123"

    def test_sse_token_events_contain_text(self):
        """Token events should contain non-empty text data."""
        app = _make_sse_test_app()
        with patch("src.agent.graph.chat_stream", side_effect=_mock_chat_stream):
            with TestClient(app) as client:
                response = client.post(
                    "/chat",
                    json={"message": "Tell me about dining"},
                )
                events = _parse_sse_events(response.text)
                token_events = [e for e in events if e.get("event") == "token"]
                assert len(token_events) >= 1
                # All tokens should have non-empty data
                for token in token_events:
                    assert token.get("data"), "Token event has empty data"

    def test_sse_replace_event_for_non_streaming(self):
        """Non-streaming responses use replace event instead of tokens."""
        app = _make_sse_test_app()
        with patch("src.agent.graph.chat_stream", side_effect=_mock_replace_stream):
            with TestClient(app) as client:
                response = client.post(
                    "/chat",
                    json={"message": "Hello"},
                )
                events = _parse_sse_events(response.text)
                event_types = [e.get("event") for e in events]

                assert "metadata" in event_types
                assert "replace" in event_types
                assert "done" in event_types
                # No token events for non-streaming paths
                assert "token" not in event_types

    def test_sse_done_event_signals_completion(self):
        """Done event contains {done: true}."""
        app = _make_sse_test_app()
        with patch("src.agent.graph.chat_stream", side_effect=_mock_chat_stream):
            with TestClient(app) as client:
                response = client.post(
                    "/chat",
                    json={"message": "Test"},
                )
                events = _parse_sse_events(response.text)
                done_event = next(e for e in events if e.get("event") == "done")
                data = json.loads(done_event["data"])
                assert data.get("done") is True

    def test_sse_sources_event_contains_categories(self):
        """Sources event lists knowledge-base categories."""
        app = _make_sse_test_app()
        with patch("src.agent.graph.chat_stream", side_effect=_mock_chat_stream):
            with TestClient(app) as client:
                response = client.post(
                    "/chat",
                    json={"message": "What restaurants are open?"},
                )
                events = _parse_sse_events(response.text)
                sources_events = [e for e in events if e.get("event") == "sources"]
                assert len(sources_events) >= 1
                sources_data = json.loads(sources_events[0]["data"])
                assert isinstance(sources_data, list)
                assert len(sources_data) > 0


class TestSSEErrorHandling:
    """Error handling in SSE streams."""

    def test_agent_not_initialized_returns_503(self):
        """When agent is None, /chat returns 503."""
        __import__("src.api.app")
        app_module = sys.modules["src.api.app"]

        @asynccontextmanager
        async def no_agent_lifespan(app):
            app.state.agent = None
            app.state.property_data = {}
            app.state.ready = False
            yield

        original = app_module.lifespan
        app_module.lifespan = no_agent_lifespan
        try:
            app = app_module.create_app()
        finally:
            app_module.lifespan = original

        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={"message": "Hello"},
            )
            assert response.status_code == 503

    def test_empty_message_returns_422(self):
        """Empty message in ChatRequest is rejected."""
        app = _make_sse_test_app()
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={"message": ""},
            )
            assert response.status_code == 422

    def test_message_too_long_returns_422(self):
        """Message exceeding max_length (4096) is rejected."""
        app = _make_sse_test_app()
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={"message": "x" * 4097},
            )
            assert response.status_code == 422
