"""Tests for the LangGraph agent (graph compilation, chat extraction, streaming)."""

import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage


class TestBuildGraph:
    def test_graph_compiles(self):
        """build_graph() returns a compiled graph without errors."""
        from src.agent.graph import build_graph

        graph = build_graph()
        assert graph is not None
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    def test_graph_has_11_nodes(self):
        """The compiled graph contains exactly 11 user-defined nodes (v2.1)."""
        from src.agent.graph import build_graph

        graph = build_graph()
        # get_graph() returns a DrawableGraph; its nodes include __start__ and __end__
        all_nodes = set(graph.get_graph().nodes)
        user_nodes = all_nodes - {"__start__", "__end__"}
        expected = {
            "compliance_gate", "router", "retrieve", "whisper_planner",
            "generate", "validate", "persona_envelope", "respond",
            "fallback", "greeting", "off_topic",
        }
        assert user_nodes == expected

    def test_graph_accepts_custom_checkpointer(self):
        """build_graph() accepts an explicit checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver
        from src.agent.graph import build_graph

        cp = MemorySaver()
        graph = build_graph(checkpointer=cp)
        assert graph is not None


class TestChatResponseExtraction:
    """Unit tests for chat() response extraction logic (no API key needed)."""

    @pytest.mark.asyncio
    async def test_extracts_last_ai_message(self):
        """chat() returns the last AIMessage content."""
        from src.agent.graph import chat

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="What restaurants?"),
                AIMessage(content="Mohegan Sun has great restaurants!"),
            ],
            "sources_used": ["restaurants"],
        }

        result = await chat(mock_graph, "What restaurants?")
        assert result["response"] == "Mohegan Sun has great restaurants!"
        assert "thread_id" in result

    @pytest.mark.asyncio
    async def test_extracts_sources_from_state(self):
        """chat() extracts sources from result['sources_used']."""
        from src.agent.graph import chat

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="Tell me about dining and shows"),
                AIMessage(content="Here is dining and entertainment info."),
            ],
            "sources_used": ["restaurants", "entertainment"],
        }

        result = await chat(mock_graph, "Tell me about dining and shows")
        assert "restaurants" in result["sources"]
        assert "entertainment" in result["sources"]

    @pytest.mark.asyncio
    async def test_empty_response_when_no_ai_message(self):
        """chat() returns empty string if no AI message found."""
        from src.agent.graph import chat

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [],
            "sources_used": [],
        }

        result = await chat(mock_graph, "test")
        assert result["response"] == ""
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_thread_id_preserved(self):
        """chat() preserves thread_id when provided."""
        from src.agent.graph import chat

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="Response")],
            "sources_used": [],
        }

        result = await chat(mock_graph, "test", thread_id="my-thread-123")
        assert result["thread_id"] == "my-thread-123"

    @pytest.mark.asyncio
    async def test_thread_id_generated_when_none(self):
        """chat() generates a UUID thread_id when none is provided."""
        from src.agent.graph import chat

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="Response")],
            "sources_used": [],
        }

        result = await chat(mock_graph, "test")
        assert result["thread_id"] is not None
        assert len(result["thread_id"]) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_multi_content_ai_message(self):
        """chat() handles AI messages with list content (multi-part)."""
        from src.agent.graph import chat

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [
                AIMessage(content=[{"type": "text", "text": "Here are the results"}]),
            ],
            "sources_used": [],
        }
        result = await chat(mock_graph, "test")
        # Should convert non-string content to string
        assert len(result["response"]) > 0


class TestChatSourceDedup:
    """Verify source deduplication in chat() response."""

    @pytest.mark.asyncio
    async def test_sources_from_state_are_not_duplicated(self):
        """Sources from sources_used are passed through as-is."""
        from src.agent.graph import chat

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="Response")],
            "sources_used": ["restaurants"],
        }
        result = await chat(mock_graph, "test")
        assert result["sources"] == ["restaurants"]


class TestChatStream:
    """Tests for the chat_stream async generator."""

    @pytest.mark.asyncio
    async def test_stream_yields_metadata_first(self):
        """First event from chat_stream is metadata with thread_id."""
        from src.agent.graph import chat_stream

        mock_graph = AsyncMock()

        async def empty_stream(*args, **kwargs):
            return
            yield  # make it an async generator

        mock_graph.astream_events = empty_stream

        events = []
        async for event in chat_stream(mock_graph, "test"):
            events.append(event)

        assert events[0]["event"] == "metadata"
        data = json.loads(events[0]["data"])
        assert "thread_id" in data

    @pytest.mark.asyncio
    async def test_stream_yields_done_last(self):
        """Last event from chat_stream is done."""
        from src.agent.graph import chat_stream

        mock_graph = AsyncMock()

        async def empty_stream(*args, **kwargs):
            return
            yield

        mock_graph.astream_events = empty_stream

        events = []
        async for event in chat_stream(mock_graph, "test"):
            events.append(event)

        assert events[-1]["event"] == "done"

    @pytest.mark.asyncio
    async def test_stream_preserves_thread_id(self):
        """chat_stream uses provided thread_id."""
        from src.agent.graph import chat_stream

        mock_graph = AsyncMock()

        async def empty_stream(*args, **kwargs):
            return
            yield

        mock_graph.astream_events = empty_stream

        events = []
        async for event in chat_stream(mock_graph, "test", thread_id="my-uuid-123"):
            events.append(event)

        data = json.loads(events[0]["data"])
        assert data["thread_id"] == "my-uuid-123"

    @pytest.mark.asyncio
    async def test_stream_error_emits_error_event(self):
        """Exceptions during streaming emit an error event."""
        from src.agent.graph import chat_stream

        mock_graph = AsyncMock()

        async def failing_stream(*args, **kwargs):
            raise RuntimeError("LLM API failure")
            yield  # make it an async generator

        mock_graph.astream_events = failing_stream

        events = []
        async for event in chat_stream(mock_graph, "test"):
            events.append(event)

        event_types = [e["event"] for e in events]
        assert "error" in event_types
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_stream_token_events(self):
        """Token events from generate node are yielded."""
        from src.agent.graph import chat_stream

        mock_graph = AsyncMock()

        async def token_stream(*args, **kwargs):
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "generate"},
                "data": {"chunk": AIMessageChunk(content="Hello ")},
            }
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "generate"},
                "data": {"chunk": AIMessageChunk(content="world!")},
            }

        mock_graph.astream_events = token_stream

        events = []
        async for event in chat_stream(mock_graph, "test"):
            events.append(event)

        token_events = [e for e in events if e["event"] == "token"]
        # StreamingPIIRedactor may combine short chunks in the lookahead buffer.
        # Verify content integrity (no data loss) rather than per-chunk emission.
        assert len(token_events) >= 1
        combined = "".join(
            json.loads(e["data"])["content"] for e in token_events
        )
        assert "Hello " in combined
        assert "world!" in combined

    @pytest.mark.asyncio
    async def test_stream_replace_event_for_greeting(self):
        """Greeting node output emits a 'replace' event."""
        from src.agent.graph import chat_stream

        mock_graph = AsyncMock()

        async def greeting_stream(*args, **kwargs):
            yield {
                "event": "on_chain_end",
                "metadata": {"langgraph_node": "greeting"},
                "data": {
                    "output": {
                        "messages": [AIMessage(content="Welcome to Mohegan Sun!")],
                        "sources_used": [],
                    }
                },
            }

        mock_graph.astream_events = greeting_stream

        events = []
        async for event in chat_stream(mock_graph, "test"):
            events.append(event)

        replace_events = [e for e in events if e["event"] == "replace"]
        assert len(replace_events) == 1
        assert json.loads(replace_events[0]["data"])["content"] == "Welcome to Mohegan Sun!"

    @pytest.mark.asyncio
    async def test_stream_sources_from_respond_node(self):
        """Sources from respond node are captured and emitted."""
        from src.agent.graph import chat_stream

        mock_graph = AsyncMock()

        async def respond_stream(*args, **kwargs):
            yield {
                "event": "on_chain_end",
                "metadata": {"langgraph_node": "respond"},
                "data": {
                    "output": {
                        "sources_used": ["restaurants", "entertainment"],
                        "retry_feedback": None,
                    }
                },
            }

        mock_graph.astream_events = respond_stream

        events = []
        async for event in chat_stream(mock_graph, "test"):
            events.append(event)

        sources_events = [e for e in events if e["event"] == "sources"]
        assert len(sources_events) == 1
        sources_data = json.loads(sources_events[0]["data"])
        assert "restaurants" in sources_data["sources"]
        assert "entertainment" in sources_data["sources"]


class TestNodeConstants:
    """Node name constants prevent silent breakage from string typos."""

    def test_constants_exported(self):
        """All 11 node constants are importable from graph module."""
        from src.agent.graph import (
            NODE_COMPLIANCE_GATE,
            NODE_FALLBACK,
            NODE_GENERATE,
            NODE_GREETING,
            NODE_OFF_TOPIC,
            NODE_PERSONA,
            NODE_RESPOND,
            NODE_RETRIEVE,
            NODE_ROUTER,
            NODE_VALIDATE,
            NODE_WHISPER,
        )

        assert NODE_COMPLIANCE_GATE == "compliance_gate"
        assert NODE_ROUTER == "router"
        assert NODE_RETRIEVE == "retrieve"
        assert NODE_GENERATE == "generate"
        assert NODE_VALIDATE == "validate"
        assert NODE_PERSONA == "persona_envelope"
        assert NODE_RESPOND == "respond"
        assert NODE_FALLBACK == "fallback"
        assert NODE_GREETING == "greeting"
        assert NODE_OFF_TOPIC == "off_topic"
        assert NODE_WHISPER == "whisper_planner"

    def test_graph_nodes_match_constants(self):
        """Graph node names match the defined constants."""
        from src.agent.graph import (
            NODE_COMPLIANCE_GATE,
            NODE_FALLBACK,
            NODE_GENERATE,
            NODE_GREETING,
            NODE_OFF_TOPIC,
            NODE_PERSONA,
            NODE_RESPOND,
            NODE_RETRIEVE,
            NODE_ROUTER,
            NODE_VALIDATE,
            NODE_WHISPER,
            build_graph,
        )

        graph = build_graph()
        all_nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
        expected = {
            NODE_COMPLIANCE_GATE, NODE_ROUTER, NODE_RETRIEVE, NODE_WHISPER,
            NODE_GENERATE, NODE_VALIDATE, NODE_PERSONA, NODE_RESPOND,
            NODE_FALLBACK, NODE_GREETING, NODE_OFF_TOPIC,
        }
        assert all_nodes == expected


class TestHitlInterrupt:
    """HITL interrupt support via ENABLE_HITL_INTERRUPT setting."""

    def test_hitl_disabled_by_default(self):
        """HITL interrupt is disabled by default."""
        from src.config import Settings

        s = Settings()
        assert s.ENABLE_HITL_INTERRUPT is False

    def test_hitl_graph_compiles_with_interrupt(self):
        """Graph compiles with ENABLE_HITL_INTERRUPT=True."""
        import os
        from unittest.mock import patch as mock_patch

        from src.agent.graph import build_graph

        with mock_patch.dict(os.environ, {"ENABLE_HITL_INTERRUPT": "true"}):
            graph = build_graph()
            assert graph is not None


class TestExtractNodeMetadata:
    """Unit tests for _extract_node_metadata() helper."""

    def test_router_metadata(self):
        """Router node extracts query_type and confidence."""
        from src.agent.graph import _extract_node_metadata

        output = {"query_type": "property_qa", "router_confidence": 0.95}
        meta = _extract_node_metadata("router", output)
        assert meta["query_type"] == "property_qa"
        assert meta["confidence"] == 0.95

    def test_retrieve_metadata(self):
        """Retrieve node extracts doc_count."""
        from src.agent.graph import _extract_node_metadata

        output = {"retrieved_context": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}
        meta = _extract_node_metadata("retrieve", output)
        assert meta["doc_count"] == 3

    def test_validate_metadata(self):
        """Validate node extracts result."""
        from src.agent.graph import _extract_node_metadata

        output = {"validation_result": "PASS"}
        meta = _extract_node_metadata("validate", output)
        assert meta["result"] == "PASS"

    def test_respond_metadata(self):
        """Respond node extracts sources."""
        from src.agent.graph import _extract_node_metadata

        output = {"sources_used": ["restaurants", "entertainment"]}
        meta = _extract_node_metadata("respond", output)
        assert meta["sources"] == ["restaurants", "entertainment"]

    def test_whisper_metadata(self):
        """Whisper planner node extracts has_plan."""
        from src.agent.graph import _extract_node_metadata

        output_with_plan = {"whisper_plan": {"next_topic": "dining"}}
        meta = _extract_node_metadata("whisper_planner", output_with_plan)
        assert meta["has_plan"] is True

        output_no_plan = {"whisper_plan": None}
        meta = _extract_node_metadata("whisper_planner", output_no_plan)
        assert meta["has_plan"] is False

    def test_unknown_node_returns_empty(self):
        """Unknown or unhandled nodes return empty dict."""
        from src.agent.graph import _extract_node_metadata

        assert _extract_node_metadata("generate", {"some": "data"}) == {}
        assert _extract_node_metadata("fallback", {}) == {}

    def test_non_dict_output_returns_empty(self):
        """Non-dict output returns empty dict."""
        from src.agent.graph import _extract_node_metadata

        assert _extract_node_metadata("router", "not a dict") == {}
        assert _extract_node_metadata("router", None) == {}
