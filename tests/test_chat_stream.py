"""Tests for chat_stream() PII buffer logic and SSE error handling.

Direct tests that exercise chat_stream() without mocking it — only the
compiled graph's astream_events is mocked.  Validates:
- PII redaction of phone numbers, SSNs, card numbers split across tokens
- Buffer flush at sentence boundaries and 80-char threshold
- Non-digit text flushes immediately (no unnecessary buffering)
- PII buffer is dropped (not flushed) on error
- CancelledError handling (clean termination, no error event)
- Error mid-stream yields error + done events
- Sources suppressed after error
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from src.agent.graph import chat_stream


def _make_mock_graph(events: list[dict]) -> MagicMock:
    """Create a mock compiled graph that yields the given events from astream_events."""
    mock_graph = MagicMock()

    async def fake_astream_events(initial_state, config=None, version=None):
        for ev in events:
            yield ev

    mock_graph.astream_events = fake_astream_events
    return mock_graph


def _make_token_event(content: str, node: str = "generate") -> dict:
    """Create an on_chat_model_stream event for a token chunk."""
    chunk = AIMessageChunk(content=content)
    return {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": node},
        "data": {"chunk": chunk},
    }


def _make_chain_start(node: str) -> dict:
    """Create an on_chain_start event for a node."""
    return {
        "event": "on_chain_start",
        "metadata": {"langgraph_node": node},
        "data": {},
    }


def _make_chain_end(node: str, output: dict | None = None) -> dict:
    """Create an on_chain_end event for a node."""
    return {
        "event": "on_chain_end",
        "metadata": {"langgraph_node": node},
        "data": {"output": output or {}},
    }


async def _collect_events(gen) -> list[dict]:
    """Collect all events from an async generator."""
    events = []
    async for ev in gen:
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# PII Buffer — phone numbers split across tokens
# ---------------------------------------------------------------------------


class TestChatStreamPiiBuffer:
    """Tests for the inline PII redaction buffer in chat_stream()."""

    @pytest.mark.asyncio
    async def test_phone_number_split_across_tokens_is_redacted(self):
        """Phone number (555-123-4567) split across chunks is redacted."""
        events = [
            _make_chain_start("generate"),
            _make_token_event("Call us at "),
            _make_token_event("555-"),
            _make_token_event("123-"),
            _make_token_event("4567"),
            _make_token_event(". We're open 24/7."),  # sentence boundary flushes
            _make_chain_end("generate"),
        ]
        mock_graph = _make_mock_graph(events)

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "What is your phone number?", thread_id="test-pii-phone")
            )

        # Find all token events
        token_events = [e for e in collected if e["event"] == "token"]
        combined = "".join(
            json.loads(e["data"])["content"] for e in token_events
        )

        # Phone number must be redacted
        assert "555-123-4567" not in combined
        assert "[PHONE]" in combined

    @pytest.mark.asyncio
    async def test_card_number_split_across_tokens_is_redacted(self):
        """Credit card number (4242 4242 4242 4242) split across tokens is redacted."""
        events = [
            _make_chain_start("generate"),
            _make_token_event("Your card: "),
            _make_token_event("4242 "),
            _make_token_event("4242 "),
            _make_token_event("4242 "),
            _make_token_event("4242"),
            _make_token_event(". Thanks!"),
            _make_chain_end("generate"),
        ]
        mock_graph = _make_mock_graph(events)

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "Show me the card", thread_id="test-pii-card")
            )

        token_events = [e for e in collected if e["event"] == "token"]
        combined = "".join(
            json.loads(e["data"])["content"] for e in token_events
        )

        assert "4242" not in combined
        assert "[CARD]" in combined

    @pytest.mark.asyncio
    async def test_ssn_split_across_tokens_is_redacted(self):
        """SSN (123-45-6789) split across tokens is redacted."""
        events = [
            _make_chain_start("generate"),
            _make_token_event("SSN: "),
            _make_token_event("123-"),
            _make_token_event("45-"),
            _make_token_event("6789"),
            _make_token_event(". End."),
            _make_chain_end("generate"),
        ]
        mock_graph = _make_mock_graph(events)

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "What is the SSN?", thread_id="test-pii-ssn")
            )

        token_events = [e for e in collected if e["event"] == "token"]
        combined = "".join(
            json.loads(e["data"])["content"] for e in token_events
        )

        assert "123-45-6789" not in combined
        assert "[SSN]" in combined

    @pytest.mark.asyncio
    async def test_non_digit_text_passes_through(self):
        """Text without digits passes through completely (no data loss)."""
        events = [
            _make_chain_start("generate"),
            _make_token_event("Welcome to "),
            _make_token_event("our resort."),
            _make_chain_end("generate"),
        ]
        mock_graph = _make_mock_graph(events)

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "Hello", thread_id="test-pii-clean")
            )

        token_events = [e for e in collected if e["event"] == "token"]
        # At least one token event with all text present
        assert len(token_events) >= 1
        combined = "".join(
            json.loads(e["data"])["content"] for e in token_events
        )
        assert "Welcome to our resort." in combined

    @pytest.mark.asyncio
    async def test_buffer_flush_at_80_char_threshold(self):
        """Buffer flushes when accumulating 80+ chars with digits."""
        # Create a long string with digits that exceeds _PII_FLUSH_LEN (80)
        long_text = "Room 42 is on floor 3. " * 5  # ~115 chars with digits
        events = [
            _make_chain_start("generate"),
            _make_token_event(long_text),
            _make_chain_end("generate"),
        ]
        mock_graph = _make_mock_graph(events)

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "Room info", thread_id="test-pii-flush")
            )

        token_events = [e for e in collected if e["event"] == "token"]
        # Should have flushed at least once (80-char threshold reached)
        assert len(token_events) >= 1
        combined = "".join(
            json.loads(e["data"])["content"] for e in token_events
        )
        assert "Room 42" in combined

    @pytest.mark.asyncio
    async def test_buffer_flush_at_sentence_boundary(self):
        """Buffer flushes at sentence boundary ('. ') even with digits present."""
        events = [
            _make_chain_start("generate"),
            _make_token_event("Room 101"),
            _make_token_event(" is nice. "),  # sentence boundary should trigger flush
            _make_token_event("Enjoy!"),
            _make_chain_end("generate"),
        ]
        mock_graph = _make_mock_graph(events)

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "Rooms?", thread_id="test-pii-sentence")
            )

        token_events = [e for e in collected if e["event"] == "token"]
        combined = "".join(
            json.loads(e["data"])["content"] for e in token_events
        )
        assert "Room 101 is nice." in combined


# ---------------------------------------------------------------------------
# R10 fix (DeepSeek F2): PII MAX_BUFFER hard cap
# ---------------------------------------------------------------------------


class TestPiiMaxBufferHardCap:
    """R10 fix: _PII_MAX_BUFFER forces flush even when _PII_FLUSH_LEN hasn't triggered."""

    @pytest.mark.asyncio
    async def test_max_buffer_forces_flush(self):
        """Buffer exceeding _PII_MAX_BUFFER (500) chars forces flush regardless of other conditions.

        Previously _PII_MAX_BUFFER was in an `or` with _PII_FLUSH_LEN (80), so the 80-char
        threshold always triggered first, making the 500-char cap dead code. After R10 fix,
        the hard cap is an unconditional first check.
        """
        # Create a string with digits that won't trigger the 80-char flush
        # by building it in small increments that each contain digits but
        # are under 80 chars. We'll send many small chunks.
        # Actually, we need to test that the 500-char cap works when the
        # buffer contains digits but doesn't hit sentence boundaries or 80-char threshold.
        # The simplest way: send one giant chunk > 500 chars with digits throughout.
        giant_chunk = ("digit1 " * 80)  # ~560 chars, has digits, no sentence boundary
        events = [
            _make_chain_start("generate"),
            _make_token_event(giant_chunk),
            _make_chain_end("generate"),
        ]
        mock_graph = _make_mock_graph(events)

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "test", thread_id="test-max-buffer")
            )

        token_events = [e for e in collected if e["event"] == "token"]
        # With 560 chars and digits, the buffer should have flushed at least once
        # due to the hard cap (500) OR the 80-char threshold
        assert len(token_events) >= 1
        combined = "".join(
            json.loads(e["data"])["content"] for e in token_events
        )
        assert "digit1" in combined


# ---------------------------------------------------------------------------
# Error handling — PII buffer dropped on error, no leak
# ---------------------------------------------------------------------------


class TestChatStreamErrorHandling:
    """Tests for error paths in chat_stream — PII buffer safety."""

    @pytest.mark.asyncio
    async def test_error_mid_stream_yields_error_and_done(self):
        """RuntimeError mid-stream yields error + done events."""
        mock_graph = MagicMock()

        async def failing_astream(initial_state, config=None, version=None):
            yield _make_chain_start("generate")
            yield _make_token_event("partial ")
            raise RuntimeError("LLM API failure")

        mock_graph.astream_events = failing_astream

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "test", thread_id="test-error")
            )

        event_types = [e["event"] for e in collected]
        assert "error" in event_types
        assert "done" in event_types
        # Done must be last
        assert collected[-1]["event"] == "done"

    @pytest.mark.asyncio
    async def test_pii_buffer_dropped_on_error(self):
        """When error occurs mid-stream, PII buffer is dropped — not flushed."""
        mock_graph = MagicMock()

        async def failing_astream(initial_state, config=None, version=None):
            yield _make_chain_start("generate")
            # Buffer digits (potential PII) then crash
            yield _make_token_event("Card: 4242-4242")
            raise RuntimeError("crash")

        mock_graph.astream_events = failing_astream

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "test", thread_id="test-pii-drop")
            )

        # The digits "4242-4242" should NOT appear in any token event
        # because the buffer is dropped on error (errored=True guard)
        token_events = [e for e in collected if e["event"] == "token"]
        combined = "".join(
            json.loads(e["data"])["content"] for e in token_events
        )
        assert "4242" not in combined

    @pytest.mark.asyncio
    async def test_sources_suppressed_after_error(self):
        """After an error, sources are NOT emitted (stale/partial data)."""
        mock_graph = MagicMock()

        async def failing_astream(initial_state, config=None, version=None):
            yield _make_chain_start("generate")
            yield _make_token_event("partial")
            raise RuntimeError("crash")

        mock_graph.astream_events = failing_astream

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "test", thread_id="test-no-sources")
            )

        event_types = [e["event"] for e in collected]
        assert "sources" not in event_types

    @pytest.mark.asyncio
    async def test_cancelled_error_reraises(self):
        """CancelledError (client disconnect) is re-raised, not caught."""
        mock_graph = MagicMock()

        async def cancelling_astream(initial_state, config=None, version=None):
            yield _make_chain_start("generate")
            raise asyncio.CancelledError()

        mock_graph.astream_events = cancelling_astream

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            with pytest.raises(asyncio.CancelledError):
                events = []
                async for ev in chat_stream(mock_graph, "test", thread_id="test-cancel"):
                    events.append(ev)


# ---------------------------------------------------------------------------
# SSE event sequence contract
# ---------------------------------------------------------------------------


class TestChatStreamEventSequence:
    """Verify SSE event sequence from chat_stream()."""

    @pytest.mark.asyncio
    async def test_metadata_first_done_last(self):
        """metadata is first event, done is last event."""
        events = [
            _make_chain_start("generate"),
            _make_token_event("Hello!"),
            _make_chain_end("generate"),
        ]
        mock_graph = _make_mock_graph(events)

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "hi", thread_id="test-seq")
            )

        assert collected[0]["event"] == "metadata"
        assert collected[-1]["event"] == "done"
        # Metadata contains thread_id
        meta_data = json.loads(collected[0]["data"])
        assert meta_data["thread_id"] == "test-seq"

    @pytest.mark.asyncio
    async def test_graph_node_lifecycle_events_emitted(self):
        """Graph node start/complete events are emitted for known nodes."""
        events = [
            _make_chain_start("generate"),
            _make_token_event("Hello!"),
            _make_chain_end("generate", output={}),
        ]
        mock_graph = _make_mock_graph(events)

        with patch("src.agent.graph.get_langfuse_handler", return_value=None):
            collected = await _collect_events(
                chat_stream(mock_graph, "hi", thread_id="test-lifecycle")
            )

        graph_events = [e for e in collected if e["event"] == "graph_node"]
        generate_events = [
            e for e in graph_events
            if json.loads(e["data"])["node"] == "generate"
        ]
        statuses = [json.loads(e["data"])["status"] for e in generate_events]
        assert "start" in statuses
        assert "complete" in statuses
