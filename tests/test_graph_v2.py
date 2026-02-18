"""Tests for v2 graph topology: compliance gate, persona envelope, routing.

Validates the 10-node graph (v2) changes without requiring API keys.
All LLM calls are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import (
    NODE_COMPLIANCE_GATE,
    NODE_FALLBACK,
    NODE_GENERATE,
    NODE_GREETING,
    NODE_OFF_TOPIC,
    NODE_PERSONA,
    NODE_ROUTER,
    NODE_WHISPER,
    _extract_node_metadata,
    _route_after_validate_v2,
    build_graph,
    route_from_compliance,
)
from src.agent.state import CasinoHostState, PropertyQAState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    """Build a minimal PropertyQAState dict with defaults."""
    base = {
        "messages": [],
        "query_type": None,
        "router_confidence": 0.0,
        "retrieved_context": [],
        "validation_result": None,
        "retry_count": 0,
        "skip_validation": False,
        "retry_feedback": None,
        "current_time": "Monday 3 PM",
        "sources_used": [],
        "extracted_fields": {},
        "whisper_plan": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


class TestGraphV2Compilation:
    """v2.1 graph compiles with 11 nodes."""

    def test_compiles_without_error(self):
        """build_graph() returns a compiled graph without errors."""
        graph = build_graph()
        assert graph is not None

    def test_has_11_nodes(self):
        """The compiled graph contains exactly 11 user-defined nodes."""
        graph = build_graph()
        all_nodes = set(graph.get_graph().nodes)
        user_nodes = all_nodes - {"__start__", "__end__"}
        expected = {
            "compliance_gate", "router", "retrieve", "whisper_planner",
            "generate", "validate", "persona_envelope", "respond",
            "fallback", "greeting", "off_topic",
        }
        assert user_nodes == expected

    def test_start_goes_to_compliance_gate(self):
        """START edge goes to compliance_gate (not router)."""
        graph = build_graph()
        drawable = graph.get_graph()
        # Find edges from __start__
        start_edges = [
            e for e in drawable.edges
            if e.source == "__start__"
        ]
        assert len(start_edges) == 1
        assert start_edges[0].target == "compliance_gate"

    def test_retrieve_to_whisper_to_generate(self):
        """retrieve → whisper_planner → generate edge chain exists."""
        graph = build_graph()
        drawable = graph.get_graph()

        # retrieve → whisper_planner
        retrieve_edges = [e for e in drawable.edges if e.source == "retrieve"]
        assert any(e.target == "whisper_planner" for e in retrieve_edges)

        # whisper_planner → generate
        whisper_edges = [e for e in drawable.edges if e.source == "whisper_planner"]
        assert any(e.target == "generate" for e in whisper_edges)

        # No direct retrieve → generate edge
        assert not any(e.target == "generate" for e in retrieve_edges)


# ---------------------------------------------------------------------------
# route_from_compliance
# ---------------------------------------------------------------------------


class TestRouteFromCompliance:
    """Tests for the compliance gate routing function."""

    def test_none_routes_to_router(self):
        """query_type=None means all guardrails passed, route to router."""
        state = _state(query_type=None)
        assert route_from_compliance(state) == NODE_ROUTER

    def test_greeting_routes_to_greeting(self):
        """query_type='greeting' routes directly to greeting node."""
        state = _state(query_type="greeting")
        assert route_from_compliance(state) == NODE_GREETING

    def test_off_topic_routes_to_off_topic(self):
        """query_type='off_topic' routes to off_topic node."""
        state = _state(query_type="off_topic")
        assert route_from_compliance(state) == NODE_OFF_TOPIC

    def test_gambling_advice_routes_to_off_topic(self):
        """query_type='gambling_advice' routes to off_topic node."""
        state = _state(query_type="gambling_advice")
        assert route_from_compliance(state) == NODE_OFF_TOPIC

    def test_patron_privacy_routes_to_off_topic(self):
        """query_type='patron_privacy' routes to off_topic node."""
        state = _state(query_type="patron_privacy")
        assert route_from_compliance(state) == NODE_OFF_TOPIC

    def test_age_verification_routes_to_off_topic(self):
        """query_type='age_verification' routes to off_topic node."""
        state = _state(query_type="age_verification")
        assert route_from_compliance(state) == NODE_OFF_TOPIC


# ---------------------------------------------------------------------------
# _route_after_validate_v2
# ---------------------------------------------------------------------------


class TestRouteAfterValidateV2:
    """Tests for v2 validate routing (PASS → persona_envelope)."""

    def test_pass_routes_to_persona(self):
        """PASS routes to persona_envelope (not directly to respond)."""
        state = _state(validation_result="PASS")
        assert _route_after_validate_v2(state) == NODE_PERSONA

    def test_retry_routes_to_generate(self):
        """RETRY routes back to generate (host_agent) node."""
        state = _state(validation_result="RETRY")
        assert _route_after_validate_v2(state) == NODE_GENERATE

    def test_fail_routes_to_fallback(self):
        """FAIL routes to fallback node."""
        state = _state(validation_result="FAIL")
        assert _route_after_validate_v2(state) == NODE_FALLBACK

    def test_default_routes_to_fallback(self):
        """Missing/unknown validation_result routes to fallback (safe default)."""
        state = _state(validation_result=None)
        assert _route_after_validate_v2(state) == NODE_FALLBACK


# ---------------------------------------------------------------------------
# Persona envelope node
# ---------------------------------------------------------------------------


class TestPersonaEnvelope:
    """Tests for the persona_envelope_node."""

    @pytest.mark.asyncio
    async def test_web_mode_passthrough(self):
        """PERSONA_MAX_CHARS=0 (web mode): returns empty dict (no modification)."""
        from src.agent.persona import persona_envelope_node

        state = _state(messages=[
            HumanMessage(content="What restaurants?"),
            AIMessage(content="Mohegan Sun has great dining options."),
        ])

        with patch("src.agent.persona.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(PERSONA_MAX_CHARS=0)
            result = await persona_envelope_node(state)

        assert result == {}

    @pytest.mark.asyncio
    async def test_sms_mode_truncates_long_message(self):
        """PERSONA_MAX_CHARS=160 (SMS mode): truncates long messages with ellipsis."""
        from src.agent.persona import persona_envelope_node

        long_content = "A" * 300
        state = _state(messages=[
            HumanMessage(content="Tell me about dining"),
            AIMessage(content=long_content),
        ])

        with patch("src.agent.persona.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(PERSONA_MAX_CHARS=160)
            result = await persona_envelope_node(state)

        assert "messages" in result
        truncated = result["messages"][0].content
        assert len(truncated) == 160
        assert truncated.endswith("...")

    @pytest.mark.asyncio
    async def test_sms_mode_short_message_not_truncated(self):
        """Short messages are not truncated even in SMS mode."""
        from src.agent.persona import persona_envelope_node

        state = _state(messages=[
            HumanMessage(content="Hi"),
            AIMessage(content="Welcome!"),
        ])

        with patch("src.agent.persona.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(PERSONA_MAX_CHARS=160)
            result = await persona_envelope_node(state)

        assert result == {}

    @pytest.mark.asyncio
    async def test_no_ai_message_returns_empty(self):
        """No AI message in state returns empty dict."""
        from src.agent.persona import persona_envelope_node

        state = _state(messages=[HumanMessage(content="Hello")])

        with patch("src.agent.persona.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(PERSONA_MAX_CHARS=160)
            result = await persona_envelope_node(state)

        assert result == {}


# ---------------------------------------------------------------------------
# Node metadata extraction
# ---------------------------------------------------------------------------


class TestExtractNodeMetadataV2:
    """Tests for _extract_node_metadata with v2 nodes."""

    def test_compliance_gate_metadata(self):
        """Compliance gate extracts query_type and confidence."""
        output = {"query_type": "gambling_advice", "router_confidence": 1.0}
        meta = _extract_node_metadata("compliance_gate", output)
        assert meta["query_type"] == "gambling_advice"
        assert meta["confidence"] == 1.0

    def test_compliance_gate_passthrough(self):
        """Compliance gate passing through (query_type=None)."""
        output = {"query_type": None, "router_confidence": 0.0}
        meta = _extract_node_metadata("compliance_gate", output)
        assert meta["query_type"] is None

    def test_persona_returns_empty(self):
        """Persona envelope has no special metadata."""
        output = {"messages": []}
        meta = _extract_node_metadata("persona_envelope", output)
        assert meta == {}


# ---------------------------------------------------------------------------
# Initial state v2 fields
# ---------------------------------------------------------------------------


class TestInitialStateV2:
    """Tests for v2 fields in _initial_state."""

    def test_v2_fields_present(self):
        """_initial_state includes v2 fields with correct defaults."""
        from src.agent.graph import _initial_state

        state = _initial_state("Hello")
        assert state["extracted_fields"] == {}
        assert state["whisper_plan"] is None

    def test_v1_fields_unchanged(self):
        """_initial_state preserves all v1 field defaults."""
        from src.agent.graph import _initial_state

        state = _initial_state("Hello")
        assert state["query_type"] is None
        assert state["router_confidence"] == 0.0
        assert state["retrieved_context"] == []
        assert state["validation_result"] is None
        assert state["retry_count"] == 0
        assert state["skip_validation"] is False
        assert state["retry_feedback"] is None
        assert state["sources_used"] == []


# ---------------------------------------------------------------------------
# CasinoHostState alias
# ---------------------------------------------------------------------------


class TestCasinoHostStateAlias:
    """CasinoHostState is a backward-compatible alias."""

    def test_alias_is_same_type(self):
        """CasinoHostState is PropertyQAState."""
        assert CasinoHostState is PropertyQAState

    def test_exported_from_package(self):
        """CasinoHostState is importable from the agent package."""
        from src.agent import CasinoHostState as imported

        assert imported is PropertyQAState


# ---------------------------------------------------------------------------
# Node constants
# ---------------------------------------------------------------------------


class TestNodeConstantsV2:
    """v2 node constants are correct."""

    def test_compliance_gate_constant(self):
        assert NODE_COMPLIANCE_GATE == "compliance_gate"

    def test_persona_constant(self):
        assert NODE_PERSONA == "persona_envelope"

    def test_generate_still_exists(self):
        """NODE_GENERATE is preserved for backward compat."""
        assert NODE_GENERATE == "generate"

    def test_known_nodes_has_11(self):
        """_KNOWN_NODES includes all 11 v2.1 nodes."""
        from src.agent.graph import _KNOWN_NODES

        assert len(_KNOWN_NODES) == 11
        assert NODE_COMPLIANCE_GATE in _KNOWN_NODES
        assert NODE_PERSONA in _KNOWN_NODES
        assert NODE_GENERATE in _KNOWN_NODES
        assert NODE_WHISPER in _KNOWN_NODES

    def test_non_stream_nodes_includes_new_nodes(self):
        """_NON_STREAM_NODES includes compliance_gate, persona_envelope, and whisper_planner."""
        from src.agent.graph import _NON_STREAM_NODES

        assert NODE_COMPLIANCE_GATE in _NON_STREAM_NODES
        assert NODE_PERSONA in _NON_STREAM_NODES
        assert NODE_WHISPER in _NON_STREAM_NODES
        # generate should NOT be in non-stream (it streams tokens)
        assert NODE_GENERATE not in _NON_STREAM_NODES
