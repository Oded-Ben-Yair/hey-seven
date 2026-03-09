"""Tests for the LangGraph agent (graph compilation, node constants, metadata).

Mock-based tests (TestChatResponseExtraction, TestChatSourceDedup, TestChatStream)
removed per NO-MOCK ground rule. Validate chat() and chat_stream() via live eval.
"""

import os

import pytest


class TestBuildGraph:
    def test_graph_compiles(self):
        """build_graph() returns a compiled graph without errors."""
        from src.agent.graph import build_graph

        graph = build_graph()
        assert graph is not None
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    def test_graph_has_13_nodes(self):
        """The compiled graph contains exactly 13 user-defined nodes (v2.4)."""
        from src.agent.graph import build_graph

        graph = build_graph()
        # get_graph() returns a DrawableGraph; its nodes include __start__ and __end__
        all_nodes = set(graph.get_graph().nodes)
        user_nodes = all_nodes - {"__start__", "__end__"}
        expected = {
            "compliance_gate",
            "router",
            "retrieve",
            "whisper_planner",
            "pre_extract",
            "generate",
            "validate",
            "persona_envelope",
            "respond",
            "fallback",
            "greeting",
            "off_topic",
            "profiling_enrichment",
        }
        assert user_nodes == expected

    def test_graph_accepts_custom_checkpointer(self):
        """build_graph() accepts an explicit checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver
        from src.agent.graph import build_graph

        cp = MemorySaver()
        graph = build_graph(checkpointer=cp)
        assert graph is not None


class TestNodeConstants:
    """Node name constants prevent silent breakage from string typos."""

    def test_constants_exported(self):
        """All 13 node constants are importable from graph module."""
        from src.agent.graph import (
            NODE_COMPLIANCE_GATE,
            NODE_FALLBACK,
            NODE_GENERATE,
            NODE_GREETING,
            NODE_OFF_TOPIC,
            NODE_PERSONA,
            NODE_PRE_EXTRACT,
            NODE_PROFILING,
            NODE_RESPOND,
            NODE_RETRIEVE,
            NODE_ROUTER,
            NODE_VALIDATE,
            NODE_WHISPER,
        )

        assert NODE_COMPLIANCE_GATE == "compliance_gate"
        assert NODE_ROUTER == "router"
        assert NODE_RETRIEVE == "retrieve"
        assert NODE_PRE_EXTRACT == "pre_extract"
        assert NODE_GENERATE == "generate"
        assert NODE_VALIDATE == "validate"
        assert NODE_PERSONA == "persona_envelope"
        assert NODE_RESPOND == "respond"
        assert NODE_FALLBACK == "fallback"
        assert NODE_GREETING == "greeting"
        assert NODE_OFF_TOPIC == "off_topic"
        assert NODE_WHISPER == "whisper_planner"
        assert NODE_PROFILING == "profiling_enrichment"

    def test_graph_nodes_match_constants(self):
        """Graph node names match the defined constants."""
        from src.agent.graph import (
            NODE_COMPLIANCE_GATE,
            NODE_FALLBACK,
            NODE_GENERATE,
            NODE_GREETING,
            NODE_OFF_TOPIC,
            NODE_PERSONA,
            NODE_PRE_EXTRACT,
            NODE_PROFILING,
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
            NODE_COMPLIANCE_GATE,
            NODE_ROUTER,
            NODE_RETRIEVE,
            NODE_WHISPER,
            NODE_PRE_EXTRACT,
            NODE_GENERATE,
            NODE_VALIDATE,
            NODE_PERSONA,
            NODE_RESPOND,
            NODE_FALLBACK,
            NODE_GREETING,
            NODE_OFF_TOPIC,
            NODE_PROFILING,
        }
        assert all_nodes == expected


class TestHitlInterrupt:
    """HITL interrupt support via ENABLE_HITL_INTERRUPT setting."""

    def test_hitl_disabled_by_default(self):
        """HITL interrupt is disabled by default."""
        from src.config import Settings

        s = Settings()
        assert s.ENABLE_HITL_INTERRUPT is False

    def test_hitl_graph_compiles_with_interrupt(self, monkeypatch):
        """Graph compiles with ENABLE_HITL_INTERRUPT=True."""
        from src.agent.graph import build_graph

        monkeypatch.setenv("ENABLE_HITL_INTERRUPT", "true")
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

    def test_generate_node_returns_specialist(self):
        """Generate node returns specialist name metadata (R41 fix D1-M002)."""
        from src.agent.graph import _extract_node_metadata

        assert _extract_node_metadata("generate", {"specialist_name": "dining"}) == {
            "specialist": "dining"
        }
        assert _extract_node_metadata("generate", {"some": "data"}) == {
            "specialist": None
        }

    def test_unknown_node_returns_empty(self):
        """Unknown or unhandled nodes return empty dict."""
        from src.agent.graph import _extract_node_metadata

        assert _extract_node_metadata("fallback", {}) == {}

    def test_non_dict_output_returns_empty(self):
        """Non-dict output returns empty dict."""
        from src.agent.graph import _extract_node_metadata

        assert _extract_node_metadata("router", "not a dict") == {}
        assert _extract_node_metadata("router", None) == {}
