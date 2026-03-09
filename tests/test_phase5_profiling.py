"""Integration tests for profiling in the full graph pipeline.

Tests profiling_enrichment_node wired into the 13-node StateGraph:
topology verification, feature flag gating, and state schema parity.

Mock-based pipeline integration tests removed per NO-MOCK ground rule.
Validate profiling via live eval (tests/evaluation/).
"""

import types

import pytest

from src.agent.constants import (
    NODE_GENERATE,
    NODE_PROFILING,
    NODE_VALIDATE,
    _KNOWN_NODES,
)
from src.agent.graph import build_graph
from src.agent.state import PropertyQAState
from src.casino.feature_flags import DEFAULT_FEATURES


# ---------------------------------------------------------------------------
# Graph topology with profiling
# ---------------------------------------------------------------------------


class TestProfilingGraphTopology:
    """Verify profiling_enrichment node is correctly wired in the graph."""

    def test_graph_has_13_nodes(self):
        """Graph with profiling enabled has 13 user nodes."""
        graph = build_graph()
        graph_data = graph.get_graph()
        user_nodes = {n for n in graph_data.nodes if not str(n).startswith("__")}
        assert len(user_nodes) == 13, (
            f"Expected 13 nodes, got {len(user_nodes)}: {sorted(user_nodes)}"
        )

    def test_profiling_node_exists_in_graph(self):
        """profiling_enrichment node is present in the compiled graph."""
        graph = build_graph()
        graph_data = graph.get_graph()
        node_names = (
            set(graph_data.nodes.keys())
            if isinstance(graph_data.nodes, dict)
            else {n.id if hasattr(n, "id") else str(n) for n in graph_data.nodes}
        )
        assert NODE_PROFILING in node_names

    def test_profiling_between_generate_and_validate(self):
        """profiling_enrichment sits between generate and validate in the edge map."""
        graph = build_graph()
        graph_data = graph.get_graph()
        adj: dict[str, set[str]] = {}
        for edge in graph_data.edges:
            adj.setdefault(edge.source, set()).add(edge.target)

        # generate -> profiling_enrichment -> validate
        assert NODE_PROFILING in adj.get(NODE_GENERATE, set()), (
            f"generate should connect to profiling. Targets: {adj.get(NODE_GENERATE)}"
        )
        assert NODE_VALIDATE in adj.get(NODE_PROFILING, set()), (
            f"profiling should connect to validate. Targets: {adj.get(NODE_PROFILING)}"
        )

    def test_profiling_disabled_removes_node_from_edges(self, monkeypatch):
        """When profiling_enabled=False, generate connects directly to validate."""
        import src.agent.graph as graph_mod

        disabled = types.MappingProxyType(
            {**DEFAULT_FEATURES, "profiling_enabled": False}
        )
        monkeypatch.setattr(graph_mod, "DEFAULT_FEATURES", disabled)
        graph = build_graph()
        graph_data = graph.get_graph()
        adj: dict[str, set[str]] = {}
        for edge in graph_data.edges:
            adj.setdefault(edge.source, set()).add(edge.target)

        # generate -> validate directly (no profiling)
        assert NODE_VALIDATE in adj.get(NODE_GENERATE, set()), (
            "When profiling disabled, generate should connect directly to validate"
        )

    def test_profiling_node_in_known_nodes(self):
        """NODE_PROFILING is registered in _KNOWN_NODES constant."""
        assert NODE_PROFILING in _KNOWN_NODES

    def test_profiling_does_not_break_retry_loop(self):
        """Validation RETRY still routes back to generate even with profiling in between."""
        graph = build_graph()
        graph_data = graph.get_graph()
        adj: dict[str, set[str]] = {}
        for edge in graph_data.edges:
            adj.setdefault(edge.source, set()).add(edge.target)

        # validate can still route back to generate (retry)
        assert NODE_GENERATE in adj.get(NODE_VALIDATE, set()), (
            "validate should still have retry edge to generate with profiling present"
        )

    def test_profiling_disabled_graph_has_correct_topology(self, monkeypatch):
        """With profiling disabled, graph still has correct topology."""
        import src.agent.graph as graph_mod

        disabled = types.MappingProxyType(
            {**DEFAULT_FEATURES, "profiling_enabled": False}
        )
        monkeypatch.setattr(graph_mod, "DEFAULT_FEATURES", disabled)
        graph = build_graph()
        graph_data = graph.get_graph()

        adj: dict[str, set[str]] = {}
        for edge in graph_data.edges:
            adj.setdefault(edge.source, set()).add(edge.target)

        assert NODE_VALIDATE in adj.get(NODE_GENERATE, set())


# ---------------------------------------------------------------------------
# State schema parity
# ---------------------------------------------------------------------------


class TestProfilingStateParity:
    """Verify profiling state fields are wired correctly."""

    def test_profiling_phase_in_state_schema(self):
        assert "profiling_phase" in PropertyQAState.__annotations__

    def test_profile_completeness_in_state_schema(self):
        assert "profile_completeness_score" in PropertyQAState.__annotations__

    def test_profiling_question_injected_in_state_schema(self):
        assert "profiling_question_injected" in PropertyQAState.__annotations__

    def test_initial_state_has_profiling_fields(self):
        """_initial_state() includes all profiling fields for parity check."""
        from src.agent.graph import _initial_state

        state = _initial_state("test")
        assert "profiling_phase" in state
        assert "profile_completeness_score" in state
        assert "profiling_question_injected" in state
        assert state["profiling_phase"] is None
        assert state["profile_completeness_score"] == 0.0
        assert state["profiling_question_injected"] is False
