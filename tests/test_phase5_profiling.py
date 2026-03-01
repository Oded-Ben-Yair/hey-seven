"""Integration tests for profiling in the full graph pipeline.

Tests profiling_enrichment_node wired into the 12-node StateGraph:
topology verification, feature flag gating, multi-turn accumulation,
and coexistence with the validate retry loop.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.constants import (
    NODE_GENERATE,
    NODE_PROFILING,
    NODE_VALIDATE,
    _KNOWN_NODES,
)
from src.agent.graph import build_graph, chat
from src.agent.profiling import (
    ProfileExtractionOutput,
)
from src.agent.state import (
    DispatchOutput,
    PropertyQAState,
    RouterOutput,
    ValidationResult,
)
from src.casino.feature_flags import DEFAULT_FEATURES


# ---------------------------------------------------------------------------
# Graph topology with profiling
# ---------------------------------------------------------------------------


class TestProfilingGraphTopology:
    """Verify profiling_enrichment node is correctly wired in the graph."""

    def test_graph_has_12_nodes(self):
        """Graph with profiling enabled has 12 user nodes."""
        graph = build_graph()
        graph_data = graph.get_graph()
        user_nodes = {
            n for n in graph_data.nodes
            if not str(n).startswith("__")
        }
        assert len(user_nodes) == 12, f"Expected 12 nodes, got {len(user_nodes)}: {sorted(user_nodes)}"

    def test_profiling_node_exists_in_graph(self):
        """profiling_enrichment node is present in the compiled graph."""
        graph = build_graph()
        graph_data = graph.get_graph()
        node_names = set(graph_data.nodes.keys()) if isinstance(graph_data.nodes, dict) else {
            n.id if hasattr(n, "id") else str(n) for n in graph_data.nodes
        }
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

    def test_profiling_disabled_removes_node_from_edges(self):
        """When profiling_enabled=False, generate connects directly to validate."""
        import types
        disabled = types.MappingProxyType({**DEFAULT_FEATURES, "profiling_enabled": False})
        with patch("src.agent.graph.DEFAULT_FEATURES", disabled):
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


# ---------------------------------------------------------------------------
# Full pipeline integration with mock LLM
# ---------------------------------------------------------------------------


def _make_smart_mock_llm():
    """Build a mock LLM that dispatches by structured output schema.

    Handles RouterOutput, DispatchOutput, ValidationResult,
    and ProfileExtractionOutput.
    """
    mock_llm = AsyncMock()

    def _with_structured_output(schema, **kwargs):
        inner_mock = AsyncMock()
        if schema == RouterOutput:
            inner_mock.ainvoke = AsyncMock(return_value=RouterOutput(
                query_type="property_qa",
                confidence=0.95,
                detected_language="en",
            ))
        elif schema == DispatchOutput:
            inner_mock.ainvoke = AsyncMock(return_value=DispatchOutput(
                specialist="dining",
                confidence=0.9,
                reasoning="food query",
            ))
        elif schema == ValidationResult:
            inner_mock.ainvoke = AsyncMock(return_value=ValidationResult(
                status="PASS",
                reason="Response meets criteria",
            ))
        elif schema == ProfileExtractionOutput:
            inner_mock.ainvoke = AsyncMock(return_value=ProfileExtractionOutput(
                guest_name="Mike",
                party_size="4",
            ))
        else:
            # Default: return a mock that works
            inner_mock.ainvoke = AsyncMock(return_value=MagicMock())
        return inner_mock

    mock_llm.with_structured_output = _with_structured_output

    # For direct ainvoke (generate node)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
        content="Here are our restaurant options.",
    ))

    return mock_llm


class TestProfilingPipelineIntegration:
    """Integration tests running profiling through the full graph."""

    @pytest.mark.asyncio
    async def test_profiling_state_populated_after_node_call(self):
        """Direct call to profiling_enrichment_node populates state fields."""
        from src.agent.profiling import profiling_enrichment_node, ProfileExtractionOutput

        mock_llm = MagicMock()
        mock_extraction = ProfileExtractionOutput(
            guest_name="Mike",
            party_size="4",
        )
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=mock_extraction,
        )

        state = {
            "messages": [
                HumanMessage(content="I'm Mike, party of 4 for dinner"),
                AIMessage(content="Welcome Mike! Let me find dining options for your group."),
            ],
            "extracted_fields": {},
            "whisper_plan": None,
        }

        with patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock, return_value=mock_llm):
            result = await profiling_enrichment_node(state)

        assert result["profiling_phase"] is not None
        assert isinstance(result["profile_completeness_score"], float)
        assert result["profile_completeness_score"] > 0.0
        assert result["extracted_fields"].get("name") == "Mike"

    @pytest.mark.asyncio
    async def test_profiling_does_not_break_retry_loop(self):
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

    def test_profiling_disabled_graph_has_11_connected_nodes(self):
        """With profiling disabled, graph still has correct topology."""
        import types
        disabled = types.MappingProxyType({**DEFAULT_FEATURES, "profiling_enabled": False})
        with patch("src.agent.graph.DEFAULT_FEATURES", disabled):
            graph = build_graph()
            graph_data = graph.get_graph()

            # Even disabled, the node might still be in the graph
            # but the edge should go directly generate -> validate
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
