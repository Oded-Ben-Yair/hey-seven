"""Formal verification of graph topology -- no unreachable nodes, no stuck states.

BFS-based verification that the compiled StateGraph has:
1. No unreachable nodes from START
2. All nodes can reach END (no stuck states)
3. No self-loops
4. Only allowed loops exist (validate->generate retry)
5. Node count matches expected architecture (11 user nodes)
"""

import pytest

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
    _KNOWN_NODES,
    build_graph,
)


class TestGraphTopology:
    """Formal graph topology verification using BFS."""

    @pytest.fixture(autouse=True)
    def _build(self):
        """Build graph once for all topology tests."""
        self.compiled = build_graph()
        self.g = self.compiled.get_graph()

    def _get_adjacency(self):
        """Build forward adjacency dict from graph edges."""
        adj: dict[str, set[str]] = {}
        for edge in self.g.edges:
            adj.setdefault(edge.source, set()).add(edge.target)
        return adj

    def _get_reverse_adjacency(self):
        """Build reverse adjacency dict from graph edges."""
        rev: dict[str, set[str]] = {}
        for edge in self.g.edges:
            rev.setdefault(edge.target, set()).add(edge.source)
        return rev

    def _get_all_nodes(self):
        """Get all node IDs from the graph."""
        return set(self.g.nodes.keys())

    def _get_user_nodes(self):
        """Get all non-internal node IDs (excluding __start__, __end__)."""
        return self._get_all_nodes() - {"__start__", "__end__"}

    # ------------------------------------------------------------------
    # Reachability tests
    # ------------------------------------------------------------------

    def test_no_unreachable_nodes(self):
        """Every node is reachable from __start__ via BFS."""
        adj = self._get_adjacency()
        visited: set[str] = set()
        queue = ["__start__"]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for target in adj.get(node, set()):
                if target not in visited:
                    queue.append(target)

        all_nodes = self._get_all_nodes()
        unreachable = all_nodes - visited
        assert not unreachable, f"Unreachable nodes from __start__: {unreachable}"

    def test_all_nodes_can_reach_end(self):
        """Every node has a path to __end__ (no stuck states)."""
        rev = self._get_reverse_adjacency()
        # BFS backwards from __end__
        can_reach_end: set[str] = set()
        queue = ["__end__"]
        while queue:
            node = queue.pop(0)
            if node in can_reach_end:
                continue
            can_reach_end.add(node)
            for source in rev.get(node, set()):
                if source not in can_reach_end:
                    queue.append(source)

        all_nodes = self._get_all_nodes()
        # __end__ itself is trivially "can reach end"
        stuck = all_nodes - can_reach_end
        assert not stuck, f"Stuck nodes (no path to __end__): {stuck}"

    # ------------------------------------------------------------------
    # Structural invariants
    # ------------------------------------------------------------------

    def test_no_self_loops(self):
        """No node has an edge directly to itself."""
        for edge in self.g.edges:
            assert edge.source != edge.target, f"Self-loop detected on {edge.source}"

    def test_start_has_outgoing_edges(self):
        """__start__ has at least one outgoing edge."""
        adj = self._get_adjacency()
        assert "__start__" in adj, "__start__ has no outgoing edges"
        assert len(adj["__start__"]) >= 1

    def test_end_has_incoming_edges(self):
        """__end__ has at least one incoming edge."""
        rev = self._get_reverse_adjacency()
        assert "__end__" in rev, "__end__ has no incoming edges"
        assert len(rev["__end__"]) >= 1

    def test_start_has_no_incoming_edges(self):
        """__start__ should not be a target of any edge."""
        rev = self._get_reverse_adjacency()
        assert "__start__" not in rev, (
            f"__start__ has incoming edges from: {rev.get('__start__')}"
        )

    def test_end_has_no_outgoing_edges(self):
        """__end__ should not be a source of any edge."""
        adj = self._get_adjacency()
        assert "__end__" not in adj, (
            f"__end__ has outgoing edges to: {adj.get('__end__')}"
        )

    # ------------------------------------------------------------------
    # Node count and identity
    # ------------------------------------------------------------------

    def test_node_count_matches_expected(self):
        """Graph has exactly 11 user-defined nodes (plus __start__ and __end__)."""
        user_nodes = self._get_user_nodes()
        assert len(user_nodes) == 11, (
            f"Expected 11 user nodes, got {len(user_nodes)}: {sorted(user_nodes)}"
        )

    def test_all_known_nodes_present(self):
        """Every node name in _KNOWN_NODES exists in the compiled graph."""
        user_nodes = self._get_user_nodes()
        missing = _KNOWN_NODES - user_nodes
        assert not missing, f"Nodes in _KNOWN_NODES but missing from graph: {missing}"

    def test_no_unknown_nodes(self):
        """No unexpected nodes exist beyond _KNOWN_NODES and internals."""
        user_nodes = self._get_user_nodes()
        unknown = user_nodes - _KNOWN_NODES
        assert not unknown, f"Unexpected nodes not in _KNOWN_NODES: {unknown}"

    # ------------------------------------------------------------------
    # Allowed loops
    # ------------------------------------------------------------------

    def test_only_validate_generate_loop_exists(self):
        """The only cycle in the graph is validate -> generate (retry).

        Uses DFS-based back-edge detection: an edge (u, v) is a back-edge
        if v is an ancestor of u in the DFS tree (i.e., v is currently on
        the recursion stack). This correctly handles DAGs with convergent
        paths (e.g., both compliance_gate and router routing to greeting).
        """
        adj = self._get_adjacency()
        allowed_back_edges = {(NODE_VALIDATE, NODE_GENERATE)}

        back_edges: set[tuple[str, str]] = set()
        visited: set[str] = set()
        on_stack: set[str] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            on_stack.add(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in on_stack:
                    back_edges.add((node, neighbor))
            on_stack.discard(node)

        dfs("__start__")

        unexpected = back_edges - allowed_back_edges
        assert not unexpected, (
            f"Unexpected back-edges (cycles): {unexpected}. "
            f"Only {allowed_back_edges} is allowed."
        )

    # ------------------------------------------------------------------
    # Terminal node verification
    # ------------------------------------------------------------------

    def test_terminal_nodes_reach_end_directly(self):
        """Terminal nodes (greeting, off_topic, fallback) connect directly to __end__."""
        adj = self._get_adjacency()
        terminal_nodes = {NODE_GREETING, NODE_OFF_TOPIC, NODE_FALLBACK}
        for node in terminal_nodes:
            targets = adj.get(node, set())
            assert "__end__" in targets, (
                f"Terminal node {node} does not connect to __end__. "
                f"Targets: {targets}"
            )

    def test_respond_is_final_node_before_end(self):
        """respond connects directly to __end__ (last node in happy path)."""
        adj = self._get_adjacency()
        assert "__end__" in adj.get(NODE_RESPOND, set()), (
            "respond does not connect to __end__"
        )

    # ------------------------------------------------------------------
    # Entry point verification
    # ------------------------------------------------------------------

    def test_compliance_gate_is_first_node(self):
        """compliance_gate is the first node after __start__."""
        adj = self._get_adjacency()
        start_targets = adj.get("__start__", set())
        assert start_targets == {NODE_COMPLIANCE_GATE}, (
            f"Expected __start__ -> compliance_gate only, got: {start_targets}"
        )

    # ------------------------------------------------------------------
    # Happy-path chain verification
    # ------------------------------------------------------------------

    def test_happy_path_chain(self):
        """The full happy path is: compliance_gate -> router -> retrieve ->
        whisper_planner -> generate -> validate -> persona_envelope -> respond -> __end__."""
        adj = self._get_adjacency()

        happy_path = [
            (NODE_COMPLIANCE_GATE, NODE_ROUTER),
            (NODE_ROUTER, NODE_RETRIEVE),
            (NODE_RETRIEVE, NODE_WHISPER),
            (NODE_WHISPER, NODE_GENERATE),
            (NODE_GENERATE, NODE_VALIDATE),
            (NODE_VALIDATE, NODE_PERSONA),
            (NODE_PERSONA, NODE_RESPOND),
            (NODE_RESPOND, "__end__"),
        ]
        for source, target in happy_path:
            assert target in adj.get(source, set()), (
                f"Happy path broken: {source} -> {target} edge missing. "
                f"Actual targets of {source}: {adj.get(source, set())}"
            )

    # ------------------------------------------------------------------
    # Conditional edge verification
    # ------------------------------------------------------------------

    def test_conditional_edges_exist(self):
        """Conditional edges exist for compliance_gate, router, and validate."""
        conditional_sources = set()
        for edge in self.g.edges:
            if edge.conditional:
                conditional_sources.add(edge.source)

        expected_conditional = {NODE_COMPLIANCE_GATE, NODE_ROUTER, NODE_VALIDATE}
        assert expected_conditional <= conditional_sources, (
            f"Missing conditional edges. Expected at least {expected_conditional}, "
            f"got {conditional_sources}"
        )
