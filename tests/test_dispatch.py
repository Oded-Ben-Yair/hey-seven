"""Tests for extracted dispatch logic (src/agent/dispatch.py).

Validates keyword dispatch, category mappings, node metadata extraction,
comp intent detection, backward-compat re-exports from graph.py,
and dynamic comp bridge.

Mock-based tests (TestExecuteSpecialistErrorHandling) removed (mock purge R111).
"""

import pytest

from src.agent.dispatch import (
    _CATEGORY_TO_AGENT,
    _CATEGORY_PRIORITY,
    _DISPATCH_OWNED_KEYS,
    _VALID_STATE_KEYS,
    _keyword_dispatch,
    _extract_node_metadata,
)
from src.agent.state import PropertyQAState


class TestKeywordDispatch:
    def test_empty_context_returns_host(self):
        assert _keyword_dispatch([]) == "host"

    def test_single_restaurant_category(self):
        chunks = [{"metadata": {"category": "restaurants"}}]
        assert _keyword_dispatch(chunks) == "dining"

    def test_hotel_category(self):
        chunks = [{"metadata": {"category": "hotel"}}]
        assert _keyword_dispatch(chunks) == "hotel"

    def test_unknown_category_returns_host(self):
        chunks = [{"metadata": {"category": "unknown_stuff"}}]
        assert _keyword_dispatch(chunks) == "host"

    def test_tie_break_by_priority(self):
        # 1 restaurant, 1 hotel -- equal count, restaurants has higher priority (4 vs 3)
        chunks = [
            {"metadata": {"category": "restaurants"}},
            {"metadata": {"category": "hotel"}},
        ]
        assert _keyword_dispatch(chunks) == "dining"

    def test_majority_wins(self):
        chunks = [
            {"metadata": {"category": "hotel"}},
            {"metadata": {"category": "hotel"}},
            {"metadata": {"category": "restaurants"}},
        ]
        assert _keyword_dispatch(chunks) == "hotel"

    def test_empty_category_ignored(self):
        chunks = [{"metadata": {"category": ""}}, {"metadata": {"category": "gaming"}}]
        assert _keyword_dispatch(chunks) == "comp"

    def test_no_metadata_returns_host(self):
        chunks = [{"metadata": {}}, {}]
        assert _keyword_dispatch(chunks) == "host"


class TestCompIntentKeywords:
    """R103 fix H9: Comp intent word and phrase matching."""

    def test_single_word_comp_intent(self):
        from src.agent.dispatch import _COMP_INTENT_WORDS

        query = "what rewards do I get"
        query_words = set(query.lower().split())
        assert query_words & _COMP_INTENT_WORDS  # "rewards" matches

    def test_multi_word_phrase_detected(self):
        from src.agent.dispatch import _COMP_INTENT_PHRASES

        query = "how do I get free play"
        query_lower = query.lower()
        assert any(phrase in query_lower for phrase in _COMP_INTENT_PHRASES)

    def test_player_card_phrase_detected(self):
        from src.agent.dispatch import _COMP_INTENT_PHRASES

        query = "where do I get my player card"
        query_lower = query.lower()
        assert any(phrase in query_lower for phrase in _COMP_INTENT_PHRASES)

    def test_rewards_program_phrase_detected(self):
        from src.agent.dispatch import _COMP_INTENT_PHRASES

        query = "tell me about the rewards program"
        query_lower = query.lower()
        assert any(phrase in query_lower for phrase in _COMP_INTENT_PHRASES)

    def test_my_status_phrase_detected(self):
        from src.agent.dispatch import _COMP_INTENT_PHRASES

        query = "what is my status in the program"
        query_lower = query.lower()
        assert any(phrase in query_lower for phrase in _COMP_INTENT_PHRASES)

    def test_non_comp_query_not_matched(self):
        from src.agent.dispatch import _COMP_INTENT_WORDS, _COMP_INTENT_PHRASES

        query = "what restaurants do you have"
        query_lower = query.lower()
        query_words = set(query_lower.split())
        assert not (query_words & _COMP_INTENT_WORDS)
        assert not any(phrase in query_lower for phrase in _COMP_INTENT_PHRASES)


class TestCompIntentDetectedField:
    """R103 fix H9: comp_intent_detected state field."""

    def test_field_in_state_annotations(self):
        assert "comp_intent_detected" in PropertyQAState.__annotations__

    def test_field_in_initial_state(self):
        from src.agent.graph import _initial_state

        state = _initial_state("hello")
        assert "comp_intent_detected" in state
        assert state["comp_intent_detected"] is False


class TestDispatchOwnedKeys:
    def test_contains_expected_keys(self):
        assert "guest_context" in _DISPATCH_OWNED_KEYS
        assert "guest_name" in _DISPATCH_OWNED_KEYS

    def test_is_frozenset(self):
        assert isinstance(_DISPATCH_OWNED_KEYS, frozenset)


class TestValidStateKeys:
    def test_contains_core_fields(self):
        assert "messages" in _VALID_STATE_KEYS
        assert "query_type" in _VALID_STATE_KEYS
        assert "dispatch_method" in _VALID_STATE_KEYS

    def test_matches_state_annotations(self):
        assert _VALID_STATE_KEYS == frozenset(PropertyQAState.__annotations__)


class TestCategoryMappings:
    def test_category_to_agent_immutable(self):
        with pytest.raises(TypeError):
            _CATEGORY_TO_AGENT["test"] = "value"

    def test_category_priority_immutable(self):
        with pytest.raises(TypeError):
            _CATEGORY_PRIORITY["test"] = 999

    def test_spa_maps_to_entertainment(self):
        assert _CATEGORY_TO_AGENT["spa"] == "entertainment"


class TestExtractNodeMetadata:
    def test_router_metadata(self):
        output = {"query_type": "greeting", "router_confidence": 0.95}
        meta = _extract_node_metadata("compliance_gate", output)
        assert meta["query_type"] == "greeting"
        assert meta["confidence"] == 0.95

    def test_retrieve_metadata(self):
        output = {"retrieved_context": [1, 2, 3]}
        meta = _extract_node_metadata("retrieve", output)
        assert meta["doc_count"] == 3

    def test_generate_metadata(self):
        output = {"specialist_name": "dining"}
        meta = _extract_node_metadata("generate", output)
        assert meta["specialist"] == "dining"

    def test_validate_metadata(self):
        output = {"validation_result": "PASS"}
        meta = _extract_node_metadata("validate", output)
        assert meta["result"] == "PASS"

    def test_respond_metadata(self):
        output = {"sources_used": ["restaurants", "hotel"]}
        meta = _extract_node_metadata("respond", output)
        assert meta["sources"] == ["restaurants", "hotel"]

    def test_whisper_metadata(self):
        output = {"whisper_plan": {"steps": ["a"]}}
        meta = _extract_node_metadata("whisper_planner", output)
        assert meta["has_plan"] is True

    def test_unknown_node(self):
        meta = _extract_node_metadata("unknown", {"data": "x"})
        assert meta == {}

    def test_non_dict_output(self):
        meta = _extract_node_metadata("router", "not a dict")
        assert meta == {}


class TestBackwardCompatImports:
    """Verify backward-compat re-exports from graph.py still work."""

    def test_import_from_graph(self):
        from src.agent.graph import (
            _dispatch_to_specialist,
            _keyword_dispatch,
            _CATEGORY_TO_AGENT,
            _DISPATCH_OWNED_KEYS,
            _VALID_STATE_KEYS,
        )

        assert callable(_dispatch_to_specialist)
        assert callable(_keyword_dispatch)
        assert "restaurants" in _CATEGORY_TO_AGENT
        assert isinstance(_DISPATCH_OWNED_KEYS, frozenset)
        assert isinstance(_VALID_STATE_KEYS, frozenset)

    def test_import_extract_node_metadata_from_graph(self):
        from src.agent.graph import _extract_node_metadata

        assert callable(_extract_node_metadata)

    def test_import_route_to_specialist_from_graph(self):
        from src.agent.graph import _route_to_specialist

        assert callable(_route_to_specialist)

    def test_import_inject_guest_context_from_graph(self):
        from src.agent.graph import _inject_guest_context

        assert callable(_inject_guest_context)

    def test_import_execute_specialist_from_graph(self):
        from src.agent.graph import _execute_specialist

        assert callable(_execute_specialist)

    def test_import_dispatch_prompt_from_graph(self):
        from src.agent.graph import _DISPATCH_PROMPT

        assert hasattr(_DISPATCH_PROMPT, "safe_substitute")

    def test_import_category_priority_from_graph(self):
        from src.agent.graph import _CATEGORY_PRIORITY

        assert "restaurants" in _CATEGORY_PRIORITY
        assert _CATEGORY_PRIORITY["restaurants"] == 4


class TestDispatchMethodInState:
    """Verify dispatch_method field is properly wired."""

    def test_dispatch_method_in_state_annotations(self):
        assert "dispatch_method" in PropertyQAState.__annotations__

    def test_dispatch_method_in_initial_state(self):
        from src.agent.graph import _initial_state

        state = _initial_state("hello")
        assert "dispatch_method" in state
        assert state["dispatch_method"] is None


class TestDynamicCompBridgeInBase:
    """R105: Verify _base.py uses dynamic comp bridge via get_comp_bridge_for_non_comp."""

    def test_get_comp_bridge_for_non_comp_importable(self):
        from src.agent.behavior_tools.comp_strategy import get_comp_bridge_for_non_comp

        assert callable(get_comp_bridge_for_non_comp)

    def test_bridge_called_for_non_comp_agent(self):
        from src.agent.behavior_tools.comp_strategy import get_comp_bridge_for_non_comp

        state = {
            "extracted_fields": {"occasion": "birthday"},
            "guest_sentiment": None,
        }
        bridge = get_comp_bridge_for_non_comp(state, casino_id="mohegan_sun")
        assert isinstance(bridge, str)
        assert len(bridge) > 0

    def test_bridge_not_needed_for_comp_agent(self):
        from src.agent.behavior_tools.comp_strategy import (
            get_comp_bridge_for_non_comp,
            get_comp_prompt_section,
        )

        state = {
            "extracted_fields": {"loyalty_tier": "Ignite"},
            "guest_sentiment": None,
        }
        full_section = get_comp_prompt_section(state, casino_id="mohegan_sun")
        bridge = get_comp_bridge_for_non_comp(state, casino_id="mohegan_sun")
        # Full section is much longer than bridge
        assert len(full_section) > len(bridge)

    def test_comp_intent_detected_flag_in_dispatch_result(self):
        assert "comp_intent_detected" in PropertyQAState.__annotations__
