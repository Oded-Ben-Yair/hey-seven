"""Tests for state schema parity, serialization safety, and reducer properties."""

import json

import pytest
from hypothesis import given, settings as h_settings, strategies as st


class TestInitialStateParity:
    """Verify _initial_state() keys match PropertyQAState annotations."""

    def test_initial_state_parity_with_state_schema(self):
        """_initial_state() keys must match PropertyQAState annotations (minus messages)."""
        from src.agent.graph import _initial_state
        from src.agent.state import PropertyQAState

        initial = _initial_state("test")
        state_fields = set(PropertyQAState.__annotations__) - {"messages"}
        initial_fields = set(initial.keys()) - {"messages"}
        assert state_fields == initial_fields, (
            f"Parity mismatch: missing={state_fields - initial_fields}, "
            f"extra={initial_fields - state_fields}"
        )


class TestInitialStateSerializable:
    """Verify _initial_state() non-message fields survive JSON roundtrip."""

    def test_initial_state_json_serializable(self):
        """_initial_state() output must survive JSON roundtrip."""
        from src.agent.graph import _initial_state

        state = _initial_state("test message")
        # Messages contain LangChain objects -- serialize the non-message fields
        non_msg = {k: v for k, v in state.items() if k != "messages"}
        roundtripped = json.loads(json.dumps(non_msg))
        assert roundtripped == non_msg


class TestRouterOutputLiterals:
    """RouterOutput query_type must accept all valid literals and reject invalid ones."""

    def test_valid_query_types_accepted(self):
        """RouterOutput accepts all defined literal query types."""
        from src.agent.state import RouterOutput

        valid_types = [
            "property_qa", "hours_schedule", "greeting", "off_topic",
            "gambling_advice", "action_request", "ambiguous",
        ]
        for qt in valid_types:
            result = RouterOutput(query_type=qt, confidence=0.9)
            assert result.query_type == qt

    def test_invalid_query_type_rejected(self):
        """RouterOutput rejects invalid query types."""
        from src.agent.state import RouterOutput

        with pytest.raises(Exception):
            RouterOutput(query_type="invalid_type", confidence=0.9)


class TestValidationResultLiterals:
    """ValidationResult status must be PASS, FAIL, or RETRY."""

    def test_valid_statuses_accepted(self):
        """ValidationResult accepts PASS, FAIL, and RETRY."""
        from src.agent.state import ValidationResult

        for status in ["PASS", "FAIL", "RETRY"]:
            result = ValidationResult(status=status, reason="test")
            assert result.status == status

    def test_invalid_status_rejected(self):
        """ValidationResult rejects invalid status values."""
        from src.agent.state import ValidationResult

        with pytest.raises(Exception):
            ValidationResult(status="INVALID", reason="test")


class TestMergeDictsProperties:
    """R38 fix C-005: Property-based tests for _merge_dicts reducer.

    Verifies algebraic invariants:
    - Identity element: merging with {} preserves existing state
    - None-filtering: None values in b do not overwrite a
    - Empty-string-filtering (R38 fix C-003): "" values in b do not overwrite a
    """

    @given(st.dictionaries(st.text(min_size=1, max_size=5), st.one_of(st.text(min_size=1), st.integers())))
    @h_settings(max_examples=50)
    def test_merge_dicts_identity(self, d):
        """Merging with empty dict preserves all keys and values."""
        from src.agent.state import _merge_dicts

        result = _merge_dicts(d, {})
        assert result == d

    @given(st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1)))
    @h_settings(max_examples=50)
    def test_merge_dicts_none_filtered(self, d):
        """None values in b do not overwrite existing keys in a."""
        from src.agent.state import _merge_dicts

        none_dict = {k: None for k in d}
        result = _merge_dicts(d, none_dict)
        assert result == d

    @given(st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1)))
    @h_settings(max_examples=50)
    def test_merge_dicts_empty_string_filtered(self, d):
        """Empty string values in b do not overwrite existing keys in a (R38 fix C-003)."""
        from src.agent.state import _merge_dicts

        empty_dict = {k: "" for k in d}
        result = _merge_dicts(d, empty_dict)
        assert result == d

    def test_merge_dicts_new_values_overwrite(self):
        """Non-None, non-empty values in b overwrite a."""
        from src.agent.state import _merge_dicts

        result = _merge_dicts({"name": "Sara"}, {"name": "Sarah"})
        assert result == {"name": "Sarah"}

    def test_merge_dicts_tombstone_deletes_key(self):
        """R47 fix C7: UNSET_SENTINEL in b removes the key from merged result."""
        from src.agent.state import UNSET_SENTINEL, _merge_dicts

        result = _merge_dicts({"name": "Sara", "dietary": "peanut allergy"}, {"dietary": UNSET_SENTINEL})
        assert result == {"name": "Sara"}

    def test_merge_dicts_tombstone_missing_key_noop(self):
        """R47 fix C7: UNSET_SENTINEL for non-existent key is a no-op."""
        from src.agent.state import UNSET_SENTINEL, _merge_dicts

        result = _merge_dicts({"name": "Sara"}, {"dietary": UNSET_SENTINEL})
        assert result == {"name": "Sara"}

    def test_merge_dicts_string_unset_not_treated_as_sentinel(self):
        """R48 fix: String '__UNSET__' in user input is NOT a tombstone.

        UNSET_SENTINEL is now object(), so string comparison fails.
        Prevents user input collision that could delete profile fields.
        """
        from src.agent.state import _merge_dicts

        result = _merge_dicts({"name": "Sara"}, {"dietary": "__UNSET__"})
        assert result == {"name": "Sara", "dietary": "__UNSET__"}

    @given(
        st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1)),
        st.lists(st.text(min_size=1, max_size=5), min_size=0, max_size=3),
    )
    @h_settings(max_examples=50)
    def test_merge_dicts_tombstone_removes_all_targeted_keys(self, d, keys_to_remove):
        """R47 fix C7: All tombstone keys are removed from result."""
        from src.agent.state import UNSET_SENTINEL, _merge_dicts

        tombstones = {k: UNSET_SENTINEL for k in keys_to_remove}
        result = _merge_dicts(d, tombstones)
        for k in keys_to_remove:
            assert k not in result

    @given(
        st.dictionaries(st.text(min_size=1, max_size=5), st.one_of(st.text(min_size=1), st.integers())),
        st.dictionaries(st.text(min_size=1, max_size=5), st.one_of(st.text(min_size=1), st.integers())),
        st.dictionaries(st.text(min_size=1, max_size=5), st.one_of(st.text(min_size=1), st.integers())),
    )
    @h_settings(max_examples=50)
    def test_merge_dicts_associativity(self, a, b, c):
        """R40 fix D5-M005: merge(merge(a, b), c) == merge(a, merge(b, c)).

        Associativity matters because extracted_fields accumulates across 3+
        turns. If merge is not associative, the order of node execution
        could produce different final states.
        """
        from src.agent.state import _merge_dicts

        left = _merge_dicts(_merge_dicts(a, b), c)
        right = _merge_dicts(a, _merge_dicts(b, c))
        assert left == right


class TestKeepMaxProperties:
    """R38 fix C-005: Property-based tests for _keep_max reducer."""

    @given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
    @h_settings(max_examples=50)
    def test_keep_max_commutative(self, a, b):
        """_keep_max(a, b) == _keep_max(b, a)."""
        from src.agent.state import _keep_max

        assert _keep_max(a, b) == _keep_max(b, a)

    @given(st.integers(min_value=0, max_value=1000))
    @h_settings(max_examples=50)
    def test_keep_max_identity(self, a):
        """_keep_max(a, 0) == a for non-negative a."""
        from src.agent.state import _keep_max

        assert _keep_max(a, 0) == a

    def test_keep_max_none_guard(self):
        """_keep_max handles None input without TypeError (R38 fix M-007)."""
        from src.agent.state import _keep_max

        assert _keep_max(5, None) == 5
        assert _keep_max(None, 3) == 3
        assert _keep_max(None, None) == 0


class TestKeepTruthyProperties:
    """R38 fix C-005: Property-based tests for _keep_truthy reducer."""

    @given(st.booleans(), st.booleans())
    @h_settings(max_examples=10)
    def test_keep_truthy_commutative(self, a, b):
        """_keep_truthy(a, b) == _keep_truthy(b, a)."""
        from src.agent.state import _keep_truthy

        assert _keep_truthy(a, b) == _keep_truthy(b, a)

    def test_keep_truthy_identity(self):
        """_keep_truthy(a, False) == a."""
        from src.agent.state import _keep_truthy

        assert _keep_truthy(True, False) is True
        assert _keep_truthy(False, False) is False

    def test_keep_truthy_sticky(self):
        """Once True, stays True."""
        from src.agent.state import _keep_truthy

        assert _keep_truthy(True, False) is True
        assert _keep_truthy(True, True) is True
