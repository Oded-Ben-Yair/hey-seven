"""Property-based tests for LangGraph state reducers and graph invariants.

Uses Hypothesis to verify algebraic properties of state reducers and
ensure _initial_state() always produces valid state.
"""

import pytest
from hypothesis import given, strategies as st, settings as hyp_settings

from src.agent.state import (
    _merge_dicts,
    _keep_max,
    _keep_truthy,
    UNSET_SENTINEL,
)


# ---------------------------------------------------------------------------
# _initial_state always produces all required keys
# ---------------------------------------------------------------------------

class TestInitialStateProperties:
    """Verify _initial_state() invariants."""

    @given(message=st.text(min_size=0, max_size=500))
    @hyp_settings(max_examples=50)
    def test_initial_state_has_all_required_keys(self, message):
        """_initial_state() must produce all PropertyQAState keys."""
        from src.agent.graph import _initial_state
        from src.agent.state import PropertyQAState

        state = _initial_state(message)
        expected_keys = set(PropertyQAState.__annotations__.keys())
        actual_keys = set(state.keys())
        assert expected_keys == actual_keys, f"Missing: {expected_keys - actual_keys}, Extra: {actual_keys - expected_keys}"

    @given(message=st.text(min_size=1, max_size=200))
    @hyp_settings(max_examples=20)
    def test_initial_state_messages_contain_input(self, message):
        """The input message must appear in _initial_state().messages."""
        from src.agent.graph import _initial_state

        state = _initial_state(message)
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == message


# ---------------------------------------------------------------------------
# _merge_dicts reducer properties
# ---------------------------------------------------------------------------

class TestMergeDictsProperties:
    """Algebraic properties of _merge_dicts reducer."""

    @given(d=st.dictionaries(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=50)))
    @hyp_settings(max_examples=50)
    def test_merge_dicts_identity(self, d):
        """Merging with empty dict returns original (identity element)."""
        result = _merge_dicts(d, {})
        assert result == d

    @given(d=st.dictionaries(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=50)))
    @hyp_settings(max_examples=50)
    def test_merge_dicts_from_empty(self, d):
        """Merging from empty dict returns new dict."""
        result = _merge_dicts({}, d)
        assert result == d

    @given(
        k=st.text(min_size=1, max_size=10),
        v1=st.text(min_size=1, max_size=50),
        v2=st.text(min_size=1, max_size=50),
    )
    @hyp_settings(max_examples=30)
    def test_merge_dicts_latest_wins(self, k, v1, v2):
        """Later value overwrites earlier for same key."""
        result = _merge_dicts({k: v1}, {k: v2})
        assert result[k] == v2

    def test_merge_dicts_unset_sentinel_deletes(self):
        """UNSET_SENTINEL removes the key from the merged result."""
        result = _merge_dicts({"name": "Alice", "age": "30"}, {"name": UNSET_SENTINEL})
        assert "name" not in result
        assert result["age"] == "30"

    @given(d=st.dictionaries(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=50)))
    @hyp_settings(max_examples=30)
    def test_merge_dicts_none_preserves_existing(self, d):
        """None values in b should NOT overwrite existing values in a."""
        none_dict = {k: None for k in d}
        result = _merge_dicts(d, none_dict)
        assert result == d

    def test_merge_dicts_none_inputs(self):
        """Both None inputs return empty dict."""
        assert _merge_dicts(None, None) == {}
        assert _merge_dicts(None, {}) == {}
        assert _merge_dicts({}, None) == {}


# ---------------------------------------------------------------------------
# _keep_max reducer properties
# ---------------------------------------------------------------------------

class TestKeepMaxProperties:
    """Algebraic properties of _keep_max reducer."""

    @given(a=st.integers(min_value=0, max_value=1000), b=st.integers(min_value=0, max_value=1000))
    @hyp_settings(max_examples=50)
    def test_keep_max_commutativity(self, a, b):
        """max(a, b) == max(b, a)."""
        assert _keep_max(a, b) == _keep_max(b, a)

    @given(a=st.integers(min_value=0, max_value=1000))
    @hyp_settings(max_examples=30)
    def test_keep_max_idempotent(self, a):
        """max(a, a) == a."""
        assert _keep_max(a, a) == a

    @given(a=st.integers(min_value=0, max_value=1000), b=st.integers(min_value=0, max_value=1000))
    @hyp_settings(max_examples=50)
    def test_keep_max_correctness(self, a, b):
        """_keep_max returns the larger value."""
        assert _keep_max(a, b) == max(a, b)

    def test_keep_max_none_inputs(self):
        """None is treated as 0."""
        assert _keep_max(None, None) == 0
        assert _keep_max(None, 5) == 5
        assert _keep_max(5, None) == 5


# ---------------------------------------------------------------------------
# _keep_truthy reducer properties
# ---------------------------------------------------------------------------

class TestKeepTruthyProperties:
    """Algebraic properties of _keep_truthy reducer."""

    @given(a=st.booleans(), b=st.booleans())
    @hyp_settings(max_examples=10)
    def test_keep_truthy_commutativity(self, a, b):
        """_keep_truthy(a, b) == _keep_truthy(b, a) (logical OR is commutative)."""
        assert _keep_truthy(a, b) == _keep_truthy(b, a)

    def test_keep_truthy_once_true_stays_true(self):
        """Once True, always True (sticky flag)."""
        assert _keep_truthy(True, False) is True
        assert _keep_truthy(False, True) is True
        assert _keep_truthy(True, True) is True

    def test_keep_truthy_false_identity(self):
        """False is identity element for OR."""
        assert _keep_truthy(False, False) is False

    @given(a=st.booleans())
    @hyp_settings(max_examples=10)
    def test_keep_truthy_idempotent(self, a):
        """_keep_truthy(a, a) == a."""
        assert _keep_truthy(a, a) == a

    def test_keep_truthy_returns_bool(self):
        """Return type is always bool, not truthy value."""
        result = _keep_truthy(None, None)
        assert result is False
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# State reducer edge cases
# ---------------------------------------------------------------------------

class TestReducerEdgeCases:
    """Edge cases for all reducers."""

    def test_merge_dicts_empty_string_ignored(self):
        """Empty string values should not overwrite existing."""
        result = _merge_dicts({"name": "Alice"}, {"name": ""})
        assert result["name"] == "Alice"

    @given(
        a=st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=20), max_size=5),
        b=st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=20), max_size=5),
        c=st.dictionaries(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=20), max_size=5),
    )
    @hyp_settings(max_examples=20)
    def test_merge_dicts_associativity(self, a, b, c):
        """_merge_dicts should be associative: merge(merge(a,b), c) == merge(a, merge(b,c))."""
        left = _merge_dicts(_merge_dicts(a, b), c)
        right = _merge_dicts(a, _merge_dicts(b, c))
        assert left == right
