"""State serialization round-trip tests.

Verifies that graph state survives JSON serialization (required for
FirestoreSaver checkpointer which serializes state to Firestore).
"""

import json

import pytest


def test_initial_state_json_roundtrip():
    """_initial_state() output must survive JSON serialization."""
    from src.agent.graph import _initial_state

    state = _initial_state("test message")
    # Remove messages (HumanMessage objects need special serialization)
    state_without_messages = {k: v for k, v in state.items() if k != "messages"}
    roundtripped = json.loads(json.dumps(state_without_messages))
    assert roundtripped == state_without_messages


def test_unset_sentinel_json_roundtrip():
    """UNSET_SENTINEL must survive JSON serialization (string-based, not object())."""
    from src.agent.state import UNSET_SENTINEL

    assert json.loads(json.dumps(UNSET_SENTINEL)) == UNSET_SENTINEL
    assert isinstance(UNSET_SENTINEL, str)
    assert UNSET_SENTINEL.startswith("$$UNSET:")


def test_unset_sentinel_uniqueness():
    """UNSET_SENTINEL should not appear in natural language."""
    from src.agent.state import UNSET_SENTINEL

    natural_inputs = [
        "Please unset my dietary preference",
        "Remove the peanut allergy",
        "$$UNSET$$",
        "UNSET",
        "",
    ]
    for text in natural_inputs:
        assert text != UNSET_SENTINEL
