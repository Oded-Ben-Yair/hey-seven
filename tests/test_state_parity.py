"""Tests for state schema parity and serialization safety."""

import json

import pytest


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
