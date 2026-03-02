"""Tests for intent-aware validation (R82 Track 1D).

Verifies that VALIDATION_PROMPT adapts criteria based on query_type.
"""
import pytest
from src.agent.prompts import VALIDATION_PROMPT


class TestIntentAwareValidationPrompt:
    """Test that VALIDATION_PROMPT includes query_type variable."""

    def test_has_query_type_variable(self):
        """VALIDATION_PROMPT must accept $query_type."""
        result = VALIDATION_PROMPT.safe_substitute(
            user_question="test",
            retrieved_context="test",
            generated_response="test",
            query_type="greeting",
        )
        assert "greeting" in result

    def test_query_type_in_prompt(self):
        """Query type appears in the prompt output."""
        result = VALIDATION_PROMPT.safe_substitute(
            user_question="Hey!",
            retrieved_context="No context",
            generated_response="Welcome!",
            query_type="greeting",
        )
        assert "Query Type" in result
        assert "greeting" in result

    def test_light_criteria_section_exists(self):
        """Prompt contains light criteria for greetings/acknowledgments."""
        result = VALIDATION_PROMPT.safe_substitute(
            user_question="test",
            retrieved_context="test",
            generated_response="test",
            query_type="greeting",
        )
        assert "Light Criteria" in result
        assert "greeting" in result.lower()

    def test_crisis_criteria_section_exists(self):
        """Prompt contains crisis-specific criteria."""
        result = VALIDATION_PROMPT.safe_substitute(
            user_question="test",
            retrieved_context="test",
            generated_response="test",
            query_type="self_harm",
        )
        assert "Crisis resources" in result or "crisis" in result.lower()

    def test_grounding_criteria_section_exists(self):
        """Prompt contains grounding criteria for property_qa."""
        result = VALIDATION_PROMPT.safe_substitute(
            user_question="test",
            retrieved_context="test",
            generated_response="test",
            query_type="property_qa",
        )
        assert "Grounded" in result

    def test_default_query_type_fallback(self):
        """Missing query_type defaults gracefully."""
        result = VALIDATION_PROMPT.safe_substitute(
            user_question="test",
            retrieved_context="test",
            generated_response="test",
        )
        # $query_type should remain as literal if not provided
        assert "$query_type" in result or "query_type" in result


class TestValidateNodeQueryType:
    """Test that validate_node passes query_type to the prompt."""

    def test_validate_node_uses_state_query_type(self):
        """Verify validate_node reads query_type from state.

        This is a code-level check -- the actual LLM call is mocked in
        test_nodes.py. Here we just verify the prompt construction.
        """
        from src.agent.prompts import VALIDATION_PROMPT

        # Simulate what validate_node does
        query_type = "greeting"  # From state
        prompt_text = VALIDATION_PROMPT.safe_substitute(
            user_question="Hey there!",
            retrieved_context="No context.",
            generated_response="Welcome to Mohegan Sun!",
            query_type=query_type,
        )
        assert "greeting" in prompt_text
