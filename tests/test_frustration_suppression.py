"""Tests for frustration/crisis suppression and response length budgets.

Mock purge R111: Retained only deterministic tests that do not depend on
MagicMock/AsyncMock/@patch for LLM calls. Behavioral validation uses live eval.

R82 Track 1E: Frustration suppression of promotional content for comp agent.
R82 Track 1E: Crisis state suppression across ALL specialists.
R82 Track 1G (partial): Response length budgets per intent.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.agents._base import (
    _build_behavioral_prompt_sections,
    _should_inject_suggestion,
)


# ---------------------------------------------------------------------------
# Track 1E: Frustration Suppression — deterministic pure functions
# ---------------------------------------------------------------------------


class TestFrustrationSuppression:
    """Test that promotional content is suppressed for frustrated guests."""

    def test_behavioral_sections_detect_frustration(self):
        """_build_behavioral_prompt_sections returns frustrated as effective_sentiment."""
        state = {
            "messages": [HumanMessage(content="This is terrible service!")],
            "query_type": None,
            "router_confidence": 0.0,
            "retrieved_context": [],
            "validation_result": None,
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": None,
            "guest_sentiment": "frustrated",
            "guest_context": {},
            "guest_name": None,
            "domains_discussed": [],
            "crisis_active": False,
            "crisis_turn_count": 0,
            "detected_language": None,
            "handoff_request": None,
            "profiling_phase": None,
            "profile_completeness_score": 0.0,
            "profiling_question_injected": False,
            "suggestion_offered": False,
            "responsible_gaming_count": 0,
            "specialist_name": None,
            "dispatch_method": None,
        }
        sections, eff_sent, dynamics, frust_count = _build_behavioral_prompt_sections(
            state,
            "This is terrible service!",
            "this is terrible service!",
            {},
            "frustrated",
        )
        assert eff_sent == "frustrated"

    def test_suggestion_blocked_on_frustration(self):
        """Proactive suggestions are blocked when guest is frustrated."""
        state = {
            "messages": [HumanMessage(content="test")],
            "query_type": None,
            "router_confidence": 0.0,
            "retrieved_context": [{"content": "test", "metadata": {}, "score": 0.9}],
            "validation_result": None,
            "retry_count": 0,
            "skip_validation": False,
            "retry_feedback": None,
            "current_time": "Monday 3 PM",
            "sources_used": [],
            "extracted_fields": {},
            "whisper_plan": {
                "proactive_suggestion": "Try our spa!",
                "suggestion_confidence": "0.9",
            },
            "guest_sentiment": None,
            "guest_context": {},
            "guest_name": None,
            "domains_discussed": [],
            "crisis_active": False,
            "crisis_turn_count": 0,
            "detected_language": None,
            "handoff_request": None,
            "profiling_phase": None,
            "profile_completeness_score": 0.0,
            "profiling_question_injected": False,
            "suggestion_offered": False,
            "responsible_gaming_count": 0,
            "specialist_name": None,
            "dispatch_method": None,
        }
        section, offered = _should_inject_suggestion(
            state,
            "frustrated",
            {"turn_count": 5},
        )
        assert section == ""
        assert offered is False
