"""Tests for HandoffOrchestrator tool (P9: Host Handoff).

Pure deterministic tests — no LLM calls, no mocks needed.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.behavior_tools.handoff import (
    HandoffSummary,
    build_handoff_summary,
    format_handoff_for_prompt,
    _build_conversation_narrative,
    _build_recommended_actions,
    _build_next_actions,
    _detect_risk_flags,
    _partition_stated_inferred,
)


def _make_state(**overrides):
    """Build a minimal graph state for testing."""
    base = {
        "messages": [],
        "extracted_fields": {},
        "domains_discussed": [],
        "guest_sentiment": None,
        "crisis_active": False,
        "responsible_gaming_count": 0,
        "profile_completeness_score": 0.0,
        "crisis_turn_count": 0,
    }
    base.update(overrides)
    return base


class TestDetectRiskFlags:
    """Test risk flag detection."""

    def test_crisis_active_flagged(self):
        state = _make_state(crisis_active=True)
        flags = _detect_risk_flags(state, [])
        assert any("crisis" in f.lower() for f in flags)

    def test_frustrated_sentiment_flagged(self):
        state = _make_state(guest_sentiment="frustrated")
        flags = _detect_risk_flags(state, [])
        assert any("frustration" in f.lower() for f in flags)

    def test_responsible_gaming_flagged(self):
        state = _make_state(responsible_gaming_count=2)
        flags = _detect_risk_flags(state, [])
        assert any("responsible gaming" in f.lower() for f in flags)

    def test_complaint_in_messages_flagged(self):
        messages = [HumanMessage(content="This is unacceptable service")]
        state = _make_state()
        flags = _detect_risk_flags(state, messages)
        assert any("complaint" in f.lower() for f in flags)

    def test_no_flags_for_happy_guest(self):
        state = _make_state(guest_sentiment="positive")
        flags = _detect_risk_flags(state, [HumanMessage(content="I love this place")])
        assert len(flags) == 0


class TestBuildConversationNarrative:
    """Test conversation narrative builder."""

    def test_includes_guest_name(self):
        narrative = _build_conversation_narrative([], {"name": "Sarah"}, [])
        assert "Sarah" in narrative

    def test_includes_occasion(self):
        narrative = _build_conversation_narrative([], {"occasion": "birthday"}, [])
        assert "birthday" in narrative

    def test_includes_domains(self):
        narrative = _build_conversation_narrative([], {}, ["dining", "entertainment"])
        assert "dining" in narrative
        assert "entertainment" in narrative

    def test_includes_last_request(self):
        messages = [HumanMessage(content="Can you recommend a steakhouse?")]
        narrative = _build_conversation_narrative(messages, {}, [])
        assert "steakhouse" in narrative

    def test_truncates_long_request(self):
        long_msg = "x" * 200
        messages = [HumanMessage(content=long_msg)]
        narrative = _build_conversation_narrative(messages, {}, [])
        assert "..." in narrative

    def test_empty_state_still_returns(self):
        narrative = _build_conversation_narrative([], {}, [])
        assert isinstance(narrative, str)
        assert len(narrative) > 0


class TestBuildRecommendedActions:
    """Test recommended actions builder."""

    def test_crisis_actions(self):
        state = _make_state(crisis_active=True)
        actions = _build_recommended_actions(state, {}, [])
        assert any("welfare" in a.lower() for a in actions)

    def test_frustrated_actions(self):
        state = _make_state(guest_sentiment="frustrated")
        actions = _build_recommended_actions(state, {}, [])
        assert any("frustration" in a.lower() for a in actions)

    def test_occasion_actions(self):
        state = _make_state()
        actions = _build_recommended_actions(state, {"occasion": "birthday"}, [])
        assert any("birthday" in a.lower() for a in actions)

    def test_loyalty_actions(self):
        state = _make_state()
        actions = _build_recommended_actions(state, {"loyalty_tier": "Ignite"}, [])
        assert any("loyalty" in a.lower() or "ignite" in a.lower() for a in actions)

    def test_default_actions_for_empty_state(self):
        state = _make_state()
        actions = _build_recommended_actions(state, {}, [])
        assert len(actions) >= 1


class TestBuildHandoffSummary:
    """Test full handoff summary builder."""

    def test_basic_summary(self):
        state = _make_state(
            messages=[HumanMessage(content="Hi"), AIMessage(content="Hello")],
            extracted_fields={"name": "Mike"},
        )
        summary = build_handoff_summary(state)
        assert isinstance(summary, HandoffSummary)
        assert summary.guest_name == "Mike"
        assert summary.turn_count == 1

    def test_crisis_urgency(self):
        state = _make_state(crisis_active=True)
        summary = build_handoff_summary(state)
        assert summary.urgency == "urgent"

    def test_frustrated_urgency(self):
        state = _make_state(guest_sentiment="frustrated")
        summary = build_handoff_summary(state)
        assert summary.urgency == "priority"

    def test_routine_urgency(self):
        state = _make_state(guest_sentiment="positive")
        summary = build_handoff_summary(state)
        assert summary.urgency == "routine"

    def test_preferences_extracted(self):
        state = _make_state(
            extracted_fields={"preferences": "Italian", "dietary": "gluten-free"},
        )
        summary = build_handoff_summary(state)
        assert len(summary.key_preferences) == 2

    def test_domains_preserved(self):
        state = _make_state(domains_discussed=["dining", "hotel"])
        summary = build_handoff_summary(state)
        assert summary.domains_discussed == ["dining", "hotel"]

    def test_handoff_reason_passed(self):
        state = _make_state()
        summary = build_handoff_summary(state, handoff_reason="Test reason")
        assert summary.handoff_reason == "Test reason"

    def test_serializable(self):
        """Summary must be JSON-serializable (crosses SSE boundary)."""
        state = _make_state(
            messages=[HumanMessage(content="Help me")],
            extracted_fields={"name": "Sarah"},
            crisis_active=True,
        )
        summary = build_handoff_summary(state)
        serialized = json.dumps(summary.model_dump())
        assert json.loads(serialized)

    def test_multi_turn_conversation(self):
        messages = [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello"),
            HumanMessage(content="What dining options do you have?"),
            AIMessage(content="We have several restaurants..."),
            HumanMessage(content="Can I speak to someone?"),
        ]
        state = _make_state(
            messages=messages,
            extracted_fields={"name": "John", "occasion": "anniversary"},
            domains_discussed=["dining"],
        )
        summary = build_handoff_summary(state)
        assert summary.turn_count == 3
        assert summary.guest_name == "John"

    def test_complaint_in_risk_flags(self):
        messages = [HumanMessage(content="This is terrible, I want the manager")]
        state = _make_state(messages=messages)
        summary = build_handoff_summary(state)
        assert any("complaint" in f.lower() for f in summary.risk_flags)


class TestPartitionStatedInferred:
    """R103 fix P9: Stated vs inferred partition."""

    def test_name_is_stated(self):
        stated, inferred = _partition_stated_inferred({"name": "Sarah"})
        assert any("Sarah" in s for s in stated)
        assert len(inferred) == 0

    def test_party_size_is_stated(self):
        stated, inferred = _partition_stated_inferred({"party_size": 4})
        assert any("4" in s for s in stated)

    def test_visit_purpose_is_inferred(self):
        stated, inferred = _partition_stated_inferred({"visit_purpose": "leisure"})
        assert len(stated) == 0
        assert any("leisure" in s for s in inferred)

    def test_mixed_fields_partitioned(self):
        fields = {
            "name": "Mike",
            "occasion": "birthday",
            "visit_purpose": "celebration",
            "budget_signal": "willing to spend",
        }
        stated, inferred = _partition_stated_inferred(fields)
        assert len(stated) == 2  # name, occasion
        assert len(inferred) == 2  # visit_purpose, budget_signal

    def test_empty_fields_skipped(self):
        stated, inferred = _partition_stated_inferred({"name": "", "occasion": None})
        assert len(stated) == 0
        assert len(inferred) == 0

    def test_boolean_fields_skipped(self):
        stated, inferred = _partition_stated_inferred(
            {"urgency": True, "fatigue": True}
        )
        assert len(stated) == 0
        assert len(inferred) == 0


class TestBuildNextActions:
    """R103 fix P9: Conversation-specific next actions."""

    def test_crisis_action_first(self):
        state = _make_state(crisis_active=True)
        actions = _build_next_actions(state, {}, [], [])
        assert any("IMMEDIATE" in a for a in actions)

    def test_named_guest_gets_name_action(self):
        state = _make_state()
        actions = _build_next_actions(state, {"name": "Sarah"}, [], [])
        assert any("Sarah" in a for a in actions)

    def test_unnamed_guest_gets_intro_action(self):
        state = _make_state()
        actions = _build_next_actions(state, {}, [], [])
        assert any("Introduce" in a for a in actions)

    def test_occasion_gets_comp_check(self):
        state = _make_state()
        actions = _build_next_actions(state, {"occasion": "anniversary"}, [], [])
        assert any("anniversary" in a for a in actions)

    def test_last_question_generates_followup(self):
        messages = [HumanMessage(content="What time does the spa close?")]
        state = _make_state(messages=messages)
        actions = _build_next_actions(state, {}, [], [])
        assert any("spa" in a.lower() for a in actions)

    def test_capped_at_five(self):
        state = _make_state(
            crisis_active=True,
            guest_sentiment="frustrated",
            messages=[HumanMessage(content="What about dining?")],
        )
        extracted = {"name": "John", "occasion": "birthday", "loyalty_tier": "Gold"}
        actions = _build_next_actions(state, extracted, [], ["dining", "hotel"])
        assert len(actions) <= 5


class TestHandoffSummaryNewFields:
    """R103 fix P9: Verify new fields in build_handoff_summary."""

    def test_stated_facts_populated(self):
        state = _make_state(
            extracted_fields={"name": "Sarah", "party_size": 4},
        )
        summary = build_handoff_summary(state)
        assert len(summary.guest_stated_facts) == 2

    def test_inferred_facts_populated(self):
        state = _make_state(
            extracted_fields={"visit_purpose": "leisure", "budget_signal": "high"},
        )
        summary = build_handoff_summary(state)
        assert len(summary.agent_inferences) == 2

    def test_next_actions_populated(self):
        state = _make_state(
            messages=[HumanMessage(content="Hi"), AIMessage(content="Hello")],
            extracted_fields={"name": "Mike"},
        )
        summary = build_handoff_summary(state)
        assert len(summary.next_actions) >= 1
        assert any("Mike" in a for a in summary.next_actions)

    def test_new_fields_serializable(self):
        """New fields must survive JSON serialization."""
        state = _make_state(
            extracted_fields={"name": "Sarah", "visit_purpose": "business"},
            messages=[HumanMessage(content="Help")],
        )
        summary = build_handoff_summary(state)
        serialized = json.dumps(summary.model_dump())
        data = json.loads(serialized)
        assert "guest_stated_facts" in data
        assert "agent_inferences" in data
        assert "next_actions" in data


class TestFormatHandoffForPrompt:
    """Test prompt formatting."""

    def test_basic_format(self):
        summary = HandoffSummary(
            guest_name="Sarah",
            conversation_summary="Sarah asked about dining.",
            urgency="routine",
        )
        prompt = format_handoff_for_prompt(summary)
        assert "Sarah" in prompt
        assert "Handoff Preparation" in prompt

    def test_risk_flags_in_prompt(self):
        summary = HandoffSummary(
            risk_flags=["Crisis indicators detected"],
            urgency="urgent",
        )
        prompt = format_handoff_for_prompt(summary)
        assert "Crisis" in prompt
        assert "urgent" in prompt

    def test_recommended_actions_in_prompt(self):
        summary = HandoffSummary(
            recommended_actions=["Check loyalty status"],
            urgency="routine",
        )
        prompt = format_handoff_for_prompt(summary)
        assert "loyalty" in prompt.lower()

    def test_stated_inferred_in_prompt(self):
        """R103 fix P9: stated/inferred partition appears in prompt."""
        summary = HandoffSummary(
            guest_stated_facts=["Name: Sarah", "Party Size: 4"],
            agent_inferences=["Visit Purpose: Leisure"],
            next_actions=["Greet as Sarah"],
            urgency="routine",
        )
        prompt = format_handoff_for_prompt(summary)
        assert "Guest told us" in prompt
        assert "Sarah" in prompt
        assert "We inferred" in prompt
        assert "Your first actions" in prompt

    def test_empty_summary_still_valid(self):
        summary = HandoffSummary()
        prompt = format_handoff_for_prompt(summary)
        assert "Handoff Preparation" in prompt
