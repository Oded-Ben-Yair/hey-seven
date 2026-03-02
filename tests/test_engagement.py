"""Tests for conversation engagement dynamics (Phase 2, B3).

Covers: list format detection, conversation phase detection,
terse reply guidance, format adaptation prompt sections.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.agents._base import (
    _build_behavioral_prompt_sections,
    _build_cross_domain_hint,
    _detect_conversation_dynamics,
)


# ---------------------------------------------------------------------------
# List format detection
# ---------------------------------------------------------------------------


class TestListFormatDetection:
    """Detect when guests send numbered or bullet lists."""

    def test_numbered_list_detected(self):
        msgs = [HumanMessage(content="I need:\n1. Restaurant\n2. Show\n3. Hotel")]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["list_format"] is True

    def test_numbered_list_with_parens(self):
        msgs = [HumanMessage(content="Looking for:\n1) Steakhouse\n2) Pool")]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["list_format"] is True

    def test_bullet_list_detected(self):
        msgs = [HumanMessage(content="Looking for:\n- Steakhouse\n- Pool\n- Spa")]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["list_format"] is True

    def test_asterisk_bullets_detected(self):
        msgs = [HumanMessage(content="Want to know about:\n* Pool\n* Spa")]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["list_format"] is True

    def test_no_list_format(self):
        msgs = [HumanMessage(content="What restaurants do you have?")]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["list_format"] is False

    def test_only_last_message_checked(self):
        """List format is based on the LAST human message, not earlier ones."""
        msgs = [
            HumanMessage(content="I need:\n1. Restaurant\n2. Show"),
            HumanMessage(content="What time is dinner?"),
        ]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["list_format"] is False


# ---------------------------------------------------------------------------
# Conversation phase detection
# ---------------------------------------------------------------------------


class TestConversationPhase:
    """Detect conversation phase based on turn count."""

    def test_opening_phase_1_message(self):
        msgs = [HumanMessage(content="Hello")]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["conversation_phase"] == "opening"

    def test_opening_phase_2_messages(self):
        msgs = [HumanMessage(content="Hi"), HumanMessage(content="What's up?")]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["conversation_phase"] == "opening"

    def test_exploring_phase_3_messages(self):
        msgs = [HumanMessage(content=f"Question {i}") for i in range(3)]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["conversation_phase"] == "exploring"

    def test_exploring_phase_6_messages(self):
        msgs = [HumanMessage(content=f"Question {i}") for i in range(6)]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["conversation_phase"] == "exploring"

    def test_deciding_phase_7_messages(self):
        msgs = [HumanMessage(content=f"Question {i}") for i in range(7)]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["conversation_phase"] == "deciding"

    def test_empty_messages_is_opening(self):
        dynamics = _detect_conversation_dynamics([])
        assert dynamics["conversation_phase"] == "opening"

    def test_ai_messages_not_counted(self):
        """Only HumanMessage instances count for turn count."""
        msgs = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="Question"),
            AIMessage(content="Answer"),
        ]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["turn_count"] == 2
        assert dynamics["conversation_phase"] == "opening"


# ---------------------------------------------------------------------------
# Terse reply detection
# ---------------------------------------------------------------------------


class TestTerseReplyDetection:
    """Detect when guests give very short replies."""

    def test_terse_replies_counted(self):
        msgs = [
            HumanMessage(content="ok"),
            HumanMessage(content="sure"),
            HumanMessage(content="fine"),
        ]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["terse_replies"] == 3

    def test_no_terse_replies(self):
        msgs = [HumanMessage(content="I would like to know about restaurants available at the resort")]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["terse_replies"] == 0

    def test_brevity_preference(self):
        msgs = [
            HumanMessage(content="ok"),
            HumanMessage(content="next"),
            HumanMessage(content="sure"),
        ]
        dynamics = _detect_conversation_dynamics(msgs)
        assert dynamics["brevity_preference"] is True


# ---------------------------------------------------------------------------
# Cross-domain engagement hint
# ---------------------------------------------------------------------------


class TestCrossDomainHint:
    """R74 B3: Cross-domain engagement variation."""

    def test_empty_domains_no_hint(self):
        assert _build_cross_domain_hint([]) == ""

    def test_some_explored_suggests_unexplored(self):
        hint = _build_cross_domain_hint(["dining", "entertainment"])
        assert "Cross-Domain Awareness" in hint
        assert "dining" in hint
        assert "entertainment" in hint
        # Should suggest unexplored domains
        assert "spa" in hint or "hotel" in hint or "gaming" in hint

    def test_all_explored_no_hint(self):
        all_domains = list(
            {"dining", "entertainment", "hotel", "spa", "gaming", "shopping", "promotions", "comp"}
        )
        hint = _build_cross_domain_hint(all_domains)
        assert hint == ""

    def test_max_3_suggestions(self):
        hint = _build_cross_domain_hint(["dining"])
        # R82 Track 2C: bridge templates replace generic "you could mention" for known pairs.
        # At most 3 bridge lines (each starts with "- <domain>:")
        bridge_lines = [line for line in hint.split("\n") if line.startswith("- ")]
        assert len(bridge_lines) <= 3
