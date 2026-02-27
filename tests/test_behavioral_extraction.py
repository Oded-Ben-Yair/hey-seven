"""Tests for behavioral signal extraction (B2) and conversation dynamics (B3).

Covers: loyalty signals, urgency cues, fatigue indicators, budget signals,
conversation dynamics detection (terse replies, repeated questions, brevity).
"""

import pytest

from src.agent.extraction import extract_fields


# ---------------------------------------------------------------------------
# B2: Loyalty signal extraction
# ---------------------------------------------------------------------------


class TestLoyaltyExtraction:
    """Loyalty signals from guest messages."""

    @pytest.mark.parametrize(
        "text",
        [
            "I'm a Momentum member, 20 years now",
            "Been a member for 10 years",
            "I've been coming here for 5 years",
            "Gold tier member",
            "I'm a Platinum member",
            "Used to be Gold",
            "I spend a lot here every month",
            "I come here every weekend",
        ],
    )
    def test_loyalty_detected(self, text):
        result = extract_fields(text)
        assert "loyalty_signal" in result, f"Expected loyalty_signal for: {text}"

    @pytest.mark.parametrize(
        "text",
        [
            "What restaurants do you have?",
            "Where is the pool?",
            "I need a restaurant for 4",
        ],
    )
    def test_no_false_positive_loyalty(self, text):
        result = extract_fields(text)
        assert "loyalty_signal" not in result


# ---------------------------------------------------------------------------
# B2: Urgency signal extraction
# ---------------------------------------------------------------------------


class TestUrgencyExtraction:
    """Urgency cues from guest messages."""

    @pytest.mark.parametrize(
        "text",
        [
            "Checking out in an hour. Anywhere I can grab breakfast?",
            "We're leaving soon, need a quick bite",
            "I need something fast",
            "Flight in 3 hours, hurry",
            "Don't have much time",
            "Running late, where's the buffet?",
        ],
    )
    def test_urgency_detected(self, text):
        result = extract_fields(text)
        assert result.get("urgency") is True, f"Expected urgency for: {text}"

    @pytest.mark.parametrize(
        "text",
        [
            "What time does the steakhouse open?",
            "Tell me about the spa",
            "We're celebrating an anniversary",
        ],
    )
    def test_no_false_positive_urgency(self, text):
        result = extract_fields(text)
        assert "urgency" not in result


# ---------------------------------------------------------------------------
# B2: Fatigue signal extraction
# ---------------------------------------------------------------------------


class TestFatigueExtraction:
    """Fatigue indicators from guest messages."""

    @pytest.mark.parametrize(
        "text",
        [
            "We're exhausted from the drive",
            "Long day at the conference. Tired.",
            "We drove 4 hours to get here",
            "On our feet all day, need to unwind",
            "I just want to relax after a long flight",
            "Wiped out from traveling",
        ],
    )
    def test_fatigue_detected(self, text):
        result = extract_fields(text)
        assert result.get("fatigue") is True, f"Expected fatigue for: {text}"

    @pytest.mark.parametrize(
        "text",
        [
            "What shows are on tonight?",
            "We're excited to be here",
            "Where's the steakhouse?",
        ],
    )
    def test_no_false_positive_fatigue(self, text):
        result = extract_fields(text)
        assert "fatigue" not in result


# ---------------------------------------------------------------------------
# B2: Budget signal extraction
# ---------------------------------------------------------------------------


class TestBudgetExtraction:
    """Budget consciousness signals from guest messages."""

    @pytest.mark.parametrize(
        "text",
        [
            "What's good for dinner? Nothing too expensive",
            "Any affordable dining options?",
            "We're on a budget this trip",
            "Are there any free shows?",
            "Don't want to spend too much on dinner",
            "Something cheap and quick",
        ],
    )
    def test_budget_detected(self, text):
        result = extract_fields(text)
        assert result.get("budget_conscious") is True, f"Expected budget for: {text}"

    @pytest.mark.parametrize(
        "text",
        [
            "What's the best steakhouse?",
            "Where should we go for dinner tonight?",
            "We're celebrating our anniversary",
        ],
    )
    def test_no_false_positive_budget(self, text):
        result = extract_fields(text)
        assert "budget_conscious" not in result


# ---------------------------------------------------------------------------
# B2: Combined implicit signal extraction
# ---------------------------------------------------------------------------


class TestCombinedImplicitSignals:
    """Multiple implicit signals in a single message."""

    def test_fatigue_and_urgency(self):
        result = extract_fields("Exhausted and leaving soon, where can I eat quick?")
        assert result.get("fatigue") is True
        assert result.get("urgency") is True

    def test_loyalty_and_occasion(self):
        result = extract_fields("Momentum member, here for my birthday")
        assert "loyalty_signal" in result
        assert result.get("occasion") == "birthday"

    def test_budget_and_party_size(self):
        result = extract_fields("Party of 6, nothing too expensive")
        assert result.get("party_size") == 6
        assert result.get("budget_conscious") is True


# ---------------------------------------------------------------------------
# B3: Conversation dynamics detection
# ---------------------------------------------------------------------------


class TestConversationDynamics:
    """Tests for _detect_conversation_dynamics from _base.py."""

    def test_terse_replies_detected(self):
        from langchain_core.messages import AIMessage, HumanMessage
        from src.agent.agents._base import _detect_conversation_dynamics

        messages = [
            HumanMessage(content="What should we do?"),
            AIMessage(content="Here are some options: spa, dining, shows..."),
            HumanMessage(content="ok"),
            AIMessage(content="Would you like to try the steakhouse?"),
            HumanMessage(content="fine"),
            AIMessage(content="The steakhouse opens at 5 PM..."),
            HumanMessage(content="sure"),
        ]
        dynamics = _detect_conversation_dynamics(messages)
        assert dynamics["terse_replies"] >= 2

    def test_repeated_question_detected(self):
        from langchain_core.messages import HumanMessage
        from src.agent.agents._base import _detect_conversation_dynamics

        messages = [
            HumanMessage(content="What time does the buffet open?"),
            HumanMessage(content="When does the buffet open?"),
        ]
        dynamics = _detect_conversation_dynamics(messages)
        assert dynamics["repeated_question"] is True

    def test_brevity_preference_detected(self):
        from langchain_core.messages import HumanMessage
        from src.agent.agents._base import _detect_conversation_dynamics

        messages = [
            HumanMessage(content="steakhouse hours"),
            HumanMessage(content="spa hours"),
            HumanMessage(content="pool open?"),
        ]
        dynamics = _detect_conversation_dynamics(messages)
        assert dynamics["brevity_preference"] is True

    def test_no_dynamics_for_normal_conversation(self):
        from langchain_core.messages import AIMessage, HumanMessage
        from src.agent.agents._base import _detect_conversation_dynamics

        messages = [
            HumanMessage(content="What restaurants are available tonight for dinner?"),
            AIMessage(content="Great question! We have several options..."),
            HumanMessage(content="The steakhouse sounds good, what time does it open?"),
        ]
        dynamics = _detect_conversation_dynamics(messages)
        assert dynamics["terse_replies"] == 0
        assert dynamics["repeated_question"] is False
        assert dynamics["brevity_preference"] is False

    def test_empty_messages(self):
        from src.agent.agents._base import _detect_conversation_dynamics

        dynamics = _detect_conversation_dynamics([])
        assert dynamics["turn_count"] == 0
        assert dynamics["terse_replies"] == 0

    def test_turn_count_accurate(self):
        from langchain_core.messages import AIMessage, HumanMessage
        from src.agent.agents._base import _detect_conversation_dynamics

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Welcome!"),
            HumanMessage(content="What's for dinner?"),
            AIMessage(content="We have options..."),
            HumanMessage(content="The steakhouse please"),
        ]
        dynamics = _detect_conversation_dynamics(messages)
        assert dynamics["turn_count"] == 3
