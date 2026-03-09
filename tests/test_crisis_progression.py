"""Tests for multi-turn crisis response progression (R82 Track 2D).

Verifies that crisis responses adapt across turns:
- _build_crisis_followup is a pure function (no LLM needed)
- Spanish crisis followup and turn 1 responses
- Crisis level detection (pure regex-based classification)
- Multi-turn crisis response variation

Mock-based tests (off_topic_node with @patch is_feature_enabled)
removed (mock purge R111).
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.crisis import (
    detect_crisis_level,
    get_crisis_followup_es,
    get_crisis_response_es,
)
from src.agent.nodes import _build_crisis_followup


# ---------------------------------------------------------------------------
# _build_crisis_followup -- pure function tests
# ---------------------------------------------------------------------------


class TestBuildCrisisFollowup:
    """Test _build_crisis_followup generates context-aware responses."""

    def test_default_followup_includes_988(self):
        result = _build_crisis_followup(
            "I'm still struggling", "Mohegan Sun", "1-888-226-7711"
        )
        assert "988" in result

    def test_default_followup_includes_property_name(self):
        result = _build_crisis_followup(
            "I'm still feeling bad", "Mohegan Sun", "1-888-226-7711"
        )
        assert "Mohegan Sun" in result

    def test_default_followup_empathetic_tone(self):
        result = _build_crisis_followup(
            "things are hard", "Mohegan Sun", "1-888-226-7711"
        )
        lower = result.lower()
        assert any(
            phrase in lower for phrase in ["hear you", "going through", "alone", "real"]
        ), f"Expected empathetic opening, got: {result[:120]}"

    def test_on_site_keywords_trigger_team_member_guidance(self):
        result = _build_crisis_followup(
            "Is there someone here I can talk to?", "Mohegan Sun", "1-888-226-7711"
        )
        lower = result.lower()
        assert (
            "team member" in lower or "front desk" in lower or "guest services" in lower
        )

    def test_in_person_keyword_triggers_on_site_path(self):
        result = _build_crisis_followup(
            "I want to talk to someone in person", "Mohegan Sun", "1-888-226-7711"
        )
        lower = result.lower()
        assert "front desk" in lower or "team member" in lower

    def test_face_to_face_keyword_triggers_on_site_path(self):
        result = _build_crisis_followup(
            "Can I speak to someone face to face?", "Mohegan Sun", "1-888-226-7711"
        )
        lower = result.lower()
        assert "team member" in lower or "front desk" in lower

    def test_on_site_path_includes_phone_number(self):
        result = _build_crisis_followup(
            "talk to someone here please", "Mohegan Sun", "1-888-226-7711"
        )
        assert "1-888-226-7711" in result

    def test_on_site_path_still_mentions_988(self):
        result = _build_crisis_followup(
            "I need to talk to someone in person", "Mohegan Sun", "1-888-226-7711"
        )
        assert "988" in result

    def test_default_and_on_site_responses_differ(self):
        default = _build_crisis_followup(
            "I'm still struggling", "Mohegan Sun", "1-888-226-7711"
        )
        on_site = _build_crisis_followup(
            "I want to talk to someone in person", "Mohegan Sun", "1-888-226-7711"
        )
        assert default != on_site


# ---------------------------------------------------------------------------
# Spanish crisis followup
# ---------------------------------------------------------------------------


class TestSpanishCrisisFollowup:
    """Test get_crisis_followup_es generates Spanish-language responses."""

    def test_returns_spanish_text(self):
        result = get_crisis_followup_es(
            "Necesito ayuda", "Mohegan Sun", "1-888-226-7711"
        )
        assert any(
            w in result.lower()
            for w in ["ayuda", "llamar", "apoyo", "hablar", "crisis", "escucho"]
        ), f"Response doesn't appear to be Spanish: {result[:100]}"

    def test_includes_988_helpline(self):
        result = get_crisis_followup_es("Sigo mal", "Mohegan Sun", "1-888-226-7711")
        assert "988" in result

    def test_includes_property_name(self):
        result = get_crisis_followup_es(
            "Necesito hablar con alguien", "Mohegan Sun", "1-888-226-7711"
        )
        assert "Mohegan Sun" in result

    def test_on_site_spanish_keywords(self):
        result = get_crisis_followup_es(
            "Quiero hablar con alguien en persona", "Mohegan Sun", "1-888-226-7711"
        )
        lower = result.lower()
        assert "equipo" in lower or "recepci" in lower

    def test_alguien_aqui_triggers_on_site(self):
        result = get_crisis_followup_es(
            "Hay alguien aqui que me pueda ayudar?", "Mohegan Sun", "1-888-226-7711"
        )
        lower = result.lower()
        assert "equipo" in lower or "recepci" in lower


class TestSpanishCrisisResponse:
    """Test get_crisis_response_es (turn 1 full response)."""

    def test_turn1_includes_988_spanish(self):
        result = get_crisis_response_es("Mohegan Sun", "1-888-226-7711")
        assert "988" in result

    def test_turn1_includes_741741(self):
        result = get_crisis_response_es("Mohegan Sun", "1-888-226-7711")
        assert "741741" in result

    def test_turn1_includes_911(self):
        result = get_crisis_response_es("Mohegan Sun", "1-888-226-7711")
        assert "911" in result

    def test_turn1_different_from_followup(self):
        turn1 = get_crisis_response_es("Mohegan Sun", "1-888-226-7711")
        turn2 = get_crisis_followup_es("Sigo mal", "Mohegan Sun", "1-888-226-7711")
        assert turn1 != turn2, "Turn 1 and turn 2+ Spanish responses are identical"


# ---------------------------------------------------------------------------
# Crisis level detection (regression tests for detect_crisis_level)
# ---------------------------------------------------------------------------


class TestCrisisLevelDetection:
    """Test crisis level classification across severity levels."""

    def test_immediate_active_ideation(self):
        assert detect_crisis_level("I'm thinking about ending it") == "immediate"

    def test_immediate_dont_want_to_live(self):
        assert detect_crisis_level("I don't want to live anymore") == "immediate"

    def test_urgent_lost_everything(self):
        assert (
            detect_crisis_level("I've lost everything, my life is ruined") == "urgent"
        )

    def test_urgent_cant_face_family(self):
        assert detect_crisis_level("I can't face my wife after this") == "urgent"

    def test_concern_chasing_losses(self):
        assert (
            detect_crisis_level("I just need one more win to make it back") == "concern"
        )

    def test_concern_gambling_problem(self):
        assert detect_crisis_level("I think I have a gambling problem") == "concern"

    def test_none_normal_question(self):
        assert detect_crisis_level("What restaurants are open?") == "none"

    def test_none_empty_string(self):
        assert detect_crisis_level("") == "none"

    def test_none_on_none_input(self):
        assert detect_crisis_level(None) == "none"

    def test_gambling_frustration_not_immediate(self):
        level = detect_crisis_level("I lost all my money today, this sucks")
        assert level != "immediate", (
            f"Simple frustration should not be immediate crisis, got: {level}"
        )


# ---------------------------------------------------------------------------
# Multi-turn progression simulation
# ---------------------------------------------------------------------------


class TestMultiTurnCrisisProgression:
    """Simulate a multi-turn crisis conversation to verify response variation."""

    def test_followup_messages_are_not_identical(self):
        turn2 = _build_crisis_followup(
            "I'm still struggling", "Mohegan Sun", "1-888-226-7711"
        )
        turn3 = _build_crisis_followup(
            "Can I talk to someone here?", "Mohegan Sun", "1-888-226-7711"
        )
        assert turn2 != turn3

    def test_all_followup_turns_include_help_resource(self):
        test_messages = [
            "I'm still feeling bad",
            "I don't know what to do",
            "Is anyone here?",
            "I just want to talk",
            "help me please",
        ]
        for msg in test_messages:
            result = _build_crisis_followup(msg, "Mohegan Sun", "1-888-226-7711")
            lower = result.lower()
            assert any(
                kw in lower
                for kw in ["988", "help", "call", "contact", "support", "team"]
            ), f"Turn with message '{msg}' missing help resources: {result[:100]}"

    def test_turn1_response_is_longer_than_followup(self):
        # Simulate turn 1 content (inline, matching off_topic_node logic)
        turn1_content = (
            "I can hear that you're going through a really difficult time, and I want "
            "you to know that help is available right now.\n\n"
            "**Please reach out to these confidential resources:**\n\n"
            "- **988 Suicide & Crisis Lifeline**: Call or text **988** (24/7, free, confidential)\n"
            "- **Crisis Text Line**: Text **HOME** to **741741**\n"
            "- **Emergency**: Call **911** if you or someone is in immediate danger\n\n"
            "You don't have to face this alone."
        )
        followup_content = _build_crisis_followup(
            "I'm still feeling bad", "Mohegan Sun", "1-888-226-7711"
        )
        assert len(turn1_content) > len(followup_content), (
            f"Turn 1 ({len(turn1_content)} chars) should be longer than followup ({len(followup_content)} chars)"
        )
