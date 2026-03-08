"""Tests for Rapport Ladder tool (H6: Rapport Depth).

Pure deterministic tests — no LLM calls, no mocks needed.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.behavior_tools.rapport_ladder import (
    RapportPattern,
    get_rapport_patterns,
    get_rapport_prompt_section,
    _infer_guest_type,
    _TECHNIQUES,
)


class TestTechniqueCatalog:
    """Test technique catalog structure."""

    def test_all_techniques_have_required_fields(self):
        for t in _TECHNIQUES:
            assert t.technique
            assert t.description
            assert t.example
            assert len(t.applicable_guest_types) > 0

    def test_technique_names_unique(self):
        names = [t.technique for t in _TECHNIQUES]
        assert len(names) == len(set(names))

    def test_at_least_7_techniques(self):
        assert len(_TECHNIQUES) >= 7

    def test_examples_are_substantial(self):
        for t in _TECHNIQUES:
            assert len(t.example) > 20, f"Short example for {t.technique}"


class TestInferGuestType:
    """Test guest type inference."""

    def test_grief_sentiment(self):
        assert _infer_guest_type({}, "grief", 3) == "grieving"

    def test_celebration_sentiment(self):
        assert _infer_guest_type({}, "celebration", 3) == "celebrating"

    def test_birthday_occasion(self):
        assert _infer_guest_type({"occasion": "birthday"}, None, 3) == "celebrating"

    def test_anniversary_occasion(self):
        assert _infer_guest_type({"occasion": "anniversary"}, None, 3) == "celebrating"

    def test_family_party(self):
        assert (
            _infer_guest_type({"party_composition": "with kids"}, None, 3) == "family"
        )

    def test_couple(self):
        assert (
            _infer_guest_type({"party_composition": "with my wife"}, None, 3)
            == "couple"
        )

    def test_vip_loyalty(self):
        assert _infer_guest_type({"loyalty_tier": "Soar"}, None, 3) == "vip"

    def test_regular_loyalty(self):
        assert _infer_guest_type({"loyalty_tier": "Ignite"}, None, 3) == "regular"

    def test_regular_from_signal(self):
        assert (
            _infer_guest_type({"loyalty_signal": "I come here regularly"}, None, 3)
            == "regular"
        )

    def test_first_timer_default(self):
        assert _infer_guest_type({}, None, 1) == "first_timer"

    def test_solo_from_party_size(self):
        assert _infer_guest_type({"party_size": "1"}, None, 3) == "solo"

    def test_sentiment_overrides_occasion(self):
        """Grief sentiment takes priority over birthday occasion."""
        assert _infer_guest_type({"occasion": "birthday"}, "grief", 3) == "grieving"


class TestGetRapportPatterns:
    """Test pattern selection."""

    def test_returns_max_3_patterns(self):
        patterns = get_rapport_patterns("first_timer")
        assert len(patterns) <= 3

    def test_first_timer_gets_patterns(self):
        patterns = get_rapport_patterns("first_timer")
        assert len(patterns) > 0

    def test_vip_gets_patterns(self):
        patterns = get_rapport_patterns("vip")
        assert len(patterns) > 0

    def test_grieving_gets_empathetic(self):
        patterns = get_rapport_patterns("grieving")
        techniques = [p.technique for p in patterns]
        assert "empathetic_mirror" in techniques

    def test_family_gets_family_focus(self):
        patterns = get_rapport_patterns("family")
        techniques = [p.technique for p in patterns]
        assert "family_focus" in techniques

    def test_name_callback_requires_name(self):
        """name_callback should only appear when guest name is known."""
        patterns_no_name = get_rapport_patterns("regular")
        patterns_with_name = get_rapport_patterns("regular", guest_name="Sarah")
        no_name_techniques = [p.technique for p in patterns_no_name]
        with_name_techniques = [p.technique for p in patterns_with_name]
        assert "name_callback" not in no_name_techniques
        # May or may not be in with_name depending on scoring

    def test_closing_phase_boosts_return_anchor(self):
        patterns = get_rapport_patterns("regular", conversation_phase="closing")
        techniques = [p.technique for p in patterns]
        # return_anchor should rank higher in closing phase
        assert "return_anchor" in techniques

    def test_experience_bridge_boosted_multi_domain(self):
        patterns = get_rapport_patterns(
            "regular",
            domains_discussed=["dining", "spa", "entertainment"],
        )
        techniques = [p.technique for p in patterns]
        assert "experience_bridge" in techniques

    def test_all_guest_types_get_patterns(self):
        for guest_type in (
            "first_timer",
            "regular",
            "vip",
            "family",
            "couple",
            "solo",
            "grieving",
            "celebrating",
        ):
            patterns = get_rapport_patterns(guest_type)
            assert len(patterns) > 0, f"No patterns for {guest_type}"


class TestGetRapportPromptSection:
    """Test prompt section generation."""

    def test_basic_section(self):
        state = {
            "messages": [HumanMessage(content="Hi")] * 3,
            "extracted_fields": {},
            "guest_sentiment": None,
            "domains_discussed": [],
            "guest_name": None,
        }
        section = get_rapport_prompt_section(state)
        assert "Rapport Technique" in section

    def test_frustrated_returns_empty(self):
        state = {
            "messages": [HumanMessage(content="Hi")] * 3,
            "extracted_fields": {},
            "guest_sentiment": "frustrated",
            "domains_discussed": [],
            "guest_name": None,
        }
        section = get_rapport_prompt_section(state)
        assert section == ""

    def test_negative_returns_empty(self):
        state = {
            "messages": [HumanMessage(content="Hi")] * 3,
            "extracted_fields": {},
            "guest_sentiment": "negative",
            "domains_discussed": [],
            "guest_name": None,
        }
        section = get_rapport_prompt_section(state)
        assert section == ""

    def test_grieving_gets_empathetic(self):
        state = {
            "messages": [HumanMessage(content="My dad passed")] * 2,
            "extracted_fields": {},
            "guest_sentiment": "grief",
            "domains_discussed": [],
            "guest_name": None,
        }
        section = get_rapport_prompt_section(state)
        assert "empathetic_mirror" in section

    def test_celebrating_guest(self):
        state = {
            "messages": [HumanMessage(content="It's my birthday!")] * 3,
            "extracted_fields": {"occasion": "birthday"},
            "guest_sentiment": "excited",
            "domains_discussed": ["dining"],
            "guest_name": "Sarah",
        }
        section = get_rapport_prompt_section(state)
        assert "celebrating" in section

    def test_section_has_adapt_instruction(self):
        state = {
            "messages": [HumanMessage(content="Hi")] * 3,
            "extracted_fields": {},
            "guest_sentiment": None,
            "domains_discussed": [],
            "guest_name": None,
        }
        section = get_rapport_prompt_section(state)
        assert "Adapt" in section

    def test_conversation_phase_detection(self):
        """Short conversation should show 'opening' phase."""
        state = {
            "messages": [HumanMessage(content="Hi")],
            "extracted_fields": {},
            "guest_sentiment": None,
            "domains_discussed": [],
            "guest_name": None,
        }
        section = get_rapport_prompt_section(state)
        assert "opening" in section

    def test_closing_phase_for_long_conversation(self):
        state = {
            "messages": [HumanMessage(content="x")] * 10,
            "extracted_fields": {},
            "guest_sentiment": None,
            "domains_discussed": ["dining", "hotel"],
            "guest_name": "Mike",
        }
        section = get_rapport_prompt_section(state)
        assert "closing" in section


class TestRapportModels:
    """Test Pydantic model validation."""

    def test_pattern_serializable(self):
        pattern = RapportPattern(
            technique="test",
            description="Test technique",
            example="Test example text",
            applicable_guest_types=["first_timer"],
        )
        serialized = json.dumps(pattern.model_dump())
        assert json.loads(serialized)


# ---------------------------------------------------------------------------
# R105: Phase-aware rapport patterns
# ---------------------------------------------------------------------------


class TestR105PhaseAwareRapport:
    """R105: Rapport patterns use profiling_phase from state, not just turn_count."""

    def test_foundation_phase_uses_opening_patterns(self):
        """Foundation profiling phase should map to opening conversation phase."""
        state = {
            "messages": [HumanMessage(content="x")]
            * 5,  # turn_count=5 would be "deciding"
            "extracted_fields": {},
            "guest_sentiment": None,
            "domains_discussed": [],
            "guest_name": None,
            "profiling_phase": "foundation",
        }
        section = get_rapport_prompt_section(state)
        assert "opening" in section

    def test_preference_phase_uses_exploring_patterns(self):
        """Preference profiling phase should map to exploring conversation phase."""
        state = {
            "messages": [HumanMessage(content="x")]
            * 1,  # turn_count=1 would be "opening"
            "extracted_fields": {"name": "Mike", "party_size": "4"},
            "guest_sentiment": None,
            "domains_discussed": [],
            "guest_name": "Mike",
            "profiling_phase": "preference",
        }
        section = get_rapport_prompt_section(state)
        assert "exploring" in section

    def test_relationship_phase_uses_deciding_patterns(self):
        """Relationship profiling phase should map to deciding conversation phase."""
        state = {
            "messages": [HumanMessage(content="x")]
            * 3,  # turn_count=3 would be "exploring"
            "extracted_fields": {"name": "Sarah", "preferences": "steak"},
            "guest_sentiment": None,
            "domains_discussed": ["dining"],
            "guest_name": "Sarah",
            "profiling_phase": "relationship",
        }
        section = get_rapport_prompt_section(state)
        assert "deciding" in section

    def test_behavioral_phase_uses_closing_patterns(self):
        """Behavioral profiling phase should map to closing conversation phase."""
        state = {
            "messages": [HumanMessage(content="x")]
            * 3,  # turn_count=3 would be "exploring"
            "extracted_fields": {
                "name": "Mike",
                "preferences": "seafood",
                "occasion": "birthday",
            },
            "guest_sentiment": None,
            "domains_discussed": ["dining", "entertainment"],
            "guest_name": "Mike",
            "profiling_phase": "behavioral",
        }
        section = get_rapport_prompt_section(state)
        assert "closing" in section

    def test_profiling_phase_overrides_turn_count(self):
        """Profiling phase from state should take priority over turn_count inference."""
        state = {
            "messages": [HumanMessage(content="x")]
            * 10,  # turn_count=10 would be "closing"
            "extracted_fields": {},
            "guest_sentiment": None,
            "domains_discussed": [],
            "guest_name": None,
            "profiling_phase": "foundation",  # Should override to "opening"
        }
        section = get_rapport_prompt_section(state)
        # With 10 turns, turn_count would give "closing"
        # But profiling_phase="foundation" should override to "opening"
        assert "opening" in section

    def test_fallback_to_turn_count_when_no_phase(self):
        """Without profiling_phase in state, should fall back to turn_count."""
        state = {
            "messages": [HumanMessage(content="x")] * 10,
            "extracted_fields": {},
            "guest_sentiment": None,
            "domains_discussed": [],
            "guest_name": None,
            # No profiling_phase key
        }
        section = get_rapport_prompt_section(state)
        # 10 human messages = closing phase
        assert "closing" in section
