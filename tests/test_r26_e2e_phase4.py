"""R26 E2E integration tests for Phase 4 features.

Validates that Phase 4 features (frustration escalation, proactive
suggestions, persona drift, property-aware helplines, CASINO_PROFILES,
suggestion_offered persistence) work together correctly.

All LLM calls are mocked. No API keys required.
"""

from string import Template
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.agents._base import (
    _PERSONA_REINJECT_THRESHOLD,
    _count_consecutive_frustrated,
    execute_specialist,
)
from src.agent.prompts import (
    HEART_ESCALATION_LANGUAGE,
    SENTIMENT_TONE_GUIDES,
    get_persona_style,
    get_responsible_gaming_helplines,
)
from src.agent.state import PropertyQAState, _keep_max, _merge_dicts
from src.casino.config import CASINO_PROFILES, get_casino_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    """Build a minimal PropertyQAState dict with defaults."""
    base = {
        "messages": [HumanMessage(content="What restaurants do you have?")],
        "query_type": "property_qa",
        "router_confidence": 0.9,
        "retrieved_context": [
            {
                "content": "Todd English's Tuscany offers Italian cuisine.",
                "metadata": {"category": "dining"},
                "score": 0.85,
            }
        ],
        "validation_result": None,
        "retry_count": 0,
        "skip_validation": False,
        "retry_feedback": None,
        "current_time": "Saturday, February 22, 2026 10:00 AM UTC",
        "sources_used": [],
        "extracted_fields": {},
        "whisper_plan": None,
        "responsible_gaming_count": 0,
        "guest_sentiment": None,
        "guest_context": {},
        "guest_name": None,
        "suggestion_offered": 0,
    }
    base.update(overrides)
    return base


def _make_llm_mock(response_text: str = "Here are our dining options..."):
    """Create a mock LLM that returns a fixed response."""
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    return mock_llm


def _make_cb_mock(allow: bool = True):
    """Create a mock circuit breaker."""
    cb = MagicMock()
    cb.allow_request = AsyncMock(return_value=allow)
    cb.record_success = AsyncMock()
    cb.record_failure = AsyncMock()
    return cb


# ---------------------------------------------------------------------------
# 1. Frustration -> HEART Escalation Flow
# ---------------------------------------------------------------------------


class TestFrustrationHEARTEscalation:
    """Test that sustained frustration triggers HEART framework language."""

    @pytest.mark.asyncio()
    async def test_two_frustrated_triggers_hear_empathize(self):
        """2 consecutive frustrated messages inject HEAR + EMPATHIZE steps."""
        messages = [
            HumanMessage(content="This is terrible, I can't find anything!"),
            AIMessage(content="I understand your frustration..."),
            HumanMessage(content="I'm frustrated, the service is awful!"),
            AIMessage(content="Let me help you..."),
            HumanMessage(content="This is ridiculous, unacceptable!"),
        ]

        state = _state(
            messages=messages,
            guest_sentiment="frustrated",
        )

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "I completely understand how frustrating that must be."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        result = await execute_specialist(
            state,
            agent_name="host",
            system_prompt_template=Template("You are a concierge for $property_name."),
            context_header="Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
        )

        # Verify HEART language in system prompt
        system_msg = captured_messages[0]
        assert isinstance(system_msg, SystemMessage)
        system_text = system_msg.content

        assert "HEART Framework" in system_text
        assert HEART_ESCALATION_LANGUAGE["hear"] in system_text
        assert HEART_ESCALATION_LANGUAGE["empathize"] in system_text

    @pytest.mark.asyncio()
    async def test_three_frustrated_triggers_full_heart(self):
        """3+ consecutive frustrated messages inject full HEART (5 steps)."""
        messages = [
            HumanMessage(content="This is terrible!"),
            AIMessage(content="I'm sorry..."),
            HumanMessage(content="I can't believe this!"),
            AIMessage(content="Let me help..."),
            HumanMessage(content="I'm fed up with everything!"),
            AIMessage(content="I understand..."),
            HumanMessage(content="This is ridiculous and unacceptable!"),
        ]

        state = _state(
            messages=messages,
            guest_sentiment="frustrated",
        )

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "I'm truly sorry for this experience."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        result = await execute_specialist(
            state,
            agent_name="host",
            system_prompt_template=Template("You are a concierge for $property_name."),
            context_header="Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
        )

        system_text = captured_messages[0].content
        # Full HEART: all 5 steps should be present
        assert HEART_ESCALATION_LANGUAGE["hear"] in system_text
        assert HEART_ESCALATION_LANGUAGE["empathize"] in system_text
        assert HEART_ESCALATION_LANGUAGE["apologize"] in system_text
        assert HEART_ESCALATION_LANGUAGE["resolve"] in system_text
        assert HEART_ESCALATION_LANGUAGE["thank"] in system_text

    @pytest.mark.asyncio()
    async def test_no_escalation_on_positive_sentiment(self):
        """Positive sentiment does NOT trigger HEART escalation."""
        messages = [
            HumanMessage(content="This is great, love the restaurants!"),
        ]

        state = _state(
            messages=messages,
            guest_sentiment="positive",
        )

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "We are glad you are enjoying your visit."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        await execute_specialist(
            state,
            agent_name="host",
            system_prompt_template=Template("You are a concierge for $property_name."),
            context_header="Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
        )

        system_text = captured_messages[0].content
        assert "HEART Framework" not in system_text


# ---------------------------------------------------------------------------
# 2. Proactive Suggestion with Sentiment Gate
# ---------------------------------------------------------------------------


class TestProactiveSuggestionSentimentGate:
    """Test that proactive suggestions require positive evidence of sentiment."""

    @pytest.mark.asyncio()
    async def test_suggestion_injected_with_positive_sentiment(self):
        """High-confidence suggestion + positive sentiment = suggestion injected."""
        whisper_plan = {
            "next_topic": "dining",
            "extraction_targets": ["cuisine_preference"],
            "offer_readiness": 0.5,
            "conversation_note": "Guest interested in dining",
            "proactive_suggestion": "You might enjoy the new sushi bar on Level 2.",
            "suggestion_confidence": 0.9,
        }

        state = _state(
            guest_sentiment="positive",
            whisper_plan=whisper_plan,
            suggestion_offered=0,
        )

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "Great choice! You might also enjoy our sushi bar."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        result = await execute_specialist(
            state,
            agent_name="dining",
            system_prompt_template=Template("You are the dining agent for $property_name."),
            context_header="Dining Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
            include_whisper=True,
        )

        system_text = captured_messages[0].content
        assert "Proactive Suggestion" in system_text
        assert "sushi bar" in system_text
        # suggestion_offered should be set
        assert result.get("suggestion_offered") == 1

    @pytest.mark.asyncio()
    async def test_suggestion_not_injected_with_none_sentiment(self):
        """High-confidence suggestion + None sentiment = NO injection (R23 fix C-002)."""
        whisper_plan = {
            "next_topic": "dining",
            "extraction_targets": [],
            "offer_readiness": 0.5,
            "conversation_note": "Guest interested",
            "proactive_suggestion": "Try the sushi bar!",
            "suggestion_confidence": 0.9,
        }

        state = _state(
            guest_sentiment=None,  # No sentiment evidence
            whisper_plan=whisper_plan,
            suggestion_offered=0,
        )

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "Here are our dining options."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        result = await execute_specialist(
            state,
            agent_name="dining",
            system_prompt_template=Template("You are the dining agent for $property_name."),
            context_header="Dining Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
            include_whisper=True,
        )

        system_text = captured_messages[0].content
        assert "Proactive Suggestion" not in system_text
        assert result.get("suggestion_offered") is None or result.get("suggestion_offered") == 0

    @pytest.mark.asyncio()
    async def test_suggestion_not_injected_when_already_offered(self):
        """suggestion_offered=1 prevents second suggestion (R23 fix C-003)."""
        whisper_plan = {
            "next_topic": "entertainment",
            "extraction_targets": [],
            "offer_readiness": 0.5,
            "conversation_note": "Guest interested",
            "proactive_suggestion": "Check out the comedy show tonight!",
            "suggestion_confidence": 0.95,
        }

        state = _state(
            guest_sentiment="positive",
            whisper_plan=whisper_plan,
            suggestion_offered=1,  # Already offered
        )

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "Here are our entertainment options."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        result = await execute_specialist(
            state,
            agent_name="entertainment",
            system_prompt_template=Template("You are the entertainment agent for $property_name."),
            context_header="Entertainment Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
            include_whisper=True,
        )

        system_text = captured_messages[0].content
        assert "Proactive Suggestion" not in system_text

    @pytest.mark.asyncio()
    async def test_low_confidence_suggestion_not_injected(self):
        """Suggestion with confidence < 0.8 is NOT injected."""
        whisper_plan = {
            "next_topic": "dining",
            "extraction_targets": [],
            "offer_readiness": 0.3,
            "conversation_note": "Guest browsing",
            "proactive_suggestion": "Maybe try the buffet?",
            "suggestion_confidence": 0.5,  # Below threshold
        }

        state = _state(
            guest_sentiment="positive",
            whisper_plan=whisper_plan,
            suggestion_offered=0,
        )

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "Here are our dining options."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        await execute_specialist(
            state,
            agent_name="dining",
            system_prompt_template=Template("You are the dining agent for $property_name."),
            context_header="Dining Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
            include_whisper=True,
        )

        system_text = captured_messages[0].content
        assert "Proactive Suggestion" not in system_text


# ---------------------------------------------------------------------------
# 3. Persona Drift Prevention
# ---------------------------------------------------------------------------


class TestPersonaDriftPrevention:
    """Test persona reminder injection for long conversations."""

    @pytest.mark.asyncio()
    async def test_persona_reminder_injected_after_threshold(self):
        """Persona reminder appears when human turns exceed threshold."""
        # Build a conversation with enough human turns to trigger reinject
        # Threshold is 10 messages, checked as human_turn_count > threshold // 2
        # So human_turn_count > 5 triggers it
        messages = []
        for i in range(7):
            messages.append(HumanMessage(content=f"Question {i}?"))
            messages.append(AIMessage(content=f"Answer {i}"))
        # Add the current question
        messages.append(HumanMessage(content="What about the spa?"))

        state = _state(messages=messages)

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "Our Elemis Spa offers wonderful treatments."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        await execute_specialist(
            state,
            agent_name="host",
            system_prompt_template=Template("You are a concierge for $property_name."),
            context_header="Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
        )

        # Find the persona reminder SystemMessage
        system_msgs = [m for m in captured_messages if isinstance(m, SystemMessage)]
        persona_reminders = [m for m in system_msgs if "PERSONA REMINDER" in m.content]
        assert len(persona_reminders) >= 1
        assert "Seven" in persona_reminders[0].content or "concierge" in persona_reminders[0].content.lower()

    @pytest.mark.asyncio()
    async def test_no_persona_reminder_for_short_conversation(self):
        """Short conversations do NOT get persona reminder."""
        messages = [
            HumanMessage(content="What restaurants do you have?"),
        ]

        state = _state(messages=messages)

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "We have several great restaurants."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        await execute_specialist(
            state,
            agent_name="host",
            system_prompt_template=Template("You are a concierge for $property_name."),
            context_header="Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
        )

        system_msgs = [m for m in captured_messages if isinstance(m, SystemMessage)]
        persona_reminders = [m for m in system_msgs if "PERSONA REMINDER" in m.content]
        assert len(persona_reminders) == 0


# ---------------------------------------------------------------------------
# 4. Property-Aware Helplines
# ---------------------------------------------------------------------------


class TestPropertyAwareHelplines:
    """Test that helplines are correct per property/state."""

    def test_hard_rock_ac_returns_nj_helplines(self):
        """Hard Rock AC (NJ) returns NJ-specific helplines."""
        helplines = get_responsible_gaming_helplines("hard_rock_ac")
        assert "1-800-GAMBLER" in helplines
        assert "NJ" in helplines
        # Should still include national helpline
        assert "1-800-MY-RESET" in helplines

    def test_mohegan_sun_returns_ct_helplines(self):
        """Mohegan Sun (CT) returns CT-specific helplines."""
        helplines = get_responsible_gaming_helplines("mohegan_sun")
        assert "CT" in helplines
        assert "1-888-789-7777" in helplines
        assert "1-800-MY-RESET" in helplines

    def test_foxwoods_returns_ct_helplines(self):
        """Foxwoods (CT) returns CT-specific helplines."""
        helplines = get_responsible_gaming_helplines("foxwoods")
        assert "CT" in helplines
        assert "1-800-MY-RESET" in helplines

    def test_unknown_casino_returns_default(self):
        """Unknown casino ID returns default (CT) helplines."""
        helplines = get_responsible_gaming_helplines("unknown_casino")
        assert "1-800-MY-RESET" in helplines
        assert "Connecticut" in helplines or "CT" in helplines

    def test_none_casino_returns_default(self):
        """No casino_id returns default helplines."""
        helplines = get_responsible_gaming_helplines(None)
        assert "1-800-MY-RESET" in helplines

    def test_nj_helplines_not_in_ct_property(self):
        """CT property should NOT return NJ helplines."""
        helplines = get_responsible_gaming_helplines("mohegan_sun")
        assert "1-800-GAMBLER" not in helplines


# ---------------------------------------------------------------------------
# 5. CASINO_PROFILES Lookup
# ---------------------------------------------------------------------------


class TestCasinoProfiles:
    """Test CASINO_PROFILES data integrity and lookup."""

    def test_all_profiles_have_required_sections(self):
        """Every profile has branding, regulations, operational, prompts sections."""
        required_sections = {"branding", "regulations", "operational", "prompts"}
        for casino_id, profile in CASINO_PROFILES.items():
            for section in required_sections:
                assert section in profile, (
                    f"Casino '{casino_id}' missing section '{section}'"
                )

    def test_mohegan_sun_profile(self):
        """Mohegan Sun has correct state, persona, and helpline data."""
        profile = get_casino_profile("mohegan_sun")
        assert profile["regulations"]["state"] == "CT"
        assert profile["branding"]["persona_name"] == "Seven"
        assert profile["regulations"]["responsible_gaming_helpline"] == "1-800-522-4700"

    def test_hard_rock_ac_profile(self):
        """Hard Rock AC has correct NJ state and persona."""
        profile = get_casino_profile("hard_rock_ac")
        assert profile["regulations"]["state"] == "NJ"
        assert profile["branding"]["persona_name"] == "Ace"
        assert profile["regulations"]["responsible_gaming_helpline"] == "1-800-GAMBLER"

    def test_foxwoods_profile(self):
        """Foxwoods has correct CT state and persona."""
        profile = get_casino_profile("foxwoods")
        assert profile["regulations"]["state"] == "CT"
        assert profile["branding"]["persona_name"] == "Foxy"

    def test_unknown_casino_returns_default_config(self):
        """Unknown casino_id returns DEFAULT_CONFIG."""
        from src.casino.config import DEFAULT_CONFIG

        profile = get_casino_profile("nonexistent_casino")
        assert profile["_id"] == "default"

    def test_each_profile_has_features_section(self):
        """Every profile has a features section matching DEFAULT_FEATURES keys."""
        from src.casino.feature_flags import DEFAULT_FEATURES

        for casino_id, profile in CASINO_PROFILES.items():
            assert "features" in profile, f"Casino '{casino_id}' missing 'features'"
            profile_features = set(profile["features"].keys())
            default_features = set(DEFAULT_FEATURES.keys())
            assert profile_features == default_features, (
                f"Casino '{casino_id}' features mismatch: "
                f"missing={default_features - profile_features}, "
                f"extra={profile_features - default_features}"
            )

    def test_all_profiles_have_id(self):
        """Every profile has a _id matching its dict key."""
        for casino_id, profile in CASINO_PROFILES.items():
            assert profile["_id"] == casino_id


# ---------------------------------------------------------------------------
# 6. suggestion_offered Persistence (_keep_max reducer)
# ---------------------------------------------------------------------------


class TestSuggestionOfferedPersistence:
    """Test that suggestion_offered persists across turns via _keep_max reducer."""

    def test_keep_max_preserves_one(self):
        """_keep_max(1, 0) = 1 -- once offered, stays offered."""
        assert _keep_max(1, 0) == 1

    def test_keep_max_initial(self):
        """_keep_max(0, 0) = 0 -- initial state stays zero."""
        assert _keep_max(0, 0) == 0

    def test_keep_max_updates(self):
        """_keep_max(0, 1) = 1 -- new offering is recorded."""
        assert _keep_max(0, 1) == 1

    def test_keep_max_true_persists(self):
        """Once set to 1, subsequent 0 resets don't reduce it."""
        current = 0
        # Turn 1: suggestion offered
        current = _keep_max(current, 1)
        assert current == 1
        # Turn 2: no suggestion (initial_state sends 0)
        current = _keep_max(current, 0)
        assert current == 1
        # Turn 3: still persists
        current = _keep_max(current, 0)
        assert current == 1


# ---------------------------------------------------------------------------
# 7. _merge_dicts Reducer
# ---------------------------------------------------------------------------


class TestMergeDictsReducer:
    """Test that extracted_fields accumulate across turns."""

    def test_merge_empty_preserves_existing(self):
        """Merging empty dict preserves existing fields."""
        existing = {"name": "Sarah", "party_size": 4}
        result = _merge_dicts(existing, {})
        assert result == {"name": "Sarah", "party_size": 4}

    def test_merge_new_field_adds(self):
        """New field is added to existing fields."""
        existing = {"name": "Sarah"}
        result = _merge_dicts(existing, {"party_size": 4})
        assert result == {"name": "Sarah", "party_size": 4}

    def test_merge_overwrites_same_key(self):
        """Same key from new dict overwrites existing value."""
        existing = {"name": "Sarah"}
        result = _merge_dicts(existing, {"name": "Mike"})
        assert result == {"name": "Mike"}

    def test_merge_multiple_turns(self):
        """Simulate 3 turns of field accumulation."""
        state = {}
        state = _merge_dicts(state, {"name": "Sarah"})
        state = _merge_dicts(state, {"party_size": 6})
        state = _merge_dicts(state, {"occasion": "birthday"})
        assert state == {"name": "Sarah", "party_size": 6, "occasion": "birthday"}


# ---------------------------------------------------------------------------
# 8. Sentiment Tone Guides Integration
# ---------------------------------------------------------------------------


class TestSentimentToneGuides:
    """Test sentiment tone guide injection into specialist agents."""

    @pytest.mark.asyncio()
    async def test_frustrated_tone_guide_injected(self):
        """Frustrated sentiment injects empathetic tone guidance."""
        state = _state(guest_sentiment="frustrated")

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "I understand your frustration. Let me help."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        await execute_specialist(
            state,
            agent_name="host",
            system_prompt_template=Template("You are a concierge for $property_name."),
            context_header="Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
        )

        system_text = captured_messages[0].content
        assert "Tone Guidance" in system_text
        assert "frustrated" in system_text.lower() or "empathetic" in system_text.lower()

    @pytest.mark.asyncio()
    async def test_positive_tone_guide_injected(self):
        """Positive sentiment injects upbeat tone guidance."""
        state = _state(guest_sentiment="positive")

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "Great to hear! Here are our options."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        await execute_specialist(
            state,
            agent_name="host",
            system_prompt_template=Template("You are a concierge for $property_name."),
            context_header="Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
        )

        system_text = captured_messages[0].content
        assert "Tone Guidance" in system_text
        assert "enthusiasm" in system_text.lower() or "upbeat" in system_text.lower()

    @pytest.mark.asyncio()
    async def test_neutral_no_tone_guide(self):
        """Neutral sentiment has empty tone guide -- no section injected."""
        state = _state(guest_sentiment="neutral")

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "Here are our options."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        await execute_specialist(
            state,
            agent_name="host",
            system_prompt_template=Template("You are a concierge for $property_name."),
            context_header="Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
        )

        system_text = captured_messages[0].content
        # Neutral has empty tone guide, so no "Tone Guidance" section
        assert "Tone Guidance" not in system_text


# ---------------------------------------------------------------------------
# 9. Guest Context Injection
# ---------------------------------------------------------------------------


class TestGuestContextInjection:
    """Test guest profile context injection into specialist agents."""

    @pytest.mark.asyncio()
    async def test_guest_context_injected_into_prompt(self):
        """Guest context with name and preferences appears in system prompt."""
        state = _state(
            guest_context={
                "name": "Sarah",
                "party_size": 4,
                "preferences": "vegetarian",
            },
        )

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "Sarah, here are our vegetarian options for your group."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        await execute_specialist(
            state,
            agent_name="dining",
            system_prompt_template=Template("You are the dining agent for $property_name."),
            context_header="Dining Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
        )

        system_text = captured_messages[0].content
        assert "Guest Context" in system_text
        assert "Sarah" in system_text
        assert "4" in system_text
        assert "vegetarian" in system_text

    @pytest.mark.asyncio()
    async def test_empty_guest_context_not_injected(self):
        """Empty guest context does NOT add a section to system prompt."""
        state = _state(guest_context={})

        captured_messages = []

        async def capture_ainvoke(msgs):
            captured_messages.extend(msgs)
            resp = MagicMock()
            resp.content = "Here are our dining options."
            return resp

        mock_llm = MagicMock()
        mock_llm.ainvoke = capture_ainvoke

        await execute_specialist(
            state,
            agent_name="dining",
            system_prompt_template=Template("You are the dining agent for $property_name."),
            context_header="Dining Context",
            no_context_fallback="Please contact us.",
            get_llm_fn=AsyncMock(return_value=mock_llm),
            get_cb_fn=AsyncMock(return_value=_make_cb_mock()),
        )

        system_text = captured_messages[0].content
        assert "Guest Context" not in system_text
