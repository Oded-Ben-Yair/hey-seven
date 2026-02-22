"""R26 conversation quality tests: topic switching, context retention, sentiment transitions.

Tests verify conversation-level behavior: correct handling of topic
transitions, multi-turn context persistence, and sentiment state
management. All LLM calls are mocked; deterministic functions
(detect_sentiment, extract_fields) are tested directly.
"""

from string import Template
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.extraction import extract_fields
from src.agent.nodes import route_from_router
from src.agent.sentiment import detect_sentiment
from src.agent.state import PropertyQAState, RouterOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    """Build a minimal PropertyQAState dict with defaults."""
    base = {
        "messages": [],
        "query_type": None,
        "router_confidence": 0.0,
        "retrieved_context": [],
        "validation_result": None,
        "retry_count": 0,
        "skip_validation": False,
        "retry_feedback": None,
        "current_time": "Saturday 10:00 AM",
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


def _mock_router_result(query_type: str, confidence: float = 0.9):
    """Create a mock that returns a RouterOutput for a given query_type."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        return_value=RouterOutput(query_type=query_type, confidence=confidence)
    )
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


# ---------------------------------------------------------------------------
# 1. Topic Switching Tests
# ---------------------------------------------------------------------------


class TestTopicSwitching:
    """Tests for topic transition handling across conversation turns."""

    @pytest.mark.asyncio()
    @patch("src.agent.nodes._get_llm", new_callable=AsyncMock)
    async def test_dining_to_entertainment_routing(self, mock_get_llm):
        """Guest asks about dining, then entertainment -- each routed correctly."""
        from src.agent.nodes import router_node

        # Turn 1: Dining query
        mock_get_llm.return_value = _mock_router_result("property_qa")
        state_turn1 = _state(
            messages=[HumanMessage(content="What restaurants do you have?")]
        )
        result1 = await router_node(state_turn1)
        assert result1["query_type"] == "property_qa"

        # Turn 2: Entertainment query (different topic)
        mock_get_llm.return_value = _mock_router_result("property_qa", 0.92)
        state_turn2 = _state(
            messages=[
                HumanMessage(content="What restaurants do you have?"),
                AIMessage(content="We have several great restaurants..."),
                HumanMessage(content="What shows are playing this weekend?"),
            ]
        )
        result2 = await router_node(state_turn2)
        assert result2["query_type"] == "property_qa"

    @pytest.mark.asyncio()
    @patch("src.agent.nodes._get_llm", new_callable=AsyncMock)
    async def test_dining_to_entertainment_to_hotel(self, mock_get_llm):
        """Three consecutive topic switches route correctly."""
        from src.agent.nodes import router_node

        topics = [
            ("What are the best restaurants?", "property_qa"),
            ("Any concerts this weekend?", "property_qa"),
            ("What room types do you have?", "property_qa"),
        ]
        messages = []
        for query, expected_type in topics:
            messages.append(HumanMessage(content=query))
            mock_get_llm.return_value = _mock_router_result(expected_type)
            state = _state(messages=list(messages))
            result = await router_node(state)
            assert result["query_type"] == expected_type
            messages.append(AIMessage(content=f"Here is info about {query}..."))

    @pytest.mark.asyncio()
    @patch("src.agent.nodes._get_llm", new_callable=AsyncMock)
    async def test_property_qa_to_greeting(self, mock_get_llm):
        """Mid-conversation greeting routes to greeting node."""
        from src.agent.nodes import router_node

        mock_get_llm.return_value = _mock_router_result("greeting")
        state = _state(
            messages=[
                HumanMessage(content="What restaurants do you have?"),
                AIMessage(content="We have Todd English's Tuscany..."),
                HumanMessage(content="Thanks! Hello again!"),
            ]
        )
        result = await router_node(state)
        assert result["query_type"] == "greeting"

    @pytest.mark.asyncio()
    @patch("src.agent.nodes._get_llm", new_callable=AsyncMock)
    async def test_property_qa_to_off_topic(self, mock_get_llm):
        """Guest switches from property question to off-topic."""
        from src.agent.nodes import router_node

        mock_get_llm.return_value = _mock_router_result("off_topic")
        state = _state(
            messages=[
                HumanMessage(content="What restaurants do you have?"),
                AIMessage(content="We have several options..."),
                HumanMessage(content="What is the weather like today?"),
            ]
        )
        result = await router_node(state)
        assert result["query_type"] == "off_topic"

    def test_route_from_router_property_qa(self):
        """property_qa with high confidence routes to retrieve."""
        state = _state(query_type="property_qa", router_confidence=0.9)
        assert route_from_router(state) == "retrieve"

    def test_route_from_router_greeting(self):
        """greeting routes to greeting node."""
        state = _state(query_type="greeting", router_confidence=0.99)
        assert route_from_router(state) == "greeting"


# ---------------------------------------------------------------------------
# 2. Context Retention Tests
# ---------------------------------------------------------------------------


class TestContextRetention:
    """Tests for multi-turn context persistence."""

    def test_name_persists_across_topics(self):
        """Guest name extracted on turn 1 persists through topic switches."""
        # Turn 1: Guest introduces themselves
        fields1 = extract_fields("Hi, I'm Sarah and I'm looking for restaurants")
        assert fields1.get("name") == "Sarah"

        # Turn 2: Different topic -- name should still be in extracted_fields
        # Simulate state accumulation via _merge_dicts reducer behavior
        accumulated = dict(fields1)
        fields2 = extract_fields("What shows are playing tonight?")
        accumulated.update(fields2)  # merge new fields (empty for this query)
        assert accumulated.get("name") == "Sarah"

        # Turn 3: Another topic -- name still persists
        fields3 = extract_fields("Do you have a pool?")
        accumulated.update(fields3)
        assert accumulated.get("name") == "Sarah"

    def test_party_size_persists(self):
        """Party size from early turn persists through subsequent turns."""
        fields1 = extract_fields("There are 4 of us visiting this weekend")
        assert fields1.get("party_size") == 4

        accumulated = dict(fields1)
        fields2 = extract_fields("What time does the buffet open?")
        accumulated.update(fields2)
        assert accumulated.get("party_size") == 4

    def test_dietary_preference_retained(self):
        """Dietary preference extracted early persists."""
        fields1 = extract_fields("I'm vegetarian, what dining options do you have?")
        assert "vegetarian" in fields1.get("preferences", "")

        accumulated = dict(fields1)
        fields2 = extract_fields("Also, what about the steakhouse?")
        accumulated.update(fields2)
        assert "vegetarian" in accumulated.get("preferences", "")

    def test_multiple_fields_accumulate(self):
        """Fields from different turns accumulate correctly."""
        # Turn 1: name
        accumulated = {}
        fields1 = extract_fields("I'm Sarah")
        accumulated.update(fields1)
        assert accumulated.get("name") == "Sarah"

        # Turn 2: party size
        fields2 = extract_fields("There are 6 of us")
        accumulated.update(fields2)
        assert accumulated.get("name") == "Sarah"
        assert accumulated.get("party_size") == 6

        # Turn 3: occasion
        fields3 = extract_fields("We're celebrating a birthday")
        accumulated.update(fields3)
        assert accumulated.get("name") == "Sarah"
        assert accumulated.get("party_size") == 6
        assert accumulated.get("occasion") == "birthday"

    def test_return_to_previous_topic(self):
        """Guest returns to a previous topic -- context is retained."""
        # This tests that the _merge_dicts reducer doesn't lose earlier data.
        # Simulate 3 turns: dining -> spa -> "back to dinner"
        turn1_fields = extract_fields("I'm Sarah, looking for dinner for party of 4")
        assert turn1_fields.get("name") == "Sarah"
        assert turn1_fields.get("party_size") == 4

        accumulated = dict(turn1_fields)

        # Turn 2: Spa (different topic, may extract nothing new)
        turn2_fields = extract_fields("What about the spa?")
        accumulated.update(turn2_fields)

        # Turn 3: Return to dinner -- original fields still there
        turn3_fields = extract_fields("Back to dinner options, I'm vegetarian by the way")
        accumulated.update(turn3_fields)
        assert accumulated.get("name") == "Sarah"
        assert accumulated.get("party_size") == 4
        assert "vegetarian" in accumulated.get("preferences", "")


# ---------------------------------------------------------------------------
# 3. Sentiment Transition Tests
# ---------------------------------------------------------------------------


class TestSentimentTransitions:
    """Tests for sentiment state changes across conversation turns."""

    def test_neutral_to_positive_transition(self):
        """Neutral start, positive response -- sentiment updates correctly."""
        s1 = detect_sentiment("What restaurants do you have?")
        assert s1 == "neutral"

        s2 = detect_sentiment("That sounds amazing, thank you so much!")
        assert s2 == "positive"

    def test_frustrated_to_positive_transition(self):
        """Guest starts frustrated, gets good answer, mood improves."""
        s1 = detect_sentiment("This is ridiculous, I can't find anything!")
        assert s1 == "frustrated"

        # After getting a helpful answer, guest is happy
        s2 = detect_sentiment("Oh that's great, exactly what I needed!")
        assert s2 == "positive"

    def test_positive_does_not_trigger_escalation(self):
        """Positive sentiment on latest turn should not trigger escalation."""
        from src.agent.agents._base import _count_consecutive_frustrated

        messages = [
            HumanMessage(content="This is terrible!"),  # frustrated
            AIMessage(content="I understand your frustration..."),
            HumanMessage(content="I can't believe this!"),  # frustrated
            AIMessage(content="Let me help you with that..."),
            HumanMessage(content="Oh that's great, thank you!"),  # positive
        ]
        # Positive message at end breaks the consecutive frustrated chain
        count = _count_consecutive_frustrated(messages)
        assert count == 0  # positive breaks the chain

    def test_sustained_frustration_triggers_escalation(self):
        """Two consecutive frustrated messages trigger escalation count."""
        from src.agent.agents._base import _count_consecutive_frustrated

        messages = [
            HumanMessage(content="This is ridiculous!"),  # frustrated
            AIMessage(content="I understand..."),
            HumanMessage(content="I'm so frustrated, nothing works!"),  # frustrated
            AIMessage(content="I'm sorry..."),
            HumanMessage(content="This is unacceptable, I'm fed up!"),  # frustrated
        ]
        count = _count_consecutive_frustrated(messages)
        assert count >= 2  # Should detect consecutive frustration

    def test_sarcasm_detected_as_frustrated(self):
        """Sarcastic comments are detected as frustrated, not positive."""
        assert detect_sentiment("Great, another closed restaurant") == "frustrated"
        assert detect_sentiment("Thanks for nothing") == "frustrated"
        assert detect_sentiment("Oh wonderful, more waiting") == "frustrated"

    def test_casino_positive_overrides(self):
        """Casino domain positive phrases are correctly classified."""
        assert detect_sentiment("I'm killing it at the tables!") == "positive"
        assert detect_sentiment("Hit the jackpot tonight!") == "positive"
        assert detect_sentiment("I'm on a hot streak!") == "positive"

    def test_neutral_stays_neutral(self):
        """Neutral questions stay neutral."""
        assert detect_sentiment("What time does the pool close?") == "neutral"
        assert detect_sentiment("How many restaurants do you have?") == "neutral"


# ---------------------------------------------------------------------------
# 4. Routing Flow Tests
# ---------------------------------------------------------------------------


class TestRoutingFlow:
    """Tests for greeting -> property_qa -> off_topic flow transitions."""

    def test_greeting_routes_to_greeting_node(self):
        """greeting query_type routes to greeting node."""
        state = _state(query_type="greeting", router_confidence=0.99)
        assert route_from_router(state) == "greeting"

    def test_property_qa_routes_to_retrieve(self):
        """property_qa routes to retrieve node."""
        state = _state(query_type="property_qa", router_confidence=0.9)
        assert route_from_router(state) == "retrieve"

    def test_off_topic_routes_to_off_topic(self):
        """off_topic routes to off_topic node."""
        state = _state(query_type="off_topic", router_confidence=0.8)
        assert route_from_router(state) == "off_topic"

    def test_gambling_advice_routes_to_off_topic(self):
        """gambling_advice routes to off_topic node."""
        state = _state(query_type="gambling_advice", router_confidence=0.95)
        assert route_from_router(state) == "off_topic"

    def test_action_request_routes_to_off_topic(self):
        """action_request routes to off_topic node."""
        state = _state(query_type="action_request", router_confidence=0.9)
        assert route_from_router(state) == "off_topic"

    def test_ambiguous_routes_to_retrieve(self):
        """ambiguous queries route to retrieve (not off_topic)."""
        state = _state(query_type="ambiguous", router_confidence=0.5)
        assert route_from_router(state) == "retrieve"

    def test_low_confidence_routes_to_off_topic(self):
        """Very low confidence routes to off_topic regardless of query_type."""
        state = _state(query_type="property_qa", router_confidence=0.1)
        assert route_from_router(state) == "off_topic"

    def test_hours_schedule_routes_to_retrieve(self):
        """hours_schedule routes to retrieve node."""
        state = _state(query_type="hours_schedule", router_confidence=0.85)
        assert route_from_router(state) == "retrieve"

    @pytest.mark.asyncio()
    @patch("src.agent.nodes._get_llm", new_callable=AsyncMock)
    async def test_greeting_then_property_qa_sequence(self, mock_get_llm):
        """Verify sequential routing: greeting first, then property_qa."""
        from src.agent.nodes import router_node

        # Turn 1: greeting
        mock_get_llm.return_value = _mock_router_result("greeting")
        state1 = _state(messages=[HumanMessage(content="Hello!")])
        r1 = await router_node(state1)
        assert r1["query_type"] == "greeting"
        assert route_from_router({**state1, **r1}) == "greeting"

        # Turn 2: property_qa
        mock_get_llm.return_value = _mock_router_result("property_qa")
        state2 = _state(
            messages=[
                HumanMessage(content="Hello!"),
                AIMessage(content="Welcome to Mohegan Sun!"),
                HumanMessage(content="What restaurants do you have?"),
            ]
        )
        r2 = await router_node(state2)
        assert r2["query_type"] == "property_qa"
        assert route_from_router({**state2, **r2}) == "retrieve"


# ---------------------------------------------------------------------------
# 5. Extraction Edge Cases
# ---------------------------------------------------------------------------


class TestExtractionEdgeCases:
    """Tests for field extraction edge cases across conversation."""

    def test_vegetarian_not_extracted_as_name(self):
        """'I'm vegetarian' should NOT extract 'Vegetarian' as a name."""
        fields = extract_fields("I'm vegetarian, what dining options do you have?")
        assert fields.get("name") is None
        assert "vegetarian" in fields.get("preferences", "")

    def test_here_not_extracted_as_name(self):
        """'I'm here for a birthday' should NOT extract 'Here' as a name."""
        fields = extract_fields("I'm here for a birthday")
        assert fields.get("name") is None

    def test_sure_not_extracted_as_name(self):
        """'Sure thing' should NOT extract 'Sure' as a name."""
        fields = extract_fields("Sure thing, I'd like to know about dinner")
        assert fields.get("name") is None

    def test_valid_name_extraction(self):
        """Valid name patterns are correctly extracted."""
        assert extract_fields("My name is Sarah")["name"] == "Sarah"
        assert extract_fields("I'm Michael")["name"] == "Michael"
        assert extract_fields("Call me David")["name"] == "David"

    def test_occasion_extraction(self):
        """Occasion fields are extracted correctly."""
        assert extract_fields("We're celebrating our anniversary")["occasion"] == "anniversary"
        assert extract_fields("It's my birthday!")["occasion"] == "birthday"

    def test_visit_date_extraction(self):
        """Visit date patterns are extracted."""
        fields = extract_fields("We're visiting next Saturday")
        assert fields.get("visit_date") == "Saturday"

    def test_empty_input_returns_empty(self):
        """Empty or None input returns empty dict."""
        assert extract_fields("") == {}
        assert extract_fields(None) == {}
