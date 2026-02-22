"""Full-graph E2E tests: build_graph() -> chat() with mocked LLMs.

These tests exercise the FULL 11-node StateGraph pipeline. Unlike unit tests
that test individual nodes in isolation, these compile the real graph via
build_graph() and send messages through chat(), verifying that node wiring,
conditional edges, and state transitions work end-to-end.

All LLM calls are mocked at the lowest level (_get_llm, _get_validator_llm,
_get_whisper_llm) to avoid API key requirements. RAG retrieval tools are
mocked to return deterministic test data. Circuit breakers are mocked to
always allow requests.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, chat
from src.agent.guardrails import InjectionClassification
from src.agent.state import (
    DispatchOutput,
    RouterOutput,
    ValidationResult,
)
from src.agent.whisper_planner import WhisperPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_permissive_cb() -> AsyncMock:
    """Create a mock circuit breaker that always allows requests."""
    cb = AsyncMock()
    cb.allow_request = AsyncMock(return_value=True)
    cb.record_success = AsyncMock()
    cb.record_failure = AsyncMock()
    cb.record_cancellation = AsyncMock()
    cb.is_open = False
    cb.state = "closed"
    return cb


def _make_mock_llm(
    *,
    router_output: RouterOutput | None = None,
    dispatch_output: DispatchOutput | None = None,
    validation_output: ValidationResult | None = None,
    agent_response: str = "This is a test response from the specialist agent.",
) -> MagicMock:
    """Create a mock LLM that handles with_structured_output for multiple schemas.

    The mock handles:
    - RouterOutput (router node)
    - DispatchOutput (specialist dispatch in _dispatch_to_specialist)
    - ValidationResult (validate node)
    - Plain ainvoke (specialist agent generate call)
    """
    mock_llm = MagicMock()

    # Track what structured output type is requested
    def _with_structured_output(schema):
        chain = AsyncMock()
        if schema is RouterOutput:
            output = router_output or RouterOutput(
                query_type="property_qa", confidence=0.95,
            )
            chain.ainvoke = AsyncMock(return_value=output)
        elif schema is DispatchOutput:
            output = dispatch_output or DispatchOutput(
                specialist="host",
                confidence=0.9,
                reasoning="General property question",
            )
            chain.ainvoke = AsyncMock(return_value=output)
        elif schema is ValidationResult:
            output = validation_output or ValidationResult(
                status="PASS", reason="Response meets all criteria",
            )
            chain.ainvoke = AsyncMock(return_value=output)
        elif schema is InjectionClassification:
            # Semantic injection classifier: return safe (no injection)
            chain.ainvoke = AsyncMock(return_value=InjectionClassification(
                is_injection=False,
                confidence=0.05,
                reason="Legitimate property question",
            ))
        else:
            # WhisperPlan or other structured output
            chain.ainvoke = AsyncMock(return_value=WhisperPlan(
                next_topic="dining",
                extraction_targets=["dietary_restrictions"],
                offer_readiness=0.3,
                conversation_note="Guest is asking about dining options",
                proactive_suggestion=None,
                suggestion_confidence=0.0,
            ))
        return chain

    mock_llm.with_structured_output = MagicMock(side_effect=_with_structured_output)

    # Plain ainvoke for specialist agent generate call
    mock_response = MagicMock()
    mock_response.content = agent_response
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    return mock_llm


def _make_test_retrieved_context(category: str = "restaurants") -> list[dict]:
    """Return test retrieved context chunks."""
    return [
        {
            "content": "Todd English's Tuscany offers authentic Italian cuisine with handmade pasta.",
            "metadata": {"category": category, "source": "property_data"},
            "score": 0.92,
        },
        {
            "content": "Bobby Flay's Bar Americain features American grill favorites.",
            "metadata": {"category": category, "source": "property_data"},
            "score": 0.85,
        },
    ]


# ---------------------------------------------------------------------------
# E2E Test Class
# ---------------------------------------------------------------------------


class TestFullGraphE2E:
    """Full pipeline E2E tests through build_graph() -> chat().

    Each test compiles the real graph and sends a message through the full
    pipeline, verifying end-to-end behavior with mocked LLMs.
    """

    @pytest.mark.asyncio
    async def test_dining_query_full_pipeline(self):
        """Dining question traverses all nodes: compliance -> router -> retrieve ->
        whisper -> generate -> validate -> persona -> respond."""
        mock_llm = _make_mock_llm(
            router_output=RouterOutput(query_type="property_qa", confidence=0.95),
            dispatch_output=DispatchOutput(
                specialist="dining",
                confidence=0.95,
                reasoning="Guest is asking about restaurants",
            ),
            validation_output=ValidationResult(status="PASS", reason="Looks good"),
            agent_response="Todd English's Tuscany is an excellent choice for Italian dining at Mohegan Sun.",
        )
        mock_cb = _make_permissive_cb()

        with (
            patch("src.agent.nodes._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.nodes._get_validator_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.circuit_breaker._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb),
            patch("src.agent.nodes.search_knowledge_base", return_value=_make_test_retrieved_context("restaurants")),
            patch("src.agent.nodes.search_hours", return_value=[]),
            patch("src.observability.langfuse_client.get_langfuse_handler", return_value=None),
        ):
            graph = build_graph()
            result = await chat(graph, "What restaurants do you have?")

        # Verify we got a response
        assert "response" in result
        assert len(result["response"]) > 0
        assert "thread_id" in result

        # Verify the response content came from the specialist agent
        response_text = result["response"]
        assert "Tuscany" in response_text or "Italian" in response_text or "dining" in response_text.lower()

    @pytest.mark.asyncio
    async def test_greeting_full_pipeline(self):
        """'Hello!' is classified as greeting by router and returns a welcome message
        with property name, without entering the RAG pipeline."""
        mock_llm = _make_mock_llm(
            router_output=RouterOutput(query_type="greeting", confidence=0.99),
        )
        mock_cb = _make_permissive_cb()

        with (
            patch("src.agent.nodes._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.nodes._get_validator_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.circuit_breaker._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb),
            patch("src.agent.nodes.search_knowledge_base", return_value=[]) as mock_search,
            patch("src.agent.nodes.search_hours", return_value=[]),
            patch("src.observability.langfuse_client.get_langfuse_handler", return_value=None),
        ):
            graph = build_graph()
            result = await chat(graph, "Hello!")

        response_text = result["response"]

        # Greeting should mention property name (from settings: "Mohegan Sun")
        assert "Mohegan Sun" in response_text

        # Greeting node produces a welcome message with "Seven" persona
        assert "Seven" in response_text

        # RAG retrieval should NOT have been called (greeting bypasses retrieve)
        mock_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_off_topic_full_pipeline(self):
        """Non-property question gets classified as off_topic and returns a redirect."""
        mock_llm = _make_mock_llm(
            router_output=RouterOutput(query_type="off_topic", confidence=0.90),
        )
        mock_cb = _make_permissive_cb()

        with (
            patch("src.agent.nodes._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.nodes._get_validator_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.circuit_breaker._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb),
            patch("src.agent.nodes.search_knowledge_base", return_value=[]) as mock_search,
            patch("src.agent.nodes.search_hours", return_value=[]),
            patch("src.observability.langfuse_client.get_langfuse_handler", return_value=None),
        ):
            graph = build_graph()
            result = await chat(graph, "What is the capital of France?")

        response_text = result["response"]

        # Off-topic should redirect to property topics
        assert "Mohegan Sun" in response_text or "concierge" in response_text.lower()

        # RAG retrieval should NOT have been called (off_topic bypasses retrieve)
        mock_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_responsible_gaming_trigger(self):
        """'I can't stop gambling' triggers the responsible gaming guardrail in
        compliance_gate, bypassing the LLM router entirely, and returns helpline info."""
        mock_llm = _make_mock_llm()
        mock_cb = _make_permissive_cb()

        with (
            patch("src.agent.nodes._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.nodes._get_validator_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.circuit_breaker._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb),
            patch("src.agent.nodes.search_knowledge_base", return_value=[]) as mock_search,
            patch("src.agent.nodes.search_hours", return_value=[]),
            patch("src.observability.langfuse_client.get_langfuse_handler", return_value=None),
        ):
            graph = build_graph()
            result = await chat(graph, "I can't stop gambling")

        response_text = result["response"]

        # Response should contain helpline information
        assert "1-800-522-4700" in response_text or "helpline" in response_text.lower() or "problem gambling" in response_text.lower()

        # RAG retrieval should NOT be called (guardrail bypasses router+retrieve)
        mock_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """3 sequential messages with the same thread_id maintain conversation context."""
        mock_llm = _make_mock_llm(
            router_output=RouterOutput(query_type="property_qa", confidence=0.95),
            dispatch_output=DispatchOutput(
                specialist="dining",
                confidence=0.9,
                reasoning="Restaurant query",
            ),
            validation_output=ValidationResult(status="PASS", reason="OK"),
            agent_response="We have many great restaurants at Mohegan Sun including Tuscany.",
        )
        mock_cb = _make_permissive_cb()

        thread_id = "test-multi-turn-e2e"

        with (
            patch("src.agent.nodes._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.nodes._get_validator_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.circuit_breaker._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb),
            patch("src.agent.nodes.search_knowledge_base", return_value=_make_test_retrieved_context("restaurants")),
            patch("src.agent.nodes.search_hours", return_value=[]),
            patch("src.observability.langfuse_client.get_langfuse_handler", return_value=None),
        ):
            graph = build_graph()

            # Turn 1: Ask about restaurants
            result1 = await chat(graph, "What restaurants do you have?", thread_id=thread_id)
            assert result1["thread_id"] == thread_id
            assert len(result1["response"]) > 0

            # Turn 2: Follow-up question (same thread)
            result2 = await chat(graph, "Do any of them serve Italian?", thread_id=thread_id)
            assert result2["thread_id"] == thread_id
            assert len(result2["response"]) > 0

            # Turn 3: Another follow-up (same thread)
            result3 = await chat(graph, "What are the hours?", thread_id=thread_id)
            assert result3["thread_id"] == thread_id
            assert len(result3["response"]) > 0

        # Verify all 3 turns returned responses and maintained the same thread
        assert result1["thread_id"] == result2["thread_id"] == result3["thread_id"]

        # The MemorySaver checkpointer maintains state across turns for the same
        # thread_id. Each subsequent call accumulates messages in the state.
        # This verifies the wiring: build_graph() creates a checkpointer,
        # chat() uses thread_id in config, and ainvoke persists state.

    @pytest.mark.asyncio
    async def test_retry_loop_full_pipeline(self):
        """Validator returns RETRY on first attempt, then PASS on second.

        Verifies the generate -> validate -> generate -> validate -> respond
        path through the real compiled graph (the RETRY conditional edge).
        """
        # Track validation calls to return RETRY first, then PASS
        validate_call_count = {"n": 0}

        def _with_structured_output(schema):
            chain = AsyncMock()
            if schema is RouterOutput:
                chain.ainvoke = AsyncMock(return_value=RouterOutput(
                    query_type="property_qa", confidence=0.95,
                ))
            elif schema is DispatchOutput:
                chain.ainvoke = AsyncMock(return_value=DispatchOutput(
                    specialist="dining",
                    confidence=0.9,
                    reasoning="Restaurant query",
                ))
            elif schema is ValidationResult:
                async def _validate_side_effect(prompt):
                    validate_call_count["n"] += 1
                    if validate_call_count["n"] == 1:
                        return ValidationResult(
                            status="RETRY",
                            reason="Missing source attribution in response",
                        )
                    return ValidationResult(
                        status="PASS",
                        reason="Source attribution added, response is correct",
                    )
                chain.ainvoke = AsyncMock(side_effect=_validate_side_effect)
            elif schema is InjectionClassification:
                chain.ainvoke = AsyncMock(return_value=InjectionClassification(
                    is_injection=False, confidence=0.05,
                    reason="Legitimate property question",
                ))
            else:
                chain.ainvoke = AsyncMock(return_value=WhisperPlan(
                    next_topic="dining",
                    extraction_targets=["dietary_restrictions"],
                    offer_readiness=0.3,
                    conversation_note="Guest asking about dining",
                    proactive_suggestion=None,
                    suggestion_confidence=0.0,
                ))
            return chain

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(side_effect=_with_structured_output)
        mock_response = MagicMock()
        mock_response.content = "Todd English's Tuscany offers excellent Italian cuisine at Mohegan Sun."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_cb = _make_permissive_cb()

        with (
            patch("src.agent.nodes._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.nodes._get_validator_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.circuit_breaker._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb),
            patch("src.agent.nodes.search_knowledge_base", return_value=_make_test_retrieved_context("restaurants")),
            patch("src.agent.nodes.search_hours", return_value=[]),
            patch("src.observability.langfuse_client.get_langfuse_handler", return_value=None),
        ):
            graph = build_graph()
            result = await chat(graph, "What Italian restaurants do you have?")

        assert "response" in result
        assert len(result["response"]) > 0
        assert "thread_id" in result
        # Validator was called twice (RETRY then PASS)
        assert validate_call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_fail_to_fallback_full_pipeline(self):
        """Validator always returns non-PASS, triggering RETRY then FAIL.

        Verifies the generate -> validate (RETRY) -> generate -> validate (FAIL)
        -> fallback path through the real compiled graph.
        The fallback_node returns a message with the property phone number.
        """
        def _with_structured_output(schema):
            chain = AsyncMock()
            if schema is RouterOutput:
                chain.ainvoke = AsyncMock(return_value=RouterOutput(
                    query_type="property_qa", confidence=0.95,
                ))
            elif schema is DispatchOutput:
                chain.ainvoke = AsyncMock(return_value=DispatchOutput(
                    specialist="dining",
                    confidence=0.9,
                    reasoning="Restaurant query",
                ))
            elif schema is ValidationResult:
                # Always return RETRY — validate_node logic:
                # retry_count=0 -> RETRY, retry_count=1 -> FAIL
                chain.ainvoke = AsyncMock(return_value=ValidationResult(
                    status="RETRY",
                    reason="Response contains factual errors about menu items",
                ))
            elif schema is InjectionClassification:
                chain.ainvoke = AsyncMock(return_value=InjectionClassification(
                    is_injection=False, confidence=0.05,
                    reason="Legitimate property question",
                ))
            else:
                chain.ainvoke = AsyncMock(return_value=WhisperPlan(
                    next_topic="dining",
                    extraction_targets=["dietary_restrictions"],
                    offer_readiness=0.3,
                    conversation_note="Guest asking about dining",
                    proactive_suggestion=None,
                    suggestion_confidence=0.0,
                ))
            return chain

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(side_effect=_with_structured_output)
        mock_response = MagicMock()
        mock_response.content = "Some generated response that will fail validation."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        mock_cb = _make_permissive_cb()

        with (
            patch("src.agent.nodes._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.nodes._get_validator_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.circuit_breaker._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb),
            patch("src.agent.nodes.search_knowledge_base", return_value=_make_test_retrieved_context("restaurants")),
            patch("src.agent.nodes.search_hours", return_value=[]),
            patch("src.observability.langfuse_client.get_langfuse_handler", return_value=None),
        ):
            graph = build_graph()
            result = await chat(graph, "Tell me about your Italian restaurants")

        response_text = result["response"]
        assert len(response_text) > 0
        # Fallback node includes the property phone number (1-888-226-7711)
        assert "1-888-226-7711" in response_text
        assert "thread_id" in result

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_full_pipeline(self):
        """Circuit breaker is open, so the specialist agent returns CB fallback.

        Verifies that when CB.allow_request() returns False, the specialist
        agent (_base.py:135) returns a fallback message with skip_validation=True,
        and the graph routes through validate (auto-PASS) -> persona -> respond.
        """
        mock_llm = _make_mock_llm(
            router_output=RouterOutput(query_type="property_qa", confidence=0.95),
            dispatch_output=DispatchOutput(
                specialist="dining",
                confidence=0.9,
                reasoning="Restaurant query",
            ),
            validation_output=ValidationResult(status="PASS", reason="OK"),
            agent_response="This should not appear because CB is open.",
        )

        # Create a CB that blocks all requests (open state)
        mock_cb_open = AsyncMock()
        mock_cb_open.allow_request = AsyncMock(return_value=False)
        mock_cb_open.record_success = AsyncMock()
        mock_cb_open.record_failure = AsyncMock()
        mock_cb_open.record_cancellation = AsyncMock()
        mock_cb_open.is_open = True
        mock_cb_open.state = "open"

        # Patch _get_circuit_breaker at all import sites:
        # - graph.py imports it for dispatch CB check
        # - each specialist agent module imports it as get_cb_fn
        # Python's "from X import Y" creates a local binding, so we must
        # patch where the reference is USED, not just where it's DEFINED.
        with (
            patch("src.agent.nodes._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.nodes._get_validator_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.circuit_breaker._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb_open),
            patch("src.agent.graph._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb_open),
            patch("src.agent.agents.dining_agent._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb_open),
            patch("src.agent.agents.host_agent._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb_open),
            patch("src.agent.agents.entertainment_agent._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb_open),
            patch("src.agent.agents.comp_agent._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb_open),
            patch("src.agent.agents.hotel_agent._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb_open),
            patch("src.agent.nodes.search_knowledge_base", return_value=_make_test_retrieved_context("restaurants")),
            patch("src.agent.nodes.search_hours", return_value=[]),
            patch("src.observability.langfuse_client.get_langfuse_handler", return_value=None),
        ):
            graph = build_graph()
            result = await chat(graph, "What restaurants do you have?")

        response_text = result["response"]
        assert len(response_text) > 0
        # CB fallback message from _base.py contains "technical difficulties"
        assert "technical difficulties" in response_text.lower() or "contact" in response_text.lower()
        assert "thread_id" in result

    @pytest.mark.asyncio
    async def test_whisper_planner_failure_silent(self):
        """Whisper planner LLM raises an exception but the graph continues.

        Verifies that whisper_planner_node is fail-silent: when its LLM call
        fails, it returns {"whisper_plan": None} and the graph proceeds to
        the generate node without crashing.
        """
        mock_llm = _make_mock_llm(
            router_output=RouterOutput(query_type="property_qa", confidence=0.95),
            dispatch_output=DispatchOutput(
                specialist="dining",
                confidence=0.9,
                reasoning="Restaurant query",
            ),
            validation_output=ValidationResult(status="PASS", reason="OK"),
            agent_response="Todd English's Tuscany offers Italian dining at Mohegan Sun.",
        )
        mock_cb = _make_permissive_cb()

        # Create a broken whisper LLM that raises on any structured output call
        mock_whisper_llm = MagicMock()
        whisper_chain = AsyncMock()
        whisper_chain.ainvoke = AsyncMock(side_effect=Exception("LLM timeout — whisper planner unavailable"))
        mock_whisper_llm.with_structured_output = MagicMock(return_value=whisper_chain)

        with (
            patch("src.agent.nodes._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.nodes._get_validator_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.whisper_planner._get_whisper_llm", new_callable=AsyncMock, return_value=mock_whisper_llm),
            patch("src.agent.circuit_breaker._get_circuit_breaker", new_callable=AsyncMock, return_value=mock_cb),
            patch("src.agent.nodes.search_knowledge_base", return_value=_make_test_retrieved_context("restaurants")),
            patch("src.agent.nodes.search_hours", return_value=[]),
            patch("src.observability.langfuse_client.get_langfuse_handler", return_value=None),
        ):
            graph = build_graph()
            result = await chat(graph, "What Italian restaurants do you have?")

        # Graph should still produce a valid response despite whisper failure
        assert "response" in result
        assert len(result["response"]) > 0
        assert "thread_id" in result
        # The specialist agent still generates a response
        response_text = result["response"]
        assert "Tuscany" in response_text or "Italian" in response_text or "dining" in response_text.lower()
