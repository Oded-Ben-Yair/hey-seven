"""Phase 2 integration tests: whisper planner wiring, host agent injection,
comp agent profile gate, SMS webhook endpoint, graph endpoint v2.

Validates that Phase 2 modules (guest profile, whisper planner, SMS) are
correctly wired into the existing graph, agents, and API.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> dict:
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
        "current_time": "Monday 3 PM",
        "sources_used": [],
        "extracted_fields": {},
        "whisper_plan": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestWhisperGraphWiring
# ---------------------------------------------------------------------------


class TestWhisperGraphWiring:
    """Verify whisper_planner is correctly wired in the graph topology."""

    def test_whisper_node_in_graph(self):
        """whisper_planner is one of the 11 nodes."""
        from src.agent.graph import build_graph

        graph = build_graph()
        all_nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
        assert "whisper_planner" in all_nodes
        assert len(all_nodes) == 11

    def test_retrieve_to_whisper_edge(self):
        """retrieve has an edge to whisper_planner."""
        from src.agent.graph import build_graph

        graph = build_graph()
        drawable = graph.get_graph()
        retrieve_targets = {e.target for e in drawable.edges if e.source == "retrieve"}
        assert "whisper_planner" in retrieve_targets

    def test_whisper_to_generate_edge(self):
        """whisper_planner has an edge to generate."""
        from src.agent.graph import build_graph

        graph = build_graph()
        drawable = graph.get_graph()
        whisper_targets = {e.target for e in drawable.edges if e.source == "whisper_planner"}
        assert "generate" in whisper_targets

    def test_no_direct_retrieve_to_generate(self):
        """No direct edge from retrieve to generate (whisper is in between)."""
        from src.agent.graph import build_graph

        graph = build_graph()
        drawable = graph.get_graph()
        retrieve_targets = {e.target for e in drawable.edges if e.source == "retrieve"}
        assert "generate" not in retrieve_targets

    def test_retry_bypasses_whisper(self):
        """validate RETRY goes directly to generate, not through whisper."""
        from src.agent.graph import build_graph

        graph = build_graph()
        drawable = graph.get_graph()
        validate_targets = {e.target for e in drawable.edges if e.source == "validate"}
        assert "generate" in validate_targets  # RETRY path
        assert "whisper_planner" not in validate_targets  # No re-whisper on retry

    def test_whisper_constant_exported(self):
        """NODE_WHISPER constant is importable and correct."""
        from src.agent.graph import NODE_WHISPER

        assert NODE_WHISPER == "whisper_planner"

    def test_whisper_in_known_nodes(self):
        """whisper_planner is in _KNOWN_NODES."""
        from src.agent.graph import _KNOWN_NODES

        assert "whisper_planner" in _KNOWN_NODES

    def test_whisper_in_non_stream_nodes(self):
        """whisper_planner is in _NON_STREAM_NODES (no token streaming)."""
        from src.agent.graph import _NON_STREAM_NODES

        assert "whisper_planner" in _NON_STREAM_NODES


# ---------------------------------------------------------------------------
# TestWhisperNodeMetadata
# ---------------------------------------------------------------------------


class TestWhisperNodeMetadata:
    """Verify _extract_node_metadata handles whisper_planner."""

    def test_metadata_with_plan(self):
        """whisper_planner with a plan returns has_plan=True."""
        from src.agent.graph import _extract_node_metadata

        output = {"whisper_plan": {"next_topic": "dining", "offer_readiness": 0.5}}
        meta = _extract_node_metadata("whisper_planner", output)
        assert meta == {"has_plan": True}

    def test_metadata_without_plan(self):
        """whisper_planner with None plan returns has_plan=False."""
        from src.agent.graph import _extract_node_metadata

        output = {"whisper_plan": None}
        meta = _extract_node_metadata("whisper_planner", output)
        assert meta == {"has_plan": False}

    def test_metadata_empty_output(self):
        """whisper_planner with empty dict returns has_plan=False."""
        from src.agent.graph import _extract_node_metadata

        output = {}
        meta = _extract_node_metadata("whisper_planner", output)
        assert meta == {"has_plan": False}


# ---------------------------------------------------------------------------
# TestHostAgentWhisperInjection
# ---------------------------------------------------------------------------


class TestHostAgentWhisperInjection:
    """Verify host_agent injects whisper plan into system prompt."""

    @pytest.mark.asyncio
    async def test_whisper_plan_injected_when_present(self):
        """Whisper plan is appended to system prompt when state has a plan."""
        from src.agent.agents.host_agent import host_agent

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))

        state = _base_state(
            messages=[HumanMessage(content="Tell me about dining")],
            retrieved_context=[
                {"content": "Test restaurant data", "metadata": {"category": "restaurants"}, "score": 0.9},
            ],
            whisper_plan={
                "next_topic": "dining",
                "extraction_targets": ["cuisine_preferences"],
                "offer_readiness": 0.3,
                "conversation_note": "Guest seems interested in Italian food",
            },
        )

        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()

        with (
            patch("src.agent.agents.host_agent._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.agents.host_agent._get_circuit_breaker", return_value=mock_cb),
        ):
            await host_agent(state)

        # Verify the LLM was called with a system prompt containing whisper guidance
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msg = call_args[0]  # First message is SystemMessage
        assert "Whisper Track Guidance" in system_msg.content
        assert "dining" in system_msg.content
        assert "cuisine_preferences" in system_msg.content

    @pytest.mark.asyncio
    async def test_no_whisper_plan_no_injection(self):
        """No whisper plan (None) means no guidance injected."""
        from src.agent.agents.host_agent import host_agent

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))

        state = _base_state(
            messages=[HumanMessage(content="Tell me about dining")],
            retrieved_context=[
                {"content": "Test restaurant data", "metadata": {"category": "restaurants"}, "score": 0.9},
            ],
            whisper_plan=None,
        )

        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()

        with (
            patch("src.agent.agents.host_agent._get_llm", new_callable=AsyncMock, return_value=mock_llm),
            patch("src.agent.agents.host_agent._get_circuit_breaker", return_value=mock_cb),
        ):
            await host_agent(state)

        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msg = call_args[0]
        assert "Whisper Track Guidance" not in system_msg.content

    @pytest.mark.asyncio
    async def test_empty_context_skips_whisper(self):
        """Empty retrieved context returns early before whisper injection."""
        from src.agent.agents.host_agent import host_agent

        state = _base_state(
            messages=[HumanMessage(content="Tell me about something")],
            retrieved_context=[],
            whisper_plan={"next_topic": "dining", "extraction_targets": [], "offer_readiness": 0.5, "conversation_note": "test"},
        )

        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.allow_request = AsyncMock(return_value=True)

        with patch("src.agent.agents.host_agent._get_circuit_breaker", return_value=mock_cb):
            result = await host_agent(state)

        # Empty context triggers skip_validation early return
        assert result["skip_validation"] is True
        assert "don't have specific information" in result["messages"][0].content


# ---------------------------------------------------------------------------
# TestCompAgentProfileGate
# ---------------------------------------------------------------------------


class TestCompAgentProfileGate:
    """Verify comp_agent's 60% profile completeness gate."""

    @pytest.mark.asyncio
    async def test_below_60_returns_deflection(self):
        """Profile completeness < 60% returns a friendly deflection."""
        from src.agent.agents.comp_agent import comp_agent

        state = _base_state(
            messages=[HumanMessage(content="What promotions do you have?")],
            retrieved_context=[
                {"content": "Loyalty program info", "metadata": {"category": "promotions"}, "score": 0.9},
            ],
            extracted_fields={},  # Empty = 0% completeness
        )

        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.allow_request = AsyncMock(return_value=True)

        with patch("src.agent.agents.comp_agent._get_circuit_breaker", return_value=mock_cb):
            result = await comp_agent(state)

        assert result["skip_validation"] is True
        content = result["messages"][0].content
        assert "rewards and promotions" in content
        assert "tell me a bit more" in content

    @pytest.mark.asyncio
    async def test_partial_profile_below_60(self):
        """Partial profile (some fields) still below 60% returns deflection."""
        from src.agent.agents.comp_agent import comp_agent

        # Only 2/8 flat fields filled = 25% < 60% threshold
        state = _base_state(
            messages=[HumanMessage(content="Any deals?")],
            retrieved_context=[
                {"content": "Promo data", "metadata": {"category": "promotions"}, "score": 0.9},
            ],
            extracted_fields={
                "name": "John",
                "visit_date": None,
                "party_size": None,
                "dining": None,
                "entertainment": "comedy show",
                "gaming": None,
                "occasions": None,
                "companions": None,
            },
        )

        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.allow_request = AsyncMock(return_value=True)

        with patch("src.agent.agents.comp_agent._get_circuit_breaker", return_value=mock_cb):
            result = await comp_agent(state)

        assert result["skip_validation"] is True
        assert "rewards and promotions" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_above_60_proceeds_normally(self):
        """Profile completeness >= 60% proceeds to LLM generation."""
        from src.agent.agents.comp_agent import comp_agent

        # Fill 5/8 flat fields = 62.5% > 60% threshold
        extracted_fields = {
            "name": "John Doe",
            "visit_date": "2026-03-01",
            "party_size": 4,
            "dining": "steakhouse",
            "entertainment": "comedy show",
            "gaming": None,
            "occasions": None,
            "companions": None,
        }

        state = _base_state(
            messages=[HumanMessage(content="What promotions do you have?")],
            retrieved_context=[
                {"content": "Loyalty program info", "metadata": {"category": "promotions"}, "score": 0.9},
            ],
            extracted_fields=extracted_fields,
        )

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Here are our promotions!"))
        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.allow_request = AsyncMock(return_value=True)
        mock_cb.record_success = AsyncMock()

        with (
            patch("src.agent.agents.comp_agent._get_circuit_breaker", return_value=mock_cb),
            patch("src.agent.agents.comp_agent._get_llm", new_callable=AsyncMock, return_value=mock_llm),
        ):
            result = await comp_agent(state)

        # Should NOT be the deflection message
        assert "rewards and promotions" not in result["messages"][0].content
        assert "skip_validation" not in result or result.get("skip_validation") is not True

    @pytest.mark.asyncio
    async def test_circuit_breaker_takes_priority(self):
        """Circuit breaker open returns fallback after passing profile gate."""
        from src.agent.agents.comp_agent import comp_agent

        # Fill enough fields to pass the profile completeness gate (>=60%),
        # so execution reaches execute_specialist where CB is checked.
        # core=10 + visit=4.5 + dining.dietary=1.0 = 15.5/25 = 62%
        extracted_fields = {
            "core_identity": {
                "name": {"value": "Test", "confidence": 0.9, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                "email": {"value": "test@example.com", "confidence": 0.8, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                "language": {"value": "en", "confidence": 0.95, "source": "contextual_extraction", "collected_at": "2026-01-01T00:00:00Z"},
                "full_name": {"value": "Test User", "confidence": 0.85, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                "date_of_birth": {"value": "1990-01-01", "confidence": 0.7, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
            },
            "visit_context": {
                "planned_visit_date": {"value": "2026-03-01", "confidence": 0.9, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                "party_size": {"value": 2, "confidence": 0.8, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                "occasion": {"value": "anniversary", "confidence": 0.8, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
            },
            "preferences": {
                "dining": {
                    "dietary_restrictions": {"value": "none", "confidence": 0.7, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                },
            },
        }
        state = _base_state(
            messages=[HumanMessage(content="What promotions?")],
            extracted_fields=extracted_fields,
        )

        mock_cb = MagicMock()
        mock_cb.is_open = True
        mock_cb.allow_request = AsyncMock(return_value=False)

        with patch("src.agent.agents.comp_agent._get_circuit_breaker", return_value=mock_cb):
            result = await comp_agent(state)

        assert result["skip_validation"] is True
        assert "technical difficulties" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_empty_context_returns_no_info_fallback(self):
        """Empty retrieved context returns no-info fallback after passing profile gate."""
        from src.agent.agents.comp_agent import comp_agent

        # Fill enough fields to pass profile gate (>=60%) but provide no context
        extracted_fields = {
            "core_identity": {
                "name": {"value": "John", "confidence": 0.9, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                "email": {"value": "john@test.com", "confidence": 0.8, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                "language": {"value": "en", "confidence": 0.95, "source": "contextual_extraction", "collected_at": "2026-01-01T00:00:00Z"},
                "full_name": {"value": "John Doe", "confidence": 0.85, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                "date_of_birth": {"value": "1985-06-15", "confidence": 0.7, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
            },
            "visit_context": {
                "planned_visit_date": {"value": "2026-03-01", "confidence": 0.9, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                "party_size": {"value": 4, "confidence": 0.85, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                "occasion": {"value": "birthday", "confidence": 0.8, "source": "contextual_extraction", "collected_at": "2026-01-01T00:00:00Z"},
            },
            "preferences": {
                "dining": {
                    "dietary_restrictions": {"value": "none", "confidence": 0.7, "source": "self_reported", "collected_at": "2026-01-01T00:00:00Z"},
                },
            },
        }

        state = _base_state(
            messages=[HumanMessage(content="What loyalty rewards?")],
            retrieved_context=[],
            extracted_fields=extracted_fields,
        )

        mock_cb = MagicMock()
        mock_cb.is_open = False
        mock_cb.allow_request = AsyncMock(return_value=True)

        with patch("src.agent.agents.comp_agent._get_circuit_breaker", return_value=mock_cb):
            result = await comp_agent(state)

        assert result["skip_validation"] is True
        assert "loyalty programs" in result["messages"][0].content


# ---------------------------------------------------------------------------
# TestSMSWebhookEndpoint
# ---------------------------------------------------------------------------


class TestSMSWebhookEndpoint:
    """Test POST /sms/webhook endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        app.state.agent = MagicMock()
        app.state.property_data = {"property": {"name": "Test Casino"}}
        app.state.ready = True
        return TestClient(app)

    def test_inbound_message_returns_200(self, client):
        """Valid inbound SMS webhook returns 200."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "from": {"phone_number": "+12035551234"},
                    "to": [{"phone_number": "+18605559876"}],
                    "text": "What restaurants do you have?",
                    "id": "msg-123-456",
                    "media": [],
                },
            }
        }
        response = client.post("/sms/webhook", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "received"
        assert data["from"] == "+12035551234"

    def test_stop_keyword_handled(self, client):
        """STOP keyword returns keyword_handled status."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "from": {"phone_number": "+12035551234"},
                    "to": [{"phone_number": "+18605559876"}],
                    "text": "STOP",
                    "id": "msg-stop-123",
                    "media": [],
                },
            }
        }
        response = client.post("/sms/webhook", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "keyword_handled"
        assert "unsubscribed" in data["response"].lower()

    def test_help_keyword_handled(self, client):
        """HELP keyword returns keyword_handled status."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "from": {"phone_number": "+12035551234"},
                    "to": [{"phone_number": "+18605559876"}],
                    "text": "HELP",
                    "id": "msg-help-123",
                    "media": [],
                },
            }
        }
        response = client.post("/sms/webhook", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "keyword_handled"
        assert "STOP" in data["response"]

    def test_start_keyword_handled(self, client):
        """START keyword returns keyword_handled status."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "from": {"phone_number": "+12035551234"},
                    "to": [{"phone_number": "+18605559876"}],
                    "text": "start",
                    "id": "msg-start-123",
                    "media": [],
                },
            }
        }
        response = client.post("/sms/webhook", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "keyword_handled"
        assert "resubscribed" in data["response"].lower()

    def test_non_message_event_ignored(self, client):
        """Non-message.received events are ignored."""
        payload = {
            "data": {
                "event_type": "message.finalized",
                "payload": {},
            }
        }
        response = client.post("/sms/webhook", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_invalid_json_returns_500(self, client):
        """Invalid webhook body returns 500."""
        response = client.post(
            "/sms/webhook",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        # Should catch the exception and return 500
        assert response.status_code in (422, 500)

    def test_spanish_stop_keyword(self, client):
        """Spanish STOP keyword (parar) is recognized."""
        payload = {
            "data": {
                "event_type": "message.received",
                "payload": {
                    "from": {"phone_number": "+12035551234"},
                    "to": [{"phone_number": "+18605559876"}],
                    "text": "parar",
                    "id": "msg-parar-123",
                    "media": [],
                },
            }
        }
        response = client.post("/sms/webhook", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "keyword_handled"


# ---------------------------------------------------------------------------
# TestGraphEndpointV2
# ---------------------------------------------------------------------------


class TestGraphEndpointV2:
    """Test GET /graph returns v2.1 topology."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        app.state.agent = MagicMock()
        app.state.property_data = {"property": {"name": "Test Casino"}}
        app.state.ready = True
        return TestClient(app)

    def test_graph_returns_11_nodes(self, client):
        """GET /graph returns exactly 11 nodes."""
        response = client.get("/graph")
        assert response.status_code == 200
        data = response.json()
        assert len(data["nodes"]) == 11

    def test_graph_includes_whisper_planner(self, client):
        """whisper_planner is in the node list."""
        response = client.get("/graph")
        data = response.json()
        assert "whisper_planner" in data["nodes"]

    def test_graph_includes_compliance_gate(self, client):
        """compliance_gate is in the node list."""
        response = client.get("/graph")
        data = response.json()
        assert "compliance_gate" in data["nodes"]

    def test_graph_includes_persona_envelope(self, client):
        """persona_envelope is in the node list."""
        response = client.get("/graph")
        data = response.json()
        assert "persona_envelope" in data["nodes"]

    def test_graph_start_to_compliance_gate(self, client):
        """First edge is __start__ -> compliance_gate."""
        response = client.get("/graph")
        data = response.json()
        first_edge = data["edges"][0]
        assert first_edge["from"] == "__start__"
        assert first_edge["to"] == "compliance_gate"

    def test_graph_retrieve_to_whisper_edge(self, client):
        """retrieve -> whisper_planner edge exists."""
        response = client.get("/graph")
        data = response.json()
        edges = data["edges"]
        matching = [e for e in edges if e["from"] == "retrieve" and e["to"] == "whisper_planner"]
        assert len(matching) == 1

    def test_graph_whisper_to_generate_edge(self, client):
        """whisper_planner -> generate edge exists."""
        response = client.get("/graph")
        data = response.json()
        edges = data["edges"]
        matching = [e for e in edges if e["from"] == "whisper_planner" and e["to"] == "generate"]
        assert len(matching) == 1

    def test_graph_persona_to_respond_edge(self, client):
        """persona_envelope -> respond edge exists."""
        response = client.get("/graph")
        data = response.json()
        edges = data["edges"]
        matching = [e for e in edges if e["from"] == "persona_envelope" and e["to"] == "respond"]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# TestWhisperPlannerNodeUnit
# ---------------------------------------------------------------------------


class TestWhisperPlannerNodeUnit:
    """Unit tests for the whisper_planner_node function."""

    @pytest.mark.asyncio
    async def test_returns_plan_on_success(self):
        """whisper_planner_node returns a plan dict on success."""
        from src.agent.whisper_planner import WhisperPlan, whisper_planner_node

        plan = WhisperPlan(
            next_topic="dining",
            extraction_targets=["dietary_restrictions"],
            offer_readiness=0.4,
            conversation_note="Guest mentioned anniversary",
        )

        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=plan)
        mock_llm.with_structured_output.return_value = mock_structured

        state = _base_state(
            messages=[HumanMessage(content="We're celebrating our anniversary")],
        )

        with patch("src.agent.whisper_planner._get_whisper_llm", return_value=mock_llm):
            result = await whisper_planner_node(state)

        assert result["whisper_plan"] is not None
        assert result["whisper_plan"]["next_topic"] == "dining"
        assert result["whisper_plan"]["offer_readiness"] == 0.4

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_failure(self):
        """whisper_planner_node returns None plan on LLM failure."""
        from src.agent.whisper_planner import whisper_planner_node

        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM API down"))
        mock_llm.with_structured_output.return_value = mock_structured

        state = _base_state(
            messages=[HumanMessage(content="Tell me about dining")],
        )

        with patch("src.agent.whisper_planner._get_whisper_llm", return_value=mock_llm):
            result = await whisper_planner_node(state)

        assert result["whisper_plan"] is None

    @pytest.mark.asyncio
    async def test_returns_none_on_parsing_error(self):
        """whisper_planner_node returns None plan on parsing error."""
        from src.agent.whisper_planner import whisper_planner_node

        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(side_effect=ValueError("Invalid output"))
        mock_llm.with_structured_output.return_value = mock_structured

        state = _base_state(
            messages=[HumanMessage(content="Hello")],
        )

        with patch("src.agent.whisper_planner._get_whisper_llm", return_value=mock_llm):
            result = await whisper_planner_node(state)

        assert result["whisper_plan"] is None


# ---------------------------------------------------------------------------
# TestFormatWhisperPlan
# ---------------------------------------------------------------------------


class TestFormatWhisperPlan:
    """Unit tests for format_whisper_plan."""

    def test_none_returns_empty_string(self):
        """None plan returns empty string."""
        from src.agent.whisper_planner import format_whisper_plan

        assert format_whisper_plan(None) == ""

    def test_valid_plan_returns_formatted_string(self):
        """Valid plan dict returns formatted guidance string."""
        from src.agent.whisper_planner import format_whisper_plan

        plan = {
            "next_topic": "dining",
            "extraction_targets": ["dietary_restrictions", "cuisine_preferences"],
            "offer_readiness": 0.65,
            "conversation_note": "Guest seems interested in Italian",
        }
        result = format_whisper_plan(plan)
        assert "Whisper Track Guidance" in result
        assert "dining" in result
        assert "dietary_restrictions" in result
        assert "65%" in result
        assert "Italian" in result

    def test_empty_targets_shows_none(self):
        """Empty extraction targets shows (none)."""
        from src.agent.whisper_planner import format_whisper_plan

        plan = {
            "next_topic": "none",
            "extraction_targets": [],
            "offer_readiness": 0.0,
            "conversation_note": "",
        }
        result = format_whisper_plan(plan)
        assert "(none)" in result

    def test_plan_includes_internal_label(self):
        """Plan output includes 'never reveal to guest' warning."""
        from src.agent.whisper_planner import format_whisper_plan

        plan = {
            "next_topic": "gaming",
            "extraction_targets": ["level"],
            "offer_readiness": 0.5,
            "conversation_note": "Ask about games",
        }
        result = format_whisper_plan(plan)
        assert "never reveal to guest" in result


# ---------------------------------------------------------------------------
# TestProfileCompletenessCalculation
# ---------------------------------------------------------------------------


class TestProfileCompletenessCalculation:
    """Verify calculate_completeness works correctly for the comp gate."""

    def test_empty_profile_is_zero(self):
        """Empty profile returns 0.0."""
        from src.data.models import calculate_completeness

        assert calculate_completeness({}) == 0.0
        assert calculate_completeness(None) == 0.0

    def test_full_core_identity(self):
        """Filling all core identity fields gives partial completeness."""
        from src.data.models import calculate_completeness

        profile = {
            "core_identity": {
                "name": {"value": "John", "confidence": 0.9, "source": "self_reported", "collected_at": "2026-01-01"},
                "email": {"value": "john@test.com", "confidence": 0.8, "source": "self_reported", "collected_at": "2026-01-01"},
                "language": {"value": "en", "confidence": 0.95, "source": "contextual_extraction", "collected_at": "2026-01-01"},
                "full_name": {"value": "John Doe", "confidence": 0.85, "source": "self_reported", "collected_at": "2026-01-01"},
                "date_of_birth": {"value": "1985-06-15", "confidence": 0.7, "source": "self_reported", "collected_at": "2026-01-01"},
            },
        }
        completeness = calculate_completeness(profile)
        assert completeness > 0.0
        # 5 core fields * 2.0 weight = 10.0 out of total weight
        assert completeness > 0.3  # Should be significant

    def test_sixty_percent_threshold(self):
        """Profile with enough fields reaches 60% threshold."""
        from src.data.models import calculate_completeness

        # Fill all core (5*2=10) + all visit (3*1.5=4.5) = 14.5
        # total_weight = 5*2 + 3*1.5 + 10*1.0 + 0.5 = 10 + 4.5 + 10 + 0.5 = 25.0
        # 14.5/25 = 0.58 (just under 60%)
        profile = {
            "core_identity": {
                "name": {"value": "John", "confidence": 0.9, "source": "self_reported", "collected_at": "2026-01-01"},
                "email": {"value": "j@t.com", "confidence": 0.8, "source": "self_reported", "collected_at": "2026-01-01"},
                "language": {"value": "en", "confidence": 0.9, "source": "contextual_extraction", "collected_at": "2026-01-01"},
                "full_name": {"value": "John D", "confidence": 0.8, "source": "self_reported", "collected_at": "2026-01-01"},
                "date_of_birth": {"value": "1985-01-01", "confidence": 0.7, "source": "self_reported", "collected_at": "2026-01-01"},
            },
            "visit_context": {
                "planned_visit_date": {"value": "2026-03-01", "confidence": 0.9, "source": "self_reported", "collected_at": "2026-01-01"},
                "party_size": {"value": 4, "confidence": 0.85, "source": "self_reported", "collected_at": "2026-01-01"},
                "occasion": {"value": "bday", "confidence": 0.8, "source": "contextual_extraction", "collected_at": "2026-01-01"},
            },
            "preferences": {
                "dining": {
                    "dietary_restrictions": {"value": "none", "confidence": 0.7, "source": "self_reported", "collected_at": "2026-01-01"},
                },
            },
        }
        completeness = calculate_completeness(profile)
        assert completeness >= 0.60


# ---------------------------------------------------------------------------
# TestConftestCleanup
# ---------------------------------------------------------------------------


class TestConftestCleanup:
    """Verify conftest cleanup includes guest_profile memory store."""

    def test_clear_memory_store_exists(self):
        """clear_memory_store function is importable and callable."""
        from src.data.guest_profile import clear_memory_store

        # Should not raise
        clear_memory_store()

    def test_memory_store_is_cleared(self):
        """Memory store is empty after clear."""
        from src.data.guest_profile import _memory_store, clear_memory_store

        # Add something
        _memory_store["test:key"] = {"test": "data"}
        assert len(_memory_store) > 0

        clear_memory_store()
        assert len(_memory_store) == 0
