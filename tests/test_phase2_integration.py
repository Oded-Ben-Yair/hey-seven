"""Phase 2 integration tests — deterministic subset.

Mock purge R111: Retained only deterministic tests that do not depend on
mock-based LLM tests removed. All behavioral validation uses live eval.

Covers: graph topology (whisper wiring, node counts, edges), whisper plan
formatting, profile completeness calculation, conftest cleanup, SMS webhook,
graph endpoint, format_whisper_plan.
"""

import pytest

from langchain_core.messages import AIMessage, HumanMessage


class _Placeholder:
    """Minimal placeholder object for app.state fields (replaces MagicMock)."""

    pass


# ---------------------------------------------------------------------------
# TestWhisperGraphWiring (deterministic — graph topology inspection)
# ---------------------------------------------------------------------------


class TestWhisperGraphWiring:
    """Verify whisper_planner is correctly wired in the graph topology."""

    def test_whisper_node_in_graph(self):
        """whisper_planner is one of the 13 nodes."""
        from src.agent.graph import build_graph

        graph = build_graph()
        all_nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
        assert "whisper_planner" in all_nodes
        assert len(all_nodes) == 13

    def test_retrieve_to_whisper_edge(self):
        """retrieve has an edge to whisper_planner."""
        from src.agent.graph import build_graph

        graph = build_graph()
        drawable = graph.get_graph()
        retrieve_targets = {e.target for e in drawable.edges if e.source == "retrieve"}
        assert "whisper_planner" in retrieve_targets

    def test_whisper_to_pre_extract_edge(self):
        """whisper_planner has an edge to pre_extract."""
        from src.agent.graph import build_graph

        graph = build_graph()
        drawable = graph.get_graph()
        whisper_targets = {
            e.target for e in drawable.edges if e.source == "whisper_planner"
        }
        assert "pre_extract" in whisper_targets

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
        assert "generate" in validate_targets
        assert "whisper_planner" not in validate_targets

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
# TestWhisperNodeMetadata (deterministic — pure function)
# ---------------------------------------------------------------------------


class TestWhisperNodeMetadata:
    """Verify _extract_node_metadata handles whisper_planner."""

    def test_metadata_with_plan(self):
        """whisper_planner with a plan returns has_plan=True."""
        from src.agent.graph import _extract_node_metadata

        output = {"whisper_plan": {"next_topic": "dining", "conversation_note": "test"}}
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
# TestSMSWebhookEndpoint (deterministic — real FastAPI TestClient, no LLM mocks)
# ---------------------------------------------------------------------------


class TestSMSWebhookEndpoint:
    """Test POST /sms/webhook endpoint."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a test client for the FastAPI app with SMS enabled."""
        from fastapi.testclient import TestClient
        from src.api.app import create_app

        monkeypatch.setenv("SMS_ENABLED", "true")
        monkeypatch.setenv("CONSENT_HMAC_SECRET", "test-secret-for-sms-webhook-tests")
        app = create_app()
        app.state.agent = _Placeholder()
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
# TestGraphEndpointV2 (deterministic — real FastAPI TestClient, topology checks)
# ---------------------------------------------------------------------------


class TestGraphEndpointV2:
    """Test GET /graph returns v2.1 topology."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        from fastapi.testclient import TestClient
        from src.api.app import create_app

        app = create_app()
        app.state.agent = _Placeholder()
        app.state.property_data = {"property": {"name": "Test Casino"}}
        app.state.ready = True
        return TestClient(app)

    def test_graph_returns_13_nodes(self, client):
        """GET /graph returns exactly 13 nodes."""
        response = client.get("/graph")
        assert response.status_code == 200
        data = response.json()
        assert len(data["nodes"]) == 13

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
        matching = [
            e for e in edges if e["from"] == "retrieve" and e["to"] == "whisper_planner"
        ]
        assert len(matching) == 1

    def test_graph_whisper_to_pre_extract_edge(self, client):
        """whisper_planner -> pre_extract edge exists."""
        response = client.get("/graph")
        data = response.json()
        edges = data["edges"]
        matching = [
            e
            for e in edges
            if e["from"] == "whisper_planner" and e["to"] == "pre_extract"
        ]
        assert len(matching) == 1

    def test_graph_persona_to_respond_edge(self, client):
        """persona_envelope -> respond edge exists."""
        response = client.get("/graph")
        data = response.json()
        edges = data["edges"]
        matching = [
            e for e in edges if e["from"] == "persona_envelope" and e["to"] == "respond"
        ]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# TestFormatWhisperPlan (deterministic — pure function)
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
            "conversation_note": "Guest seems interested in Italian",
        }
        result = format_whisper_plan(plan)
        assert "Whisper Track Guidance" in result
        assert "dining" in result
        assert "Italian" in result

    def test_empty_targets_shows_none(self):
        """Empty extraction targets shows (none)."""
        from src.agent.whisper_planner import format_whisper_plan

        plan = {
            "next_topic": "none",
            "conversation_note": "",
        }
        result = format_whisper_plan(plan)
        assert "none" in result

    def test_plan_includes_internal_label(self):
        """Plan output includes 'never reveal to guest' warning."""
        from src.agent.whisper_planner import format_whisper_plan

        plan = {
            "next_topic": "gaming",
            "conversation_note": "Ask about games",
        }
        result = format_whisper_plan(plan)
        assert "never reveal to guest" in result


# ---------------------------------------------------------------------------
# TestProfileCompletenessCalculation (deterministic — pure function)
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
                "name": {
                    "value": "John",
                    "confidence": 0.9,
                    "source": "self_reported",
                    "collected_at": "2026-01-01",
                },
                "email": {
                    "value": "john@test.com",
                    "confidence": 0.8,
                    "source": "self_reported",
                    "collected_at": "2026-01-01",
                },
                "language": {
                    "value": "en",
                    "confidence": 0.95,
                    "source": "contextual_extraction",
                    "collected_at": "2026-01-01",
                },
                "full_name": {
                    "value": "John Doe",
                    "confidence": 0.85,
                    "source": "self_reported",
                    "collected_at": "2026-01-01",
                },
                "date_of_birth": {
                    "value": "1985-06-15",
                    "confidence": 0.7,
                    "source": "self_reported",
                    "collected_at": "2026-01-01",
                },
            },
        }
        completeness = calculate_completeness(profile)
        assert completeness > 0.0
        assert completeness > 0.3

    def test_sixty_percent_threshold(self):
        """Profile with enough fields reaches 60% threshold."""
        from src.data.models import calculate_completeness

        profile = {
            "core_identity": {
                "name": {
                    "value": "John",
                    "confidence": 0.9,
                    "source": "self_reported",
                    "collected_at": "2026-01-01",
                },
                "email": {
                    "value": "j@t.com",
                    "confidence": 0.8,
                    "source": "self_reported",
                    "collected_at": "2026-01-01",
                },
                "language": {
                    "value": "en",
                    "confidence": 0.9,
                    "source": "contextual_extraction",
                    "collected_at": "2026-01-01",
                },
                "full_name": {
                    "value": "John D",
                    "confidence": 0.8,
                    "source": "self_reported",
                    "collected_at": "2026-01-01",
                },
                "date_of_birth": {
                    "value": "1985-01-01",
                    "confidence": 0.7,
                    "source": "self_reported",
                    "collected_at": "2026-01-01",
                },
            },
            "visit_context": {
                "planned_visit_date": {
                    "value": "2026-03-01",
                    "confidence": 0.9,
                    "source": "self_reported",
                    "collected_at": "2026-01-01",
                },
                "party_size": {
                    "value": 4,
                    "confidence": 0.85,
                    "source": "self_reported",
                    "collected_at": "2026-01-01",
                },
                "occasion": {
                    "value": "bday",
                    "confidence": 0.8,
                    "source": "contextual_extraction",
                    "collected_at": "2026-01-01",
                },
            },
            "preferences": {
                "dining": {
                    "dietary_restrictions": {
                        "value": "none",
                        "confidence": 0.7,
                        "source": "self_reported",
                        "collected_at": "2026-01-01",
                    },
                },
            },
        }
        completeness = calculate_completeness(profile)
        assert completeness >= 0.60


# ---------------------------------------------------------------------------
# TestConftestCleanup (deterministic)
# ---------------------------------------------------------------------------


class TestConftestCleanup:
    """Verify conftest cleanup includes guest_profile memory store."""

    def test_clear_memory_store_exists(self):
        """clear_memory_store function is importable and callable."""
        from src.data.guest_profile import clear_memory_store

        clear_memory_store()

    def test_memory_store_is_cleared(self):
        """Memory store is empty after clear."""
        from src.data.guest_profile import _memory_store, clear_memory_store

        _memory_store["test:key"] = {"test": "data"}
        assert len(_memory_store) > 0

        clear_memory_store()
        assert len(_memory_store) == 0
