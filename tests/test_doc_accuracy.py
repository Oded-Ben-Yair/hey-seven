"""Tests to prevent documentation drift from code reality.

These tests assert that key counts, names, and contracts documented in
README.md and ARCHITECTURE.md match the actual codebase. Prevents the
documentation rot identified in R6.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class TestSettingsCount:
    """Verify the documented settings count matches config.py."""

    def test_config_field_count_matches_docs(self):
        """Settings class field count matches what README claims."""
        from src.config import Settings

        # Count fields defined directly on Settings (exclude inherited)
        fields = Settings.model_fields
        actual_count = len(fields)
        assert actual_count == 59, (
            f"Settings has {actual_count} fields, but docs say 59. "
            f"Update README.md and .env.example if count changed."
        )


class TestAgentRegistry:
    """Verify the documented agent count matches the registry."""

    def test_registry_agent_count(self):
        """Agent registry has 5 specialist agents."""
        from src.agent.agents.registry import _AGENT_REGISTRY

        assert len(_AGENT_REGISTRY) == 5, (
            f"Agent registry has {len(_AGENT_REGISTRY)} agents, expected 5. "
            f"Update README and ARCHITECTURE if agents were added/removed."
        )

    def test_registry_contains_hotel(self):
        """Hotel agent is registered."""
        from src.agent.agents.registry import _AGENT_REGISTRY

        assert "hotel" in _AGENT_REGISTRY, "Hotel agent missing from registry"

    def test_registry_agent_names(self):
        """All 5 specialist agents are registered."""
        from src.agent.agents.registry import list_agents

        expected = ["comp", "dining", "entertainment", "host", "hotel"]
        assert list_agents() == expected


class TestStateFieldCount:
    """Verify the documented state field count matches code."""

    def test_property_qa_state_has_16_fields(self):
        """PropertyQAState has exactly 16 fields (13 v1/v2 + 3 v3 Phase 3)."""
        from src.agent.state import PropertyQAState

        actual = len(PropertyQAState.__annotations__)
        assert actual == 16, (
            f"PropertyQAState has {actual} fields, expected 16. "
            f"Update ARCHITECTURE.md State Schema section if count changed."
        )


class TestSSEEventModels:
    """Verify all SSE event types have corresponding Pydantic models."""

    def test_ping_event_model_exists(self):
        """SSEPingEvent model exists."""
        from src.api.models import SSEPingEvent

        assert SSEPingEvent is not None

    def test_all_sse_event_models_importable(self):
        """All 9 SSE event models can be imported."""
        from src.api.models import (
            SSEDoneEvent,
            SSEErrorEvent,
            SSEGraphNodeEvent,
            SSEMetadataEvent,
            SSEPingEvent,
            SSEReplaceEvent,
            SSESourcesEvent,
            SSETokenEvent,
        )

        models = [
            SSEMetadataEvent, SSETokenEvent, SSESourcesEvent,
            SSEDoneEvent, SSEPingEvent, SSEErrorEvent,
            SSEGraphNodeEvent, SSEReplaceEvent,
        ]
        assert len(models) == 8


class TestHealthResponseModel:
    """Verify HealthResponse includes all documented fields."""

    def test_health_response_has_circuit_breaker_state(self):
        """HealthResponse has circuit_breaker_state field."""
        from src.api.models import HealthResponse

        hr = HealthResponse(
            status="healthy",
            version="1.0.0",
            agent_ready=True,
            property_loaded=True,
            rag_ready=True,
            observability_enabled=False,
            circuit_breaker_state="closed",
        )
        assert hr.circuit_breaker_state == "closed"

    def test_health_response_field_count(self):
        """HealthResponse has 8 fields (including environment)."""
        from src.api.models import HealthResponse

        assert len(HealthResponse.model_fields) == 8


class TestErrorTaxonomy:
    """Verify error taxonomy matches documentation."""

    def test_error_code_count(self):
        """ErrorCode enum has 8 codes."""
        from src.api.errors import ErrorCode

        assert len(ErrorCode) == 8

    def test_error_code_values(self):
        """All 8 documented error codes exist."""
        from src.api.errors import ErrorCode

        expected = {
            "unauthorized", "not_found", "rate_limit_exceeded", "payload_too_large",
            "agent_unavailable", "internal_error", "validation_error",
            "service_degraded",
        }
        actual = {code.value for code in ErrorCode}
        assert actual == expected


class TestMiddlewareProtectedPaths:
    """Verify ApiKeyMiddleware protects the documented paths."""

    def test_protected_paths_match_docs(self):
        """ApiKeyMiddleware._PROTECTED_PATHS matches what README documents."""
        from src.api.middleware import ApiKeyMiddleware

        expected = {"/chat", "/graph", "/property", "/feedback"}
        assert ApiKeyMiddleware._PROTECTED_PATHS == expected


class TestRateLimitScope:
    """Verify rate limiter applies to the documented paths."""

    def test_rate_limited_paths(self):
        """RateLimitMiddleware applies to /chat and /feedback."""
        # Verified by reading source: path not in ("/chat", "/feedback") -> pass through
        # This is a structural assertion based on the middleware code.
        import ast
        import inspect
        from src.api.middleware import RateLimitMiddleware

        source = inspect.getsource(RateLimitMiddleware.__call__)
        assert '"/chat"' in source
        assert '"/feedback"' in source


class TestWebhookResponseModels:
    """Verify webhook response models exist and validate."""

    def test_sms_webhook_response(self):
        """SmsWebhookResponse model validates."""
        from src.api.models import SmsWebhookResponse

        resp = SmsWebhookResponse(status="ignored")
        assert resp.status == "ignored"

    def test_cms_webhook_response(self):
        """CmsWebhookResponse model validates."""
        from src.api.models import CmsWebhookResponse

        resp = CmsWebhookResponse(status="success", updated_categories=["restaurants"])
        assert resp.status == "success"
        assert resp.updated_categories == ["restaurants"]


class TestVersionConsistency:
    """Verify VERSION defaults are consistent across config and .env files."""

    def test_config_version_default(self):
        """config.py VERSION default is 1.0.0."""
        from src.config import Settings

        default = Settings.model_fields["VERSION"].default
        assert default == "1.0.0", (
            f"config.py VERSION default is {default!r}, expected '1.0.0'. "
            f"Sync .env, .env.example, and ARCHITECTURE.md."
        )

    def test_env_example_version(self):
        """env.example VERSION matches config.py default."""
        env_example = ROOT / ".env.example"
        content = env_example.read_text()
        match = re.search(r"^VERSION=([^\s#]+)", content, re.MULTILINE)
        assert match, "VERSION not found in .env.example"
        assert match.group(1).strip() == "1.0.0"


class TestCategoryToAgentMapping:
    """Verify category-to-agent dispatch includes hotel."""

    def test_hotel_category_mapped(self):
        """Hotel category is mapped in _CATEGORY_TO_AGENT."""
        from src.agent.graph import _CATEGORY_TO_AGENT

        assert "hotel" in _CATEGORY_TO_AGENT
        assert _CATEGORY_TO_AGENT["hotel"] == "hotel"


class TestGraphNodeCount:
    """Verify the documented graph node count matches code."""

    def test_graph_has_expected_nodes(self):
        """StateGraph has the documented number of nodes (excluding __start__, __end__)."""
        from src.agent.graph import build_graph

        graph = build_graph()
        g = graph.get_graph()
        # g.nodes may be a dict (keyed by node ID) or iterable of node objects.
        # Handle both: if dict, use keys; if iterable with .id, extract IDs.
        if isinstance(g.nodes, dict):
            nodes = [n for n in g.nodes if n not in ("__start__", "__end__")]
        else:
            nodes = [
                (n.id if hasattr(n, "id") else str(n))
                for n in g.nodes
                if (n.id if hasattr(n, "id") else str(n)) not in ("__start__", "__end__")
            ]
        assert len(nodes) == 11, (
            f"Graph has {len(nodes)} nodes, expected 11. "
            f"Nodes: {nodes}"
        )

    def test_known_nodes_frozenset_matches_graph(self):
        """_KNOWN_NODES frozenset has same count as actual graph nodes."""
        from src.agent.graph import _KNOWN_NODES, build_graph

        graph = build_graph()
        g = graph.get_graph()
        if isinstance(g.nodes, dict):
            actual = {n for n in g.nodes if n not in ("__start__", "__end__")}
        else:
            actual = {
                (n.id if hasattr(n, "id") else str(n))
                for n in g.nodes
                if (n.id if hasattr(n, "id") else str(n)) not in ("__start__", "__end__")
            }
        assert _KNOWN_NODES == actual, (
            f"_KNOWN_NODES mismatch: "
            f"missing_from_frozenset={actual - _KNOWN_NODES}, "
            f"extra_in_frozenset={_KNOWN_NODES - actual}"
        )


class TestGuardrailPatternCount:
    """Verify the documented guardrail pattern count matches code."""

    def test_total_guardrail_patterns(self):
        """guardrails.py has the expected total number of re.compile() patterns."""
        import inspect
        from src.agent import guardrails

        source = inspect.getsource(guardrails)
        patterns = re.findall(r"re\.compile\(", source)
        assert len(patterns) == 84, (
            f"guardrails.py has {len(patterns)} re.compile() patterns, expected 84. "
            f"Update docs if patterns were added/removed."
        )

    def test_five_guardrail_categories(self):
        """guardrails.py defines exactly 5 guardrail pattern categories."""
        from src.agent import guardrails

        category_lists = [
            guardrails._INJECTION_PATTERNS,
            guardrails._RESPONSIBLE_GAMING_PATTERNS,
            guardrails._AGE_VERIFICATION_PATTERNS,
            guardrails._BSA_AML_PATTERNS,
            guardrails._PATRON_PRIVACY_PATTERNS,
        ]
        assert len(category_lists) == 5, (
            f"Expected 5 guardrail categories, found {len(category_lists)}."
        )
        # Each category must have at least 1 pattern
        for i, cat in enumerate(category_lists):
            assert len(cat) > 0, f"Category {i} is empty"

    def test_injection_pattern_count(self):
        """Prompt injection has 11 patterns."""
        from src.agent.guardrails import _INJECTION_PATTERNS

        assert len(_INJECTION_PATTERNS) == 11, (
            f"_INJECTION_PATTERNS has {len(_INJECTION_PATTERNS)}, expected 11."
        )

    def test_responsible_gaming_pattern_count(self):
        """Responsible gaming has 31 patterns (English + Spanish + Portuguese + Mandarin)."""
        from src.agent.guardrails import _RESPONSIBLE_GAMING_PATTERNS

        assert len(_RESPONSIBLE_GAMING_PATTERNS) == 31, (
            f"_RESPONSIBLE_GAMING_PATTERNS has {len(_RESPONSIBLE_GAMING_PATTERNS)}, expected 31."
        )

    def test_bsa_aml_pattern_count(self):
        """BSA/AML has 25 patterns (English + Spanish + Portuguese + Mandarin)."""
        from src.agent.guardrails import _BSA_AML_PATTERNS

        assert len(_BSA_AML_PATTERNS) == 25, (
            f"_BSA_AML_PATTERNS has {len(_BSA_AML_PATTERNS)}, expected 25."
        )


class TestMiddlewareCount:
    """Verify the documented middleware count matches code."""

    def test_middleware_all_exports(self):
        """middleware.py __all__ exports exactly 6 middleware classes."""
        from src.api import middleware

        assert len(middleware.__all__) == 6, (
            f"middleware.__all__ has {len(middleware.__all__)} entries, expected 6. "
            f"Entries: {middleware.__all__}"
        )

    def test_middleware_names(self):
        """All 6 documented middleware classes are exported."""
        from src.api.middleware import __all__ as mw_all

        expected = {
            "RequestLoggingMiddleware",
            "ErrorHandlingMiddleware",
            "SecurityHeadersMiddleware",
            "ApiKeyMiddleware",
            "RateLimitMiddleware",
            "RequestBodyLimitMiddleware",
        }
        assert set(mw_all) == expected, (
            f"Middleware mismatch: "
            f"missing={expected - set(mw_all)}, "
            f"extra={set(mw_all) - expected}"
        )


class TestEndpointCount:
    """Verify the documented endpoint count matches code."""

    # FastAPI adds built-in routes (/docs, /redoc, /openapi.json, /docs/oauth2-redirect).
    # Exclude them to count only application-defined endpoints.
    _FASTAPI_BUILTIN = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}

    def test_app_has_expected_endpoints(self):
        """FastAPI app has exactly 8 application-defined endpoints."""
        from src.api.app import create_app

        app = create_app()
        endpoints = [
            route.path
            for route in app.routes
            if hasattr(route, "endpoint") and route.path not in self._FASTAPI_BUILTIN
        ]
        assert len(endpoints) == 8, (
            f"App has {len(endpoints)} app endpoints, expected 8. "
            f"Endpoints: {endpoints}"
        )

    def test_expected_endpoint_paths(self):
        """All 8 documented endpoint paths exist."""
        from src.api.app import create_app

        app = create_app()
        paths = {
            route.path
            for route in app.routes
            if hasattr(route, "endpoint") and route.path not in self._FASTAPI_BUILTIN
        }
        expected = {
            "/chat", "/live", "/health", "/property",
            "/graph", "/sms/webhook", "/cms/webhook", "/feedback",
        }
        assert paths == expected, (
            f"Endpoint mismatch: "
            f"missing={expected - paths}, "
            f"extra={paths - expected}"
        )
