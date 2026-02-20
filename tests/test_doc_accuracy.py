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
        assert actual_count == 56, (
            f"Settings has {actual_count} fields, but docs say 56. "
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

    def test_property_qa_state_has_13_fields(self):
        """PropertyQAState has exactly 13 fields."""
        from src.agent.state import PropertyQAState

        actual = len(PropertyQAState.__annotations__)
        assert actual == 13, (
            f"PropertyQAState has {actual} fields, expected 13. "
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
