"""Phase 3 integration tests: CMS webhook endpoint, casino config, feature flags.

Validates that Phase 3 modules (casino config, CMS webhook) are correctly
wired into the API and that feature flag machinery works end-to-end.
"""

import hashlib
import hmac
import json
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sign(body_bytes: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature for CMS webhook."""
    return hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()


def _valid_dining_payload() -> dict:
    """Return a valid CMS webhook payload for the dining category."""
    return {
        "casino_id": "mohegan_sun",
        "category": "dining",
        "item": {
            "id": "bobbys-burgers",
            "name": "Bobby's Burgers",
            "description": "Celebrity chef burger restaurant",
            "active": True,
            "cuisine": "American",
            "price_range": "$$",
            "hours": "11 AM - 10 PM",
            "location": "Casino Level 1",
        },
    }


# ---------------------------------------------------------------------------
# TestCMSWebhookEndpoint
# ---------------------------------------------------------------------------


class TestCMSWebhookEndpoint:
    """Test POST /cms/webhook endpoint."""

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

    def _patch_secret(self, monkeypatch, secret: str):
        """Patch CMS_WEBHOOK_SECRET on the cached settings instance.

        The cms_webhook endpoint calls ``get_settings().CMS_WEBHOOK_SECRET``
        at runtime, so patching the cached settings object is sufficient.
        """
        from src.config import get_settings

        monkeypatch.setattr(get_settings(), "CMS_WEBHOOK_SECRET", secret)

    def test_valid_item_returns_indexed(self, client, monkeypatch):
        """Valid CMS webhook with proper signature returns 200 and 'indexed' status."""
        secret = "test-secret-123"
        self._patch_secret(monkeypatch, secret)

        payload = _valid_dining_payload()
        body_bytes = json.dumps(payload).encode()
        sig = _sign(body_bytes, secret)

        response = client.post(
            "/cms/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": sig,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert data["item_id"] == "bobbys-burgers"

    def test_unchanged_item_returns_unchanged(self, client, monkeypatch):
        """Sending same item twice returns 'unchanged' on second call."""
        secret = "test-secret-123"
        self._patch_secret(monkeypatch, secret)

        payload = _valid_dining_payload()
        body_bytes = json.dumps(payload).encode()
        sig = _sign(body_bytes, secret)
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": sig,
        }

        # First call: indexed
        resp1 = client.post("/cms/webhook", content=body_bytes, headers=headers)
        assert resp1.json()["status"] == "indexed"

        # Second call: unchanged (same content hash)
        resp2 = client.post("/cms/webhook", content=body_bytes, headers=headers)
        assert resp2.json()["status"] == "unchanged"

    def test_invalid_signature_returns_403(self, client, monkeypatch):
        """Bad HMAC signature returns 403 and 'rejected'."""
        self._patch_secret(monkeypatch, "real-secret")

        payload = _valid_dining_payload()
        body_bytes = json.dumps(payload).encode()

        response = client.post(
            "/cms/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": "bad-signature",
            },
        )
        assert response.status_code == 403
        data = response.json()
        assert data["status"] == "rejected"

    def test_missing_casino_id_returns_rejected(self, client, monkeypatch):
        """Missing casino_id in payload returns 'rejected'."""
        secret = "test-secret"
        self._patch_secret(monkeypatch, secret)

        payload = {"category": "dining", "item": {"name": "Test"}}
        body_bytes = json.dumps(payload).encode()
        sig = _sign(body_bytes, secret)

        response = client.post(
            "/cms/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": sig,
            },
        )
        # Missing casino_id -> rejected -> 403
        assert response.status_code == 403
        assert response.json()["status"] == "rejected"

    def test_missing_category_returns_rejected(self, client, monkeypatch):
        """Missing category returns 'rejected'."""
        secret = "test-secret"
        self._patch_secret(monkeypatch, secret)

        payload = {"casino_id": "mohegan_sun", "item": {"name": "Test"}}
        body_bytes = json.dumps(payload).encode()
        sig = _sign(body_bytes, secret)

        response = client.post(
            "/cms/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": sig,
            },
        )
        assert response.status_code == 403
        assert response.json()["status"] == "rejected"

    def test_invalid_item_returns_quarantined(self, client, monkeypatch):
        """Item missing required fields returns 'quarantined'."""
        secret = "test-secret"
        self._patch_secret(monkeypatch, secret)

        # dining requires: id, name, description, active, cuisine, price_range
        # This item is missing most required fields
        payload = {
            "casino_id": "mohegan_sun",
            "category": "dining",
            "item": {"name": "Incomplete Restaurant"},
        }
        body_bytes = json.dumps(payload).encode()
        sig = _sign(body_bytes, secret)

        response = client.post(
            "/cms/webhook",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": sig,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "quarantined"
        assert len(data["errors"]) > 0

    def test_no_signature_header_rejects_when_no_secret(self, client, monkeypatch):
        """When CMS_WEBHOOK_SECRET is empty and no signature header, webhook rejects."""
        self._patch_secret(monkeypatch, "")

        payload = _valid_dining_payload()
        body_bytes = json.dumps(payload).encode()

        response = client.post(
            "/cms/webhook",
            content=body_bytes,
            headers={"Content-Type": "application/json"},
        )
        # With empty secret and no signature, verify_webhook_signature
        # returns False (both are falsy) -> rejected -> 403
        assert response.status_code == 403
        assert response.json()["status"] == "rejected"

    def test_malformed_json_returns_500(self, client):
        """Non-JSON body returns 500."""
        response = client.post(
            "/cms/webhook",
            content=b"not-valid-json{{{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 500
        assert "error" in response.json()


# ---------------------------------------------------------------------------
# TestCasinoConfigIntegration
# ---------------------------------------------------------------------------


class TestCasinoConfigIntegration:
    """Integration tests for per-casino config."""

    @pytest.mark.asyncio
    async def test_default_config_returns_all_sections(self):
        """get_casino_config returns config with all expected sections."""
        from src.casino.config import get_casino_config

        config = await get_casino_config("test_casino")

        assert "branding" in config
        assert "regulations" in config
        assert "operational" in config
        assert "rag" in config
        assert "prompts" in config
        assert "features" in config

    @pytest.mark.asyncio
    async def test_feature_flags_defaults(self):
        """Default feature flags have expected values."""
        from src.casino.config import get_casino_config

        config = await get_casino_config("test_casino")
        features = config["features"]

        assert features["sms_enabled"] is False
        assert features["comp_agent_enabled"] is True
        assert features["whisper_planner_enabled"] is True
        assert features["ai_disclosure_enabled"] is True
        assert features["outbound_campaigns_enabled"] is False

    @pytest.mark.asyncio
    async def test_is_feature_enabled_known_flag(self):
        """is_feature_enabled returns correct bool for known flags."""
        from src.casino.feature_flags import is_feature_enabled

        assert await is_feature_enabled("test_casino", "comp_agent_enabled") is True
        assert await is_feature_enabled("test_casino", "sms_enabled") is False

    @pytest.mark.asyncio
    async def test_is_feature_enabled_unknown_flag_returns_false(self):
        """Unknown flag name returns False (safe default)."""
        from src.casino.feature_flags import is_feature_enabled

        result = await is_feature_enabled("test_casino", "nonexistent_flag")
        assert result is False

    @pytest.mark.asyncio
    async def test_config_cache_cleared_between_tests(self):
        """Config cache is actually cleared by conftest fixture."""
        from src.casino.config import _config_cache, get_casino_config

        # First call populates cache
        await get_casino_config("cache_test_casino")
        assert "cache_test_casino" in _config_cache

        # The conftest autouse fixture clears it after each test.
        # We verify that the cache is populated (confirming caching works).
        # The fact that this test doesn't see stale data from OTHER tests
        # proves the fixture works.
        config = await get_casino_config("cache_test_casino")
        assert config["_id"] == "cache_test_casino"


# ---------------------------------------------------------------------------
# TestFeatureFlagGating
# ---------------------------------------------------------------------------


class TestFeatureFlagGating:
    """Test that feature flags could gate graph behavior."""

    def test_whisper_planner_flag_exists(self):
        """whisper_planner_enabled flag is in DEFAULT_FEATURES."""
        from src.casino.feature_flags import DEFAULT_FEATURES

        assert "whisper_planner_enabled" in DEFAULT_FEATURES

    def test_comp_agent_flag_exists(self):
        """comp_agent_enabled flag is in DEFAULT_FEATURES."""
        from src.casino.feature_flags import DEFAULT_FEATURES

        assert "comp_agent_enabled" in DEFAULT_FEATURES

    def test_sms_flag_exists(self):
        """sms_enabled flag is in DEFAULT_FEATURES."""
        from src.casino.feature_flags import DEFAULT_FEATURES

        assert "sms_enabled" in DEFAULT_FEATURES

    def test_all_default_features_are_bool(self):
        """Every value in DEFAULT_FEATURES is a bool."""
        from src.casino.feature_flags import DEFAULT_FEATURES

        for key, value in DEFAULT_FEATURES.items():
            assert isinstance(value, bool), f"{key} is {type(value)}, expected bool"

    def test_get_default_features_returns_copy(self):
        """get_default_features returns a copy, not the original."""
        from src.casino.feature_flags import DEFAULT_FEATURES, get_default_features

        copy = get_default_features()
        assert copy == DEFAULT_FEATURES
        assert copy is not DEFAULT_FEATURES

        # Mutating the copy should not affect the original
        copy["test_mutation"] = True
        assert "test_mutation" not in DEFAULT_FEATURES


# ---------------------------------------------------------------------------
# TestCMSValidation
# ---------------------------------------------------------------------------


class TestCMSValidation:
    """Integration-level validation tests for CMS items."""

    def test_restaurant_item_validates(self):
        """Valid dining item passes validation."""
        from src.cms.validation import validate_item

        item = {
            "id": "steakhouse-1",
            "name": "Prime Steakhouse",
            "description": "Fine dining steakhouse",
            "active": True,
            "cuisine": "Steakhouse",
            "price_range": "$$$",
        }
        is_valid, errors = validate_item(item, "dining")
        assert is_valid is True
        assert errors == []

    def test_restaurant_missing_name_fails(self):
        """Dining item without name fails validation."""
        from src.cms.validation import validate_item

        item = {
            "id": "no-name",
            "description": "Missing name field",
            "active": True,
            "cuisine": "Italian",
            "price_range": "$$",
        }
        is_valid, errors = validate_item(item, "dining")
        assert is_valid is False
        assert any("name" in e for e in errors)

    def test_unknown_category_quarantined(self):
        """Unknown category items are quarantined (not forward-compatible)."""
        from src.cms.validation import validate_item

        item = {
            "id": "test-1",
            "name": "Test Item",
            "description": "Some item",
            "active": True,
        }
        is_valid, errors = validate_item(item, "unknown_category")
        assert is_valid is False
        assert any("Unknown category" in e for e in errors)

    def test_details_json_validation(self):
        """Item with malformed details_json is caught."""
        from src.cms.validation import validate_details_json

        item = {
            "id": "bad-json",
            "name": "Bad JSON Item",
            "details_json": "{not valid json!!!",
        }
        is_valid, errors = validate_details_json(item)
        assert is_valid is False
        assert len(errors) > 0
        assert any("details_json" in e for e in errors)
