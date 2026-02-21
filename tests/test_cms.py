"""Tests for the CMS package: Google Sheets client, content validation, and webhook handler.

Covers content category definitions, Sheets API client read operations,
SHA-256 content hashing, per-category validation with quarantine logging,
Details JSON validation, HMAC-SHA256 webhook signature verification,
change detection via content hashing, and webhook edge cases.
"""

import hashlib
import hmac
import json
import logging
import time
from unittest.mock import MagicMock

import pytest


# ============================================================================
# Content categories
# ============================================================================


class TestContentCategories:
    """Verify all 8 content categories are defined."""

    def test_all_eight_categories_defined(self):
        from src.cms.sheets_client import CONTENT_CATEGORIES

        assert len(CONTENT_CATEGORIES) == 8

    def test_expected_categories_present(self):
        from src.cms.sheets_client import CONTENT_CATEGORIES

        expected = {
            "dining",
            "entertainment",
            "spa",
            "gaming",
            "promotions",
            "regulations",
            "hours",
            "general_info",
        }
        assert set(CONTENT_CATEGORIES) == expected

    def test_categories_is_tuple(self):
        """Categories should be immutable."""
        from src.cms.sheets_client import CONTENT_CATEGORIES

        assert isinstance(CONTENT_CATEGORIES, tuple)

    def test_categories_exported_from_package(self):
        from src.cms import CONTENT_CATEGORIES

        assert len(CONTENT_CATEGORIES) == 8


# ============================================================================
# SheetsClient tests
# ============================================================================


class TestSheetsClient:
    """Google Sheets API client operations."""

    @pytest.mark.asyncio
    async def test_read_category_returns_list(self):
        """When Sheets API is unavailable, read_category returns empty list."""
        from src.cms.sheets_client import SheetsClient

        client = SheetsClient()
        result = await client.read_category("sheet-id-123", "dining")
        assert isinstance(result, list)
        assert result == []

    @pytest.mark.asyncio
    async def test_read_category_unknown_category_returns_empty(self):
        from src.cms.sheets_client import SheetsClient

        client = SheetsClient()
        result = await client.read_category("sheet-id-123", "nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_read_all_categories_returns_dict(self):
        """read_all_categories returns a dict with all 8 categories."""
        from src.cms.sheets_client import CONTENT_CATEGORIES, SheetsClient

        client = SheetsClient()
        result = await client.read_all_categories("sheet-id-123")
        assert isinstance(result, dict)
        assert set(result.keys()) == set(CONTENT_CATEGORIES)
        for category_items in result.values():
            assert isinstance(category_items, list)

    @pytest.mark.asyncio
    async def test_read_category_with_mock_service(self):
        """When Sheets API returns data, rows are parsed into dicts."""
        from src.cms.sheets_client import SheetsClient

        client = SheetsClient()

        mock_service = MagicMock()
        mock_result = {
            "values": [
                ["ID", "Name", "Description", "Active", "Cuisine", "Price Range"],
                ["1", "Test Restaurant", "Great food", "true", "Italian", "$$"],
                ["2", "Sushi Bar", "Fresh sushi", "true", "Japanese", "$$$"],
            ]
        }
        mock_service.spreadsheets().values().get().execute.return_value = mock_result

        client._service = mock_service
        client._available = True

        result = await client.read_category("sheet-id", "dining")
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[0]["name"] == "Test Restaurant"
        assert result[0]["cuisine"] == "Italian"
        assert result[1]["name"] == "Sushi Bar"

    @pytest.mark.asyncio
    async def test_read_category_header_only_returns_empty(self):
        """A sheet with only a header row returns no items."""
        from src.cms.sheets_client import SheetsClient

        client = SheetsClient()
        mock_service = MagicMock()
        mock_result = {"values": [["ID", "Name", "Description"]]}
        mock_service.spreadsheets().values().get().execute.return_value = mock_result

        client._service = mock_service
        client._available = True

        result = await client.read_category("sheet-id", "dining")
        assert result == []

    @pytest.mark.asyncio
    async def test_read_category_handles_short_rows(self):
        """Rows shorter than the header get empty strings for missing columns."""
        from src.cms.sheets_client import SheetsClient

        client = SheetsClient()
        mock_service = MagicMock()
        mock_result = {
            "values": [
                ["ID", "Name", "Description"],
                ["1"],  # Only ID, missing Name and Description
            ]
        }
        mock_service.spreadsheets().values().get().execute.return_value = mock_result

        client._service = mock_service
        client._available = True

        result = await client.read_category("sheet-id", "dining")
        assert len(result) == 1
        assert result[0]["id"] == "1"
        assert result[0]["name"] == ""
        assert result[0]["description"] == ""


class TestComputeContentHash:
    """SHA-256 content hashing for change detection."""

    def test_deterministic_hash(self):
        """Same input always produces the same hash."""
        from src.cms.sheets_client import compute_content_hash

        items = [{"id": "1", "name": "Test"}]
        hash1 = compute_content_hash(items)
        hash2 = compute_content_hash(items)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        from src.cms.sheets_client import compute_content_hash

        items_a = [{"id": "1", "name": "Restaurant A"}]
        items_b = [{"id": "1", "name": "Restaurant B"}]
        assert compute_content_hash(items_a) != compute_content_hash(items_b)

    def test_hash_is_sha256_hex(self):
        from src.cms.sheets_client import compute_content_hash

        result = compute_content_hash([{"id": "1"}])
        assert len(result) == 64  # SHA-256 hex = 64 chars
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_list_produces_valid_hash(self):
        from src.cms.sheets_client import compute_content_hash

        result = compute_content_hash([])
        assert len(result) == 64

    def test_key_order_independent(self):
        """Dict key order should not affect the hash (sort_keys=True)."""
        from src.cms.sheets_client import compute_content_hash

        items_a = [{"name": "Test", "id": "1"}]
        items_b = [{"id": "1", "name": "Test"}]
        assert compute_content_hash(items_a) == compute_content_hash(items_b)


# ============================================================================
# Validation tests
# ============================================================================


class TestValidation:
    """Content validation per category with quarantine."""

    def _make_valid_item(self, **overrides):
        """Create a base valid item with all common fields."""
        item = {
            "id": "item-001",
            "name": "Test Item",
            "description": "A test item",
            "active": "true",
        }
        item.update(overrides)
        return item

    def test_valid_dining_item_passes(self):
        from src.cms.validation import validate_item

        item = self._make_valid_item(cuisine="Italian", price_range="$$")
        is_valid, errors = validate_item(item, "dining")
        assert is_valid is True
        assert errors == []

    def test_missing_required_field_fails(self):
        from src.cms.validation import validate_item

        item = self._make_valid_item(cuisine="Italian")
        # Missing price_range for dining
        is_valid, errors = validate_item(item, "dining")
        assert is_valid is False
        assert any("price_range" in e for e in errors)

    def test_empty_string_field_fails(self):
        from src.cms.validation import validate_item

        item = self._make_valid_item(cuisine="Italian", price_range="  ")
        is_valid, errors = validate_item(item, "dining")
        assert is_valid is False
        assert any("price_range" in e for e in errors)

    def test_missing_base_field_fails(self):
        from src.cms.validation import validate_item

        item = {"cuisine": "Italian", "price_range": "$$"}
        # Missing id, name, description, active
        is_valid, errors = validate_item(item, "dining")
        assert is_valid is False
        assert len(errors) >= 4

    def test_unknown_category_fails(self):
        from src.cms.validation import validate_item

        item = self._make_valid_item()
        is_valid, errors = validate_item(item, "nonexistent_category")
        assert is_valid is False
        assert any("Unknown category" in e for e in errors)

    @pytest.mark.parametrize(
        "category,extra_fields",
        [
            ("dining", {"cuisine": "Italian", "price_range": "$$"}),
            ("entertainment", {"event_type": "Concert", "venue": "Main Arena"}),
            ("spa", {"treatment_type": "Massage", "duration": "60 min"}),
            ("gaming", {"game_type": "Blackjack", "location": "Main Floor"}),
            ("promotions", {"value": "$100", "valid_until": "2026-12-31"}),
            ("regulations", {"state": "CT", "effective_date": "2026-01-01"}),
            ("hours", {"weekday_hours": "9 AM - 5 PM"}),
            ("general_info", {}),
        ],
    )
    def test_valid_item_per_category(self, category, extra_fields):
        from src.cms.validation import validate_item

        item = self._make_valid_item(**extra_fields)
        is_valid, errors = validate_item(item, category)
        assert is_valid is True, f"Failed for {category}: {errors}"
        assert errors == []

    def test_quarantine_logging(self, caplog):
        from src.cms.validation import validate_item

        item = self._make_valid_item()
        # Missing cuisine and price_range for dining
        with caplog.at_level(logging.WARNING):
            is_valid, errors = validate_item(item, "dining")
        assert is_valid is False
        assert "Quarantined" in caplog.text

    def test_valid_details_json(self):
        from src.cms.validation import validate_details_json

        item = {"id": "1", "details_json": '{"hours": "9-5"}'}
        is_valid, errors = validate_details_json(item)
        assert is_valid is True
        assert errors == []

    def test_invalid_details_json_fails(self):
        from src.cms.validation import validate_details_json

        item = {"id": "1", "details_json": "{invalid json}"}
        is_valid, errors = validate_details_json(item)
        assert is_valid is False
        assert len(errors) == 1
        assert "Invalid details_json" in errors[0]

    def test_no_details_json_passes(self):
        from src.cms.validation import validate_details_json

        item = {"id": "1", "name": "Test"}
        is_valid, errors = validate_details_json(item)
        assert is_valid is True
        assert errors == []

    def test_empty_details_json_passes(self):
        from src.cms.validation import validate_details_json

        item = {"id": "1", "details_json": ""}
        is_valid, errors = validate_details_json(item)
        assert is_valid is True


# ============================================================================
# Webhook tests
# ============================================================================


class TestWebhookSignatureVerification:
    """HMAC-SHA256 webhook signature verification."""

    def test_valid_signature_passes(self):
        from src.cms.webhook import verify_webhook_signature

        body = b'{"casino_id": "mohegan_sun"}'
        secret = "test-webhook-secret"
        signature = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()

        assert verify_webhook_signature(body, signature, secret) is True

    def test_invalid_signature_fails(self):
        from src.cms.webhook import verify_webhook_signature

        body = b'{"casino_id": "mohegan_sun"}'
        assert verify_webhook_signature(body, "bad-signature", "secret") is False

    def test_empty_signature_fails(self):
        from src.cms.webhook import verify_webhook_signature

        assert verify_webhook_signature(b"body", "", "secret") is False

    def test_empty_secret_fails(self):
        from src.cms.webhook import verify_webhook_signature

        assert verify_webhook_signature(b"body", "sig", "") is False


class TestWebhookHandler:
    """CMS webhook handler for content updates."""

    def _make_valid_payload(self, **item_overrides):
        """Create a valid webhook payload."""
        item = {
            "id": "dining-001",
            "name": "Seasons Buffet",
            "description": "All-you-can-eat buffet",
            "active": "true",
            "cuisine": "Buffet",
            "price_range": "$$",
        }
        item.update(item_overrides)
        return {
            "casino_id": "mohegan_sun",
            "category": "dining",
            "item": item,
        }

    @pytest.mark.asyncio
    async def test_valid_item_gets_indexed(self):
        from src.cms.webhook import _content_hashes, handle_cms_webhook

        # Clear any stored hashes from prior tests
        _content_hashes.clear()

        payload = self._make_valid_payload()
        result = await handle_cms_webhook(payload, webhook_secret="test-secret")
        assert result["status"] == "indexed"
        assert result["item_id"] == "dining-001"

    @pytest.mark.asyncio
    async def test_invalid_item_gets_quarantined(self):
        from src.cms.webhook import handle_cms_webhook

        payload = {
            "casino_id": "mohegan_sun",
            "category": "dining",
            "item": {
                "id": "bad-001",
                "name": "Incomplete Item",
                # Missing description, active, cuisine, price_range
            },
        }
        result = await handle_cms_webhook(payload, webhook_secret="test-secret")
        assert result["status"] == "quarantined"
        assert result["item_id"] == "bad-001"
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_unchanged_content_returns_unchanged(self):
        from src.cms.webhook import _content_hashes, handle_cms_webhook

        _content_hashes.clear()

        payload = self._make_valid_payload()
        # First call: indexed
        result1 = await handle_cms_webhook(payload, webhook_secret="secret")
        assert result1["status"] == "indexed"

        # Second call with same content: unchanged
        result2 = await handle_cms_webhook(payload, webhook_secret="secret")
        assert result2["status"] == "unchanged"

    @pytest.mark.asyncio
    async def test_changed_content_gets_reindexed(self):
        from src.cms.webhook import _content_hashes, handle_cms_webhook

        _content_hashes.clear()

        payload = self._make_valid_payload()
        await handle_cms_webhook(payload, webhook_secret="secret")

        # Update the item description
        payload["item"]["description"] = "Updated buffet description"
        result = await handle_cms_webhook(payload, webhook_secret="secret")
        assert result["status"] == "indexed"

    @pytest.mark.asyncio
    async def test_signature_verification_passes(self):
        from src.cms.webhook import _content_hashes, handle_cms_webhook

        _content_hashes.clear()

        payload = self._make_valid_payload()
        raw_body = json.dumps(payload).encode("utf-8")
        secret = "test-webhook-secret"
        signature = hmac.new(
            secret.encode("utf-8"), raw_body, hashlib.sha256
        ).hexdigest()

        result = await handle_cms_webhook(
            payload,
            webhook_secret=secret,
            raw_body=raw_body,
            signature=signature,
        )
        assert result["status"] == "indexed"

    @pytest.mark.asyncio
    async def test_signature_verification_fails(self):
        from src.cms.webhook import handle_cms_webhook

        payload = self._make_valid_payload()
        raw_body = json.dumps(payload).encode("utf-8")

        result = await handle_cms_webhook(
            payload,
            webhook_secret="real-secret",
            raw_body=raw_body,
            signature="invalid-signature",
        )
        assert result["status"] == "rejected"
        assert "invalid signature" in result["errors"]


class TestWebhookReplayProtection:
    """CMS webhook replay protection via timestamp validation."""

    def test_valid_timestamp_passes(self):
        from src.cms.webhook import verify_webhook_signature

        body = b'{"casino_id": "mohegan_sun"}'
        secret = "test-webhook-secret"
        signature = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        timestamp = str(int(time.time()))

        assert verify_webhook_signature(body, signature, secret, timestamp=timestamp) is True

    def test_stale_timestamp_rejected(self):
        from src.cms.webhook import verify_webhook_signature

        body = b'{"casino_id": "mohegan_sun"}'
        secret = "test-webhook-secret"
        signature = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        # Timestamp from 10 minutes ago (outside 5-minute tolerance)
        timestamp = str(int(time.time()) - 600)

        assert verify_webhook_signature(body, signature, secret, timestamp=timestamp) is False

    def test_future_timestamp_rejected(self):
        from src.cms.webhook import verify_webhook_signature

        body = b'{"casino_id": "mohegan_sun"}'
        secret = "test-webhook-secret"
        signature = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        # Timestamp 10 minutes in the future
        timestamp = str(int(time.time()) + 600)

        assert verify_webhook_signature(body, signature, secret, timestamp=timestamp) is False

    def test_invalid_timestamp_rejected(self):
        from src.cms.webhook import verify_webhook_signature

        body = b'{"casino_id": "mohegan_sun"}'
        secret = "test-webhook-secret"
        signature = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()

        assert verify_webhook_signature(body, signature, secret, timestamp="not-a-number") is False

    def test_no_timestamp_passes(self):
        """Backward compatibility: no timestamp still passes if signature is valid."""
        from src.cms.webhook import verify_webhook_signature

        body = b'{"casino_id": "mohegan_sun"}'
        secret = "test-webhook-secret"
        signature = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()

        assert verify_webhook_signature(body, signature, secret) is True
        assert verify_webhook_signature(body, signature, secret, timestamp=None) is True


class TestWebhookEdgeCases:
    """Webhook handler edge cases and missing fields."""

    @pytest.mark.asyncio
    async def test_missing_casino_id(self):
        from src.cms.webhook import handle_cms_webhook

        payload = {
            "category": "dining",
            "item": {"id": "1", "name": "Test"},
        }
        result = await handle_cms_webhook(payload, webhook_secret="secret")
        assert result["status"] == "rejected"
        assert "missing casino_id" in result["errors"]

    @pytest.mark.asyncio
    async def test_missing_category(self):
        from src.cms.webhook import handle_cms_webhook

        payload = {
            "casino_id": "mohegan_sun",
            "item": {"id": "1", "name": "Test"},
        }
        result = await handle_cms_webhook(payload, webhook_secret="secret")
        assert result["status"] == "rejected"
        assert "missing category" in result["errors"]

    @pytest.mark.asyncio
    async def test_empty_item(self):
        from src.cms.webhook import handle_cms_webhook

        payload = {
            "casino_id": "mohegan_sun",
            "category": "dining",
            "item": {},
        }
        result = await handle_cms_webhook(payload, webhook_secret="secret")
        # Empty dict should fail validation (missing required fields)
        assert result["status"] == "quarantined"

    @pytest.mark.asyncio
    async def test_missing_item(self):
        from src.cms.webhook import handle_cms_webhook

        payload = {
            "casino_id": "mohegan_sun",
            "category": "dining",
        }
        result = await handle_cms_webhook(payload, webhook_secret="secret")
        assert result["status"] == "rejected"
        assert "missing or invalid item" in result["errors"]

    @pytest.mark.asyncio
    async def test_item_not_a_dict(self):
        from src.cms.webhook import handle_cms_webhook

        payload = {
            "casino_id": "mohegan_sun",
            "category": "dining",
            "item": "not a dict",
        }
        result = await handle_cms_webhook(payload, webhook_secret="secret")
        assert result["status"] == "rejected"
        assert "missing or invalid item" in result["errors"]

    @pytest.mark.asyncio
    async def test_invalid_details_json_quarantines(self):
        from src.cms.webhook import _content_hashes, handle_cms_webhook

        _content_hashes.clear()

        payload = {
            "casino_id": "mohegan_sun",
            "category": "dining",
            "item": {
                "id": "dining-bad-json",
                "name": "Bad JSON Restaurant",
                "description": "Has invalid details_json",
                "active": "true",
                "cuisine": "Italian",
                "price_range": "$$$",
                "details_json": "{not valid json!!!}",
            },
        }
        result = await handle_cms_webhook(payload, webhook_secret="secret")
        assert result["status"] == "quarantined"
        assert any("Invalid details_json" in e for e in result["errors"])


# ============================================================================
# Config integration test
# ============================================================================


class TestCMSConfigFields:
    """Verify CMS config fields exist in Settings."""

    def test_cms_config_fields_exist(self):
        from src.config import Settings

        s = Settings()
        assert hasattr(s, "CMS_WEBHOOK_SECRET")
        assert hasattr(s, "GOOGLE_SHEETS_ID")

    def test_cms_config_defaults(self):
        from src.config import Settings

        s = Settings()
        assert s.CMS_WEBHOOK_SECRET.get_secret_value() == ""
        assert s.GOOGLE_SHEETS_ID == ""

    def test_cms_webhook_secret_is_secretstr(self):
        """CMS_WEBHOOK_SECRET uses SecretStr to prevent accidental logging exposure."""
        from pydantic import SecretStr

        from src.config import Settings

        s = Settings()
        assert isinstance(s.CMS_WEBHOOK_SECRET, SecretStr)


# ============================================================================
# Content hash cache bounding test
# ============================================================================


class TestContentHashCacheBounded:
    """Verify _content_hashes is bounded (not an unbounded dict)."""

    def test_content_hashes_is_ttl_cache(self):
        """_content_hashes is a TTLCache (bounded) not a plain dict."""
        from cachetools import TTLCache

        from src.cms.webhook import _content_hashes

        assert isinstance(_content_hashes, TTLCache)

    def test_content_hashes_has_maxsize(self):
        """_content_hashes has a maxsize to prevent unbounded memory growth."""
        from src.cms.webhook import _content_hashes

        assert _content_hashes.maxsize >= 1000

    def test_content_hashes_has_ttl(self):
        """_content_hashes has a TTL to evict stale entries."""
        from src.cms.webhook import _content_hashes

        assert _content_hashes.ttl > 0


# ============================================================================
# Live re-indexing tests
# ============================================================================


class TestWebhookLiveReindexing:
    """CMS webhook triggers live vector store upsert on content change."""

    def _make_valid_payload(self, **item_overrides):
        """Create a valid webhook payload."""
        item = {
            "id": "dining-reindex-001",
            "name": "Reindex Test Restaurant",
            "description": "Restaurant for reindex testing",
            "active": "true",
            "cuisine": "Italian",
            "price_range": "$$$",
        }
        item.update(item_overrides)
        return {
            "casino_id": "mohegan_sun",
            "category": "dining",
            "item": item,
        }

    @pytest.mark.asyncio
    async def test_webhook_calls_reingest_item_on_change(self):
        """When content changes, webhook calls reingest_item with correct args."""
        from unittest.mock import AsyncMock, patch

        from src.cms.webhook import _content_hashes, handle_cms_webhook

        _content_hashes.clear()

        payload = self._make_valid_payload()

        with patch(
            "src.rag.pipeline.reingest_item",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_reingest:
            result = await handle_cms_webhook(payload, webhook_secret="secret")

        assert result["status"] == "indexed"
        mock_reingest.assert_called_once_with(
            "dining",
            "dining-reindex-001",
            payload["item"],
            casino_id="mohegan_sun",
        )

    @pytest.mark.asyncio
    async def test_webhook_indexed_even_when_reingest_fails(self):
        """Hash is updated even if reingest_item fails (non-critical failure)."""
        from unittest.mock import AsyncMock, patch

        from src.cms.webhook import _content_hashes, handle_cms_webhook

        _content_hashes.clear()

        payload = self._make_valid_payload()

        with patch(
            "src.rag.pipeline.reingest_item",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await handle_cms_webhook(payload, webhook_secret="secret")

        # Status is still "indexed" (hash updated) even though re-indexing failed
        assert result["status"] == "indexed"
        # Hash was stored (subsequent identical call returns "unchanged")
        result2 = await handle_cms_webhook(payload, webhook_secret="secret")
        assert result2["status"] == "unchanged"

    @pytest.mark.asyncio
    async def test_webhook_does_not_call_reingest_for_unchanged(self):
        """Unchanged content does not trigger re-indexing."""
        from unittest.mock import AsyncMock, patch

        from src.cms.webhook import _content_hashes, handle_cms_webhook

        _content_hashes.clear()

        payload = self._make_valid_payload()

        with patch(
            "src.rag.pipeline.reingest_item",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_reingest:
            # First call: indexed
            await handle_cms_webhook(payload, webhook_secret="secret")
            mock_reingest.assert_called_once()
            mock_reingest.reset_mock()

            # Second call: unchanged — should NOT call reingest
            result = await handle_cms_webhook(payload, webhook_secret="secret")
            assert result["status"] == "unchanged"
            mock_reingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_webhook_does_not_call_reingest_for_quarantined(self):
        """Quarantined items do not trigger re-indexing."""
        from unittest.mock import AsyncMock, patch

        from src.cms.webhook import handle_cms_webhook

        payload = {
            "casino_id": "mohegan_sun",
            "category": "dining",
            "item": {
                "id": "bad-item",
                "name": "Bad Item",
                # Missing required fields -> quarantined
            },
        }

        with patch(
            "src.rag.pipeline.reingest_item",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_reingest:
            result = await handle_cms_webhook(payload, webhook_secret="secret")

        assert result["status"] == "quarantined"
        mock_reingest.assert_not_called()
