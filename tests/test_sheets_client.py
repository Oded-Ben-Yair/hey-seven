"""Tests for Google Sheets CMS client (src/cms/sheets_client.py).

Covers:
- SheetsClient initialization and lazy service creation
- read_category: valid tab, unknown category, empty sheet, API errors
- read_all_categories: iterates all tabs
- compute_content_hash: deterministic SHA-256
"""

from unittest.mock import MagicMock, patch

import pytest

from src.cms.sheets_client import (
    CONTENT_CATEGORIES,
    SheetsClient,
    compute_content_hash,
)


class TestSheetsClientInit:
    """SheetsClient initialization and lazy service creation."""

    def test_init_defaults(self):
        """SheetsClient initializes with no credentials and no service."""
        client = SheetsClient()
        assert client._credentials is None
        assert client._service is None
        assert client._available is None

    def test_init_with_credentials(self):
        """SheetsClient stores provided credentials."""
        creds = MagicMock()
        client = SheetsClient(credentials=creds)
        assert client._credentials is creds

    def test_ensure_service_no_google_lib(self):
        """When google-api-python-client is missing, service is unavailable."""
        client = SheetsClient()
        with patch.dict("sys.modules", {"googleapiclient": None, "googleapiclient.discovery": None}):
            # Force re-check by resetting _available
            client._available = None
            result = client._ensure_service()
        assert result is False
        assert client._available is False

    def test_ensure_service_cached_after_first_call(self):
        """_ensure_service returns cached result on subsequent calls."""
        client = SheetsClient()
        client._available = True  # Simulate successful init
        client._service = MagicMock()
        assert client._ensure_service() is True


class TestReadCategory:
    """SheetsClient.read_category with mocked Sheets API."""

    @pytest.mark.asyncio
    async def test_unknown_category_returns_empty(self):
        """Unknown category name returns empty list without API call."""
        client = SheetsClient()
        result = await client.read_category("sheet-id", "nonexistent_category")
        assert result == []

    @pytest.mark.asyncio
    async def test_service_unavailable_returns_empty(self):
        """When service is unavailable, returns empty list."""
        client = SheetsClient()
        client._available = False
        result = await client.read_category("sheet-id", "dining")
        assert result == []

    @pytest.mark.asyncio
    async def test_successful_read_returns_dicts(self):
        """Successful API call returns list of dicts keyed by header row."""
        client = SheetsClient()
        client._available = True

        # Mock the Sheets API chain
        mock_service = MagicMock()
        mock_execute = MagicMock(return_value={
            "values": [
                ["Name", "Cuisine", "Hours"],
                ["Todd English's Tuscany", "Italian", "5-10 PM"],
                ["Bobby's Burger Palace", "American", "11 AM-11 PM"],
            ]
        })
        mock_service.spreadsheets().values().get().execute = mock_execute
        client._service = mock_service

        result = await client.read_category("sheet-id", "dining")

        assert len(result) == 2
        assert result[0]["name"] == "Todd English's Tuscany"
        assert result[0]["cuisine"] == "Italian"
        assert result[1]["hours"] == "11 AM-11 PM"

    @pytest.mark.asyncio
    async def test_empty_sheet_returns_empty(self):
        """Sheet with only header row (no data) returns empty list."""
        client = SheetsClient()
        client._available = True

        mock_service = MagicMock()
        mock_service.spreadsheets().values().get().execute = MagicMock(
            return_value={"values": [["Name", "Cuisine"]]}  # Header only
        )
        client._service = mock_service

        result = await client.read_category("sheet-id", "dining")
        assert result == []

    @pytest.mark.asyncio
    async def test_api_error_returns_empty(self):
        """API exception returns empty list (graceful degradation)."""
        client = SheetsClient()
        client._available = True

        mock_service = MagicMock()
        mock_service.spreadsheets().values().get().execute = MagicMock(
            side_effect=RuntimeError("API quota exceeded")
        )
        client._service = mock_service

        result = await client.read_category("sheet-id", "dining")
        assert result == []

    @pytest.mark.asyncio
    async def test_short_row_pads_with_empty_string(self):
        """Rows shorter than header are padded with empty strings."""
        client = SheetsClient()
        client._available = True

        mock_service = MagicMock()
        mock_service.spreadsheets().values().get().execute = MagicMock(
            return_value={
                "values": [
                    ["Name", "Cuisine", "Hours"],
                    ["Buffet"],  # Missing cuisine and hours
                ]
            }
        )
        client._service = mock_service

        result = await client.read_category("sheet-id", "dining")
        assert len(result) == 1
        assert result[0]["name"] == "Buffet"
        assert result[0]["cuisine"] == ""
        assert result[0]["hours"] == ""


class TestReadAllCategories:
    """SheetsClient.read_all_categories covers all tabs."""

    @pytest.mark.asyncio
    async def test_reads_all_content_categories(self):
        """read_all_categories iterates all CONTENT_CATEGORIES."""
        client = SheetsClient()
        client._available = False  # No service — each read returns []

        result = await client.read_all_categories("sheet-id")

        assert set(result.keys()) == set(CONTENT_CATEGORIES)
        for cat in CONTENT_CATEGORIES:
            assert result[cat] == []


class TestComputeContentHash:
    """Deterministic SHA-256 hash for change detection."""

    def test_same_input_produces_same_hash(self):
        """Identical inputs produce the same hash (idempotent)."""
        items = [{"name": "A", "value": "1"}, {"name": "B", "value": "2"}]
        h1 = compute_content_hash(items)
        h2 = compute_content_hash(items)
        assert h1 == h2

    def test_different_input_produces_different_hash(self):
        """Different inputs produce different hashes."""
        h1 = compute_content_hash([{"name": "A"}])
        h2 = compute_content_hash([{"name": "B"}])
        assert h1 != h2

    def test_empty_list_produces_consistent_hash(self):
        """Empty input list produces a consistent hash."""
        h1 = compute_content_hash([])
        h2 = compute_content_hash([])
        assert h1 == h2

    def test_hash_is_hex_sha256(self):
        """Hash is a 64-character hex string (SHA-256)."""
        h = compute_content_hash([{"name": "test"}])
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_dict_key_order_independent(self):
        """Hash is the same regardless of dict key insertion order."""
        h1 = compute_content_hash([{"z": 1, "a": 2}])
        h2 = compute_content_hash([{"a": 2, "z": 1}])
        assert h1 == h2
