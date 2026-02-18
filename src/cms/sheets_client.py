"""Google Sheets CMS client for reading casino content.

Each casino has one Google Sheets spreadsheet with tabs for each content category:
Dining | Entertainment | Spa | Gaming | Promotions | Regulations | Hours | General Info

When Sheets API is unavailable (local dev), falls back to loading from
the local JSON file at PROPERTY_DATA_PATH.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

CONTENT_CATEGORIES: tuple[str, ...] = (
    "dining",
    "entertainment",
    "spa",
    "gaming",
    "promotions",
    "regulations",
    "hours",
    "general_info",
)


class SheetsClient:
    """Google Sheets API client for reading casino content.

    Lazily imports ``googleapiclient.discovery`` to avoid requiring the
    dependency in local development.  When the library is unavailable,
    all read operations return empty lists.

    Args:
        credentials: Optional Google API credentials object.  When ``None``,
            the client attempts Application Default Credentials (ADC).
    """

    def __init__(self, credentials: Any = None) -> None:
        self._credentials = credentials
        self._service: Any = None
        self._available: bool | None = None

    def _ensure_service(self) -> bool:
        """Lazily create the Sheets API service.

        Returns:
            ``True`` if the service is available, ``False`` otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            from googleapiclient.discovery import build  # type: ignore[import-untyped]

            kwargs: dict[str, Any] = {"serviceName": "sheets", "version": "v4"}
            if self._credentials:
                kwargs["credentials"] = self._credentials
            self._service = build(**kwargs)
            self._available = True
            logger.info("Google Sheets API service initialized.")
        except ImportError:
            logger.info(
                "google-api-python-client not installed. "
                "Sheets API unavailable; returning empty results."
            )
            self._available = False
        except Exception:
            logger.warning(
                "Failed to initialize Google Sheets API service.",
                exc_info=True,
            )
            self._available = False

        return self._available

    async def read_category(
        self,
        sheet_id: str,
        category: str,
    ) -> list[dict[str, Any]]:
        """Read all rows from a single sheet tab (category).

        The first row is treated as the header.  Each subsequent row becomes
        a dict keyed by the header values.

        Args:
            sheet_id: Google Sheets spreadsheet ID.
            category: Tab name (must be one of ``CONTENT_CATEGORIES``).

        Returns:
            A list of dicts, one per row.  Returns an empty list when the
            Sheets API is unavailable or the tab is empty/missing.
        """
        if category not in CONTENT_CATEGORIES:
            logger.warning("Unknown content category: %s", category)
            return []

        if not self._ensure_service():
            return []

        try:
            # Tab name uses title case (e.g., "Dining", "General Info")
            tab_name = category.replace("_", " ").title()
            result = (
                self._service.spreadsheets()
                .values()
                .get(spreadsheetId=sheet_id, range=f"{tab_name}!A:P")
                .execute()
            )
            rows: list[list[str]] = result.get("values", [])
            if len(rows) < 2:
                return []

            headers = [h.strip().lower().replace(" ", "_") for h in rows[0]]
            items: list[dict[str, Any]] = []
            for row in rows[1:]:
                item: dict[str, Any] = {}
                for i, header in enumerate(headers):
                    item[header] = row[i].strip() if i < len(row) else ""
                items.append(item)

            logger.info(
                "Read %d items from sheet=%s category=%s",
                len(items),
                sheet_id[:12],
                category,
            )
            return items

        except Exception:
            logger.warning(
                "Failed to read sheet=%s category=%s",
                sheet_id[:12],
                category,
                exc_info=True,
            )
            return []

    async def read_all_categories(
        self,
        sheet_id: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Read all content category tabs from a spreadsheet.

        Args:
            sheet_id: Google Sheets spreadsheet ID.

        Returns:
            A dict mapping category name to list of row dicts.
        """
        results: dict[str, list[dict[str, Any]]] = {}
        for category in CONTENT_CATEGORIES:
            results[category] = await self.read_category(sheet_id, category)
        return results


def compute_content_hash(items: list[dict[str, Any]]) -> str:
    """Compute a deterministic SHA-256 hash over a list of content items.

    Used for change detection: if the hash matches the previously stored
    value, the content has not changed and re-indexing can be skipped.

    Args:
        items: List of content item dicts.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    # Sort keys for deterministic serialization regardless of dict ordering
    canonical = json.dumps(items, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
