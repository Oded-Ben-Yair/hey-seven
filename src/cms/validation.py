"""Content validation for CMS items.

Each content category has required fields. Items failing validation are
quarantined (logged, not indexed) rather than silently dropped.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Required fields per content category.
# Every item must have the base fields (id, name, description, active)
# plus any category-specific fields listed here.
_BASE_FIELDS: tuple[str, ...] = ("id", "name", "description", "active")

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "dining": _BASE_FIELDS + ("cuisine", "price_range"),
    "entertainment": _BASE_FIELDS + ("event_type", "venue"),
    "spa": _BASE_FIELDS + ("treatment_type", "duration"),
    "gaming": _BASE_FIELDS + ("game_type", "location"),
    "promotions": _BASE_FIELDS + ("value", "valid_until"),
    "regulations": _BASE_FIELDS + ("state", "effective_date"),
    "hours": _BASE_FIELDS + ("weekday_hours",),
    "general_info": _BASE_FIELDS,
}


def validate_item(
    item: dict[str, Any],
    category: str,
) -> tuple[bool, list[str]]:
    """Validate a single CMS content item against category requirements.

    Items that fail validation are *quarantined* -- the errors are returned
    so the caller can log them and skip indexing without silently dropping
    content.

    Args:
        item: The content item dict (keys from Google Sheets header row).
        category: One of the ``CONTENT_CATEGORIES`` values.

    Returns:
        A ``(is_valid, errors)`` tuple.  ``errors`` is an empty list when
        the item is valid.
    """
    errors: list[str] = []

    required = REQUIRED_FIELDS.get(category)
    if required is None:
        errors.append(f"Unknown category: {category}")
        logger.warning("Quarantined item: unknown category '%s'", category)
        return False, errors

    for field in required:
        value = item.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"Missing required field: {field}")

    if errors:
        item_id = item.get("id", item.get("name", "<unknown>"))
        logger.warning(
            "Quarantined item id=%s category=%s errors=%s",
            item_id,
            category,
            errors,
        )

    return len(errors) == 0, errors


def validate_details_json(item: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate the ``details_json`` column if present.

    Some sheet rows include a ``details_json`` column containing
    supplemental structured data (e.g., menu items, schedule details).
    This function verifies that the value is valid JSON.

    Args:
        item: The content item dict.

    Returns:
        A ``(is_valid, errors)`` tuple.  Returns ``(True, [])`` when
        there is no ``details_json`` field.
    """
    raw = item.get("details_json")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return True, []

    if isinstance(raw, str):
        try:
            json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            error_msg = f"Invalid details_json: {exc}"
            item_id = item.get("id", item.get("name", "<unknown>"))
            logger.warning(
                "Quarantined item id=%s: %s", item_id, error_msg
            )
            return False, [error_msg]

    return True, []
