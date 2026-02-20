"""CMS webhook handler for Google Sheets content updates.

Receives POST from Google Apps Script when a casino staff member edits content.
Validates the payload, checks HMAC signature with replay protection, validates
the item, and triggers re-indexing of the changed item.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

from cachetools import TTLCache

from .validation import validate_details_json, validate_item

logger = logging.getLogger(__name__)

# Replay protection: reject webhooks with timestamps older than this (seconds).
_REPLAY_TOLERANCE_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------


def verify_webhook_signature(
    payload_bytes: bytes,
    signature: str,
    secret: str,
    timestamp: str | None = None,
) -> bool:
    """Verify HMAC-SHA256 signature from the CMS webhook with replay protection.

    The Google Apps Script ``onEdit`` trigger signs the POST body with
    a shared secret using HMAC-SHA256 and includes a timestamp header.

    Replay protection: if ``timestamp`` is provided, reject requests
    where ``abs(now - timestamp) > 300 seconds`` (5-minute window).
    This mirrors the SMS webhook's Ed25519 replay protection pattern.

    Args:
        payload_bytes: Raw request body bytes.
        signature: Value of the ``X-Webhook-Signature`` header.
        secret: Shared webhook secret (from ``CMS_WEBHOOK_SECRET`` config).
        timestamp: Value of the ``X-Webhook-Timestamp`` header (Unix epoch string).

    Returns:
        ``True`` if the signature is valid and within the replay window.
    """
    if not signature or not secret:
        logger.warning("Missing webhook signature or secret")
        return False

    # Replay protection: validate timestamp freshness
    if timestamp:
        try:
            ts_int = int(timestamp)
            if abs(int(time.time()) - ts_int) > _REPLAY_TOLERANCE_SECONDS:
                logger.warning(
                    "CMS webhook timestamp outside tolerance: ts=%d now=%d",
                    ts_int,
                    int(time.time()),
                )
                return False
        except (ValueError, TypeError):
            logger.warning("CMS webhook invalid timestamp: %r", timestamp)
            return False

    expected = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Re-indexing helper
# ---------------------------------------------------------------------------

# In-memory hash store for change detection.  In production this would be
# backed by Firestore.  Bounded via TTLCache to prevent unbounded memory
# growth in long-running containers receiving frequent CMS updates.
# maxsize=10_000 covers ~10K distinct items; TTL=86400 (24h) ensures
# stale hashes are evicted even without container restart.
_CONTENT_HASH_MAXSIZE = 10_000
_CONTENT_HASH_TTL = 86400  # 24 hours
_content_hashes: TTLCache[str, str] = TTLCache(
    maxsize=_CONTENT_HASH_MAXSIZE, ttl=_CONTENT_HASH_TTL
)


def _compute_item_hash(item: dict[str, Any], source: str) -> str:
    """Compute a SHA-256 content hash for a single item.

    Follows the same pattern as ``src.rag.pipeline.ingest_property`` which
    uses ``hashlib.sha256((text + source).encode()).hexdigest()``.

    Args:
        item: The content item dict.
        source: Source identifier (e.g., ``"cms:{casino_id}:{category}"``).

    Returns:
        Hex-encoded SHA-256 hash.
    """
    canonical = json.dumps(item, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(
        (canonical + source).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Main webhook handler
# ---------------------------------------------------------------------------


async def handle_cms_webhook(
    payload: dict[str, Any],
    webhook_secret: str,
    *,
    raw_body: bytes | None = None,
    signature: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Process a CMS webhook from Google Apps Script.

    Workflow:
        1. Verify HMAC-SHA256 signature with replay protection
           (if ``raw_body`` and ``signature`` provided).
        2. Extract ``casino_id``, ``category``, and ``item`` from the payload.
        3. Validate the item via :func:`validate_item`.
        4. Compute SHA-256 content hash and compare with stored hash.
        5. If hash changed, mark the item for re-indexing.

    Args:
        payload: Parsed JSON body from the webhook request.
        webhook_secret: Shared HMAC secret for signature verification.
        raw_body: Raw request body bytes (for signature verification).
        signature: Value of the ``X-Webhook-Signature`` header.
        timestamp: Value of the ``X-Webhook-Timestamp`` header (Unix epoch string).

    Returns:
        A status dict::

            {
                "status": "indexed" | "unchanged" | "quarantined" | "rejected",
                "item_id": "<item id>",
                "errors": [...]  # only when quarantined
            }
    """
    # Step 1: Signature verification with replay protection
    if raw_body is not None and signature is not None:
        if not verify_webhook_signature(raw_body, signature, webhook_secret, timestamp=timestamp):
            logger.warning("CMS webhook signature verification failed")
            return {"status": "rejected", "item_id": None, "errors": ["invalid signature"]}

    # Step 2: Extract required fields
    casino_id: str = payload.get("casino_id", "")
    category: str = payload.get("category", "")
    item: dict[str, Any] | None = payload.get("item")

    if not casino_id:
        logger.warning("CMS webhook missing casino_id")
        return {"status": "rejected", "item_id": None, "errors": ["missing casino_id"]}

    if not category:
        logger.warning("CMS webhook missing category")
        return {"status": "rejected", "item_id": None, "errors": ["missing category"]}

    if item is None or not isinstance(item, dict):
        logger.warning("CMS webhook missing or invalid item")
        return {"status": "rejected", "item_id": None, "errors": ["missing or invalid item"]}

    item_id: str = item.get("id", item.get("name", "<unknown>"))

    # Step 3: Validate item
    is_valid, validation_errors = validate_item(item, category)

    # Also validate details_json if present
    details_valid, details_errors = validate_details_json(item)
    if not details_valid:
        is_valid = False
        validation_errors.extend(details_errors)

    if not is_valid:
        logger.warning(
            "CMS webhook item quarantined: casino=%s category=%s item_id=%s errors=%s",
            casino_id,
            category,
            item_id,
            validation_errors,
        )
        return {
            "status": "quarantined",
            "item_id": item_id,
            "errors": validation_errors,
        }

    # Step 4: Content hash for change detection
    source = f"cms:{casino_id}:{category}"
    new_hash = _compute_item_hash(item, source)
    hash_key = f"{casino_id}:{category}:{item_id}"
    stored_hash = _content_hashes.get(hash_key)

    if stored_hash == new_hash:
        logger.info(
            "CMS item unchanged: casino=%s category=%s item_id=%s",
            casino_id,
            category,
            item_id,
        )
        return {"status": "unchanged", "item_id": item_id}

    # Step 5: Mark for re-indexing (store new hash)
    _content_hashes[hash_key] = new_hash

    logger.info(
        "CMS item indexed: casino=%s category=%s item_id=%s hash=%s",
        casino_id,
        category,
        item_id,
        new_hash[:12],
    )
    return {"status": "indexed", "item_id": item_id}
