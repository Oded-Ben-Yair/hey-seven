"""SMS webhook handler with HMAC-SHA256 signature verification.

Provides webhook signature verification (HMAC-SHA256 placeholder for
Telnyx Ed25519 -- will be upgraded to Ed25519 when production webhook
signing keys are provisioned), message parsing, TCPA compliance checking,
and idempotency tracking.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from typing import Any

from cachetools import TTLCache

from .compliance import handle_mandatory_keywords

logger = logging.getLogger(__name__)

# Module-level idempotency tracker for deduplicating Telnyx webhook retries.
# Instantiated here so all inbound SMS requests share the same tracker.
_idempotency_tracker: WebhookIdempotencyTracker | None = None


def _get_idempotency_tracker() -> WebhookIdempotencyTracker:
    """Lazy singleton for the idempotency tracker."""
    global _idempotency_tracker  # noqa: PLW0603
    if _idempotency_tracker is None:
        _idempotency_tracker = WebhookIdempotencyTracker()
    return _idempotency_tracker


# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------


async def verify_webhook_signature(
    request_body: bytes,
    signature: str,
    timestamp: str,
    public_key: str,
) -> bool:
    """Verify a Telnyx webhook signature.

    Currently uses HMAC-SHA256 as a placeholder implementation. The Telnyx
    webhook spec uses ED25519 signatures, but that requires the ``cryptography``
    package which is not in the dependency list.

    .. todo::
        Replace with proper Ed25519 verification when the ``cryptography``
        package or ``telnyx`` SDK is available::

            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key))
            key.verify(bytes.fromhex(signature), signed_payload)

    Args:
        request_body: Raw webhook request body bytes.
        signature: Value of the ``telnyx-signature-ed25519`` header.
        timestamp: Value of the ``telnyx-timestamp`` header.
        public_key: Telnyx public key for verification.

    Returns:
        ``True`` if the signature is valid.
    """
    if not signature or not timestamp or not public_key:
        logger.warning("Missing signature, timestamp, or public key")
        return False

    # Replay protection: reject webhooks older than 5 minutes
    try:
        ts_int = int(timestamp)
    except (ValueError, TypeError):
        logger.warning("Invalid timestamp format: %s", timestamp)
        return False

    tolerance = 300  # 5 minutes
    if abs(int(time.time()) - ts_int) > tolerance:
        logger.warning("Webhook timestamp expired: %s", timestamp)
        return False

    # HMAC-SHA256 verification (placeholder for Ed25519)
    signed_payload = f"{timestamp}.{request_body.decode('utf-8', errors='replace')}"
    expected = hmac.new(
        public_key.encode(),
        signed_payload.encode(),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Inbound SMS handling
# ---------------------------------------------------------------------------


async def handle_inbound_sms(payload: dict[str, Any]) -> dict[str, Any]:
    """Process an inbound SMS webhook payload from Telnyx.

    Extracts the message fields, checks for mandatory STOP/HELP/START
    keywords FIRST (before any LLM processing), and returns either a
    keyword response or the parsed message for agent processing.

    Args:
        payload: Telnyx webhook ``data.payload`` dict.

    Returns:
        A dict with:
        - ``type``: ``"keyword_response"`` or ``"message"``
        - ``from_``: Sender phone (E.164)
        - ``to``: Destination phone (E.164)
        - ``text``: Inbound message text
        - ``message_id``: Telnyx message UUID
        - ``media_urls``: List of MMS media URLs (may be empty)
        - ``keyword_response``: Canned response text (only if ``type == "keyword_response"``)
    """
    from_number: str = payload.get("from", {}).get("phone_number", "")
    to_number: str = payload.get("to", [{}])[0].get("phone_number", "") if isinstance(
        payload.get("to"), list
    ) else payload.get("to", {}).get("phone_number", "")
    text: str = payload.get("text", "")
    message_id: str = payload.get("id", "")
    media_urls: list[str] = [m.get("url", "") for m in payload.get("media", []) if m.get("url")]

    logger.info(
        "Inbound SMS from=%s to=%s len=%d message_id=%s",
        from_number[-4:] if from_number else "????",
        to_number[-4:] if to_number else "????",
        len(text),
        message_id[:12] if message_id else "none",
    )

    # Idempotency: deduplicate Telnyx webhook retries (4xx/5xx/timeout).
    if message_id and await _get_idempotency_tracker().is_duplicate(message_id):
        logger.info("Duplicate webhook skipped: %s", message_id[:12])
        return {
            "type": "duplicate",
            "from_": from_number,
            "to": to_number,
            "text": text,
            "message_id": message_id,
            "media_urls": media_urls,
        }

    # MANDATORY: Check keywords BEFORE any LLM call
    keyword_response = await handle_mandatory_keywords(text, from_number)
    if keyword_response is not None:
        return {
            "type": "keyword_response",
            "from_": from_number,
            "to": to_number,
            "text": text,
            "message_id": message_id,
            "media_urls": media_urls,
            "keyword_response": keyword_response,
        }

    return {
        "type": "message",
        "from_": from_number,
        "to": to_number,
        "text": text,
        "message_id": message_id,
        "media_urls": media_urls,
    }


# ---------------------------------------------------------------------------
# Delivery receipt handling
# ---------------------------------------------------------------------------

# Track delivery statuses for monitoring
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "delivered",
        "sending_failed",
        "delivery_failed",
        "delivery_unconfirmed",
    }
)

# Bounded delivery log: TTLCache prevents unbounded memory growth in
# long-running processes.  maxsize=10000 caps tracked messages; TTL=86400
# (24h) auto-expires old entries.  OrderedDict-based LRU eviction when full.
_DELIVERY_LOG: TTLCache = TTLCache(maxsize=10000, ttl=86400)


async def handle_delivery_receipt(payload: dict[str, Any]) -> None:
    """Process a Telnyx delivery receipt (DLR) webhook.

    Logs the delivery status and updates the in-memory delivery log.
    In production this would persist to Firestore and trigger retry
    logic for failed deliveries.

    Args:
        payload: Telnyx DLR webhook ``data.payload`` dict.
    """
    message_id: str = payload.get("id", "")
    to_list = payload.get("to", [])

    if isinstance(to_list, list) and to_list:
        status: str = to_list[0].get("status", "unknown")
        address: str = to_list[0].get("address", "")
    else:
        status = "unknown"
        address = ""

    _DELIVERY_LOG[message_id] = {
        "status": status,
        "address": address,
        "received_at": time.time(),
    }

    if status == "delivered":
        logger.info("SMS delivered: message_id=%s to=%s", message_id[:12], address[-4:])
    elif status in ("sending_failed", "delivery_failed"):
        logger.warning(
            "SMS delivery failed: message_id=%s to=%s status=%s",
            message_id[:12],
            address[-4:],
            status,
        )
    elif status == "delivery_unconfirmed":
        logger.info(
            "SMS delivery unconfirmed: message_id=%s to=%s",
            message_id[:12],
            address[-4:],
        )
    else:
        logger.debug("SMS DLR: message_id=%s status=%s", message_id[:12], status)


def get_delivery_status(message_id: str) -> str | None:
    """Look up the most recent delivery status for a message.

    Returns:
        Status string or ``None`` if no DLR has been received.
    """
    entry = _DELIVERY_LOG.get(message_id)
    return entry["status"] if entry else None


# ---------------------------------------------------------------------------
# Idempotency tracker
# ---------------------------------------------------------------------------


class WebhookIdempotencyTracker:
    """In-memory de-duplication for Telnyx webhook retries.

    Telnyx retries webhooks on 4xx/5xx and timeouts, which can cause
    duplicate processing. This tracker stores processed message IDs
    with a configurable TTL and cleans up expired entries periodically.

    Thread-safe via ``asyncio.Lock``.

    Args:
        ttl_seconds: How long to remember a message ID (default 3600 = 1h).
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        self._processed: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_duplicate(self, message_id: str) -> bool:
        """Check if a message ID has already been processed.

        Also triggers cleanup of expired entries.

        Args:
            message_id: Telnyx webhook message UUID.

        Returns:
            ``True`` if the message has already been seen within the TTL.
        """
        async with self._lock:
            await self._cleanup()
            if message_id in self._processed:
                logger.debug("Duplicate webhook detected: %s", message_id[:12])
                return True
            self._processed[message_id] = time.time()
            return False

    async def _cleanup(self) -> None:
        """Remove expired entries older than TTL."""
        cutoff = time.time() - self._ttl
        expired = [mid for mid, ts in self._processed.items() if ts < cutoff]
        for mid in expired:
            del self._processed[mid]
        if expired:
            logger.debug("Cleaned up %d expired idempotency entries", len(expired))

    @property
    def size(self) -> int:
        """Current number of tracked message IDs."""
        return len(self._processed)
