"""Telnyx SMS client using raw HTTP via httpx.

Sends SMS messages through the Telnyx v2 REST API, handles GSM-7 vs UCS-2
encoding detection, and provides smart message segmentation at word boundaries.

No Telnyx SDK dependency -- uses httpx.AsyncClient for direct API calls.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# GSM 03.38 Basic Character Set (single-shift table).
# Extended chars ({, }, [, ], \, |, ^, ~, euro sign) each consume 2 septets.
_GSM7_BASIC: frozenset[str] = frozenset(
    "@\u00a3$\u00a5\u00e8\u00e9\u00f9\u00ec\u00f2\u00c7\n\u00d8\u00f8\r\u00c5\u00e5"
    "\u0394_\u03a6\u0393\u039b\u03a9\u03a0\u03a8\u03a3\u0398\u039e"
    "\x1b\u00c6\u00e6\u00df\u00c9 !\"#\u00a4%&'()*+,-./0123456789:;<=>?"
    "\u00a1ABCDEFGHIJKLMNOPQRSTUVWXYZ\u00c4\u00d6\u00d1\u00dc\u00a7"
    "\u00bfabcdefghijklmnopqrstuvwxyz\u00e4\u00f6\u00f1\u00fc\u00e0"
)

_GSM7_EXTENSION: frozenset[str] = frozenset("{}[]\\|^~\u20ac")

TELNYX_API_BASE = "https://api.telnyx.com/v2"


class TelnyxSMSClient:
    """Async SMS client for the Telnyx v2 REST API.

    Args:
        api_key: Telnyx API v2 key.
        messaging_profile_id: Telnyx messaging profile UUID.
        base_url: Override for testing; defaults to Telnyx v2 production API.
    """

    def __init__(
        self,
        api_key: str,
        messaging_profile_id: str,
        *,
        base_url: str = TELNYX_API_BASE,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._messaging_profile_id = messaging_profile_id
        self._base_url = base_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(10.0),
        )
        self._owns_client = http_client is None

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    # -- Public API ----------------------------------------------------------

    async def send_message(self, to: str, from_: str, text: str) -> dict[str, Any]:
        """Send an SMS message via Telnyx v2 API.

        Args:
            to: Destination phone number in E.164 format (e.g. ``+12125551234``).
            from_: Sender phone number in E.164 format.
            text: Message body.

        Returns:
            Telnyx API response ``data`` dict containing ``id``, ``to``,
            ``from``, ``text``, and ``record_type`` fields.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses from Telnyx.
        """
        payload: dict[str, Any] = {
            "from": from_,
            "to": to,
            "text": text,
            "messaging_profile_id": self._messaging_profile_id,
        }

        encoding, max_chars = self.detect_encoding(text)
        logger.info(
            "Sending SMS to=%s encoding=%s chars=%d",
            to[-4:],
            encoding,
            len(text),
        )

        response = await self._client.post("/messages", json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json().get("data", response.json())
        logger.info("SMS sent message_id=%s", data.get("id", "unknown"))
        return data

    async def check_delivery_status(self, message_id: str) -> str:
        """Check delivery receipt status for a sent message.

        Args:
            message_id: Telnyx message UUID returned from ``send_message``.

        Returns:
            Status string such as ``queued``, ``sent``, ``delivered``,
            ``sending_failed``, or ``delivery_failed``.
        """
        response = await self._client.get(f"/messages/{message_id}")
        response.raise_for_status()
        data = response.json().get("data", response.json())
        status: str = data.get("to", [{}])[0].get("status", "unknown")
        return status

    # -- Encoding & segmentation helpers ------------------------------------

    @staticmethod
    def detect_encoding(text: str) -> tuple[str, int]:
        """Detect whether *text* fits GSM-7 or requires UCS-2 encoding.

        Returns:
            A ``(encoding, max_chars_per_segment)`` tuple.
            ``("gsm7", 160)`` for plain ASCII/GSM text, or
            ``("ucs2", 70)`` for unicode text.
        """
        for char in text:
            if char not in _GSM7_BASIC and char not in _GSM7_EXTENSION:
                return ("ucs2", 70)
        return ("gsm7", 160)

    @staticmethod
    def segment_message(text: str, max_chars: int = 0) -> list[str]:
        """Split a long message into SMS-sized segments at word boundaries.

        If *max_chars* is ``0``, the limit is auto-detected from the text
        encoding (160 for GSM-7, 70 for UCS-2).

        For multi-part messages the per-segment limit drops to account for
        the User Data Header (UDH): 153 for GSM-7, 67 for UCS-2.

        Args:
            text: Full message text.
            max_chars: Override per-segment character limit.

        Returns:
            List of message segments.
        """
        if not text:
            return []

        if max_chars <= 0:
            for char in text:
                if char not in _GSM7_BASIC and char not in _GSM7_EXTENSION:
                    max_chars = 70
                    break
            else:
                max_chars = 160

        if len(text) <= max_chars:
            return [text]

        # Multi-part: account for UDH overhead
        if max_chars == 160:
            max_chars = 153
        elif max_chars == 70:
            max_chars = 67

        segments: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= max_chars:
                segments.append(remaining)
                break

            # Prefer sentence boundary
            split_at = remaining[:max_chars].rfind(". ")
            if split_at > max_chars * 0.4:
                segments.append(remaining[: split_at + 1].strip())
                remaining = remaining[split_at + 2 :].strip()
                continue

            # Fall back to word boundary
            split_at = remaining[:max_chars].rfind(" ")
            if split_at > 0:
                segments.append(remaining[:split_at].strip())
                remaining = remaining[split_at + 1 :].strip()
            else:
                # No word boundary -- hard split (rare for English/Spanish)
                segments.append(remaining[:max_chars])
                remaining = remaining[max_chars:]

        return segments
