"""Tests for the SMS package: TCPA compliance, formatting, and webhook handling.

Covers GSM-7/UCS-2 encoding detection, message segmentation, keyword handling
(EN + ES), quiet hours with timezone support, consent hash chain integrity,
consent level checking, Ed25519 webhook signature verification, delivery receipts,
inbound SMS parsing, idempotency tracking, and config fields.

Mock-based tests (TelnyxSMSClient.send_message, check_delivery_status,
InboundSmsIdempotency with mock tracker) removed (mock purge R111).
"""

import time
from datetime import datetime, timezone

import pytest


# ============================================================================
# TelnyxSMSClient encoding tests
# ============================================================================


class TestTelnyxSMSClientEncoding:
    """GSM-7 vs UCS-2 encoding detection."""

    def test_ascii_text_is_gsm7(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        encoding, max_chars = TelnyxSMSClient.detect_encoding("Hello world!")
        assert encoding == "gsm7"
        assert max_chars == 160

    def test_basic_punctuation_is_gsm7(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        encoding, _ = TelnyxSMSClient.detect_encoding(
            "Price: $50 (call 1-800-555-1234)"
        )
        assert encoding == "gsm7"

    def test_emoji_triggers_ucs2(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        encoding, max_chars = TelnyxSMSClient.detect_encoding("Hello! \U0001f600")
        assert encoding == "ucs2"
        assert max_chars == 70

    def test_spanish_accents_trigger_ucs2(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        encoding, _ = TelnyxSMSClient.detect_encoding(
            "\u00bf Hola, c\u00f3mo est\u00e1s?"
        )
        assert encoding == "ucs2"

    def test_gsm7_extended_chars(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        encoding, _ = TelnyxSMSClient.detect_encoding("code: {test}")
        assert encoding == "gsm7"

    def test_empty_string_is_gsm7(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        encoding, max_chars = TelnyxSMSClient.detect_encoding("")
        assert encoding == "gsm7"
        assert max_chars == 160

    def test_chinese_characters_trigger_ucs2(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        encoding, _ = TelnyxSMSClient.detect_encoding("\u4f60\u597d")
        assert encoding == "ucs2"


class TestTelnyxSMSClientSegmentation:
    """Message segmentation at word boundaries."""

    def test_short_message_single_segment(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        segments = TelnyxSMSClient.segment_message("Hello!")
        assert segments == ["Hello!"]

    def test_empty_string_returns_empty_list(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        segments = TelnyxSMSClient.segment_message("")
        assert segments == []

    def test_exactly_160_chars_single_segment(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        text = "A" * 160
        segments = TelnyxSMSClient.segment_message(text)
        assert len(segments) == 1
        assert segments[0] == text

    def test_long_message_splits_into_multiple_segments(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        # 400 chars of words
        text = ("Hello world. " * 35).strip()
        segments = TelnyxSMSClient.segment_message(text)
        assert len(segments) > 1
        # Each segment should not exceed 153 chars (multi-part GSM-7 limit)
        for seg in segments:
            assert len(seg) <= 153

    def test_segmentation_prefers_sentence_boundaries(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        text = "First sentence here. Second sentence follows. " * 5
        segments = TelnyxSMSClient.segment_message(text.strip())
        # Verify split happens at period
        for seg in segments[:-1]:
            assert seg.endswith(".")

    def test_unicode_segmentation_uses_67_char_limit(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        # 150 emoji chars -> should split at UCS-2 multi-segment limit (67)
        text = "\U0001f600" * 150
        segments = TelnyxSMSClient.segment_message(text)
        assert len(segments) > 1
        for seg in segments:
            assert len(seg) <= 67

    def test_explicit_max_chars_override(self):
        from src.sms.telnyx_client import TelnyxSMSClient

        text = "A " * 50  # 100 chars of "A "
        segments = TelnyxSMSClient.segment_message(text.strip(), max_chars=30)
        assert len(segments) > 1
        for seg in segments:
            assert len(seg) <= 30


# ============================================================================
# Webhook tests (deterministic)
# ============================================================================


class TestWebhookSignatureVerification:
    """Ed25519 signature verification (Telnyx webhook spec)."""

    @pytest.mark.asyncio
    async def test_verify_webhook_signature_valid_ed25519(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        from src.sms.webhook import verify_webhook_signature

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        public_key_hex = public_key.public_bytes_raw().hex()

        body = b'{"data": "test"}'
        timestamp = str(int(time.time()))
        signed_payload = f"{timestamp}.{body.decode()}".encode()
        signature_bytes = private_key.sign(signed_payload)
        signature_hex = signature_bytes.hex()

        result = await verify_webhook_signature(
            body, signature_hex, timestamp, public_key_hex
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_webhook_signature_invalid_signature(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        from src.sms.webhook import verify_webhook_signature

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        public_key_hex = public_key.public_bytes_raw().hex()

        body = b'{"data": "test"}'
        timestamp = str(int(time.time()))

        wrong_payload = b"wrong-payload"
        wrong_sig = private_key.sign(wrong_payload).hex()

        result = await verify_webhook_signature(
            body, wrong_sig, timestamp, public_key_hex
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_webhook_signature_expired_timestamp(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        from src.sms.webhook import verify_webhook_signature

        private_key = Ed25519PrivateKey.generate()
        public_key_hex = private_key.public_key().public_bytes_raw().hex()

        body = b'{"data": "test"}'
        old_timestamp = str(int(time.time()) - 600)
        signed_payload = f"{old_timestamp}.{body.decode()}".encode()
        signature_hex = private_key.sign(signed_payload).hex()

        result = await verify_webhook_signature(
            body, signature_hex, old_timestamp, public_key_hex
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_webhook_signature_missing_params(self):
        from src.sms.webhook import verify_webhook_signature

        result = await verify_webhook_signature(b"body", "", "123", "aabbcc")
        assert result is False

        result = await verify_webhook_signature(b"body", "aabbcc", "", "aabbcc")
        assert result is False

        result = await verify_webhook_signature(b"body", "aabbcc", "123", "")
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_webhook_signature_invalid_hex(self):
        from src.sms.webhook import verify_webhook_signature

        timestamp = str(int(time.time()))

        result = await verify_webhook_signature(
            b"body", "aa" * 32, timestamp, "not-hex!"
        )
        assert result is False

        result = await verify_webhook_signature(
            b"body", "not-hex!", timestamp, "aa" * 32
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_timestamp_format_fails(self):
        from src.sms.webhook import verify_webhook_signature

        result = await verify_webhook_signature(b"body", "sig", "not-a-number", "key")
        assert result is False


class TestInboundSMS:
    """Inbound SMS webhook parsing."""

    @pytest.mark.asyncio
    async def test_normal_message_parsed(self):
        from src.sms.webhook import handle_inbound_sms

        payload = {
            "id": "msg-123",
            "from": {"phone_number": "+12125551234"},
            "to": [{"phone_number": "+18605559999"}],
            "text": "What restaurants are open?",
            "media": [],
        }
        result = await handle_inbound_sms(payload)
        assert result["type"] == "message"
        assert result["from_"] == "+12125551234"
        assert result["to"] == "+18605559999"
        assert result["text"] == "What restaurants are open?"
        assert result["message_id"] == "msg-123"
        assert result["media_urls"] == []

    @pytest.mark.asyncio
    async def test_stop_keyword_intercepted(self):
        from src.sms.webhook import handle_inbound_sms

        payload = {
            "id": "msg-456",
            "from": {"phone_number": "+12125551234"},
            "to": [{"phone_number": "+18605559999"}],
            "text": "STOP",
            "media": [],
        }
        result = await handle_inbound_sms(payload)
        assert result["type"] == "keyword_response"
        assert "unsubscribed" in result["keyword_response"].lower()

    @pytest.mark.asyncio
    async def test_media_urls_extracted(self):
        from src.sms.webhook import handle_inbound_sms

        payload = {
            "id": "msg-789",
            "from": {"phone_number": "+12125551234"},
            "to": [{"phone_number": "+18605559999"}],
            "text": "Check this out",
            "media": [
                {"url": "https://example.com/image.jpg", "content_type": "image/jpeg"},
            ],
        }
        result = await handle_inbound_sms(payload)
        assert result["media_urls"] == ["https://example.com/image.jpg"]


class TestDeliveryReceipts:
    """DLR webhook handling."""

    @pytest.mark.asyncio
    async def test_delivered_status_tracked(self):
        from src.sms.webhook import get_delivery_status, handle_delivery_receipt

        payload = {
            "id": "msg-dlr-001",
            "to": [{"status": "delivered", "address": "+12125551234"}],
        }
        await handle_delivery_receipt(payload)
        assert get_delivery_status("msg-dlr-001") == "delivered"

    @pytest.mark.asyncio
    async def test_failed_status_tracked(self):
        from src.sms.webhook import get_delivery_status, handle_delivery_receipt

        payload = {
            "id": "msg-dlr-002",
            "to": [{"status": "delivery_failed", "address": "+12125551234"}],
        }
        await handle_delivery_receipt(payload)
        assert get_delivery_status("msg-dlr-002") == "delivery_failed"

    @pytest.mark.asyncio
    async def test_unknown_message_returns_none(self):
        from src.sms.webhook import get_delivery_status

        assert get_delivery_status("nonexistent-id") is None

    def test_delivery_log_is_bounded(self):
        from cachetools import TTLCache

        from src.sms.webhook import _DELIVERY_LOG

        assert isinstance(_DELIVERY_LOG, TTLCache)
        assert _DELIVERY_LOG.maxsize == 10000


class TestIdempotencyTracker:
    """WebhookIdempotencyTracker de-duplication."""

    @pytest.mark.asyncio
    async def test_first_message_not_duplicate(self):
        from src.sms.webhook import WebhookIdempotencyTracker

        tracker = WebhookIdempotencyTracker()
        assert await tracker.is_duplicate("msg-001") is False

    @pytest.mark.asyncio
    async def test_same_message_is_duplicate(self):
        from src.sms.webhook import WebhookIdempotencyTracker

        tracker = WebhookIdempotencyTracker()
        await tracker.is_duplicate("msg-002")
        assert await tracker.is_duplicate("msg-002") is True

    @pytest.mark.asyncio
    async def test_different_messages_not_duplicate(self):
        from src.sms.webhook import WebhookIdempotencyTracker

        tracker = WebhookIdempotencyTracker()
        await tracker.is_duplicate("msg-003")
        assert await tracker.is_duplicate("msg-004") is False

    @pytest.mark.asyncio
    async def test_expired_entries_cleaned_up(self):
        from src.sms.webhook import WebhookIdempotencyTracker

        tracker = WebhookIdempotencyTracker(ttl_seconds=1)
        await tracker.is_duplicate("msg-005")
        assert tracker.size == 1

        # Manually expire the entry
        tracker._processed["msg-005"] = time.time() - 2

        # Next call triggers cleanup
        await tracker.is_duplicate("msg-006")
        assert "msg-005" not in tracker._processed
        assert tracker.size == 1  # Only msg-006


# ============================================================================
# Compliance tests
# ============================================================================


class TestMandatoryKeywords:
    """STOP/HELP/START keyword handling."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "keyword",
        ["stop", "STOP", "Stop", "stopall", "unsubscribe", "cancel", "end", "quit"],
    )
    async def test_stop_keywords_en(self, keyword):
        from src.sms.compliance import handle_mandatory_keywords

        result = await handle_mandatory_keywords(keyword, "+12125551234")
        assert result is not None
        assert "unsubscribed" in result.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("keyword", ["parar", "detener", "cancelar", "PARAR"])
    async def test_stop_keywords_es(self, keyword):
        from src.sms.compliance import handle_mandatory_keywords

        result = await handle_mandatory_keywords(keyword, "+12125551234")
        assert result is not None
        assert "unsubscribed" in result.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("keyword", ["help", "HELP", "info", "INFO"])
    async def test_help_keywords_en(self, keyword):
        from src.sms.compliance import handle_mandatory_keywords

        result = await handle_mandatory_keywords(keyword, "+12125551234")
        assert result is not None
        assert "stop" in result.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("keyword", ["ayuda", "informacion", "AYUDA"])
    async def test_help_keywords_es(self, keyword):
        from src.sms.compliance import handle_mandatory_keywords

        result = await handle_mandatory_keywords(keyword, "+12125551234")
        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "keyword", ["start", "START", "subscribe", "iniciar", "comenzar"]
    )
    async def test_start_keywords(self, keyword):
        from src.sms.compliance import handle_mandatory_keywords

        result = await handle_mandatory_keywords(keyword, "+12125551234")
        assert result is not None
        assert "resubscribed" in result.lower()

    @pytest.mark.asyncio
    async def test_non_keyword_returns_none(self):
        from src.sms.compliance import handle_mandatory_keywords

        result = await handle_mandatory_keywords(
            "What restaurants are open?", "+12125551234"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_keyword_with_whitespace(self):
        from src.sms.compliance import handle_mandatory_keywords

        result = await handle_mandatory_keywords("  STOP  ", "+12125551234")
        assert result is not None
        assert "unsubscribed" in result.lower()

    @pytest.mark.asyncio
    async def test_partial_keyword_not_matched(self):
        from src.sms.compliance import handle_mandatory_keywords

        result = await handle_mandatory_keywords("stopping by tonight", "+12125551234")
        assert result is None


class TestQuietHours:
    """Quiet hours enforcement with timezone support."""

    def test_during_quiet_hours_night(self):
        from src.sms.compliance import is_quiet_hours

        now = datetime(2026, 3, 15, 22, 0, 0, tzinfo=timezone.utc)
        assert is_quiet_hours("UTC", now=now) is True

    def test_during_business_hours(self):
        from src.sms.compliance import is_quiet_hours

        now = datetime(2026, 3, 15, 14, 0, 0, tzinfo=timezone.utc)
        assert is_quiet_hours("UTC", now=now) is False

    def test_quiet_hours_boundary_start(self):
        from src.sms.compliance import is_quiet_hours

        now = datetime(2026, 3, 15, 21, 0, 0, tzinfo=timezone.utc)
        assert is_quiet_hours("UTC", now=now) is True

    def test_quiet_hours_boundary_end(self):
        from src.sms.compliance import is_quiet_hours

        now = datetime(2026, 3, 15, 8, 0, 0, tzinfo=timezone.utc)
        assert is_quiet_hours("UTC", now=now) is False

    def test_early_morning_is_quiet(self):
        from src.sms.compliance import is_quiet_hours

        now = datetime(2026, 3, 15, 3, 0, 0, tzinfo=timezone.utc)
        assert is_quiet_hours("UTC", now=now) is True

    def test_timezone_conversion(self):
        from src.sms.compliance import is_quiet_hours

        now = datetime(2026, 3, 15, 17, 0, 0, tzinfo=timezone.utc)
        assert is_quiet_hours("America/New_York", now=now) is False

    def test_invalid_timezone_defaults_safely(self):
        from src.sms.compliance import is_quiet_hours

        now = datetime(2026, 3, 15, 14, 0, 0, tzinfo=timezone.utc)
        result = is_quiet_hours("Invalid/Timezone", now=now)
        assert result is False


class TestAreaCodeTimezone:
    """US area code to timezone mapping."""

    def test_new_york_area_code(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("+12125551234") == "America/New_York"

    def test_chicago_area_code(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("+13125551234") == "America/Chicago"

    def test_denver_area_code(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("+13035551234") == "America/Denver"

    def test_los_angeles_area_code(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("+12135551234") == "America/Los_Angeles"

    def test_las_vegas_area_code(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("+17025551234") == "America/Los_Angeles"

    def test_connecticut_area_code(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("+18605551234") == "America/New_York"

    def test_hawaii_area_code(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("+18085551234") == "Pacific/Honolulu"

    def test_alaska_area_code(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("+19075551234") == "America/Anchorage"

    def test_unknown_area_code_defaults(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("+19995551234") == "America/New_York"

    def test_without_country_code(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("2125551234") == "America/New_York"

    def test_short_number_defaults(self):
        from src.sms.compliance import _get_timezone_from_area_code

        assert _get_timezone_from_area_code("12") == "America/New_York"


class TestConsentHashChain:
    """SHA-256 tamper-evident consent hash chain."""

    def test_empty_chain_verifies(self):
        from src.sms.compliance import ConsentHashChain

        chain = ConsentHashChain()
        assert chain.verify_chain() is True

    def test_single_event_chain(self):
        from src.sms.compliance import ConsentHashChain

        chain = ConsentHashChain()
        h = chain.add_event(
            "opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form"
        )
        assert isinstance(h, str)
        assert len(h) == 64
        assert chain.verify_chain() is True

    def test_multi_event_chain(self):
        from src.sms.compliance import ConsentHashChain

        chain = ConsentHashChain()
        chain.add_event("opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form")
        chain.add_event(
            "opt_out", "+12125551234", "2026-03-16T10:00:00Z", "STOP keyword"
        )
        chain.add_event(
            "opt_in", "+12125551234", "2026-03-17T09:00:00Z", "START keyword"
        )
        assert chain.verify_chain() is True
        assert len(chain.events) == 3
        assert len(chain.hashes) == 3

    def test_tampered_chain_detected(self):
        from src.sms.compliance import ConsentHashChain

        chain = ConsentHashChain()
        chain.add_event("opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form")
        chain.add_event(
            "opt_out", "+12125551234", "2026-03-16T10:00:00Z", "STOP keyword"
        )

        chain._events[0]["evidence"] = "tampered_evidence"
        assert chain.verify_chain() is False

    def test_chain_links_are_dependent(self):
        from src.sms.compliance import ConsentHashChain

        chain = ConsentHashChain()
        h1 = chain.add_event(
            "opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form"
        )
        h2 = chain.add_event("opt_out", "+12125551234", "2026-03-16T10:00:00Z", "STOP")

        assert chain.events[1]["previous_hash"] == h1
        assert h1 != h2

    def test_first_event_uses_zero_hash(self):
        from src.sms.compliance import ConsentHashChain

        chain = ConsentHashChain()
        chain.add_event("opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form")
        assert chain.events[0]["previous_hash"] == "0" * 64

    def test_scope_change_event(self):
        from src.sms.compliance import ConsentHashChain

        chain = ConsentHashChain()
        chain.add_event(
            "scope_change",
            "+12125551234",
            "2026-03-15T14:00:00Z",
            "upgraded to marketing",
        )
        assert chain.verify_chain() is True
        assert chain.events[0]["event_type"] == "scope_change"


class TestConsentHashChainHMAC:
    """HMAC-SHA256 tamper-evident consent hash chain (production mode)."""

    def test_hmac_chain_verifies(self):
        from src.sms.compliance import ConsentHashChain

        chain = ConsentHashChain(hmac_secret="test-secret-key")
        chain.add_event("opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form")
        chain.add_event(
            "opt_out", "+12125551234", "2026-03-16T10:00:00Z", "STOP keyword"
        )
        assert chain.verify_chain() is True

    def test_hmac_produces_different_hashes_than_plain(self):
        from src.sms.compliance import ConsentHashChain

        plain_chain = ConsentHashChain()
        hmac_chain = ConsentHashChain(hmac_secret="secret")

        h_plain = plain_chain.add_event(
            "opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web"
        )
        h_hmac = hmac_chain.add_event(
            "opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web"
        )

        assert h_plain != h_hmac, "HMAC hash must differ from plain SHA-256"

    def test_hmac_tampered_chain_detected(self):
        from src.sms.compliance import ConsentHashChain

        chain = ConsentHashChain(hmac_secret="secret-key")
        chain.add_event("opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form")
        chain.add_event(
            "opt_out", "+12125551234", "2026-03-16T10:00:00Z", "STOP keyword"
        )
        chain._events[0]["evidence"] = "tampered"
        assert chain.verify_chain() is False

    def test_hmac_wrong_secret_fails_verification(self):
        from src.sms.compliance import ConsentHashChain

        chain = ConsentHashChain(hmac_secret="correct-secret")
        chain.add_event("opt_in", "+12125551234", "2026-03-15T14:00:00Z", "web_form")

        chain._hmac_secret = "wrong-secret"
        assert chain.verify_chain() is False

    def test_settings_consent_hmac_secret_exists(self):
        from src.config import Settings

        s = Settings()
        assert hasattr(s, "CONSENT_HMAC_SECRET")
        from pydantic import SecretStr

        assert isinstance(s.CONSENT_HMAC_SECRET, SecretStr)
        assert s.CONSENT_HMAC_SECRET.get_secret_value() == "change-me-in-production"


class TestConsentChecking:
    """Consent level verification for different message types."""

    def test_transactional_always_allowed(self):
        from src.sms.compliance import check_consent

        profile = {"consent": {}}
        assert check_consent(profile, "transactional") is True

    def test_transactional_allowed_even_with_no_opt_in(self):
        from src.sms.compliance import check_consent

        profile = {"consent": {"sms_opt_in": False}}
        assert check_consent(profile, "transactional") is True

    def test_informational_requires_consent(self):
        from src.sms.compliance import check_consent

        profile_no_consent = {"consent": {}}
        assert check_consent(profile_no_consent, "informational") is False

        profile_opted_in = {
            "consent": {"sms_opt_in": True, "sms_opt_in_method": "text_keyword"}
        }
        assert check_consent(profile_opted_in, "informational") is True

    def test_marketing_requires_written_consent(self):
        from src.sms.compliance import check_consent

        profile_keyword = {
            "consent": {"sms_opt_in": True, "sms_opt_in_method": "text_keyword"}
        }
        assert check_consent(profile_keyword, "marketing") is False

        profile_web = {"consent": {"sms_opt_in": True, "sms_opt_in_method": "web_form"}}
        assert check_consent(profile_web, "marketing") is True

    def test_opted_out_blocks_non_transactional(self):
        from src.sms.compliance import check_consent

        profile = {"consent": {"sms_opt_in": False, "sms_opt_in_method": "web_form"}}
        assert check_consent(profile, "informational") is False
        assert check_consent(profile, "marketing") is False

    def test_paper_form_satisfies_marketing(self):
        from src.sms.compliance import check_consent

        profile = {"consent": {"sms_opt_in": True, "sms_opt_in_method": "paper_form"}}
        assert check_consent(profile, "marketing") is True


# ============================================================================
# Config integration test
# ============================================================================


class TestSMSConfigFields:
    """Verify SMS config fields exist in Settings."""

    def test_sms_config_fields_exist(self):
        from src.config import Settings

        s = Settings()
        assert hasattr(s, "TELNYX_API_KEY")
        assert hasattr(s, "TELNYX_MESSAGING_PROFILE_ID")
        assert hasattr(s, "TELNYX_PUBLIC_KEY")
        assert hasattr(s, "QUIET_HOURS_START")
        assert hasattr(s, "QUIET_HOURS_END")
        assert hasattr(s, "SMS_FROM_NUMBER")

    def test_sms_config_defaults(self):
        from src.config import Settings

        s = Settings()
        assert s.TELNYX_MESSAGING_PROFILE_ID == ""
        assert s.TELNYX_PUBLIC_KEY == ""
        assert s.QUIET_HOURS_START == 21
        assert s.QUIET_HOURS_END == 8
        assert s.SMS_FROM_NUMBER == ""

    def test_sms_api_key_is_secret(self, monkeypatch):
        from src.config import Settings

        monkeypatch.setenv("TELNYX_API_KEY", "test-telnyx-key")
        s = Settings()
        assert s.TELNYX_API_KEY.get_secret_value() == "test-telnyx-key"
