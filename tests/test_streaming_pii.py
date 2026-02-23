"""Tests for StreamingPIIRedactor.

Validates that PII patterns (phone numbers, SSNs, credit cards, emails,
account numbers) are correctly redacted even when split across multiple
streaming chunks. Also verifies buffer safety invariants and clean text
pass-through.
"""

import pytest

from src.agent.streaming_pii import StreamingPIIRedactor


class TestStreamingPIIRedactor:
    """Core redaction tests for the streaming PII redactor."""

    def test_redacts_phone_split_across_chunks(self):
        """Phone number split across chunks is caught and redacted."""
        r = StreamingPIIRedactor()
        output = (
            list(r.feed("Call me at 860-"))
            + list(r.feed("555-0123 please"))
            + list(r.flush())
        )
        text = "".join(output)
        assert "860-555-0123" not in text
        assert "[PHONE]" in text

    def test_redacts_email(self):
        """Email address is redacted."""
        r = StreamingPIIRedactor()
        output = (
            list(r.feed("Email bob@example.com for details"))
            + list(r.flush())
        )
        text = "".join(output)
        assert "bob@example.com" not in text
        assert "[EMAIL]" in text

    def test_clean_text_passes_through(self):
        """Non-PII text passes through without modification."""
        r = StreamingPIIRedactor()
        output = (
            list(r.feed("What restaurants are open tonight?"))
            + list(r.flush())
        )
        text = "".join(output)
        assert "restaurants" in text

    def test_redacts_ssn(self):
        """SSN in XXX-XX-XXXX format is redacted."""
        r = StreamingPIIRedactor()
        output = list(r.feed("SSN: 123-45-6789")) + list(r.flush())
        text = "".join(output)
        assert "123-45-6789" not in text
        assert "[SSN]" in text

    def test_redacts_card_number(self):
        """Credit card number with spaces is redacted."""
        r = StreamingPIIRedactor()
        output = (
            list(r.feed("Card: 4111 1111 1111 1111"))
            + list(r.flush())
        )
        text = "".join(output)
        assert "4111 1111 1111 1111" not in text
        assert "[CARD]" in text

    def test_buffer_never_exceeds_max(self):
        """Internal buffer never grows beyond MAX_BUFFER."""
        r = StreamingPIIRedactor()
        for _ in range(100):
            list(r.feed("a" * 50))
        assert len(r._buffer) <= r.MAX_BUFFER

    def test_redacts_player_card_number(self):
        """Player card number with label prefix is redacted.

        Uses the ``player card`` prefix which matches the
        ``pii_redaction.py`` player/loyalty card pattern.
        """
        r = StreamingPIIRedactor()
        output = (
            list(r.feed("Your player card: 12345678"))
            + list(r.flush())
        )
        text = "".join(output)
        assert "12345678" not in text
        assert "[PLAYER_ID]" in text

    def test_redacts_member_number(self):
        """Member number with label prefix is redacted."""
        r = StreamingPIIRedactor()
        output = (
            list(r.feed("Member ID: 987654321"))
            + list(r.flush())
        )
        text = "".join(output)
        assert "987654321" not in text

    def test_card_split_across_many_chunks(self):
        """Credit card number split across 4 chunks is caught."""
        r = StreamingPIIRedactor()
        output = (
            list(r.feed("Card: "))
            + list(r.feed("4242 "))
            + list(r.feed("4242 "))
            + list(r.feed("4242 "))
            + list(r.feed("4242"))
            + list(r.flush())
        )
        text = "".join(output)
        assert "4242 4242 4242 4242" not in text
        assert "[CARD]" in text

    def test_ssn_split_across_chunks(self):
        """SSN split across 3 chunks is caught."""
        r = StreamingPIIRedactor()
        output = (
            list(r.feed("SSN: 123-"))
            + list(r.feed("45-"))
            + list(r.feed("6789. End."))
            + list(r.flush())
        )
        text = "".join(output)
        assert "123-45-6789" not in text
        assert "[SSN]" in text

    def test_flush_idempotent(self):
        """Calling flush() twice does not duplicate output."""
        r = StreamingPIIRedactor()
        list(r.feed("hello world"))
        first = list(r.flush())
        second = list(r.flush())
        assert len(first) >= 1
        assert len(second) == 0

    def test_empty_chunks_no_output(self):
        """Empty string chunks produce no output."""
        r = StreamingPIIRedactor()
        output = list(r.feed("")) + list(r.feed("")) + list(r.flush())
        assert output == []

    def test_multiple_pii_in_one_stream(self):
        """Multiple PII types in a single stream are all redacted."""
        r = StreamingPIIRedactor()
        output = (
            list(r.feed("Phone: 555-123-4567, SSN: 123-45-6789"))
            + list(r.flush())
        )
        text = "".join(output)
        assert "555-123-4567" not in text
        assert "123-45-6789" not in text

    def test_short_text_only_emitted_on_flush(self):
        """Text shorter than lookahead window is buffered until flush."""
        r = StreamingPIIRedactor()
        output = list(r.feed("hi"))
        # Short text stays in buffer (within lookahead window)
        assert len(output) == 0
        flush_output = list(r.flush())
        text = "".join(flush_output)
        assert "hi" in text

    def test_redacts_long_email_with_subdomain(self):
        """Email with long subdomain (60+ chars total) is caught by the 80-char buffer.

        Validates that the increased _MAX_PATTERN_LEN=80 covers real-world
        email addresses with subdomains and long TLDs that exceed the
        previous 40-char buffer.
        """
        long_email = "firstname.lastname@hospitality.mohegan-sun-resort.com"
        assert len(long_email) > 50  # Confirm this is a genuinely long pattern
        r = StreamingPIIRedactor()
        # Split the email across chunk boundaries to test lookahead
        output = (
            list(r.feed("Contact "))
            + list(r.feed(long_email[:30]))
            + list(r.feed(long_email[30:]))
            + list(r.feed(" for reservations"))
            + list(r.flush())
        )
        text = "".join(output)
        assert long_email not in text
        assert "[EMAIL]" in text


class TestStreamingVsFullTextParity:
    """R37 fix C-002: Verify streaming and full-text PII redaction produce
    identical output for known PII patterns.

    StreamingPIIRedactor delegates to redact_pii() internally, so parity
    is expected. This test catches divergence if the code paths ever split.
    """

    @pytest.mark.parametrize(
        "pii_text",
        [
            "My SSN is 123-45-6789 thanks",
            "Call me at 860-555-0123 please",
            "Card number 4111 1111 1111 1111 on file",
            "Email bob@example.com for details",
            "My player card: 12345678901 is expired",
            "member number: 9876543210 for rewards",
        ],
        ids=["ssn", "phone", "card", "email", "player_card", "member"],
    )
    def test_streaming_matches_full_text_redaction(self, pii_text):
        """Feeding text token-by-token through StreamingPIIRedactor and then
        flushing must produce the same result as redact_pii() on the full text.
        """
        from src.api.pii_redaction import redact_pii

        # Full-text path
        full_result = redact_pii(pii_text)

        # Streaming path: feed one character at a time (worst case)
        r = StreamingPIIRedactor()
        streaming_chunks = []
        for char in pii_text:
            streaming_chunks.extend(r.feed(char))
        streaming_chunks.extend(r.flush())
        streaming_result = "".join(streaming_chunks)

        assert streaming_result == full_result, (
            f"Parity mismatch:\n"
            f"  Full-text: {full_result!r}\n"
            f"  Streaming: {streaming_result!r}"
        )
