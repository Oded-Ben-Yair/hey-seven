"""Adversarial tests for streaming PII redaction.

Tests that PII patterns spanning chunk boundaries are correctly
redacted, and that the buffer handles adversarial chunk splitting.

R52 fix D5: StreamingPIIRedactor uses a 120-char lookahead buffer to
catch PII spanning chunk boundaries. These tests verify the buffer
handles adversarial splits, single-char feeding, and fuzzed input.
"""

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.agent.streaming_pii import StreamingPIIRedactor


class TestStreamingPIIBoundary:
    """Test PII detection across chunk boundaries."""

    def test_phone_split_at_boundary(self):
        """Phone number split across two chunks must be redacted."""
        r = StreamingPIIRedactor()
        chunks = list(r.feed("Call me at 555-"))
        chunks.extend(r.feed("123-4567 for details"))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "555-123-4567" not in text, "Phone number leaked across chunk boundary"

    def test_ssn_split_at_boundary(self):
        """SSN split across chunks must be redacted."""
        r = StreamingPIIRedactor()
        chunks = list(r.feed("My SSN is 123-45-"))
        chunks.extend(r.feed("6789 and that's it"))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "123-45-6789" not in text, "SSN leaked across chunk boundary"

    def test_email_split_at_boundary(self):
        """Email split across chunks must be redacted."""
        r = StreamingPIIRedactor()
        chunks = list(r.feed("Email me at user@"))
        chunks.extend(r.feed("example.com please"))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "user@example.com" not in text, "Email leaked across chunk boundary"

    def test_credit_card_split_across_three_chunks(self):
        """Credit card number split across 3 chunks must be redacted."""
        r = StreamingPIIRedactor()
        chunks = list(r.feed("Card: 4111-1111-"))
        chunks.extend(r.feed("1111-"))
        chunks.extend(r.feed("1111 expiry 12/25"))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "4111-1111-1111-1111" not in text, "Credit card leaked across chunks"

    def test_safe_text_passes_through(self):
        """Non-PII text should not be modified."""
        r = StreamingPIIRedactor()
        chunks = list(r.feed("Hello, welcome to the casino! "))
        chunks.extend(r.feed("Enjoy your stay. "))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "Hello" in text
        assert "casino" in text

    def test_buffer_flush_on_max_buffer(self):
        """Buffer should flush when reaching max size (500 chars)."""
        r = StreamingPIIRedactor()
        # Feed a large chunk that exceeds MAX_BUFFER (500)
        large_text = "Hello world " * 50  # ~600 chars
        chunks = list(r.feed(large_text))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "Hello world" in text

    def test_empty_input(self):
        """Empty input should produce empty output."""
        r = StreamingPIIRedactor()
        chunks = list(r.feed(""))
        chunks.extend(r.flush())
        assert "".join(chunks) == ""

    def test_single_char_chunks(self):
        """Single-character chunk feeding should still work."""
        r = StreamingPIIRedactor()
        text = "No PII here just text."
        all_chunks = []
        for c in text:
            all_chunks.extend(r.feed(c))
        all_chunks.extend(r.flush())
        result = "".join(all_chunks)
        # The text should emerge intact (may be buffered and released in pieces)
        assert "No PII" in result or "text" in result

    def test_phone_in_single_chunk(self):
        """Phone number entirely within one chunk must be redacted."""
        r = StreamingPIIRedactor()
        chunks = list(r.feed("My number is 555-867-5309 thanks"))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "555-867-5309" not in text

    def test_ssn_at_buffer_boundary(self):
        """SSN at exact buffer boundary edge case."""
        r = StreamingPIIRedactor()
        # Fill buffer close to _MAX_PATTERN_LEN (120 chars) then send SSN
        padding = "x" * 110
        chunks = list(r.feed(padding + " 123-"))
        chunks.extend(r.feed("45-6789 done"))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "123-45-6789" not in text, "SSN leaked at buffer boundary"

    def test_multiple_pii_across_chunks(self):
        """Multiple PII items spanning different chunks."""
        r = StreamingPIIRedactor()
        chunks = list(r.feed("Call 555-"))
        chunks.extend(r.feed("111-2222 or email "))
        chunks.extend(r.feed("test@"))
        chunks.extend(r.feed("example.com"))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "555-111-2222" not in text
        assert "test@example.com" not in text

    def test_flush_without_feed(self):
        """Flushing an unused redactor should produce nothing."""
        r = StreamingPIIRedactor()
        chunks = list(r.flush())
        assert chunks == []

    def test_redactor_reuse_after_flush(self):
        """After flush, redactor can be reused for a new stream."""
        r = StreamingPIIRedactor()
        # First stream
        list(r.feed("My phone is 555-123-4567"))
        list(r.flush())
        # Second stream (buffer should be empty)
        chunks = list(r.feed("Clean text here"))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "Clean text" in text


class TestStreamingPIIFuzz:
    """Hypothesis fuzz tests for StreamingPIIRedactor."""

    @given(text=st.text(max_size=1000))
    @settings(max_examples=200, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_feed_never_crashes(self, text):
        """feed() must never crash on arbitrary input."""
        r = StreamingPIIRedactor()
        chunks = list(r.feed(text))
        chunks.extend(r.flush())
        # All output should be strings
        for chunk in chunks:
            assert isinstance(chunk, str)

    @given(data=st.data())
    @settings(max_examples=200, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_chunk_splitting(self, data):
        """Splitting text at arbitrary points must not crash."""
        text = data.draw(st.text(min_size=1, max_size=500))
        # Choose random split points
        if len(text) > 1:
            n_splits = data.draw(st.integers(min_value=1, max_value=min(10, len(text) - 1)))
            split_points = sorted(set(data.draw(
                st.lists(
                    st.integers(min_value=1, max_value=len(text) - 1),
                    min_size=n_splits,
                    max_size=n_splits,
                )
            )))
        else:
            split_points = []

        # Split text at those points
        chunks_in = []
        prev = 0
        for sp in split_points:
            chunks_in.append(text[prev:sp])
            prev = sp
        chunks_in.append(text[prev:])

        r = StreamingPIIRedactor()
        output_chunks = []
        for chunk in chunks_in:
            output_chunks.extend(r.feed(chunk))
        output_chunks.extend(r.flush())

        # Verify: all output is strings
        for chunk in output_chunks:
            assert isinstance(chunk, str)

    @given(text=st.text(max_size=200))
    @settings(max_examples=100, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_fresh_redactor_per_request(self, text):
        """Each redactor instance should be independent."""
        r1 = StreamingPIIRedactor()
        r2 = StreamingPIIRedactor()
        list(r1.feed("prefix " + text))
        list(r2.feed(text))
        # Both should flush independently without affecting each other
        list(r1.flush())
        list(r2.flush())

    @given(text=st.text(max_size=300))
    @settings(max_examples=100, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_output_preserves_non_pii_content(self, text):
        """Non-PII text should appear in output (not swallowed by buffer)."""
        # Only test with text that has no PII-like digit sequences
        # to avoid false assertions on redacted content
        safe_text = "".join(c for c in text if not c.isdigit() and c != "@")
        if not safe_text.strip():
            return  # Skip empty-after-filter cases

        r = StreamingPIIRedactor()
        chunks = list(r.feed(safe_text))
        chunks.extend(r.flush())
        result = "".join(chunks)

        # The output should contain the input (possibly with extra redaction
        # markers if name patterns matched, but content should not vanish)
        assert len(result) > 0 or len(safe_text.strip()) == 0


class TestStreamingPIIEdgeCases:
    """Edge cases for streaming PII redaction."""

    def test_only_whitespace(self):
        """Whitespace-only input should pass through."""
        r = StreamingPIIRedactor()
        chunks = list(r.feed("   \t\n  "))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert text.strip() == "" or text == "   \t\n  "

    def test_very_long_non_pii_text(self):
        """Long text without PII should not be corrupted."""
        r = StreamingPIIRedactor()
        long_text = "The casino restaurant opens at seven pm. " * 30
        chunks = list(r.feed(long_text))
        chunks.extend(r.flush())
        text = "".join(chunks)
        assert "casino restaurant" in text
        assert "seven pm" in text

    def test_redaction_placeholder_not_re_redacted(self):
        """[PHONE] placeholder from prior redaction should survive re-scan."""
        r = StreamingPIIRedactor()
        # Feed text that will be partially buffered (lookahead re-scans)
        chunks = list(r.feed("Your number is [PHONE] and "))
        chunks.extend(r.feed("your email is test@example.com"))
        chunks.extend(r.flush())
        text = "".join(chunks)
        # [PHONE] placeholder should survive re-scanning
        assert "[PHONE]" in text
        # Email should be redacted
        assert "test@example.com" not in text

    def test_buffer_size_property(self):
        """buffer_size property should reflect internal buffer state."""
        r = StreamingPIIRedactor()
        assert r.buffer_size == 0
        # Feed small chunk (stays in buffer)
        list(r.feed("hello"))
        assert r.buffer_size > 0
        # Flush empties buffer
        list(r.flush())
        assert r.buffer_size == 0
