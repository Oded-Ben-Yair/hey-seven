"""Streaming-safe PII redaction using regex pattern matching.

Buffers incoming text chunks and scans for PII patterns at configurable
boundaries. Releases clean text as soon as safe, redacts PII in place.

Reuses the compiled PII patterns from ``src.api.pii_redaction`` to ensure
consistency between streaming and non-streaming redaction paths.

Gaming-specific patterns (defined in pii_redaction): card numbers, SSN,
phone, email, and player/loyalty card numbers used in casino patron
communications.
"""

from collections.abc import Iterator

from src.api.pii_redaction import redact_pii

# Maximum pattern length we need to detect.
# Credit card with spaces: "4111 1111 1111 1111" = 19 chars.
# Player card prefix + number: "player card number: 123456789012" = ~32 chars.
# Use 40 as safety margin for lookahead buffer.
_MAX_PATTERN_LEN = 40


class StreamingPIIRedactor:
    """Streaming PII redactor with lookahead buffering.

    Buffers incoming text and applies PII regex patterns before releasing
    safe text to the caller. A trailing window of ``_MAX_PATTERN_LEN``
    characters is retained as lookahead to catch PII patterns that span
    chunk boundaries.

    The same ``redact_pii()`` function used for non-streaming responses
    is applied to the buffer contents, ensuring identical redaction
    behavior across both paths.

    Usage::

        redactor = StreamingPIIRedactor()
        for chunk in incoming_stream:
            for safe_text in redactor.feed(chunk):
                yield safe_text
        for remaining in redactor.flush():
            yield remaining
    """

    MAX_BUFFER = 500  # Hard cap -- force-flush regardless

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> Iterator[str]:
        """Feed a text chunk, yield safe (redacted) text.

        Text is buffered until enough characters accumulate for PII
        pattern detection. A trailing lookahead window is always
        retained to catch patterns that span chunk boundaries.

        Args:
            chunk: Incoming text fragment from the LLM stream.

        Yields:
            Redacted text that is safe to emit to the client.
        """
        self._buffer += chunk

        # Hard cap: force-flush if buffer exceeds maximum
        if len(self._buffer) >= self.MAX_BUFFER:
            yield from self._scan_and_release(force=True)
            return

        # Release safe prefix: everything before the last _MAX_PATTERN_LEN chars
        if len(self._buffer) > _MAX_PATTERN_LEN:
            yield from self._scan_and_release(force=False)

    def flush(self) -> Iterator[str]:
        """Flush remaining buffer (end of stream).

        Must be called after all chunks have been fed to ensure the
        trailing lookahead window is emitted with redaction applied.

        Yields:
            Any remaining redacted text.
        """
        if self._buffer:
            yield from self._scan_and_release(force=True)

    def _scan_and_release(self, force: bool) -> Iterator[str]:
        """Scan buffer for PII, release safe prefix.

        Uses ``redact_pii()`` from the shared PII module to apply
        all compiled patterns (phone, SSN, card, email, player ID,
        name patterns) consistently.

        Args:
            force: If True, release entire buffer (end-of-stream or
                hard cap). If False, retain trailing lookahead window.

        Yields:
            Redacted safe text.
        """
        # Apply redaction to the full buffer so patterns that span
        # the safe/lookahead boundary are caught.
        redacted = redact_pii(self._buffer)

        if force:
            self._buffer = ""
            if redacted:
                yield redacted
        else:
            # Keep last _MAX_PATTERN_LEN chars of the ORIGINAL buffer
            # as lookahead (not the redacted text, since redaction may
            # change lengths). The lookahead will be re-scanned on
            # next feed() or flush().
            safe = redacted[:-_MAX_PATTERN_LEN]
            self._buffer = self._buffer[-_MAX_PATTERN_LEN:]
            if safe:
                yield safe
