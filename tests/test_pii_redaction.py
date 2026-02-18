"""Tests for PII redaction module."""


class TestRedactPII:
    """Test redact_pii function."""

    def test_phone_e164(self):
        """E.164 phone numbers are redacted."""
        from src.api.pii_redaction import redact_pii
        assert "[PHONE]" in redact_pii("Call me at +12035551234")

    def test_phone_us_format(self):
        """US formatted phone numbers are redacted."""
        from src.api.pii_redaction import redact_pii
        assert "[PHONE]" in redact_pii("My number is (203) 555-1234")

    def test_phone_dashes(self):
        """Dashed phone numbers are redacted."""
        from src.api.pii_redaction import redact_pii
        assert "[PHONE]" in redact_pii("Call 203-555-1234 for info")

    def test_email(self):
        """Email addresses are redacted."""
        from src.api.pii_redaction import redact_pii
        result = redact_pii("Email me at john.doe@example.com please")
        assert "[EMAIL]" in result
        assert "john.doe@example.com" not in result

    def test_credit_card(self):
        """Credit card numbers are redacted."""
        from src.api.pii_redaction import redact_pii
        assert "[CARD]" in redact_pii("Card: 4111 1111 1111 1111")

    def test_ssn_dashed(self):
        """SSN with dashes is redacted."""
        from src.api.pii_redaction import redact_pii
        assert "[SSN]" in redact_pii("SSN: 123-45-6789")

    def test_ssn_with_label(self):
        """SSN with 'social security' label is redacted."""
        from src.api.pii_redaction import redact_pii
        assert "[SSN]" in redact_pii("social security: 123456789")

    def test_player_card(self):
        """Player card numbers are redacted."""
        from src.api.pii_redaction import redact_pii
        assert "[PLAYER_ID]" in redact_pii("Player card number: 12345678")

    def test_name_self_id(self):
        """'My name is' pattern redacts names."""
        from src.api.pii_redaction import redact_pii
        result = redact_pii("My name is John Smith")
        assert "[NAME]" in result
        assert "John Smith" not in result

    def test_name_honorific(self):
        """Honorific + name pattern redacts names."""
        from src.api.pii_redaction import redact_pii
        result = redact_pii("Please help Mr. Johnson")
        assert "[NAME]" in result

    def test_no_pii_unchanged(self):
        """Text without PII is returned unchanged."""
        from src.api.pii_redaction import redact_pii
        text = "What restaurants do you have?"
        assert redact_pii(text) == text

    def test_empty_string(self):
        """Empty string returns empty string."""
        from src.api.pii_redaction import redact_pii
        assert redact_pii("") == ""

    def test_none_returns_none(self):
        """None input returns None."""
        from src.api.pii_redaction import redact_pii
        assert redact_pii(None) is None

    def test_multiple_pii_types(self):
        """Multiple PII types in one string are all redacted."""
        from src.api.pii_redaction import redact_pii
        text = "My name is Jane Doe, email jane@test.com, phone 203-555-1234"
        result = redact_pii(text)
        assert "[NAME]" in result
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "jane@test.com" not in result


class TestRedactDict:
    """Test redact_dict function."""

    def test_redact_all_string_values(self):
        """All string values are redacted by default."""
        from src.api.pii_redaction import redact_dict
        data = {"message": "Call 203-555-1234", "count": 5}
        result = redact_dict(data)
        assert "[PHONE]" in result["message"]
        assert result["count"] == 5

    def test_redact_specific_keys(self):
        """Only specified keys are redacted."""
        from src.api.pii_redaction import redact_dict
        data = {"message": "Call 203-555-1234", "query": "Call 203-555-1234"}
        result = redact_dict(data, keys_to_redact={"message"})
        assert "[PHONE]" in result["message"]
        assert "203-555-1234" in result["query"]  # Not redacted


class TestContainsPII:
    """Test contains_pii function."""

    def test_detects_phone(self):
        """Detects phone number."""
        from src.api.pii_redaction import contains_pii
        assert contains_pii("Call 203-555-1234") is True

    def test_no_pii(self):
        """No PII returns False."""
        from src.api.pii_redaction import contains_pii
        assert contains_pii("What restaurants do you have?") is False

    def test_empty_string(self):
        """Empty string returns False."""
        from src.api.pii_redaction import contains_pii
        assert contains_pii("") is False

    def test_detects_email(self):
        """Detects email address."""
        from src.api.pii_redaction import contains_pii
        assert contains_pii("Email: test@example.com") is True

    def test_detects_name(self):
        """Detects name pattern."""
        from src.api.pii_redaction import contains_pii
        assert contains_pii("My name is John Smith") is True
