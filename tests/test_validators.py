"""Tests for runtime data boundary validators.

Verifies validate_retrieved_chunk() and validate_guest_profile() catch
invalid data shapes at the RAG retrieval → graph state and
Firestore → guest profile boundaries.
"""

import pytest

from src.data.validators import validate_retrieved_chunk, validate_guest_profile


# ---------------------------------------------------------------------------
# validate_retrieved_chunk
# ---------------------------------------------------------------------------


class TestValidateRetrievedChunk:
    """Tests for RAG chunk validation."""

    def test_valid_chunk(self):
        chunk = {
            "content": "Restaurant info",
            "metadata": {"category": "dining", "source": "menu.json"},
            "score": 0.85,
        }
        assert validate_retrieved_chunk(chunk) is True

    def test_missing_content(self):
        chunk = {"metadata": {}, "score": 0.5}
        assert validate_retrieved_chunk(chunk) is False

    def test_content_not_string(self):
        chunk = {"content": 123, "metadata": {}, "score": 0.5}
        assert validate_retrieved_chunk(chunk) is False

    def test_missing_metadata(self):
        chunk = {"content": "text", "score": 0.5}
        assert validate_retrieved_chunk(chunk) is False

    def test_metadata_not_dict(self):
        chunk = {"content": "text", "metadata": "not a dict", "score": 0.5}
        assert validate_retrieved_chunk(chunk) is False

    def test_missing_score(self):
        chunk = {"content": "text", "metadata": {}}
        assert validate_retrieved_chunk(chunk) is False

    def test_score_not_numeric(self):
        chunk = {"content": "text", "metadata": {}, "score": "high"}
        assert validate_retrieved_chunk(chunk) is False

    def test_not_a_dict(self):
        assert validate_retrieved_chunk("not a dict") is False
        assert validate_retrieved_chunk(None) is False
        assert validate_retrieved_chunk([]) is False

    def test_valid_chunk_with_optional_fields(self):
        chunk = {
            "content": "text",
            "metadata": {"category": "dining"},
            "score": 0.9,
            "rrf_score": 0.015,
            "source_name": "menu.json",
        }
        assert validate_retrieved_chunk(chunk) is True

    def test_score_zero_is_valid(self):
        chunk = {"content": "text", "metadata": {}, "score": 0.0}
        assert validate_retrieved_chunk(chunk) is True

    def test_score_float_string_is_invalid(self):
        chunk = {"content": "text", "metadata": {}, "score": "0.5"}
        # String "0.5" is castable to float, so this should be valid
        assert validate_retrieved_chunk(chunk) is True


# ---------------------------------------------------------------------------
# validate_guest_profile
# ---------------------------------------------------------------------------


class TestValidateGuestProfile:
    """Tests for guest profile validation."""

    def test_valid_profile(self):
        profile = {
            "_id": "+12035551234",
            "_version": 1,
            "core_identity": {"phone": "+12035551234"},
        }
        assert validate_guest_profile(profile) is True

    def test_missing_id(self):
        profile = {"_version": 1, "core_identity": {}}
        assert validate_guest_profile(profile) is False

    def test_missing_version(self):
        profile = {"_id": "phone", "core_identity": {}}
        assert validate_guest_profile(profile) is False

    def test_version_not_int(self):
        profile = {"_id": "phone", "_version": "1", "core_identity": {}}
        assert validate_guest_profile(profile) is False

    def test_missing_core_identity(self):
        profile = {"_id": "phone", "_version": 1}
        assert validate_guest_profile(profile) is False

    def test_core_identity_not_dict(self):
        profile = {"_id": "phone", "_version": 1, "core_identity": "not a dict"}
        assert validate_guest_profile(profile) is False

    def test_not_a_dict(self):
        assert validate_guest_profile("not a dict") is False
        assert validate_guest_profile(None) is False
        assert validate_guest_profile([]) is False

    def test_full_profile(self):
        """A complete profile with all sections should validate."""
        profile = {
            "_id": "+12035551234",
            "_version": 2,
            "_schema_version": 2,
            "_created_at": "2026-01-01T00:00:00Z",
            "_updated_at": "2026-01-01T00:00:00Z",
            "core_identity": {"phone": "+12035551234"},
            "visit_context": {},
            "preferences": {"dining": {}, "entertainment": {}},
            "companions": [],
            "consent": {},
            "engagement": {"total_conversations": 0},
        }
        assert validate_guest_profile(profile) is True
