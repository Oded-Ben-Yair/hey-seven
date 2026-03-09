"""Tests for the guest profile data models.

Covers:
- ProfileField creation and validation
- Profile completeness calculation (empty, partial, full)
- Confidence decay (90-day threshold)
- Confidence update rules (confirm, contradict)
- Low-confidence field filtering
- Agent context (decay + filter pipeline)
- Edge cases (JSON serialization, custom threshold)
- Firestore client caching structure

Mock-based tests (CRUD with @patch Firestore, batch overflow, multi-tenant)
removed (mock purge R111).
"""

from datetime import datetime, timedelta, timezone

import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_profile_field(
    value,
    confidence=0.85,
    source="self_reported",
    collected_at=None,
    consent_scope="personalization",
):
    """Create a ProfileField dict for testing."""
    if collected_at is None:
        collected_at = datetime.now(timezone.utc).isoformat()
    return {
        "value": value,
        "confidence": confidence,
        "source": source,
        "collected_at": collected_at,
        "consent_scope": consent_scope,
    }


def _make_full_profile(phone="+12035551234"):
    """Create a fully-populated guest profile for testing."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "_id": phone,
        "_version": 3,
        "_created_at": now,
        "_updated_at": now,
        "core_identity": {
            "phone": phone,
            "guest_uuid": "g-test-uuid-1234",
            "name": _make_profile_field("Maria", confidence=0.95),
            "email": _make_profile_field(
                "maria@test.com", confidence=0.90, source="incentive_exchange"
            ),
            "language": _make_profile_field("es", confidence=0.85),
            "full_name": _make_profile_field("Maria Garcia", confidence=0.90),
            "date_of_birth": _make_profile_field("1985-06-15", confidence=0.95),
        },
        "visit_context": {
            "planned_visit_date": _make_profile_field("2026-04-10", confidence=0.90),
            "party_size": _make_profile_field(
                4, confidence=0.85, source="contextual_extraction"
            ),
            "occasion": _make_profile_field(
                "anniversary", confidence=0.80, source="contextual_extraction"
            ),
            "visit_history": [{"date": "2026-01-20", "source": "crm_import"}],
        },
        "preferences": {
            "dining": {
                "dietary_restrictions": _make_profile_field(
                    ["gluten-free"], confidence=0.90
                ),
                "cuisine_preferences": _make_profile_field(
                    ["italian", "seafood"], confidence=0.75
                ),
                "budget_range": _make_profile_field("$$", confidence=0.70),
                "kids_menu_needed": _make_profile_field(True, confidence=0.85),
            },
            "entertainment": {
                "interests": _make_profile_field(
                    ["comedy", "live_music"], confidence=0.70
                ),
                "accessibility_needs": _make_profile_field(
                    "wheelchair", confidence=0.50
                ),
            },
            "gaming": {
                "level": _make_profile_field("casual", confidence=0.65),
                "preferred_games": _make_profile_field(
                    ["slots", "blackjack"], confidence=0.60
                ),
                "typical_spend": _make_profile_field(200, confidence=0.55),
            },
            "spa": {
                "treatments_interested": _make_profile_field(
                    ["massage", "facial"], confidence=0.70
                ),
            },
        },
        "companions": [
            {
                "relationship": "spouse",
                "name": _make_profile_field(
                    "Carlos", confidence=0.80, source="contextual_extraction"
                ),
                "preferences": {
                    "dining": _make_profile_field(
                        "steak", confidence=0.65, source="inferred"
                    )
                },
            },
        ],
        "consent": {
            "sms_opt_in": True,
            "marketing_consent": True,
            "consent_version": "1.0",
        },
        "engagement": {
            "total_conversations": 3,
            "total_messages_sent": 12,
            "total_messages_received": 15,
            "profile_completeness": 0.45,
            "offers_sent": 1,
            "offers_redeemed": 0,
            "escalations": 0,
        },
    }


def _make_empty_profile(phone="+12035551234"):
    """Create a minimal empty profile for testing."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "_id": phone,
        "_version": 0,
        "_created_at": now,
        "_updated_at": now,
        "core_identity": {"phone": phone},
        "visit_context": {},
        "preferences": {"dining": {}, "entertainment": {}, "gaming": {}, "spa": {}},
        "companions": [],
        "consent": {},
        "engagement": {
            "total_conversations": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "profile_completeness": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# ProfileField tests
# ---------------------------------------------------------------------------


class TestProfileField:
    def test_create_profile_field_with_all_keys(self):
        """ProfileField dict has all required keys."""
        pf = _make_profile_field("Maria", confidence=0.95, source="self_reported")
        assert pf["value"] == "Maria"
        assert pf["confidence"] == 0.95
        assert pf["source"] == "self_reported"
        assert "collected_at" in pf
        assert pf["consent_scope"] == "personalization"

    def test_profile_field_confidence_bounds(self):
        """Confidence should be a float between 0.0 and 1.0."""
        pf = _make_profile_field("test", confidence=0.0)
        assert pf["confidence"] == 0.0
        pf2 = _make_profile_field("test", confidence=1.0)
        assert pf2["confidence"] == 1.0

    def test_profile_field_source_types(self):
        """All five source types are accepted."""
        sources = [
            "self_reported",
            "contextual_extraction",
            "inferred",
            "crm_import",
            "incentive_exchange",
        ]
        for source in sources:
            pf = _make_profile_field("val", source=source)
            assert pf["source"] == source

    def test_profile_field_none_consent_scope(self):
        """consent_scope can be None."""
        pf = _make_profile_field("val", consent_scope=None)
        assert pf["consent_scope"] is None

    def test_profile_field_iso8601_collected_at(self):
        """collected_at is a valid ISO 8601 string."""
        pf = _make_profile_field("val")
        # Should not raise
        dt = datetime.fromisoformat(pf["collected_at"].replace("Z", "+00:00"))
        assert isinstance(dt, datetime)


# ---------------------------------------------------------------------------
# Completeness tests
# ---------------------------------------------------------------------------


class TestCalculateCompleteness:
    def test_empty_profile_returns_zero(self):
        """Empty profile has 0.0 completeness."""
        from src.data.models import calculate_completeness

        result = calculate_completeness({})
        assert result == 0.0

    def test_none_profile_returns_zero(self):
        """None-ish empty dict returns 0.0."""
        from src.data.models import calculate_completeness

        assert calculate_completeness({}) == 0.0

    def test_empty_skeleton_returns_zero(self):
        """Empty skeleton profile with sections but no fields returns 0.0."""
        from src.data.models import calculate_completeness

        profile = _make_empty_profile()
        assert calculate_completeness(profile) == 0.0

    def test_full_profile_returns_one(self):
        """Fully populated profile returns 1.0 completeness."""
        from src.data.models import calculate_completeness

        profile = _make_full_profile()
        result = calculate_completeness(profile)
        assert result == 1.0

    def test_partial_profile_between_zero_and_one(self):
        """Profile with some fields returns a value between 0 and 1."""
        from src.data.models import calculate_completeness

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field("Maria")
        result = calculate_completeness(profile)
        assert 0.0 < result < 1.0

    def test_core_fields_weighted_higher(self):
        """Core identity fields contribute more to completeness than preferences."""
        from src.data.models import calculate_completeness

        # Profile with only one core field
        profile_core = _make_empty_profile()
        profile_core["core_identity"]["name"] = _make_profile_field("Maria")

        # Profile with only one preference field
        profile_pref = _make_empty_profile()
        profile_pref["preferences"]["dining"]["dietary_restrictions"] = (
            _make_profile_field(["vegan"])
        )

        core_score = calculate_completeness(profile_core)
        pref_score = calculate_completeness(profile_pref)
        assert core_score > pref_score

    def test_companions_contribute_to_completeness(self):
        """Non-empty companions list adds to completeness."""
        from src.data.models import calculate_completeness

        profile = _make_empty_profile()
        score_no_companions = calculate_completeness(profile)
        profile["companions"] = [{"relationship": "spouse"}]
        score_with_companions = calculate_completeness(profile)
        assert score_with_companions > score_no_companions

    def test_null_profile_field_value_not_counted(self):
        """A ProfileField with value=None does not count toward completeness."""
        from src.data.models import calculate_completeness

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field(None, confidence=0.5)
        assert calculate_completeness(profile) == 0.0


# ---------------------------------------------------------------------------
# Confidence decay tests
# ---------------------------------------------------------------------------


class TestConfidenceDecay:
    def test_no_decay_within_90_days(self):
        from src.data.models import apply_confidence_decay

        now = datetime.now(timezone.utc)
        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field(
            "Maria",
            confidence=0.95,
            collected_at=(now - timedelta(days=30)).isoformat(),
        )
        result = apply_confidence_decay(profile, now=now)
        assert result["core_identity"]["name"]["confidence"] == 0.95

    def test_decay_after_90_days(self):
        from src.data.models import apply_confidence_decay

        now = datetime.now(timezone.utc)
        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field(
            "Maria",
            confidence=0.95,
            collected_at=(now - timedelta(days=100)).isoformat(),
        )
        result = apply_confidence_decay(profile, now=now)
        expected = round(0.95 * 0.90, 4)
        assert result["core_identity"]["name"]["confidence"] == expected

    def test_decay_exactly_90_days_no_decay(self):
        from src.data.models import apply_confidence_decay

        now = datetime.now(timezone.utc)
        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field(
            "Maria",
            confidence=0.80,
            collected_at=(now - timedelta(days=90)).isoformat(),
        )
        result = apply_confidence_decay(profile, now=now)
        assert result["core_identity"]["name"]["confidence"] == 0.80

    def test_decay_applies_to_nested_preferences(self):
        from src.data.models import apply_confidence_decay

        now = datetime.now(timezone.utc)
        profile = _make_empty_profile()
        profile["preferences"]["dining"]["cuisine_preferences"] = _make_profile_field(
            ["italian"],
            confidence=0.80,
            collected_at=(now - timedelta(days=120)).isoformat(),
        )
        result = apply_confidence_decay(profile, now=now)
        expected = round(0.80 * 0.90, 4)
        assert (
            result["preferences"]["dining"]["cuisine_preferences"]["confidence"]
            == expected
        )

    def test_decay_applies_to_companions(self):
        from src.data.models import apply_confidence_decay

        now = datetime.now(timezone.utc)
        profile = _make_empty_profile()
        profile["companions"] = [
            {
                "relationship": "spouse",
                "name": _make_profile_field(
                    "Carlos",
                    confidence=0.80,
                    collected_at=(now - timedelta(days=100)).isoformat(),
                ),
            }
        ]
        result = apply_confidence_decay(profile, now=now)
        expected = round(0.80 * 0.90, 4)
        assert result["companions"][0]["name"]["confidence"] == expected

    def test_decay_handles_missing_collected_at(self):
        from src.data.models import apply_confidence_decay

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = {"value": "Maria", "confidence": 0.80}
        result = apply_confidence_decay(profile)
        assert result["core_identity"]["name"]["confidence"] == 0.80

    def test_decay_handles_invalid_collected_at(self):
        from src.data.models import apply_confidence_decay

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = {
            "value": "Maria",
            "confidence": 0.80,
            "collected_at": "not-a-date",
        }
        result = apply_confidence_decay(profile)
        assert result["core_identity"]["name"]["confidence"] == 0.80


# ---------------------------------------------------------------------------
# Confidence update tests
# ---------------------------------------------------------------------------


class TestConfidenceUpdate:
    def test_confirm_boosts_confidence(self):
        from src.data.models import update_confidence

        existing = _make_profile_field("Maria", confidence=0.80)
        result = update_confidence(existing, "crm_import", "Maria")
        assert result["confidence"] == 0.95
        assert result["source"] == "crm_import"

    def test_confirm_caps_at_one(self):
        from src.data.models import update_confidence

        existing = _make_profile_field("Maria", confidence=0.95)
        result = update_confidence(existing, "crm_import", "Maria")
        assert result["confidence"] == 1.0

    def test_contradict_reduces_confidence(self):
        from src.data.models import update_confidence

        existing = _make_profile_field("Maria", confidence=0.80)
        result = update_confidence(existing, "self_reported", "Mary")
        assert result["confidence"] == 0.50
        assert result["_contradicted"] is True
        assert result["_previous_value"] == "Maria"

    def test_contradict_floors_at_zero(self):
        from src.data.models import update_confidence

        existing = _make_profile_field("Maria", confidence=0.20)
        result = update_confidence(existing, "self_reported", "Mary")
        assert result["confidence"] == 0.0
        assert result["_contradicted"] is True

    def test_contradict_preserves_consent_scope(self):
        from src.data.models import update_confidence

        existing = _make_profile_field(
            "Maria", confidence=0.80, consent_scope="marketing"
        )
        result = update_confidence(existing, "self_reported", "Mary")
        assert result["consent_scope"] == "marketing"

    def test_update_sets_new_collected_at(self):
        from src.data.models import update_confidence

        now = datetime(2026, 3, 15, 14, 0, 0, tzinfo=timezone.utc)
        existing = _make_profile_field("Maria", confidence=0.80)
        result = update_confidence(existing, "crm_import", "Maria", now=now)
        assert result["collected_at"] == now.isoformat()


# ---------------------------------------------------------------------------
# Low-confidence filtering tests
# ---------------------------------------------------------------------------


class TestFilterLowConfidence:
    def test_high_confidence_fields_retained(self):
        from src.data.models import filter_low_confidence

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field("Maria", confidence=0.95)
        result = filter_low_confidence(profile)
        assert result["core_identity"]["name"]["value"] == "Maria"

    def test_low_confidence_fields_nullified(self):
        from src.data.models import filter_low_confidence

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field("Maria", confidence=0.30)
        result = filter_low_confidence(profile)
        assert result["core_identity"]["name"] is None

    def test_exactly_at_threshold_is_retained(self):
        from src.data.models import filter_low_confidence

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field("Maria", confidence=0.40)
        result = filter_low_confidence(profile)
        assert result["core_identity"]["name"]["value"] == "Maria"

    def test_filter_does_not_mutate_original(self):
        from src.data.models import filter_low_confidence

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field("Maria", confidence=0.30)
        filter_low_confidence(profile)
        # Original still has the low-confidence field
        assert profile["core_identity"]["name"]["confidence"] == 0.30

    def test_filter_nested_preferences(self):
        from src.data.models import filter_low_confidence

        profile = _make_empty_profile()
        profile["preferences"]["dining"]["cuisine_preferences"] = _make_profile_field(
            ["italian"], confidence=0.20
        )
        result = filter_low_confidence(profile)
        assert result["preferences"]["dining"]["cuisine_preferences"] is None

    def test_filter_companion_fields(self):
        from src.data.models import filter_low_confidence

        profile = _make_empty_profile()
        profile["companions"] = [
            {
                "relationship": "spouse",
                "name": _make_profile_field("Carlos", confidence=0.30),
            }
        ]
        result = filter_low_confidence(profile)
        assert result["companions"][0]["name"] is None


# ---------------------------------------------------------------------------
# get_agent_context integration
# ---------------------------------------------------------------------------


class TestGetAgentContext:
    def test_agent_context_filters_low_confidence(self):
        from src.data.guest_profile import get_agent_context

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field("Maria", confidence=0.30)
        result = get_agent_context(profile)
        assert result["core_identity"]["name"] is None

    def test_agent_context_applies_decay_then_filters(self):
        from src.data.guest_profile import get_agent_context

        now = datetime.now(timezone.utc)
        profile = _make_empty_profile()
        # confidence 0.44 * 0.90 = 0.396, which is below 0.40 threshold
        profile["core_identity"]["name"] = _make_profile_field(
            "Maria",
            confidence=0.44,
            collected_at=(now - timedelta(days=100)).isoformat(),
        )
        result = get_agent_context(profile, now=now)
        assert result["core_identity"]["name"] is None

    def test_agent_context_retains_high_confidence_recent(self):
        from src.data.guest_profile import get_agent_context

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field("Maria", confidence=0.95)
        result = get_agent_context(profile)
        assert result["core_identity"]["name"]["value"] == "Maria"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_completeness_with_none_value_field(self):
        from src.data.models import calculate_completeness

        profile = _make_full_profile()
        # Set one core field to value=None
        profile["core_identity"]["name"]["value"] = None
        result = calculate_completeness(profile)
        assert result < 1.0

    def test_decay_with_utc_z_suffix(self):
        from src.data.models import apply_confidence_decay

        now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        profile = _make_empty_profile()
        profile["core_identity"]["name"] = {
            "value": "Maria",
            "confidence": 0.80,
            "collected_at": "2026-01-01T00:00:00Z",  # ~165 days before now
        }
        result = apply_confidence_decay(profile, now=now)
        assert result["core_identity"]["name"]["confidence"] == round(0.80 * 0.90, 4)

    def test_filter_custom_threshold(self):
        from src.data.models import filter_low_confidence

        profile = _make_empty_profile()
        profile["core_identity"]["name"] = _make_profile_field("Maria", confidence=0.60)
        # Default threshold (0.40) keeps it
        assert (
            filter_low_confidence(profile, threshold=0.40)["core_identity"]["name"]
            is not None
        )
        # Higher threshold filters it out
        assert (
            filter_low_confidence(profile, threshold=0.70)["core_identity"]["name"]
            is None
        )

    def test_profile_serializable(self):
        import json

        profile = _make_full_profile()
        serialized = json.dumps(profile)
        deserialized = json.loads(serialized)
        assert deserialized["core_identity"]["name"]["value"] == "Maria"
        assert deserialized["_version"] == 3


class TestFirestoreClientCaching:
    """Verify Firestore client is cached as a singleton (not recreated per request)."""

    def test_firestore_client_cache_dict_exists(self):
        from src.data.guest_profile import _firestore_client_cache

        assert isinstance(_firestore_client_cache, dict)

    def test_clear_firestore_client_cache(self):
        from src.data.guest_profile import (
            _firestore_client_cache,
            clear_firestore_client_cache,
        )

        _firestore_client_cache["client"] = "mock_client"
        clear_firestore_client_cache()
        assert _firestore_client_cache == {}

    @pytest.mark.asyncio
    async def test_get_firestore_client_returns_none_without_gcp(self):
        """Without GCP SDK, _get_firestore_client returns None (in-memory fallback)."""
        from src.data.guest_profile import (
            _get_firestore_client,
            clear_firestore_client_cache,
        )

        clear_firestore_client_cache()
        result = await _get_firestore_client()
        assert result is None
