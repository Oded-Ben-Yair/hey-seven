"""Tests for casino profile regulatory completeness.

R52 fix D10: Every casino profile must have complete regulatory
fields to avoid compliance gaps.
"""

import pytest


class TestJurisdictionCompleteness:
    """Verify casino profiles have required regulatory fields."""

    def test_all_profiles_have_regulations(self):
        from src.casino.config import CASINO_PROFILES

        for casino_id, profile in CASINO_PROFILES.items():
            assert "regulations" in profile, f"{casino_id} missing regulations"

    def test_all_profiles_have_gaming_age(self):
        from src.casino.config import CASINO_PROFILES

        for casino_id, profile in CASINO_PROFILES.items():
            regs = profile.get("regulations", {})
            assert "gaming_age_minimum" in regs, (
                f"{casino_id} missing gaming_age_minimum"
            )

    def test_all_profiles_have_helplines(self):
        from src.casino.config import CASINO_PROFILES

        for casino_id, profile in CASINO_PROFILES.items():
            regs = profile.get("regulations", {})
            assert "responsible_gaming_helpline" in regs, (
                f"{casino_id} missing responsible_gaming_helpline"
            )
            assert "state_helpline" in regs, (
                f"{casino_id} missing state_helpline"
            )

    def test_all_profiles_have_self_exclusion(self):
        from src.casino.config import CASINO_PROFILES

        for casino_id, profile in CASINO_PROFILES.items():
            regs = profile.get("regulations", {})
            assert "self_exclusion_authority" in regs, (
                f"{casino_id} missing self_exclusion_authority"
            )

    def test_all_profiles_have_state(self):
        from src.casino.config import CASINO_PROFILES

        for casino_id, profile in CASINO_PROFILES.items():
            regs = profile.get("regulations", {})
            assert "state" in regs, f"{casino_id} missing state"

    def test_required_fields_frozenset_matches_profiles(self):
        """Verify _REQUIRED_PROFILE_FIELDS covers what tests check."""
        from src.casino.config import _REQUIRED_PROFILE_FIELDS

        expected = {
            "state",
            "gaming_age_minimum",
            "responsible_gaming_helpline",
            "state_helpline",
            "self_exclusion_authority",
        }
        assert _REQUIRED_PROFILE_FIELDS == expected

    def test_default_profile_has_all_fields(self):
        from src.casino.config import get_casino_profile

        profile = get_casino_profile("mohegan_sun")
        assert profile is not None
        assert "branding" in profile
        assert "regulations" in profile

    def test_unknown_casino_returns_default(self):
        from src.casino.config import get_casino_profile

        profile = get_casino_profile("nonexistent_casino")
        assert profile is not None  # Should return default, not crash
