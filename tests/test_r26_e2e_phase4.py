"""R26 E2E integration tests for Phase 4 features — deterministic subset.

Mock purge R111: Retained only deterministic tests that do not depend on
MagicMock/AsyncMock/@patch. All behavioral validation uses live eval.

Covers: property-aware helplines, CASINO_PROFILES lookup, suggestion_offered
persistence (_keep_max reducer), _merge_dicts reducer.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.prompts import (
    HEART_ESCALATION_LANGUAGE,
    SENTIMENT_TONE_GUIDES,
    get_persona_style,
    get_responsible_gaming_helplines,
)
from src.agent.state import _keep_max, _merge_dicts
from src.casino.config import CASINO_PROFILES, get_casino_profile


# ---------------------------------------------------------------------------
# 4. Property-Aware Helplines (deterministic)
# ---------------------------------------------------------------------------


class TestPropertyAwareHelplines:
    """Test that helplines are correct per property/state."""

    def test_hard_rock_ac_returns_nj_helplines(self):
        """Hard Rock AC (NJ) returns NJ-specific helplines."""
        helplines = get_responsible_gaming_helplines("hard_rock_ac")
        assert "1-800-GAMBLER" in helplines
        assert "NJ" in helplines
        assert "National Problem Gambling Helpline" in helplines

    def test_mohegan_sun_returns_ct_helplines(self):
        """Mohegan Sun (CT) returns CT-specific helplines."""
        helplines = get_responsible_gaming_helplines("mohegan_sun")
        assert "CT" in helplines
        assert "1-888-789-7777" in helplines
        assert "1-800-MY-RESET" in helplines

    def test_foxwoods_returns_ct_helplines(self):
        """Foxwoods (CT) returns CT-specific helplines."""
        helplines = get_responsible_gaming_helplines("foxwoods")
        assert "CT" in helplines
        assert "1-800-MY-RESET" in helplines

    def test_unknown_casino_returns_default(self):
        """Unknown casino ID returns default (CT) helplines."""
        helplines = get_responsible_gaming_helplines("unknown_casino")
        assert "1-800-MY-RESET" in helplines
        assert "Connecticut" in helplines or "CT" in helplines

    def test_none_casino_returns_default(self):
        """No casino_id returns default helplines."""
        helplines = get_responsible_gaming_helplines(None)
        assert "1-800-MY-RESET" in helplines

    def test_nj_helplines_not_in_ct_property(self):
        """CT property should NOT return NJ-specific state helplines."""
        helplines = get_responsible_gaming_helplines("mohegan_sun")
        assert "NJ " not in helplines


# ---------------------------------------------------------------------------
# 5. CASINO_PROFILES Lookup (deterministic)
# ---------------------------------------------------------------------------


class TestCasinoProfiles:
    """Test CASINO_PROFILES data integrity and lookup."""

    def test_all_profiles_have_required_sections(self):
        """Every profile has branding, regulations, operational, prompts sections."""
        required_sections = {"branding", "regulations", "operational", "prompts"}
        for casino_id, profile in CASINO_PROFILES.items():
            for section in required_sections:
                assert section in profile, (
                    f"Casino '{casino_id}' missing section '{section}'"
                )

    def test_mohegan_sun_profile(self):
        """Mohegan Sun has correct state, persona, and helpline data."""
        profile = get_casino_profile("mohegan_sun")
        assert profile["regulations"]["state"] == "CT"
        assert profile["branding"]["persona_name"] == "Seven"
        assert profile["regulations"]["responsible_gaming_helpline"] == "1-800-MY-RESET"

    def test_hard_rock_ac_profile(self):
        """Hard Rock AC has correct NJ state and persona."""
        profile = get_casino_profile("hard_rock_ac")
        assert profile["regulations"]["state"] == "NJ"
        assert profile["branding"]["persona_name"] == "Ace"
        assert profile["regulations"]["responsible_gaming_helpline"] == "1-800-GAMBLER"

    def test_foxwoods_profile(self):
        """Foxwoods has correct CT state and persona."""
        profile = get_casino_profile("foxwoods")
        assert profile["regulations"]["state"] == "CT"
        assert profile["branding"]["persona_name"] == "Foxy"

    def test_unknown_casino_returns_default_config(self):
        """Unknown casino_id returns DEFAULT_CONFIG."""
        from src.casino.config import DEFAULT_CONFIG

        profile = get_casino_profile("nonexistent_casino")
        assert profile["_id"] == "default"

    def test_each_profile_has_features_section(self):
        """Every profile has a features section matching DEFAULT_FEATURES keys."""
        from src.casino.feature_flags import DEFAULT_FEATURES

        for casino_id, profile in CASINO_PROFILES.items():
            assert "features" in profile, f"Casino '{casino_id}' missing 'features'"
            profile_features = set(profile["features"].keys())
            default_features = set(DEFAULT_FEATURES.keys())
            assert profile_features == default_features, (
                f"Casino '{casino_id}' features mismatch: "
                f"missing={default_features - profile_features}, "
                f"extra={profile_features - default_features}"
            )

    def test_all_profiles_have_id(self):
        """Every profile has a _id matching its dict key."""
        for casino_id, profile in CASINO_PROFILES.items():
            assert profile["_id"] == casino_id


# ---------------------------------------------------------------------------
# 6. suggestion_offered Persistence (_keep_max reducer — deterministic)
# ---------------------------------------------------------------------------


class TestSuggestionOfferedPersistence:
    """Test that suggestion_offered persists across turns via _keep_max reducer."""

    def test_keep_max_preserves_one(self):
        """_keep_max(1, 0) = 1 -- once offered, stays offered."""
        assert _keep_max(1, 0) == 1

    def test_keep_max_initial(self):
        """_keep_max(0, 0) = 0 -- initial state stays zero."""
        assert _keep_max(0, 0) == 0

    def test_keep_max_updates(self):
        """_keep_max(0, 1) = 1 -- new offering is recorded."""
        assert _keep_max(0, 1) == 1

    def test_keep_max_true_persists(self):
        """Once set to 1, subsequent 0 resets don't reduce it."""
        current = 0
        current = _keep_max(current, 1)
        assert current == 1
        current = _keep_max(current, 0)
        assert current == 1
        current = _keep_max(current, 0)
        assert current == 1


# ---------------------------------------------------------------------------
# 7. _merge_dicts Reducer (deterministic)
# ---------------------------------------------------------------------------


class TestMergeDictsReducer:
    """Test that extracted_fields accumulate across turns."""

    def test_merge_empty_preserves_existing(self):
        """Merging empty dict preserves existing fields."""
        existing = {"name": "Sarah", "party_size": 4}
        result = _merge_dicts(existing, {})
        assert result == {"name": "Sarah", "party_size": 4}

    def test_merge_new_field_adds(self):
        """New field is added to existing fields."""
        existing = {"name": "Sarah"}
        result = _merge_dicts(existing, {"party_size": 4})
        assert result == {"name": "Sarah", "party_size": 4}

    def test_merge_overwrites_same_key(self):
        """Same key from new dict overwrites existing value."""
        existing = {"name": "Sarah"}
        result = _merge_dicts(existing, {"name": "Mike"})
        assert result == {"name": "Mike"}

    def test_merge_multiple_turns(self):
        """Simulate 3 turns of field accumulation."""
        state = {}
        state = _merge_dicts(state, {"name": "Sarah"})
        state = _merge_dicts(state, {"party_size": 6})
        state = _merge_dicts(state, {"occasion": "birthday"})
        assert state == {"name": "Sarah", "party_size": 6, "occasion": "birthday"}
