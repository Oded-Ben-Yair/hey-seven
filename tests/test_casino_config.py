"""Tests for per-casino configuration and feature flags.

Covers:
- Default config structure and values
- Deep merge logic
- Feature flag defaults, overrides, and unknown flags
- Branding and regulation default values
- Casino profiles (static CASINO_PROFILES entries)
- Edge cases (empty casino_id, greeting templates)

Mock-based Firestore tests removed (mock purge R111).
"""

import copy
import types

import pytest

from src.casino.config import (
    CASINO_PROFILES,
    DEFAULT_CONFIG,
    _deep_merge,
    clear_config_cache,
    get_casino_profile,
)
from src.casino.feature_flags import (
    DEFAULT_FEATURES,
    FeatureFlags,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear config cache between tests."""
    clear_config_cache()
    yield
    clear_config_cache()


# ---------------------------------------------------------------------------
# TestDeepMerge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_shallow_merge(self):
        """Non-nested keys are replaced."""
        base = {"a": 1, "b": 2}
        overrides = {"b": 3, "c": 4}
        result = _deep_merge(base, overrides)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_dict_merge(self):
        """Nested dicts are merged recursively."""
        base = {"x": {"a": 1, "b": 2}}
        overrides = {"x": {"b": 3, "c": 4}}
        result = _deep_merge(base, overrides)
        assert result == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_override_replaces_non_dict_with_dict(self):
        """Override can replace a non-dict value with a dict."""
        base = {"x": 1}
        overrides = {"x": {"nested": True}}
        result = _deep_merge(base, overrides)
        assert result == {"x": {"nested": True}}

    def test_does_not_mutate_base(self):
        """Deep merge does not mutate the base dict."""
        base = {"x": {"a": 1}}
        original_base = copy.deepcopy(base)
        _deep_merge(base, {"x": {"b": 2}})
        assert base == original_base


# ---------------------------------------------------------------------------
# TestFeatureFlagDefaults
# ---------------------------------------------------------------------------


class TestFeatureFlagDefaults:
    def test_dict_of_default_features_is_a_copy(self):
        """dict(DEFAULT_FEATURES) returns a mutable copy, not the original."""
        defaults = dict(DEFAULT_FEATURES)
        assert defaults == DEFAULT_FEATURES
        defaults["sms_enabled"] = True
        # Original unchanged
        assert DEFAULT_FEATURES["sms_enabled"] is False

    def test_default_features_is_immutable(self):
        """DEFAULT_FEATURES cannot be mutated (MappingProxyType)."""
        assert isinstance(DEFAULT_FEATURES, types.MappingProxyType)
        with pytest.raises(TypeError):
            DEFAULT_FEATURES["sms_enabled"] = True

    def test_feature_flags_typeddict_parity(self):
        """FeatureFlags TypedDict declares every key in DEFAULT_FEATURES (and vice versa)."""
        typeddict_keys = set(FeatureFlags.__annotations__)
        default_keys = set(DEFAULT_FEATURES.keys())
        assert typeddict_keys == default_keys, (
            f"FeatureFlags drift: "
            f"missing={default_keys - typeddict_keys}, "
            f"extra={typeddict_keys - default_keys}"
        )

    def test_default_config_features_parity_with_default_features(self):
        """DEFAULT_CONFIG['features'] keys must match DEFAULT_FEATURES keys (no drift)."""
        config_keys = set(DEFAULT_CONFIG["features"].keys())
        default_keys = set(DEFAULT_FEATURES.keys())
        assert config_keys == default_keys, (
            f"Feature flag drift: "
            f"missing={default_keys - config_keys}, "
            f"extra={config_keys - default_keys}"
        )


# ---------------------------------------------------------------------------
# TestDefaultConfigStructure
# ---------------------------------------------------------------------------


class TestDefaultConfigStructure:
    def test_greeting_templates_use_safe_substitute_syntax(self):
        """Greeting templates use $placeholder syntax (safe_substitute), not {placeholder} (.format)."""
        for key in ("greeting_template", "greeting_template_es"):
            template = DEFAULT_CONFIG["prompts"][key]
            assert "{" not in template, (
                f"{key} uses unsafe .format() placeholder syntax"
            )
            assert "$" in template, f"{key} missing $placeholder for safe_substitute"

    def test_default_config_has_all_sections(self):
        """DEFAULT_CONFIG includes all defined sections."""
        for section in (
            "features",
            "prompts",
            "branding",
            "regulations",
            "operational",
            "rag",
        ):
            assert section in DEFAULT_CONFIG, f"Missing section: {section}"

    def test_default_branding_values(self):
        """Default branding config has expected values."""
        branding = DEFAULT_CONFIG["branding"]
        assert branding["persona_name"] == "Seven"
        assert branding["tone"] == "warm_professional"
        assert branding["formality_level"] == "casual_respectful"
        assert branding["emoji_allowed"] is False
        assert branding["exclamation_limit"] == 1

    def test_default_regulation_values(self):
        """Default regulation config has expected values."""
        regs = DEFAULT_CONFIG["regulations"]
        assert regs["state"] == "CT"
        assert regs["gaming_age_minimum"] == 21
        assert regs["ai_disclosure_required"] is True
        assert regs["quiet_hours_start"] == "21:00"
        assert regs["quiet_hours_end"] == "08:00"
        assert regs["responsible_gaming_helpline"] == "1-800-MY-RESET"

    def test_default_rag_values(self):
        """Default RAG config has expected values."""
        rag = DEFAULT_CONFIG["rag"]
        assert rag["min_relevance_score"] == 0.35
        assert rag["top_k"] == 5
        assert rag["embedding_model"] == "gemini-embedding-001"
        assert rag["embedding_dimensions"] == 768

    def test_default_operational_values(self):
        """Default operational config has expected values."""
        ops = DEFAULT_CONFIG["operational"]
        assert ops["max_messages_per_guest_per_day"] == 20
        assert ops["session_timeout_hours"] == 48

    def test_default_prompts_values(self):
        """Default prompts have greeting templates and fallback message."""
        prompts = DEFAULT_CONFIG["prompts"]
        assert "guest_name" in prompts["greeting_template"]
        assert "persona_name" in prompts["greeting_template"]
        assert prompts["system_prompt_override"] is None
        assert len(prompts["fallback_message"]) > 0

    def test_config_is_json_serializable(self):
        """Config round-trips through JSON serialization."""
        import json

        serialized = json.dumps(DEFAULT_CONFIG)
        deserialized = json.loads(serialized)
        assert deserialized["branding"]["persona_name"] == "Seven"
        assert deserialized["regulations"]["gaming_age_minimum"] == 21


# ---------------------------------------------------------------------------
# TestCasinoProfiles (Phase 6: Step 4c)
# ---------------------------------------------------------------------------


class TestCasinoProfiles:
    """Tests for static CASINO_PROFILES entries and get_casino_profile()."""

    def test_parx_casino_profile_pa_helplines(self):
        """Parx Casino profile has correct PA helplines and state."""
        profile = get_casino_profile("parx_casino")
        regs = profile["regulations"]
        assert regs["state"] == "PA"
        assert regs["gaming_age_minimum"] == 21
        assert regs["responsible_gaming_helpline"] == "1-800-GAMBLER"
        assert regs["state_helpline"] == "1-800-848-1880"
        assert regs["self_exclusion_authority"] == "PA Gaming Control Board"

    def test_parx_casino_branding(self):
        """Parx Casino uses 'Lucky' persona."""
        profile = get_casino_profile("parx_casino")
        assert profile["branding"]["persona_name"] == "Lucky"

    def test_wynn_las_vegas_profile_nv_helplines(self):
        """Wynn Las Vegas profile has correct NV helplines, state, and timezone."""
        profile = get_casino_profile("wynn_las_vegas")
        regs = profile["regulations"]
        assert regs["state"] == "NV"
        assert regs["gaming_age_minimum"] == 21
        # R52 fix C2: 1-800-522-4700 rebranded to 1-800-GAMBLER (NCPG, 2022)
        assert regs["responsible_gaming_helpline"] == "1-800-GAMBLER"
        assert regs["self_exclusion_authority"] == "Nevada Gaming Control Board"
        # R57 fix D10: NV self-exclusion per NGC Regulation 5.170 (not NRS 463.368)
        assert "petition" in regs["self_exclusion_options"].lower()
        assert profile["operational"]["timezone"] == "America/Los_Angeles"

    def test_wynn_las_vegas_luxury_branding(self):
        """Wynn Las Vegas uses luxury branding with no exclamation marks."""
        profile = get_casino_profile("wynn_las_vegas")
        branding = profile["branding"]
        assert branding["persona_name"] == "Wynn Host"
        assert branding["tone"] == "luxury"
        assert branding["formality_level"] == "formal"
        assert branding["exclamation_limit"] == 0

    def test_all_profiles_have_required_sections(self):
        """All 5 casino profiles have required top-level config sections."""
        required_sections = {
            "regulations",
            "branding",
            "operational",
            "features",
            "prompts",
            "rag",
        }
        for casino_id, profile in CASINO_PROFILES.items():
            for section in required_sections:
                assert section in profile, (
                    f"Profile '{casino_id}' missing required section: {section}"
                )

    def test_unknown_casino_returns_default_config(self):
        """get_casino_profile with unknown ID returns a deepcopy of DEFAULT_CONFIG."""
        profile = get_casino_profile("unknown_casino")
        # R36: returns deepcopy, not identity reference (prevents mutation of global)
        assert profile == DEFAULT_CONFIG
        assert profile is not DEFAULT_CONFIG, (
            "Should return deepcopy, not direct reference"
        )
