"""Tests for per-casino configuration and feature flags.

Covers:
- Default config loading and structure
- Cache TTL behavior and invalidation
- Firestore fallback to defaults
- Deep merge of config overrides
- Feature flag defaults, overrides, and unknown flags
- Branding and regulation default values
- Edge cases (empty casino_id, None values)
"""

import copy
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.casino.config import (
    DEFAULT_CONFIG,
    _CONFIG_TTL_SECONDS,
    _deep_merge,
    clear_config_cache,
    get_casino_config,
)
from src.casino.feature_flags import (
    DEFAULT_FEATURES,
    get_default_features,
    get_feature_flags,
    is_feature_enabled,
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
# TestCasinoConfig
# ---------------------------------------------------------------------------


class TestCasinoConfig:
    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_returns_defaults_when_firestore_unavailable(self, mock_client):
        """Without Firestore, get_casino_config returns DEFAULT_CONFIG values."""
        config = await get_casino_config("mohegan_sun")
        assert config["branding"]["persona_name"] == "Seven"
        assert config["regulations"]["gaming_age_minimum"] == 21
        assert config["operational"]["timezone"] == "America/New_York"
        assert config["rag"]["top_k"] == 5

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_config_id_matches_casino_id(self, mock_client):
        """The _id field is set to the requested casino_id."""
        config = await get_casino_config("foxwoods")
        assert config["_id"] == "foxwoods"

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_config_has_all_sections(self, mock_client):
        """Returned config includes all defined sections."""
        config = await get_casino_config("mohegan_sun")
        for section in ("features", "prompts", "branding", "regulations", "operational", "rag"):
            assert section in config, f"Missing section: {section}"

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_cache_returns_same_object_within_ttl(self, mock_client):
        """Second call within TTL returns cached config without re-fetching."""
        config1 = await get_casino_config("mohegan_sun")
        config2 = await get_casino_config("mohegan_sun")
        assert config1 is config2

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_cache_expires_after_ttl(self, mock_client):
        """After TTL expires, config is re-fetched."""
        config1 = await get_casino_config("mohegan_sun")

        # Manually expire the cache entry by clearing it
        # (TTLCache handles TTL-based expiry automatically; we simulate
        # expiry by removing the entry so the next call re-fetches.)
        from src.casino.config import _config_cache

        _config_cache.clear()

        config2 = await get_casino_config("mohegan_sun")
        # After expiry, a new deepcopy is returned
        assert config1 is not config2

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_different_casinos_get_separate_cache_entries(self, mock_client):
        """Each casino_id gets its own cache entry."""
        config_a = await get_casino_config("casino_a")
        config_b = await get_casino_config("casino_b")
        assert config_a["_id"] == "casino_a"
        assert config_b["_id"] == "casino_b"

    async def test_firestore_overrides_merged_with_defaults(self):
        """Firestore overrides are deep-merged with DEFAULT_CONFIG."""
        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {
            "branding": {"persona_name": "Lucky"},
            "regulations": {"state": "NV"},
        }

        mock_doc_ref = MagicMock()
        mock_doc_ref.get = AsyncMock(return_value=mock_doc)

        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        with patch("src.casino.config._get_firestore_client", return_value=mock_db):
            config = await get_casino_config("vegas_casino")

        # Overridden values
        assert config["branding"]["persona_name"] == "Lucky"
        assert config["regulations"]["state"] == "NV"
        # Default values preserved
        assert config["branding"]["tone"] == "warm_professional"
        assert config["regulations"]["gaming_age_minimum"] == 21
        assert config["operational"]["timezone"] == "America/New_York"

    async def test_firestore_exception_falls_back_to_defaults(self):
        """When Firestore raises, defaults are returned gracefully."""
        mock_doc_ref = MagicMock()
        mock_doc_ref.get = AsyncMock(side_effect=RuntimeError("Connection lost"))

        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        with patch("src.casino.config._get_firestore_client", return_value=mock_db):
            config = await get_casino_config("broken_casino")

        assert config["branding"]["persona_name"] == "Seven"
        assert config["_id"] == "broken_casino"

    async def test_firestore_doc_not_exists_returns_defaults(self):
        """When Firestore doc does not exist, defaults are returned."""
        mock_doc = MagicMock()
        mock_doc.exists = False

        mock_doc_ref = MagicMock()
        mock_doc_ref.get = AsyncMock(return_value=mock_doc)

        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        with patch("src.casino.config._get_firestore_client", return_value=mock_db):
            config = await get_casino_config("new_casino")

        assert config["branding"]["persona_name"] == "Seven"
        assert config["_id"] == "new_casino"

    def test_greeting_templates_use_safe_substitute_syntax(self):
        """Greeting templates use $placeholder syntax (safe_substitute), not {placeholder} (.format)."""
        from src.casino.config import DEFAULT_CONFIG

        for key in ("greeting_template", "greeting_template_es"):
            template = DEFAULT_CONFIG["prompts"][key]
            assert "{" not in template, f"{key} uses unsafe .format() placeholder syntax"
            assert "$" in template, f"{key} missing $placeholder for safe_substitute"

    def test_default_config_features_parity_with_default_features(self):
        """DEFAULT_CONFIG['features'] keys must match DEFAULT_FEATURES keys (no drift)."""
        from src.casino.config import DEFAULT_CONFIG

        config_keys = set(DEFAULT_CONFIG["features"].keys())
        default_keys = set(DEFAULT_FEATURES.keys())
        assert config_keys == default_keys, (
            f"Feature flag drift: "
            f"missing={default_keys - config_keys}, "
            f"extra={config_keys - default_keys}"
        )


# ---------------------------------------------------------------------------
# TestFeatureFlags
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_default_flags_all_present(self, mock_client):
        """All default feature flags are returned."""
        flags = await get_feature_flags("mohegan_sun")
        for flag_name, flag_value in DEFAULT_FEATURES.items():
            assert flag_name in flags
            assert flags[flag_name] == flag_value

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_core_features_on_by_default(self, mock_client):
        """Core features are enabled by default."""
        flags = await get_feature_flags("mohegan_sun")
        assert flags["ai_disclosure_enabled"] is True
        assert flags["whisper_planner_enabled"] is True
        assert flags["comp_agent_enabled"] is True
        assert flags["spanish_support_enabled"] is True

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_experimental_features_off_by_default(self, mock_client):
        """Experimental features are disabled by default."""
        flags = await get_feature_flags("mohegan_sun")
        assert flags["outbound_campaigns_enabled"] is False
        assert flags["hitl_interrupt_enabled"] is False
        assert flags["sms_enabled"] is False

    async def test_casino_overrides_merge_with_defaults(self):
        """Casino-specific flag overrides are merged with defaults."""
        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {
            "features": {
                "sms_enabled": True,
                "outbound_campaigns_enabled": True,
            },
        }

        mock_doc_ref = MagicMock()
        mock_doc_ref.get = AsyncMock(return_value=mock_doc)

        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        with patch("src.casino.config._get_firestore_client", return_value=mock_db):
            flags = await get_feature_flags("premium_casino")

        # Overridden
        assert flags["sms_enabled"] is True
        assert flags["outbound_campaigns_enabled"] is True
        # Defaults preserved
        assert flags["ai_disclosure_enabled"] is True
        assert flags["hitl_interrupt_enabled"] is False

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_unknown_flag_returns_false(self, mock_client):
        """is_feature_enabled returns False for unknown flag names."""
        result = await is_feature_enabled("mohegan_sun", "nonexistent_flag")
        assert result is False

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_is_feature_enabled_true(self, mock_client):
        """is_feature_enabled returns True for enabled flags."""
        result = await is_feature_enabled("mohegan_sun", "ai_disclosure_enabled")
        assert result is True

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_is_feature_enabled_false(self, mock_client):
        """is_feature_enabled returns False for disabled flags."""
        result = await is_feature_enabled("mohegan_sun", "sms_enabled")
        assert result is False

    def test_get_default_features_returns_copy(self):
        """get_default_features returns a copy, not the original dict."""
        defaults = get_default_features()
        assert defaults == DEFAULT_FEATURES
        defaults["sms_enabled"] = True
        # Original unchanged
        assert DEFAULT_FEATURES["sms_enabled"] is False

    def test_default_features_is_immutable(self):
        """DEFAULT_FEATURES cannot be mutated (MappingProxyType)."""
        import types

        assert isinstance(DEFAULT_FEATURES, types.MappingProxyType)
        with pytest.raises(TypeError):
            DEFAULT_FEATURES["sms_enabled"] = True

    def test_get_default_features_returns_mutable_dict(self):
        """get_default_features returns a mutable dict, not a MappingProxyType."""
        defaults = get_default_features()
        assert isinstance(defaults, dict)
        # Should be mutable
        defaults["test_flag"] = True
        assert defaults["test_flag"] is True

    def test_feature_flags_typeddict_parity(self):
        """FeatureFlags TypedDict declares every key in DEFAULT_FEATURES (and vice versa)."""
        from src.casino.feature_flags import FeatureFlags

        typeddict_keys = set(FeatureFlags.__annotations__)
        default_keys = set(DEFAULT_FEATURES.keys())
        assert typeddict_keys == default_keys, (
            f"FeatureFlags drift: "
            f"missing={default_keys - typeddict_keys}, "
            f"extra={typeddict_keys - default_keys}"
        )


# ---------------------------------------------------------------------------
# TestBrandingConfig
# ---------------------------------------------------------------------------


class TestBrandingConfig:
    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_default_branding_values(self, mock_client):
        """Default branding config has expected values."""
        config = await get_casino_config("mohegan_sun")
        branding = config["branding"]
        assert branding["persona_name"] == "Seven"
        assert branding["tone"] == "warm_professional"
        assert branding["formality_level"] == "casual_respectful"
        assert branding["emoji_allowed"] is False
        assert branding["exclamation_limit"] == 1


# ---------------------------------------------------------------------------
# TestRegulationConfig
# ---------------------------------------------------------------------------


class TestRegulationConfig:
    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_default_regulation_values(self, mock_client):
        """Default regulation config has expected values."""
        config = await get_casino_config("mohegan_sun")
        regs = config["regulations"]
        assert regs["state"] == "CT"
        assert regs["gaming_age_minimum"] == 21
        assert regs["ai_disclosure_required"] is True
        assert regs["quiet_hours_start"] == "21:00"
        assert regs["quiet_hours_end"] == "08:00"
        assert regs["responsible_gaming_helpline"] == "1-800-522-4700"


# ---------------------------------------------------------------------------
# TestClearCache
# ---------------------------------------------------------------------------


class TestClearCache:
    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_clear_cache_removes_all_entries(self, mock_client):
        """clear_config_cache removes all cached configs."""
        await get_casino_config("casino_a")
        await get_casino_config("casino_b")

        from src.casino.config import _config_cache

        assert len(_config_cache) == 2
        clear_config_cache()
        assert len(_config_cache) == 0

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_clear_cache_forces_reload(self, mock_client):
        """After clearing cache, next call returns a fresh config."""
        config1 = await get_casino_config("mohegan_sun")
        clear_config_cache()
        config2 = await get_casino_config("mohegan_sun")
        # Different object after cache clear
        assert config1 is not config2


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
# TestCasinoConfigEdgeCases
# ---------------------------------------------------------------------------


class TestCasinoConfigEdgeCases:
    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_empty_casino_id(self, mock_client):
        """Empty string casino_id still returns valid config."""
        config = await get_casino_config("")
        assert config["_id"] == ""
        assert config["branding"]["persona_name"] == "Seven"

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_config_is_json_serializable(self, mock_client):
        """Config round-trips through JSON serialization."""
        import json

        config = await get_casino_config("mohegan_sun")
        serialized = json.dumps(config)
        deserialized = json.loads(serialized)
        assert deserialized["branding"]["persona_name"] == "Seven"
        assert deserialized["regulations"]["gaming_age_minimum"] == 21

    async def test_firestore_override_with_none_values(self):
        """None values in Firestore overrides are preserved."""
        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {
            "prompts": {"system_prompt_override": None},
        }

        mock_doc_ref = MagicMock()
        mock_doc_ref.get = AsyncMock(return_value=mock_doc)

        mock_collection = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = MagicMock()
        mock_db.collection.return_value = mock_collection

        with patch("src.casino.config._get_firestore_client", return_value=mock_db):
            config = await get_casino_config("test_casino")

        assert config["prompts"]["system_prompt_override"] is None

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_default_config_not_mutated_across_calls(self, mock_client):
        """Getting config for different casinos does not mutate DEFAULT_CONFIG."""
        original_default = copy.deepcopy(DEFAULT_CONFIG)
        await get_casino_config("casino_a")
        await get_casino_config("casino_b")
        # DEFAULT_CONFIG unchanged (deepcopy is used internally)
        assert DEFAULT_CONFIG["branding"]["persona_name"] == original_default["branding"]["persona_name"]
        assert DEFAULT_CONFIG["_id"] == original_default["_id"]

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_prompts_config_defaults(self, mock_client):
        """Default prompts have greeting templates and fallback message."""
        config = await get_casino_config("mohegan_sun")
        prompts = config["prompts"]
        assert "guest_name" in prompts["greeting_template"]
        assert "persona_name" in prompts["greeting_template"]
        assert prompts["system_prompt_override"] is None
        assert len(prompts["fallback_message"]) > 0

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_rag_config_defaults(self, mock_client):
        """Default RAG config has expected values."""
        config = await get_casino_config("mohegan_sun")
        rag = config["rag"]
        assert rag["min_relevance_score"] == 0.35
        assert rag["top_k"] == 5
        assert rag["embedding_model"] == "gemini-embedding-001"
        assert rag["embedding_dimensions"] == 768

    @patch("src.casino.config._get_firestore_client", return_value=None)
    async def test_operational_config_defaults(self, mock_client):
        """Default operational config has expected values."""
        config = await get_casino_config("mohegan_sun")
        ops = config["operational"]
        assert ops["max_messages_per_guest_per_day"] == 20
        assert ops["session_timeout_hours"] == 48
