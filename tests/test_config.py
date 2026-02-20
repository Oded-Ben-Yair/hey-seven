"""Tests for centralized configuration (src/config.py)."""

import os
from unittest.mock import patch


class TestSettings:
    def test_default_settings_load(self):
        """Settings load with defaults when no env vars are set."""
        from src.config import Settings

        s = Settings()
        assert s.PROPERTY_NAME == "Mohegan Sun"
        assert s.MODEL_NAME == "gemini-2.5-flash"
        assert s.MODEL_TEMPERATURE == 0.3
        assert s.RAG_TOP_K == 5
        assert s.RAG_CHUNK_SIZE == 800
        assert s.RATE_LIMIT_CHAT == 20
        assert s.VERSION == "1.0.0"

    def test_env_var_overrides(self):
        """Environment variables override default settings."""
        from src.config import Settings

        with patch.dict(os.environ, {"MODEL_NAME": "gemini-2.5-pro", "RAG_TOP_K": "10"}):
            s = Settings()
            assert s.MODEL_NAME == "gemini-2.5-pro"
            assert s.RAG_TOP_K == 10

    def test_settings_are_reusable(self):
        """get_settings returns a working Settings instance each time."""
        from src.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1.PROPERTY_NAME == s2.PROPERTY_NAME
        assert s1.VERSION == s2.VERSION

    def test_allowed_origins_default(self):
        """Default ALLOWED_ORIGINS is a list with localhost."""
        from src.config import Settings

        s = Settings()
        assert isinstance(s.ALLOWED_ORIGINS, list)
        assert "http://localhost:8080" in s.ALLOWED_ORIGINS

    def test_google_api_key_field_exists(self):
        """GOOGLE_API_KEY is declared in Settings."""
        from src.config import Settings

        s = Settings()
        assert hasattr(s, "GOOGLE_API_KEY")

    def test_google_api_key_from_env(self):
        """GOOGLE_API_KEY can be set via environment variable."""
        from src.config import Settings

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key-123"}):
            s = Settings()
            assert s.GOOGLE_API_KEY.get_secret_value() == "test-key-123"

    def test_llm_safety_params_exist(self):
        """LLM safety parameters exist with correct defaults."""
        from src.config import Settings

        s = Settings()
        assert s.MODEL_TIMEOUT == 30
        assert s.MODEL_MAX_RETRIES == 2
        assert s.MODEL_MAX_OUTPUT_TOKENS == 2048

    def test_new_round1_settings_exist(self):
        """Round 1 review-driven settings exist with correct defaults."""
        from src.config import Settings

        s = Settings()
        assert s.PROPERTY_WEBSITE == "mohegansun.com"
        assert s.PROPERTY_PHONE == "1-888-226-7711"
        assert s.RAG_MIN_RELEVANCE_SCORE == 0.3
        assert s.RATE_LIMIT_MAX_CLIENTS == 10000
        assert s.MAX_REQUEST_BODY_SIZE == 65536
        assert s.MAX_MESSAGE_LIMIT == 40

    def test_graph_recursion_limit_default(self):
        """GRAPH_RECURSION_LIMIT has sensible default of 10."""
        from src.config import Settings

        s = Settings()
        assert s.GRAPH_RECURSION_LIMIT == 10

    def test_graph_recursion_limit_overridable(self):
        """GRAPH_RECURSION_LIMIT can be overridden via env var."""
        from src.config import Settings

        with patch.dict(os.environ, {"GRAPH_RECURSION_LIMIT": "20"}):
            s = Settings()
            assert s.GRAPH_RECURSION_LIMIT == 20

    def test_chunk_overlap_must_be_less_than_chunk_size(self):
        """RAG_CHUNK_OVERLAP >= RAG_CHUNK_SIZE raises ValueError."""
        import pytest

        from src.config import Settings

        with patch.dict(os.environ, {"RAG_CHUNK_OVERLAP": "800", "RAG_CHUNK_SIZE": "800"}):
            with pytest.raises(ValueError, match="RAG_CHUNK_OVERLAP"):
                Settings()

    def test_valid_chunk_params_pass(self):
        """RAG_CHUNK_OVERLAP < RAG_CHUNK_SIZE validates successfully."""
        from src.config import Settings

        with patch.dict(os.environ, {"RAG_CHUNK_OVERLAP": "100", "RAG_CHUNK_SIZE": "800"}):
            s = Settings()
            assert s.RAG_CHUNK_OVERLAP == 100
            assert s.RAG_CHUNK_SIZE == 800

    def test_consent_hmac_rejects_default_when_sms_enabled(self):
        """SMS_ENABLED=True with default HMAC secret raises ValueError."""
        import pytest

        from src.config import Settings

        with patch.dict(os.environ, {"SMS_ENABLED": "true"}):
            with pytest.raises(ValueError, match="CONSENT_HMAC_SECRET must be set"):
                Settings()

    def test_consent_hmac_accepts_custom_when_sms_enabled(self):
        """SMS_ENABLED=True with custom HMAC secret passes validation."""
        from src.config import Settings

        with patch.dict(os.environ, {
            "SMS_ENABLED": "true",
            "CONSENT_HMAC_SECRET": "my-secure-production-secret",
        }):
            s = Settings()
            assert s.SMS_ENABLED is True
            assert s.CONSENT_HMAC_SECRET.get_secret_value() == "my-secure-production-secret"

    def test_consent_hmac_allows_default_when_sms_disabled(self):
        """SMS_ENABLED=False (default) allows the default HMAC secret."""
        from src.config import Settings

        s = Settings()
        assert s.SMS_ENABLED is False
        assert s.CONSENT_HMAC_SECRET.get_secret_value() == "change-me-in-production"


class TestProductionSecretValidation:
    """Tests for production secret hard-fail validation (R2 security fix)."""

    def test_production_rejects_empty_api_key(self):
        """ENVIRONMENT=production with empty API_KEY raises ValueError."""
        import pytest

        from src.config import Settings

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with pytest.raises(ValueError, match="API_KEY must be set"):
                Settings()

    def test_production_rejects_empty_cms_webhook_secret(self):
        """ENVIRONMENT=production with empty CMS_WEBHOOK_SECRET raises ValueError."""
        import pytest

        from src.config import Settings

        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "API_KEY": "test-api-key",
        }):
            with pytest.raises(ValueError, match="CMS_WEBHOOK_SECRET must be set"):
                Settings()

    def test_production_rejects_missing_telnyx_key_when_sms_enabled(self):
        """ENVIRONMENT=production with SMS_ENABLED=True but no TELNYX_PUBLIC_KEY raises ValueError."""
        import pytest

        from src.config import Settings

        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "API_KEY": "test-api-key",
            "CMS_WEBHOOK_SECRET": "test-cms-secret",
            "SMS_ENABLED": "true",
            "CONSENT_HMAC_SECRET": "secure-hmac-secret",
        }):
            with pytest.raises(ValueError, match="TELNYX_PUBLIC_KEY must be set"):
                Settings()

    def test_production_passes_with_all_secrets_set(self):
        """ENVIRONMENT=production with all required secrets passes validation."""
        from src.config import Settings

        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "API_KEY": "prod-api-key",
            "CMS_WEBHOOK_SECRET": "prod-cms-secret",
        }):
            s = Settings()
            assert s.ENVIRONMENT == "production"
            assert s.API_KEY.get_secret_value() == "prod-api-key"
            assert s.CMS_WEBHOOK_SECRET.get_secret_value() == "prod-cms-secret"

    def test_development_allows_empty_secrets(self):
        """ENVIRONMENT=development (default) allows empty secrets for local testing."""
        from src.config import Settings

        s = Settings()
        assert s.ENVIRONMENT == "development"
        assert s.API_KEY.get_secret_value() == ""
        assert s.CMS_WEBHOOK_SECRET.get_secret_value() == ""

    def test_trusted_proxies_defaults_to_none(self):
        """TRUSTED_PROXIES defaults to None (trust nobody's XFF headers)."""
        from src.config import Settings

        s = Settings()
        assert s.TRUSTED_PROXIES is None
