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
        assert s.VERSION == "0.1.0"

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
            assert s.GOOGLE_API_KEY == "test-key-123"

    def test_llm_safety_params_exist(self):
        """LLM safety parameters exist with correct defaults."""
        from src.config import Settings

        s = Settings()
        assert s.MODEL_TIMEOUT == 30
        assert s.MODEL_MAX_RETRIES == 2
        assert s.MODEL_MAX_OUTPUT_TOKENS == 2048
