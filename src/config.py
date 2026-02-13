"""Centralized configuration via pydantic-settings.

All magic numbers, model names, paths, and tuning knobs live here.
Override any value via environment variable (e.g., ``MODEL_NAME=gemini-2.5-pro``).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Property ---
    PROPERTY_NAME: str = "Mohegan Sun"
    PROPERTY_DATA_PATH: str = "data/mohegan_sun.json"

    # --- LLM ---
    MODEL_NAME: str = "gemini-2.5-flash"
    MODEL_TEMPERATURE: float = 0.3

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "models/text-embedding-004"

    # --- RAG ---
    CHROMA_PERSIST_DIR: str = "data/chroma"
    RAG_TOP_K: int = 5
    RAG_CHUNK_SIZE: int = 800
    RAG_CHUNK_OVERLAP: int = 100

    # --- API ---
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8080"]
    RATE_LIMIT_CHAT: int = 20  # requests per minute
    SSE_TIMEOUT_SECONDS: int = 60

    # --- Observability ---
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    VERSION: str = "0.1.0"

    model_config = {"env_prefix": "", "case_sensitive": True}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
