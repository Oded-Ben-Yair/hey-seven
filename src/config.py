"""Centralized configuration via pydantic-settings.

All magic numbers, model names, paths, and tuning knobs live here.
Override any value via environment variable (e.g., ``MODEL_NAME=gemini-2.5-pro``).
"""

from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Google API ---
    GOOGLE_API_KEY: SecretStr = SecretStr("")  # Required for production; langchain auto-detects from env

    # --- Property ---
    PROPERTY_NAME: str = "Mohegan Sun"
    PROPERTY_DATA_PATH: str = "data/mohegan_sun.json"
    PROPERTY_WEBSITE: str = "mohegansun.com"
    PROPERTY_PHONE: str = "1-888-226-7711"

    # --- LLM ---
    MODEL_NAME: str = "gemini-2.5-flash"
    MODEL_TEMPERATURE: float = 0.3
    MODEL_TIMEOUT: int = 30  # LLM call timeout in seconds
    MODEL_MAX_RETRIES: int = 2  # LLM call retry count
    MODEL_MAX_OUTPUT_TOKENS: int = 2048  # Max response tokens

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # --- RAG ---
    CHROMA_PERSIST_DIR: str = "data/chroma"
    RAG_TOP_K: int = 5
    RAG_CHUNK_SIZE: int = 800
    RAG_CHUNK_OVERLAP: int = 120  # 15% of chunk size (industry recommendation)
    RAG_MIN_RELEVANCE_SCORE: float = 0.3  # Minimum relevance score (0-1, higher = more relevant)

    # --- API ---
    API_KEY: SecretStr = SecretStr("")  # When set, /chat requires X-API-Key header
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8080"]
    RATE_LIMIT_CHAT: int = 20  # requests per minute
    RATE_LIMIT_MAX_CLIENTS: int = 10000  # max tracked client IPs (memory guard)
    SSE_TIMEOUT_SECONDS: int = 60
    MAX_REQUEST_BODY_SIZE: int = 65536  # 64 KB max request body

    # --- Agent ---
    MAX_MESSAGE_LIMIT: int = 40  # max total messages (human + AI) before forcing conversation end
    MAX_HISTORY_MESSAGES: int = 20  # sliding window: only last N messages sent to LLM
    ENABLE_HITL_INTERRUPT: bool = False  # When True, adds interrupt_before=["generate"] for human-in-the-loop
    GRAPH_RECURSION_LIMIT: int = 10  # LangGraph recursion limit (validate->retry loop bound)
    CB_FAILURE_THRESHOLD: int = 5  # circuit breaker: consecutive failures to open
    CB_COOLDOWN_SECONDS: int = 60  # circuit breaker: seconds before half-open probe

    # --- Observability ---
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    VERSION: str = "0.1.0"

    model_config = {"env_prefix": "", "case_sensitive": True}

    @model_validator(mode="after")
    def validate_rag_config(self) -> "Settings":
        """Validate RAG chunk parameters are consistent."""
        if self.RAG_CHUNK_OVERLAP >= self.RAG_CHUNK_SIZE:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({self.RAG_CHUNK_OVERLAP}) must be less than "
                f"RAG_CHUNK_SIZE ({self.RAG_CHUNK_SIZE})"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
