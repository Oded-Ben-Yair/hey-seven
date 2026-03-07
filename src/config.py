"""Centralized configuration via pydantic-settings.

All magic numbers, model names, paths, and tuning knobs live here.
Override any value via environment variable (e.g., ``MODEL_NAME=gemini-3.1-pro-preview``).
"""

import threading

from cachetools import TTLCache
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Google API ---
    GOOGLE_API_KEY: SecretStr = SecretStr(
        ""
    )  # Required for production; langchain auto-detects from env

    # --- Property ---
    PROPERTY_NAME: str = "Mohegan Sun"
    PROPERTY_DATA_PATH: str = "data/mohegan_sun.json"
    PROPERTY_WEBSITE: str = "mohegansun.com"
    PROPERTY_PHONE: str = "1-888-226-7711"
    PROPERTY_STATE: str = "Connecticut"  # Jurisdiction for age/gaming regulations

    # --- LLM ---
    MODEL_NAME: str = "gemini-3-flash-preview"
    COMPLEX_MODEL_NAME: str = "gemini-3.1-pro-preview"  # R83: Used for complex/emotional queries via model routing
    MODEL_TEMPERATURE: float = (
        1.0  # Gemini 3.x REQUIRES temperature=1.0; lower causes looping/degradation
    )
    MODEL_ROUTING_ENABLED: bool = (
        True  # R83: Route complex queries to COMPLEX_MODEL_NAME
    )
    MODEL_TIMEOUT: int = 30  # LLM call timeout in seconds
    MODEL_MAX_RETRIES: int = 2  # LLM call retry count
    MODEL_MAX_OUTPUT_TOKENS: int = (
        4096  # R92: Increased from 2048 to eliminate ~30% truncation
    )
    WHISPER_LLM_TEMPERATURE: float = 0.2  # Lower temperature for deterministic planning

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "gemini-embedding-001"  # Pinned. See ADR-006. Upgrade path: gemini-embedding-002 (when GA)

    # --- RAG ---
    CHROMA_PERSIST_DIR: str = "data/chroma"
    RAG_TOP_K: int = 5
    RAG_CHUNK_SIZE: int = 800
    RAG_CHUNK_OVERLAP: int = 120  # 15% of chunk size (industry recommendation)
    RAG_MIN_RELEVANCE_SCORE: float = (
        0.3  # Minimum relevance score (0-1, higher = more relevant)
    )
    RRF_K: int = 60  # Reciprocal Rank Fusion constant (see ADR-011, original RRF paper)

    # --- API ---
    API_KEY: SecretStr = SecretStr("")  # When set, /chat requires X-API-Key header
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8080", "http://localhost:3000"]
    RATE_LIMIT_CHAT: int = 20  # requests per minute
    RATE_LIMIT_MAX_CLIENTS: int = 10000  # max tracked client IPs (memory guard)
    TRUSTED_PROXIES: list[str] | None = (
        None  # CIDRs/IPs that may set X-Forwarded-For; None = trust direct peer only
    )
    SSE_TIMEOUT_SECONDS: int = 60
    MAX_REQUEST_BODY_SIZE: int = 65536  # 64 KB max request body

    # --- Agent ---
    MAX_MESSAGE_LIMIT: int = (
        40  # max total messages (human + AI) before forcing conversation end
    )
    MAX_HISTORY_MESSAGES: int = 20  # sliding window: only last N messages sent to LLM
    ENABLE_HITL_INTERRUPT: bool = (
        False  # When True, adds interrupt_before=["generate"] for human-in-the-loop
    )
    GRAPH_RECURSION_LIMIT: int = Field(
        default=10, ge=2, le=50
    )  # LangGraph recursion limit (validate->retry loop bound)
    CB_FAILURE_THRESHOLD: int = 5  # circuit breaker: consecutive failures to open
    CB_COOLDOWN_SECONDS: int = 60  # circuit breaker: seconds before half-open probe
    CB_ROLLING_WINDOW_SECONDS: float = (
        300.0  # circuit breaker: failure counting window (R10 fix — DeepSeek F8)
    )
    CB_SYNC_INTERVAL: float = (
        2.0  # R59 fix D8: Redis CB sync interval in seconds (R52 reduced from 5s)
    )
    RETRIEVAL_TIMEOUT: int = 10  # RAG retrieval timeout in seconds (R37 fix M-006)
    COMP_COMPLETENESS_THRESHOLD: float = (
        0.60  # minimum profile completeness for comp agent
    )
    SEMANTIC_INJECTION_ENABLED: bool = (
        True  # enable/disable semantic injection classifier (LLM second layer)
    )
    SEMANTIC_INJECTION_THRESHOLD: float = (
        0.8  # confidence threshold for semantic injection classifier
    )
    SEMANTIC_INJECTION_MODEL: str = (
        ""  # override model for semantic classifier (empty = use default)
    )
    PROFILING_LLM_TEMPERATURE: float = (
        0.1  # Low temperature for deterministic profiling extraction
    )
    PROFILING_MIN_CONFIDENCE: float = (
        0.7  # Minimum confidence threshold for profiling field acceptance
    )

    # --- State Backend ---
    STATE_BACKEND: str = "memory"  # "memory" | "redis"
    REDIS_URL: str = ""  # Redis connection URL for distributed state

    # --- Deployment ---
    KMS_KEY_PATH: str = ""  # GCP KMS key path for cosign image signing
    CANARY_ERROR_THRESHOLD: float = 5.0  # Max 5xx error rate (%) before canary rollback
    CANARY_STAGE_WAIT_SECONDS: int = (
        60  # Seconds to observe between canary traffic stages
    )
    LLM_SEMAPHORE_TIMEOUT: int = (
        30  # Seconds to wait for LLM semaphore before 503 backpressure
    )

    # --- Vector DB ---
    VECTOR_DB: str = "chroma"  # "chroma" (local dev) or "firestore" (GCP prod)
    FIRESTORE_PROJECT: str = ""  # GCP project ID for Firestore
    FIRESTORE_COLLECTION: str = "knowledge_base"  # Firestore collection name

    # --- Multi-tenant ---
    CASINO_ID: str = "mohegan_sun"  # Multi-tenant casino identifier

    # --- CMS ---
    CMS_WEBHOOK_SECRET: SecretStr = SecretStr(
        ""
    )  # HMAC-SHA256 secret for Google Sheets webhook verification
    GOOGLE_SHEETS_ID: str = ""  # Default Google Sheets spreadsheet ID for CMS content

    # --- SMS ---
    SMS_ENABLED: bool = False
    CONSENT_HMAC_SECRET: SecretStr = SecretStr(
        "change-me-in-production"
    )  # HMAC key for consent hash chain
    PERSONA_MAX_CHARS: int = 0  # 0 = unlimited, 160 = SMS segment limit
    TELNYX_API_KEY: SecretStr = SecretStr("")
    TELNYX_MESSAGING_PROFILE_ID: str = ""
    TELNYX_PUBLIC_KEY: str = ""  # For webhook signature verification
    QUIET_HOURS_START: int = 21  # 9 PM local time
    QUIET_HOURS_END: int = 8  # 8 AM local time
    SMS_FROM_NUMBER: str = ""  # E.164 format (e.g. +18605551234)

    # --- Observability ---
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    VERSION: str = "1.5.0"  # R83: Gemini 3.1 migration + behavioral improvements. Production deploy overrides with COMMIT_SHA.
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: SecretStr = SecretStr("")
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    model_config = {"env_prefix": "", "case_sensitive": True}

    @model_validator(mode="after")
    def normalize_embedding_model(self) -> "Settings":
        """Strip ``models/`` prefix from EMBEDDING_MODEL if present.

        Google's SDK sometimes expects the bare model name, sometimes with
        the ``models/`` prefix.  Normalizing to bare name prevents
        ingestion-vs-retrieval vector space mismatch when ``.env`` and
        ``config.py`` defaults disagree on the prefix.
        """
        if self.EMBEDDING_MODEL.startswith("models/"):
            object.__setattr__(
                self, "EMBEDDING_MODEL", self.EMBEDDING_MODEL.removeprefix("models/")
            )
        return self

    @model_validator(mode="after")
    def validate_rag_config(self) -> "Settings":
        """Validate RAG chunk parameters are consistent."""
        if self.RAG_CHUNK_OVERLAP >= self.RAG_CHUNK_SIZE:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({self.RAG_CHUNK_OVERLAP}) must be less than "
                f"RAG_CHUNK_SIZE ({self.RAG_CHUNK_SIZE})"
            )
        return self

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Hard-fail in production if critical secrets are empty.

        In production (ENVIRONMENT != 'development'), API_KEY and
        CMS_WEBHOOK_SECRET must be explicitly set. Empty defaults
        silently disable authentication and webhook verification —
        a catastrophic security bypass in a regulated casino environment.
        Development mode allows empty values for local testing.
        """
        if self.ENVIRONMENT != "development":
            if not self.API_KEY.get_secret_value():
                raise ValueError(
                    "API_KEY must be set when ENVIRONMENT != 'development'. "
                    "Empty API_KEY disables authentication on all protected endpoints. "
                    "Set API_KEY via environment variable or Secret Manager."
                )
            if not self.CMS_WEBHOOK_SECRET.get_secret_value():
                raise ValueError(
                    "CMS_WEBHOOK_SECRET must be set when ENVIRONMENT != 'development'. "
                    "Empty secret disables CMS webhook signature verification, "
                    "allowing attackers to inject fake casino content. "
                    "Set CMS_WEBHOOK_SECRET via environment variable or Secret Manager."
                )
            if self.SMS_ENABLED and not self.TELNYX_PUBLIC_KEY:
                raise ValueError(
                    "TELNYX_PUBLIC_KEY must be set when SMS_ENABLED=True in production. "
                    "Missing key disables SMS webhook signature verification, "
                    "allowing forged inbound SMS events. "
                    "Set TELNYX_PUBLIC_KEY via environment variable or Secret Manager."
                )
        return self

    @model_validator(mode="after")
    def validate_property_state(self) -> "Settings":
        """Warn if PROPERTY_STATE doesn't match expected state for CASINO_ID.

        R52 fix D10: Misconfigured PROPERTY_STATE causes regulatory helplines
        from the wrong jurisdiction to be served. This validator catches the
        mismatch early with a warning (not a hard error, since unknown
        CASINO_IDs are allowed for new property onboarding).
        """
        import logging as _logging

        _state_map = {
            "mohegan_sun": "Connecticut",
            "foxwoods": "Connecticut",
            "parx_casino": "Pennsylvania",
            "wynn_las_vegas": "Nevada",
            "hard_rock_ac": "New Jersey",
        }
        expected = _state_map.get(self.CASINO_ID)
        if expected and self.PROPERTY_STATE != expected:
            _logging.getLogger(__name__).warning(
                "PROPERTY_STATE=%r but CASINO_ID=%r expects %r — "
                "regulatory helplines may be incorrect",
                self.PROPERTY_STATE,
                self.CASINO_ID,
                expected,
            )
        return self

    @model_validator(mode="after")
    def validate_consent_hmac(self) -> "Settings":
        """Reject default CONSENT_HMAC_SECRET when SMS is enabled.

        SMS consent hashes use HMAC-SHA256 with this secret. The default
        placeholder is trivially forgeable — an attacker could fabricate
        valid consent records, causing TCPA violations and regulatory fines.
        Hard-fail prevents accidental deployment with insecure defaults.
        """
        if (
            self.SMS_ENABLED
            and self.CONSENT_HMAC_SECRET.get_secret_value() == "change-me-in-production"
        ):
            raise ValueError(
                "CONSENT_HMAC_SECRET must be set to a secure value when SMS_ENABLED=True. "
                "The default 'change-me-in-production' is insecure and would allow "
                "SMS consent hash forgery. Set CONSENT_HMAC_SECRET via environment variable."
            )
        return self


# R40 fix D8-C001: Add TTL jitter to prevent thundering herd on synchronized
# cache expiry. All singletons previously used identical ttl=3600, causing
# all 6+ caches to expire within a 2-second window on container startup + 1 hour.
# Jitter spreads reconstruction over a ~5-minute window.
import random as _random

_settings_cache: TTLCache = TTLCache(maxsize=1, ttl=3600 + _random.randint(0, 300))
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """Return a cached Settings instance with 1-hour TTL.

    Uses TTLCache (not @lru_cache) to allow runtime config changes
    without container restart. Environment variables are re-read when
    the cache expires (every 3600s).

    Thread-safe via threading.Lock (Settings is synchronous — not async).
    """
    cached = _settings_cache.get("settings")
    if cached is not None:
        return cached
    with _settings_lock:
        # Double-check after acquiring lock
        cached = _settings_cache.get("settings")
        if cached is not None:
            return cached
        settings = Settings()
        _settings_cache["settings"] = settings
        return settings


def clear_settings_cache() -> None:
    """Clear the settings cache, forcing re-read from environment on next access.

    Call during incident response to pick up tuned thresholds (CB_FAILURE_THRESHOLD,
    RATE_LIMIT_CHAT, SEMANTIC_INJECTION_THRESHOLD, etc.) without container restart.

    R17 fix: GPT F-001 — every other singleton has a cache-clear function;
    Settings was the exception, blocking incident response config changes.
    """
    _settings_cache.clear()


# Backward compatibility: many test files call get_settings.cache_clear() directly.
# This shim avoids breaking every test module that uses the @lru_cache API.
get_settings.cache_clear = _settings_cache.clear  # type: ignore[attr-defined]
