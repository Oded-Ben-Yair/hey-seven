"""Shared test fixtures for Hey Seven tests."""

import asyncio
import json
import os

import pytest


@pytest.fixture(autouse=True)
def _disable_semantic_injection_in_tests(monkeypatch):
    """Disable semantic injection classifier in tests by default.

    The semantic classifier requires a GOOGLE_API_KEY and fails CLOSED
    (blocks all messages) when unavailable.  Tests that specifically
    test semantic injection behavior mock the classifier directly.
    """
    monkeypatch.setenv("SEMANTIC_INJECTION_ENABLED", "false")


@pytest.fixture(autouse=True)
def _clear_singleton_caches():
    """Reset all @lru_cache singletons between tests.

    Prevents test pollution from cached Settings, LLM instances, and
    CircuitBreaker singletons leaking state across test modules.
    """
    yield
    # Clear all singleton caches after each test.
    # Uses (ImportError, AttributeError) consistently to handle both missing
    # modules and renamed cache attributes without misleading test failures.
    from src.config import get_settings

    get_settings.cache_clear()

    try:
        from src.agent.nodes import _llm_cache, _validator_cache

        _llm_cache.clear()
        _validator_cache.clear()
    except (ImportError, AttributeError):
        pass

    try:
        from src.agent.nodes import _greeting_cache

        _greeting_cache.clear()
    except (ImportError, AttributeError):
        pass

    try:
        from src.agent.whisper_planner import _whisper_cache

        _whisper_cache.clear()
    except (ImportError, AttributeError):
        pass

    try:
        import src.agent.whisper_planner as _wp

        _wp._telemetry.count = 0
        _wp._telemetry.alerted = False
    except (ImportError, AttributeError):
        pass

    try:
        from src.agent.circuit_breaker import _cb_cache

        _cb_cache.clear()
    except (ImportError, AttributeError):
        pass

    try:
        from src.rag.embeddings import _embeddings_cache

        _embeddings_cache.clear()
    except (ImportError, AttributeError):
        pass

    try:
        from src.rag.pipeline import clear_retriever_cache

        clear_retriever_cache()
    except (ImportError, AttributeError):
        pass

    try:
        from src.agent.memory import clear_checkpointer_cache

        clear_checkpointer_cache()
    except (ImportError, AttributeError):
        pass

    try:
        from src.data.guest_profile import clear_firestore_client_cache, clear_memory_store

        clear_memory_store()
        clear_firestore_client_cache()
    except (ImportError, AttributeError):
        pass

    try:
        from src.casino.config import clear_config_cache

        clear_config_cache()
    except (ImportError, AttributeError):
        pass

    try:
        from src.casino.feature_flags import _flag_cache

        _flag_cache.clear()
    except (ImportError, AttributeError):
        pass

    try:
        from src.cms.webhook import _content_hashes

        _content_hashes.clear()
    except (ImportError, AttributeError):
        pass

    try:
        from src.sms import webhook as _sms_webhook

        _sms_webhook._idempotency_tracker = None
    except (ImportError, AttributeError):
        pass

    try:
        from src.api.middleware import _access_logger

        # Remove all handlers to prevent handler accumulation across tests
        _access_logger.handlers.clear()
    except (ImportError, AttributeError):
        pass

    try:
        from src.sms.webhook import _DELIVERY_LOG

        _DELIVERY_LOG.clear()
    except (ImportError, AttributeError):
        pass

    try:
        from src.state_backend import get_state_backend

        get_state_backend.cache_clear()
    except (ImportError, AttributeError):
        pass

    try:
        from src.observability.langfuse_client import _get_langfuse_client

        _get_langfuse_client.cache_clear()
    except (ImportError, AttributeError):
        pass

    try:
        import src.agent.sentiment as _sent

        _sent._vader_analyzer = None
    except (ImportError, AttributeError):
        pass

    try:
        import src.agent.agents._base as _base_mod

        # Recreate the semaphore to reset its internal counter.
        # Prevents permanent count decrement from crashed test acquisitions.
        _base_mod._LLM_SEMAPHORE = asyncio.Semaphore(20)
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def test_property_data():
    """Small test dataset for property Q&A tests."""
    return {
        "property": {"name": "Test Casino", "location": "Test City, NV"},
        "restaurants": [
            {
                "name": "Test Steakhouse",
                "cuisine": "Steakhouse",
                "price_range": "$$$",
                "hours": "5-10 PM",
                "location": "Main Floor",
                "description": "A fine steakhouse",
            },
            {
                "name": "Test Buffet",
                "cuisine": "Buffet",
                "price_range": "$$",
                "hours": "11 AM-9 PM",
                "location": "Level 2",
                "description": "All you can eat",
            },
        ],
        "entertainment": [
            {
                "name": "Test Arena",
                "type": "Concert Venue",
                "capacity": "10000",
                "description": "Main arena",
            }
        ],
        "hotel": {
            "towers": [
                {
                    "name": "Sky Tower",
                    "description": "Luxury tower with mountain views",
                    "floors": 34,
                }
            ],
            "room_types": [
                {
                    "name": "Deluxe King",
                    "size": "400 sq ft",
                    "rate": "$199/night",
                    "description": "Spacious room with king bed",
                }
            ],
        },
        "gaming": {
            "casino_size_sqft": 300000,
            "slot_machines": 5000,
            "table_games": 300,
        },
        "faq": [{"question": "What are the hours?", "answer": "Open 24/7"}],
        "amenities": [
            {
                "name": "Elemis Spa",
                "type": "Full-service spa",
                "description": "Luxury day spa offering massages, facials, and body treatments.",
                "hours": "9 AM - 9 PM daily",
                "location": "Level 2, Casino of the Earth",
            },
            {
                "name": "Swimming Pool",
                "type": "Indoor pool",
                "description": "Heated indoor pool with lounge seating.",
                "hours": "6 AM - 10 PM daily",
            },
        ],
        "promotions": [
            {
                "name": "Momentum Rewards",
                "description": "Mohegan Sun's loyalty program with four tiers of benefits.",
                "benefits": ["Free play", "Dining discounts", "Hotel upgrades", "Event access"],
                "how_to_join": "Sign up at any Momentum desk with valid photo ID.",
            },
            {
                "name": "Ascend Tier",
                "description": "Premium tier for frequent players with enhanced benefits.",
                "requirements": "Earn 25,000 tier credits within a calendar year.",
                "benefits": ["Priority check-in", "Dedicated host", "Complimentary valet"],
            },
        ],
    }


@pytest.fixture
def test_property_file(tmp_path, test_property_data):
    """Write test data to a temp JSON file and return its path."""
    p = tmp_path / "test_property.json"
    p.write_text(json.dumps(test_property_data))
    return str(p)


