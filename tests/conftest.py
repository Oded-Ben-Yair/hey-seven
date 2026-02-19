"""Shared test fixtures for Hey Seven tests."""

import json

import pytest


@pytest.fixture(autouse=True)
def _clear_singleton_caches():
    """Reset all @lru_cache singletons between tests.

    Prevents test pollution from cached Settings, LLM instances, and
    CircuitBreaker singletons leaking state across test modules.
    """
    yield
    # Clear all singleton caches after each test
    from src.config import get_settings

    get_settings.cache_clear()

    try:
        from src.agent.nodes import _llm_cache

        _llm_cache.clear()
    except ImportError:
        pass

    try:
        from src.agent.nodes import _build_greeting_categories

        _build_greeting_categories.cache_clear()
    except ImportError:
        pass

    try:
        from src.agent.whisper_planner import _whisper_cache

        _whisper_cache.clear()
    except ImportError:
        pass

    try:
        from src.agent.circuit_breaker import _get_circuit_breaker

        _get_circuit_breaker.cache_clear()
    except ImportError:
        pass

    try:
        from src.rag.embeddings import get_embeddings

        get_embeddings.cache_clear()
    except ImportError:
        pass

    try:
        from src.rag.pipeline import _get_retriever_cached

        _get_retriever_cached.cache_clear()
    except (ImportError, AttributeError):
        pass

    try:
        from src.agent.memory import get_checkpointer

        get_checkpointer.cache_clear()
    except ImportError:
        pass

    try:
        from src.data.guest_profile import clear_memory_store

        clear_memory_store()
    except ImportError:
        pass

    try:
        from src.casino.config import clear_config_cache

        clear_config_cache()
    except ImportError:
        pass

    try:
        from src.cms.webhook import _content_hashes

        _content_hashes.clear()
    except ImportError:
        pass

    try:
        from src.observability.langfuse_client import _get_langfuse_client

        _get_langfuse_client.cache_clear()
    except ImportError:
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
    }


@pytest.fixture
def test_property_file(tmp_path, test_property_data):
    """Write test data to a temp JSON file and return its path."""
    p = tmp_path / "test_property.json"
    p.write_text(json.dumps(test_property_data))
    return str(p)


