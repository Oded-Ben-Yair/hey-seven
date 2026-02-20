"""Checkpointer factory for conversation state persistence.

Returns ``MemorySaver`` for local development (default) or
``FirestoreSaver`` for production when ``VECTOR_DB=firestore``.

Cached with ``@lru_cache`` consistent with ``_get_llm()``,
``get_settings()``, and ``get_embeddings()`` singleton patterns.
"""

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_checkpointer() -> Any:
    """Get or create the checkpointer singleton (cached).

    Returns ``FirestoreSaver`` when ``VECTOR_DB=firestore`` (production),
    otherwise ``MemorySaver`` (local development).

    Returns:
        A LangGraph checkpointer instance.
    """
    from src.config import get_settings

    settings = get_settings()

    if settings.VECTOR_DB == "firestore":
        try:
            from langgraph.checkpoint.firestore import FirestoreSaver

            checkpointer = FirestoreSaver(project=settings.FIRESTORE_PROJECT)
            logger.info("Using FirestoreSaver checkpointer (project=%s)", settings.FIRESTORE_PROJECT)
            return checkpointer
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-firestore not installed. "
                "Falling back to MemorySaver."
            )
        except Exception:
            logger.exception("Failed to create FirestoreSaver. Falling back to MemorySaver.")

    from langgraph.checkpoint.memory import MemorySaver

    if settings.ENVIRONMENT == "production":
        logger.warning(
            "ENVIRONMENT=production but using MemorySaver (in-memory). "
            "Conversation state will be lost on container recycle and is not "
            "shared across Cloud Run instances. Set VECTOR_DB=firestore for "
            "durable checkpointing."
        )
    logger.info("Using MemorySaver checkpointer (in-memory, dev mode).")
    return MemorySaver()
