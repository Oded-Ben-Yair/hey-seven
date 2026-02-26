"""Runtime validation for data boundary crossings.

Validates data shape at the boundary between RAG retrieval -> graph state
and between Firestore -> guest profile. Catches data corruption, schema
drift, and deserialization bugs that static type checking misses.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["validate_retrieved_chunk", "validate_guest_profile"]


def validate_retrieved_chunk(chunk: dict[str, Any]) -> bool:
    """Validate a RetrievedChunk dict has required fields with correct types.

    Args:
        chunk: A dict expected to match the RetrievedChunk schema.

    Returns:
        True if valid, False otherwise (logs warning on failure).
    """
    if not isinstance(chunk, dict):
        logger.warning("Retrieved chunk is not a dict: %s", type(chunk).__name__)
        return False
    if "content" not in chunk or not isinstance(chunk["content"], str):
        logger.warning("Retrieved chunk missing or invalid 'content' field")
        return False
    if "metadata" not in chunk or not isinstance(chunk["metadata"], dict):
        logger.warning("Retrieved chunk missing or invalid 'metadata' field")
        return False
    if "score" not in chunk:
        logger.warning("Retrieved chunk missing 'score' field")
        return False
    try:
        float(chunk["score"])
    except (TypeError, ValueError):
        logger.warning("Retrieved chunk 'score' is not numeric: %s", chunk["score"])
        return False
    return True


def validate_guest_profile(profile: dict[str, Any]) -> bool:
    """Validate a guest profile dict has required structure.

    Checks for required top-level keys and basic type constraints.
    Does NOT validate individual ProfileField contents (that's the
    confidence/decay system's job).

    Args:
        profile: A dict expected to match the GuestProfile schema.

    Returns:
        True if valid, False otherwise (logs warning on failure).
    """
    if not isinstance(profile, dict):
        logger.warning("Guest profile is not a dict: %s", type(profile).__name__)
        return False
    required_keys = {"_id", "_version", "core_identity"}
    missing = required_keys - set(profile.keys())
    if missing:
        logger.warning("Guest profile missing required keys: %s", missing)
        return False
    if not isinstance(profile.get("_version"), int):
        logger.warning(
            "Guest profile '_version' is not int: %s",
            type(profile.get("_version")).__name__,
        )
        return False
    if not isinstance(profile.get("core_identity"), dict):
        logger.warning("Guest profile 'core_identity' is not dict")
        return False
    return True
