"""Guest profile CRUD operations with Firestore and in-memory fallback.

Provides async functions for reading, updating, and deleting guest profiles.
Firestore paths follow the multi-tenant namespace:
``casinos/{casino_id}/guests/{phone}``.

When Firestore is unavailable (local dev, tests), a module-level in-memory
dict serves as the backing store, ensuring the same async API works
everywhere without mocking infrastructure.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from src.data.models import (
    CONFIDENCE_MIN_THRESHOLD,
    apply_confidence_decay,
    calculate_completeness,
    filter_low_confidence,
    update_confidence,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory fallback store (keyed by "casino_id:phone")
# ---------------------------------------------------------------------------

_memory_store: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Firestore client accessor
# ---------------------------------------------------------------------------


# Cached Firestore client — avoids creating a new AsyncClient per CRUD call.
# Under load, per-request instantiation exhausts file descriptors and SSL
# handshakes, crashing the app.  The singleton is cleared by
# ``clear_firestore_client_cache()`` (exposed for tests).
_firestore_client_cache: dict[str, Any] = {}


def _get_firestore_client() -> Any | None:
    """Return the cached Firestore AsyncClient if available, else None.

    Lazy-imports ``google.cloud.firestore`` to avoid import failures when
    the dependency is not installed (local dev without GCP SDK).

    The client is cached as a module-level singleton to prevent connection
    exhaustion under load.  Creating a new ``AsyncClient`` per CRUD call
    opens new HTTP/2 connections and SSL handshakes — unsustainable at scale.

    Note: A near-identical helper exists in ``src.casino.config``.
    Both are intentionally kept separate to avoid coupling the guest
    data layer to the casino config module.  See comment there for
    the full trade-off rationale.
    """
    cached = _firestore_client_cache.get("client")
    if cached is not None:
        return cached

    try:
        from google.cloud.firestore import AsyncClient  # noqa: F401

        from src.config import get_settings

        settings = get_settings()
        if settings.VECTOR_DB == "firestore" and settings.FIRESTORE_PROJECT:
            client = AsyncClient(
                project=settings.FIRESTORE_PROJECT,
                database=settings.CASINO_ID,
            )
            _firestore_client_cache["client"] = client
            return client
    except ImportError:
        logger.debug("google-cloud-firestore not installed; using in-memory store")
    except Exception:
        logger.warning("Firestore client init failed; falling back to in-memory store", exc_info=True)
    return None


def clear_firestore_client_cache() -> None:
    """Clear the cached Firestore client (for testing and credential rotation)."""
    _firestore_client_cache.clear()


def _store_key(phone: str, casino_id: str) -> str:
    """Build a composite key for the in-memory store."""
    return f"{casino_id}:{phone}"


def _collection_path(casino_id: str) -> str:
    """Return the Firestore collection path for guests."""
    return f"casinos/{casino_id}/guests"


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


async def get_guest_profile(phone: str, casino_id: str) -> dict:
    """Load a guest profile from Firestore or the in-memory fallback.

    Args:
        phone: Guest phone in E.164 format (e.g., ``"+12035551234"``).
        casino_id: Casino identifier for multi-tenant isolation.

    Returns:
        The guest profile dict, or an empty profile skeleton if not found.
    """
    db = _get_firestore_client()
    if db is not None:
        try:
            doc_ref = db.collection(_collection_path(casino_id)).document(phone)
            doc = await doc_ref.get()
            if doc.exists:
                profile = doc.to_dict()
                logger.info("Loaded guest profile for %s...%s", phone[:4], phone[-4:])
                return profile
            logger.debug("No profile found for %s...%s; returning empty", phone[:4], phone[-4:])
            return _empty_profile(phone)
        except Exception:
            logger.warning("Firestore read failed; falling back to in-memory", exc_info=True)

    # In-memory fallback
    key = _store_key(phone, casino_id)
    profile = _memory_store.get(key)
    if profile is not None:
        return profile
    return _empty_profile(phone)


async def update_guest_profile(phone: str, casino_id: str, updates: dict) -> dict:
    """Merge updates into a guest profile with confidence tracking.

    For each field in ``updates``:
    - If the field is a ProfileField (has ``value`` key), apply confidence
      update rules (confirm/contradict) against the existing field.
    - Otherwise, set the field directly (for metadata, engagement counters, etc.).

    After merging, recalculates ``engagement.profile_completeness``.

    Args:
        phone: Guest phone in E.164 format.
        casino_id: Casino identifier for multi-tenant isolation.
        updates: Dict of fields to merge. Supports nested dot-paths and
            section-level dicts.

    Returns:
        The updated profile dict.
    """
    profile = await get_guest_profile(phone, casino_id)
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()

    # Merge updates into profile
    for section_key, section_updates in updates.items():
        if not isinstance(section_updates, dict):
            # Direct field set (e.g., _version, _updated_at)
            profile[section_key] = section_updates
            continue

        if section_key not in profile:
            profile[section_key] = {}

        existing_section = profile[section_key]
        if not isinstance(existing_section, dict):
            profile[section_key] = section_updates
            continue

        for field_key, field_value in section_updates.items():
            if isinstance(field_value, dict) and "value" in field_value:
                # ProfileField update with confidence tracking
                existing_field = existing_section.get(field_key)
                if existing_field and isinstance(existing_field, dict) and "value" in existing_field:
                    # Update existing field with confirm/contradict logic
                    updated = update_confidence(
                        existing_field,
                        field_value.get("source", "contextual_extraction"),
                        field_value["value"],
                        now=now,
                    )
                    # Preserve consent_scope from update if provided
                    if "consent_scope" in field_value and field_value["consent_scope"] is not None:
                        updated["consent_scope"] = field_value["consent_scope"]
                    existing_section[field_key] = updated
                else:
                    # New field, set directly
                    existing_section[field_key] = field_value
            else:
                # Non-ProfileField, set directly
                existing_section[field_key] = field_value

    # Update metadata
    profile["_updated_at"] = timestamp
    if "_version" in profile:
        profile["_version"] = profile["_version"] + 1
    else:
        profile["_version"] = 1

    # Recalculate completeness
    completeness = calculate_completeness(profile)
    if "engagement" not in profile:
        profile["engagement"] = {}
    profile["engagement"]["profile_completeness"] = round(completeness, 4)

    # Persist
    db = _get_firestore_client()
    if db is not None:
        try:
            doc_ref = db.collection(_collection_path(casino_id)).document(phone)
            await doc_ref.set(profile, merge=True)
            logger.info("Updated guest profile for %s...%s", phone[:4], phone[-4:])
        except Exception:
            logger.warning("Firestore write failed; stored in-memory only", exc_info=True)
            _memory_store[_store_key(phone, casino_id)] = profile
    else:
        _memory_store[_store_key(phone, casino_id)] = profile

    return profile


async def delete_guest_profile(phone: str, casino_id: str) -> bool:
    """CCPA cascade delete: remove profile and all conversation threads.

    Implements the deletion pipeline from Section 2.4a of the architecture doc:
    1. Delete all messages in conversation subcollections.
    2. Delete all conversation documents.
    3. Delete behavioral signals.
    4. Delete the guest document itself.

    Audit log entries are NOT deleted (regulatory requirement) -- they are
    de-identified by hashing the phone number.

    Args:
        phone: Guest phone in E.164 format.
        casino_id: Casino identifier for multi-tenant isolation.

    Returns:
        True if deletion succeeded, False if the profile did not exist.
    """
    db = _get_firestore_client()
    if db is not None:
        try:
            guest_ref = db.collection(_collection_path(casino_id)).document(phone)
            guest_doc = await guest_ref.get()
            if not guest_doc.exists:
                logger.info("No profile to delete for %s...%s", phone[:4], phone[-4:])
                return False

            # Collect all document references for batch deletion.
            # Firestore batches are atomic: either all operations complete or
            # none do, preventing the partially-deleted zombie state that
            # violates CCPA when individual deletes fail mid-cascade.
            # Firestore batch limit is 500 operations; cascade unlikely to
            # exceed this for a single guest profile.
            batch = db.batch()
            ops_count = 0

            # Cascade: conversations -> messages
            async for conv in guest_ref.collection("conversations").stream():
                async for msg in conv.reference.collection("messages").stream():
                    batch.delete(msg.reference)
                    ops_count += 1
                batch.delete(conv.reference)
                ops_count += 1

            # Cascade: behavioral signals
            async for signal in guest_ref.collection("behavioral_signals").stream():
                batch.delete(signal.reference)
                ops_count += 1

            # De-identify audit log references (update, not delete — regulatory requirement)
            hashed = hashlib.sha256(phone.encode()).hexdigest()[:16]
            audit_entries = db.collection("audit_log").where("entity_phone", "==", phone).stream()
            async for entry in audit_entries:
                batch.update(entry.reference, {
                    "entity_phone": f"deleted:{hashed}",
                    "entity_name": "[deleted]",
                })
                ops_count += 1

            # Delete guest document
            batch.delete(guest_ref)
            ops_count += 1

            # Commit atomically — all-or-nothing
            await batch.commit()
            logger.info(
                "CCPA cascade delete completed for %s...%s (%d batch ops)",
                phone[:4], phone[-4:], ops_count,
            )
            return True

        except Exception:
            logger.error("CCPA delete failed for %s...%s", phone[:4], phone[-4:], exc_info=True)
            raise

    # In-memory fallback
    key = _store_key(phone, casino_id)
    if key in _memory_store:
        del _memory_store[key]
        # Also clean up any conversation keys that might exist
        conv_prefix = f"{casino_id}:{phone}:conversations:"
        to_delete = [k for k in _memory_store if k.startswith(conv_prefix)]
        for k in to_delete:
            del _memory_store[k]
        logger.info("In-memory delete completed for %s...%s", phone[:4], phone[-4:])
        return True

    return False


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------


def get_agent_context(profile: dict, *, now: datetime | None = None) -> dict:
    """Prepare a profile for agent context by applying decay and filtering.

    Applies confidence decay for stale fields, then removes fields below
    the confidence threshold so the agent only sees reliable data.

    Args:
        profile: A guest profile dict.
        now: Current datetime for decay calculation.

    Returns:
        A filtered copy of the profile suitable for LLM context injection.
    """
    import copy

    working = copy.deepcopy(profile)
    working = apply_confidence_decay(working, now=now)
    return filter_low_confidence(working, threshold=CONFIDENCE_MIN_THRESHOLD)


def _empty_profile(phone: str) -> dict:
    """Return a minimal empty profile skeleton.

    Args:
        phone: Guest phone in E.164 format.

    Returns:
        A dict with required structure but no populated ProfileFields.
    """
    now = datetime.now(timezone.utc).isoformat()
    return {
        "_id": phone,
        "_version": 0,
        "_created_at": now,
        "_updated_at": now,
        "core_identity": {
            "phone": phone,
        },
        "visit_context": {},
        "preferences": {
            "dining": {},
            "entertainment": {},
            "gaming": {},
            "spa": {},
        },
        "companions": [],
        "consent": {},
        "engagement": {
            "total_conversations": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "profile_completeness": 0.0,
            "offers_sent": 0,
            "offers_redeemed": 0,
            "escalations": 0,
        },
    }


def clear_memory_store() -> None:
    """Clear the in-memory fallback store (for testing)."""
    _memory_store.clear()
