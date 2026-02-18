"""Guest profile data models for progressive profiling.

Defines the ``ProfileField`` schema and ``GuestProfile`` TypedDict matching
the Firestore document structure from Section 2.1 of the v2 architecture doc.

Key design decisions:
- TypedDict (not dataclass) for JSON-serializable Firestore compatibility.
- Per-field confidence scoring with source tracking.
- Weighted completeness calculation (core > visit > preferences > companions).
- 90-day confidence decay (0.90 multiplier) for stale fields.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source types for profile fields
# ---------------------------------------------------------------------------

SourceType = Literal[
    "self_reported",
    "contextual_extraction",
    "inferred",
    "crm_import",
    "incentive_exchange",
]

# ---------------------------------------------------------------------------
# ProfileField: every collected field wraps its value with metadata
# ---------------------------------------------------------------------------


class ProfileField(TypedDict, total=False):
    """A single profiled field with confidence and provenance metadata.

    Matches the Firestore schema from Section 2.2 of the architecture doc.
    ``total=False`` allows partial construction (e.g., omitting consent_scope).
    """

    value: Any
    confidence: float  # 0.0 - 1.0
    source: SourceType
    collected_at: str  # ISO 8601
    consent_scope: str | None


# ---------------------------------------------------------------------------
# Section-level TypedDicts
# ---------------------------------------------------------------------------


class CoreIdentity(TypedDict, total=False):
    """Core guest identity fields."""

    phone: str  # E.164 format, also the document key
    guest_uuid: str  # UUID v4, stable cross-reference
    name: ProfileField | None
    email: ProfileField | None
    language: ProfileField | None
    full_name: ProfileField | None
    date_of_birth: ProfileField | None


class VisitContext(TypedDict, total=False):
    """Visit planning and history fields."""

    planned_visit_date: ProfileField | None
    party_size: ProfileField | None
    occasion: ProfileField | None
    visit_history: list[dict[str, Any]]


class DiningPreferences(TypedDict, total=False):
    """Dining preference fields."""

    dietary_restrictions: ProfileField | None
    cuisine_preferences: ProfileField | None
    budget_range: ProfileField | None
    kids_menu_needed: ProfileField | None


class EntertainmentPreferences(TypedDict, total=False):
    """Entertainment preference fields."""

    interests: ProfileField | None
    accessibility_needs: ProfileField | None


class GamingPreferences(TypedDict, total=False):
    """Gaming preference fields."""

    level: ProfileField | None
    preferred_games: ProfileField | None
    typical_spend: ProfileField | None


class SpaPreferences(TypedDict, total=False):
    """Spa preference fields."""

    treatments_interested: ProfileField | None


class Preferences(TypedDict, total=False):
    """Aggregated guest preferences across all categories."""

    dining: DiningPreferences
    entertainment: EntertainmentPreferences
    gaming: GamingPreferences
    spa: SpaPreferences


class Companion(TypedDict, total=False):
    """A companion traveling with the guest."""

    relationship: str
    name: ProfileField | None
    age: ProfileField | None
    preferences: dict[str, Any]


class Consent(TypedDict, total=False):
    """Guest consent tracking for TCPA/CCPA compliance."""

    sms_opt_in: bool
    sms_opt_in_method: str
    sms_opt_in_timestamp: str
    ai_disclosure_sent: bool
    ai_disclosure_timestamp: str
    marketing_consent: bool
    data_retention_consent: bool
    privacy_policy_link_sent: bool
    consent_version: str


class Engagement(TypedDict, total=False):
    """Aggregate engagement metrics."""

    total_conversations: int
    total_messages_sent: int
    total_messages_received: int
    last_message_at: str
    profile_completeness: float
    offers_sent: int
    offers_redeemed: int
    escalations: int
    sentiment_trend: str


class GuestProfile(TypedDict, total=False):
    """Full guest profile document matching Firestore schema (Section 2.1).

    TypedDict is used (not dataclass) so the profile round-trips cleanly
    through ``json.loads(json.dumps(profile))`` and Firestore serialization.
    """

    _id: str
    _version: int
    _created_at: str
    _updated_at: str
    core_identity: CoreIdentity
    visit_context: VisitContext
    preferences: Preferences
    companions: list[Companion]
    consent: Consent
    engagement: Engagement


# ---------------------------------------------------------------------------
# Completeness weights (higher = more important for profile quality)
# ---------------------------------------------------------------------------

# core_identity fields that are ProfileField-shaped (not phone/guest_uuid)
_CORE_FIELDS = ("name", "email", "language", "full_name", "date_of_birth")
_VISIT_FIELDS = ("planned_visit_date", "party_size", "occasion")
_PREF_FIELDS = (
    "dining.dietary_restrictions",
    "dining.cuisine_preferences",
    "dining.budget_range",
    "dining.kids_menu_needed",
    "entertainment.interests",
    "entertainment.accessibility_needs",
    "gaming.level",
    "gaming.preferred_games",
    "gaming.typical_spend",
    "spa.treatments_interested",
)

FIELD_WEIGHTS: dict[str, float] = {}
for _f in _CORE_FIELDS:
    FIELD_WEIGHTS[f"core_identity.{_f}"] = 2.0
for _f in _VISIT_FIELDS:
    FIELD_WEIGHTS[f"visit_context.{_f}"] = 1.5
for _f in _PREF_FIELDS:
    FIELD_WEIGHTS[f"preferences.{_f}"] = 1.0
FIELD_WEIGHTS["companions"] = 0.5

_TOTAL_WEIGHT = sum(FIELD_WEIGHTS.values())

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

CONFIDENCE_MIN_THRESHOLD = 0.40  # Fields below this excluded from agent context
CONFIDENCE_DECAY_DAYS = 90  # Days before confidence decays
CONFIDENCE_DECAY_FACTOR = 0.90  # Multiplier applied after decay threshold
CONFIDENCE_CONFIRM_BOOST = 0.15  # Boost when same field confirmed by 2nd source
CONFIDENCE_CONTRADICT_PENALTY = 0.30  # Penalty when field contradicted


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _is_profile_field(value: Any) -> bool:
    """Check if a value looks like a ProfileField (dict with 'value' key)."""
    return isinstance(value, dict) and "value" in value


def _get_nested(data: dict, dotted_path: str) -> Any:
    """Retrieve a nested value via dot-separated path.

    Args:
        data: The dict to traverse.
        dotted_path: A dot-separated key path (e.g., ``"dining.budget_range"``).

    Returns:
        The value at the path, or ``None`` if any segment is missing.
    """
    current = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def calculate_completeness(profile: dict) -> float:
    """Calculate weighted profile completeness (0.0 - 1.0).

    Weights are assigned per the architecture doc:
    - core_identity fields: 2.0 each
    - visit_context fields: 1.5 each
    - preferences fields: 1.0 each
    - companions: 0.5

    Args:
        profile: A guest profile dict (GuestProfile-shaped).

    Returns:
        A float between 0.0 (empty) and 1.0 (all fields populated).
    """
    if not profile:
        return 0.0

    filled_weight = 0.0

    for dotted_path, weight in FIELD_WEIGHTS.items():
        if dotted_path == "companions":
            companions = profile.get("companions")
            if companions and isinstance(companions, list) and len(companions) > 0:
                filled_weight += weight
            continue

        # Split into section + field path
        parts = dotted_path.split(".", 1)
        section = parts[0]
        field_path = parts[1] if len(parts) > 1 else ""

        section_data = profile.get(section)
        if not isinstance(section_data, dict):
            continue

        field_value = _get_nested(section_data, field_path) if field_path else section_data
        if field_value is None:
            continue

        # A ProfileField is populated if it has a non-None value
        if _is_profile_field(field_value) and field_value.get("value") is not None:
            filled_weight += weight

    if _TOTAL_WEIGHT == 0:
        return 0.0
    return min(1.0, filled_weight / _TOTAL_WEIGHT)


def apply_confidence_decay(profile: dict, now: datetime | None = None) -> dict:
    """Apply 90-day confidence decay to all ProfileFields in the profile.

    Fields older than ``CONFIDENCE_DECAY_DAYS`` have their confidence
    multiplied by ``CONFIDENCE_DECAY_FACTOR`` (0.90).

    Args:
        profile: A guest profile dict (modified in-place and returned).
        now: Current datetime (defaults to ``datetime.now(timezone.utc)``).

    Returns:
        The profile dict with decayed confidence values.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    def _decay_field(field: Any) -> None:
        """Apply decay to a single ProfileField if stale."""
        if not _is_profile_field(field):
            return
        collected_at_str = field.get("collected_at")
        if not collected_at_str:
            return
        try:
            collected_at = datetime.fromisoformat(collected_at_str.replace("Z", "+00:00"))
            if collected_at.tzinfo is None:
                collected_at = collected_at.replace(tzinfo=timezone.utc)
            age_days = (now - collected_at).days
            if age_days > CONFIDENCE_DECAY_DAYS:
                old_conf = field.get("confidence", 1.0)
                field["confidence"] = round(old_conf * CONFIDENCE_DECAY_FACTOR, 4)
        except (ValueError, TypeError):
            logger.warning("Invalid collected_at in profile field: %s", collected_at_str)

    def _walk_and_decay(data: Any) -> None:
        """Recursively walk a dict/list and decay all ProfileFields."""
        if isinstance(data, dict):
            if _is_profile_field(data):
                _decay_field(data)
            else:
                for v in data.values():
                    _walk_and_decay(v)
        elif isinstance(data, list):
            for item in data:
                _walk_and_decay(item)

    _walk_and_decay(profile)
    return profile


def update_confidence(
    existing_field: dict,
    new_source: SourceType,
    new_value: Any,
    *,
    now: datetime | None = None,
) -> dict:
    """Update a ProfileField's confidence based on confirmation or contradiction.

    Rules from Section 2.2:
    - Same value confirmed by second source: confidence += 0.15 (capped at 1.0)
    - Value contradicted: confidence -= 0.30 (floored at 0.0), flagged for review

    Args:
        existing_field: The current ProfileField dict.
        new_source: The source of the new observation.
        new_value: The new observed value.
        now: Current datetime for updating ``collected_at``.

    Returns:
        Updated ProfileField dict.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    old_confidence = existing_field.get("confidence", 0.5)
    old_value = existing_field.get("value")
    timestamp = now.isoformat()

    if old_value == new_value:
        # Confirmed by second source
        new_confidence = min(1.0, old_confidence + CONFIDENCE_CONFIRM_BOOST)
        return {
            "value": new_value,
            "confidence": round(new_confidence, 4),
            "source": new_source,
            "collected_at": timestamp,
            "consent_scope": existing_field.get("consent_scope"),
        }
    else:
        # Contradicted
        new_confidence = max(0.0, old_confidence - CONFIDENCE_CONTRADICT_PENALTY)
        return {
            "value": new_value,
            "confidence": round(new_confidence, 4),
            "source": new_source,
            "collected_at": timestamp,
            "consent_scope": existing_field.get("consent_scope"),
            "_contradicted": True,
            "_previous_value": old_value,
        }


def filter_low_confidence(profile: dict, threshold: float = CONFIDENCE_MIN_THRESHOLD) -> dict:
    """Return a copy of the profile excluding fields below confidence threshold.

    Fields with ``confidence < threshold`` are replaced with ``None`` so the
    agent does not see unreliable data.

    Args:
        profile: A guest profile dict.
        threshold: Minimum confidence to retain a field.

    Returns:
        A new dict with low-confidence fields nullified.
    """
    import copy

    filtered = copy.deepcopy(profile)

    def _filter_section(section: Any) -> Any:
        if isinstance(section, dict):
            if _is_profile_field(section):
                conf = section.get("confidence", 0.0)
                if conf < threshold:
                    return None
                return section
            return {k: _filter_section(v) for k, v in section.items()}
        if isinstance(section, list):
            return [_filter_section(item) for item in section]
        return section

    for key in ("core_identity", "visit_context", "preferences"):
        if key in filtered:
            filtered[key] = _filter_section(filtered[key])

    # Filter companion fields
    if "companions" in filtered and isinstance(filtered["companions"], list):
        filtered["companions"] = [_filter_section(c) for c in filtered["companions"]]

    return filtered
