"""Guest profile data models and CRUD operations.

Public API:

    from src.data import (
        GuestProfile,
        ProfileField,
        get_guest_profile,
        update_guest_profile,
        delete_guest_profile,
        calculate_completeness,
    )
"""

from src.data.guest_profile import (
    delete_guest_profile,
    get_guest_profile,
    update_guest_profile,
)
from src.data.models import GuestProfile, ProfileField, calculate_completeness

__all__ = [
    "GuestProfile",
    "ProfileField",
    "calculate_completeness",
    "delete_guest_profile",
    "get_guest_profile",
    "update_guest_profile",
]
