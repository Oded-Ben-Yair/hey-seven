"""Google Sheets CMS for casino content management.

Provides a Sheets API client, per-category content validation with quarantine,
and a webhook handler for real-time content updates from casino operators.
"""

from .sheets_client import CONTENT_CATEGORIES, SheetsClient
from .validation import validate_item
from .webhook import handle_cms_webhook

__all__ = [
    "SheetsClient",
    "validate_item",
    "handle_cms_webhook",
    "CONTENT_CATEGORIES",
]
