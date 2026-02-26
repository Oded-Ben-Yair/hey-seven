"""RFC 7807 Problem Details error taxonomy for the Hey Seven API.

Provides a canonical set of error codes and RFC 7807-compliant error response
builder that clients can switch on, ensuring consistent error handling across
all endpoints and middleware layers.

Usage::

    from src.api.errors import ErrorCode, error_response

    return JSONResponse(
        status_code=429,
        content=error_response(ErrorCode.RATE_LIMITED, "Too many requests.", status=429),
        media_type="application/problem+json",
    )
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Canonical error codes for API responses.

    Client applications should switch on ``code`` (not HTTP status)
    to differentiate error handling paths.
    """

    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limit_exceeded"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    AGENT_UNAVAILABLE = "agent_unavailable"
    INTERNAL_ERROR = "internal_error"
    VALIDATION_ERROR = "validation_error"
    SERVICE_DEGRADED = "service_degraded"


def error_response(code: ErrorCode, message: str, status: int = 500) -> dict:
    """Build an RFC 7807 Problem Details response body.

    Conforms to RFC 7807 (Problem Details for HTTP APIs) for standardized
    error handling across all middleware and endpoint layers.

    Args:
        code: One of the ``ErrorCode`` enum values.
        message: Human-readable error description.
        status: HTTP status code.

    Returns:
        Dict conforming to RFC 7807 Problem Details.
    """
    return {
        "type": "about:blank",
        "title": code.value.replace("_", " ").title(),
        "status": status,
        "detail": message,
        "code": code.value,
    }
