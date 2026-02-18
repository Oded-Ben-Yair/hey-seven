"""Structured error taxonomy for the Hey Seven API.

Provides a canonical set of error codes that clients can switch on, ensuring
consistent error handling across all endpoints and middleware layers.

Usage::

    from src.api.errors import ErrorCode, error_response

    return JSONResponse(
        status_code=429,
        content=error_response(ErrorCode.RATE_LIMITED, "Too many requests."),
    )
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Canonical error codes for API responses.

    Client applications should switch on ``error.code`` (not HTTP status)
    to differentiate error handling paths.
    """

    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limit_exceeded"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    AGENT_UNAVAILABLE = "agent_unavailable"
    INTERNAL_ERROR = "internal_error"
    VALIDATION_ERROR = "validation_error"
    SERVICE_DEGRADED = "service_degraded"


def error_response(code: ErrorCode, message: str) -> dict:
    """Build a structured error response body.

    Args:
        code: One of the ``ErrorCode`` enum values.
        message: Human-readable error description.

    Returns:
        Dict with ``error`` object containing ``code`` and ``message``.
    """
    return {"error": {"code": code.value, "message": message}}
