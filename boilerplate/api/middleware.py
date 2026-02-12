"""Middleware for the Casino Host API.

Configures CORS, request logging, error handling, and rate limiting
for the FastAPI application.
"""

import logging
import time
import uuid
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------


def configure_cors(app: FastAPI) -> None:
    """Add CORS middleware with appropriate settings.

    In production, restrict origins to the frontend domain. During
    development, allows all origins.

    Args:
        app: The FastAPI application instance.
    """
    import os

    allowed_origins = os.environ.get(
        "CORS_ORIGINS", "*"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Request Logging
# ---------------------------------------------------------------------------


async def request_logging_middleware(request: Request, call_next: Any) -> Response:
    """Log all incoming requests with timing information.

    Assigns a unique request ID, logs the method/path/status, and records
    the response time in milliseconds.

    Args:
        request: The incoming HTTP request.
        call_next: The next middleware or route handler.

    Returns:
        The HTTP response.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.monotonic()

    # Inject request_id into request state for downstream use
    request.state.request_id = request_id

    logger.info(
        "[%s] %s %s started",
        request_id,
        request.method,
        request.url.path,
    )

    response = await call_next(request)

    duration_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "[%s] %s %s -> %d (%.1fms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    # Add timing header
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"

    return response


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


async def error_handling_middleware(request: Request, call_next: Any) -> Response:
    """Catch unhandled exceptions and return structured error responses.

    Prevents stack traces from leaking to clients in production. Logs
    the full exception server-side.

    Args:
        request: The incoming HTTP request.
        call_next: The next middleware or route handler.

    Returns:
        The HTTP response (500 error on unhandled exceptions).
    """
    try:
        return await call_next(request)
    except Exception:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(
            "[%s] Unhandled exception on %s %s",
            request_id,
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again.",
                "request_id": request_id,
            },
        )


# ---------------------------------------------------------------------------
# Rate Limiting (Simple In-Memory)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple in-memory rate limiter for development and single-instance deploys.

    For production, use Redis-backed rate limiting (e.g., via Cloud Memorystore)
    or GCP API Gateway rate limiting.

    Limits requests per client IP with a sliding window.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_limit: int = 10,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum sustained request rate per IP.
            burst_limit: Maximum requests in a 1-second burst.
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self._request_log: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if a request from the given IP is allowed.

        Args:
            client_ip: The client's IP address.

        Returns:
            True if the request is allowed, False if rate-limited.
        """
        now = time.monotonic()
        window_start = now - 60.0
        burst_start = now - 1.0

        # Clean old entries
        self._request_log[client_ip] = [
            t for t in self._request_log[client_ip] if t > window_start
        ]

        # Check minute window
        if len(self._request_log[client_ip]) >= self.requests_per_minute:
            return False

        # Check burst window
        burst_count = sum(
            1 for t in self._request_log[client_ip] if t > burst_start
        )
        if burst_count >= self.burst_limit:
            return False

        self._request_log[client_ip].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


async def rate_limiting_middleware(request: Request, call_next: Any) -> Response:
    """Apply rate limiting based on client IP.

    Skips rate limiting for health check endpoints.

    Args:
        request: The incoming HTTP request.
        call_next: The next middleware or route handler.

    Returns:
        The HTTP response (429 if rate-limited).
    """
    # Skip rate limiting for health checks
    if request.url.path in ("/health", "/api/v1/health"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
            },
        )

    return await call_next(request)
