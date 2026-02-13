"""Pure ASGI middleware for the Hey Seven Property Q&A API.

Uses raw ASGI middleware (NOT BaseHTTPMiddleware) to preserve SSE streaming.
Provides request logging, error handling, security headers, and rate limiting.
"""

import asyncio
import collections
import json
import logging
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.config import get_settings

logger = logging.getLogger(__name__)


def _get_access_logger() -> logging.Logger:
    """Return a logger configured for structured JSON output (Cloud Logging compatible)."""
    log = logging.getLogger("hey_seven.access")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


_access_logger = _get_access_logger()


class RequestLoggingMiddleware:
    """Pure ASGI middleware for structured request logging.

    Injects X-Request-ID, emits a JSON log line per request,
    and adds X-Response-Time-Ms header.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        request_id = (
            headers.get(b"x-request-id", b"").decode() or str(uuid.uuid4())[:8]
        )
        start_time = time.monotonic()
        method = scope.get("method", "WS")
        path = scope.get("path", "/")

        status_code: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                duration_ms = (time.monotonic() - start_time) * 1000
                extra_headers = [
                    (b"x-request-id", request_id.encode()),
                    (b"x-response-time-ms", f"{duration_ms:.1f}".encode()),
                ]
                message["headers"] = list(message.get("headers", [])) + extra_headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.monotonic() - start_time) * 1000
            _access_logger.info(
                json.dumps(
                    {
                        "severity": "INFO",
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "status": status_code,
                        "duration_ms": round(duration_ms, 1),
                    }
                )
            )


class ErrorHandlingMiddleware:
    """Catch unhandled exceptions and return a structured 500 JSON body.

    Prevents stack traces from leaking to clients.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except asyncio.CancelledError:
            # Client disconnected (normal for SSE) — log at INFO, not ERROR
            logger.info(
                "Client disconnected: %s %s",
                scope.get("method", "?"),
                scope.get("path", "/"),
            )
        except Exception:
            logger.exception(
                "Unhandled exception on %s %s",
                scope.get("method", "?"),
                scope.get("path", "/"),
            )
            if not response_started:
                body = json.dumps(
                    {
                        "error": "internal_server_error",
                        "message": "An unexpected error occurred.",
                    }
                ).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware:
    """Add security headers to every HTTP response."""

    HEADERS = [
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (
            b"content-security-policy",
            b"default-src 'self'; script-src 'self' 'unsafe-inline'; "
            b"style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            b"connect-src 'self'",
        ),
    ]

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + list(
                    self.HEADERS
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)


class RateLimitMiddleware:
    """Token-bucket rate limiter per client IP.

    Only applies to ``/chat`` endpoint. ``/health`` and static files are exempt.
    Returns 429 with ``Retry-After`` header when the limit is exceeded.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        settings = get_settings()
        self.max_tokens = settings.RATE_LIMIT_CHAT
        self.window_seconds = 60.0
        # {ip: deque of request timestamps}
        self._requests: dict[str, collections.deque] = {}

    def _is_allowed(self, client_ip: str) -> bool:
        """Check if a request from client_ip is within the rate limit."""
        now = time.monotonic()
        window_start = now - self.window_seconds

        if client_ip not in self._requests:
            self._requests[client_ip] = collections.deque()

        bucket = self._requests[client_ip]

        # Evict expired entries
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        # Clean up empty buckets to prevent unbounded dict growth
        if not bucket:
            del self._requests[client_ip]

        if client_ip not in self._requests:
            self._requests[client_ip] = collections.deque()
            bucket = self._requests[client_ip]

        if len(bucket) >= self.max_tokens:
            return False

        bucket.append(now)
        return True

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")

        # Only rate-limit /chat
        if path != "/chat":
            await self.app(scope, receive, send)
            return

        # Extract client IP
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        if self._is_allowed(client_ip):
            await self.app(scope, receive, send)
            return

        # Rate limited
        body = json.dumps(
            {"error": "rate_limit_exceeded", "message": "Too many requests."}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", b"60"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
