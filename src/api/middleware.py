"""Pure ASGI middleware for the Hey Seven Property Q&A API.

Uses raw ASGI middleware (NOT BaseHTTPMiddleware) to preserve SSE streaming.
Provides request logging, error handling, security headers, and rate limiting.
"""

import asyncio
import collections
import hmac
import json
import logging
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.config import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "RequestLoggingMiddleware",
    "ErrorHandlingMiddleware",
    "SecurityHeadersMiddleware",
    "ApiKeyMiddleware",
    "RateLimitMiddleware",
    "RequestBodyLimitMiddleware",
]


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

    Prevents stack traces from leaking to clients.  When this middleware
    is the outermost layer, its error responses bypass SecurityHeadersMiddleware.
    To ensure security headers are always present (even on 500s), the error
    response includes the same headers that SecurityHeadersMiddleware adds.
    """

    # Security headers included in error responses so that 500s generated
    # by this outermost middleware still carry all security headers.
    _SECURITY_HEADERS = [
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (b"x-xss-protection", b"1; mode=block"),
    ]

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
                        ]
                        + list(self._SECURITY_HEADERS),
                    }
                )
                await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware:
    """Add security headers to every HTTP response."""

    # CSP: 'unsafe-inline' required for single-file demo HTML with inline scripts.
    # Production deployment should externalize scripts and use nonce-based CSP.
    # Trade-off documented: demo simplicity vs strict CSP -- acceptable for
    # interview demo where the frontend is secondary to the agent architecture.
    # No user-generated content is rendered as HTML, so XSS surface is minimal.
    # Production path: externalize CSS/JS into separate static files and
    # replace 'unsafe-inline' with nonce-based CSP (generate per-request
    # nonce in middleware, inject into <script nonce="..."> tags).
    HEADERS = [
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (b"strict-transport-security", b"max-age=63072000; includeSubDomains"),
        (
            b"content-security-policy",
            b"default-src 'self'; script-src 'self' 'unsafe-inline'; "
            b"style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            b"font-src 'self' https://fonts.gstatic.com; "
            b"img-src 'self' data:; connect-src 'self'",
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


class ApiKeyMiddleware:
    """Validate ``X-API-Key`` header on protected endpoints.

    When ``API_KEY`` is empty (default), authentication is disabled and all
    requests pass through.  When set, ``/chat`` requires a matching key.
    Uses ``hmac.compare_digest`` to prevent timing attacks.
    """

    _PROTECTED_PATHS = {"/chat"}

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._api_key = get_settings().API_KEY.get_secret_value()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._api_key:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        if path not in self._PROTECTED_PATHS:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        provided = headers.get(b"x-api-key", b"").decode()

        if not provided or not hmac.compare_digest(provided, self._api_key):
            body = json.dumps(
                {"error": "unauthorized", "message": "Invalid or missing API key."}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)


class RateLimitMiddleware:
    """Sliding-window rate limiter per client IP.

    Only applies to ``/chat`` endpoint. ``/health`` and static files are exempt.
    Returns 429 with ``Retry-After`` header when the limit is exceeded.
    Respects ``X-Forwarded-For`` behind reverse proxies (Cloud Run, nginx).
    Caps tracked clients to ``RATE_LIMIT_MAX_CLIENTS`` to bound memory.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        settings = get_settings()
        self.max_tokens = settings.RATE_LIMIT_CHAT
        self.max_clients = settings.RATE_LIMIT_MAX_CLIENTS
        self.window_seconds = 60.0
        # {ip: deque of request timestamps} — defaultdict avoids check-then-set race
        self._requests: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        # Protects _requests mutations under concurrent async requests
        self._lock = asyncio.Lock()

    def _get_client_ip(self, scope: Scope) -> str:
        """Extract client IP, preferring X-Forwarded-For behind proxies."""
        headers = dict(scope.get("headers", []))
        forwarded = headers.get(b"x-forwarded-for", b"").decode()
        if forwarded:
            # X-Forwarded-For: client, proxy1, proxy2 — take the first (client) IP
            return forwarded.split(",")[0].strip()
        client = scope.get("client")
        return client[0] if client else "unknown"

    async def _is_allowed(self, client_ip: str) -> bool:
        """Check if a request from client_ip is within the rate limit."""
        now = time.monotonic()
        window_start = now - self.window_seconds

        async with self._lock:
            # Memory guard: evict oldest client if at capacity
            if client_ip not in self._requests and len(self._requests) >= self.max_clients:
                oldest_ip = next(iter(self._requests))
                del self._requests[oldest_ip]

            bucket = self._requests[client_ip]

            # Evict expired entries
            while bucket and bucket[0] < window_start:
                bucket.popleft()

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

        client_ip = self._get_client_ip(scope)

        if await self._is_allowed(client_ip):
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


class RequestBodyLimitMiddleware:
    """Reject HTTP requests whose body exceeds a configurable limit.

    Prevents resource exhaustion from oversized payloads. Two layers:
    1. Fast-path: checks ``Content-Length`` header when present.
    2. Streaming enforcement: counts actual bytes received via ``receive()``
       to catch chunked transfers or missing ``Content-Length``.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._max_size = get_settings().MAX_REQUEST_BODY_SIZE

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length", b"0")

        try:
            size = int(content_length)
        except (ValueError, TypeError):
            size = 0

        if size > self._max_size:
            await self._send_413(send)
            return

        # Streaming enforcement: count actual bytes for chunked/missing Content-Length
        bytes_received = 0
        exceeded = False

        async def receive_wrapper() -> Message:
            nonlocal bytes_received, exceeded
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                bytes_received += len(body)
                if bytes_received > self._max_size:
                    exceeded = True
            return message

        async def send_wrapper(message: Message) -> None:
            if exceeded and message.get("type") == "http.response.start":
                await self._send_413(send)
                return
            await send(message)

        await self.app(scope, receive_wrapper, send_wrapper)

    async def _send_413(self, send: Send) -> None:
        body = json.dumps(
            {
                "error": "payload_too_large",
                "message": f"Request body exceeds {self._max_size} bytes.",
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
