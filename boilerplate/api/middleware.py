"""Pure ASGI middleware for the Casino Host API.

Uses raw ASGI middleware (not BaseHTTPMiddleware) to avoid breaking
streaming responses. Provides request logging, request ID injection,
timing headers, and structured error handling.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


def _get_structured_logger() -> logging.Logger:
    """Return a logger configured for structured JSON output.

    Cloud Run / Cloud Logging parses JSON lines from stdout automatically.
    """
    log = logging.getLogger("hey_seven.access")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


_access_logger = _get_structured_logger()


# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------


def configure_cors(app: Any) -> None:
    """Add CORS middleware with explicit origins.

    NEVER uses ``allow_origins=["*"]`` with ``allow_credentials=True`` --
    that combination is invalid per the Fetch spec and browsers will reject
    the response.

    Args:
        app: The FastAPI / Starlette application instance.
    """
    from fastapi.middleware.cors import CORSMiddleware

    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    origins = [o.strip() for o in raw.split(",") if o.strip()]

    # Credentials + wildcard is spec-invalid. If someone sets "*", treat it
    # as dev-only without credentials.
    if "*" in origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
        )


# ---------------------------------------------------------------------------
# Pure ASGI Middleware: Request Logging + Timing + Request ID
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware:
    """Pure ASGI middleware for structured request logging.

    - Injects ``X-Request-ID`` (from the incoming header or a new UUID).
    - Emits a structured JSON log line per request (Cloud Logging compatible).
    - Adds ``X-Response-Time-Ms`` header to the response.

    Does NOT use ``BaseHTTPMiddleware`` so streaming responses are not broken.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Extract or generate request ID
        headers = dict(scope.get("headers", []))
        request_id = (
            headers.get(b"x-request-id", b"").decode()
            or str(uuid.uuid4())[:8]
        )
        start_time = time.monotonic()

        method = scope.get("method", "WS")
        path = scope.get("path", "/")

        # For HTTP requests, intercept the first response message to inject headers
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


# ---------------------------------------------------------------------------
# Pure ASGI Middleware: Unhandled Error Catch
# ---------------------------------------------------------------------------


class ErrorHandlingMiddleware:
    """Catch unhandled exceptions and return a structured 500 JSON body.

    Prevents stack traces from leaking to clients. Logs full exception
    server-side for debugging.
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
