"""Pure ASGI middleware for the Hey Seven Property Q&A API.

Uses raw ASGI middleware (NOT BaseHTTPMiddleware) to preserve SSE streaming.
Provides request logging, error handling, security headers, and rate limiting.
"""

import asyncio
import base64
import collections
import hmac
import json
import logging
import secrets
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.api.errors import ErrorCode, error_response
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
        raw_id = headers.get(b"x-request-id", b"").decode()
        # Sanitize client-provided request IDs to prevent log injection:
        # strip non-alphanumeric/hyphen chars, cap length at 64.
        sanitized = "".join(c for c in raw_id if c.isalnum() or c == "-")[:64]
        request_id = sanitized or str(uuid.uuid4())[:8]
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

    # R36 fix A13: Unified with SecurityHeadersMiddleware._STATIC_HEADERS.
    # Removed deprecated x-xss-protection (Chrome removed in 2019), added
    # HSTS for parity. Error responses now carry identical headers.
    _SECURITY_HEADERS = [
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (b"strict-transport-security", b"max-age=63072000; includeSubDomains"),
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
            # Client disconnected (normal for SSE) — log at INFO, not ERROR.
            # Re-raise to preserve asyncio task cancellation semantics;
            # without re-raise, parent tasks waiting on cancellation hang.
            logger.info(
                "Client disconnected: %s %s",
                scope.get("method", "?"),
                scope.get("path", "/"),
            )
            raise
        except Exception:
            logger.exception(
                "Unhandled exception on %s %s",
                scope.get("method", "?"),
                scope.get("path", "/"),
            )
            if not response_started:
                body = json.dumps(
                    error_response(ErrorCode.INTERNAL_ERROR, "An unexpected error occurred.")
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
    """Add security headers to every HTTP response.

    R36 fix A11: CSP uses strict policy without nonce for API endpoints.
    R39 fix M-007: CSP is NOT applied to static file paths. The app mounts
    StaticFiles(html=True) at "/" for the Next.js frontend, which uses inline
    styles. Applying strict CSP to those HTML files would block rendering.
    Static files get standard security headers but no CSP; API endpoints get
    full CSP. Production recommendation: serve frontend from a separate CDN
    origin to avoid this split entirely.
    """

    # Static security headers (shared with ErrorHandlingMiddleware).
    _STATIC_HEADERS = [
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (b"strict-transport-security", b"max-age=63072000; includeSubDomains"),
    ]

    # R36 fix A11: Static CSP for API-only endpoints — no nonce needed.
    _CSP = (
        b"default-src 'self'; "
        b"script-src 'self'; "
        b"style-src 'self' https://fonts.googleapis.com; "
        b"font-src 'self' https://fonts.gstatic.com; "
        b"img-src 'self' data:; connect-src 'self'"
    )

    # API paths that benefit from strict CSP. Paths NOT in this set
    # (including static files served at /) get security headers but no CSP.
    _API_PATHS = frozenset({"/chat", "/health", "/live", "/metrics", "/graph",
                            "/property", "/feedback", "/docs", "/openapi.json"})

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/").rstrip("/") or "/"
        apply_csp = path in self._API_PATHS

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", [])) + list(self._STATIC_HEADERS)
                if apply_csp:
                    headers.append((b"content-security-policy", self._CSP))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class ApiKeyMiddleware:
    """Validate ``X-API-Key`` header on protected endpoints.

    When ``API_KEY`` is empty (default), authentication is disabled and all
    requests pass through.  When set, ``/chat`` requires a matching key.
    Uses ``hmac.compare_digest`` to prevent timing attacks.

    The API key is re-read from settings every 60 seconds (TTL) to support
    secret rotation without container restart. For demo deployments, this
    means updating the ``API_KEY`` environment variable takes effect within
    60 seconds.
    """

    _PROTECTED_PATHS = {"/chat", "/graph", "/property", "/feedback"}
    _KEY_TTL = 60  # seconds

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        # Atomic tuple: (key, timestamp) — single read/write prevents torn
        # pair races where one coroutine sees new key + old timestamp or vice
        # versa. R5 fix per DeepSeek F5 analysis.
        self._cached: tuple[str, float] = ("", 0.0)

    def _get_api_key(self) -> str:
        """Return the current API key, refreshing from settings if TTL expired."""
        now = time.monotonic()
        cached = self._cached  # Single atomic read
        if now - cached[1] > self._KEY_TTL:
            key = get_settings().API_KEY.get_secret_value()
            self._cached = (key, now)  # Single atomic write
            return key
        return cached[0]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/").rstrip("/") or "/"
        if path not in self._PROTECTED_PATHS:
            await self.app(scope, receive, send)
            return

        api_key = self._get_api_key()
        if not api_key:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        provided = headers.get(b"x-api-key", b"").decode()

        if not provided or not hmac.compare_digest(provided, api_key):
            body = json.dumps(
                error_response(ErrorCode.UNAUTHORIZED, "Invalid or missing API key.")
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    # R36 fix A9: Include security headers on 401 responses
                    # for parity with ErrorHandlingMiddleware 500 responses.
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ] + list(SecurityHeadersMiddleware._STATIC_HEADERS),
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

    **Cloud Run scaling limitation (accepted for demo, must fix for production)**:
    This uses in-memory state, so each Cloud Run instance maintains independent
    counters.  Effective rate limit across N instances = ``RATE_LIMIT_CHAT * N``.
    For the current demo deployment (``min-instances=1``, ``max-instances=10``),
    this is acceptable.

    **TODO (pre-production)**: Migrate to distributed rate limiting via one of:
    1. Cloud Armor rate limiting (zero-code, GCP-native) -- recommended
    2. Redis (Cloud Memorystore) sliding window via ``StateBackend``
    3. API Gateway quotas (if using Apigee or similar)
    Without this, bot storms can bypass limits by hitting different instances.
    Tracked as a known limitation per R11 review (3/3 reviewer consensus).

    **ADR: In-memory rate limiting for MVP**

    Decision:
        In-memory sliding-window rate limiting is acceptable for the current
        MVP deployment phase. No distributed backend is required yet.

    Context:
        The demo deployment runs on a single Cloud Run instance with
        ``min-instances=1`` to ensure a warm instance is always available.
        Traffic is low (< 100 req/min), and the primary threat model is
        accidental abuse, not coordinated bot storms.

    Failure modes under multi-instance scaling:
        Each Cloud Run instance maintains independent counters. Under
        autoscaling (up to ``max-instances=10``), the effective rate limit
        becomes ``RATE_LIMIT_CHAT * N`` where N is the active instance
        count. A client sending 20 req/min could slip through if requests
        are load-balanced across 10 instances (2 req/instance < 20 limit).

    Mitigation:
        - ``min-instances=1``: most demo traffic hits a single instance.
        - ``max-instances=10``: bounds the multiplier to 10x.
        - Cloud Armor is the recommended next step (zero-code, CDN-level).

    3-tier upgrade path:
        1. **In-memory** (current) -- single instance, demo-grade. No
           external dependencies. Suitable for < 100 req/min, single
           instance. Memory bounded by ``RATE_LIMIT_MAX_CLIENTS``.
        2. **Cloud Memorystore (Redis)** -- shared counters across all
           instances. Requires Redis instance (~$30/mo for basic tier).
           Suitable for multi-instance production with < 10K req/min.
        3. **Cloud Armor rate limiting** -- zero-code, CDN-level enforcement.
           No application changes needed. Suitable for production scale
           with DDoS protection. Recommended for GA launch.

    Upgrade trigger (R40 D8-M001):
        Upgrade to Cloud Armor when ANY of: (a) daily traffic exceeds 1000
        requests, (b) before any paid client deployment, (c) max-instances
        regularly scales above 3 (checked via Cloud Run metrics).
    """

    def __init__(self, app: ASGIApp) -> None:
        # TODO(production): Migrate to Cloud Memorystore (Redis) or Cloud Armor
        # for distributed rate limiting. See ADR in class docstring for upgrade path.
        self.app = app
        settings = get_settings()
        self.max_tokens = settings.RATE_LIMIT_CHAT
        self.max_clients = settings.RATE_LIMIT_MAX_CLIENTS
        self._trusted_proxies = frozenset(settings.TRUSTED_PROXIES) if settings.TRUSTED_PROXIES is not None else None
        self.window_seconds = 60.0
        # {ip: deque of request timestamps} — OrderedDict for LRU eviction semantics
        self._requests: collections.OrderedDict[str, collections.deque] = collections.OrderedDict()
        # R39 CRITICAL fix D8-C001: Removed per-client asyncio.Lock objects.
        # asyncio is single-threaded cooperative multitasking — context switches
        # only happen at await points. The deque operations in _is_allowed()
        # (popleft, append, len) have ZERO await points, making them inherently
        # atomic. Per-client locks added 10K Lock objects at max capacity for
        # no correctness benefit. The _requests_lock IS still needed for
        # structural dict mutations (adding/removing keys, LRU eviction) because
        # _ensure_sweep_task() can context-switch between dict membership check
        # and mutation.
        self._requests_lock = asyncio.Lock()
        # Stale-client sweep counter. R16 fix: initialized in __init__
        # instead of lazy getattr (Gemini F-009, Grok M-006 consensus).
        self._request_counter: int = 0
        # R36 fix B6: Time-based sweep fallback. Under low traffic, stale
        # clients accumulate without cleanup (never reach 100 request threshold).
        self._last_sweep: float = time.monotonic()
        # R38 fix: Background sweep task (lazy-started on first request).
        # Sweep runs outside the request path so it never blocks incoming requests.
        self._sweep_task: asyncio.Task | None = None

    @staticmethod
    def _normalize_ip(ip: str) -> str:
        """Normalize IP address for consistent rate limit keying.

        Strips IPv6 brackets, normalizes IPv4-mapped IPv6 (::ffff:1.2.3.4 -> 1.2.3.4),
        and strips port numbers if accidentally included.
        """
        ip = ip.strip()
        # Remove brackets from IPv6 (e.g., [::1] -> ::1)
        if ip.startswith("[") and "]" in ip:
            ip = ip[1:ip.index("]")]
        # Normalize IPv4-mapped IPv6 to plain IPv4
        if ip.startswith("::ffff:"):
            ip = ip[7:]
        return ip

    def _get_client_ip(self, scope: Scope) -> str:
        """Extract client IP, preferring X-Forwarded-For behind trusted proxies.

        When TRUSTED_PROXIES is None (default), XFF is NEVER trusted — use
        direct peer IP only.  This prevents IP spoofing via forged XFF headers.
        When TRUSTED_PROXIES is explicitly set (e.g., Cloud Run LB IPs),
        XFF is trusted only from those peers.

        All returned IPs are normalized via ``_normalize_ip()`` for consistent
        rate limit keying across IPv4 and IPv6 address formats.
        """
        headers = dict(scope.get("headers", []))
        forwarded = headers.get(b"x-forwarded-for", b"").decode()
        if forwarded:
            client = scope.get("client")
            peer_ip = client[0] if client else "unknown"
            trusted = self._trusted_proxies
            # None = trust nobody's XFF (default — prevents IP spoofing)
            if trusted is None:
                return self._normalize_ip(peer_ip)
            # Explicit list: trust XFF only from listed proxy IPs
            if peer_ip in trusted:
                return self._normalize_ip(forwarded.split(",")[0].strip())
            # Untrusted peer sent XFF — ignore it, use peer IP
            return self._normalize_ip(peer_ip)
        client = scope.get("client")
        return self._normalize_ip(client[0] if client else "unknown")

    async def _ensure_sweep_task(self) -> None:
        """Lazily start the background sweep task on first request.

        R38 fix: Sweep runs in a background asyncio.Task every 60s instead
        of inline during _is_allowed(). This prevents the sweep from blocking
        concurrent requests under high load.
        """
        if self._sweep_task is None or self._sweep_task.done():
            self._sweep_task = asyncio.create_task(self._background_sweep())

    async def _background_sweep(self) -> None:
        """Periodically remove stale clients whose deques are fully expired.

        Runs every 60s in the background. Acquires _requests_lock only briefly
        to collect stale keys and delete them. Does NOT block _is_allowed().
        """
        try:
            while True:
                await asyncio.sleep(60)
                try:
                    now = time.monotonic()
                    window_start = now - self.window_seconds
                    async with self._requests_lock:
                        stale = [
                            ip for ip, dq in self._requests.items()
                            if not dq or dq[-1] < window_start
                        ]
                        for ip in stale:
                            del self._requests[ip]
                        self._last_sweep = now
                except Exception:
                    # R45 fix D8-M001: Catch unexpected errors (e.g., RuntimeError
                    # from dict size change) to keep the sweep task alive. A dead
                    # sweep task causes slow memory leak from unreaped stale clients.
                    logger.warning("Background sweep iteration failed", exc_info=True)
        except asyncio.CancelledError:
            pass

    async def _is_allowed(self, client_ip: str) -> bool:
        """Check if a request from client_ip is within the rate limit.

        R39 CRITICAL fix D8-C001: Removed per-client asyncio.Lock objects.
        asyncio is single-threaded cooperative multitasking — deque operations
        (popleft, append, len) have zero await points and are inherently atomic.
        The _requests_lock protects only structural dict mutations (add/remove
        keys, LRU eviction). After the lock releases, bucket operations proceed
        without locking since no await points exist between them.

        Only allowed requests are recorded in the deque. Rejected requests
        do NOT consume deque memory. Per-client deque is bounded to
        max_tokens entries (the maximum possible within a window when all
        requests are allowed). This prevents memory exhaustion DoS: even at
        10K req/s, each client stores at most max_tokens * ~36 bytes.
        R5 fix: added maxlen per DeepSeek F2 analysis.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds

        # Start background sweep if not running
        await self._ensure_sweep_task()

        # Brief structural lock: ensure bucket exists
        async with self._requests_lock:
            # Memory guard: evict LEAST recently used client if at capacity
            if client_ip not in self._requests and len(self._requests) >= self.max_clients:
                self._requests.popitem(last=False)  # LRU eviction

            if client_ip not in self._requests:
                self._requests[client_ip] = collections.deque(maxlen=self.max_tokens)

            # Move to end (most recently used) on access
            self._requests.move_to_end(client_ip)
            bucket = self._requests[client_ip]

        # No per-client lock needed: deque ops below have zero await points,
        # so they execute atomically in the single-threaded event loop.
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

        path = scope.get("path", "/").rstrip("/") or "/"

        # Rate-limit mutable endpoints: /chat (LLM cost), /feedback (log/DoS)
        if path not in ("/chat", "/feedback"):
            await self.app(scope, receive, send)
            return

        client_ip = self._get_client_ip(scope)

        if await self._is_allowed(client_ip):
            await self.app(scope, receive, send)
            return

        # Rate limited
        body = json.dumps(
            error_response(ErrorCode.RATE_LIMITED, "Too many requests.")
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

        sent_413 = False

        async def send_wrapper(message: Message) -> None:
            nonlocal sent_413
            if exceeded:
                if not sent_413 and message.get("type") == "http.response.start":
                    await self._send_413(send)
                    sent_413 = True
                return  # Suppress all messages after exceeding limit
            await send(message)

        await self.app(scope, receive_wrapper, send_wrapper)

    async def _send_413(self, send: Send) -> None:
        body = json.dumps(
            error_response(
                ErrorCode.PAYLOAD_TOO_LARGE,
                f"Request body exceeds {self._max_size} bytes.",
            )
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
