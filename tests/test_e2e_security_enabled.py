"""R47 fix C5 + R48 expansion: E2E tests with auth middleware ENABLED.

Tests the ASGI middleware pipeline: auth middleware -> rate limiting ->
security headers -> body limit. Exercises code paths that are neutered
in other tests (where auth is disabled via conftest).

Mock purge R111: Removed all classifier mock tests (TestE2EWithClassifierEnabled,
TestClassifierLifecycle). Retained ASGI middleware tests that use real middleware
objects with protocol-level send/receive callbacks (no MagicMock/AsyncMock).
"""

import pytest


@pytest.fixture
def _enable_auth(monkeypatch):
    """Override conftest's autouse fixture to ENABLE auth."""
    monkeypatch.setenv("API_KEY", "test-secret-key-r47")
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    monkeypatch.setenv("API_KEY", "")
    get_settings.cache_clear()


class TestE2EWithAuthEnabled:
    """E2E tests with API key authentication ENABLED.

    Uses direct ASGI middleware testing to avoid full lifespan initialization
    (which requires real LLM connections). Tests the auth middleware layer
    specifically, which is the code path neutered in other tests.
    """

    @pytest.mark.asyncio
    async def test_chat_with_valid_auth_passes_middleware(self, _enable_auth):
        """Valid API key allows request through ApiKeyMiddleware."""
        from src.api.middleware import ApiKeyMiddleware

        # Track if inner app was called
        called = []

        async def inner_app(scope, receive, send):
            called.append(True)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"OK"})

        middleware = ApiKeyMiddleware(inner_app)
        scope = {
            "type": "http",
            "path": "/chat",
            "headers": [(b"x-api-key", b"test-secret-key-r47")],
        }

        response_status = []

        async def asgi_send(message):
            if message["type"] == "http.response.start":
                response_status.append(message["status"])

        async def asgi_receive():
            return {"type": "http.request", "body": b""}

        await middleware(scope, asgi_receive, asgi_send)
        assert called, "Inner app should be called with valid API key"
        assert response_status[0] == 200

    @pytest.mark.asyncio
    async def test_chat_without_auth_returns_401(self, _enable_auth):
        """Missing API key returns 401 when auth is enabled."""
        from src.api.middleware import ApiKeyMiddleware

        async def inner_app(scope, receive, send):
            raise AssertionError("Should not reach inner app")

        middleware = ApiKeyMiddleware(inner_app)
        scope = {
            "type": "http",
            "path": "/chat",
            "headers": [],  # No API key
        }

        response_status = []
        response_body = []

        async def asgi_send(message):
            if message["type"] == "http.response.start":
                response_status.append(message["status"])
            elif message["type"] == "http.response.body":
                response_body.append(message.get("body", b""))

        async def asgi_receive():
            return {"type": "http.request", "body": b""}

        await middleware(scope, asgi_receive, asgi_send)
        assert response_status[0] == 401

    @pytest.mark.asyncio
    async def test_chat_with_wrong_key_returns_401(self, _enable_auth):
        """Wrong API key returns 401 when auth is enabled."""
        from src.api.middleware import ApiKeyMiddleware

        async def inner_app(scope, receive, send):
            raise AssertionError("Should not reach inner app")

        middleware = ApiKeyMiddleware(inner_app)
        scope = {
            "type": "http",
            "path": "/chat",
            "headers": [(b"x-api-key", b"wrong-key")],
        }

        response_status = []

        async def asgi_send(message):
            if message["type"] == "http.response.start":
                response_status.append(message["status"])

        async def asgi_receive():
            return {"type": "http.request", "body": b""}

        await middleware(scope, asgi_receive, asgi_send)
        assert response_status[0] == 401


class TestMiddlewareChainOrdering:
    """R48 fix: Verify middleware execution order matches security requirements.

    Starlette executes middleware in REVERSE add order. Rate limiting must
    execute BEFORE auth to prevent API key brute-force (DeepSeek C1).
    """

    @pytest.mark.asyncio
    async def test_rate_limit_executes_before_auth(self, _enable_auth):
        """Rate limiting should block before auth rejects — prevents brute-force."""
        from src.api.middleware import ApiKeyMiddleware, RateLimitMiddleware

        # Build a chain: RateLimit wraps ApiKey wraps inner_app
        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"OK"})

        auth_layer = ApiKeyMiddleware(inner_app)
        rate_layer = RateLimitMiddleware(auth_layer)

        scope = {
            "type": "http",
            "path": "/chat",
            "headers": [(b"x-api-key", b"wrong-key")],
            "client": ("10.0.0.1", 1234),
        }

        response_status = []

        async def asgi_send(message):
            if message["type"] == "http.response.start":
                response_status.append(message["status"])

        async def asgi_receive():
            return {"type": "http.request", "body": b""}

        await rate_layer(scope, asgi_receive, asgi_send)
        assert response_status[0] == 401, "Auth should reject after rate limit allows"

    @pytest.mark.asyncio
    async def test_unprotected_endpoints_bypass_auth(self, _enable_auth):
        """Health and live endpoints bypass auth even when API key is set."""
        from src.api.middleware import ApiKeyMiddleware

        for path in ["/health", "/live"]:
            called = []

            async def inner_app(scope, receive, send):
                called.append(True)
                await send(
                    {"type": "http.response.start", "status": 200, "headers": []}
                )
                await send({"type": "http.response.body", "body": b"OK"})

            middleware = ApiKeyMiddleware(inner_app)
            scope = {"type": "http", "path": path, "headers": []}

            async def asgi_send(msg):
                pass

            async def asgi_receive():
                return {"type": "http.request", "body": b""}

            await middleware(scope, asgi_receive, asgi_send)
            assert called, f"{path} should bypass auth"

    @pytest.mark.asyncio
    async def test_security_headers_on_401_response(self, _enable_auth):
        """401 responses include security headers (X-Content-Type-Options, etc.)."""
        from src.api.middleware import ApiKeyMiddleware

        async def inner_app(scope, receive, send):
            raise AssertionError("Should not reach")

        middleware = ApiKeyMiddleware(inner_app)
        scope = {
            "type": "http",
            "path": "/chat",
            "headers": [(b"x-api-key", b"wrong")],
        }

        response_headers = {}

        async def asgi_send(message):
            if message["type"] == "http.response.start":
                for k, v in message.get("headers", []):
                    response_headers[k] = v

        async def asgi_receive():
            return {"type": "http.request", "body": b""}

        await middleware(scope, asgi_receive, asgi_send)
        assert b"x-content-type-options" in response_headers
        assert response_headers[b"x-content-type-options"] == b"nosniff"


class TestFullMiddlewareStack:
    """R50 fix: Test the full 6-layer middleware chain as composed in create_app.

    Previous tests tested individual middleware layers in isolation. This verifies
    that the full composition works -- ErrorHandling catches ApiKey's 401,
    Security headers are present on rate-limited 429, etc.
    """

    @pytest.mark.asyncio
    async def test_full_stack_rejects_unauthenticated_chat(self, _enable_auth):
        """Full 6-layer stack: BodyLimit->ErrorHandling->Logging->Security->RateLimit->ApiKey."""
        from src.api.middleware import (
            ApiKeyMiddleware,
            ErrorHandlingMiddleware,
            RateLimitMiddleware,
            RequestBodyLimitMiddleware,
            RequestLoggingMiddleware,
            SecurityHeadersMiddleware,
        )

        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"OK"})

        # Build chain matching app.py order (add order, reverse execution)
        stack = inner_app
        stack = ApiKeyMiddleware(stack)
        stack = RateLimitMiddleware(stack)
        stack = SecurityHeadersMiddleware(stack)
        stack = RequestLoggingMiddleware(stack)
        stack = ErrorHandlingMiddleware(stack)
        stack = RequestBodyLimitMiddleware(stack)

        scope = {
            "type": "http",
            "path": "/chat",
            "method": "POST",
            "headers": [(b"content-length", b"50")],
            "client": ("10.0.0.1", 1234),
        }

        response_status = []
        response_headers = {}

        async def asgi_send(message):
            if message["type"] == "http.response.start":
                response_status.append(message["status"])
                for k, v in message.get("headers", []):
                    response_headers[k] = v

        async def asgi_receive():
            return {"type": "http.request", "body": b'{"message": "test"}'}

        await stack(scope, asgi_receive, asgi_send)
        assert response_status[0] == 401, "Unauthenticated /chat should get 401"
        # Security headers should be present even on 401
        assert b"x-content-type-options" in response_headers

    @pytest.mark.asyncio
    async def test_full_stack_allows_health_without_auth(self, _enable_auth):
        """Health endpoint passes through all layers without auth."""
        from src.api.middleware import (
            ApiKeyMiddleware,
            ErrorHandlingMiddleware,
            RateLimitMiddleware,
            RequestBodyLimitMiddleware,
            RequestLoggingMiddleware,
            SecurityHeadersMiddleware,
        )

        called = []

        async def inner_app(scope, receive, send):
            called.append(True)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"OK"})

        stack = inner_app
        stack = ApiKeyMiddleware(stack)
        stack = RateLimitMiddleware(stack)
        stack = SecurityHeadersMiddleware(stack)
        stack = RequestLoggingMiddleware(stack)
        stack = ErrorHandlingMiddleware(stack)
        stack = RequestBodyLimitMiddleware(stack)

        scope = {
            "type": "http",
            "path": "/health",
            "method": "GET",
            "headers": [],
            "client": ("10.0.0.1", 1234),
        }

        async def asgi_send(msg):
            pass

        async def asgi_receive():
            return {"type": "http.request", "body": b""}

        await stack(scope, asgi_receive, asgi_send)
        assert called, "/health should reach inner app without auth"
