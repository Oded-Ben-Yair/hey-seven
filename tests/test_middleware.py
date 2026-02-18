"""Tests for pure ASGI middleware (middleware.py).

Tests each middleware in isolation using Starlette test utilities.
"""

from unittest.mock import MagicMock

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient


def _ok_app(request: Request) -> JSONResponse:
    """Simple handler that returns 200 OK."""
    return JSONResponse({"ok": True})


def _error_app(request: Request):
    """Handler that raises an unhandled exception."""
    raise RuntimeError("Intentional test error")


class TestRequestLoggingMiddleware:
    def test_adds_request_id_header(self):
        """Response includes X-Request-ID header."""
        from src.api.middleware import RequestLoggingMiddleware

        app = Starlette(routes=[Route("/test", _ok_app)])
        app.add_middleware(RequestLoggingMiddleware)
        client = TestClient(app)
        resp = client.get("/test")
        assert "x-request-id" in resp.headers

    def test_preserves_existing_request_id(self):
        """If client sends X-Request-ID, it is preserved."""
        from src.api.middleware import RequestLoggingMiddleware

        app = Starlette(routes=[Route("/test", _ok_app)])
        app.add_middleware(RequestLoggingMiddleware)
        client = TestClient(app)
        resp = client.get("/test", headers={"X-Request-ID": "custom-id-42"})
        assert resp.headers["x-request-id"] == "custom-id-42"

    def test_adds_response_time_header(self):
        """Response includes X-Response-Time-Ms header."""
        from src.api.middleware import RequestLoggingMiddleware

        app = Starlette(routes=[Route("/test", _ok_app)])
        app.add_middleware(RequestLoggingMiddleware)
        client = TestClient(app)
        resp = client.get("/test")
        assert "x-response-time-ms" in resp.headers
        ms = float(resp.headers["x-response-time-ms"])
        assert ms >= 0


class TestErrorHandlingMiddleware:
    def test_returns_500_json_on_unhandled_exception(self):
        """Unhandled exceptions return structured 500 JSON."""
        from src.api.middleware import ErrorHandlingMiddleware

        app = Starlette(routes=[Route("/error", _error_app)])
        app.add_middleware(ErrorHandlingMiddleware)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/error")
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"]["code"] == "internal_error"
        assert "message" in data["error"]

    def test_passes_through_normal_requests(self):
        """Normal requests pass through without modification."""
        from src.api.middleware import ErrorHandlingMiddleware

        app = Starlette(routes=[Route("/ok", _ok_app)])
        app.add_middleware(ErrorHandlingMiddleware)
        client = TestClient(app)
        resp = client.get("/ok")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


class TestSecurityHeadersMiddleware:
    def test_all_security_headers_present(self):
        """All expected security headers are set on responses."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = Starlette(routes=[Route("/test", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.get("/test")
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["x-frame-options"] == "DENY"
        assert resp.headers["referrer-policy"] == "strict-origin-when-cross-origin"
        assert "content-security-policy" in resp.headers

    def test_hsts_header_present(self):
        """HSTS header is set with appropriate max-age."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = Starlette(routes=[Route("/test", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.get("/test")
        assert "strict-transport-security" in resp.headers
        assert "max-age=" in resp.headers["strict-transport-security"]

    def test_csp_includes_self(self):
        """CSP header includes 'self' directive."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = Starlette(routes=[Route("/test", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.get("/test")
        assert "'self'" in resp.headers["content-security-policy"]


class TestRateLimitMiddleware:
    def test_allows_requests_within_limit(self):
        """Requests within the rate limit pass through."""
        from src.api.middleware import RateLimitMiddleware

        app = Starlette(routes=[Route("/chat", _ok_app, methods=["POST"])])
        app.add_middleware(RateLimitMiddleware)
        client = TestClient(app)
        resp = client.post("/chat")
        assert resp.status_code == 200

    def test_blocks_over_limit(self):
        """Requests over the rate limit return 429."""
        from src.api.middleware import RateLimitMiddleware

        app = Starlette(routes=[Route("/chat", _ok_app, methods=["POST"])])
        app.add_middleware(RateLimitMiddleware)
        client = TestClient(app)

        # Send requests up to the limit (default 20)
        for _ in range(20):
            resp = client.post("/chat")
            assert resp.status_code == 200

        # Next request should be rate limited
        resp = client.post("/chat")
        assert resp.status_code == 429
        data = resp.json()
        assert data["error"]["code"] == "rate_limit_exceeded"
        assert "retry-after" in resp.headers

    def test_respects_x_forwarded_for(self):
        """Rate limiter uses X-Forwarded-For header for client IP."""
        from src.api.middleware import RateLimitMiddleware

        app = Starlette(routes=[Route("/chat", _ok_app, methods=["POST"])])
        app.add_middleware(RateLimitMiddleware)
        client = TestClient(app)

        # Exhaust limit for IP "1.2.3.4"
        for _ in range(20):
            client.post("/chat", headers={"X-Forwarded-For": "1.2.3.4"})

        # "1.2.3.4" should be rate limited
        resp = client.post("/chat", headers={"X-Forwarded-For": "1.2.3.4"})
        assert resp.status_code == 429

        # Different IP should still work
        resp = client.post("/chat", headers={"X-Forwarded-For": "5.6.7.8"})
        assert resp.status_code == 200

    def test_health_exempt_from_rate_limit(self):
        """GET /health is not rate limited."""
        from src.api.middleware import RateLimitMiddleware

        app = Starlette(
            routes=[
                Route("/chat", _ok_app, methods=["POST"]),
                Route("/health", _ok_app),
            ]
        )
        app.add_middleware(RateLimitMiddleware)
        client = TestClient(app)

        # Exhaust rate limit on /chat
        for _ in range(21):
            client.post("/chat")

        # /health should still work
        resp = client.get("/health")
        assert resp.status_code == 200


class TestApiKeyMiddleware:
    def test_no_key_configured_passes_through(self):
        """When API_KEY is empty, all requests pass through."""
        from unittest.mock import patch

        from src.api.middleware import ApiKeyMiddleware

        with patch("src.api.middleware.get_settings") as mock_settings:
            mock_settings.return_value.API_KEY = MagicMock(get_secret_value=lambda: "")
            app = Starlette(routes=[Route("/chat", _ok_app, methods=["POST"])])
            app.add_middleware(ApiKeyMiddleware)
            client = TestClient(app)
            resp = client.post("/chat")
            assert resp.status_code == 200

    def test_valid_key_passes(self):
        """Valid X-API-Key header passes through."""
        from unittest.mock import patch

        from src.api.middleware import ApiKeyMiddleware

        with patch("src.api.middleware.get_settings") as mock_settings:
            mock_settings.return_value.API_KEY = MagicMock(get_secret_value=lambda: "test-secret-key")
            app = Starlette(routes=[Route("/chat", _ok_app, methods=["POST"])])
            app.add_middleware(ApiKeyMiddleware)
            client = TestClient(app)
            resp = client.post("/chat", headers={"X-API-Key": "test-secret-key"})
            assert resp.status_code == 200

    def test_invalid_key_returns_401(self):
        """Invalid API key returns 401."""
        from unittest.mock import patch

        from src.api.middleware import ApiKeyMiddleware

        with patch("src.api.middleware.get_settings") as mock_settings:
            mock_settings.return_value.API_KEY = MagicMock(get_secret_value=lambda: "test-secret-key")
            app = Starlette(routes=[Route("/chat", _ok_app, methods=["POST"])])
            app.add_middleware(ApiKeyMiddleware)
            client = TestClient(app)
            resp = client.post("/chat", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "unauthorized"

    def test_missing_key_returns_401(self):
        """Missing API key returns 401 when key is configured."""
        from unittest.mock import patch

        from src.api.middleware import ApiKeyMiddleware

        with patch("src.api.middleware.get_settings") as mock_settings:
            mock_settings.return_value.API_KEY = MagicMock(get_secret_value=lambda: "test-secret-key")
            app = Starlette(routes=[Route("/chat", _ok_app, methods=["POST"])])
            app.add_middleware(ApiKeyMiddleware)
            client = TestClient(app)
            resp = client.post("/chat")
            assert resp.status_code == 401

    def test_health_exempt_from_auth(self):
        """GET /health bypasses API key check."""
        from unittest.mock import patch

        from src.api.middleware import ApiKeyMiddleware

        with patch("src.api.middleware.get_settings") as mock_settings:
            mock_settings.return_value.API_KEY = MagicMock(get_secret_value=lambda: "test-secret-key")
            app = Starlette(
                routes=[
                    Route("/chat", _ok_app, methods=["POST"]),
                    Route("/health", _ok_app),
                ]
            )
            app.add_middleware(ApiKeyMiddleware)
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200


class TestRequestBodyLimitMiddleware:
    def test_allows_normal_request(self):
        """Requests within the body size limit pass through."""
        from unittest.mock import patch

        from src.api.middleware import RequestBodyLimitMiddleware

        with patch("src.api.middleware.get_settings") as mock_settings:
            mock_settings.return_value.MAX_REQUEST_BODY_SIZE = 65536
            app = Starlette(routes=[Route("/chat", _ok_app, methods=["POST"])])
            app.add_middleware(RequestBodyLimitMiddleware)
            client = TestClient(app)
            resp = client.post("/chat", content=b'{"message": "hello"}')
            assert resp.status_code == 200

    def test_rejects_oversized_request(self):
        """Requests exceeding the body size limit return 413."""
        from unittest.mock import patch

        from src.api.middleware import RequestBodyLimitMiddleware

        with patch("src.api.middleware.get_settings") as mock_settings:
            mock_settings.return_value.MAX_REQUEST_BODY_SIZE = 100  # Very small limit
            app = Starlette(routes=[Route("/chat", _ok_app, methods=["POST"])])
            app.add_middleware(RequestBodyLimitMiddleware)
            client = TestClient(app)
            # Send a payload larger than 100 bytes
            large_body = b"x" * 200
            resp = client.post(
                "/chat",
                content=large_body,
                headers={"Content-Length": str(len(large_body))},
            )
            assert resp.status_code == 413
            assert resp.json()["error"]["code"] == "payload_too_large"

    def test_rejects_oversized_chunked_request(self):
        """Streaming enforcement catches oversized body without Content-Length header."""
        from unittest.mock import patch

        from src.api.middleware import RequestBodyLimitMiddleware

        with patch("src.api.middleware.get_settings") as mock_settings:
            mock_settings.return_value.MAX_REQUEST_BODY_SIZE = 100
            app = Starlette(routes=[Route("/chat", _ok_app, methods=["POST"])])
            app.add_middleware(RequestBodyLimitMiddleware)
            client = TestClient(app)
            # Send oversized payload via content= (simulates chunked transfer)
            large_body = b"x" * 200
            resp = client.post("/chat", content=large_body)
            assert resp.status_code == 413
