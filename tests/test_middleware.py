"""Tests for pure ASGI middleware (middleware.py).

Tests each middleware in isolation using Starlette test utilities.
"""


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
        assert data["error"] == "internal_server_error"

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
        assert data["error"] == "rate_limit_exceeded"
        assert "retry-after" in resp.headers

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
