"""Tests for pure ASGI middleware (middleware.py).

Tests each middleware in isolation using Starlette test utilities.
Only deterministic tests retained (no MagicMock/patch).

Mock-based tests (ApiKey with mock settings, RateLimit with mock settings,
TTL refresh, LRU eviction, concurrency, body limit with mock settings)
removed (mock purge R111).
"""

import pytest

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
        # RFC 7807 Problem Details format
        assert data["code"] == "internal_error"
        assert data["type"] == "about:blank"
        assert data["status"] == 500
        assert "detail" in data

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

        app = Starlette(routes=[Route("/health", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.get("/health")
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
        """CSP header includes 'self' directive on API paths."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = Starlette(routes=[Route("/health", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.get("/health")
        assert "'self'" in resp.headers["content-security-policy"]

    def test_csp_omitted_for_static_paths(self):
        """R39 fix M-007: Non-API paths skip CSP (static files may need inline styles)."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = Starlette(routes=[Route("/some-page", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.get("/some-page")
        # Security headers still present
        assert resp.headers["x-content-type-options"] == "nosniff"
        # But CSP is NOT present for non-API paths
        assert "content-security-policy" not in resp.headers


class TestCSPNonce:
    """CSP uses strict policy for API endpoints (R36: nonce removed, R39: API-only)."""

    def test_csp_has_no_unsafe_inline(self):
        """CSP header must not contain unsafe-inline."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = Starlette(routes=[Route("/health", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.get("/health")
        csp = resp.headers["content-security-policy"]
        assert "unsafe-inline" not in csp, (
            f"CSP must not contain unsafe-inline, got: {csp}"
        )

    def test_csp_is_static_no_nonce(self):
        """R36: CSP is static for API-only backend -- no per-request nonce."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = Starlette(routes=[Route("/health", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        csp = client.get("/health").headers["content-security-policy"]
        assert "'nonce-" not in csp, (
            f"API-only CSP should not contain nonce, got: {csp}"
        )

    def test_csp_consistent_across_requests(self):
        """R36: Static CSP produces identical headers across requests."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = Starlette(routes=[Route("/health", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        csp1 = client.get("/health").headers["content-security-policy"]
        csp2 = client.get("/health").headers["content-security-policy"]
        assert csp1 == csp2, "Static CSP should be identical across requests"

    def test_csp_preserves_google_fonts(self):
        """CSP still allows Google Fonts."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = Starlette(routes=[Route("/health", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        csp = client.get("/health").headers["content-security-policy"]
        assert "https://fonts.googleapis.com" in csp
        assert "https://fonts.gstatic.com" in csp

    def test_csp_script_and_style_directives(self):
        """CSP includes script-src and style-src directives."""
        from src.api.middleware import SecurityHeadersMiddleware

        app = Starlette(routes=[Route("/health", _ok_app)])
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)
        csp = client.get("/health").headers["content-security-policy"]
        assert "script-src" in csp
        assert "style-src" in csp


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
        # RFC 7807 Problem Details format
        assert data["code"] == "rate_limit_exceeded"
        assert data["type"] == "about:blank"
        assert data["status"] == 429
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


class TestMiddlewareOrdering:
    """R55 fix D5: Verify middleware is applied in the documented order."""

    def test_middleware_stack_contains_all_required_types(self):
        """All 6 custom middleware types are present in the app configuration."""
        from src.api.app import app
        from src.api.middleware import (
            ApiKeyMiddleware,
            ErrorHandlingMiddleware,
            RateLimitMiddleware,
            RequestBodyLimitMiddleware,
            RequestLoggingMiddleware,
            SecurityHeadersMiddleware,
        )

        required_types = {
            ApiKeyMiddleware,
            ErrorHandlingMiddleware,
            RateLimitMiddleware,
            RequestBodyLimitMiddleware,
            RequestLoggingMiddleware,
            SecurityHeadersMiddleware,
        }
        found_types = {m.cls for m in app.user_middleware if m.cls in required_types}

        assert found_types == required_types, (
            f"Missing middleware: {required_types - found_types}"
        )

    def test_middleware_execution_order(self):
        """Middleware executes in the documented order (ADR-010)."""
        from src.api.app import app
        from src.api.middleware import (
            ApiKeyMiddleware,
            ErrorHandlingMiddleware,
            RateLimitMiddleware,
            RequestBodyLimitMiddleware,
            RequestLoggingMiddleware,
            SecurityHeadersMiddleware,
        )

        # Build ordered list of our middleware (skip CORSMiddleware)
        our_middleware = {
            RequestBodyLimitMiddleware,
            ErrorHandlingMiddleware,
            RequestLoggingMiddleware,
            SecurityHeadersMiddleware,
            RateLimitMiddleware,
            ApiKeyMiddleware,
        }
        order = [m.cls.__name__ for m in app.user_middleware if m.cls in our_middleware]

        # BodyLimit must be outermost (first in list)
        assert order[0] == "RequestBodyLimitMiddleware", (
            f"BodyLimit must be outermost (index 0). Got: {order}"
        )

        # ErrorHandling must be before Logging, Security, RateLimit, ApiKey
        eh_idx = order.index("ErrorHandlingMiddleware")
        rl_idx = order.index("RequestLoggingMiddleware")
        assert eh_idx < rl_idx, (
            f"ErrorHandling must be before RequestLogging. Got: {order}"
        )

        # RateLimit BEFORE ApiKey (prevents brute-force: R48 fix)
        rate_idx = order.index("RateLimitMiddleware")
        auth_idx = order.index("ApiKeyMiddleware")
        assert rate_idx < auth_idx, (
            f"RateLimit must execute before ApiKey (prevent brute-force). Got: {order}"
        )

        # ApiKey must be innermost (last in our middleware list)
        assert order[-1] == "ApiKeyMiddleware", (
            f"ApiKey must be innermost (last). Got: {order}"
        )
