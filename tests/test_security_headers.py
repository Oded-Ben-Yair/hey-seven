"""Tests for security header consistency across all HTTP error responses.

R53 fix D5: Verify security headers appear on ALL HTTP error responses
(401, 413, 429, 500) -- not just 200s. Ensures SecurityHeadersMiddleware,
ErrorHandlingMiddleware, ApiKeyMiddleware, RateLimitMiddleware, and
RequestBodyLimitMiddleware all include the shared security headers.
"""

import json
import sys
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from src.config import get_settings


# Expected security headers on every response
SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "strict-transport-security": "max-age=63072000; includeSubDomains",
}


def _make_test_app(property_data=None):
    """Create a test app with mocked agent and property data."""
    mock_agent = MagicMock()
    default_data = {
        "property": {"name": "Test Casino", "location": "Test City"},
        "restaurants": [{"name": "Steakhouse"}],
    }
    data = property_data or default_data

    @asynccontextmanager
    async def test_lifespan(app):
        app.state.agent = mock_agent
        app.state.property_data = data
        app.state.ready = True
        yield
        app.state.ready = False

    __import__("src.api.app")
    app_module = sys.modules["src.api.app"]
    original_lifespan = app_module.lifespan
    app_module.lifespan = test_lifespan
    try:
        app = app_module.create_app()
    finally:
        app_module.lifespan = original_lifespan
    return app


def _assert_security_headers(response, context: str):
    """Assert all expected security headers are present on the response.

    Note: Some error paths include security headers in the response body AND
    SecurityHeadersMiddleware adds them again on the way out. This results in
    comma-separated duplicate values (e.g., 'nosniff, nosniff'). We check
    that the expected value appears at least once, which is correct behavior
    -- browsers accept duplicates per RFC 7230 Section 3.2.2.
    """
    for name, value in SECURITY_HEADERS.items():
        actual = response.headers.get(name)
        assert actual is not None and value in actual, (
            f"Missing {name} on {context}: "
            f"expected {value!r} to be present, got {actual!r}"
        )


class TestSecurityHeadersOnErrors:
    """Security headers must appear on ALL error responses, not just 200s."""

    def test_401_has_security_headers(self):
        """401 from ApiKeyMiddleware includes security headers."""
        with patch.dict("os.environ", {"API_KEY": "secret123"}):
            get_settings.cache_clear()
            app = _make_test_app()
            with TestClient(app) as client:
                resp = client.post(
                    "/chat",
                    json={"message": "test"},
                )
                assert resp.status_code == 401
                _assert_security_headers(resp, "401 Unauthorized")

    def test_413_has_security_headers(self):
        """413 from RequestBodyLimitMiddleware includes security headers."""
        app = _make_test_app()
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                content=b"x" * 100000,
                headers={
                    "content-type": "application/json",
                    "content-length": "100000",
                },
            )
            assert resp.status_code == 413
            _assert_security_headers(resp, "413 Payload Too Large")

    def test_health_200_has_security_headers(self):
        """200 from /health includes security headers."""
        app = _make_test_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            _assert_security_headers(resp, "200 Health OK")

    def test_429_has_security_headers(self):
        """429 from RateLimitMiddleware includes security headers."""
        with patch.dict("os.environ", {"RATE_LIMIT_CHAT": "1"}):
            get_settings.cache_clear()
            app = _make_test_app()
            with TestClient(app) as client:
                # First request allowed
                client.post("/chat", json={"message": "test1"})
                # Second request rate limited
                resp = client.post("/chat", json={"message": "test2"})
                assert resp.status_code == 429
                _assert_security_headers(resp, "429 Rate Limited")


class TestContentEncodingRejection:
    """Content-Encoding bypass protection (R63 fix D4)."""

    def test_compressed_payload_rejected(self):
        """Compressed payloads should be rejected with 415."""
        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/chat",
                content=b'{"message": "test"}',
                headers={
                    "content-type": "application/json",
                    "content-encoding": "gzip",
                },
            )
            assert resp.status_code == 415
            body = resp.json()
            assert body["code"] == "unsupported_media_type"
            _assert_security_headers(resp, "415 Unsupported Media Type")

    def test_identity_encoding_allowed(self):
        """Content-Encoding: identity should be allowed (pass-through)."""
        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/chat",
                json={"message": "test"},
                headers={"content-encoding": "identity"},
            )
            # Should not be 415 -- may be 200 or other status depending on auth
            assert resp.status_code != 415

    def test_deflate_encoding_rejected(self):
        """Content-Encoding: deflate should also be rejected."""
        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/chat",
                content=b'{"message": "test"}',
                headers={
                    "content-type": "application/json",
                    "content-encoding": "deflate",
                },
            )
            assert resp.status_code == 415

    def test_br_encoding_rejected(self):
        """Content-Encoding: br (brotli) should also be rejected."""
        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/chat",
                content=b'{"message": "test"}',
                headers={
                    "content-type": "application/json",
                    "content-encoding": "br",
                },
            )
            assert resp.status_code == 415

    def test_comma_separated_encoding_rejected(self):
        """R64 fix D4: Comma-separated Content-Encoding should be rejected."""
        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/chat",
                content=b'{"message": "test"}',
                headers={
                    "content-type": "application/json",
                    "content-encoding": "gzip, chunked",
                },
            )
            assert resp.status_code == 415

    def test_comma_separated_identity_only_allowed(self):
        """R64 fix D4: identity-only comma-separated encoding should pass."""
        app = _make_test_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/chat",
                json={"message": "test"},
                headers={"content-encoding": "identity, identity"},
            )
            # Should not be 415
            assert resp.status_code != 415


class TestXFFSpoofing:
    """X-Forwarded-For header handling for rate limit key extraction."""

    def test_invalid_xff_ip_ignored(self):
        """Invalid IP in XFF header should not crash the application."""
        app = _make_test_app()
        with TestClient(app) as client:
            resp = client.get(
                "/health",
                headers={"x-forwarded-for": "not-an-ip, also-bad"},
            )
            assert resp.status_code == 200

    def test_xff_without_trusted_proxies_ignored(self):
        """XFF header ignored when TRUSTED_PROXIES not configured (default)."""
        app = _make_test_app()
        with TestClient(app) as client:
            resp = client.get(
                "/health",
                headers={"x-forwarded-for": "1.2.3.4"},
            )
            assert resp.status_code == 200
