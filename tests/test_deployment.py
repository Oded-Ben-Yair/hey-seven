"""Deployment readiness tests for Hey Seven Property Q&A Agent.

Tests verify: Dockerfile correctness, cloudbuild.yaml configuration,
dependency pinning, Cloud Run probe setup, health/liveness separation,
and security hardening for production deployment.
"""

import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_test_app(agent=True, property_data=None):
    """Create a test app with optional mocked agent."""
    default_data = {
        "property": {"name": "Test Casino", "location": "Test City"},
        "restaurants": [{"name": "Steakhouse"}],
    }
    data = property_data or default_data

    @asynccontextmanager
    async def test_lifespan(app):
        app.state.agent = MagicMock() if agent else None
        app.state.property_data = data
        app.state.ready = True
        yield
        app.state.ready = False

    __import__("src.api.app")
    app_module = sys.modules["src.api.app"]
    original = app_module.lifespan
    app_module.lifespan = test_lifespan
    try:
        return app_module.create_app()
    finally:
        app_module.lifespan = original


# ---------------------------------------------------------------------------
# 1. Dependency Pinning
# ---------------------------------------------------------------------------


class TestDependencyPinning:
    """All production dependencies must be pinned to exact versions."""

    def test_prod_requirements_all_pinned(self):
        """Every dependency in requirements-prod.txt uses == (no >= or ~=)."""
        req_path = PROJECT_ROOT / "requirements-prod.txt"
        assert req_path.exists(), "requirements-prod.txt not found"
        lines = req_path.read_text().splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue
            assert ">=" not in stripped, (
                f"Unpinned dependency in requirements-prod.txt: {stripped}"
            )
            assert "~=" not in stripped, (
                f"Approximately-pinned dependency in requirements-prod.txt: {stripped}"
            )
            assert "==" in stripped, (
                f"Dependency without exact pin: {stripped}"
            )

    def test_dev_requirements_all_pinned(self):
        """Every dependency in requirements.txt uses == (no >= or ~=)."""
        req_path = PROJECT_ROOT / "requirements.txt"
        assert req_path.exists()
        lines = req_path.read_text().splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue
            assert ">=" not in stripped, f"Unpinned dep: {stripped}"
            assert "==" in stripped, f"No exact pin: {stripped}"

    def test_langfuse_pinned_exact(self):
        """langfuse must be pinned to an exact version (not >=2.0)."""
        req_path = PROJECT_ROOT / "requirements-prod.txt"
        content = req_path.read_text()
        match = re.search(r"langfuse==(\d+\.\d+\.\d+)", content)
        assert match, "langfuse must be pinned with == in requirements-prod.txt"

    def test_cryptography_pinned_exact(self):
        """cryptography must be pinned to an exact version (not >=43.0)."""
        req_path = PROJECT_ROOT / "requirements-prod.txt"
        content = req_path.read_text()
        match = re.search(r"cryptography==(\d+\.\d+\.\d+)", content)
        assert match, "cryptography must be pinned with == in requirements-prod.txt"


# ---------------------------------------------------------------------------
# 2. Dockerfile Best Practices
# ---------------------------------------------------------------------------


class TestDockerfile:
    """Dockerfile follows production best practices."""

    @pytest.fixture(autouse=True)
    def _load_dockerfile(self):
        self.dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()

    def test_multi_stage_build(self):
        """Dockerfile uses multi-stage build (builder + production)."""
        assert "AS builder" in self.dockerfile
        assert "FROM python:" in self.dockerfile
        assert self.dockerfile.count("FROM ") >= 2

    def test_non_root_user(self):
        """Container runs as non-root user."""
        assert "USER appuser" in self.dockerfile

    def test_exec_form_cmd(self):
        """CMD uses exec form (receives SIGTERM at PID 1)."""
        # Exec form starts with CMD [
        assert 'CMD ["python"' in self.dockerfile

    def test_python_unbuffered(self):
        """PYTHONUNBUFFERED=1 set for proper log streaming."""
        assert "PYTHONUNBUFFERED=1" in self.dockerfile

    def test_python_no_bytecode(self):
        """PYTHONDONTWRITEBYTECODE=1 set to avoid .pyc in container."""
        assert "PYTHONDONTWRITEBYTECODE=1" in self.dockerfile

    def test_python_hashseed(self):
        """PYTHONHASHSEED set for deterministic behavior."""
        assert "PYTHONHASHSEED" in self.dockerfile

    def test_healthcheck_documented_as_cloud_run_ignored(self):
        """HEALTHCHECK has comment noting Cloud Run ignores it."""
        assert "Cloud Run ignores" in self.dockerfile

    def test_graceful_shutdown_timeout(self):
        """Graceful shutdown timeout is >= 15s to allow SSE drain."""
        match = re.search(r"--timeout-graceful-shutdown.*?(\d+)", self.dockerfile)
        assert match, "graceful shutdown timeout not found"
        timeout = int(match.group(1))
        assert timeout >= 15, (
            f"Graceful shutdown {timeout}s too short for SSE streams (60s SSE timeout)"
        )

    def test_apt_lists_cleaned(self):
        """apt-get lists cleaned in same RUN layer (prevents bloated image)."""
        assert "rm -rf /var/lib/apt/lists" in self.dockerfile

    def test_prod_requirements_used(self):
        """Builder stage uses requirements-prod.txt (not requirements.txt)."""
        assert "requirements-prod.txt" in self.dockerfile


# ---------------------------------------------------------------------------
# 3. Cloud Build Pipeline
# ---------------------------------------------------------------------------


class TestCloudBuild:
    """cloudbuild.yaml has proper CI/CD pipeline configuration."""

    @pytest.fixture(autouse=True)
    def _load_cloudbuild(self):
        self.cloudbuild = (PROJECT_ROOT / "cloudbuild.yaml").read_text()

    def test_trivy_scanner_pinned(self):
        """Trivy scanner image is pinned (not :latest)."""
        assert "trivy:latest" not in self.cloudbuild, (
            "Trivy scanner must be pinned to a specific version"
        )
        assert re.search(r"trivy:\d+\.\d+\.\d+", self.cloudbuild), (
            "Trivy must use a version-pinned image tag"
        )

    def test_trivy_blocks_critical_high(self):
        """Trivy scan blocks deploys on CRITICAL/HIGH vulnerabilities."""
        assert "--severity=CRITICAL,HIGH" in self.cloudbuild
        assert "--exit-code=1" in self.cloudbuild

    def test_coverage_gate(self):
        """Tests must pass with >= 90% coverage."""
        assert "--cov-fail-under=90" in self.cloudbuild

    def test_lint_and_typecheck(self):
        """Pipeline runs ruff (lint) and mypy (typecheck)."""
        assert "ruff check" in self.cloudbuild
        assert "mypy" in self.cloudbuild

    def test_cloud_run_startup_probe_configured(self):
        """Cloud Run deploy includes startup probe flags."""
        assert "--startup-probe-path" in self.cloudbuild
        assert "/health" in self.cloudbuild

    def test_cloud_run_liveness_probe_configured(self):
        """Cloud Run deploy includes liveness probe on /live."""
        assert "--liveness-probe-path" in self.cloudbuild
        assert "/live" in self.cloudbuild

    def test_cpu_flag_present(self):
        """Cloud Run deploy specifies --cpu (not default 1 vCPU)."""
        assert "--cpu=" in self.cloudbuild

    def test_concurrency_flag_present(self):
        """Cloud Run deploy specifies --concurrency for SSE streams."""
        assert "--concurrency=" in self.cloudbuild

    def test_no_traffic_deploy(self):
        """Deploy uses --no-traffic for canary-safe rollout."""
        assert "--no-traffic" in self.cloudbuild

    def test_traffic_migration_after_smoke(self):
        """Traffic is migrated to latest only after smoke test."""
        assert "--to-latest" in self.cloudbuild

    def test_smoke_test_step_exists(self):
        """Pipeline includes post-deploy smoke test."""
        assert "SMOKE TEST" in self.cloudbuild

    def test_rollback_on_failure(self):
        """Pipeline includes rollback to previous revision on smoke failure."""
        assert "Rolling back" in self.cloudbuild or "rollback" in self.cloudbuild.lower()

    def test_version_in_env_vars(self):
        """VERSION is set from COMMIT_SHA (not hardcoded)."""
        assert "VERSION=$COMMIT_SHA" in self.cloudbuild

    def test_log_level_info_in_production(self):
        """LOG_LEVEL=INFO in production (not WARNING — suppresses app logs)."""
        assert "LOG_LEVEL=INFO" in self.cloudbuild

    def test_timeout_accommodates_llm_pipeline(self):
        """Cloud Run timeout >= 120s for 6-LLM-call agent pipeline."""
        match = re.search(r"--timeout=(\d+)s", self.cloudbuild)
        assert match, "Cloud Run --timeout flag not found"
        timeout = int(match.group(1))
        assert timeout >= 120, (
            f"Cloud Run timeout {timeout}s too short for multi-LLM agent pipeline"
        )

    def test_previous_revision_captured(self):
        """Pipeline captures previous revision for rollback."""
        assert "previous-revision" in self.cloudbuild


# ---------------------------------------------------------------------------
# 4. Health / Liveness Endpoint Separation
# ---------------------------------------------------------------------------


class TestLivenessEndpoint:
    """GET /live is a lightweight probe that always returns 200."""

    def test_live_returns_200(self):
        """GET /live returns 200 even when agent is healthy."""
        app = _make_test_app(agent=True)
        with TestClient(app) as client:
            resp = client.get("/live")
            assert resp.status_code == 200
            assert resp.json()["status"] == "alive"

    def test_live_returns_200_when_agent_unavailable(self):
        """GET /live returns 200 even when agent is None (degraded)."""
        app = _make_test_app(agent=False)
        with TestClient(app) as client:
            resp = client.get("/live")
            assert resp.status_code == 200
            assert resp.json()["status"] == "alive"


class TestHealthEndpointDeployment:
    """GET /health serves as readiness/startup probe."""

    def test_health_includes_environment(self):
        """GET /health response includes environment field."""
        app = _make_test_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            data = resp.json()
            assert "environment" in data

    def test_health_200_when_healthy(self):
        """GET /health returns 200 when all components ready."""
        app = _make_test_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"

    def test_health_503_when_degraded(self):
        """GET /health returns 503 when agent unavailable (Cloud Run stops routing)."""
        app = _make_test_app(agent=False)
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 503
            assert resp.json()["status"] == "degraded"

    def test_health_has_version(self):
        """GET /health includes version for post-deploy assertion."""
        app = _make_test_app()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert "version" in resp.json()


# ---------------------------------------------------------------------------
# 5. Security Hardening
# ---------------------------------------------------------------------------


class TestSecurityDeployment:
    """Security configuration for production deployment."""

    def test_sms_webhook_returns_404_when_disabled(self):
        """POST /sms/webhook returns 404 when SMS_ENABLED=False."""
        app = _make_test_app()
        with TestClient(app) as client:
            resp = client.post(
                "/sms/webhook",
                json={"data": {"event_type": "message.received", "payload": {}}},
            )
            assert resp.status_code == 404
            assert resp.json()["error"]["code"] == "not_found"

    def test_not_found_error_code_exists(self):
        """ErrorCode.NOT_FOUND exists in error taxonomy."""
        from src.api.errors import ErrorCode

        assert hasattr(ErrorCode, "NOT_FOUND")
        assert ErrorCode.NOT_FOUND.value == "not_found"

    def test_rate_limit_middleware_has_cloud_run_tradeoff_doc(self):
        """RateLimitMiddleware documents in-memory trade-off for Cloud Run."""
        from src.api.middleware import RateLimitMiddleware

        docstring = RateLimitMiddleware.__doc__ or ""
        assert "Cloud Run" in docstring, (
            "RateLimitMiddleware must document Cloud Run scaling trade-off"
        )

    def test_dockerignore_excludes_sensitive_dirs(self):
        """`.dockerignore` excludes .env, tests, reviews, research."""
        ignore_path = PROJECT_ROOT / ".dockerignore"
        content = ignore_path.read_text()
        for pattern in [".env", "tests/", "reviews/", "research/"]:
            assert pattern in content, f".dockerignore missing {pattern}"


# ---------------------------------------------------------------------------
# 6. Configuration Validation
# ---------------------------------------------------------------------------


class TestConfigDeployment:
    """Configuration validates correctly for production deployment."""

    def test_production_requires_api_key(self, monkeypatch):
        """Settings hard-fail when ENVIRONMENT=production and API_KEY is empty."""
        from src.config import Settings

        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "production")
        with pytest.raises(ValueError, match="API_KEY must be set"):
            Settings()

    def test_production_requires_cms_webhook_secret(self, monkeypatch):
        """Settings hard-fail when ENVIRONMENT=production and CMS_WEBHOOK_SECRET empty."""
        from src.config import Settings

        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.delenv("CMS_WEBHOOK_SECRET", raising=False)
        with pytest.raises(ValueError, match="CMS_WEBHOOK_SECRET must be set"):
            Settings()

    def test_embedding_model_normalized(self, monkeypatch):
        """EMBEDDING_MODEL strips models/ prefix for vector space consistency."""
        from src.config import Settings

        monkeypatch.setenv("EMBEDDING_MODEL", "models/text-embedding-004")
        s = Settings()
        assert s.EMBEDDING_MODEL == "text-embedding-004"

    def test_rag_chunk_overlap_validation(self, monkeypatch):
        """RAG_CHUNK_OVERLAP >= RAG_CHUNK_SIZE raises ValueError."""
        from src.config import Settings

        monkeypatch.setenv("RAG_CHUNK_OVERLAP", "900")
        monkeypatch.setenv("RAG_CHUNK_SIZE", "800")
        with pytest.raises(ValueError, match="RAG_CHUNK_OVERLAP"):
            Settings()


# ---------------------------------------------------------------------------
# 7. Models
# ---------------------------------------------------------------------------


class TestLiveResponseModel:
    """LiveResponse model is correctly defined."""

    def test_live_response_default_status(self):
        from src.api.models import LiveResponse

        resp = LiveResponse()
        assert resp.status == "alive"

    def test_health_response_has_environment(self):
        from src.api.models import HealthResponse

        resp = HealthResponse(
            status="healthy",
            version="1.0.0",
            agent_ready=True,
            property_loaded=True,
        )
        assert resp.environment == "development"
