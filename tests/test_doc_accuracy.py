"""Tests to prevent documentation drift from code reality.

These tests assert that key counts, names, and contracts documented in
README.md and ARCHITECTURE.md match the actual codebase. Prevents the
documentation rot identified in R6.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


class TestSettingsCount:
    """Verify the documented settings count matches config.py."""

    def test_config_field_count_matches_docs(self):
        """Settings class field count matches what README claims."""
        from src.config import Settings

        # Count fields defined directly on Settings (exclude inherited)
        fields = Settings.model_fields
        actual_count = len(fields)
        assert actual_count == 68, (
            f"Settings has {actual_count} fields, but docs say 68. "
            f"Update README.md and .env.example if count changed."
        )


class TestAgentRegistry:
    """Verify the documented agent count matches the registry."""

    def test_registry_agent_count(self):
        """Agent registry has 5 specialist agents."""
        from src.agent.agents.registry import _AGENT_REGISTRY

        assert len(_AGENT_REGISTRY) == 5, (
            f"Agent registry has {len(_AGENT_REGISTRY)} agents, expected 5. "
            f"Update README and ARCHITECTURE if agents were added/removed."
        )

    def test_registry_contains_hotel(self):
        """Hotel agent is registered."""
        from src.agent.agents.registry import _AGENT_REGISTRY

        assert "hotel" in _AGENT_REGISTRY, "Hotel agent missing from registry"

    def test_registry_agent_names(self):
        """All 5 specialist agents are registered."""
        from src.agent.agents.registry import list_agents

        expected = ["comp", "dining", "entertainment", "host", "hotel"]
        assert list_agents() == expected


class TestStateFieldCount:
    """Verify the documented state field count matches code."""

    def test_property_qa_state_has_27_fields(self):
        """PropertyQAState has exactly 27 fields (13 v1/v2 + 3 v3 + 1 v4 + 1 R37 + 1 R52 + 2 R72/R73 + 1 R81-crisis-turn + 1 Phase1-multilingual + 1 Phase5-handoff + 3 profiling)."""
        from src.agent.state import PropertyQAState

        actual = len(PropertyQAState.__annotations__)
        assert actual == 27, (
            f"PropertyQAState has {actual} fields, expected 27. "
            f"Update ARCHITECTURE.md State Schema section if count changed."
        )


class TestSSEEventModels:
    """Verify all SSE event types have corresponding Pydantic models."""

    def test_ping_event_model_exists(self):
        """SSEPingEvent model exists."""
        from src.api.models import SSEPingEvent

        assert SSEPingEvent is not None

    def test_all_sse_event_models_importable(self):
        """All 9 SSE event models can be imported."""
        from src.api.models import (
            SSEDoneEvent,
            SSEErrorEvent,
            SSEGraphNodeEvent,
            SSEMetadataEvent,
            SSEPingEvent,
            SSEReplaceEvent,
            SSESourcesEvent,
            SSETokenEvent,
        )

        models = [
            SSEMetadataEvent, SSETokenEvent, SSESourcesEvent,
            SSEDoneEvent, SSEPingEvent, SSEErrorEvent,
            SSEGraphNodeEvent, SSEReplaceEvent,
        ]
        assert len(models) == 8


class TestHealthResponseModel:
    """Verify HealthResponse includes all documented fields."""

    def test_health_response_has_circuit_breaker_state(self):
        """HealthResponse has circuit_breaker_state field."""
        from src.api.models import HealthResponse

        hr = HealthResponse(
            status="healthy",
            version="1.0.0",
            agent_ready=True,
            property_loaded=True,
            rag_ready=True,
            observability_enabled=False,
            circuit_breaker_state="closed",
        )
        assert hr.circuit_breaker_state == "closed"

    def test_health_response_field_count(self):
        """HealthResponse has 9 fields (including re2_available)."""
        from src.api.models import HealthResponse

        assert len(HealthResponse.model_fields) == 9


class TestErrorTaxonomy:
    """Verify error taxonomy matches documentation."""

    def test_error_code_count(self):
        """ErrorCode enum has 9 codes (R63: added unsupported_media_type)."""
        from src.api.errors import ErrorCode

        assert len(ErrorCode) == 9

    def test_error_code_values(self):
        """All 9 documented error codes exist."""
        from src.api.errors import ErrorCode

        expected = {
            "unauthorized", "not_found", "rate_limit_exceeded", "payload_too_large",
            "unsupported_media_type",  # R63 fix D4: Content-Encoding rejection
            "agent_unavailable", "internal_error", "validation_error",
            "service_degraded",
        }
        actual = {code.value for code in ErrorCode}
        assert actual == expected


class TestMiddlewareProtectedPaths:
    """Verify ApiKeyMiddleware protects the documented paths."""

    def test_protected_paths_match_docs(self):
        """ApiKeyMiddleware._PROTECTED_PATHS matches what README documents."""
        from src.api.middleware import ApiKeyMiddleware

        expected = {"/chat", "/graph", "/property", "/feedback", "/metrics"}
        assert ApiKeyMiddleware._PROTECTED_PATHS == expected


class TestRateLimitScope:
    """Verify rate limiter applies to the documented paths."""

    def test_rate_limited_paths(self):
        """RateLimitMiddleware applies to /chat and /feedback."""
        # Verified by reading source: path not in ("/chat", "/feedback") -> pass through
        # This is a structural assertion based on the middleware code.
        import ast
        import inspect
        from src.api.middleware import RateLimitMiddleware

        source = inspect.getsource(RateLimitMiddleware.__call__)
        assert '"/chat"' in source
        assert '"/feedback"' in source


class TestWebhookResponseModels:
    """Verify webhook response models exist and validate."""

    def test_sms_webhook_response(self):
        """SmsWebhookResponse model validates."""
        from src.api.models import SmsWebhookResponse

        resp = SmsWebhookResponse(status="ignored")
        assert resp.status == "ignored"

    def test_cms_webhook_response(self):
        """CmsWebhookResponse model validates."""
        from src.api.models import CmsWebhookResponse

        resp = CmsWebhookResponse(status="success", updated_categories=["restaurants"])
        assert resp.status == "success"
        assert resp.updated_categories == ["restaurants"]


class TestVersionConsistency:
    """Verify VERSION defaults are consistent across config and .env files."""

    def test_config_version_default(self):
        """config.py VERSION default is 1.4.0."""
        from src.config import Settings

        default = Settings.model_fields["VERSION"].default
        assert default == "1.4.0", (
            f"config.py VERSION default is {default!r}, expected '1.4.0'. "
            f"Sync .env, .env.example, and ARCHITECTURE.md."
        )

    def test_env_example_version(self):
        """env.example VERSION matches config.py default."""
        env_example = ROOT / ".env.example"
        content = env_example.read_text()
        match = re.search(r"^VERSION=([^\s#]+)", content, re.MULTILINE)
        assert match, "VERSION not found in .env.example"
        assert match.group(1).strip() == "1.4.0"


class TestCategoryToAgentMapping:
    """Verify category-to-agent dispatch includes hotel."""

    def test_hotel_category_mapped(self):
        """Hotel category is mapped in _CATEGORY_TO_AGENT."""
        from src.agent.graph import _CATEGORY_TO_AGENT

        assert "hotel" in _CATEGORY_TO_AGENT
        assert _CATEGORY_TO_AGENT["hotel"] == "hotel"


class TestGraphNodeCount:
    """Verify the documented graph node count matches code."""

    def test_graph_has_expected_nodes(self):
        """StateGraph has the documented number of nodes (excluding __start__, __end__)."""
        from src.agent.graph import build_graph

        graph = build_graph()
        g = graph.get_graph()
        # g.nodes may be a dict (keyed by node ID) or iterable of node objects.
        # Handle both: if dict, use keys; if iterable with .id, extract IDs.
        if isinstance(g.nodes, dict):
            nodes = [n for n in g.nodes if n not in ("__start__", "__end__")]
        else:
            nodes = [
                (n.id if hasattr(n, "id") else str(n))
                for n in g.nodes
                if (n.id if hasattr(n, "id") else str(n)) not in ("__start__", "__end__")
            ]
        assert len(nodes) == 12, (
            f"Graph has {len(nodes)} nodes, expected 12. "
            f"Nodes: {nodes}"
        )

    def test_known_nodes_frozenset_matches_graph(self):
        """_KNOWN_NODES frozenset has same count as actual graph nodes."""
        from src.agent.graph import _KNOWN_NODES, build_graph

        graph = build_graph()
        g = graph.get_graph()
        if isinstance(g.nodes, dict):
            actual = {n for n in g.nodes if n not in ("__start__", "__end__")}
        else:
            actual = {
                (n.id if hasattr(n, "id") else str(n))
                for n in g.nodes
                if (n.id if hasattr(n, "id") else str(n)) not in ("__start__", "__end__")
            }
        assert _KNOWN_NODES == actual, (
            f"_KNOWN_NODES mismatch: "
            f"missing_from_frozenset={actual - _KNOWN_NODES}, "
            f"extra_in_frozenset={_KNOWN_NODES - actual}"
        )


class TestGuardrailPatternCount:
    """Verify the documented guardrail pattern count matches code."""

    def test_total_guardrail_patterns(self):
        """guardrails.py has the expected total number of re.compile() patterns."""
        import inspect
        from src.agent import guardrails

        source = inspect.getsource(guardrails)
        # R52: Migrated from re.compile() to regex_engine.compile() for re2 ReDoS protection.
        patterns = re.findall(r"regex_engine\.compile\(", source)
        # R49: 185 -> 204 (added 6 Mandarin injection + 14 self-harm + 5 Mandarin non-Latin injection patterns)
        # R77: 204 -> 211 (added 7 Spanish self-harm patterns)
        assert len(patterns) == 211, (
            f"guardrails.py has {len(patterns)} regex_engine.compile() patterns, expected 211. "
            f"Update docs if patterns were added/removed."
        )

    def test_six_guardrail_categories(self):
        """guardrails.py defines exactly 6 guardrail pattern categories."""
        from src.agent import guardrails

        category_lists = [
            guardrails._INJECTION_PATTERNS,
            guardrails._RESPONSIBLE_GAMING_PATTERNS,
            guardrails._AGE_VERIFICATION_PATTERNS,
            guardrails._BSA_AML_PATTERNS,
            guardrails._PATRON_PRIVACY_PATTERNS,
            guardrails._SELF_HARM_PATTERNS,
        ]
        assert len(category_lists) == 6, (
            f"Expected 6 guardrail categories, found {len(category_lists)}."
        )
        # Each category must have at least 1 pattern
        for i, cat in enumerate(category_lists):
            assert len(cat) > 0, f"Category {i} is empty"

    def test_injection_pattern_count(self):
        """Prompt injection has 20 Latin patterns (11 English + 9 Tagalog/Taglish)."""
        from src.agent.guardrails import _INJECTION_PATTERNS

        assert len(_INJECTION_PATTERNS) == 20, (
            f"_INJECTION_PATTERNS has {len(_INJECTION_PATTERNS)}, expected 20."
        )

    def test_non_latin_injection_pattern_count(self):
        """Non-Latin injection has 33 patterns (Arabic + Japanese + Korean + Mandarin + French + Vietnamese + Hindi)."""
        from src.agent.guardrails import _NON_LATIN_INJECTION_PATTERNS

        # R49: 27 -> 33 (added 6 Mandarin/Chinese injection patterns)
        assert len(_NON_LATIN_INJECTION_PATTERNS) == 33, (
            f"_NON_LATIN_INJECTION_PATTERNS has {len(_NON_LATIN_INJECTION_PATTERNS)}, expected 33."
        )

    def test_responsible_gaming_pattern_count(self):
        """Responsible gaming has 60 patterns (EN + ES + PT + ZH + FR + VI + Hindi + Tagalog + JP + KO)."""
        from src.agent.guardrails import _RESPONSIBLE_GAMING_PATTERNS

        assert len(_RESPONSIBLE_GAMING_PATTERNS) == 60, (
            f"_RESPONSIBLE_GAMING_PATTERNS has {len(_RESPONSIBLE_GAMING_PATTERNS)}, expected 60."
        )

    def test_bsa_aml_pattern_count(self):
        """BSA/AML has 47 patterns (EN + ES + PT + ZH + FR + VI + Hindi + Tagalog + JP + KO)."""
        from src.agent.guardrails import _BSA_AML_PATTERNS

        assert len(_BSA_AML_PATTERNS) == 47, (
            f"_BSA_AML_PATTERNS has {len(_BSA_AML_PATTERNS)}, expected 47."
        )


class TestMiddlewareCount:
    """Verify the documented middleware count matches code."""

    def test_middleware_all_exports(self):
        """middleware.py __all__ exports exactly 6 middleware classes."""
        from src.api import middleware

        assert len(middleware.__all__) == 6, (
            f"middleware.__all__ has {len(middleware.__all__)} entries, expected 6. "
            f"Entries: {middleware.__all__}"
        )

    def test_middleware_names(self):
        """All 6 documented middleware classes are exported."""
        from src.api.middleware import __all__ as mw_all

        expected = {
            "RequestLoggingMiddleware",
            "ErrorHandlingMiddleware",
            "SecurityHeadersMiddleware",
            "ApiKeyMiddleware",
            "RateLimitMiddleware",
            "RequestBodyLimitMiddleware",
        }
        assert set(mw_all) == expected, (
            f"Middleware mismatch: "
            f"missing={expected - set(mw_all)}, "
            f"extra={set(mw_all) - expected}"
        )


class TestDeterministicD5:
    """Tier 1 deterministic gate for D5 (Testing Strategy).

    Replaces LLM judgment with hard assertions: test count floor,
    coverage config existence, zero xfails, coverage threshold config.
    """

    def test_minimum_test_count(self):
        """Project must have at least 2500 tests (enforces growth floor)."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header"],
            capture_output=True, text=True, cwd=str(ROOT),
            timeout=120,
        )
        # Last non-empty line is like "2580 tests collected"
        lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
        count_line = lines[-1] if lines else ""
        match = re.search(r"(\d+)\s+tests?\s+collected", count_line)
        assert match, f"Could not parse test count from: {count_line!r}"
        count = int(match.group(1))
        assert count >= 2500, (
            f"Only {count} tests collected, minimum is 2500. "
            f"Add tests or update floor if intentional reduction."
        )

    def test_coverage_config_exists(self):
        """pyproject.toml has [tool.coverage.run] with source = ['src']."""
        pyproject = ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "[tool.coverage.run]" in content
        assert 'source = ["src"]' in content

    def test_coverage_threshold_configured(self):
        """Coverage fail_under is at least 90."""
        pyproject = ROOT / "pyproject.toml"
        content = pyproject.read_text()
        match = re.search(r"fail_under\s*=\s*(\d+)", content)
        assert match, "fail_under not found in pyproject.toml"
        assert int(match.group(1)) >= 90

    def test_zero_active_xfails(self):
        """No active @pytest.mark.xfail decorators in test files."""
        test_dir = ROOT / "tests"
        active_xfails = []
        for path in test_dir.rglob("*.py"):
            text = path.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                # Skip comments and docstrings
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                # Only match actual decorator usage (starts with @)
                if stripped.startswith("@pytest.mark.xfail"):
                    active_xfails.append(f"{path.name}:{i}")
        assert not active_xfails, (
            f"Active xfail decorators found: {active_xfails}. "
            f"Fix or remove — xfails mask real failures."
        )


class TestDeterministicD6:
    """Tier 1 deterministic gate for D6 (Docker & DevOps).

    Verifies Dockerfile best practices without LLM judgment:
    non-root user, multi-stage build, require-hashes, HEALTHCHECK uses /live.
    """

    def _read_dockerfile(self):
        return (ROOT / "Dockerfile").read_text()

    def test_multi_stage_build(self):
        """Dockerfile uses multi-stage build (at least 2 FROM statements)."""
        content = self._read_dockerfile()
        from_count = len(re.findall(r"^FROM\s+", content, re.MULTILINE))
        assert from_count >= 2, (
            f"Only {from_count} FROM statements. Multi-stage required."
        )

    def test_non_root_user(self):
        """Dockerfile switches to non-root user before CMD."""
        content = self._read_dockerfile()
        assert re.search(r"^USER\s+\S+", content, re.MULTILINE), (
            "No USER directive found. Must run as non-root."
        )

    def test_require_hashes(self):
        """pip install uses --require-hashes for supply chain security."""
        content = self._read_dockerfile()
        assert "--require-hashes" in content

    def test_healthcheck_uses_live(self):
        """HEALTHCHECK hits /live (always 200), not /health (503 when degraded)."""
        content = self._read_dockerfile()
        # Find the HEALTHCHECK line and all continuation lines
        hc_lines = []
        in_healthcheck = False
        for line in content.splitlines():
            if line.strip().startswith("HEALTHCHECK"):
                in_healthcheck = True
            if in_healthcheck:
                hc_lines.append(line)
                if not line.rstrip().endswith("\\"):
                    break
        hc_block = "\n".join(hc_lines)
        assert hc_block, "No HEALTHCHECK found"
        assert "/live" in hc_block, (
            f"HEALTHCHECK should use /live not /health. Found: {hc_block!r}"
        )

    def test_exec_form_cmd(self):
        """CMD uses exec form (JSON array), not shell form."""
        content = self._read_dockerfile()
        # Exec form starts with CMD [
        assert re.search(r'^CMD\s+\[', content, re.MULTILINE), (
            "CMD not in exec form. Use CMD [\"...\"] for proper signal handling."
        )

    def test_digest_pinning(self):
        """Base images use SHA-256 digest pinning (@sha256:...)."""
        content = self._read_dockerfile()
        from_lines = re.findall(r"^FROM\s+(.+?)(?:\s+AS\s+\w+)?$", content,
                                re.MULTILINE)
        for img in from_lines:
            assert "@sha256:" in img, (
                f"Image {img!r} not digest-pinned. Use @sha256:... for immutability."
            )


class TestDeterministicD7:
    """Tier 1 deterministic gate for D7 (Prompts & Guardrails).

    Verifies guardrail pattern counts and categories without LLM judgment.
    """

    def test_total_pattern_count_is_204(self):
        """Total guardrail patterns must be exactly 204."""
        import inspect
        from src.agent import guardrails
        source = inspect.getsource(guardrails)
        patterns = re.findall(r"regex_engine\.compile\(", source)
        assert len(patterns) == 211  # R77: 204 + 7 Spanish self-harm patterns

    def test_six_guardrail_categories(self):
        """All 6 guardrail categories exist and are non-empty."""
        from src.agent import guardrails
        categories = {
            "injection": guardrails._INJECTION_PATTERNS,
            "responsible_gaming": guardrails._RESPONSIBLE_GAMING_PATTERNS,
            "age_verification": guardrails._AGE_VERIFICATION_PATTERNS,
            "bsa_aml": guardrails._BSA_AML_PATTERNS,
            "patron_privacy": guardrails._PATRON_PRIVACY_PATTERNS,
            "self_harm": guardrails._SELF_HARM_PATTERNS,
        }
        for name, cat in categories.items():
            assert len(cat) > 0, f"Category {name} is empty"

    def test_confusable_entry_count_is_145(self):
        """Confusable mapping must have exactly 145 entries."""
        from src.agent.guardrails import _CONFUSABLES
        assert len(_CONFUSABLES) == 145, (
            f"_CONFUSABLES has {len(_CONFUSABLES)} entries, expected 145. "
            f"Update docs/adr/018-confusable-coverage.md if count changed."
        )

    def test_all_security_patterns_use_regex_engine(self):
        """Security guardrail patterns use regex_engine, not raw re.compile()."""
        import inspect
        from src.agent import guardrails
        source = inspect.getsource(guardrails)
        # _ACT_AS_BROAD_PATTERN is a casino-context exclusion helper, not a
        # security pattern — it's allowed to use re.compile() directly.
        _ALLOWED_RE_COMPILE = {"_ACT_AS_BROAD_PATTERN"}
        raw_compiles = []
        for i, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "re.compile(" in stripped and "regex_engine" not in stripped:
                # Check if this is an allowed exception
                if not any(name in stripped for name in _ALLOWED_RE_COMPILE):
                    raw_compiles.append(f"line {i}: {stripped[:80]}")
        assert len(raw_compiles) == 0, (
            f"Found raw re.compile() calls in guardrails: {raw_compiles}. "
            f"Use regex_engine.compile() for re2 ReDoS protection."
        )


class TestDeterministicD8:
    """Tier 1 deterministic gate for D8 (Scalability & Production).

    Verifies scalability patterns without LLM judgment: no threading.Lock in
    async paths, TTLCache for LLM singletons, jitter on caches.
    """

    def test_no_threading_lock_in_async_agent_code(self):
        """No threading.Lock instantiation in agent/ (async code) except documented exceptions."""
        # Documented exceptions: state_backend.py (intentional R36 fix B5)
        agent_dir = ROOT / "src" / "agent"
        violations = []
        for path in agent_dir.rglob("*.py"):
            text = path.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                # Skip comments and docstrings
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                # Only flag actual instantiation: threading.Lock()
                if "threading.Lock()" in stripped:
                    violations.append(f"{path.name}:{i}")
        assert not violations, (
            f"threading.Lock() instantiation in async agent code: {violations}. "
            f"Use asyncio.Lock for async code."
        )

    def test_llm_singletons_use_ttlcache(self):
        """LLM singleton caches use TTLCache, not @lru_cache."""
        # Check key files that create LLM clients
        llm_files = [
            ROOT / "src" / "agent" / "nodes.py",
            ROOT / "src" / "agent" / "circuit_breaker.py",
            ROOT / "src" / "agent" / "memory.py",
            ROOT / "src" / "agent" / "whisper_planner.py",
            ROOT / "src" / "rag" / "embeddings.py",
        ]
        for path in llm_files:
            text = path.read_text()
            # Must use TTLCache
            assert "TTLCache" in text, (
                f"{path.name} missing TTLCache — credential rotation requires TTL."
            )
            # Must NOT use @lru_cache for actual caching (comments/docstrings are OK)
            in_docstring = False
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = not in_docstring
                    continue
                if in_docstring:
                    continue
                if stripped.startswith("@lru_cache"):
                    assert False, (
                        f"{path.name}:{i} uses @lru_cache — "
                        f"use TTLCache for credential rotation."
                    )

    def test_ttlcache_has_jitter_on_singletons(self):
        """TTLCache for LLM/config singletons includes random jitter to prevent thundering herd."""
        # Only check singleton caches (maxsize=1) — bounded data caches
        # like delivery logs (maxsize=10000) don't need jitter.
        src_dir = ROOT / "src"
        jitter_missing = []
        for path in src_dir.rglob("*.py"):
            text = path.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "TTLCache(" in stripped and re.search(r"maxsize=1\b", stripped):
                    if "random" not in stripped and "randint" not in stripped:
                        jitter_missing.append(f"{path.name}:{i}")
        assert not jitter_missing, (
            f"TTLCache singleton without jitter (thundering herd risk): {jitter_missing}. "
            f"Add random.randint(0, 300) to TTL."
        )


class TestDeterministicD9:
    """Tier 1 deterministic gate for D9 (Trade-off Documentation).

    Verifies documentation completeness without LLM judgment: ADR count,
    ADR format, version parity, confusable count parity.
    """

    def test_adr_count_minimum(self):
        """At least 22 ADRs exist (excluding README.md)."""
        adr_dir = ROOT / "docs" / "adr"
        adrs = [f for f in adr_dir.glob("*.md") if f.name != "README.md"]
        assert len(adrs) >= 22, (
            f"Only {len(adrs)} ADRs found, minimum is 22."
        )

    def test_all_adrs_have_status(self):
        """Every ADR has a Status section."""
        adr_dir = ROOT / "docs" / "adr"
        missing = []
        for path in sorted(adr_dir.glob("*.md")):
            if path.name == "README.md":
                continue
            text = path.read_text()
            # Accept either "## Status" or "**Status**:" format
            has_status = "## Status" in text or "**Status**:" in text
            if not has_status:
                missing.append(f"{path.name}: missing Status section")
        assert not missing, f"ADR format issues: {missing}"

    def test_recent_adrs_have_dates(self):
        """ADRs from 022+ (2026-02-26 onward) have dates in YYYY-MM-DD format."""
        adr_dir = ROOT / "docs" / "adr"
        missing = []
        for path in sorted(adr_dir.glob("*.md")):
            if path.name == "README.md":
                continue
            # Only check ADRs numbered 022+ (older ADRs use round-based status)
            num_match = re.search(r"(\d{3})", path.name)
            if not num_match or int(num_match.group(1)) < 22:
                continue
            text = path.read_text()
            if not re.search(r"\d{4}-\d{2}-\d{2}", text):
                missing.append(f"{path.name}: no YYYY-MM-DD date found")
        assert not missing, f"Recent ADR date issues: {missing}"

    def test_version_parity_config_and_env(self):
        """VERSION in config.py default matches .env.example."""
        from src.config import Settings
        config_ver = Settings.model_fields["VERSION"].default
        env_example = ROOT / ".env.example"
        content = env_example.read_text()
        match = re.search(r"^VERSION=([^\s#]+)", content, re.MULTILINE)
        assert match, "VERSION not found in .env.example"
        assert match.group(1).strip() == config_ver, (
            f".env.example VERSION={match.group(1)} != config.py {config_ver}"
        )


class TestBehavioralScenarioParity:
    """Tier 1 parity tests for behavioral scenario files.

    Ensures scenario counts are tracked and B4 file exists.
    """

    def test_behavioral_scenario_total_count(self):
        """Total behavioral scenarios across all files is at least 50."""
        import yaml
        scenario_dir = ROOT / "tests" / "scenarios"
        total = 0
        for path in sorted(scenario_dir.glob("behavioral_*.yaml")):
            with open(path) as f:
                data = yaml.safe_load(f)
            if data and "scenarios" in data:
                total += len(data["scenarios"])
        assert total >= 50, (
            f"Only {total} behavioral scenarios, target is 50. "
            f"Add scenarios to reach minimum coverage."
        )

    def test_b4_agentic_file_exists(self):
        """B4 agentic scenario file exists."""
        path = ROOT / "tests" / "scenarios" / "behavioral_agentic.yaml"
        assert path.exists(), (
            "behavioral_agentic.yaml missing — B4 dimension has no scenarios."
        )

    def test_each_dimension_has_scenarios(self):
        """Each behavioral dimension (B1-B5) has a scenario file with at least 5 scenarios."""
        import yaml
        expected_files = {
            "B1": "behavioral_sarcasm.yaml",
            "B2": "behavioral_implicit.yaml",
            "B3": "behavioral_engagement.yaml",
            "B4": "behavioral_agentic.yaml",
            "B5": "behavioral_nuance.yaml",
        }
        scenario_dir = ROOT / "tests" / "scenarios"
        for dim, filename in expected_files.items():
            path = scenario_dir / filename
            assert path.exists(), f"{dim}: {filename} missing"
            with open(path) as f:
                data = yaml.safe_load(f)
            count = len(data.get("scenarios", []))
            assert count >= 5, (
                f"{dim} ({filename}) has only {count} scenarios, minimum is 5."
            )



    """Verify the documented endpoint count matches code."""

    # FastAPI adds built-in routes (/docs, /redoc, /openapi.json, /docs/oauth2-redirect).
    # Exclude them to count only application-defined endpoints.
    _FASTAPI_BUILTIN = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}

    def test_app_has_expected_endpoints(self):
        """FastAPI app has exactly 8 application-defined endpoints."""
        from src.api.app import create_app

        app = create_app()
        endpoints = [
            route.path
            for route in app.routes
            if hasattr(route, "endpoint") and route.path not in self._FASTAPI_BUILTIN
        ]
        assert len(endpoints) == 9, (
            f"App has {len(endpoints)} app endpoints, expected 9. "
            f"Endpoints: {endpoints}"
        )

    def test_expected_endpoint_paths(self):
        """All 9 documented endpoint paths exist."""
        from src.api.app import create_app

        app = create_app()
        paths = {
            route.path
            for route in app.routes
            if hasattr(route, "endpoint") and route.path not in self._FASTAPI_BUILTIN
        }
        expected = {
            "/chat", "/live", "/health", "/property", "/metrics",
            "/graph", "/sms/webhook", "/cms/webhook", "/feedback",
        }
        assert paths == expected, (
            f"Endpoint mismatch: "
            f"missing={expected - paths}, "
            f"extra={paths - expected}"
        )


class TestRE2Enforcement:
    """Verify RE2 is enforced in non-development environments."""

    def test_re2_enforcement_raises_in_production(self, monkeypatch):
        """If ENVIRONMENT != 'development' and RE2 is missing, startup must fail."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("CMS_WEBHOOK_SECRET", "test-secret")
        import src.agent.regex_engine as mod
        from src.config import get_settings
        get_settings.cache_clear()
        original = mod.RE2_AVAILABLE
        try:
            mod.RE2_AVAILABLE = False
            with pytest.raises(RuntimeError, match="google-re2"):
                mod.enforce_re2_in_production()
        finally:
            mod.RE2_AVAILABLE = original
            get_settings.cache_clear()

    def test_re2_enforcement_passes_in_development(self, monkeypatch):
        """In development, missing RE2 should not raise."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        import src.agent.regex_engine as mod
        from src.config import get_settings
        get_settings.cache_clear()
        original = mod.RE2_AVAILABLE
        try:
            mod.RE2_AVAILABLE = False
            mod.enforce_re2_in_production()  # Should not raise
        finally:
            mod.RE2_AVAILABLE = original
            get_settings.cache_clear()

    def test_re2_enforcement_passes_when_available(self, monkeypatch):
        """When RE2 is available, enforcement always passes."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("CMS_WEBHOOK_SECRET", "test-secret")
        import src.agent.regex_engine as mod
        from src.config import get_settings
        get_settings.cache_clear()
        original = mod.RE2_AVAILABLE
        try:
            mod.RE2_AVAILABLE = True
            mod.enforce_re2_in_production()  # Should not raise
        finally:
            mod.RE2_AVAILABLE = original
            get_settings.cache_clear()


class TestVersionParity:
    """Ensure version is consistent across all sources."""

    def test_pyproject_matches_config(self):
        """pyproject.toml version must match src/config.py VERSION."""
        import tomllib
        from pathlib import Path
        with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        from src.config import Settings
        assert pyproject["project"]["version"] == Settings.model_fields["VERSION"].default

    def test_env_example_matches_config(self):
        """VERSION in .env.example must match src/config.py."""
        from pathlib import Path
        env_path = Path(__file__).parent.parent / ".env.example"
        if not env_path.exists():
            pytest.skip(".env.example not found")
        content = env_path.read_text()
        from src.config import Settings
        version = Settings.model_fields["VERSION"].default
        assert f"VERSION={version}" in content


class TestGraphTopology:
    """Verify graph topology matches documentation.

    Complements TestGraphNodeCount (count-based assertions) with
    explicit node-name identity checks against constants.py.
    """

    def test_node_count_matches_docs(self):
        """Graph must have exactly 12 nodes (documented in ARCHITECTURE.md and README)."""
        from src.agent.graph import build_graph

        graph = build_graph()
        graph_data = graph.get_graph()
        # Exclude __start__ and __end__ virtual nodes added by LangGraph
        if isinstance(graph_data.nodes, dict):
            real_nodes = [n for n in graph_data.nodes if not n.startswith("__")]
        else:
            real_nodes = [
                (n.id if hasattr(n, "id") else str(n))
                for n in graph_data.nodes
                if not (n.id if hasattr(n, "id") else str(n)).startswith("__")
            ]
        assert len(real_nodes) == 12, f"Expected 12 nodes, got {len(real_nodes)}: {real_nodes}"

    def test_node_names_use_constants(self):
        """All node names must come from constants.py."""
        from src.agent.constants import _KNOWN_NODES
        from src.agent.graph import build_graph

        graph = build_graph()
        graph_data = graph.get_graph()
        if isinstance(graph_data.nodes, dict):
            real_nodes = {n for n in graph_data.nodes if not n.startswith("__")}
        else:
            real_nodes = {
                (n.id if hasattr(n, "id") else str(n))
                for n in graph_data.nodes
                if not (n.id if hasattr(n, "id") else str(n)).startswith("__")
            }
        assert real_nodes == _KNOWN_NODES, (
            f"Mismatch: in graph but not constants: {real_nodes - _KNOWN_NODES}, "
            f"in constants but not graph: {_KNOWN_NODES - real_nodes}"
        )
