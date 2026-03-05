# Component 12: Config + Feature Flags + Casino Config — Architecture Audit

**Auditor**: auditor-api
**Date**: 2026-03-05
**Scope**: Centralized settings, per-casino configuration, feature flags, circuit breaker, checkpointer

---

## 1. Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/config.py` | 313 | Pydantic-settings `Settings` class (~60 settings). Model validators: embedding normalization, RAG config, production secrets, property state, consent HMAC. TTLCache with jitter for `get_settings()`. |
| `src/casino/config.py` | 904 | Per-casino configuration: BrandingConfig, RegulationConfig, OperationalConfig, RagConfig, PromptsConfig, CasinoConfig TypedDicts. DEFAULT_CONFIG + 5 CASINO_PROFILES (Mohegan Sun, Foxwoods, Parx, Wynn, Hard Rock AC). Firestore hot-reload with TTLCache. Deep merge. Profile completeness validation at import time. |
| `src/casino/feature_flags.py` | 181 | 17 feature flags in MappingProxyType. FeatureFlags TypedDict with import-time parity check. Cross-module parity with DEFAULT_CONFIG["features"]. `get_feature_flags()` and `is_feature_enabled()` async with TTLCache + asyncio.Lock (thundering herd protection). |
| `src/agent/circuit_breaker.py` | 604 | CircuitBreaker: closed->open->half_open states, rolling window failures, asyncio.Lock. Redis L2 sync with pipeline reads/writes. `_read_backend_state()` (I/O outside lock) + `_apply_backend_state()` (mutation inside lock). Bidirectional sync. TTL-cached singleton. |
| `src/agent/memory.py` | 199 | BoundedMemorySaver wrapping MemorySaver with LRU eviction (MAX_ACTIVE_THREADS=1000). `get_checkpointer()` returns FirestoreSaver (prod) or BoundedMemorySaver (dev). |

**Total**: 5 files, 2,201 lines

---

## 2. Wiring Verification

All modules are **heavily wired** — these are foundational infrastructure used by virtually every other component:

| Module | Imported By | Import Count |
|--------|-------------|-------------|
| `src/config.py` | 20+ files across agent, rag, casino, observability, api | 20+ |
| `src/casino/config.py` | agent/_base.py, persona.py, prompts.py, feature_flags.py | 5+ |
| `src/casino/feature_flags.py` | agent/dispatch.py, nodes.py, pre_extract.py, graph.py, whisper_planner.py | 5+ |
| `src/agent/circuit_breaker.py` | agent/_base.py (all specialists use it) | 1 (but called per-specialist) |
| `src/agent/memory.py` | agent/graph.py (checkpointer for StateGraph) | 1 (critical path) |

**Grep proof**:
```
# config.py — ubiquitous
src/agent/nodes.py:from src.config import get_settings
src/agent/graph.py:from src.config import get_settings
src/rag/pipeline.py:from src.config import get_settings
src/api/app.py:from src.config import get_settings
src/casino/config.py:from src.config import get_settings
(... 15+ more files)

# casino/config.py
src/agent/agents/_base.py:from src.casino.config import get_casino_config, get_casino_profile
src/agent/persona.py:from src.casino.config import get_casino_profile
src/agent/prompts.py:from src.casino.config import get_casino_config

# feature_flags.py
src/agent/dispatch.py:from src.casino.feature_flags import is_feature_enabled
src/agent/nodes.py:from src.casino.feature_flags import is_feature_enabled
src/agent/graph.py:from src.casino.feature_flags import DEFAULT_FEATURES

# circuit_breaker.py
src/agent/agents/_base.py:from src.agent.circuit_breaker import _get_circuit_breaker

# memory.py
src/agent/graph.py:from src.agent.memory import get_checkpointer
```

**Verdict**: All 5 files are REAL production infrastructure. `config.py` is the single most-imported module in the codebase.

---

## 3. Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `tests/test_casino_config.py` | 42 | Casino profiles, deep merge, Firestore hot-reload, default config, profile completeness, branding, regulations |
| `tests/test_deployment.py` | ~20 (config subset) | Settings validation, production secret checks, embedding normalization, RAG config, property state validation |
| `tests/test_r5_scalability.py` | 31 | Circuit breaker states, rolling window, half-open probes, Redis sync, TTLCache jitter, singleton caching |
| `tests/test_r46_scalability.py` | ~15 (CB subset) | Redis pipeline operations, bidirectional sync, concurrent access |
| `tests/conftest.py` | N/A | 17 singleton cache clears (including config, feature flags, circuit breaker) — ensures test isolation |

**Total**: ~108 tests covering config + flags + circuit breaker + memory

---

## 4. Live vs Mock Assessment

| Test File | Mock Count | Live Calls | Assessment |
|-----------|-----------|------------|------------|
| `test_casino_config.py` | 91 | 0 | **Firestore mocked** — appropriate. Tests validate deep merge, profile completeness, parity checks. |
| `test_deployment.py` | ~5 | 0 | **Env var manipulation** — tests use monkeypatch for settings, no external deps. |
| `test_r5_scalability.py` | ~10 | 0 | **Redis mocked** — appropriate for unit testing CB state machine. |
| `test_r46_scalability.py` | ~8 | 0 | **Redis mocked** — tests validate Lua scripts and pipeline operations. |

**Summary**: Mocking is appropriate throughout. Config, feature flags, and circuit breaker are primarily deterministic logic with optional external backends (Firestore, Redis). The core state machines and merge logic are tested without mocks. External service integration (Firestore hot-reload, Redis CB sync) is mocked because CI doesn't have those services.

---

## 5. Known Gaps

### GAP-1: No Live Redis Circuit Breaker Test (MEDIUM)
Circuit breaker Redis L2 sync is tested with mocked Redis. No integration test with a real Redis instance validates that the Lua scripts, pipeline operations, and bidirectional sync actually work.

**Impact**: A Redis API change or Lua script error would not be caught until production.
**Mitigation**: Add a `@pytest.mark.live` test with a real Redis instance. The code is well-structured for this.

### GAP-2: No Live Firestore Config Test (MEDIUM)
Casino config Firestore hot-reload is tested with mocked Firestore client. No test validates actual Firestore read/write/merge behavior.

**Impact**: Firestore schema changes or permission issues not caught until deployment.
**Mitigation**: Same as GAP-1 — add `@pytest.mark.live` test.

### GAP-3: Feature Flag Lifecycle Not Tested (LOW)
Tests verify flag values and parity checks, but don't test the full lifecycle: flag change in Firestore -> cache expiry -> new value picked up by graph node.

**Impact**: Cache-related bugs in flag propagation wouldn't be caught.
**Mitigation**: An integration test that mutates flags and verifies graph behavior changes would close this gap.

### GAP-4: BoundedMemorySaver LRU Eviction Under Concurrent Load (LOW)
`BoundedMemorySaver` uses a dict with manual LRU eviction. Under high concurrency, the dict operations are not atomic (though they're protected by the GIL in CPython). No concurrent stress test exists.

**Impact**: Theoretical — CPython GIL protects dict operations. Would only matter under PyPy or if async locking assumptions change.

### GAP-5: CASINO_PROFILES Returns Mutable Data (PREVIOUSLY FIXED)
`get_casino_profile()` was previously identified as returning mutable module-level data (R35-R36). The fix uses `copy.deepcopy()`. Verified in current code — this gap is CLOSED.

---

## 6. Confidence: 90%

**Strengths**:
- Pydantic-settings with model validators for production safety (hard-fail for empty secrets)
- Import-time parity checks between FeatureFlags TypedDict, DEFAULT_FEATURES, and DEFAULT_CONFIG (catches drift at startup)
- MappingProxyType for immutable defaults (prevents cross-request mutation)
- TTLCache with jitter on ALL singletons (prevents thundering herd)
- Double-check locking pattern on all caches (thread-safe for config, asyncio-safe for flags)
- Circuit breaker with I/O outside lock / mutation inside lock (correct TOCTOU pattern)
- Bidirectional Redis sync (open->closed recovery propagation)
- `get_settings.cache_clear` backward compatibility shim for tests
- 5 casino profiles with complete regulatory, branding, and operational configs
- Profile completeness validation at import time

**Weaknesses**:
- No live integration tests for Redis or Firestore backends
- Feature flag lifecycle not end-to-end tested

---

## 7. Verdict: PRODUCTION-READY

This is the strongest component in the system. The config layer is meticulously designed with multiple defense layers: import-time parity checks, immutable defaults, production secret validation, TTL jitter, and thread-safe caching. The circuit breaker implementation follows the correct pattern for async + distributed state (I/O outside lock, Lua scripts for atomicity, bidirectional sync). The only gaps are live integration tests for external backends, which is a testing completeness issue rather than a code quality issue. High confidence that this code works correctly in production.
