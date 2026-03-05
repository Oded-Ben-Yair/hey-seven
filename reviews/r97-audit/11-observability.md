# Component 11: Observability — Architecture Audit

**Auditor**: auditor-api
**Date**: 2026-03-05
**Scope**: LangFuse tracing, distributed trace context, evaluation framework

---

## 1. Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/observability/langfuse_client.py` | 163 | LangFuse client with TTLCache (1hr + jitter). Sampling: 10% prod, 100% dev. Returns LangChain CallbackHandler. |
| `src/observability/traces.py` | 148 | Lightweight trace context: `NodeSpan` and `TraceContext` dataclasses. `create_trace_context()`, `record_node_span()`, `end_trace()`. |
| `src/observability/evaluation.py` | 459 | Evaluation framework: `GoldenTestCase` (20 cases), `EvalScore`. Deterministic scoring: groundedness, helpfulness, safety, persona. `run_evaluation()` aggregator. |

**Total**: 3 files, 770 lines

---

## 2. Wiring Verification

| Module | Imported By | Status |
|--------|-------------|--------|
| `langfuse_client.py` | `src/api/app.py` (health check), `src/agent/graph.py` (tracing callbacks) | REAL |
| `traces.py` | `src/agent/graph.py` (trace context per invocation) | REAL |
| `evaluation.py` | `tests/` (test evaluation framework), standalone eval scripts | REAL |

**Grep proof**:
```
src/api/app.py:from src.observability.langfuse_client import is_observability_enabled
src/agent/graph.py:from src.observability.langfuse_client import get_langfuse_handler, should_sample
src/agent/graph.py:from src.observability.traces import create_trace_context, record_node_span, end_trace
```

**Verdict**: All 3 files are actively wired. `evaluation.py` is used by the evaluation framework (outside the main request path but production-relevant for scoring).

---

## 3. Test Coverage

| Test File | Test Count | What It Tests |
|-----------|-----------|---------------|
| `tests/test_deployment.py` | ~10 (observability subset) | LangFuse configuration, sampling rates, cache TTL, handler creation |
| `tests/test_phase2_integration.py` | ~5 (observability subset) | Trace context creation, span recording |

**Total**: ~15 tests directly covering observability

Additional coverage:
- `evaluation.py` is exercised by evaluation scripts in `tests/evaluation/` (judge panels, scenario runners)
- LangFuse handler is passed through graph invocations in integration tests (indirect coverage)

---

## 4. Live vs Mock Assessment

| Test Area | Mock Status | Assessment |
|-----------|-------------|------------|
| LangFuse client | Mocked (no LANGFUSE_PUBLIC_KEY in CI) | **Appropriate** — LangFuse is external SaaS. Mock validates integration logic. |
| Trace context | No mocks needed | **Pure dataclass logic** — fully deterministic. |
| Evaluation scoring | No mocks needed | **Fully deterministic** — keyword/regex scoring, no LLM calls. |

**Summary**: Observability tests are appropriately structured. LangFuse integration can't be tested live without credentials. Trace context and evaluation scoring are deterministic and don't need mocks.

---

## 5. Known Gaps

### GAP-1: Traces Are Local-Only (MEDIUM)
`TraceContext` and `NodeSpan` are in-memory dataclasses. They're created per-request and passed through the graph, but the spans are not exported to any external system (no OpenTelemetry, no Cloud Trace, no Jaeger).

**Impact**: In production, you can see LangFuse traces (LLM calls) but NOT internal graph execution traces (which node took how long, what the routing decision was). Debugging latency requires correlating logs manually.
**Mitigation**: The trace infrastructure is wired and collecting data. Exporting to Cloud Trace or LangFuse custom events would be straightforward.

### GAP-2: No Alerting on LangFuse Errors (LOW)
`get_langfuse_handler()` returns `None` if LangFuse is not configured or if credentials fail. The caller gracefully skips tracing. But there's no alert when tracing silently stops working.

**Impact**: Tracing could fail in production and nobody would notice until they check the LangFuse dashboard.
**Mitigation**: Add a metric or health check for LangFuse connectivity.

### GAP-3: Evaluation Framework is Deterministic Only (BY DESIGN)
`evaluation.py` scores are based on keyword/regex matching, not LLM-as-judge. This is intentional — the live behavioral evaluation uses a separate system (`tests/evaluation/v2/`).

**Impact**: None. The deterministic scoring is for automated regression testing. Live behavioral quality is measured by the GPT-5.2 judge panel.

### GAP-4: 10% Production Sampling May Miss Issues (LOW)
LangFuse samples 10% of production requests. Edge case behaviors that occur in <10% of requests may not be captured.

**Impact**: Rare but important scenarios (crisis, BSA/AML triggers) may not get traced.
**Mitigation**: Could add "always trace" for specific query types (crisis, responsible_gaming) regardless of sampling rate.

---

## 6. Confidence: 72%

**Strengths**:
- LangFuse integration with TTLCache + jitter (credential rotation safe)
- Sampling rate differentiation (100% dev, 10% prod)
- Lightweight trace context that doesn't add latency
- Deterministic evaluation framework for regression testing (no LLM dependency in CI)
- `is_observability_enabled()` check on health endpoint

**Weaknesses**:
- Trace spans not exported to external APM (local-only)
- No alerting on tracing failure
- No request-ID propagation from API middleware to trace context
- Relatively thin test coverage compared to other components

---

## 7. Verdict: NEEDS-FIXES

The observability stack is functional for development but has gaps for production:
1. Trace spans need export to Cloud Trace or equivalent APM for production debugging
2. LangFuse failure detection/alerting is missing
3. Request-ID correlation between API middleware and trace context is not implemented

The code quality is high and the architecture is sound — the gaps are about completeness of the observability pipeline, not code defects. Upgrading to production-ready requires: (a) span export to Cloud Trace, (b) LangFuse health alerting, (c) request-ID propagation. Estimated effort: 1-2 days.
