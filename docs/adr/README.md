# Architecture Decision Records (ADRs)

Architectural decisions documented inline in source code, extracted here for discoverability.

## Status Lifecycle

| Status | Meaning |
|--------|---------|
| **Proposed** | Under discussion, not yet implemented |
| **Accepted** | Decision made, implementation in place |
| **Accepted (with caveats)** | Decision made with known limitations documented |
| **Deferred** | Acknowledged but implementation postponed |
| **Superseded** | Replaced by a newer ADR (link to successor) |
| **Deprecated** | No longer relevant; kept for historical context |

## Index

| # | Decision | Status | Source | Last Reviewed |
|---|----------|--------|--------|---------------|
| ADR-0001 | [_dispatch_to_specialist SRP Refactor](ADR-0001-dispatch-srp-refactor.md) | Accepted | `graph.py` | 2026-02-25 |
| 001 | [Custom StateGraph over create_react_agent](001-custom-stategraph.md) | Accepted | `graph.py:1-31` | 2026-02-25 |
| 002 | [In-memory rate limiting for MVP](002-rate-limiting.md) | Accepted | `middleware.py:327-366` | 2026-02-25 |
| 003 | [Feature flag dual-layer design](003-feature-flags.md) | Accepted | `graph.py:584-618` | 2026-02-25 |
| 004 | [Degraded-pass validation strategy](004-degraded-pass.md) | Accepted | `nodes.py` | 2026-02-25 |
| 005 | [English-only LLM responses](005-i18n-responses.md) | Deferred | `nodes.py:55-71` | 2026-02-25 |
| 006 | [ChromaDB dev / Vertex AI prod](006-vector-db-split.md) | Accepted | `config.py:79` | 2026-02-25 |
| 007 | [Classifier restricted mode](007-classifier-degradation.md) | Accepted | `guardrails.py:608+` | 2026-02-25 |
| 008 | [threading.Lock in InMemoryBackend](008-threading-lock.md) | Accepted | `state_backend.py:94-99` | 2026-02-25 |
| 009 | [UNSET_SENTINEL (UUID-namespaced String)](009-unset-sentinel.md) | Accepted | `state.py:28` | 2026-02-25 |
| 010 | [Middleware execution order](010-middleware-order.md) | Accepted | `app.py:173-186` | 2026-02-25 |
| 011 | [RRF Fusion Constant (k=60)](011-rrf-fusion-constant.md) | Accepted | `rag/reranking.py` | 2026-02-25 |
| 012 | [Retrieval Timeout (10s)](012-retrieval-timeout.md) | Accepted | `config.py:62` | 2026-02-25 |
| 013 | [SSE Timeout (60s)](013-sse-timeout.md) | Accepted | `config.py:51` | 2026-02-25 |
| 014 | [Message Limits (40/20)](014-message-limits.md) | Accepted | `config.py:55-56` | 2026-02-25 |
| 015 | [Circuit Breaker Parameters](015-circuit-breaker-parameters.md) | Accepted | `config.py:59-61` | 2026-02-25 |
| 016 | [asyncio.to_thread for Sync Retrievers](016-asyncio-to-thread-retrievers.md) | Superseded by ADR-020 | `rag/pipeline.py` | 2026-02-25 |
| 017 | [Self-Exclusion via Responsible Gaming (MVP)](017-self-exclusion-escalation.md) | Accepted | `guardrails.py`, `compliance_gate.py` | 2026-02-25 |
| 018 | [Confusable Homoglyph Coverage (~110 entries)](018-confusable-coverage.md) | Accepted (bounded scope) | `guardrails.py` | 2026-02-25 |
| 019 | [Single Tenant Per Deployment (MVP)](019-single-tenant-deployment.md) | Accepted (MVP) | `config.py`, `app.py` | 2026-02-25 |
| 020 | [Concurrent Retrieval via ThreadPoolExecutor](020-concurrent-retrieval.md) | Accepted | `agent/tools.py` | 2026-02-25 |
| 021 | [Supply Chain Security Strategy](021-supply-chain-security.md) | Accepted | `Dockerfile`, `scripts/security-audit.sh` | 2026-02-26 |
