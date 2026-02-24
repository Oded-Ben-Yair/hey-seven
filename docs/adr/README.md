# Architecture Decision Records (ADRs)

Architectural decisions documented inline in source code, extracted here for discoverability.

## Index

| # | Decision | Status | Source |
|---|----------|--------|--------|
| 001 | [Custom StateGraph over create_react_agent](001-custom-stategraph.md) | Accepted | `graph.py:1-31` |
| 002 | [In-memory rate limiting for MVP](002-rate-limiting.md) | Accepted | `middleware.py:327-366` |
| 003 | [Feature flag dual-layer design](003-feature-flags.md) | Accepted | `graph.py:584-618` |
| 004 | [Degraded-pass validation strategy](004-degraded-pass.md) | Accepted | `nodes.py` |
| 005 | [English-only LLM responses](005-i18n-responses.md) | Deferred | `nodes.py:55-71` |
| 006 | [ChromaDB dev / Vertex AI prod](006-vector-db-split.md) | Accepted | `config.py:79` |
| 007 | [Classifier restricted mode](007-classifier-degradation.md) | Accepted | `guardrails.py:608+` |
| 008 | [threading.Lock in InMemoryBackend](008-threading-lock.md) | Accepted | `state_backend.py:94-99` |
| 009 | [UNSET_SENTINEL as object()](009-unset-sentinel.md) | Accepted | `state.py:28` |
| 010 | [Middleware execution order](010-middleware-order.md) | Accepted | `app.py:173-186` |
