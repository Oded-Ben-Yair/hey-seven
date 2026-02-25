# ADR-016: asyncio.to_thread for Sync Retrievers

## Status
Superseded by [ADR-020](020-concurrent-retrieval.md)

> **Note (R60)**: Retrieval functions migrated from sync+to_thread to async-native
> using `loop.run_in_executor()`. ADR-020 documents the current architecture.

## Context
ChromaDB's retrieval API is synchronous. In the async LangGraph pipeline, blocking the event loop during retrieval stalls all concurrent SSE streams.

## Decision
Use `asyncio.to_thread()` for ChromaDB retrieval in local dev. This is acceptable because:
1. ChromaDB is dev-only (production uses Vertex AI Vector Search, which is async-native)
2. Local dev runs single-user (no concurrent stream contention)
3. Thread pool size (default 8) is sufficient for dev load

## Caveats
- **NEVER** use `asyncio.to_thread()` for Redis in production (R47 CRITICAL) -- use `redis.asyncio`
- **NEVER** use `asyncio.to_thread()` for production vector search -- use async-native client
- This pattern is a dev-only compromise, not a production architecture
