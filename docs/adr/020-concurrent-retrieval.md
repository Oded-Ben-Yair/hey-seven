# ADR-020: Concurrent Retrieval Strategies via ThreadPoolExecutor

## Status
Accepted

## Context
The RAG pipeline runs two retrieval strategies (semantic + augmented/schedule) and fuses results via RRF. Previously sequential (2x latency), then per-request ThreadPoolExecutor (R56, wasteful), now module-level pool.

## Decision
Use a module-level `ThreadPoolExecutor(max_workers=2)` for concurrent retrieval. Both strategies are I/O-bound (embedding API + vector store query), making thread-based concurrency effective.

## Design
- `_RETRIEVAL_POOL`: module-level, reused across all requests (2 threads)
- Each strategy has independent `future.result(timeout=RETRIEVAL_TIMEOUT)`
- Strategy failure is isolated: one failing doesn't affect the other
- Both failing returns empty list (graceful degradation)
- Pool is created at import time, lives for the process lifetime

## Alternatives Considered
1. **Sequential** (original): Simple but 2x latency (rejected for performance)
2. **Per-request ThreadPoolExecutor** (R56): Correct but creates/destroys pool per call (rejected for overhead)
3. **asyncio.gather** with async retrievers: Ideal but ChromaDB is sync-only; production Vertex AI is async-native (deferred to production migration)

## Consequences
- Positive: Halves retrieval latency (P50: 300ms → 150ms typical)
- Positive: Zero per-request allocation overhead
- Positive: Thread pool bounded (2 threads, no growth)
- Negative: Module-level pool requires atexit cleanup for clean shutdown (handled by Python GC)
- Negative: Thread-based concurrency adds ~50µs overhead vs direct sync calls (negligible vs I/O latency)
