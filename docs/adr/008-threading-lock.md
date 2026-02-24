# ADR 008: threading.Lock in InMemoryBackend

## Status
Accepted (R36), defended in R48

## Context
InMemoryBackend is called from async coroutines but uses `threading.Lock` instead of `asyncio.Lock`.

## R48 Review Debate
- **DeepSeek C2, GPT M8**: Flagged threading.Lock as blocking event loop under contention
- **R36 original rationale**: Sub-microsecond ops, no awaits in critical section

## Decision
Keep `threading.Lock`. Rationale:

1. **Dual-context usage**: InMemoryBackend sync methods (`set`, `get`) are called from BOTH sync contexts (`RateLimitMiddleware.__init__`, `get_state_backend()`) and async contexts (`async_set`, `async_get`). `asyncio.Lock` requires `async with` — breaks sync callers.

2. **Lock hold time bounded**: Normal operations are O(1) dict lookups (~100ns). Sweep is O(batch_size) with `_SWEEP_BATCH_SIZE=200` (~0.2ms worst case). This is 5x below the 1ms threshold for noticeable event loop blocking.

3. **Contention is theoretical**: Two coroutines contending on the lock requires both to be inside the critical section simultaneously. Since there are no `await` points inside the lock, this requires preemption — which asyncio's cooperative scheduler does not do.

## Consequences
- Positive: Works in both sync and async contexts without separate code paths
- Positive: Simpler than maintaining parallel Lock types
- Negative: 0.2ms worst case under pathological sweep + contention (acceptable)
- Risk: If future changes add `await` inside the lock → must switch to asyncio.Lock
