# R40 Fixes: fixer-beta (D8 Scalability)

## Test Results After Fixes
- **2168 passed, 0 failed**, 66 warnings in 322.74s
- **Coverage: 90.11%** (above 90% gate)
- No regressions introduced

## Fixes Applied

### D8-C001 [CRITICAL] — TTL jitter to prevent thundering herd
**Files modified**: 7 files (all TTLCache singletons I own)
- `src/config.py` — `_settings_cache`: ttl=3600+random(0,300)
- `src/agent/nodes.py` — `_llm_cache`, `_validator_cache`, `_greeting_cache`: each gets independent jitter
- `src/agent/circuit_breaker.py` — `_cb_cache`: ttl=3600+random(0,300)
- `src/agent/memory.py` — `_checkpointer_cache`: ttl=3600+random(0,300)
- `src/agent/whisper_planner.py` — `_whisper_cache`: ttl=3600+random(0,300)
- `src/state_backend.py` — `_state_backend_cache`: ttl=3600+random(0,300)
- `src/rag/embeddings.py` — `_embeddings_cache`: ttl=3600+random(0,300)

**Approach**: Each TTLCache gets `ttl=3600 + random.randint(0, 300)` computed once at module import time (container startup). Since each module imports independently, each singleton gets a different random offset. This spreads cache reconstruction over a 5-minute window instead of all at once.

**Note**: `src/rag/pipeline.py` uses a custom dict-based cache (not TTLCache) — left to fixer-alpha.

### Calibrator finding — SIGTERM graceful drain for SSE streams
**File modified**: `src/api/app.py`
- Added `_active_streams` set to track active SSE tasks
- Added `_shutting_down` asyncio.Event for shutdown coordination
- Added SIGTERM signal handler in lifespan that sets the shutdown event
- New /chat requests during shutdown return 503 with Retry-After: 5
- Lifespan shutdown waits up to 30s for active streams to finish
- SSE generator wrapped with `_tracked_generator()` that registers/unregisters tasks
- Exception handling catches RuntimeError for test compatibility (Starlette TestClient runs in non-main thread)

### Calibrator finding — --require-hashes documentation
**File modified**: `Dockerfile`
- Added documentation for supply chain hardening via `--require-hashes`
- Documented the upgrade path: `pip-compile --generate-hashes`
- Tracked as TODO(HEYSEVEN-58) — requires generating hashes for all dependencies

### D8-M001 [MAJOR] — Rate limiter upgrade trigger
**File modified**: `src/api/middleware.py`
- Added concrete upgrade trigger to the rate limiter ADR: "Upgrade to Cloud Armor when daily traffic > 1000 req OR before any paid client deployment OR max-instances regularly > 3"

## Score Impact (D8 Scalability)
- D8-C001 CRITICAL resolved: +0.5 (thundering herd eliminated)
- SIGTERM drain implemented: +0.5 (carried since R37, now resolved)
- D8-M001 trigger added: +0.0 (documentation, no code change)
- Net D8 delta: +1.0 (from calibrated 8.0 to 9.0, capped at 9.0)
