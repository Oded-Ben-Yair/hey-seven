# FastAPI Backend & GCP Deployment — Hostile Review Round 2

**Date:** 2026-02-12
**Reviewer:** api-critic (code-judge, hostile mode, Claude Opus 4.6)
**Score: 72/100** (Round 1: 38/100, Delta: +34)

## Round 1 Critical Issues — Resolution Status

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| C1 | Deprecated `@app.on_event("startup")` | FIXED | `lifespan` async context manager (main.py:44-68) |
| C2 | Zero authentication | FIXED | `verify_api_key` + `APIKeyHeader`, fail-closed (503 if unset) |
| C3 | CORS wildcard + credentials | FIXED | Wildcard → no credentials; explicit origins → credentials |
| C4 | In-memory rate limiter useless | FIXED | Removed entirely; WebSocket has per-connection limiting |
| C5 | WebSocket zero auth/limits | FIXED | Auth, 8KiB size, 5 msg/sec rate, 5-min idle timeout, JSON validation |
| C6 | Global mutable state | FIXED | Agent on `app.state`, accessed via `request.app.state.agent` |
| C7 | No input sanitization on `player_id` | FIXED | Regex patterns on all IDs, min/max lengths on messages |
| C8 | `comp_type` unvalidated | FIXED | `Literal["room", "dining", "entertainment", "freeplay", "cashback"]` |
| C9 | `BaseHTTPMiddleware` deprecated | FIXED | Pure ASGI middleware classes |
| C10 | Dependency ranges too wide | FIXED | All 18 deps pinned with `==` |
| C11 | No graceful shutdown | PARTIALLY_FIXED | SIGTERM handler exists but no `--timeout-graceful-shutdown` in Dockerfile |
| C12 | Symlink workaround | PARTIALLY_FIXED | Proper COPY but still `RUN touch __init__.py`, no pyproject.toml |
| C13 | Health check always healthy | FIXED | Returns 503 when agent is None |
| C14 | `--allow-unauthenticated` | FIXED | Both cloudbuild and deploy use `--no-allow-unauthenticated` |

**Summary: 12 FIXED, 2 PARTIALLY_FIXED**

## New Issues Found

### Critical

**N1.** Timing attack on API key comparison — `api_key != expected` at routes.py:45 and main.py:154 uses `!=` which short-circuits. **Must use `hmac.compare_digest()`.**

### Important
- **N2:** WebSocket `thread_id` not validated (main.py:202) — REST has regex, WS has nothing
- **N3:** No timeout on LLM calls in REST routes (routes.py:192) — hangs until Cloud Run 300s timeout
- **N4:** `import re` inside route handler body (routes.py:221) — should be module-level
- **N5:** No `.dockerignore` — entire repo sent as build context
- **N6:** `chromadb==0.5.23` in production requirements — ~200MB bloat
- **N7:** No security headers (HSTS, X-Content-Type-Options, X-Frame-Options, CSP)
- **N8:** No Cloud Run service account specified — defaults to Editor role

### Minor
- m1: Docker base image not pinned to digest
- m2: `latest` tag still pushed in Cloud Build
- m3: Structured logging only partial (access=JSON, app=plaintext)
- m4: `2>/dev/null` swallows errors in deploy.sh
- m5: Health endpoint defined in two places
- m6: No `pyproject.toml`
- m7: No tests of any kind
- m8: Request ID truncated to 8 chars
