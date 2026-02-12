# FastAPI Backend & GCP Deployment -- Hostile Review Round 1

Date: 2026-02-12
Reviewer: api-critic

## Overall Score: 38/100

This is a first-draft boilerplate that looks like it was written in 2023 and never updated. It uses deprecated FastAPI patterns, has zero authentication, a toy rate limiter that fails with more than one replica, CORS wide open by default, no production ASGI server config, and dependency ranges so wide they guarantee non-reproducible builds. For interview prep infrastructure at a real startup, this would not survive a single day in production.

## Section Scores

| Component | Score | Critical Issues | Important Issues |
|-----------|-------|----------------|-----------------|
| API Routes (routes.py) | 42 | 2 | 4 |
| API Main (main.py) | 35 | 3 | 3 |
| Middleware | 30 | 3 | 3 |
| WebSocket | 25 | 4 | 2 |
| Dockerfile | 50 | 1 | 4 |
| Cloud Build | 45 | 2 | 3 |
| Deploy Script | 40 | 1 | 3 |
| Dependencies | 25 | 2 | 3 |
| Security | 20 | 5 | 2 |
| Production Readiness | 30 | 3 | 4 |

---

## Critical Issues (MUST fix)

### C1. DEPRECATED: `@app.on_event("startup")` removed in Starlette 1.0+ / FastAPI 0.120+

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/main.py`, line 58

`@app.on_event("startup")` has been deprecated since FastAPI 0.103 (August 2023) and is **removed** in recent FastAPI/Starlette versions. The current pattern is `lifespan` context manager:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    agent = create_agent()
    set_agent(agent)
    yield
    # shutdown / cleanup
```

This code will fail to start on current FastAPI versions.

---

### C2. ZERO Authentication / Authorization

**Files**: All route files

There is no authentication middleware, no API key validation, no JWT verification, no OAuth2 integration -- nothing. Every endpoint including `/api/v1/chat`, `/api/v1/player/{player_id}`, and `/api/v1/comp/calculate` is completely open to the public internet. The Cloud Run deploy uses `--allow-unauthenticated`.

For a casino application handling player PII (names, tracking numbers, visit history, comp balances, preferences), this is a regulatory and legal catastrophe. Casino player data is subject to gaming commission regulations in every jurisdiction.

**Required**: At minimum, API key auth for service-to-service calls. For the frontend, JWT with proper RBAC. IAP (Identity-Aware Proxy) for Cloud Run is another option.

---

### C3. CORS Defaults to Wildcard `*` with `allow_credentials=True`

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/middleware.py`, lines 36-46

```python
allowed_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,  # <-- THIS
    allow_methods=["*"],
    allow_headers=["*"],
)
```

`allow_origins=["*"]` with `allow_credentials=True` is **invalid per the CORS spec**. Browsers will reject credentialed requests when the origin is `*`. FastAPI/Starlette will either refuse this combo or silently downgrade. Either way, this is a misconfiguration that signals the developer never tested with real browser requests.

The default MUST be a restrictive origin list, not `*`. The `*` should only exist behind an explicit `ENVIRONMENT=development` flag, and never with credentials.

---

### C4. In-Memory Rate Limiter is Useless with Cloud Run Auto-Scaling

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/middleware.py`, lines 142-231

The `RateLimiter` class stores request counts in a Python dict. Cloud Run scales from 0 to 10 instances. Each instance has its own dict. An attacker gets `60 * N` requests per minute where N is the number of active instances. With concurrency=80 and max-instances=10, that is effectively 600 requests/minute per IP -- which is no limit at all.

Worse: the dict grows unbounded. There is no TTL cleanup of old IP entries. Over time, `_request_log` becomes a memory leak. `defaultdict(list)` never removes keys even after all timestamps are cleaned from the list.

**Required**: Redis/Memorystore-backed rate limiting, or use GCP API Gateway / Cloud Armor rate limiting at the infrastructure level.

---

### C5. WebSocket Has Zero Authentication, No Rate Limiting, No Message Size Limits

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/main.py`, lines 80-147

The WebSocket endpoint:
- Accepts any connection with no auth check (line 89: `await websocket.accept()` unconditionally)
- Has no message size limit -- a client can send a 100MB JSON blob
- Has no rate limiting on messages within a connection
- Has no connection timeout / idle disconnect
- Does not validate `json.loads(data)` -- malformed JSON crashes with an unhandled exception that is caught by the generic `except Exception` but leaks to the error log
- No `thread_id` validation -- any string is accepted, including injection attempts
- Imports `from .routes import _get_agent` inside the loop body on every message (line 115-116) -- importing a private function repeatedly is wasteful and indicates poor architecture

This WebSocket is a DDoS vector. A single client can hold a connection open indefinitely, spam messages at the LLM, and exhaust your Gemini API quota.

---

### C6. Global Mutable State for Agent Injection is Not Thread-Safe

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/routes.py`, lines 103-128

```python
_agent: Any = None

def set_agent(agent: Any) -> None:
    global _agent
    _agent = agent
```

This uses a module-level global variable with no lock. While CPython's GIL provides some protection, this pattern:
- Breaks with multiple workers (the Dockerfile CMD uses `--workers 1`, but if anyone changes it, the agent is not shared across processes)
- Has no type safety (`Any` everywhere -- this is a typed codebase pretending to use types)
- Provides no way to test routes without mutating global state
- Cannot handle agent re-initialization or hot-reload

**Required**: Use FastAPI dependency injection via `app.state` or `Depends()`. The lifespan pattern naturally supports this:
```python
app.state.agent = create_agent()
```

---

### C7. No Input Sanitization on `player_id` Path Parameter

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/routes.py`, line 166

```python
@router.get("/player/{player_id}", response_model=PlayerResponse)
async def get_player(player_id: str) -> PlayerResponse:
```

`player_id` accepts any string. There is no validation of format (the examples show `PLY-482910` but nothing enforces this). Depending on how `check_player_status` uses this value downstream (Firestore query, SQL query, log injection), this is an injection vector.

**Required**: Add a Pydantic `Path` parameter with regex validation: `player_id: str = Path(..., pattern=r"^PLY-\d{6}$")`

---

### C8. `comp_type` is Unvalidated Free Text

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/routes.py`, lines 67-71

```python
comp_type: str = Field(
    ...,
    description="Type of comp to calculate.",
    examples=["room", "dining", "show", "freeplay", "travel"],
)
```

The examples list 5 valid types but there is no `Literal` constraint or enum. A client can send `comp_type: "'; DROP TABLE players; --"` and it will be passed directly to `calc_comp_tool.invoke()`. Whether this causes damage depends on the tool implementation, but the API layer should never trust downstream code to validate.

**Required**: Use `Literal["room", "dining", "show", "freeplay", "travel"]` or a Pydantic enum.

---

### C9. `BaseHTTPMiddleware` is Deprecated for Production Use

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/main.py`, lines 49-51

```python
app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limiting_middleware)
app.add_middleware(BaseHTTPMiddleware, dispatch=error_handling_middleware)
app.add_middleware(BaseHTTPMiddleware, dispatch=request_logging_middleware)
```

`BaseHTTPMiddleware` wraps the ASGI flow in a way that breaks streaming responses and adds overhead. The Starlette docs have warned against it since 2024. For production, use pure ASGI middleware or Starlette's `ASGIMiddleware` base class.

With 3 layers of `BaseHTTPMiddleware`, every request goes through 3 extra async context switches. The error handling middleware also silently catches exceptions that FastAPI's built-in exception handlers should be handling (like `RequestValidationError`), potentially masking 422 errors as 500s.

---

### C10. Dependencies Use Ranges So Wide They Guarantee Build Breakage

**File**: `/home/odedbe/projects/hey-seven/boilerplate/requirements.txt`

```
langgraph>=1.0.0,<2.0.0
langchain-core>=1.0.0,<2.0.0
langchain-google-genai>=2.0.0,<5.0.0   # <-- 3 major versions!
fastapi>=0.115.0,<1.0.0
```

- `langchain-google-genai>=2.0.0,<5.0.0` spans THREE major versions. Major versions have breaking API changes by definition.
- `langgraph>=1.0.0,<2.0.0` -- LangGraph has been evolving rapidly. A build today and a build next week will get different versions with potentially different behavior.
- No `pip-compile` or lockfile. Two developers will get different dependency trees.
- `chromadb>=0.5.0,<1.0.0` is listed for "local dev" but will be installed in the production Docker image. ChromaDB pulls in heavy C++ dependencies (hnswlib, sqlite3) that bloat the image and attack surface.

**Required**: Use `pip-tools` with a `requirements.in` -> `requirements.txt` lockfile workflow, or switch to `uv` (the standard in Feb 2026). Pin exact versions in the lockfile. Separate `requirements-dev.txt` for dev-only dependencies.

---

### C11. No Graceful Shutdown or Connection Draining

**Files**: `main.py`, `Dockerfile`

Cloud Run sends SIGTERM before terminating a container. The current setup:
- Has no shutdown handler (the deprecated `on_event("shutdown")` is not even used)
- uvicorn with `--workers 1` will handle SIGTERM, but active WebSocket connections will be killed mid-conversation
- No connection draining timeout configured
- No cleanup of Firestore clients, agent resources, or LLM connections

**Required**: Use the `lifespan` pattern with a shutdown phase. Configure `--timeout-graceful-shutdown` in uvicorn.

---

### C12. Dockerfile Uses Symlink Workaround Instead of Proper Package Structure

**File**: Looking at the directory listing:
```
drwxr-xr-x  langgraph-agent
lrwxrwxrwx  langgraph_agent -> langgraph-agent
```

The `Dockerfile` (line 33) copies `langgraph-agent/` to `./langgraph_agent/`. The local dev uses a symlink. This means:
- Local and Docker have different filesystem structures
- The `RUN touch ./langgraph_agent/__init__.py` (line 39) suggests the package structure is not reliable
- This will break if anyone uses `pip install -e .` or proper Python packaging

**Required**: Rename the directory to `langgraph_agent` (Python-importable name) and remove the symlink.

---

### C13. Health Check Lies -- Always Returns "healthy" Even When Agent Failed

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/main.py`, lines 74-77

```python
@app.get("/health")
async def root_health() -> dict:
    return {"status": "healthy"}
```

This always returns "healthy" even if the agent failed to initialize (line 68-71 catches the exception and logs but continues). Cloud Run will route traffic to an instance that cannot serve chat requests. The detailed health check at `/api/v1/health` checks Firestore but NOT whether the agent initialized successfully.

**Required**: The health check must verify `_agent is not None`. Return 503 if the agent is not ready. Use a readiness probe in addition to the liveness probe.

---

### C14. `--allow-unauthenticated` on Cloud Run for a Casino Data API

**Files**: `cloudbuild.yaml` line 52, `deploy.sh` line 111

A service that exposes player PII, comp calculations, and conversation history is deployed with `--allow-unauthenticated`. This means anyone on the internet can hit these endpoints.

Even if you add API key auth at the application level, Cloud Run's IAM layer provides defense-in-depth. For a casino application with player data, this should use IAM authentication with the frontend calling via a service account or through IAP.

---

## Important Issues (SHOULD fix)

### I1. No Structured Logging (JSON)

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/main.py`, lines 23-26

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
```

Plain text logging on Cloud Run means Cloud Logging cannot parse fields, filter by severity, or correlate traces. GCP expects JSON-formatted logs with `severity`, `message`, `trace`, and `spanId` fields.

**Required**: Use `python-json-logger` or `structlog` with GCP-compatible format. Include `trace` header from `X-Cloud-Trace-Context` for distributed tracing.

---

### I2. Request ID is Truncated UUID -- Not Useful for Tracing

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/middleware.py`, line 67

```python
request_id = str(uuid.uuid4())[:8]
```

8-character truncated UUID has a collision probability of 1 in ~4 billion. With high traffic, collisions will happen and you will not be able to trace a specific request. Also, this does not integrate with GCP's trace context (`X-Cloud-Trace-Context` header).

**Required**: Use full UUID or integrate with Cloud Trace. Propagate the trace context from incoming headers if present.

---

### I3. `calculate_comp` Endpoint Returns Untyped `dict`

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/routes.py`, line 187

```python
@router.post("/comp/calculate")
async def calculate_comp(request: CompRequest) -> dict:
```

This is the only route without a `response_model`. The return type is `dict` which means:
- No response validation
- No OpenAPI schema documentation
- The response could contain anything, including internal error details

Every other route uses a typed response model. This one does not. Inconsistency breeds bugs.

---

### I4. No Timeout on LLM/Agent Calls

**Files**: `routes.py` (chat endpoint), `main.py` (WebSocket)

The `chat()` function is awaited with no timeout:
```python
result = await chat(agent=agent, message=message, thread_id=request.thread_id)
```

LLM calls can hang for 30+ seconds or indefinitely. Cloud Run has a 300s timeout, but the client will get no response during that time. There is no circuit breaker, no retry logic, no timeout wrapping.

**Required**: Wrap in `asyncio.wait_for(chat(...), timeout=60)` and return a meaningful timeout error.

---

### I5. No `.dockerignore` File

**Observation**: The glob search found no `.dockerignore`.

Without a `.dockerignore`, `docker build` copies everything into the build context: `.git/`, `node_modules/`, `__pycache__/`, test files, the frontend directory, IDE configs, potentially `.env` files with secrets.

This slows builds, bloats the context, and risks leaking secrets into the image layer cache.

**Required**: Create a `.dockerignore` excluding at minimum: `.git`, `*.pyc`, `__pycache__`, `node_modules`, `.env*`, `*.md`, `tests/`, `frontend/`.

---

### I6. Docker Image Not Pinned to Digest

**File**: `/home/odedbe/projects/hey-seven/boilerplate/gcp/Dockerfile`, line 7

```dockerfile
FROM python:3.12-slim AS dependencies
```

`python:3.12-slim` is a floating tag. The image you build today is different from the image you build tomorrow. For reproducible, auditable builds, pin to a specific digest:
```dockerfile
FROM python:3.12-slim@sha256:<specific-digest>
```

---

### I7. `latest` Tag Used in Production

**File**: `/home/odedbe/projects/hey-seven/boilerplate/gcp/cloudbuild.yaml`, line 23

The build creates a `latest` tag AND a `SHORT_SHA` tag. The deploy uses `SHORT_SHA` (good), but the `latest` tag is still pushed. If anyone deploys manually using `latest`, they will get an unpredictable version. Remove the `latest` tag from production image tagging.

---

### I8. Firestore Client Created Per Health Check

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/routes.py`, lines 225-229

```python
db = firestore.Client()
db.collection("_health").document("ping").get()
```

A new `firestore.Client()` is created on every health check call. If Cloud Run's startup probe hits this every 10 seconds, that is a new gRPC connection and TLS handshake every 10 seconds. The client should be created once during startup and reused.

---

### I9. Error Messages Leak Information

**File**: `/home/odedbe/projects/hey-seven/boilerplate/api/routes.py`, line 182

```python
detail=f"Failed to look up player {player_id}.",
```

The error message echoes back the `player_id`. If the player_id contains HTML/JS (since there is no validation -- see C7), this could be an XSS vector if the frontend renders the error message without escaping.

---

### I10. No HTTPS/TLS Configuration Mentioned Anywhere

Cloud Run provides TLS termination, so the application does not need to handle it directly. However, there is no mention of:
- HSTS headers
- Redirecting HTTP to HTTPS
- Setting `Secure` flag on any cookies
- CSP headers

The middleware adds `X-Request-ID` and `X-Response-Time-Ms` but no security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security`, `Content-Security-Policy`).

---

### I11. No Tests of Any Kind

There are no test files anywhere in the boilerplate. No unit tests, no integration tests, no contract tests. The Cloud Build pipeline has no test step -- it builds and deploys directly.

---

### I12. Cloud Build Has No Vulnerability Scanning Step

**File**: `/home/odedbe/projects/hey-seven/boilerplate/gcp/cloudbuild.yaml`

The pipeline is: build -> push -> deploy. There is no:
- Container vulnerability scanning (`gcloud artifacts docker images scan`)
- Dependency audit (`pip-audit`)
- SAST scanning
- License compliance checking

---

## Outdated Patterns (for Feb 2026)

### O1. `@app.on_event("startup")` -- Use `lifespan` Instead
Deprecated since FastAPI 0.103 (Aug 2023). Removed in recent versions. See C1.

### O2. `BaseHTTPMiddleware` -- Use Pure ASGI Middleware
Has been discouraged for 2+ years. Breaks streaming, adds overhead. See C9.

### O3. `requirements.txt` with Ranges -- Use `uv` or `pip-tools` with Lock Files
As of Feb 2026, `uv` is the de facto Python package manager. The standard is `pyproject.toml` + `uv.lock`. `requirements.txt` with version ranges is a 2020 pattern.

### O4. `python-dotenv` -- Use Pydantic Settings with Environment Variables
Pydantic has `pydantic-settings` which provides typed, validated configuration from environment variables with defaults. `python-dotenv` is the old way.

### O5. `logging.basicConfig()` -- Use `structlog` or `python-json-logger`
Plain text logging is inadequate for cloud environments. Structured JSON logging has been the standard for GCP since 2022.

### O6. No `pyproject.toml` -- Project Metadata Missing
There is no `pyproject.toml`, no `setup.py`, no package metadata. The project cannot be installed as a package, which makes testing and dependency management harder.

### O7. Uvicorn Without Gunicorn for Production
`uvicorn` alone (line 58 of Dockerfile) without a process manager. While `--workers 1` is fine for Cloud Run (which manages scaling via instances), the standard production pattern is `gunicorn` with `uvicorn.workers.UvicornWorker` for proper process management, signal handling, and pre-fork worker support.

---

## Missing Features

### M1. No API Key or JWT Authentication
See C2. This is the most critical missing feature. A casino data API with player PII exposed to the open internet.

### M2. No OpenAPI Security Schemes
The OpenAPI docs at `/docs` and `/redoc` show no security schemes. Anyone discovering the docs URL can see and use every endpoint.

### M3. No Request Body Size Limits
Neither the FastAPI app nor the middleware enforces a maximum request body size. The only limit is `max_length=4000` on the chat message field. A client could send a massive JSON body on the `/comp/calculate` endpoint.

### M4. No Metrics / Prometheus Endpoint
No `/metrics` endpoint, no `prometheus-client` integration, no custom metrics for LLM call latency, token usage, or error rates.

### M5. No Circuit Breaker for LLM Calls
If the Gemini API is down, every request will wait until Cloud Run's 300s timeout, then fail. There is no circuit breaker to fail fast after N consecutive failures.

### M6. No Retry Logic for Transient Failures
LLM APIs return 429s (rate limit) and 503s (overloaded). There is no retry-with-backoff logic anywhere.

### M7. No Audit Logging for Player Data Access
For a casino application, every access to player data should be audit-logged (who accessed what, when, from where). There is no audit trail.

### M8. No WebSocket Heartbeat / Ping-Pong
The WebSocket connection has no keepalive mechanism. Idle connections through load balancers and proxies will be silently dropped after their idle timeout.

### M9. No API Versioning Strategy Beyond URL Prefix
`/api/v1/` is in the URL but there is no strategy for what happens with v2. No deprecation headers, no version negotiation.

### M10. No Content-Type Enforcement
The routes accept `application/json` by default, but there is no explicit enforcement. No validation that the incoming content type matches expectations.

---

## Minor Issues

### m1. Inline Imports Inside Route Handlers

```python
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    from langgraph_agent.agent import chat   # imported on every call
```

Lines 146, 173, 193, 219, 225 all have inline imports. This is done to handle the case where `langgraph_agent` might not be available, but it is a performance anti-pattern (import lock on every request) and hides import errors until runtime.

### m2. `asyncio` Imported but Never Used
**File**: `main.py`, line 7: `import asyncio` -- never referenced.

### m3. Request ID Not Returned in Error Responses from Routes
The middleware error handler includes `request_id`, but the `HTTPException` raised in routes (e.g., line 159) does not include it. A client receiving a 500 from a route handler will not get a request ID to reference in support.

### m4. Health Endpoint Defined Twice
`/health` is defined at the app level (main.py line 74) AND at `/api/v1/health` (routes.py line 212). They return different response shapes. The root `/health` returns `{"status": "healthy"}` while the versioned one returns a full `HealthResponse`. This is confusing.

### m5. `CompRequest` Does Not Validate `player_id` Format
Same issue as C7 but for the comp endpoint. The `player_id` in `CompRequest` has no pattern validation.

### m6. No `__all__` in `__init__.py`
The package init file has a docstring but no `__all__`, making it unclear what the public API surface is.

### m7. Deploy Script Sleep is Only 10 Seconds
**File**: `deploy.sh`, line 134. Cloud Run cold starts with LLM initialization can take 30-60 seconds. A 10-second sleep is not enough for the health check to succeed on first request.

### m8. `2>/dev/null` Suppresses Errors in Deploy Script
**File**: `deploy.sh`, line 72. If the repository describe fails for a reason other than "not found" (e.g., auth failure), the error is swallowed and the script tries to create the repo, which will also fail with a confusing error.

### m9. No Cloud Run Service Account Configuration
Neither `cloudbuild.yaml` nor `deploy.sh` specify a dedicated service account for the Cloud Run service. It will use the default compute service account, which typically has overly broad permissions.

### m10. `chromadb` in Production Dependencies
ChromaDB is listed under "Vector DB (local dev)" but will be installed in the Docker image. This adds ~200MB to the image and dozens of unnecessary C dependencies to the attack surface.

---

## Summary

| Category | Count |
|----------|-------|
| Critical | 14 |
| Important | 12 |
| Outdated Patterns | 7 |
| Missing Features | 10 |
| Minor | 10 |
| **Total Issues** | **53** |

### Top 3 Priorities

1. **Authentication**: Add API key auth immediately. Player PII is exposed to the entire internet. (C2, C5, C14)
2. **Modernize FastAPI patterns**: Replace deprecated `on_event`, `BaseHTTPMiddleware`, global state injection with `lifespan`, pure ASGI middleware, and `app.state` dependency injection. (C1, C6, C9)
3. **Production infrastructure**: Redis rate limiting, structured JSON logging, circuit breaker for LLM calls, dependency lockfile. (C4, C10, I1, M5)

### What Was Done Well

- Multi-stage Docker build (correct pattern, even if details need work)
- Pydantic models for request/response validation (partial -- `comp/calculate` response missing)
- Request ID propagation (concept is right, implementation is truncated)
- CORS is environment-variable-driven (concept is right, default is wrong)
- Error handling middleware prevents stack trace leaks (concept is right, implementation uses deprecated middleware)
- Cloud Build pipeline exists and uses Secret Manager for API keys
- Deploy script has `--dry-run` mode (good operational practice)
- API versioning via URL prefix (basic but functional)
