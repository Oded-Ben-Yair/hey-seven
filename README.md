# Hey Seven Property Q&A Agent

An AI-powered property concierge for Mohegan Sun casino resort, built with LangGraph and Gemini 2.5 Flash.

Guests ask natural language questions about dining, entertainment, rooms, amenities, gaming, and promotions. The agent retrieves answers from a curated knowledge base using RAG (Retrieval-Augmented Generation) and streams responses token-by-token via Server-Sent Events.

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/Oded-Ben-Yair/hey-seven.git
cd hey-seven
cp .env.example .env   # Add your GOOGLE_API_KEY

# 2. Run with Docker (recommended)
docker compose up --build

# 3. Open browser
open http://localhost:8080
```

The first boot takes ~30 seconds to embed property data into ChromaDB. Subsequent restarts are fast (data persists in a Docker volume).

### Local Development

```bash
pip install -r requirements.txt
make run
```

## Architecture

```
START --> router --+--> greeting --> END
                   +--> off_topic --> END
                   +--> retrieve --> generate --> validate --+--> respond --> END
                                      ^                      +--> generate (retry)
                                      +----------------------+--> fallback --> END
```

A custom 8-node LangGraph `StateGraph` with two conditional routing points:

1. **Router** classifies user intent (7 categories) using structured LLM output.
2. **Retrieve** searches ChromaDB for relevant property knowledge.
3. **Generate** produces a grounded response using the 11-rule concierge system prompt (VIP interaction style, competitor deflection, responsible gaming).
4. **Validate** performs adversarial review against 6 criteria (grounded, on-topic, no gambling advice, read-only, accurate, responsible gaming).
5. On validation failure, the graph retries once, then falls back to a safe contact-info response.

Tokens stream to the browser in real time via SSE (`astream_events` v2). The stream checks `request.is_disconnected()` to cancel on client disconnect.

**Circuit breaker**: The generate node is protected by an in-memory circuit breaker (5 failures -> 60s cooldown -> half-open probe) to prevent cascading LLM failures.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details: node descriptions, state schema, routing logic, prompt system, cost model, and deployment.

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Agent framework | LangGraph custom `StateGraph` (8 nodes) | Validation loop, structured routing, domain guardrails |
| LLM | Gemini 2.5 Flash | GCP-aligned, cost-effective (~$0.0014/request) |
| Vector DB | ChromaDB (embedded) | Zero infrastructure for demo; Vertex AI Vector Search for production |
| Streaming | Real token SSE via `astream_events` v2 | True progressive rendering with timeout + disconnect detection |
| Embeddings | Google `text-embedding-004` | GCP-native, free tier, 768 dimensions |
| Frontend | Vanilla HTML/CSS/JS | No build step, minimal footprint, backend is 90% of evaluation |
| Validation | Adversarial LLM review (6 criteria) | Catches hallucination, off-topic drift, gambling advice leaks |
| Config | `pydantic-settings` BaseSettings | Zero hardcoded values; every constant overridable via env var |
| Guardrails | Deterministic regex + LLM validation | Pre-LLM `audit_input()` (7 patterns) + `detect_responsible_gaming()` (22 patterns) + `detect_age_verification()` (6 patterns) |

## Project Structure

```
hey-seven/
├── src/
│   ├── config.py              # Centralized config (pydantic-settings, 20 env vars)
│   ├── agent/
│   │   ├── graph.py           # 8-node StateGraph + chat + chat_stream (SSE)
│   │   ├── nodes.py           # 8 nodes + 2 routers + CircuitBreaker + guardrails
│   │   ├── state.py           # PropertyQAState (9 fields) + Pydantic models
│   │   ├── prompts.py         # 3 prompt templates (11-rule concierge, VIP tone)
│   │   └── tools.py           # search_knowledge_base + search_hours
│   ├── rag/
│   │   ├── pipeline.py        # Ingest, chunk (800/120), embed, retrieve (ChromaDB)
│   │   └── embeddings.py      # Google text-embedding-004 config
│   └── api/
│       ├── app.py             # FastAPI app, lifespan, SSE streaming, disconnect detection
│       ├── models.py          # Pydantic request/response + SSE event schemas
│       └── middleware.py      # 6 pure ASGI middleware (logging, errors, security, auth, rate limit, body limit)
├── data/
│   └── mohegan_sun.json       # Curated property data (30 items, 7 categories)
├── static/
│   └── index.html             # Branded chat UI (Hey Seven gold/dark/cream)
├── tests/                     # 277 tests across 11 files
├── Dockerfile                 # Multi-stage Python 3.12, non-root, HEALTHCHECK
├── docker-compose.yml         # Single service, health check, named volume, 2GB limit
├── requirements.txt           # Pinned production dependencies
├── requirements-dev.txt       # Test/dev deps (pytest, ruff, black, coverage)
├── .env.example               # All 20 env vars with inline documentation
├── Makefile                   # test-ci, test-eval, lint, run, docker-up
├── cloudbuild.yaml            # GCP Cloud Build CI/CD (4-step pipeline)
└── pyproject.toml             # Project config, pytest, ruff settings
```

## Running Tests

```bash
# Unit + integration tests (no API key needed, 265 tests)
make test-ci

# Deterministic eval tests (no API key, VCR fixtures, 12 tests)
pytest tests/test_eval_deterministic.py -v

# Live eval tests (requires GOOGLE_API_KEY, 14 tests)
make test-eval

# All tests with coverage
pytest tests/ --cov=src --cov-report=term-missing --ignore=tests/test_eval.py

# Lint
make lint   # 0 errors
```

**Test pyramid**: 277 tests pass without API key (unit + integration + deterministic eval) + 14 live eval = **291 total tests** (90%+ coverage).

Tests run without `GOOGLE_API_KEY` except `test_eval.py` (live LLM). Deterministic eval tests use VCR-style fixtures with pre-recorded LLM responses.

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/chat` | API key + rate limit | Send message, receive SSE token stream |
| `GET` | `/health` | None | Health check — 200 healthy, 503 degraded (Cloud Run routing) |
| `GET` | `/property` | None | Property metadata (name, categories, doc count) |
| `GET` | `/` | None | Chat UI (static HTML) |

### POST /chat

**Request:**
```json
{
  "message": "What Italian restaurants do you have?",
  "thread_id": "optional-uuid-for-multi-turn"
}
```

**Response:** Server-Sent Events stream with typed events:

```
event: metadata
data: {"thread_id": "abc-123"}

event: token
data: {"content": "Mohegan Sun has "}

event: token
data: {"content": "excellent Italian dining..."}

event: sources
data: {"sources": ["restaurants"]}

event: done
data: {"done": true}
```

**Error responses:**
- `401` — Missing or invalid API key (when `API_KEY` is configured)
- `422` — Invalid request (empty message, non-UUID thread_id)
- `429` — Rate limited (includes `Retry-After` header)
- `503` — Agent not initialized (includes `Retry-After: 30` header)

## Security & Middleware

6 pure ASGI middleware classes (no `BaseHTTPMiddleware` — preserves SSE streaming):

| Middleware | Description |
|-----------|-------------|
| **RequestLogging** | Cloud Logging JSON format, `X-Request-ID`, `X-Response-Time-Ms` |
| **SecurityHeaders** | `nosniff`, `DENY`, CSP, `Referrer-Policy` |
| **ApiKeyAuth** | `X-API-Key` with `hmac.compare_digest` (disabled when `API_KEY` empty) |
| **RequestBodyLimit** | 64 KB max request body to prevent abuse |
| **RateLimit** | Token-bucket per IP, 20 req/min on `/chat` |
| **ErrorHandling** | `CancelledError` at INFO, structured 500 JSON for unhandled errors |

Additional safety:
- **Input auditing**: `audit_input()` regex detects 7 prompt injection patterns pre-LLM
- **Responsible gaming**: `detect_responsible_gaming()` regex detects 22 gambling concern patterns (English, Spanish, Mandarin) deterministically
- **Age verification**: `detect_age_verification()` regex detects 6 underage-related patterns, ensures 21+ requirement is always communicated
- **CORS**: Configurable allowed origins
- **SSE timeout**: Configurable timeout with clean error event on expiry
- **Request cancellation**: SSE stream cancels on client disconnect

## Configuration

All settings are configurable via environment variables (powered by `pydantic-settings`). See `.env.example` for full documentation.

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Google AI API key |
| `API_KEY` | (empty) | API key for `/chat` authentication (disabled when empty) |
| `MODEL_NAME` | `gemini-2.5-flash` | LLM model name |
| `MODEL_TEMPERATURE` | `0.3` | LLM temperature |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model |
| `PROPERTY_NAME` | `Mohegan Sun` | Property name for prompts |
| `PROPERTY_DATA_PATH` | `data/mohegan_sun.json` | Path to property JSON |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence directory |
| `RAG_TOP_K` | `5` | Number of retrieval results |
| `RAG_CHUNK_SIZE` | `800` | Chunk size for text splitting |
| `RAG_CHUNK_OVERLAP` | `120` | Chunk overlap (15% of chunk size) |
| `ALLOWED_ORIGINS` | `["http://localhost:8080"]` | CORS allowed origins |
| `RATE_LIMIT_CHAT` | `20` | Max chat requests per minute per IP |
| `SSE_TIMEOUT_SECONDS` | `60` | Stream timeout |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `ENVIRONMENT` | `development` | Environment name |
| `VERSION` | `0.1.0` | Application version |

## Cost Model

| Traffic | Requests/month | Est. Cost (Gemini 2.5 Flash) |
|---------|---------------|------------------------------|
| Demo | 1,000 | ~$1.40 |
| Medium | 50,000 | ~$70 |
| High | 500,000 | ~$700 |

Per-request: ~$0.0014 (router + generate + validate + embedding).

## Trade-offs and Production Considerations

| Component | Demo | Production |
|-----------|------|------------|
| Vector DB | ChromaDB (in-process) | Vertex AI Vector Search |
| Checkpointing | MemorySaver (lost on restart) | PostgresSaver or FirestoreSaver |
| LLM auth | API key in .env | Vertex AI IAM + GCP Secret Manager |
| Deployment | Docker Compose (local) | Cloud Run |
| Rate limiting | In-memory per-IP | Redis-backed distributed limiter |
| Monitoring | Structured logging | LangSmith + Cloud Monitoring |
| Frontend | Vanilla HTML served by FastAPI | Next.js with React 19 |
| Circuit breaker | In-memory (reset on restart) | Redis-backed with persistence |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph custom `StateGraph` (8 nodes, 2 conditional edges) |
| LLM | Gemini 2.5 Flash via `langchain-google-genai` |
| Embeddings | Google `text-embedding-004` (768 dim) |
| Vector Store | ChromaDB (embedded, persistent) |
| Backend | FastAPI + uvicorn |
| Config | pydantic-settings (20 env-overridable parameters) |
| Frontend | Vanilla HTML/CSS/JS with SSE |
| Container | Docker (multi-stage, non-root, HEALTHCHECK) |
| CI/CD | GCP Cloud Build (4-step: test, build, push, deploy) |
| Python | 3.12 |

## Data

Property data curated February 2026 from public sources (mohegansun.com). Covers 30 items across 7 categories: restaurants, entertainment, hotel rooms, amenities, casino areas, promotions, and FAQ. Mohegan Sun is a tribal casino operated by the Mohegan Tribe of Connecticut. Contact the property directly for current hours and availability.

---

Built by Oded Ben-Yair
