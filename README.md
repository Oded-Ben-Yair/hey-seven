# Hey Seven Property Q&A Agent

An AI-powered property concierge for Mohegan Sun casino resort, built with LangGraph and Gemini 2.5 Flash.

Guests ask natural language questions about dining, entertainment, rooms, amenities, gaming, and promotions. The agent retrieves answers from a curated knowledge base using RAG (Retrieval-Augmented Generation) and streams responses token-by-token via Server-Sent Events.

## Quick Start

```bash
# 1. Clone and configure
git clone <repo-url>
cd hey-seven
cp .env.example .env   # Add your GOOGLE_API_KEY

# 2. Run with Docker
docker compose up --build

# 3. Open browser
open http://localhost:8080
```

The first boot takes ~30 seconds to embed property data into ChromaDB. Subsequent restarts are fast (data persists in a Docker volume).

## Architecture

```
START --> router --+--> greeting --> END
                   +--> off_topic --> END
                   +--> retrieve --> generate --> validate --+--> respond --> END
                                       ^                     +--> generate (retry)
                                       +---------------------+--> fallback --> END
```

A custom 8-node LangGraph `StateGraph` with two conditional routing points:

1. **Router** classifies user intent (7 categories) using structured LLM output.
2. **Retrieve** searches ChromaDB for relevant property knowledge.
3. **Generate** produces a grounded response using the concierge system prompt.
4. **Validate** performs adversarial review against 6 criteria (grounded, on-topic, no gambling advice, read-only, accurate, responsible gaming).
5. On validation failure, the graph retries once, then falls back to a safe contact-info response.

Tokens stream to the browser in real time via SSE (`astream_events` v2).

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details: node descriptions, state schema, routing logic, prompt system, and deployment.

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Agent framework | LangGraph custom `StateGraph` (8 nodes) | Validation loop, structured routing, domain guardrails |
| LLM | Gemini 2.5 Flash | GCP-aligned, cost-effective ($0.30/1M input tokens) |
| Vector DB | ChromaDB (embedded) | Zero infrastructure for demo; Vertex AI Vector Search for production |
| Streaming | Real token SSE via `astream_events` v2 | True progressive rendering with timeout + error recovery |
| Embeddings | Google `text-embedding-004` | GCP-native, free tier, 768 dimensions |
| Frontend | Vanilla HTML/CSS/JS | No build step, minimal footprint, backend is 90% of evaluation |
| Validation | Adversarial LLM review (6 criteria) | Catches hallucination, off-topic drift, gambling advice leaks |
| Config | `pydantic-settings` BaseSettings | Zero hardcoded values; every constant overridable via env var |

## Project Structure

```
hey-seven/
├── src/
│   ├── config.py              # Centralized config (pydantic-settings)
│   ├── agent/
│   │   ├── graph.py           # 8-node StateGraph + chat + chat_stream (SSE)
│   │   ├── nodes.py           # 8 node functions + 2 routing functions
│   │   ├── state.py           # PropertyQAState (9 fields) + Pydantic models
│   │   ├── prompts.py         # 3 prompt templates (concierge, router, validation)
│   │   └── tools.py           # search_knowledge_base + search_hours (plain functions)
│   ├── rag/
│   │   ├── pipeline.py        # Ingest, chunk, embed, retrieve (ChromaDB)
│   │   └── embeddings.py      # Google text-embedding-004 config
│   └── api/
│       ├── app.py             # FastAPI app, lifespan, SSE streaming
│       ├── models.py          # Pydantic request/response + SSE event schemas
│       └── middleware.py      # Logging + errors + security headers + rate limit
├── data/
│   └── mohegan_sun.json       # Curated property data
├── static/
│   └── index.html             # Branded chat UI (Hey Seven colors)
├── tests/                     # Unit + integration + eval tests
├── Dockerfile                 # Multi-stage Python 3.12 + HEALTHCHECK
├── docker-compose.yml         # Single-service with health check
└── requirements.txt           # Pinned production dependencies
```

## Running Tests

```bash
# Unit + integration tests (no API key needed, LLM is mocked)
make test-ci

# Eval tests (requires GOOGLE_API_KEY, uses real Gemini)
make test-eval

# With coverage
pytest tests/ --cov=src --cov-report=term-missing --ignore=tests/test_eval.py

# Lint
make lint
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/chat` | Rate limited | Send message, receive SSE token stream |
| `GET` | `/health` | None | Health check (agent + ChromaDB status) |
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
- `422` — Invalid request (empty message, non-UUID thread_id)
- `429` — Rate limited (includes `Retry-After` header)
- `503` — Agent not initialized (includes `Retry-After: 30` header)

## Security

- **Rate limiting**: Token-bucket per client IP, 20 requests/min on `/chat`
- **Security headers**: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, CSP, `Referrer-Policy`
- **CORS**: Configurable allowed origins (default: `localhost:8080`)
- **Input validation**: Message length limits, UUID pattern for thread_id
- **SSE timeout**: Configurable timeout with clean error event on expiry

## Configuration

All settings are configurable via environment variables (powered by `pydantic-settings`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Google AI API key |
| `MODEL_NAME` | `gemini-2.5-flash` | LLM model name |
| `MODEL_TEMPERATURE` | `0.3` | LLM temperature |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model |
| `PROPERTY_NAME` | `Mohegan Sun` | Property name for prompts |
| `PROPERTY_DATA_PATH` | `data/mohegan_sun.json` | Path to property JSON |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence directory |
| `RAG_TOP_K` | `5` | Number of retrieval results |
| `RAG_CHUNK_SIZE` | `800` | Chunk size for text splitting |
| `ALLOWED_ORIGINS` | `["http://localhost:8080"]` | CORS allowed origins |
| `RATE_LIMIT_CHAT` | `20` | Max chat requests per minute per IP |
| `SSE_TIMEOUT_SECONDS` | `60` | Stream timeout |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `ENVIRONMENT` | `development` | Environment name |
| `VERSION` | `0.1.0` | Application version |

## Trade-offs and Production Considerations

| Component | Demo | Production |
|-----------|------|------------|
| Vector DB | ChromaDB (in-process) | Vertex AI Vector Search |
| Checkpointing | InMemorySaver (lost on restart) | PostgresSaver or FirestoreSaver |
| LLM auth | API key in .env | Vertex AI IAM + GCP Secret Manager |
| Deployment | Docker Compose (local) | Cloud Run |
| Rate limiting | In-memory per-IP | Redis-backed distributed limiter |
| Monitoring | Structured logging | LangSmith + Cloud Monitoring |
| Frontend | Vanilla HTML served by FastAPI | Next.js with React 19 |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph custom `StateGraph` (8 nodes, 2 conditional edges) |
| LLM | Gemini 2.5 Flash via `langchain-google-genai` |
| Embeddings | Google `text-embedding-004` (768 dim) |
| Vector Store | ChromaDB (embedded, persistent) |
| Backend | FastAPI + uvicorn |
| Config | pydantic-settings |
| Frontend | Vanilla HTML/CSS/JS with SSE |
| Container | Docker (multi-stage, non-root, HEALTHCHECK) |
| Python | 3.12 |

## Data

Property data curated February 2026 from public sources (mohegansun.com). Covers restaurants, entertainment, hotel rooms, amenities, casino areas, promotions, and FAQ. Contact Mohegan Sun directly for current hours and availability.

---

Built by Oded Ben-Yair
