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
Browser (static/index.html)
    |
    | POST /chat (SSE: metadata → token* → sources → done)
    v
FastAPI + Security Headers + Rate Limiting + Logging
    |
    v
LangGraph Agent (create_react_agent)
    |
    |-- search_property ---------> ChromaDB (general search)
    |-- get_property_hours ------> ChromaDB (schedule lookup)
    |                                  |
    v                                  v
Gemini 2.5 Flash               Property Knowledge Base
(config-driven)                (data/mohegan_sun.json)
```

1. User sends a message via the chat UI.
2. FastAPI receives the request and invokes the LangGraph agent.
3. The agent decides which tool(s) to call (`search_property` or `get_property_hours`).
4. The tool queries ChromaDB for relevant property information.
5. Gemini 2.5 Flash generates a grounded response using the retrieved context.
6. Tokens stream back to the browser in real time via SSE.

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Agent framework | LangGraph `create_react_agent` | Production-grade state machine with checkpointing |
| LLM | Gemini 2.5 Flash | GCP-aligned, cost-effective ($0.30/1M input tokens) |
| Vector DB | ChromaDB (embedded) | Zero infrastructure for demo; Vertex AI Vector Search for production |
| Streaming | Real token SSE via `astream_events` | True progressive rendering with timeout + error recovery |
| Embeddings | Google `text-embedding-004` | GCP-native, free tier, 768 dimensions |
| Frontend | Vanilla HTML/CSS/JS | No build step, minimal footprint, backend is 90% of evaluation |
| Tools | Two specialized tools | Demonstrates multi-tool orchestration and LLM tool selection |
| Config | `pydantic-settings` BaseSettings | Zero hardcoded values; every constant overridable via env var |

## Project Structure

```
hey-seven/
├── src/
│   ├── config.py              # Centralized config (pydantic-settings)
│   ├── agent/
│   │   ├── graph.py           # Agent assembly + chat + chat_stream
│   │   ├── state.py           # PropertyQAState schema
│   │   ├── prompts.py         # System prompt with few-shot examples
│   │   └── tools.py           # search_property + get_property_hours
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
├── tests/                     # 49 tests (unit + integration)
├── Dockerfile                 # Multi-stage Python 3.12 + HEALTHCHECK
├── docker-compose.yml         # Single-service with health check
└── requirements.txt           # Pinned production dependencies
```

## Running Tests

```bash
# All 49 tests (no API key needed)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Integration tests only (requires GOOGLE_API_KEY)
pytest tests/ -k "integration" -v
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
| Agent Framework | LangGraph 1.0.8 (`create_react_agent`) |
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
