# Hey Seven Property Q&A Agent

An AI-powered property concierge for Mohegan Sun casino resort, built with LangGraph and Gemini 2.5 Flash.

Guests ask natural language questions about dining, entertainment, rooms, amenities, gaming, and promotions. The agent retrieves answers from a curated knowledge base using RAG (Retrieval-Augmented Generation) and streams responses via Server-Sent Events.

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
    | POST /chat (SSE stream)
    v
FastAPI (src/api/app.py)
    |
    v
LangGraph Agent (create_react_agent)
    |
    |-- search_property tool --> ChromaDB (RAG)
    |                               |
    v                               v
Gemini 2.5 Flash            Property Knowledge Base
(generation)                 (data/mohegan_sun.json)
```

1. User sends a message via the chat UI.
2. FastAPI receives the request and invokes the LangGraph agent.
3. The agent decides whether to call the `search_property` tool.
4. The tool queries ChromaDB for relevant property information.
5. Gemini 2.5 Flash generates a grounded response using the retrieved context.
6. The response streams back to the browser via SSE.

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Agent framework | LangGraph `create_react_agent` | Production-grade state machine with checkpointing |
| LLM | Gemini 2.5 Flash | GCP-aligned, cost-effective ($0.30/1M input tokens) |
| Vector DB | ChromaDB (embedded) | Zero infrastructure for demo; Vertex AI Vector Search for production |
| Streaming | SSE over WebSocket | Industry standard for LLM streaming (OpenAI, Anthropic use SSE) |
| Embeddings | Google `text-embedding-004` | GCP-native, free tier, 768 dimensions |
| Frontend | Vanilla HTML/CSS/JS | No build step, minimal footprint, backend is 90% of evaluation |
| Search tool | Single `search_property` | Unified retrieval; LLM reformulates queries naturally |

## Project Structure

```
hey-seven/
├── src/
│   ├── agent/
│   │   ├── graph.py          # LangGraph agent assembly + chat function
│   │   ├── state.py          # PropertyQAState schema
│   │   ├── prompts.py        # System prompt templates
│   │   └── tools.py          # search_property RAG tool
│   ├── rag/
│   │   ├── pipeline.py       # Ingest, chunk, embed, retrieve (ChromaDB)
│   │   └── embeddings.py     # Google text-embedding-004 config
│   └── api/
│       ├── app.py            # FastAPI app, lifespan, SSE streaming
│       ├── models.py         # Pydantic request/response schemas
│       └── middleware.py     # Pure ASGI logging + error handling
├── data/
│   └── mohegan_sun.json      # Curated property data
├── static/
│   └── index.html            # Branded chat UI (Hey Seven colors)
├── tests/                    # Unit, integration, eval tests
├── Dockerfile                # Multi-stage Python 3.12 build
├── docker-compose.yml        # Single-service with health check
└── requirements.txt          # Pinned production dependencies
```

## Running Tests

```bash
# Unit + integration (no API key needed)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Integration tests only (requires GOOGLE_API_KEY)
pytest tests/ -k "integration" -v
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/chat` | None (demo) | Send message, receive SSE stream |
| `GET` | `/health` | None | Health check (agent + ChromaDB status) |
| `GET` | `/property` | None | Property metadata (name, categories, doc count) |
| `GET` | `/` | None | Chat UI (static HTML) |

### POST /chat

**Request:**
```json
{
  "message": "What Italian restaurants do you have?",
  "thread_id": "optional-for-multi-turn"
}
```

**Response:** Server-Sent Events stream with `data` events containing:
```json
{
  "response": "Mohegan Sun has excellent Italian dining...",
  "thread_id": "abc-123",
  "sources": ["restaurants"],
  "done": true
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | - | Google AI API key (for Gemini + embeddings) |
| `PORT` | No | `8080` | Server port |
| `ENVIRONMENT` | No | `development` | Environment name |
| `CHROMA_PERSIST_DIR` | No | `data/chroma` | ChromaDB persistence directory |

## Trade-offs and Production Considerations

This is a demo implementation. Key differences for production:

| Component | Demo | Production |
|-----------|------|------------|
| Vector DB | ChromaDB (in-process) | Vertex AI Vector Search |
| Checkpointing | InMemorySaver (lost on restart) | PostgresSaver or FirestoreSaver |
| LLM auth | API key in .env | Vertex AI IAM + GCP Secret Manager |
| Deployment | Docker Compose (local) | Cloud Run |
| Rate limiting | Not enforced | Per-IP sliding window via ASGI middleware |
| Monitoring | Structured logging | LangSmith + Cloud Monitoring |
| Frontend | Vanilla HTML served by FastAPI | Next.js with React 19 |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph 1.0.x (`create_react_agent`) |
| LLM | Gemini 2.5 Flash via `langchain-google-genai` |
| Embeddings | Google `text-embedding-004` (768 dim) |
| Vector Store | ChromaDB (embedded, persistent) |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla HTML/CSS/JS with SSE |
| Container | Docker (multi-stage, non-root) |
| Python | 3.12 |

## Data

Property data curated February 2026 from public sources (mohegansun.com). Covers restaurants, entertainment, hotel rooms, amenities, casino areas, promotions, and FAQ. Contact Mohegan Sun directly for current hours and availability.

---

Built by Oded Ben-Yair
