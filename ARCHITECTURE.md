# Architecture Document

**Property Q&A Agent for Mohegan Sun Casino Resort**

---

## 1. System Overview

A conversational AI agent that answers guest questions about a specific casino property. Built with LangGraph for agent orchestration, ChromaDB for vector retrieval, Gemini 2.5 Flash for generation, and FastAPI for the API layer.

**Design philosophy:** Build for one property, design for N. Every configuration choice (property ID, data paths, prompts) is externalized so adding a second property requires zero code changes.

```
Browser (static/index.html)
    |
    | POST /chat (SSE stream)
    v
FastAPI (src/api/app.py)
    |
    | lifespan: init agent + ingest data
    v
LangGraph create_react_agent
    |
    |-- search_property tool ---> ChromaDB
    |                                 |
    v                                 v
Gemini 2.5 Flash               Knowledge Base
(temp=0.3, generation)          (mohegan_sun.json)
```

### Component Responsibilities

| Component | Role | Technology |
|-----------|------|-----------|
| Graph Engine | Agent loop (LLM -> tools -> LLM) | LangGraph `create_react_agent` |
| LLM | Answer generation | Gemini 2.5 Flash (`langchain-google-genai`) |
| Vector Store | Semantic search over property data | ChromaDB (embedded, persistent) |
| Embeddings | Text to vector conversion | Google `text-embedding-004` (768 dim) |
| API Server | HTTP endpoints, SSE streaming | FastAPI + uvicorn |
| Frontend | Chat UI with brand styling | Vanilla HTML/CSS/JS + SSE |
| Container | Isolation, one-command startup | Docker Compose |

---

## 2. Agent Design

The agent uses LangGraph's `create_react_agent` -- a prebuilt ReAct loop that handles the LLM-to-tool-to-LLM cycle automatically. The agent has one tool: `search_property`.

### Why `create_react_agent`

The prebuilt agent provides:
- Automatic tool calling loop (LLM decides when to search)
- Built-in conversation memory via `MemorySaver` checkpointer
- Structured tool result handling
- Clean separation between agent logic and application code

### Agent Flow

```
User message
    |
    v
create_react_agent
    |
    |--> LLM decides: answer directly or search?
    |       |
    |       v (if search needed)
    |   search_property(query)
    |       |
    |       v
    |   ChromaDB similarity_search (top-5)
    |       |
    |       v
    |   Formatted results returned to LLM
    |       |
    |       v
    |   LLM generates grounded response
    |
    v
Final AI response
```

### System Prompt

The concierge prompt establishes:
- **Identity**: Property concierge for Mohegan Sun
- **Scope**: Dining, entertainment, hotel, amenities, gaming, promotions
- **Rules**: Only answer from knowledge base, never fabricate, never provide gambling advice, never claim to make bookings
- **Tone**: Warm luxury hospitality (VIP treatment for every guest)
- **Safety**: Responsible gaming helpline (1-800-522-4700, CT DMHAS 1-860-418-7000), AI disclosure, no betting advice

### Agent State

```python
class PropertyQAState(MessagesState):
    property_name: str = "Mohegan Sun"
```

Inherits `messages: Annotated[list[AnyMessage], add_messages]` from `MessagesState`, which handles message appending and deduplication. The `create_react_agent` prebuilt uses its own internal `AgentState`; `PropertyQAState` documents the expected shape and is available for custom graph builds (e.g., adding compliance or escalation nodes via `StateGraph(PropertyQAState)`).

---

## 3. RAG Pipeline

### Ingestion Flow

```
Property JSON --> Load & Parse --> Chunk (800 chars) --> Embed --> ChromaDB
```

1. **Load**: Read `mohegan_sun.json`, extract items by category (restaurants, entertainment, hotel, etc.)
2. **Format**: Category-specific formatters convert structured data to readable text
3. **Chunk**: `RecursiveCharacterTextSplitter` (800 chars, 100 overlap) for embedding quality
4. **Embed**: Google `text-embedding-004` (768 dimensions, free tier)
5. **Store**: ChromaDB with cosine similarity, persistent to disk

### Retrieval

`search_property` tool queries ChromaDB with `similarity_search(query, k=5)`. Returns formatted results with source category metadata for transparency.

**Why pure vector search (no hybrid):** Sufficient for <500 chunks where entity names appear prominently. For production with thousands of chunks, hybrid search (BM25 + vector) via Vertex AI Vector Search would improve exact name matching.

### Data Model

Property data is a single JSON file with category sections:

| Category | Content |
|----------|---------|
| restaurants | Name, cuisine, hours, price range, location, dress code |
| entertainment | Venues, shows, capacity, schedule |
| hotel_rooms | Room types, tower, features, rate |
| amenities | Spa, pool, golf, shopping, fitness |
| gaming | Casino areas, games offered (no odds/strategy) |
| promotions | Loyalty program, tiers, benefits |
| faq | Common questions with answers |

Metadata on every chunk: `category`, `item_name`, `source`, `chunk_index`.

---

## 4. API Design

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST /chat` | Send message, receive SSE stream | Main conversation endpoint |
| `GET /health` | Agent + ChromaDB status | Docker healthcheck target |
| `GET /property` | Property metadata | Name, categories, document count |
| `GET /` | Chat UI | Static HTML served by FastAPI |

### SSE Streaming

The `/chat` endpoint returns an `EventSourceResponse` (via `sse-starlette`). Currently uses single-event mode: the agent collects the full response via `ainvoke()` before yielding one SSE event with the complete reply, thread ID, and source categories. The frontend handles both single-event and token-by-token streaming, so upgrading to `astream_events()` for real token streaming requires only a backend change.

### Middleware Stack

Pure ASGI middleware (not `BaseHTTPMiddleware`, which breaks SSE):

1. **RequestLoggingMiddleware**: Structured JSON logs, `X-Request-ID`, response timing
2. **ErrorHandlingMiddleware**: Catches unhandled exceptions, returns structured 500 JSON

### Lifespan

On startup:
1. Initialize LangGraph agent (compile graph + checkpointer)
2. Check ChromaDB for existing data; ingest if empty (~30s first boot)
3. Load property metadata for `/property` endpoint

---

## 5. Data: Mohegan Sun

**Why a real property:** Authenticity demonstrates domain understanding. Mohegan Sun provides depth (40+ restaurants, 10K-seat arena, two casinos, spa, golf) and an industry connection (Gaming Analytics is a Hey Seven client).

**Regulatory awareness:** Casino data describes areas factually but excludes odds, house edges, betting strategies, and bet amounts. The FAQ references Connecticut DMHAS self-exclusion (tribal casino jurisdiction) rather than generic Nevada/NJ programs.

**Data provenance:** All data curated from mohegansun.com public pages, February 2026. `source` field on every chunk enables auditing.

---

## 6. Trade-off Tables

### Demo vs. Production

| Component | Demo Choice | Production Upgrade | Migration Cost |
|-----------|-------------|-------------------|----------------|
| Vector DB | ChromaDB (embedded) | Vertex AI Vector Search | Different API, hosting, ingestion |
| Checkpointer | InMemorySaver | PostgresSaver / FirestoreSaver | Swap in `create_agent()` |
| Embeddings | text-embedding-004 (API key) | text-embedding-005 (Vertex AI) | Change model string |
| LLM | Gemini 2.5 Flash (API key) | Gemini via Vertex AI (IAM) | Auth change only |
| Deployment | Docker Compose | Cloud Run | Add `cloudbuild.yaml` |
| Auth | None (demo) | API key + GCP Secret Manager | Add middleware |
| Monitoring | Structured logging | LangSmith + Cloud Monitoring | Add env vars |

### Key Architecture Decisions

| # | Decision | Choice | Trade-off |
|---|----------|--------|-----------|
| 1 | Agent pattern | `create_react_agent` | Less graph-level control, but faster to ship with fewer bugs |
| 2 | Search tool | Single unified tool | LLM decides when to search; every property question hits RAG |
| 3 | Vector DB | ChromaDB | No auth, no backup, in-process memory; perfect for demo |
| 4 | LLM | Gemini 2.5 Flash | GCP-aligned; slightly less reliable structured output than GPT-4o |
| 5 | Frontend | Vanilla HTML/CSS/JS | No build step; no TypeScript safety, no component library |
| 6 | Streaming | SSE (not WebSocket) | Simpler, proxy-friendly, reconnection built-in |

---

## 7. Testing Strategy

### Test Pyramid

| Layer | Scope | LLM |
|-------|-------|-----|
| Unit | Individual functions, data integrity, config | Mocked |
| Integration | Full graph flow, API endpoints, RAG pipeline | Mocked LLM, real ChromaDB |
| Eval | Answer quality, guardrails, hallucination detection | Real Gemini (temp=0) |

### Key Test Areas

- **Router accuracy**: Greetings, property questions, off-topic, gambling advice
- **Retrieval quality**: Correct results for dining/entertainment queries
- **Guardrails**: Gambling advice refusal, booking refusal, AI disclosure
- **Hallucination prevention**: "I don't know" for unknown topics
- **API behavior**: SSE streaming, health check, error responses
- **Edge cases**: Prompt injection, unicode, max-length input

### Running Tests

```bash
pytest tests/ -v                              # All tests (mocked LLM)
pytest tests/ --cov=src --cov-report=term     # With coverage
pytest tests/ -k "integration" -v             # Integration only (needs GOOGLE_API_KEY)
```

---

## 8. Deployment

### Docker

Multi-stage build (`python:3.12-slim`):
- **Stage 1** (builder): Install dependencies with `build-essential`
- **Stage 2** (production): Copy deps + app code, non-root `appuser`, expose 8080

Data ingestion happens at **startup** (FastAPI lifespan), not build time. This avoids baking `GOOGLE_API_KEY` into the image.

### docker-compose.yml

Single service with:
- Health check (60s start period for first-boot ingestion)
- ChromaDB data persisted in a Docker volume
- 2GB memory limit
- Graceful shutdown (10s timeout for in-flight SSE streams)

### Startup Sequence

```
1. docker compose up --build
2. uvicorn starts FastAPI
3. Lifespan: validate env vars (GOOGLE_API_KEY)
4. Lifespan: check ChromaDB for existing data
5. If empty -> ingest property data (~30s)
6. Lifespan: initialize LangGraph agent
7. Health returns 200 -> Docker healthcheck passes
8. System ready at http://localhost:8080
```

### Cloud Run (Production Path)

```
Cloud Build: Test -> Docker Build -> Push to GCR -> Deploy to Cloud Run
```

- `min-instances=1` to avoid cold start latency
- `concurrency=1` (synchronous graph execution)
- Scale via instances, not concurrency
- Secrets via GCP Secret Manager (not env vars)

---

## 9. Cost Model

| Component | Demo | Production (100K queries/month) |
|-----------|------|-------------------------------|
| Gemini 2.5 Flash | Free tier | ~$150/month |
| Embeddings | Free tier | ~$30/month |
| ChromaDB | $0 (embedded) | N/A (Vertex AI: ~$200/mo) |
| Cloud Run | $0 (local) | ~$150/month |
| **Total** | **$0** | **~$530/month** |

---

Built by Oded Ben-Yair | February 2026
