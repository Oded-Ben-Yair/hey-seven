# Hey Seven Property Q&A Agent

An AI concierge for Mohegan Sun casino resort, built with a custom 8-node LangGraph StateGraph.

![Screenshot](docs/screenshot-desktop.png)

## Quick Start

```bash
git clone https://github.com/Oded-Ben-Yair/hey-seven.git
cd hey-seven
cp .env.example .env   # Add your GOOGLE_API_KEY
docker compose up --build
# Open http://localhost:8080
```

## What I Built & Why

Casino guests need quick, reliable answers about dining, entertainment, rooms, and amenities. This agent retrieves answers from a curated knowledge base using RAG and streams responses token-by-token via Server-Sent Events.

I chose to build a **custom 8-node StateGraph** rather than using `create_react_agent` because the casino domain requires deterministic guardrails (responsible gaming, prompt injection, BSA/AML compliance) that must fire before the LLM — not as afterthoughts. The graph-native validation loop (generate → validate → retry/fallback) gives me control that a generic ReAct loop cannot.

The frontend includes a **real-time graph trace panel** that visualizes LangGraph node execution with timing — every query shows which nodes fired, how long each took, and what metadata they produced.

## Architecture

```
START ──> router ──┬──> greeting ──────────────────────────> END
                   ├──> off_topic ─────────────────────────> END
                   └──> retrieve ──> generate ──> validate ─┬──> respond ──> END
                                        ^                   ├──> generate (retry, max 1)
                                        └───────────────────┘
                                                            └──> fallback ──> END
```

**8 nodes, 2 conditional routing points:**

1. **Router** — Classifies user intent (7 categories) using `.with_structured_output(RouterOutput)`.
2. **Retrieve** — Searches ChromaDB with multi-strategy RRF reranking (semantic + augmented queries).
3. **Generate** — Produces a grounded response using the 11-rule concierge system prompt.
4. **Validate** — Adversarial LLM review against 6 criteria (grounded, on-topic, no gambling advice, read-only, accurate, responsible gaming).
5. **Respond** — Extracts sources, packages final response.
6. **Greeting** — Returns "Seven" persona welcome with property categories.
7. **Off-Topic** — Deterministic guardrail responses (no LLM call).
8. **Fallback** — Safe contact-info response when validation exhausts retries.

## LangGraph Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| Custom 8-node StateGraph | `graph.py` | Full control vs `create_react_agent` — validation loops, deterministic guardrails |
| Structured output routing | `router_node` | `.with_structured_output(RouterOutput)` — no string parsing |
| Conditional edges with functions | `route_from_router`, `route_after_validate` | Explicit, testable routing logic |
| Graph-native retry loop | validate → generate | Not Python retry — graph-level state loop with counter |
| State schema with reducers | `Annotated[list, add_messages]` | Proper LangGraph state management |
| `astream_events` v2 | `chat_stream()` | Most advanced streaming API with per-node event filtering |
| Dual LLM strategy | Generator (0.3) vs Validator (0.0) | Different temperatures for creativity vs strictness |
| HITL interrupt support | `interrupt_before` config | Production pattern for regulated environments |
| Real-time graph visualization | SSE `graph_node` events | Live node execution visible in UI with timing |

## Real-Time Graph Trace

Every chat request emits `graph_node` SSE events alongside content tokens. The frontend graph trace panel shows:

- Which nodes are **active** (gold pulse), **complete** (solid gold + timing), or **skipped**
- Per-node metadata: router classification + confidence, doc count, validation result, sources
- Total pipeline timing from router to respond

This is visible via the "Graph Trace" button in the bottom-right corner.

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Agent framework | LangGraph custom `StateGraph` (8 nodes) | Validation loop, structured routing, domain guardrails |
| LLM | Gemini 2.5 Flash | GCP-aligned, cost-effective (~$0.0014/request) |
| Vector DB | ChromaDB (embedded) | Zero infrastructure for demo; Vertex AI Vector Search for production |
| Streaming | Real token SSE via `astream_events` v2 | True progressive rendering with timeout + disconnect detection |
| Embeddings | Google `text-embedding-004` | GCP-native, free tier, 768 dimensions |
| Frontend | Single-file HTML/CSS/JS | No build step, minimal footprint, ships with FastAPI |
| Validation | Adversarial LLM review (6 criteria) | Catches hallucination, off-topic drift, gambling advice leaks |
| Config | `pydantic-settings` BaseSettings | 31 env-overridable settings, zero hardcoded values |
| Retrieval | Multi-strategy RRF reranking | Reciprocal Rank Fusion of semantic + augmented queries, hash-based dedup |
| Ingestion | Idempotent with deterministic IDs | SHA-256 content+source hash prevents duplicates on re-ingestion |
| Guardrails | Deterministic regex + LLM validation | Pre-LLM `audit_input()` blocks injection; 5 guardrails, 56 patterns, 3 languages |

## Safety & Guardrails

5 deterministic pre-LLM guardrails with 56 regex patterns across 3 languages (English, Spanish, Mandarin):

| Guardrail | Patterns | Trigger |
|-----------|----------|---------|
| Prompt Injection | 7 | Jailbreak attempts, system prompt extraction |
| Responsible Gaming | 22 | Problem gambling concerns, self-exclusion requests |
| Age Verification | 6 | Underage access, minimum age questions |
| BSA/AML Compliance | 10 | Money laundering, structuring, CTR/SAR evasion |
| Patron Privacy | 11 | PII requests about other guests, player tracking queries |

All guardrails fire **before** any LLM call and return deterministic responses with appropriate helpline numbers.

## Testing

```bash
make test-ci       # 368 tests, no API key needed
make test-eval     # 14 live eval tests (requires GOOGLE_API_KEY)
make lint          # ruff + mypy
```

**368 tests** across 4 layers:

| Layer | Tests | Description |
|-------|-------|-------------|
| Unit | ~200 | Nodes, guardrails, config, middleware, models |
| Integration | ~100 | Graph execution, API endpoints, SSE streaming |
| Deterministic Eval | 12 | Pre-recorded LLM fixtures, assertion-based |
| Live Eval | 14 | Real Gemini API calls, quality scoring |

**95%+ coverage**, lint clean.

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/chat` | API key + rate limit | Send message, receive SSE token stream |
| `GET` | `/health` | None | Health check (200 healthy, 503 degraded) |
| `GET` | `/property` | None | Property metadata (name, categories, doc count) |
| `GET` | `/graph` | None | Graph structure (nodes and edges for visualization) |
| `GET` | `/` | None | Chat UI |

### SSE Event Types

```
event: metadata     — {"thread_id": "..."}
event: graph_node   — {"node": "router", "status": "start"}
event: graph_node   — {"node": "router", "status": "complete", "duration_ms": 450, "metadata": {"query_type": "property_qa", "confidence": 0.95}}
event: token        — {"content": "Mohegan Sun has "}
event: sources      — {"sources": ["restaurants"]}
event: replace      — {"content": "..."} (greeting/off_topic — replaces streaming)
event: error        — {"error": "..."} (mid-stream failures)
event: done         — {"done": true}
```

## Security & Middleware

6 pure ASGI middleware classes (no `BaseHTTPMiddleware` — preserves SSE streaming):

| Middleware | Description |
|-----------|-------------|
| RequestLogging | Cloud Logging JSON format, `X-Request-ID`, `X-Response-Time-Ms` |
| SecurityHeaders | `nosniff`, `DENY`, CSP, `Referrer-Policy` |
| ApiKeyAuth | `X-API-Key` with `hmac.compare_digest` (disabled when empty) |
| RequestBodyLimit | 64 KB max request body |
| RateLimit | Token-bucket per IP, 20 req/min on `/chat` |
| ErrorHandling | `CancelledError` at INFO, structured 500 JSON |

## Configuration

All 31 settings configurable via environment variables (`pydantic-settings`). See [.env.example](.env.example) for full documentation.

## Cost Model

| Traffic | Requests/month | Est. Cost (Gemini 2.5 Flash) |
|---------|---------------|------------------------------|
| Demo | 1,000 | ~$1.40 |
| Medium | 50,000 | ~$70 |
| High | 500,000 | ~$700 |

Per-request: ~$0.0014 (router + generate + validate + embedding).

## Trade-offs I'd Revisit

| Component | Current | Production Alternative |
|-----------|---------|----------------------|
| Vector DB | ChromaDB (in-process) | Vertex AI Vector Search |
| Checkpointing | MemorySaver (lost on restart) | PostgresSaver or FirestoreSaver |
| Rate limiting | In-memory per-IP | Redis-backed distributed limiter |
| Monitoring | Structured logging | LangSmith + Cloud Monitoring |
| Frontend | Single HTML file | Next.js with React 19 |
| Circuit breaker | In-memory (reset on restart) | Redis-backed with persistence |

## Project Structure

```
hey-seven/
├── src/
│   ├── config.py              # 31 env-overridable settings (pydantic-settings)
│   ├── agent/
│   │   ├── graph.py           # 8-node StateGraph + chat + chat_stream (SSE)
│   │   ├── nodes.py           # 8 nodes + 2 routers + guardrails + circuit breaker
│   │   ├── state.py           # PropertyQAState (9 fields) + Pydantic models
│   │   ├── prompts.py         # 3 prompt templates (11-rule concierge, VIP tone)
│   │   ├── guardrails.py      # 5 deterministic guardrails (56 patterns, 3 languages)
│   │   └── tools.py           # search_knowledge_base + search_hours
│   ├── rag/
│   │   ├── pipeline.py        # Ingest, chunk (800/120), embed, retrieve (ChromaDB)
│   │   └── embeddings.py      # Google text-embedding-004
│   └── api/
│       ├── app.py             # FastAPI app, lifespan, SSE streaming
│       ├── models.py          # Pydantic schemas + SSE event models
│       └── middleware.py      # 6 pure ASGI middleware
├── data/mohegan_sun.json      # 79 items, 7 categories
├── static/
│   ├── index.html             # Branded chat UI (gold/dark/cream)
│   └── assets/                # Custom logo assets
├── tests/                     # 368 tests across 11 files
├── Dockerfile                 # Multi-stage Python 3.12, non-root, HEALTHCHECK
├── docker-compose.yml         # Health check, named volume, 2GB limit
├── cloudbuild.yaml            # GCP Cloud Build CI/CD (4-step pipeline)
└── .env.example               # All 31 env vars documented
```

---

Built by Oded Ben-Yair | February 2026
