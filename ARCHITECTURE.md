# Architecture

## System Overview

Hey Seven Property Q&A Agent is an AI concierge for Mohegan Sun casino resort. Guests ask natural-language questions about dining, entertainment, hotel rooms, amenities, gaming, and promotions. The agent uses a custom 8-node LangGraph StateGraph with RAG (Retrieval-Augmented Generation) to produce grounded, validated answers streamed token-by-token via Server-Sent Events.

The system has three layers: a vanilla HTML/JS chat frontend, a FastAPI backend with pure ASGI middleware, and a LangGraph agent backed by Gemini 2.5 Flash and ChromaDB.

**Design philosophy:** Build for one property, design for N. Every configuration choice (property name, data paths, model name, prompts) is externalized via `pydantic-settings` so adding a second property requires zero code changes.

```
Browser (static/index.html)
    |
    | POST /chat (SSE: metadata -> token* -> sources -> done)
    v
FastAPI (src/api/app.py)  <-  SecurityHeaders + RateLimit + Logging + ErrorHandler
    |
    | lifespan: build_graph() + ingest data
    v
Custom 8-node StateGraph (src/agent/graph.py)
    |
    |-- router (structured LLM output) -----> greeting / off_topic / retrieve
    |-- retrieve -----> ChromaDB
    |-- generate -----> Gemini 2.5 Flash (grounded response)
    |-- validate -----> Gemini 2.5 Flash (adversarial review)
    |-- respond / fallback / greeting / off_topic -----> END
    |
    v
Gemini 2.5 Flash               Knowledge Base
(config-driven)                (data/mohegan_sun.json -> ChromaDB)
```

---

## Custom StateGraph

The agent is a hand-built `StateGraph` (not `create_react_agent`). Every request flows through an explicit graph of 8 nodes with two conditional routing points.

```
START --> router --+--> greeting ----------------------> END
                   |
                   +--> off_topic --------------------> END
                   |
                   +--> retrieve --> generate --> validate --+--> respond --> END
                                       ^                     |
                                       |                     +--> generate  (retry, max 1)
                                       +---------------------+
                                                             +--> fallback --> END
```

Entry point: `build_graph()` in `src/agent/graph.py` compiles the graph with a checkpointer (defaults to `MemorySaver` for local development).

---

## Node Descriptions

### 1. router (`src/agent/nodes.py:44`)

**Purpose**: Classify user intent into one of 7 categories using structured LLM output.

**Input**: `messages` (conversation history).
**Output**: `query_type` (str), `router_confidence` (float 0-1).

Categories: `property_qa`, `hours_schedule`, `greeting`, `off_topic`, `gambling_advice`, `action_request`, `ambiguous`.

Pre-LLM guardrails: `audit_input()` runs regex-based prompt injection detection (7 patterns) before any LLM call. Detected injections route directly to `off_topic` without invoking the LLM.

Turn-limit guard: if `messages` exceeds 40 entries, forces `off_topic` to end the conversation. Uses `llm.with_structured_output(RouterOutput)` for reliable JSON parsing. On LLM error, defaults to `property_qa` with confidence 0.5.

### 2. retrieve (`src/agent/nodes.py:97`)

**Purpose**: Search the ChromaDB knowledge base for documents relevant to the user query.

**Input**: `messages` (extracts last `HumanMessage`).
**Output**: `retrieved_context` (list of `{content, metadata, score}` dicts).

Routes to `search_hours()` for `hours_schedule` queries (appends schedule-focused keywords), otherwise calls `search_knowledge_base()`. Both delegate to `CasinoKnowledgeRetriever.retrieve_with_scores()` with configurable `top_k` (default 5).

### 3. generate (`src/agent/nodes.py:123`)

**Purpose**: Produce a concierge response grounded in retrieved context.

**Input**: `messages`, `retrieved_context`, `current_time`, `retry_count`, `retry_feedback`.
**Output**: `messages` (appends `AIMessage`), optionally `retry_count`.

Behavior:
- Formats retrieved context as numbered sources and appends to the system prompt.
- If no context was retrieved, returns a static fallback message and sets `retry_count=99` to skip validation.
- On retry (`retry_count > 0`), injects validation feedback as a `SystemMessage` before conversation history.
- On LLM error, returns a static error message and sets `retry_count=99`.

### 4. validate (`src/agent/nodes.py:202`)

**Purpose**: Adversarial review of the generated response against 6 criteria.

**Input**: `messages` (user question + generated response), `retrieved_context`, `retry_count`.
**Output**: `validation_result` (PASS/RETRY/FAIL), optionally `retry_count`, `retry_feedback`.

6 criteria: grounded, on-topic, no gambling advice, read-only, accurate, responsible gaming.

Behavior:
- If `retry_count >= 99` (empty context or generate error), auto-PASS.
- If validation fails and `retry_count < 1`, returns RETRY with feedback.
- If `retry_count >= 1`, returns FAIL (max 1 retry).
- On validation LLM error, auto-PASS.

### 5. respond (`src/agent/nodes.py:277`)

**Purpose**: Extract source categories from retrieved context and prepare final response.

**Input**: `retrieved_context`.
**Output**: `sources_used` (deduplicated list of category strings), `retry_feedback` cleared to `None`.

### 6. fallback (`src/agent/nodes.py:300`)

**Purpose**: Safe fallback when validation fails after retry.

**Input**: `retry_feedback`.
**Output**: `messages` (appends static AIMessage with contact info), `sources_used` cleared, `retry_feedback` cleared.

Logs the validation failure reason for observability.

### 7. greeting (`src/agent/nodes.py:326`)

**Purpose**: Return a template welcome message listing available knowledge categories.

**Input**: None (uses `PROPERTY_NAME` from settings).
**Output**: `messages` (appends welcome AIMessage), `sources_used` cleared.

### 8. off_topic (`src/agent/nodes.py:351`)

**Purpose**: Handle off-topic queries, gambling advice requests, and action requests.

**Input**: `query_type`.
**Output**: `messages` (appends appropriate redirect AIMessage), `sources_used` cleared.

Three sub-cases based on `query_type`:
- `off_topic`: General redirect to property topics.
- `gambling_advice`: Redirect with responsible gaming helplines (NCPG 1-800-522-4700, CT Council 1-888-789-7777, CT DMHAS 1-860-418-7000).
- `action_request`: Explain read-only limitations, provide contact info.

---

## State Schema

`PropertyQAState` is a `TypedDict` with 9 fields (`src/agent/state.py:12`):

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `Annotated[list, add_messages]` | Conversation history (LangGraph message reducer) |
| `query_type` | `str \| None` | Router classification (7 categories) |
| `router_confidence` | `float` | Router confidence score (0.0-1.0) |
| `retrieved_context` | `list[dict]` | Retrieved documents: `{content, metadata, score}` |
| `validation_result` | `str \| None` | Validation outcome: PASS, FAIL, or RETRY |
| `retry_count` | `int` | Current retry count (max 1 before fallback) |
| `retry_feedback` | `str \| None` | Reason validation failed |
| `current_time` | `str` | UTC timestamp injected at graph entry |
| `sources_used` | `list[str]` | Knowledge-base categories cited in the response |

Two Pydantic models for structured LLM output:

- **`RouterOutput`** (`state.py:25`): `query_type` (`Literal` constrained to 7 valid categories) + `confidence` (float, 0.0-1.0).
- **`ValidationResult`** (`state.py:36`): `status` (`Literal["PASS", "FAIL"]`) + `reason` (str). The node logic handles RETRY semantics based on `retry_count`.

---

## Routing Logic

### route_from_router (`src/agent/nodes.py:403`)

Called after the `router` node. Returns the next node name:

| Condition | Next Node |
|-----------|-----------|
| `query_type == "greeting"` | `greeting` |
| `query_type in ("off_topic", "gambling_advice", "action_request")` | `off_topic` |
| `router_confidence < 0.3` | `off_topic` (low-confidence catch-all) |
| Everything else (`property_qa`, `hours_schedule`, `ambiguous`) | `retrieve` |

### route_after_validate (`src/agent/nodes.py:424`)

Called after the `validate` node. Returns the next node name:

| Condition | Next Node |
|-----------|-----------|
| `validation_result == "PASS"` | `respond` |
| `validation_result == "RETRY"` | `generate` (loops back for re-generation) |
| `validation_result == "FAIL"` | `fallback` |

---

## Prompt System

Three `string.Template` prompts in `src/agent/prompts.py`:

### CONCIERGE_SYSTEM_PROMPT

Variables: `$property_name`, `$current_time`.

The main system prompt defining the concierge persona. Contains 10 rules:
1. Only answer about the property
2. Information-only (no bookings/reservations)
3. Always search knowledge base first
4. Warm luxury hospitality tone
5. Be honest about gaps
6. Disclaim hours/prices may vary
7. No gambling advice
8. Transparent about being AI
9. Responsible gaming helplines
10. Time-aware answers using injected `$current_time`

Includes prompt injection defense and responsible gaming helpline information.

### ROUTER_PROMPT

Variables: `$user_message`.

Classifies the user message into one of 7 categories with a confidence score. Requests structured JSON output: `{"query_type": "<category>", "confidence": <float>}`.

### VALIDATION_PROMPT

Variables: `$user_question`, `$retrieved_context`, `$generated_response`.

Adversarial review prompt checking 6 criteria: grounded, on-topic, no gambling advice, read-only, accurate, responsible gaming. Includes PASS and FAIL examples for calibration. Returns `{"status": "<PASS|FAIL|RETRY>", "reason": "<explanation>"}`.

---

## Guardrails

### Deterministic: audit_input (`src/agent/nodes.py`)

Pre-LLM regex-based prompt injection detection. Runs before the router LLM call. Checks 7 patterns (e.g., "ignore previous instructions", "system:", "DAN mode", "pretend you are"). Detected injections are logged and routed directly to `off_topic` without invoking any LLM.

### LLM-based: validate node

Post-generation adversarial review against 6 criteria (see Prompt System above). Catches hallucination, off-topic drift, gambling advice, unauthorized actions. Max 1 retry, then fallback with contact info.

---

## RAG Pipeline

`src/rag/pipeline.py` handles ingestion and retrieval.

### Ingestion

```
mohegan_sun.json --> Parse by category --> Format --> Chunk (800/100) --> Embed --> ChromaDB
```

1. **Load**: Read `data/mohegan_sun.json` (configurable path).
2. **Parse**: Extract items by category. Flatten nested dicts (hotel towers, gaming sub-areas).
3. **Format**: Category-specific formatters for restaurants, entertainment, hotel rooms; generic formatter for others.
4. **Chunk**: `RecursiveCharacterTextSplitter` (800 chars, 100 overlap, separators: `\n\n`, `\n`, `. `, ` `).
5. **Embed**: Google `text-embedding-004` (768 dimensions).
6. **Store**: ChromaDB collection `property_knowledge`, persistent to disk.

Ingestion runs at FastAPI startup (lifespan) if the ChromaDB directory does not exist. First boot takes ~30 seconds.

### Retrieval

Two plain functions in `src/agent/tools.py` (no `@tool` decorators):

- **`search_knowledge_base(query, top_k=5)`**: General semantic search. Returns `list[dict]` with keys: `content`, `metadata` (category, item_name, source), `score`.
- **`search_hours(venue_name, top_k=5)`**: Appends "hours schedule open close" to the query for schedule-specific retrieval.

Both use the global `CasinoKnowledgeRetriever` singleton which wraps ChromaDB `similarity_search_with_score()`.

### Data Model

Property data is a single JSON file with category sections:

| Category | Content |
|----------|---------|
| restaurants | Name, cuisine, hours, price range, location, dress code |
| entertainment | Venues, shows, capacity, schedule |
| hotel | Room types, towers, amenities, rates |
| amenities | Spa, pool, golf, shopping, fitness |
| gaming | Casino areas, games offered (no odds/strategy) |
| promotions | Loyalty program, tiers, benefits |
| faq | Common questions with answers |

Metadata on every chunk: `category`, `item_name`, `source`, `property_id`, `last_updated`, `chunk_index`.

---

## SSE Streaming

`chat_stream()` in `src/agent/graph.py:147` uses `graph.astream_events(version="v2")`.

### Event Types

| Event | When | Payload |
|-------|------|---------|
| `metadata` | First event | `{"thread_id": "uuid"}` |
| `token` | During `generate` node | `{"content": "..."}` (incremental text chunk) |
| `replace` | After `greeting`, `off_topic`, or `fallback` node | `{"content": "..."}` (full response) |
| `sources` | After stream completes (if any) | `{"sources": ["restaurants", ...]}` |
| `done` | Always last | `{"done": true}` |
| `error` | On exception | `{"error": "message"}` |

Token streaming uses `on_chat_model_stream` events filtered to the `generate` node only. Non-streaming nodes (`greeting`, `off_topic`, `fallback`) emit `replace` events with the full response via `on_chain_end`.

The `/chat` endpoint wraps `chat_stream()` in an `EventSourceResponse` with a configurable timeout (default 60s via `SSE_TIMEOUT_SECONDS`).

---

## API Endpoints

Defined in `src/api/app.py`.

### POST /chat

Send a message, receive an SSE token stream.

**Request body** (Pydantic-validated in `src/api/models.py`):
- `message` (str, 1-4096 chars, required)
- `thread_id` (UUID string, optional -- auto-generated if omitted)

**Response**: `EventSourceResponse` with typed SSE events.

**Error responses**: 422 (validation), 429 (rate limited), 503 (agent not initialized).

### GET /health

```json
{
  "status": "healthy | degraded",
  "version": "0.1.0",
  "agent_ready": true,
  "property_loaded": true
}
```

### GET /property

```json
{
  "name": "Mohegan Sun",
  "location": "Uncasville, CT",
  "categories": ["restaurants", "entertainment", ...],
  "document_count": 42
}
```

### GET /

Static file serving (chat UI from `static/` directory).

---

## Middleware Stack

Four pure ASGI middleware classes in `src/api/middleware.py` (no `BaseHTTPMiddleware`, which breaks SSE streaming):

| Middleware | Purpose |
|------------|---------|
| `RequestLoggingMiddleware` | Structured JSON access logs (Cloud Logging compatible), `X-Request-ID` injection, `X-Response-Time-Ms` header |
| `SecurityHeadersMiddleware` | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, CSP, `Referrer-Policy` |
| `RateLimitMiddleware` | Token-bucket per client IP on `/chat` only (default: 20 req/min). Returns 429 with `Retry-After`. `/health` and static files exempt |
| `ErrorHandlingMiddleware` | Catches unhandled exceptions, returns structured 500 JSON. `CancelledError` (SSE client disconnect) logged at INFO, not ERROR |

---

## Testing Strategy

### Test Pyramid

| Layer | Scope | LLM | Command |
|-------|-------|-----|---------|
| Unit | Individual functions, state, config, middleware | Mocked | `make test-ci` |
| Integration | Full graph flow, API endpoints, RAG pipeline | Mocked LLM, real ChromaDB | `make test-ci` |
| Eval | Answer quality, guardrails, hallucination detection | Real Gemini (temp=0) | `make test-eval` |

### Makefile Targets

| Target | Command |
|--------|---------|
| `make test-ci` | `pytest tests/ -v --tb=short -x --ignore=tests/test_eval.py` |
| `make test-eval` | `pytest tests/test_eval.py -v --tb=short` |
| `make lint` | `ruff check src/ tests/` |
| `make run` | `uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload` |
| `make docker-up` | `docker compose up --build` |
| `make smoke-test` | `curl /health` |
| `make ingest` | Run RAG ingestion manually |

---

## Trade-offs

### Why Custom StateGraph vs create_react_agent

| Aspect | Custom StateGraph (chosen) | create_react_agent |
|--------|---------------------------|-------------------|
| Validation loop | Built-in: generate -> validate -> retry/fallback | Not available without wrapping |
| Routing control | Explicit 7-category router with confidence threshold | LLM decides tool calls implicitly |
| Time awareness | `current_time` injected into state at entry | Must be added to system prompt manually |
| Domain guardrails | Dedicated off_topic, gambling_advice, action_request paths | Single system prompt, no structured routing |
| Observability | Each node is individually traceable in LangSmith | Tool calls are traceable but routing is opaque |
| Code complexity | More code (8 nodes, 2 routing functions) | ~10 lines to set up |
| Flexibility | Full control over retry logic, validation criteria | Simpler but less control |

### Demo vs Production

| Component | Demo | Production |
|-----------|------|------------|
| Vector DB | ChromaDB (in-process, persistent) | Vertex AI Vector Search |
| Checkpointing | MemorySaver (lost on restart) | FirestoreSaver |
| LLM auth | API key in `.env` | Vertex AI IAM + GCP Secret Manager |
| Deployment | Docker Compose (local) | Cloud Run |
| Rate limiting | In-memory per-IP dict | Redis-backed distributed limiter |
| Monitoring | Structured logging | LangSmith + Cloud Monitoring |

---

## Configuration

All settings in `src/config.py` using `pydantic-settings`. Every value is overridable via environment variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Google AI API key (declared in Settings) |
| `PROPERTY_NAME` | `Mohegan Sun` | Property name used in prompts |
| `PROPERTY_DATA_PATH` | `data/mohegan_sun.json` | Path to property JSON |
| `MODEL_NAME` | `gemini-2.5-flash` | LLM model |
| `MODEL_TEMPERATURE` | `0.3` | LLM temperature |
| `MODEL_TIMEOUT` | `30` | LLM call timeout (seconds) |
| `MODEL_MAX_RETRIES` | `2` | LLM retry count on failure |
| `MODEL_MAX_OUTPUT_TOKENS` | `2048` | Max response tokens |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model (768 dim) |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence directory |
| `RAG_TOP_K` | `5` | Number of retrieval results |
| `RAG_CHUNK_SIZE` | `800` | Text chunk size (characters) |
| `RAG_CHUNK_OVERLAP` | `100` | Chunk overlap (characters) |
| `ALLOWED_ORIGINS` | `["http://localhost:8080"]` | CORS allowed origins |
| `RATE_LIMIT_CHAT` | `20` | Max chat requests per minute per IP |
| `SSE_TIMEOUT_SECONDS` | `60` | Stream timeout (seconds) |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `ENVIRONMENT` | `development` | Environment name |
| `VERSION` | `0.1.0` | Application version |

---

## Deployment

### Docker

Multi-stage build (`Dockerfile`):
1. **Builder stage**: Python 3.12-slim, installs dependencies to `/build/deps`.
2. **Production stage**: Python 3.12-slim, non-root `appuser`, copies deps and application code.

RAG ingestion runs at FastAPI startup (lifespan), not build time, so `GOOGLE_API_KEY` is not baked into the image.

HEALTHCHECK is defined in `docker-compose.yml` (single source of truth), not in the Dockerfile, to allow per-environment tuning.

### Docker Compose

Single-service setup with:
- `.env` file for secrets (`GOOGLE_API_KEY`).
- Named volume `chroma_data` for ChromaDB persistence across restarts.
- 2GB memory limit, 60s start period for initial embedding.

### Cloud Build

`cloudbuild.yaml` defines a 4-step CI/CD pipeline:
1. Install dependencies and run tests (`pytest`, ignoring eval tests).
2. Build Docker image tagged with commit SHA.
3. Push to Google Container Registry.
4. Deploy to Cloud Run (us-central1, 512Mi memory, 60s timeout).

---

## Scope Decisions

The following features from the initial architecture specification (`assignment/architecture.md`) were consciously deferred. Each is a production-readiness enhancement, not a demo requirement.

| Feature | Status | Rationale |
|---------|--------|-----------|
| API key authentication (`X-API-Key` + `hmac.compare_digest`) | Deferred | Adds complexity for a demo; rate limiting provides sufficient protection |
| Circuit breaker pattern | Deferred | Requires external state (Redis); overkill for single-instance demo |
| Per-category data files (8 JSON files) | Single JSON | Simpler ingestion; per-category files are a multi-property scaling concern |
| Pydantic validation of data files | Deferred | Runtime validation at ingestion is sufficient; schema enforcement adds maintenance burden |
| `structlog` structured logging | Standard `logging` | Cloud Logging compatible JSON emitted by middleware; `structlog` is a luxury, not a necessity |
| nginx frontend container | Deferred | FastAPI serves static files directly; nginx adds container orchestration complexity |
| Multi-property system (`get_property_config()`) | Single property | Demo targets one property; the config externalization enables multi-property with zero code changes |

All deferred features have clear production paths documented in the Trade-offs section above.

---

Built by Oded Ben-Yair | February 2026
