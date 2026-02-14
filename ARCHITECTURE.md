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
FastAPI (src/api/app.py)  <-  SecurityHeaders + HSTS + RateLimit + BodyLimit + Auth + Logging + ErrorHandler
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

```mermaid
graph LR
    START((START)) --> router
    router -->|greeting| greeting --> END1((END))
    router -->|off_topic / gambling / action| off_topic --> END2((END))
    router -->|property_qa / hours / ambiguous| retrieve
    retrieve --> generate
    generate --> validate
    validate -->|PASS| respond --> END3((END))
    validate -->|RETRY max 1| generate
    validate -->|FAIL| fallback --> END4((END))
```

<details><summary>ASCII fallback</summary>

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

</details>

Entry point: `build_graph()` in `src/agent/graph.py` compiles the graph with a checkpointer (defaults to `MemorySaver` for local development).

---

## Node Descriptions

### 1. router (`src/agent/nodes.py`)

**Purpose**: Classify user intent into one of 7 categories using structured LLM output.

**Input**: `messages` (conversation history).
**Output**: `query_type` (str), `router_confidence` (float 0-1).

Categories: `property_qa`, `hours_schedule`, `greeting`, `off_topic`, `gambling_advice`, `action_request`, `age_verification`, `patron_privacy`, `ambiguous`.

Pre-LLM guardrails (from `src/agent/guardrails.py`): `audit_input()` runs regex-based prompt injection detection (7 patterns) before any LLM call. Detected injections route directly to `off_topic` without invoking the LLM. `detect_responsible_gaming()` checks 22 patterns (17 English + 5 Spanish + 3 Mandarin) and routes to `gambling_advice` deterministically. `detect_age_verification()` checks 6 patterns for underage-related queries and routes to `age_verification` with the 21+ requirement. `detect_bsa_aml()` checks 10 patterns for money laundering, structuring, and CTR/SAR evasion queries and routes to `off_topic` — casinos are Money Services Businesses under the Bank Secrecy Act and must not provide guidance that could facilitate financial crime. `detect_patron_privacy()` checks 7 patterns for queries about other guests' presence, identity, or membership status — casinos must never disclose whether a specific person is at the property (privacy obligation and liability safeguard against stalking, celebrity harassment, and domestic disputes).

Message-limit guard: if `messages` exceeds `MAX_MESSAGE_LIMIT` (default 40, configurable), forces `off_topic` to end the conversation. The limit counts all messages (human + AI), not just human turns. Uses `llm.with_structured_output(RouterOutput)` for reliable JSON parsing via `ainvoke()` (fully async). On LLM error, defaults to `property_qa` with confidence 0.5.

All 8 node functions are `async def`, using `ainvoke()` for LLM calls. This ensures proper async execution throughout the LangGraph pipeline without blocking the event loop. The `retrieve` node wraps sync ChromaDB calls in `asyncio.to_thread()` since ChromaDB's LangChain wrapper only offers synchronous methods — the production path (Vertex AI Vector Search) has native async APIs.

### 2. retrieve (`src/agent/nodes.py`)

**Purpose**: Search the ChromaDB knowledge base for documents relevant to the user query.

**Input**: `messages` (extracts last `HumanMessage`).
**Output**: `retrieved_context` (list of `{content, metadata, score}` dicts).

Routes to `search_hours()` for `hours_schedule` queries, otherwise calls `search_knowledge_base()`. Both use **multi-strategy retrieval with Reciprocal Rank Fusion (RRF)**:

- `search_knowledge_base()`: Combines (1) direct semantic search with (2) entity-augmented search (`{query} name location details`) for improved proper noun matching
- `search_hours()`: Combines (1) schedule-augmented search (`{query} hours schedule open close`) with (2) direct semantic search for broader venue context
- RRF fusion score: `sum(1/(k + rank))` across rankings, with standard `k=60` dampening. Documents appearing in multiple lists receive a boost.

The ChromaDB collection uses **cosine similarity** (`hnsw:space=cosine`) instead of the default L2 distance, producing normalized scores in [0, 1] where 1.0 = exact match.

Results are filtered by `RAG_MIN_RELEVANCE_SCORE` (default 0.3) — chunks below this cosine similarity threshold are discarded before being passed to the generate node.

### 3. generate (`src/agent/nodes.py`)

**Purpose**: Produce a concierge response grounded in retrieved context.

**Input**: `messages`, `retrieved_context`, `current_time`, `retry_count`, `retry_feedback`.
**Output**: `messages` (appends `AIMessage`), optionally `skip_validation`.

Behavior:
- **Circuit breaker**: Checks `CircuitBreaker.is_open` before building prompts (early exit to avoid wasted work). If open (configurable `CB_FAILURE_THRESHOLD`, default 5 failures within `CB_COOLDOWN_SECONDS`, default 60s), returns a static fallback immediately without invoking the LLM. On success, resets the breaker; on failure, increments the counter. Thread-safe via `asyncio.Lock` to protect concurrent coroutine access to mutable state (`_failure_count`, `_last_failure_time`).
- Formats retrieved context as numbered sources and appends to the system prompt.
- If no context was retrieved, returns a static fallback message and sets `skip_validation=True` to bypass the validator.
- On retry (`retry_count > 0`), injects validation feedback as a `SystemMessage` before conversation history.
- **Message windowing**: Only the last `MAX_HISTORY_MESSAGES` (default 20) human and AI messages are sent to the LLM, bounding context size for long conversations while preserving recent context.
- On LLM error, returns a static error message and sets `skip_validation=True`.

### 4. validate (`src/agent/nodes.py`)

**Purpose**: Adversarial review of the generated response against 6 criteria.

**Input**: `messages` (user question + generated response), `retrieved_context`, `retry_count`, `skip_validation`.
**Output**: `validation_result` (PASS/RETRY/FAIL), optionally `retry_count`, `retry_feedback`.

6 criteria: grounded, on-topic, no gambling advice, read-only, accurate, responsible gaming.

Uses a **separate validator LLM** (`_get_validator_llm()`) with `temperature=0.0` for deterministic binary classification (PASS/RETRY/FAIL). This is distinct from the `_get_llm()` singleton (temperature 0.3) used for creative response generation. Both are `@lru_cache` singletons.

Behavior:
- If `skip_validation` is True (empty context, circuit breaker open, or generate error), auto-PASS (these paths produce deterministic safe responses, not LLM-generated content).
- If validation fails and `retry_count < 1`, returns RETRY with feedback.
- If `retry_count >= 1`, returns FAIL (max 1 retry).
- On validation LLM error: **degraded-pass on first attempt** (retry_count == 0) — if `generate_node` produced a response successfully but the validation LLM is unavailable, the generated response is passed through with a warning log. On retry attempts (retry_count > 0), **fail-closed** (returns FAIL, routes to fallback). This balances availability (generate already succeeded) with safety (retries indicate prior issues).

### 5. respond (`src/agent/nodes.py`)

**Purpose**: Extract source categories from retrieved context and prepare final response.

**Input**: `retrieved_context`.
**Output**: `sources_used` (deduplicated list of category strings), `retry_feedback` cleared to `None`.

### 6. fallback (`src/agent/nodes.py`)

**Purpose**: Safe fallback when validation fails after retry.

**Input**: `retry_feedback`.
**Output**: `messages` (appends static AIMessage with contact info), `sources_used` cleared, `retry_feedback` cleared.

Logs the validation failure reason for observability.

### 7. greeting (`src/agent/nodes.py`)

**Purpose**: Return a template welcome message listing available knowledge categories.

**Input**: None (uses `PROPERTY_NAME` from settings).
**Output**: `messages` (appends welcome AIMessage), `sources_used` cleared.

### 8. off_topic (`src/agent/nodes.py`)

**Purpose**: Handle off-topic queries, gambling advice requests, and action requests.

**Input**: `query_type`.
**Output**: `messages` (appends appropriate redirect AIMessage), `sources_used` cleared.

Five sub-cases based on `query_type`:
- `off_topic`: General redirect to property topics.
- `gambling_advice`: Redirect with responsible gaming helplines (NCPG 1-800-MY-RESET / 1-800-699-7378, CT Council 1-888-789-7777, CT Self-Exclusion ct.gov/selfexclusion via DCP).
- `action_request`: Explain read-only limitations, provide contact info.
- `age_verification`: Provide 21+ age requirement per CT gaming law with property contact info.
- `patron_privacy`: Decline to disclose guest presence/identity/membership with privacy explanation.

---

## State Schema

`PropertyQAState` is a `TypedDict` with 9 fields (`src/agent/state.py:12`):

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `Annotated[list, add_messages]` | Conversation history (LangGraph message reducer) |
| `query_type` | `str \| None` | Router classification (9 categories) |
| `router_confidence` | `float` | Router confidence score (0.0-1.0) |
| `retrieved_context` | `list[dict]` | Retrieved documents: `{content, metadata, score}` |
| `validation_result` | `str \| None` | Validation outcome: PASS, FAIL, or RETRY |
| `retry_count` | `int` | Current retry count (max 1 before fallback) |
| `retry_feedback` | `str \| None` | Reason validation failed |
| `current_time` | `str` | UTC timestamp injected at graph entry |
| `sources_used` | `list[str]` | Knowledge-base categories cited in the response |

Two Pydantic models for structured LLM output:

- **`RouterOutput`** (`state.py:25`): `query_type` (`Literal` constrained to 7 valid categories) + `confidence` (float, 0.0-1.0).
- **`ValidationResult`** (`state.py:36`): `status` (`Literal["PASS", "FAIL", "RETRY"]`) + `reason` (str). RETRY is a first-class schema value, ensuring the LLM can signal minor issues worth correcting versus serious violations (FAIL).

---

## Routing Logic

### route_from_router (`src/agent/nodes.py`)

Called after the `router` node. Returns the next node name:

| Condition | Next Node |
|-----------|-----------|
| `query_type == "greeting"` | `greeting` |
| `query_type in ("off_topic", "gambling_advice", "action_request", "age_verification", "patron_privacy")` | `off_topic` |
| `router_confidence < 0.3` | `off_topic` (low-confidence catch-all) |
| Everything else (`property_qa`, `hours_schedule`, `ambiguous`) | `retrieve` |

### route_after_validate (`src/agent/nodes.py`)

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

The main system prompt defining the concierge persona. Opens with a VIP-oriented Interaction Style section (status-affirming language, energy mirroring, curated suggestions). Contains 11 rules:
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
11. Competitor deflection — never discuss other properties

Includes prompt injection defense and responsible gaming helpline information.

### ROUTER_PROMPT

Variables: `$user_message`.

Classifies the user message into one of 7 categories with a confidence score. Requests structured JSON output: `{"query_type": "<category>", "confidence": <float>}`.

### VALIDATION_PROMPT

Variables: `$user_question`, `$retrieved_context`, `$generated_response`.

Adversarial review prompt checking 6 criteria: grounded, on-topic, no gambling advice, read-only, accurate, responsible gaming. Includes PASS and FAIL examples for calibration. Returns `{"status": "<PASS|FAIL|RETRY>", "reason": "<explanation>"}`.

---

## Guardrails

Four layers: three deterministic (pre-LLM, in `src/agent/guardrails.py`) and one LLM-based (post-generation, in `src/agent/nodes.py`). The deterministic guardrails are in a dedicated module to separate safety concerns from graph node logic.

### Deterministic: audit_input (`src/agent/guardrails.py`)

Pre-LLM regex-based prompt injection detection. Runs before the router LLM call. Checks 7 patterns (e.g., "ignore previous instructions", "system:", "DAN mode", "pretend you are"). Detected injections are logged and routed directly to `off_topic` without invoking any LLM.

### Deterministic: detect_responsible_gaming (`src/agent/guardrails.py`)

Pre-LLM regex-based responsible gaming safety net. Runs after `audit_input` but before the router LLM call. Checks 22 patterns across English (17) and Spanish (5), including: "gambling problem", "addicted to gambling", "self-exclusion", "can't stop gambling", "limit my gambling", "take a break from gambling", "spending too much at the casino", "family says I gamble", "cooling-off period", "want to ban myself", and Spanish equivalents ("problema de juego", "adicción al juego", "juego compulsivo"). Spanish patterns serve the diverse US casino clientele. Detected queries are routed directly to `gambling_advice` (which provides NCPG 1-800-MY-RESET, CT Council 1-888-789-7777, and CT DCP self-exclusion resources) without invoking any LLM. This ensures responsible gaming helplines are always provided deterministically, regardless of LLM routing accuracy.

Responsible gaming helplines are defined as a `RESPONSIBLE_GAMING_HELPLINES` constant in `src/agent/prompts.py` (DRY — used in both the system prompt and the `off_topic_node` response). For multi-property deployment across states, these would be loaded from the property data file.

### Deterministic: detect_age_verification (`src/agent/guardrails.py`)

Pre-LLM regex-based age verification guardrail. Runs after `detect_responsible_gaming` but before the router LLM call. Checks 6 patterns for underage-related queries (e.g., "my kid wants to play", "minimum gambling age", "can underage guests enter", "how old do you have to be to gamble", "minors allowed"). Connecticut law requires casino guests to be 21+ for gaming. Detected queries route to `age_verification` which provides a structured response listing what minors can and cannot do at the property, the 21+ requirement, and the ID requirement. This ensures the legal age requirement is always communicated deterministically, regardless of LLM routing.

### Structural: property_id metadata filter (`src/rag/pipeline.py`)

Every document ingested into ChromaDB includes a `property_id` metadata field derived from `PROPERTY_NAME` (e.g., `"mohegan_sun"`). The retriever filters by this `property_id` on every query, ensuring that only documents from the configured property are returned. This provides a structural grounding guarantee: even if multiple properties share a ChromaDB collection, cross-property data leakage is prevented at the retriever layer — not merely by prompt instructions.

### LLM-based: validate node

Post-generation adversarial review against 6 criteria (see Prompt System above). Catches hallucination, off-topic drift, gambling advice, unauthorized actions. Max 1 retry, then fallback with contact info.

---

## RAG Pipeline

`src/rag/pipeline.py` handles ingestion and retrieval.

### Ingestion

```
mohegan_sun.json --> Parse by category --> Format --> Chunk (800/120) --> Embed --> ChromaDB
```

1. **Load**: Read `data/mohegan_sun.json` (configurable path).
2. **Parse**: Extract items by category. Flatten nested dicts (hotel towers, gaming sub-areas).
3. **Format**: Category-specific formatters for restaurants, entertainment, hotel rooms; generic formatter for others.
4. **Chunk**: `RecursiveCharacterTextSplitter` (800 chars, 120 overlap [15% of chunk size], separators: `\n\n`, `\n`, `. `, ` `). 800 characters is chosen to balance context density vs retrieval precision: casino property items (restaurant descriptions, room details) average 200-400 characters, so 800 chars preserves complete items while allowing the splitter to group related smaller items. Smaller chunks (e.g., 500) would fragment multi-field items; larger chunks (e.g., 1200) would mix unrelated categories.
5. **Embed**: Google `text-embedding-004` (768 dimensions).
6. **Store**: ChromaDB collection `property_knowledge`, persistent to disk.

Ingestion runs at FastAPI startup (lifespan) if the ChromaDB `chroma.sqlite3` file does not exist. First boot takes ~30 seconds. Chunks are stored with **deterministic SHA-256 IDs** (hash of content + source metadata), so re-ingestion is idempotent — restarting the application without clearing ChromaDB will not create duplicate chunks.

### Retrieval

Two plain functions in `src/agent/tools.py` (no `@tool` decorators), both using **multi-strategy retrieval with Reciprocal Rank Fusion (RRF)**:

- **`search_knowledge_base(query)`**: Combines semantic search + entity-augmented query (`{query} name location details`) via RRF. Returns `list[dict]` with keys: `content`, `metadata` (category, item_name, source), `score`.
- **`search_hours(query)`**: Combines schedule-augmented query (`{query} hours schedule open close`) + direct semantic search via RRF.

RRF fusion merges multiple ranked lists using `score = sum(1/(k + rank))` with standard `k=60` dampening. Documents appearing in multiple strategies get boosted, improving recall for entity-heavy queries (e.g., "Todd English's") where different strategies surface different relevant docs. Document deduplication uses a hash of `page_content + source` metadata (MD5) to prevent collision when identical text appears in different categories.

Both use the global `CasinoKnowledgeRetriever` singleton which wraps ChromaDB `similarity_search_with_relevance_scores()`. The collection is configured with `hnsw:space=cosine`, producing cosine similarity scores in [0, 1] where 1.0 = exact match. The `>= RAG_MIN_RELEVANCE_SCORE` (default 0.3) filter applies **after** RRF fusion using the original cosine similarity scores (not RRF rank scores), ensuring absolute semantic relevance regardless of fusion rank.

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

Returns 200 when healthy, 503 when degraded (so Cloud Run / k8s don't route traffic to unhealthy containers).

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

Six pure ASGI middleware classes in `src/api/middleware.py` (no `BaseHTTPMiddleware`, which breaks SSE streaming):

| Middleware | Purpose |
|------------|---------|
| `RequestLoggingMiddleware` | Structured JSON access logs (Cloud Logging compatible), `X-Request-ID` injection, `X-Response-Time-Ms` header |
| `SecurityHeadersMiddleware` | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, CSP, `Referrer-Policy`, `Strict-Transport-Security` (HSTS with 2-year max-age) |
| `ApiKeyMiddleware` | `X-API-Key` header validation on `/chat` using `hmac.compare_digest`. Disabled when `API_KEY` is empty. Returns 401 on mismatch |
| `RateLimitMiddleware` | Sliding-window per client IP on `/chat` only (default: 20 req/min). Respects `X-Forwarded-For` behind reverse proxies (Cloud Run, nginx). Memory-bounded via `RATE_LIMIT_MAX_CLIENTS` cap (default 10,000). Returns 429 with `Retry-After`. `/health` and static files exempt |
| `RequestBodyLimitMiddleware` | Two-layer enforcement: fast-path `Content-Length` header check + streaming byte counting via `receive_wrapper`. Rejects requests exceeding `MAX_REQUEST_BODY_SIZE` (default 64 KB) with 413 Payload Too Large. Prevents resource exhaustion even when `Content-Length` is missing or spoofed |
| `ErrorHandlingMiddleware` | Catches unhandled exceptions, returns structured 500 JSON. `CancelledError` (SSE client disconnect) logged at INFO, not ERROR |

---

## Testing Strategy

### Test Pyramid

| Layer | Scope | LLM | Command |
|-------|-------|-----|---------|
| Unit | Individual functions, state, config, guardrails, middleware | Mocked | `make test-ci` |
| Integration | Full graph flow, API endpoints, RAG pipeline | Mocked LLM, real ChromaDB | `make test-ci` |
| Deterministic Eval | Multi-turn conversations, answer quality, guardrails | VCR fixtures (`_FixtureReplayLLM`) — no API key needed | `make test-ci` |
| Live Eval | Answer quality with real LLM, hallucination detection | Real Gemini (temp=0) | `make test-eval` |

### Deterministic Eval via VCR Fixtures

The `_FixtureReplayLLM` class in `tests/test_eval_deterministic.py` replays pre-recorded LLM responses from fixture files, enabling comprehensive evaluation tests in CI without API keys. This VCR-style pattern provides:
- Reproducible test results (no LLM variability)
- Multi-turn conversation testing with checkpointer verification (`aget_state()` confirms 4+ messages persist across turns)
- Zero API cost in CI (no Gemini calls)
- Immediate failure detection on graph logic regressions

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
| Domain guardrails | Dedicated off_topic, gambling_advice, action_request, age_verification, patron_privacy paths | Single system prompt, no structured routing |
| Observability | Each node is individually traceable in LangSmith | Tool calls are traceable but routing is opaque |
| Code complexity | More code (8 nodes, 2 routing functions) | ~10 lines to set up |
| Flexibility | Full control over retry logic, validation criteria | Simpler but less control |

### Degraded-Pass Validation Strategy

The validate node uses a **degraded-pass** strategy when the validator LLM fails:

| Attempt | Validator LLM failure behavior | Rationale |
|---------|-------------------------------|-----------|
| First attempt | **PASS** (serve unvalidated response) | Availability over safety — a failed validator should not block an otherwise grounded response |
| Retry attempt | **FAIL** (route to fallback) | Safety over availability — if validation failed twice, the response is suspect |

**Trade-off**: On first attempt, the system may serve an unvalidated LLM response to a casino guest. This is acceptable because: (1) the response is still RAG-grounded by the generate node's context, (2) deterministic guardrails (prompt injection, responsible gaming, age verification, competitor deflection) run *before* the LLM and are not affected, (3) the fallback path always provides safe, human-written responses with the property's contact information.

In production, this trade-off should be reviewed with the compliance team. A stricter policy (fail-closed on all validator failures) trades availability for safety.

### Single-Worker Assumption

Several in-memory data structures (rate limiter, circuit breaker, retriever singleton) rely on single-worker deployment (`--workers 1` in Dockerfile). This is a documented demo trade-off:

| Component | Risk with `--workers > 1` | Production path |
|-----------|--------------------------|-----------------|
| `RateLimitMiddleware._requests` | Per-worker rate limiting (each worker has its own counter) | Redis-backed distributed rate limiter |
| `CircuitBreaker` singleton | Per-worker circuit state (failures not shared) | Redis or shared-memory state |
| `get_retriever()` singleton | Multiple ChromaDB instances (memory waste, no data conflict) | Vertex AI Vector Search (stateless client) |

### CSP ``unsafe-inline``

The security headers middleware uses ``script-src 'self' 'unsafe-inline'`` and ``style-src 'self' 'unsafe-inline'`` because the demo serves a single-file chat UI (``static/index.html``) with embedded ``<style>`` and ``<script>`` blocks. No user-generated content is rendered as HTML, so the XSS attack surface is minimal.

**Production path**: Externalize CSS/JS into separate static files and replace ``'unsafe-inline'`` with nonce-based CSP — generate a per-request nonce in middleware and inject it into ``<script nonce="...">`` tags.

### Demo vs Production

| Component | Demo | Production |
|-----------|------|------------|
| Vector DB | ChromaDB (in-process, persistent) | Vertex AI Vector Search |
| Checkpointing | MemorySaver (lost on restart) | FirestoreSaver |
| LLM auth | API key in `.env` | Vertex AI IAM + GCP Secret Manager |
| Deployment | Docker Compose (local) | Cloud Run |
| Rate limiting | In-memory per-IP dict (single-instance only; reset on restart) | Redis-backed distributed limiter (shared state across instances) |
| Monitoring | Structured logging | LangSmith + Cloud Monitoring |

---

## Configuration

All settings in `src/config.py` using `pydantic-settings`. Every value is overridable via environment variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Google AI API key (`SecretStr` — redacted in logs/repr) |
| `API_KEY` | (empty) | When set, `/chat` requires `X-API-Key` header (`SecretStr`) |
| `PROPERTY_NAME` | `Mohegan Sun` | Property name used in prompts |
| `PROPERTY_DATA_PATH` | `data/mohegan_sun.json` | Path to property JSON |
| `PROPERTY_WEBSITE` | `mohegansun.com` | Property website (used in fallback messages) |
| `PROPERTY_PHONE` | `1-888-226-7711` | Property phone (used in fallback and error messages) |
| `MODEL_NAME` | `gemini-2.5-flash` | LLM model |
| `MODEL_TEMPERATURE` | `0.3` | LLM temperature |
| `MODEL_TIMEOUT` | `30` | LLM call timeout (seconds) |
| `MODEL_MAX_RETRIES` | `2` | LLM retry count on failure |
| `MODEL_MAX_OUTPUT_TOKENS` | `2048` | Max response tokens |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model (768 dim) |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence directory |
| `RAG_TOP_K` | `5` | Number of retrieval results |
| `RAG_CHUNK_SIZE` | `800` | Text chunk size (characters) |
| `RAG_CHUNK_OVERLAP` | `120` | Chunk overlap (characters, ~15% of chunk size) |
| `RAG_MIN_RELEVANCE_SCORE` | `0.3` | Minimum relevance score (0-1, higher = more relevant) to include a result |
| `ALLOWED_ORIGINS` | `["http://localhost:8080"]` | CORS allowed origins |
| `RATE_LIMIT_CHAT` | `20` | Max chat requests per minute per IP |
| `RATE_LIMIT_MAX_CLIENTS` | `10000` | Max tracked client IPs (memory bound) |
| `SSE_TIMEOUT_SECONDS` | `60` | Stream timeout (seconds) |
| `MAX_REQUEST_BODY_SIZE` | `65536` | Max request body in bytes (64 KB) |
| `MAX_MESSAGE_LIMIT` | `40` | Max total messages (human + AI) before forcing conversation end |
| `ENABLE_HITL_INTERRUPT` | `false` | When true, pauses before generate node for human-in-the-loop review |
| `CB_FAILURE_THRESHOLD` | `5` | Consecutive LLM failures before circuit opens |
| `CB_COOLDOWN_SECONDS` | `60` | Seconds before circuit transitions to half-open |
| `GRAPH_RECURSION_LIMIT` | `10` | LangGraph recursion limit (bounds validate→retry loop) |
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

HEALTHCHECK is defined in both `Dockerfile` (for standalone `docker run`) and `docker-compose.yml` (with environment-specific tuning — longer timeout, 60s start period). The docker-compose definition overrides the Dockerfile default when running via `docker compose up`.

### Docker Compose

Single-service setup with:
- `.env` file for secrets (`GOOGLE_API_KEY`).
- Named volume `chroma_data` for ChromaDB persistence across restarts.
- 2GB memory limit, 60s start period for initial embedding.

### Cloud Build

`cloudbuild.yaml` defines a 5-step CI/CD pipeline:
1. Install dev dependencies (`requirements-dev.txt`), lint (`ruff`), type-check (`mypy`), and run tests with coverage (`pytest --cov --cov-fail-under=90`).
2. Build Docker image tagged with commit SHA.
3. **Trivy container vulnerability scan** — scans for CRITICAL and HIGH severity CVEs (`--exit-code=1` fails the build on findings, `--ignore-unfixed` skips unpatched OS-level CVEs).
4. Push to Artifact Registry (`us-central1-docker.pkg.dev`).
5. Deploy to Cloud Run (us-central1, 2Gi memory, 90s timeout, `--allow-unauthenticated` for demo — API key auth is enforced at the app layer via `ApiKeyMiddleware`).

---

## Cost Model

Estimated per-request cost using Gemini 2.5 Flash pricing (as of Feb 2026):

| Operation | Input | Output | Est. Cost |
|-----------|-------|--------|-----------|
| Router (structured output) | ~200 tokens | ~30 tokens | $0.000075 |
| Generate (RAG-grounded) | ~2,000 tokens (prompt + context) | ~300 tokens | $0.000825 |
| Validate (adversarial review) | ~1,500 tokens | ~30 tokens | $0.000465 |
| Embedding (query) | ~20 tokens | — | $0.000001 |
| **Total per request** | | | **~$0.0014** |

**Pricing basis**: Gemini 2.5 Flash — $0.15/1M input tokens, $0.60/1M output tokens. Validation retry adds one extra generate + validate cycle (~$0.0013).

**Monthly projections** (single property):

| Traffic | Requests/month | Est. Cost |
|---------|---------------|-----------|
| Low (demo) | 1,000 | ~$1.40 |
| Medium | 50,000 | ~$70 |
| High | 500,000 | ~$700 |

Embedding ingestion is a one-time cost (~$0.003 for 30 documents). ChromaDB storage is free (local). Vertex AI Vector Search adds ~$70/month at production scale.

---

## Latency Budget

Expected per-request latency breakdown (non-cached, single property):

| Phase | P50 | P95 | Notes |
|-------|-----|-----|-------|
| Guardrails (audit + responsible gaming) | <1ms | <1ms | Pure regex, no I/O |
| Router LLM (structured output) | ~200ms | ~500ms | Gemini 2.5 Flash, short prompt |
| Retrieval (ChromaDB) | ~10ms | ~30ms | Local vector search, 5 results |
| Generate LLM (RAG-grounded) | ~800ms | ~1,500ms | ~2K token prompt, streaming first token ~200ms |
| Validate LLM (adversarial review) | ~300ms | ~600ms | Short prompt, structured output |
| **Total (happy path)** | **~1.3s** | **~2.6s** | Router + retrieve + generate + validate |
| **Total (with retry)** | **~2.4s** | **~4.7s** | Adds one generate + validate cycle |
| **Total (SSE first token)** | **~1.0s** | **~2.0s** | Router + retrieve + generate first token |

Deterministic paths (greeting, off_topic, gambling_advice) skip retrieval and generation — P50 <5ms.

*Estimates based on Gemini 2.5 Flash benchmarks (Google AI Studio, Feb 2026) and local ChromaDB profiling. Production latency will vary with network conditions and Vertex AI Vector Search cold starts.*

Circuit breaker open state adds 0ms (immediate static fallback, no LLM call).

---

## Scope Decisions

The following features from the initial architecture specification (`assignment/architecture.md`) were consciously deferred. Each is a production-readiness enhancement, not a demo requirement.

| Feature | Status | Rationale |
|---------|--------|-----------|
| API key authentication (`X-API-Key` + `hmac.compare_digest`) | **Implemented** | Pure ASGI `ApiKeyMiddleware`, disabled when `API_KEY` is empty |
| Circuit breaker pattern | **Implemented** | Async-safe `CircuitBreaker` with `asyncio.Lock` — configurable thresholds (`CB_FAILURE_THRESHOLD`/`CB_COOLDOWN_SECONDS`) → half-open probe |
| Per-category data files (8 JSON files) | Single JSON | Simpler ingestion; per-category files are a multi-property scaling concern |
| Pydantic validation of data files | Deferred | Runtime validation at ingestion is sufficient; schema enforcement adds maintenance burden |
| `structlog` structured logging | Standard `logging` | Cloud Logging compatible JSON emitted by middleware; `structlog` is a luxury, not a necessity |
| nginx frontend container | Deferred | FastAPI serves static files directly; nginx adds container orchestration complexity |
| Multi-property system (`get_property_config()`) | Single property | Demo targets one property; the config externalization enables multi-property with zero code changes |

### Module Organization

`src/agent/` separates concerns into focused modules:

| Module | Responsibility | Lines |
|--------|---------------|-------|
| `guardrails.py` | Deterministic pre-LLM safety (prompt injection 7 patterns, responsible gaming 25 patterns EN+ES+ZH, age verification 6 patterns, BSA/AML 10 patterns, patron privacy 7 patterns) | ~238 |
| `circuit_breaker.py` | Async-safe `CircuitBreaker` class + lazy `_get_circuit_breaker()` singleton | ~87 |
| `nodes.py` | 8 async graph nodes + 2 routing functions + dual LLM singletons + dynamic greeting categories (`@lru_cache`) | ~580 |
| `graph.py` | StateGraph compilation + node name constants + HITL interrupt support, `chat()`, `chat_stream()`, `_initial_state()` DRY helper | ~285 |
| `state.py` | TypedDict state schema (`PropertyQAState`, `RetrievedChunk`) + Pydantic structured output models | ~71 |
| `prompts.py` | 3 prompt templates + helpline constant | ~161 |
| `tools.py` | RAG retrieval with RRF reranking (hash-based dedup, multi-strategy fusion, no @tool decorators) | ~188 |

All deferred features have clear production paths documented in the Trade-offs section above.

---

Built by Oded Ben-Yair | February 2026
