# Hey Seven -- Knowledge Base

## What This System Does (Plain English)

Hey Seven is an AI casino host agent for Mohegan Sun casino resort. Guests ask natural-language questions about dining, entertainment, hotel rooms, amenities, gaming, and promotions. The agent uses a custom 12-node LangGraph StateGraph with RAG (Retrieval-Augmented Generation) to produce grounded, validated answers streamed token-by-token via Server-Sent Events. It handles compliance-sensitive topics (responsible gaming, age verification, patron privacy, BSA/AML) with deterministic regex guardrails before any LLM call.

**Users**: Casino resort guests interacting via web chat widget or SMS. Property staff update content via Google Sheets CMS.

**Business value**: 24/7 AI concierge replacing the "75% of time a human casino host spends on digital/phone tasks." Mandatory compliance guardrails in 11 languages protect against regulatory violations. Multi-property design means adding a second casino requires zero code changes.

## Architecture Overview

**Interactive diagram**: [system-arch-interactive.html](diagrams/system-arch-interactive.html) (13 nodes, 13-step guided tour) | [Macro view](diagrams/system-arch-interactive-macro.html) (7-group overview)

### Key Architectural Decisions

1. **Custom StateGraph over create_react_agent**: The 12-node graph provides explicit control over validation loops, conditional routing, and multiple terminal nodes that `create_react_agent` cannot express. Every request flows through an explicit path: compliance_gate -> router -> retrieve -> whisper_planner -> generate -> profiling -> validate -> persona_envelope -> respond.

2. **Pre-LLM deterministic guardrails**: 204 compiled regex patterns across 11 languages run before any LLM call. This catches prompt injection, responsible gaming, BSA/AML, and patron privacy queries at zero cost and deterministic speed. All patterns are re2-compatible (no backtracking, ReDoS-safe).

3. **Specialist agent DRY extraction**: All 5 agents (host, dining, entertainment, comp, hotel) share a common `execute_specialist()` base function with dependency injection. This was unanimously praised across 20+ review rounds as "the single best architectural change."

4. **Validation loop with degraded-pass**: Generate -> validate -> retry (max 1) -> fallback. If the validator LLM fails on first attempt but generate succeeded, the response passes through (availability). On retries, fail-closed (safety).

5. **Whisper Track Planner**: Silent background LLM that analyzes conversation context and produces a plan for the speaking agent. Fail-silent contract -- never crashes the pipeline. ~$0.0003/turn at Gemini Flash pricing.

6. **Build for one property, design for N**: Every configuration choice (property name, data paths, model name, prompts) is externalized via pydantic-settings so adding a second property requires zero code changes.

## Domain Knowledge

### Lessons Learned
- **Guardrail priority ordering is critical**: When two guardrail categories match the same input, the higher-priority one must run first. Example: "Is there someone I can talk to here?" matched patron privacy regex instead of maintaining crisis response. Position-based short-circuiting means first match wins.
- **Tests pass != LLM API works**: Mock LLMs bypass schema validation. A Pydantic schema that works in 3236 tests can fail 100% in production because Gemini Flash rejects schemas with >5 constrained fields.
- **Validation grounding too strict = excessive fallback**: "After dinner, try the show" was rejected because show data was not in dining RAG context. 37/236 turns hit fallback. Relaxed grounding for category-level suggestions (without specific fabricated facts).
- **LLM system prompt tone = output tone**: "Warmest, most knowledgeable person" + "genuine enthusiasm" produced "Oh, I'd be absolutely delighted." Fixed: "warmth from SUBSTANCE not ENTHUSIASM" + "NEVER start with Oh,".
- **Gambling slang normalization**: "YOLO'd" failed RAG lookup because `rstrip` removed `'` but not `d`. Must handle contractions and suffixes separately from punctuation stripping.
- **Cross-node state overwrite**: compliance_gate set `guest_sentiment="grief"`, router VADER overwrote with "neutral". Fixed with priority sentiment guard.
- **Score plateau (3+ rounds within +/-2)**: Time to change strategy (focused dims, refactor debt), not repeat the same approach.

### Common Pitfalls
- **Do not use `create_react_agent`**: It cannot express validation loops, conditional routing, or multiple terminal nodes.
- **Pin LangGraph version exactly** (`langgraph==0.2.60`): API parameters change between minor versions.
- **Use `string.Template.safe_substitute()`**: `.format()` crashes on user text with `{curly braces}`.
- **Keep Gemini Flash structured output schemas flat**: <5 constrained fields. Nested Pydantic models with Literal + bounded float = "too many schema states" error.
- **Never return mutable module-level data**: Functions returning global dicts must `copy.deepcopy()`. One caller mutating corrupts state for all subsequent requests.
- **Always use `asyncio.Lock` in async code**: `threading.Lock` deadlocks. Native `redis.asyncio` instead of `asyncio.to_thread()`.

## Operations

### What Runs Automatically
| Component | Trigger | What It Does | Failure Impact |
|-----------|---------|-------------|----------------|
| LangGraph agent | POST /chat | Processes guest queries through 12-node pipeline | Guest receives no response |
| SMS webhook | POST /sms/webhook | Processes inbound SMS with TCPA compliance | SMS channel goes silent |
| CMS webhook | POST /cms/webhook | Re-ingests knowledge base on content updates | Stale data until next restart |
| Circuit breaker | On LLM failure | Opens after 5 failures (60s cooldown), prevents cascading | Static fallback messages |
| Version-stamp purging | On ingestion | Removes stale RAG chunks from previous content versions | Ghost data in responses |

### How to Monitor Health
```bash
# Health check (returns 503 when degraded)
curl https://<cloud-run-url>/health

# Live check (always 200 -- for Docker HEALTHCHECK)
curl https://<cloud-run-url>/live

# Metrics (P50/P95/P99 latency, active streams, circuit breaker)
curl https://<cloud-run-url>/metrics

# Graph structure (visual of 12-node pipeline)
curl https://<cloud-run-url>/graph
```

### When Things Break
| Symptom | Check | Fix |
|---------|-------|-----|
| All responses are fallback text | Circuit breaker state in /metrics | Check Gemini API quota; breaker auto-resets after 60s |
| Compliance gate blocks everything | Guardrail pattern matches | Check `src/agent/guardrails.py` for overly broad regex |
| RAG returns irrelevant results | Embedding model version | Verify `text-embedding-004` is pinned in config |
| SSE streaming drops mid-response | ASGI middleware stack | Ensure no BaseHTTPMiddleware (breaks SSE) |
| Guest profile data lost between turns | State reducer check | Verify `_merge_dicts` reducer on `extracted_fields` |
| SMS responses too long | Persona envelope config | Check `PERSONA_MAX_CHARS=160` for SMS mode |
| PII leaking in responses | PII redactor logs | Check `src/api/pii_redaction.py` fail-closed behavior |
| Stale data after content update | Ingestion logs | Run re-ingestion manually; check version-stamp purging |

### Key Metrics to Watch
- **Fallback rate**: Percentage of turns hitting the fallback node (target: < 10%)
- **Validation RETRY rate**: Percentage of responses needing re-generation (target: < 20%)
- **Circuit breaker state**: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- **RAG relevance scores**: Average cosine similarity of retrieved chunks (target: > 0.5)
- **P95 latency**: End-to-end response time (target: < 3s for web, < 5s for SMS)
- **Guardrail trigger rates**: Which compliance categories fire most often

## Key Contacts and Dependencies

### Who Uses This System
- **Casino resort guests**: Primary users via web chat and SMS
- **Hey Seven team**: Israeli seed-stage startup (heyseven.ai)
- **Property staff**: Update content via Google Sheets CMS

### External Dependencies
| Service | Purpose | Failure Impact |
|---------|---------|----------------|
| Gemini 2.5 Flash / Pro | All LLM calls (router, generate, validate, whisper planner) | Circuit breaker opens; static fallback responses |
| ChromaDB / Vertex AI | Vector search for RAG retrieval | No knowledge base access; all queries hit fallback |
| Firestore | LangGraph conversation checkpointer (prod) | Conversation history lost between turns |
| Redis Memorystore | Circuit breaker state + rate limiting | Falls back to in-memory (single-instance only) |
| Telnyx | SMS inbound/outbound | SMS channel unavailable |
| Google Sheets | CMS content management | Content updates blocked; existing data unaffected |
| LangFuse | Tracing and observability | No trace visibility (non-critical) |
| Azure Key Vault | Development API keys | Dev environment stops (prod uses GCP service accounts) |

### Who to Ask About What
- **Agent graph architecture**: `src/agent/graph.py` + `ARCHITECTURE.md`
- **Compliance guardrails**: `src/agent/guardrails.py` (204 patterns, 11 languages)
- **Specialist agents**: `src/agent/agents/` (5 agents + `_base.py` shared logic)
- **RAG pipeline**: `src/rag/pipeline.py` + `src/rag/reranking.py`
- **Casino domain knowledge**: `knowledge-base/` directory + `research/` directory
- **Behavioral evaluation**: `tests/scenarios/` (195 scenarios, 27 YAML files)
- **ADRs**: `docs/adrs/` (28 architectural decision records)

## Quick Reference

### Important File Paths
| Path | Purpose |
|------|---------|
| `src/agent/graph.py` | 12-node StateGraph assembly (entry point: `build_graph()`) |
| `src/agent/state.py` | PropertyQAState TypedDict + Pydantic models |
| `src/agent/nodes.py` | Router, retrieve, generate, validate, respond, fallback |
| `src/agent/compliance_gate.py` | Pre-LLM compliance validation (204 patterns) |
| `src/agent/guardrails.py` | Compiled regex patterns (5 layers, 11 languages) |
| `src/agent/agents/_base.py` | Shared specialist execution logic (DRY) |
| `src/agent/agents/registry.py` | Agent dispatch registry |
| `src/agent/dispatch.py` | Specialist dispatch helpers (SRP extraction) |
| `src/agent/whisper_planner.py` | Silent background LLM planner |
| `src/agent/profiling.py` | Guest profiling enrichment node |
| `src/agent/incentives.py` | Incentive engine with tiered autonomy |
| `src/agent/circuit_breaker.py` | Redis L1/L2 circuit breaker |
| `src/agent/prompts.py` | System prompts with persona |
| `src/agent/streaming_pii.py` | SSE streaming PII redactor |
| `src/api/app.py` | FastAPI with lifespan, SSE streaming, middleware |
| `src/api/middleware.py` | Pure ASGI middleware (6 layers) |
| `src/rag/pipeline.py` | RAG ingestion with per-item chunking |
| `src/rag/reranking.py` | Reciprocal Rank Fusion |
| `src/config.py` | Global pydantic-settings configuration |
| `src/casino/config.py` | Multi-property casino configuration |
| `knowledge-base/` | Structured data for RAG ingestion |
| `tests/scenarios/` | 195 behavioral scenarios (27 YAML files) |

### Key Environment Variables
| Variable | Source | Description |
|----------|--------|-------------|
| `GOOGLE_API_KEY` | Key Vault (dev) / GCP SA (prod) | Gemini API key |
| `PROPERTY_NAME` | pydantic-settings | Casino property name |
| `VECTOR_DB` | pydantic-settings | `chroma` (dev) or `firestore` (prod) |
| `EMBEDDING_MODEL` | pydantic-settings | `text-embedding-004` (pinned) |
| `RAG_MIN_RELEVANCE_SCORE` | pydantic-settings | Cosine similarity threshold (default 0.3) |
| `PERSONA_MAX_CHARS` | pydantic-settings | 0 (web) or 160 (SMS) |
| `CB_FAILURE_THRESHOLD` | pydantic-settings | Circuit breaker failure threshold (default 5) |
| `CB_COOLDOWN_SECONDS` | pydantic-settings | Circuit breaker cooldown (default 60) |
| `API_KEY` | Key Vault | API authentication key |
| `REDIS_URL` | pydantic-settings | Redis Memorystore URL |
| `LANGFUSE_SECRET_KEY` | Key Vault | LangFuse tracing key |
| `TELNYX_API_KEY` | Key Vault | Telnyx SMS API key |
| `GRAPH_RECURSION_LIMIT` | pydantic-settings | Max graph recursion (default 25) |

### API Endpoints
| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/chat` | POST | API Key | SSE streaming chat (main entry point) |
| `/health` | GET | None | System health (503 when degraded) |
| `/live` | GET | None | Liveness probe (always 200) |
| `/metrics` | GET | API Key | P50/P95/P99 latency + system metrics |
| `/property` | GET | None | Property information (ETag cached) |
| `/graph` | GET | None | Graph structure visualization |
| `/sms/webhook` | POST | Telnyx | Inbound SMS handler |
| `/cms/webhook` | POST | CMS | Google Sheets content update |

### Key Statistics
| Metric | Value |
|--------|-------|
| Source modules | 66 across 10 packages |
| Tests | 3236, 0 failures, 90%+ coverage |
| Graph nodes | 12 (custom StateGraph) |
| Specialist agents | 5 (host, dining, entertainment, comp, hotel) |
| Guardrail patterns | 204 compiled regex, 11 languages |
| ADRs | 28 with status lifecycle |
| Behavioral scenarios | 195 across 27 YAML files |
| Version | v1.4.0 |

### Production URLs
| Service | URL |
|---------|-----|
| Website | heyseven.ai |
| GitHub | `Oded-Ben-Yair/hey-seven` (NOT Azure DevOps) |
| GCP Project | Cloud Run deployment (target) |
