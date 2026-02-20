# Hey Seven — Production AI Casino Host Agent

## Project Identity
Production MVP for Hey Seven (heyseven.ai) — "The Autonomous Casino Host That Never Sleeps". AI agent handling all digital casino host tasks 24/7 for US casino clients.

## Hard Rules (Project-Specific)

1. **ISOLATION**: This project is COMPLETELY isolated. NEVER reference other projects (Sentimark, QC Analyzer, etc.)
2. **NO Azure DevOps**: Git remote is GitHub ONLY (Oded-Ben-Yair/hey-seven). NEVER push to Azure DevOps.
3. **LangGraph ONLY**: Primary agent framework is LangGraph. Do not split focus with other frameworks.
4. **GCP deployment target**: Cloud Run, Firestore, Vertex AI. All infrastructure is GCP-native.
5. **NO mock data**: All demos use real casino domain data from knowledge-base/
6. **API keys**: From Azure Key Vault (kv-seekapa-apps) for development. GCP service accounts for deployment.
7. **QUALITY BAR**: Every file, every function, every decision must be production-grade. No shortcuts, no "good enough".

## Current State (Updated 2026-02-20)

- **Codebase**: 23K LOC, 51 source modules across 10 packages
- **Tests**: 1070+ tests across 32 test files
- **Agent**: 11-node LangGraph StateGraph v2.2 with 6 specialist agents
- **Review Score**: 92/100 after 20 rounds of hostile multi-model review
- **Version**: Tagged v1.0.0
- **Deployment**: Live demo on GCP Cloud Run

## Tech Stack Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Agent Framework | LangGraph 1.0 (pinned 0.2.60) | Production-grade state machine with validation loops |
| Primary LLM | Gemini 2.5 Flash | GCP alignment, cost-effective ($0.30/1M input) |
| Complex LLM | Gemini 2.5 Pro | For complex reasoning tasks |
| Cloud | GCP Cloud Run | Target deployment platform |
| Vector DB | Vertex AI Vector Search | GCP-native. ChromaDB for local dev only |
| State/Memory | Firestore + FirestoreSaver | **Community package** (NOT official LangGraph). `pip install langgraph-checkpoint-firestore` |
| Backend | FastAPI (Python) | LangGraph is Python, uses lifespan, ASGI middleware |
| Frontend | Next.js 15 + React 19 + Tailwind CSS 4 | Modern stack, SSE streaming support |
| CI/CD | Cloud Build | GCP-native |
| Observability | LangSmith + Langfuse | LangGraph ecosystem standard + open-source tracing |

## Company Context

- **Hey Seven**: Israeli seed-stage startup, building an AI casino host platform
- **Product**: AI agent that handles all digital casino host tasks 24/7
- **Problem**: Casino hosts spend 75% of time on digital/phone tasks instead of floor time with VIPs
- **Market**: US casinos, regulated environment (self-exclusion, BSA/AML, TCPA, state-specific rules)
- **Website**: heyseven.ai

## Directory Structure

```
src/                         - Production source code
  agent/                     - LangGraph agent core
    agents/                  - Specialist agents (dining, hotel, comp, entertainment, host)
      _base.py               - Shared specialist execution logic (DRY extraction)
      registry.py            - Agent dispatch registry
    graph.py                 - 11-node StateGraph assembly
    state.py                 - CasinoHostState TypedDict
    nodes.py                 - LLM nodes, router, formatter
    tools.py                 - Casino domain tools
    prompts.py               - System prompts with persona
    guardrails.py            - Pre-LLM deterministic guardrails (5 layers)
    compliance_gate.py       - Compliance validation node
    circuit_breaker.py       - LLM circuit breaker
    memory.py                - Checkpointer factory (MemorySaver dev, Firestore prod)
    whisper_planner.py       - Multi-turn conversation planner
    persona.py               - Agent persona configuration
  api/                       - FastAPI backend
    app.py                   - App with lifespan, SSE streaming
    middleware.py             - Pure ASGI middleware (logging, security, rate limit)
    models.py                - Pydantic v2 request/response models
    errors.py                - Structured error handling
    pii_redaction.py         - PII redaction (fail-closed)
  rag/                       - RAG pipeline
    pipeline.py              - Ingestion with per-item chunking
    firestore_retriever.py   - Vector search with RRF reranking
    embeddings.py            - Embedding model (pinned version)
    reranking.py             - Reciprocal Rank Fusion
  data/                      - Data models
    models.py                - Domain data models
    guest_profile.py         - Guest profile management
  casino/                    - Casino configuration
    config.py                - Multi-property casino config
    feature_flags.py         - Per-casino feature flags
  cms/                       - Content management
    sheets_client.py         - Google Sheets CMS client
    webhook.py               - CMS webhook handler
    validation.py            - Content validation
  sms/                       - SMS outreach
    telnyx_client.py         - Telnyx SMS integration
    webhook.py               - Inbound SMS handler
    compliance.py            - TCPA/DNC compliance
  observability/             - Monitoring and tracing
    langfuse_client.py       - Langfuse integration
    traces.py                - Distributed tracing
    evaluation.py            - Automated evaluation framework
  config.py                  - Global settings (Pydantic BaseSettings)
tests/                       - 32 test files, 1070+ tests
  conftest.py                - Singleton cleanup, async fixtures
  test_graph_v2.py           - Full pipeline integration tests
  test_nodes.py              - Node-level unit tests
  test_agents.py             - Specialist agent tests
  test_api.py                - API endpoint tests
  test_rag.py                - RAG pipeline tests
  test_sms.py                - SMS compliance tests
  test_phase2_integration.py - Phase 2 integration tests
  test_phase3_integration.py - Phase 3 integration tests
  test_phase4_integration.py - Phase 4 integration tests
  ...                        - (22 more test files)
knowledge-base/              - Structured data for RAG ingestion
  casino-operations/         - Comp system, host workflow
  regulations/               - State-by-state requirements
  player-psychology/         - Retention playbook
  company-context/           - Hey Seven overview
research/                    - Domain research outputs
data/                        - Runtime data directory
docs/                        - Documentation
static/                      - Static assets
```

## Architecture Overview

The agent uses a custom 11-node LangGraph StateGraph with validation loops:

```
router -> [specialist dispatch] -> generate -> validate -> respond
                                       ^          |
                                       +-- retry --+  (max 1 retry)
```

Key architectural patterns:
- **Pre-LLM deterministic guardrails** (5 layers: prompt injection, responsible gaming, age verification, BSA/AML, patron privacy)
- **Structured output routing** via Pydantic + Literal types (no substring matching)
- **Validation loop** with generate -> validate -> retry(max 1) -> fallback
- **Degraded-pass validation** (first attempt + validator failure = PASS; retry + failure = FAIL)
- **Specialist agent DRY extraction** via shared `_base.py` with dependency injection
- **Circuit breaker** per LLM call with fail-open safe fallback
- **TTL-cached LLM singletons** for credential rotation (GCP Workload Identity)
- **PII redaction** fails closed (safe placeholder, never pass-through)
- **Pure ASGI middleware** (no BaseHTTPMiddleware — breaks SSE streaming)

## Known Limitations

Documented, accepted trade-offs:
- ChromaDB in requirements.txt is for local dev only; prod uses Vertex AI Vector Search
- InMemorySaver for local dev (MAX_ACTIVE_THREADS=1000 guard; prod uses FirestoreSaver)
- LangSmith sampling requires app-level code (not just env var config)
- HITL interrupt is config-toggled but not enabled by default in production

---

## REVIEW ROUND PROTOCOL (Prevents Context Overflow)

**Problem**: Review rounds with 10 dimensions cause context overflow when findings + fixes are processed in main context.

**Solution**: Use TeamCreate swarm. Main lead NEVER sees full findings.

### Review Round Execution

```
STEP 1: Main lead creates team "review-round-N"
STEP 2: Main lead creates tasks:
  - Task A: "Review dimensions 1-5" (assigned to reviewer-alpha)
  - Task B: "Review dimensions 6-10" (assigned to reviewer-beta)
  - Task C: "Apply fixes from reviewer findings" (assigned to fixer, blocked by A+B)
  - Task D: "Write round summary" (assigned to fixer, blocked by C)

STEP 3: Teammates execute:
  - reviewer-alpha: Reads code, writes findings to reviews/round-N/alpha.md
  - reviewer-beta: Reads code, writes findings to reviews/round-N/beta.md
  - fixer: Reads BOTH finding files, applies fixes, writes summary

STEP 4: Main lead reads ONLY reviews/round-N/summary.md (5-10 lines)
STEP 5: Main lead shuts down team, reports to user
```

### Key Rules:
1. **Findings go to FILES, not parent context** — reviewers write to reviews/round-N/*.md
2. **Fixer reads files, not messages** — no large finding payloads in team messages
3. **Main lead reads only summary** — never the detailed findings
4. **Max 4 teammates** — 2 reviewers + 1 fixer + 1 reserve
5. **Each reviewer covers 5 dimensions** — parallel, no overlap
6. **Fixer works bottom-up** in the doc to minimize line shift conflicts

### Review Dimensions (10 total, split into 2 groups):
- **Group A** (reviewer-alpha): Graph Architecture, RAG Pipeline, Data Model, API Design, Testing Strategy
- **Group B** (reviewer-beta): Docker & DevOps, Prompts & Guardrails, Scalability & Production, Trade-off Documentation, Domain Intelligence
