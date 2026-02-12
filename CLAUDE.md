# Hey Seven - Interview Assignment Infrastructure

## Project Identity
Interview prep and home assignment execution for Hey Seven (heyseven.ai) - "The Autonomous Casino Host That Never Sleeps".
Oded Ben-Yair is interviewing for Senior AI/Backend Engineer position.

## Hard Rules (Project-Specific)

1. **ISOLATION**: This project is COMPLETELY isolated. NEVER reference other projects (Sentimark, QC Analyzer, etc.)
2. **NO Azure DevOps**: Git remote is GitHub ONLY (Oded-Ben-Yair/hey-seven). NEVER push to Azure DevOps.
3. **LangGraph ONLY**: Primary agent framework is LangGraph. Do not split focus with other frameworks.
4. **GCP deployment target**: Cloud Run, Firestore, Vertex AI. Show stack alignment with Hey Seven.
5. **NO mock data**: All demos use real casino domain data from knowledge-base/
6. **API keys**: From Azure Key Vault (kv-seekapa-apps) for development. GCP service accounts for deployment.
7. **Speed matters**: Fast delivery is part of the wow factor. Pre-built infrastructure enables this.
8. **QUALITY BAR**: This assignment directly affects Oded's career. Every file, every function, every decision must be production-grade. No shortcuts, no "good enough".

## Current State (Updated 2026-02-12)

- **Phase 0-2**: COMPLETE. Research, boilerplate, hostile reviews (2 rounds), fixes applied.
- **Review Score**: 80/100 overall (Round 2) + top 3 fixes applied → ~82-85 estimated.
- **Waiting for**: Assignment from Hey Seven (expected tonight/tomorrow morning).
- **33+ files ready**: LangGraph agent, FastAPI API, Next.js frontend, RAG pipeline, GCP deployment, casino domain knowledge base.

## Tech Stack Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Agent Framework | LangGraph 1.0 (pinned 0.2.60) | CTO preference, production-grade state machine |
| Primary LLM | Gemini 2.5 Flash | GCP alignment, cost-effective ($0.30/1M input). Gemini 3 Flash available (Dec 2025) |
| Complex LLM | Gemini 2.5 Pro | For complex reasoning. Gemini 3 Pro available (Nov 2025) |
| Cloud | GCP Cloud Run | Their stack |
| Vector DB | Vertex AI Vector Search | GCP-native. ChromaDB for local dev only |
| State/Memory | Firestore + FirestoreSaver | **Community package** (NOT official LangGraph). `pip install langgraph-checkpoint-firestore` |
| Backend | FastAPI (Python) | LangGraph is Python, fast development. Uses lifespan, ASGI middleware |
| Frontend | Next.js 15 + React 19 + Tailwind CSS 4 | Nice-to-have in job posting, Oded's strength |
| CI/CD | Cloud Build | GCP-native |
| Observability | LangSmith | LangGraph ecosystem standard |

## Company Context

- **Hey Seven**: Israeli seed-stage startup, 4-5 people, MVP in 5 weeks, clients waiting
- **Product**: AI agent that handles all digital casino host tasks 24/7
- **Problem**: Casino hosts spend 75% of time on digital/phone tasks instead of floor time with VIPs
- **Market**: US casinos, regulated environment (self-exclusion, BSA/AML, TCPA, state-specific rules)
- **CTO**: South American background (mentioned in interview). Name not confirmed.
- **R&D Site Manager**: Neta Rotshtein (Technion, biotech/food engineering background)
- **Website**: heyseven.ai (JS-rendered, brand extracted via browser-control)

## Key People

- **Neta Rotshtein**: R&D Site Manager, conducted HR interview. Technion graduate, biotech background. LinkedIn PDF in research/personas/.
- **CTO**: South American background (mentioned in interview). Name not confirmed.

## Oded's Strengths to Highlight

- Production agent systems (6 microservices, 240K monthly executions)
- Multi-LLM orchestration with Thompson Sampling
- RAG with vector databases on Azure
- Tool-calling agents (function calling with OpenAI/Claude/Gemini)
- Voice AI (ElevenLabs, multilingual Arabic/Hebrew)
- Regulated industry experience (CFD trading = similar to regulated gaming)
- CI/CD, testing, production-grade engineering
- Next.js/React frontend expertise

## Directory Structure

```
research/                    - All research outputs
  brand-design.md            - Hey Seven visual identity (colors, fonts, design language)
  casino-domain.md           - Casino host operations deep dive
  company-intel.md           - Hey Seven company intelligence
  langgraph-gcp.md           - LangGraph + GCP technical patterns (original)
  langgraph-latest.md        - LangGraph 1.0.8 latest intel (Feb 2026, social + deep research)
  frontend-latest.md         - React 19.2, Next.js 15/16, AI SDK 6, Tailwind 4 (Feb 2026)
  personas/                  - LinkedIn profiles + PDFs
  perplexity-deep/           - Deep research (regulations, market, psychology)
knowledge-base/              - Structured data for RAG ingestion
  casino-operations/         - Comp system, host workflow
  regulations/               - State-by-state requirements
  player-psychology/         - Retention playbook
  company-context/           - Hey Seven overview
boilerplate/                 - Pre-built code templates
  langgraph-agent/           - LangGraph agent (agent.py, tools.py, state.py, nodes.py, prompts.py, memory.py)
  api/                       - FastAPI backend (main.py, routes.py, middleware.py)
  rag/                       - RAG pipeline (indexer.py, retriever.py, embeddings.py)
  gcp/                       - Deployment configs (Dockerfile, cloudbuild.yaml, deploy.sh)
  frontend/                  - Next.js app (components, styles, API client)
  requirements.txt           - Python dependencies (pinned)
reviews/                     - Hostile review results (round-1/, round-2/)
assignment/                  - [EMPTY] Will hold assignment analysis when received
deliverables/                - [EMPTY] Final submission artifacts
```

## Boilerplate Code Map (Quick Reference)

| File | What It Does | Key Patterns |
|------|-------------|--------------|
| `langgraph-agent/agent.py` | Agent assembly, dual-mode (react + custom graph) | `create_react_agent`, `state_modifier=`, `StateGraph`, conditional edges |
| `langgraph-agent/tools.py` | 8 casino tools | `@tool` decorator, structured returns |
| `langgraph-agent/state.py` | `CasinoHostState` extends `MessagesState` | `Annotated[list, add_messages]`, typed fields |
| `langgraph-agent/nodes.py` | LLM node, compliance checker, escalation, formatter | `get_llm()` singleton, `.with_structured_output()` |
| `langgraph-agent/prompts.py` | Casino host system prompt | Regulation-aware, persona-driven |
| `langgraph-agent/memory.py` | Checkpointer factory | `MemorySaver` (dev), `FirestoreSaver` (prod) |
| `api/main.py` | FastAPI app with lifespan, WebSocket | `hmac.compare_digest()`, rate limiting, SIGTERM |
| `api/routes.py` | REST endpoints | `APIKeyHeader`, Pydantic v2, pattern validation |
| `api/middleware.py` | CORS, logging, error handling | Pure ASGI middleware |
| `rag/retriever.py` | Vector search with reranking | ChromaDB (dev), Vertex AI (prod) |
| `frontend/` | Next.js 15 chat UI | Server/Client split, SSE streaming, brand tokens |

## Research Preservation

- NEVER delete or overwrite research/ files without explicit confirmation
- Research files are append-only during active research phases
- All research findings go into knowledge-base/ for RAG ingestion

## Known Limitations (From Hostile Reviews)

These are documented, accepted trade-offs — NOT bugs:
- No tests yet (will add during assignment implementation)
- No streaming endpoint (boilerplate has SSE client code; backend endpoint TBD per assignment)
- No HITL interrupt wiring (LangGraph supports it; will wire based on assignment needs)
- No LangSmith tracing configuration (env vars only, no code changes needed)
- ChromaDB in requirements.txt (local dev only; prod uses Vertex AI Vector Search)

---

## ASSIGNMENT RECEPTION PROTOCOL (CRITICAL)

**When Oded pastes the assignment, follow this protocol EXACTLY. Do NOT skip steps.**

### Phase 3: Deep Evaluation (NO CODE)

**Step 1: Parse & Understand (15 min)**
1. Read the assignment 3 times. Extract EVERY requirement, constraint, deliverable, and evaluation criterion.
2. Write `assignment/requirements.md` with:
   - Explicit requirements (stated directly)
   - Implicit requirements (inferred from context)
   - Evaluation criteria (how they'll judge the work)
   - Constraints (time, tech stack, scope)
   - Ambiguities (things not specified that we need to decide)

**Step 2: Multi-LLM Architecture Debate (30-45 min)**
3. Launch `/multi-model-debate` with the full assignment text + our boilerplate inventory. The debate question: "What is the optimal architecture for this assignment that maximizes evaluation score while leveraging our pre-built infrastructure?"
4. Debate must cover:
   - Which boilerplate pieces apply directly vs need modification
   - What's missing that we need to build from scratch
   - Architecture trade-offs (speed vs quality vs impressiveness)
   - Risk assessment: what could go wrong?
   - What would make this submission stand out vs other candidates?

**Step 3: Architecture Document (30 min)**
5. Write `assignment/architecture.md` — full system design:
   - Component diagram (which modules, how they connect)
   - Data flow (request → agent → tools → response)
   - Technology choices with justification
   - File-by-file implementation plan
   - What we're reusing from boilerplate/ vs building new
   - Testing strategy
   - Deployment plan
   - Demo scenario (what we'll show to prove it works)

**Step 4: Hostile Review of Architecture (20 min)**
6. Launch code-judge to tear apart the architecture doc:
   - Does it actually satisfy ALL requirements?
   - Are there over-engineering risks?
   - Are there missing pieces that will block us during implementation?
   - Is the timeline realistic?
7. Fix all critical findings. Iterate until the judge passes it.

**Step 5: User Approval Gate**
8. Present the architecture to Oded with:
   - Summary of what we'll build
   - Key design decisions and trade-offs
   - Estimated implementation plan
   - Risks and mitigations
9. **DO NOT WRITE A SINGLE LINE OF CODE UNTIL ODED APPROVES.**

### Phase 4: Implementation (After Architecture Approved)

10. Create agent team (3-4 teammates, max) with clear file ownership.
11. Each teammate implements their slice in parallel.
12. Hostile review after each major milestone.
13. Integration testing after all pieces merge.

### Phase 5: Quality & Delivery

14. End-to-end testing with real casino scenarios.
15. Visual validation (Playwright screenshots + Gemini vision).
16. Code-judge final hostile review of ALL code.
17. Clean README.md for the submission.
18. GCP deployment (if time permits and assignment requires it).
19. Present deliverable to Oded for final review before submission.

---

## Agent Team Templates

### Research Team (Phase 1 — COMPLETE)
| Agent | Role | Owns |
|-------|------|------|
| brand-scout | Branding, design, personas | research/brand-design.md, research/personas/ |
| domain-expert | Casino domain, regulations | research/perplexity-deep/, knowledge-base/ |
| tech-builder | LangGraph boilerplate, GCP | boilerplate/ |

### Build Team (Phase 4 — when assignment arrives)
| Agent | Role | Owns |
|-------|------|------|
| agent-architect | LangGraph core, tools, RAG | src/agent/, src/rag/ |
| api-builder | FastAPI, GCP deployment | src/api/, GCP configs |
| ui-designer | Frontend with branding | src/frontend/ |

### Review Team (Phase 5)
| Agent | Role | Owns |
|-------|------|------|
| code-judge | Hostile code review | reviews/ |
| visual-verifier | Screenshot + Gemini vision validation | deliverables/screenshots/ |
