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

## Tech Stack Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Agent Framework | LangGraph 1.0 | CTO preference, production-grade state machine |
| Primary LLM | Gemini 2.5 Flash | GCP alignment, cost-effective ($0.30/1M input) |
| Complex LLM | Gemini 2.5 Pro | For complex reasoning tasks |
| Cloud | GCP Cloud Run | Their stack |
| Vector DB | Vertex AI Vector Search | GCP-native |
| State/Memory | Firestore + FirestoreSaver | LangGraph native, GCP-native |
| Backend | FastAPI (Python) | LangGraph is Python, fast development |
| Frontend | Next.js + React | Nice-to-have in job posting, Oded's strength |
| CI/CD | Cloud Build | GCP-native |
| Observability | LangSmith | LangGraph ecosystem standard |

## Company Context

- **Hey Seven**: Israeli seed-stage startup, 4-5 people, MVP in 5 weeks
- **Product**: AI agent that handles all digital casino host tasks 24/7
- **Problem**: Casino hosts spend 75% of time on digital/phone tasks instead of floor time with VIPs
- **Market**: US casinos, regulated environment
- **CTO**: TBD (research in progress)
- **R&D Site Manager**: Neta Rotshtein (Technion, biotech/food engineering background)
- **Website**: heyseven.ai

## Key People

- **Neta Rotshtein**: R&D Site Manager, conducted HR interview. Technion graduate, biotech background.
- **CTO**: South American background (mentioned in interview). Name TBD.

## Oded's Strengths to Highlight

- Production agent systems (6 microservices, 240K monthly executions)
- Multi-LLM orchestration with Thompson Sampling
- RAG with vector databases
- Tool-calling agents (function calling with OpenAI/Claude/Gemini)
- Voice AI (ElevenLabs, multilingual)
- Regulated industry experience (CFD trading ≈ regulated gaming)
- CI/CD, testing, production-grade engineering

## Directory Structure

```
research/              - All research outputs (casino domain, company intel, tech stack)
knowledge-base/        - Structured data for RAG ingestion
boilerplate/           - Pre-built code templates (LangGraph, RAG, API, GCP, frontend)
assignment/            - Actual assignment analysis and planning
deliverables/          - Final submission artifacts
```

## Research Preservation

- NEVER delete or overwrite research/ files without explicit confirmation
- Research files are append-only during active research phases
- All research findings go into knowledge-base/ for RAG ingestion

## Agent Team Templates

### Research Team (Phase 1)
| Agent | Role | Owns |
|-------|------|------|
| brand-scout | Branding, design, personas | research/brand-design.md, research/personas/ |
| domain-expert | Casino domain, regulations | research/perplexity-deep/, knowledge-base/ |
| tech-builder | LangGraph boilerplate, GCP | boilerplate/ |

### Build Team (Phase 4 - when assignment arrives)
| Agent | Role | Owns |
|-------|------|------|
| agent-architect | LangGraph core, tools, RAG | src/agent/, src/rag/ |
| api-builder | FastAPI, GCP deployment | src/api/, GCP configs |
| ui-designer | Frontend with branding | src/frontend/ |
