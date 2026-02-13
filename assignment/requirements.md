# Assignment Requirements — Senior AI/Backend Engineer

## Source
`Take-Home Assignment — Senior AI_Backend Engineer.pdf` from Hey Seven

## Explicit Requirements

| # | Requirement | Our Status |
|---|-------------|-----------|
| 1 | **LangGraph** agent | Boilerplate ready (agent.py, state.py, nodes.py) |
| 2 | **One property** loaded as context | Need to collect data for one casino |
| 3 | Answer guest questions (restaurants, entertainment, amenities, rooms, promos) | RAG pipeline designed |
| 4 | **Q&A only** — NO actions (no bookings, reservations, account ops) | Architecture has tools/escalation — scale DOWN |
| 5 | **Tests** included | 69 specs designed, zero implemented |
| 6 | **Docker** setup | Dockerfile ready, need docker-compose |
| 7 | **API** required if interface needs one | FastAPI boilerplate ready |

## Implicit Requirements (What They'll Judge)

1. **Clean, readable code** — they'll review every line
2. **Architecture decisions with rationale** — "why this, not that"
3. **Trade-offs documented** — honest about limitations
4. **Production-quality engineering** — senior-level patterns
5. **Knowledge ingestion approach** — how property data gets in
6. **README** — setup instructions, architecture overview, how to run
7. **Working demo** — `docker-compose up` and it works

## Evaluation Criteria (Quoted)

> "This is an open-ended assignment. There is no single correct solution. We're interested in
> how you approach the problem, the trade-offs you make, and the design decisions. This will
> serve as the foundation for a deeper technical conversation in the interview."

Translation: They care MORE about decisions than features. A clean, well-reasoned simple system beats a complex one.

## Property Choice

Examples given: Twin Arrows Navajo, Mohegan Sun, Foxwoods, Agua Caliente.
**Recommendation: Mohegan Sun** — we have existing data in knowledge-base/, tribal casino (shows regulatory awareness), CT-based (matches Hey Seven's scope).

## What's NOT Required (Don't Over-Build)

- Compliance/regulatory (BSA/AML, self-exclusion) — mention in trade-offs, don't implement
- Escalation to human hosts — not in scope
- Booking/reservation actions — explicitly excluded
- Voice AI — not mentioned
- Multi-property support — ONE property only
- Complex monitoring/observability — nice to mention, not needed
- Frontend — "chat interface is up to you" (bonus if included)

## Gap Analysis: Architecture Doc vs Assignment

### Aligned (Reuse)
- LangGraph StateGraph pattern
- RAG retrieve → generate flow
- FastAPI with SSE streaming
- Dockerfile (multi-stage, non-root)
- Casino domain knowledge structure
- Trade-off documentation pattern

### Over-Built (Scale Down)
- Validation node with retry loop — impressive but may signal over-engineering
- 8 casino tools — they want Q&A only, not tool-calling
- SecurityHeadersMiddleware — good practice but enterprise-level for a take-home
- Circuit breaker pattern — over-engineering for a demo
- Rate limiting — over-engineering for a demo
- Compliance guardrails — not required

### Gaps (Must Build)
- **Property data**: Scrape/collect Mohegan Sun data (restaurants, rooms, entertainment, promos)
- **Working code**: Boilerplate is templates, need running code
- **Tests**: Actual test files (not just specs)
- **docker-compose.yml**: Easy `docker-compose up` experience
- **README.md**: The first thing they'll read
- **Knowledge ingestion**: Script to load property data into vector store
- **.env.example**: Clear env var documentation

## Scoring Dimensions (For Multi-Model Evaluation)

1. Architecture & Design Decisions
2. LangGraph Implementation Quality
3. RAG Pipeline & Knowledge Ingestion
4. API Design & Streaming
5. Testing (coverage, quality, meaningfulness)
6. Docker & DevOps
7. Code Quality & Readability
8. Trade-off Documentation
9. Domain Understanding
10. Overall Impression (would you hire?)
