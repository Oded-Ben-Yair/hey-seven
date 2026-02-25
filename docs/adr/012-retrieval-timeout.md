# ADR-012: Retrieval Timeout (10s Default)

## Status
Accepted

## Context
RAG retrieval can stall on slow embedding API calls or vector DB queries. Without a timeout, the entire request pipeline blocks indefinitely.

## Decision
10-second timeout for retrieval operations, configurable via `RETRIEVAL_TIMEOUT`. Retrieval timeout is independent of the LLM `MODEL_TIMEOUT` (30s) because retrieval is a pre-LLM step that should fail fast.

## Consequences
- Retrieval failure returns empty context -> specialist gets no-context fallback
- 10s is generous for embedding + vector search (typical: 1-3s)
- Prevents retrieval from consuming the entire request timeout budget
