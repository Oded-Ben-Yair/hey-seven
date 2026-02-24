# ADR 006: ChromaDB Dev / Vertex AI Prod Vector DB Split

## Status
Accepted

## Context
RAG requires a vector database. ChromaDB is simple for local development (~200MB, zero infrastructure). Vertex AI Vector Search is the GCP-native production option.

## Decision
- **Dev**: ChromaDB (SQLite-backed, in-process)
- **Prod**: Vertex AI Vector Search (managed, scalable)
- Same retriever interface via LangChain abstraction

## Known Risks
1. **Embedding drift**: Different distance metrics or index configurations between ChromaDB and Vertex AI could produce different retrieval results
2. **CI blind spot**: Tests run on ChromaDB; Vertex AI regressions are invisible until production
3. **ChromaDB lazy import**: ~200MB dependency imported inside function to avoid production image bloat

## Mitigation
- Embedding model pinned (`gemini-embedding-001`) — same vectors in both DBs
- Minimum relevance score threshold (`RAG_MIN_RELEVANCE_SCORE=0.3`) catches garbage retrievals regardless of backend
- Integration tests with real Vertex AI recommended before GA launch
