# Custom StateGraph Implementation Design

**Date**: 2026-02-14
**Status**: Approved
**Objective**: Replace `create_react_agent` with custom `StateGraph` matching architecture doc v1.0

## Problem

The architecture document describes a custom StateGraph with 8 nodes, 3 LLM calls per query, and a validation/retry loop. The current implementation uses `create_react_agent` (prebuilt), which is a significant gap that an evaluator comparing doc to code will immediately notice.

## Approach: Full Rewrite

Replace the core agent entirely. Keep: API layer, RAG pipeline, middleware, config, frontend.

### State Schema (TypedDict, 9 fields)

```python
class PropertyQAState(TypedDict):
    messages: Annotated[list, add_messages]
    query_type: str | None
    router_confidence: float
    retrieved_context: list[dict]
    validation_result: str | None
    retry_count: int
    retry_feedback: str | None
    current_time: str
    sources_used: list[str]
```

### 8 Graph Nodes

| Node | LLM? | Purpose |
|------|-------|---------|
| router | Yes (structured output) | Intent classification -> RouterOutput |
| retrieve | No | ChromaDB similarity search, top-K |
| generate | Yes | Answer generation with retrieved context |
| validate | Yes (structured output) | Post-generation guardrails -> ValidationResult |
| respond | No | Extract sources, clear retry state |
| fallback | No | Safe response with contact info |
| greeting | No | Template welcome message |
| off_topic | No | Template responses (3 sub-cases) |

### Edge Map

```
START -> router
router -> {retrieve, greeting, off_topic}  (conditional)
retrieve -> generate
generate -> validate
validate -> {generate, respond, fallback}   (conditional)
respond -> END
fallback -> END
greeting -> END
off_topic -> END
```

### 3 Prompts

1. **CONCIERGE_SYSTEM_PROMPT**: Identity, VIP style, 10 rules, prompt safety, time-aware
2. **VALIDATION_PROMPT**: 6 criteria (grounded, on-topic, no gambling, read-only, accurate, responsible gaming)
3. **ROUTER_PROMPT**: Classify intent with structured output

### Streaming Strategy

Use `graph.astream(stream_mode="messages")` to stream tokens from the `generate` node. Filter for `AIMessageChunk` content, exclude router/validator output.

### Files Changed

| File | Action | Owner |
|------|--------|-------|
| src/agent/state.py | Rewrite | graph-architect |
| src/agent/nodes.py | Create | graph-architect |
| src/agent/graph.py | Rewrite | graph-architect |
| src/agent/prompts.py | Rewrite | prompt-engineer |
| src/agent/tools.py | Repurpose | graph-architect |
| src/api/app.py | Modify | api-integrator |
| src/api/models.py | Modify | api-integrator |
| tests/* | Rewrite/Create | test-writer |
| cloudbuild.yaml | Create | api-integrator |
| assignment/architecture.md | Update | api-integrator |

### Swarm Team

4 teammates, each with distinct file ownership:
- graph-architect: state.py, nodes.py, graph.py, tools.py
- prompt-engineer: prompts.py
- api-integrator: app.py, models.py, cloudbuild.yaml, architecture.md
- test-writer: tests/

### Key Constraints

- SSE event format unchanged (frontend needs zero changes)
- All tests runnable without GOOGLE_API_KEY
- 69+ tests total
- Docker must still build and run
- Architecture doc and code must be aligned
