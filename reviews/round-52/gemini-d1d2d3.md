# R52 Hostile Review — Gemini 3.1 Pro (D1, D2, D3)

**Reviewer**: Gemini 3.1 Pro (thinking=high)
**Date**: 2026-02-25
**Files Reviewed**: dispatch.py, state.py, graph.py, constants.py, reranking.py, tools.py, pipeline.py (key sections)

---

## Scores

| Dimension | Score | Weight |
|-----------|-------|--------|
| D1 Graph Architecture | 6.0 | 0.20 |
| D2 RAG Pipeline | 7.0 | 0.10 |
| D3 Data Model | 5.0 | 0.10 |

**Weighted contribution**: (6.0 * 0.20) + (7.0 * 0.10) + (5.0 * 0.10) = 1.20 + 0.70 + 0.50 = **2.40 / 4.00**

---

## D1: Graph Architecture & Execution (Score: 6.0/10)

Your graph topology looks clean on paper, but the actual execution architecture is riddled with critical flaws regarding concurrency, streaming, and multi-tenancy.

### CRITICAL | Broken Multi-Tenancy
- **File**: `src/agent/dispatch.py` ~L144
- **What's wrong**: The prompt explicitly states this is a "Multi-tenant isolation via property_id" system. Yet, you are checking `await is_feature_enabled(settings.CASINO_ID, ...)` where `settings = get_settings()`. You are pulling a global/static `CASINO_ID` from the environment instead of pulling the tenant ID from the LangGraph `config` (e.g., `config["configurable"]["casino_id"]`). This will cause all tenants running on this worker to use the feature flags and RAG data of a single globally configured casino.
- **How to fix**: Extract `casino_id` from the thread configuration passed at runtime, never from global `get_settings()`.

### CRITICAL | Silent Token Streaming Failure for Subgraphs
- **File**: `src/agent/graph.py` ~L278
- **What's wrong**: Your SSE streaming relies on `langgraph_node == NODE_GENERATE` to capture tokens: `if kind == "on_chat_model_stream" and langgraph_node == NODE_GENERATE:`. However, `NODE_GENERATE` invokes specialist agents (`agent_fn`). If these specialist agents are LangGraph compiled subgraphs (which is standard), LangGraph's `astream_events` will yield events where `langgraph_node` is the name of the node *inside the subgraph* (e.g., `generate_response`), not `NODE_GENERATE`. Your stream will be completely dead/silent when a specialist handles the query.
- **How to fix**: Check the execution path `event.get("tags")` or traverse the parent hierarchy in the event metadata rather than doing a hard string match on `langgraph_node`.

### MAJOR | Unconstrained Graph Recursion Loop
- **File**: `src/agent/graph.py` ~L101
- **What's wrong**: `_route_after_validate_v2` routes `RETRY` back to `NODE_GENERATE`. There is no retry limit check inside the routing function! You are relying entirely on `compiled.recursion_limit = settings.GRAPH_RECURSION_LIMIT` to kill the loop. Reaching the global recursion limit raises a `GraphRecursionError`, crashing the execution and returning a generic 500-style fallback message to the user, bypassing your `fallback_node` completely.
- **How to fix**: Implement a routing condition that explicitly checks `state["retry_count"] >= MAX_RETRIES` and routes to `NODE_FALLBACK` gracefully.

### MAJOR | Blocking Async Event Loop
- **File**: `src/agent/tools.py` ~L42
- **What's wrong**: `search_knowledge_base` and `search_hours` are plain synchronous functions. They call `retriever.retrieve_with_scores` synchronously. When invoked by `NODE_RETRIEVE` within the async graph (`ainvoke`/`astream_events`), this vector search I/O will block the entire Python async event loop, destroying the throughput of your "production" API.
- **How to fix**: Wrap the synchronous retriever calls in `asyncio.to_thread()` or use LangChain's native async retrievers (`aretrieve`).

### MINOR | Timeout Bypass Flaw
- **File**: `src/agent/dispatch.py` ~L207
- **What's wrong**: If a specialist times out, you catch it and return an `AIMessage` with `skip_validation: True`. However, the graph still physically routes from `NODE_GENERATE` -> `NODE_VALIDATE`. If `validate_node` doesn't explicitly check for and short-circuit on `skip_validation=True`, it will try to validate an error message.

---

## D2: RAG Pipeline (Score: 7.0/10)

The RRF implementation is mathematically sound, but the pipeline has glaring fault-tolerance and recall issues.

### CRITICAL | Fatal Fragility in Dual Retrieval
- **File**: `src/agent/tools.py` ~L42
- **What's wrong**: In `search_knowledge_base`, you execute `semantic_results`, then `augmented_results`. If `semantic_results` succeeds but `augmented_results` throws an exception (e.g., intermittent embedding API timeout), the execution drops into the `except Exception:` block and returns `[]`. You lose perfectly good semantic results because the augmentation path failed.
- **How to fix**: Wrap each retrieval call in its own try/except block. If one fails, gracefully degrade to using only the results from the successful call in the RRF fusion.

### MAJOR | Metadata Destruction in RRF Deduplication
- **File**: `src/rag/reranking.py` ~L21
- **What's wrong**: You compute `doc_id` using a SHA-256 hash of `doc.page_content` and `doc.metadata.get('source', '')`. If two distinct RAG items share the same content string (e.g., "Must be 21+ to enter.") and source URL, but have different metadata (e.g., `category: gaming` vs `category: nightlife`), they will generate the same hash. The loop `doc_map[doc_id] = (doc, score)` will overwrite one with the other, permanently deleting the metadata of the former.
- **How to fix**: Include all critical metadata (like category or item ID) in the hash payload, or better yet, use the chunk's native UUID if available.

### MAJOR | Naive Intent Detection
- **File**: `src/agent/tools.py` ~L25
- **What's wrong**: `_get_augmentation_terms` splits the query and checks `words & _TIME_WORDS`. It does not strip punctuation. If a user asks `"What are the buffet hours?"`, `words` contains `"hours?"`. `"hours?"` will not match `"hours"` in the frozen set, bypassing your schedule augmentation entirely.
- **How to fix**: Use regex or `string.punctuation` to sanitize the query before intersection checking.

---

## D3: Data Model & State Management (Score: 5.0/10)

You have fundamentally misunderstood how LangGraph state reducers work across turns. Your state mutations will permanently poison thread memory.

### CRITICAL | State Poisoning via Cross-Turn Boolean Reducer
- **File**: `src/agent/state.py` ~L35
- **What's wrong**: `def _keep_truthy(a: bool, b: bool) -> bool: return bool(a or b)`. You use this for `suggestion_offered`. In LangGraph, state persists across turns in the same thread. If a user gets a suggestion in turn 1, `suggestion_offered` becomes `True`. On turn 2, `_initial_state` passes `"suggestion_offered": False`. LangGraph applies the reducer: `_keep_truthy(True, False)` -> `True`! This flag will permanently stick to `True` forever. A user will NEVER get another suggestion in this thread.
- **How to fix**: You cannot use `a or b` if you need to reset state per-turn. Use the `UNSET_SENTINEL` pattern to allow explicit resets.

### CRITICAL | Missing Reducer for `sources_used`
- **File**: `src/agent/state.py` ~L75
- **What's wrong**: `sources_used: list[str]` lacks a reducer (e.g., `Annotated[list[str], operator.add]`). Without an explicit reducer, LangGraph treats lists as "overwrite". If `NODE_RETRIEVE` outputs `sources_used: ["A", "B"]`, and `NODE_VALIDATE` passes without emitting `sources_used`, the state keeps `["A", "B"]`. But if `NODE_RESPOND` outputs `sources_used: ["A"]` (maybe it dropped B), it overwrites the list. More dangerously, if any intermediate node outputs a partial list, the rest are lost. Your `chat_stream` SSE logic accumulates sources safely, but your `chat` API method relies on the final state, which will only contain whatever the *last* node decided to write.
- **How to fix**: Use `Annotated[list[str], add]` to accumulate sources, or strictly govern which single node is allowed to write to this key.

### MAJOR | `NoneType` Exception in Dictionary Merge
- **File**: `src/agent/state.py` ~L20
- **What's wrong**: `_merge_dicts` loops over `b.items()`. If a node accidentally returns `{"extracted_fields": None}` (which happens frequently with weak LLM structured output parsing), `b` is `None`, and `b.items()` will raise `AttributeError: 'NoneType' object has no attribute 'items'`, immediately crashing the graph.
- **How to fix**: Add a safeguard: `if not b: return dict(a)`.

### MINOR | Unsafe Pydantic Validator
- **File**: `src/agent/dispatch.py` ~L133
- **What's wrong**: You log `result.reasoning[:80]`. If the LLM generates `reasoning=None` (despite the Pydantic type hint, poorly aligned models or failing fast-paths can return None), `None[:80]` will throw a `TypeError`.
- **How to fix**: Use `str(result.reasoning)[:80]`.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 5 |
| MAJOR | 5 |
| MINOR | 2 |
| **Total** | **12** |

### CRITICAL Findings (5)
1. D1: Broken multi-tenancy — CASINO_ID from env instead of config
2. D1: Silent token streaming failure for subgraph specialists
3. D2: Fatal fragility in dual retrieval — one failure kills both results
4. D3: State poisoning via cross-turn boolean reducer (_keep_truthy)
5. D3: Missing reducer for sources_used — last-write-wins data loss

### MAJOR Findings (5)
1. D1: Unconstrained graph recursion loop — relies on global limit
2. D1: Blocking async event loop — sync tools.py in async graph
3. D2: Metadata destruction in RRF deduplication — same content different categories
4. D2: Naive intent detection — punctuation bypasses word matching
5. D3: NoneType exception in _merge_dicts — None input crashes graph

### MINOR Findings (2)
1. D1: Timeout bypass flaw — skip_validation not verified before validate
2. D3: Unsafe Pydantic validator — None[:80] throws TypeError
