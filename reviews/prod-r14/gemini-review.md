# Hey Seven Production Review -- Round 14 (Gemini Perspective)

**Reviewer**: Gemini (hostile, architecture-focused)
**Commit**: 8655074
**Date**: 2026-02-21
**Spotlight**: RAG Pipeline + Domain Intelligence (+1 severity)

---

## Scores

| # | Dimension | Score | Trend |
|---|-----------|-------|-------|
| 1 | Graph Architecture | 9 | = |
| 2 | **RAG Pipeline** (SPOTLIGHT) | 8 | = |
| 3 | Data Model | 8 | = |
| 4 | API Design | 8 | = |
| 5 | Testing Strategy | 8 | = |
| 6 | Docker & DevOps | 8 | = |
| 7 | Prompts & Guardrails | 9 | = |
| 8 | Scalability & Production | 7 | = |
| 9 | Trade-off Documentation | 9 | = |
| 10 | **Domain Intelligence** (SPOTLIGHT) | 8 | = |

**Overall: 82/100**

---

## Findings

### F1 -- CRITICAL (SPOTLIGHT +1 severity -> CRITICAL): Knowledge Base Files Referenced in Review Prompt Do Not Exist

**File**: `knowledge-base/casino-operations/dining.json`, `knowledge-base/regulations/ct-gaming.md`
**Evidence**: Neither file exists on disk. The knowledge base contains only 5 markdown files across 4 subdirectories (`casino-operations/comp-system.md`, `casino-operations/host-workflow.md`, `regulations/state-requirements.md`, `player-psychology/retention-playbook.md`, `company-context/hey-seven-overview.md`). There is no structured JSON data in the knowledge-base directory at all.

**Impact**: The review prompt references `dining.json` and `ct-gaming.md` as files to read, but they do not exist. This raises a documentation honesty concern: if external stakeholders or review prompts assume a richer knowledge base than what actually exists, the RAG pipeline's coverage is overstated. The actual dining data lives in `data/mohegan_sun.json` under the `restaurants` key -- it is NOT in the knowledge-base directory. This means the RAG pipeline has two distinct data sources (`data/mohegan_sun.json` for structured property data and `knowledge-base/` for markdown domain knowledge) but the knowledge-base directory itself is sparse (5 files total). A customer deploying this for a new property would need to understand this split, and the README/CLAUDE.md does not call it out clearly.

**Recommendation**: Document the data architecture: `data/*.json` = per-property structured data (restaurants, entertainment, hotel, gaming, amenities, promotions), `knowledge-base/` = cross-property domain knowledge (regulations, comp formulas, playbooks). Add a `knowledge-base/README.md` explaining the split and onboarding instructions for new properties.

---

### F2 -- HIGH (SPOTLIGHT +1 severity -> HIGH): `reingest_item()` Missing `_ingestion_version` Metadata -- Stale Chunk Purging Bypass

**File**: `src/rag/pipeline.py`, lines 262-340
**Evidence**: The `reingest_item()` function (used by CMS webhook for real-time content updates) adds chunks with metadata including `_schema_version`, `last_updated`, `source`, and `property_id` -- but does NOT add `_ingestion_version`. The bulk `ingest_property()` function (line 667) stamps every chunk with `_ingestion_version` and purges stale chunks based on it (lines 699-721).

When a CMS update calls `reingest_item()`, the new chunk lacks `_ingestion_version`. On the next bulk `ingest_property()` run, the purge query (`_ingestion_version != current_stamp`) will delete all chunks that lack `_ingestion_version`, including the CMS-updated ones. Conversely, if `reingest_item()` is called repeatedly for edited content, old versions are never purged (no version stamp to filter by).

This is a ghost data accumulation vector: CMS-updated content creates new chunk IDs (different SHA-256 from edited text) without deleting the old chunk. Over time, the vector store accumulates stale CMS chunks.

**Recommendation**: Add `_ingestion_version` to `reingest_item()` metadata using `datetime.now(tz=timezone.utc).isoformat()`. After successful upsert, purge chunks with the same `source` pattern (`cms:{category}/{item_id}`) but different `_ingestion_version`.

---

### F3 -- HIGH: `_get_circuit_breaker()` Is Synchronous But Called in Async Context Without Lock Protection

**File**: `src/agent/circuit_breaker.py`, lines 275-302
**Evidence**: `_get_circuit_breaker()` is a synchronous function that reads and writes to `_cb_cache` (a `TTLCache`). It is called from async contexts (`execute_specialist()` at `_base.py:92`, `_dispatch_to_specialist()` at `graph.py:204`). TTLCache is not thread-safe, and in an async context, the function can be called concurrently by multiple coroutines. The sequence `get("cb") -> None -> create -> set("cb")` has a TOCTOU (time-of-check-time-of-use) race: two coroutines can both read `None` and create duplicate CircuitBreaker instances, with one silently discarded.

Compare with the LLM singletons (`_get_llm()`, `_get_validator_llm()`) which use `async with _llm_lock:` for exactly this reason. The circuit breaker singleton does not have equivalent protection.

**Impact**: Under concurrent load (e.g., multiple SSE streams starting simultaneously after container cold start), two CircuitBreaker instances can be created. The discarded instance may have recorded failures that are lost, effectively resetting the breaker's failure tracking. In practice, this race window is narrow (TTLCache write is fast) and the consequence is a missed failure count -- LOW probability but architecturally inconsistent with the established pattern.

**Recommendation**: Convert `_get_circuit_breaker()` to async with a module-level `asyncio.Lock`, matching the pattern used by `_get_llm()`. Alternatively, document the intentional deviation (synchronous for build-time access in `graph.py:414` where async is not available).

---

### F4 -- MEDIUM: FirestoreRetriever Property ID Filtering Is Python-Side, Not Server-Side

**File**: `src/rag/firestore_retriever.py`, lines 62-128
**Evidence**: The `_single_vector_query()` method over-fetches `top_k * 2` results from Firestore and then filters by `property_id` in Python (line 105: `if doc_property_id != property_id: continue`). Firestore's `find_nearest()` supports composite pre-filtering via `where()` clauses that execute server-side BEFORE the vector search.

**Impact**: In a multi-tenant deployment (multiple casinos sharing one Firestore collection), up to 50% of fetched documents could be discarded by Python-side filtering. This doubles the data transfer cost and increases latency. The over-fetch factor of 2x is a fixed heuristic that breaks when tenant data is highly skewed (e.g., one casino has 95% of documents -- the 2x over-fetch still returns mostly that casino's data for the minority tenant).

**Recommendation**: Add a `where("metadata.property_id", "==", property_id)` composite filter to the Firestore `find_nearest()` call. This requires a composite index on `(metadata.property_id, embedding)` in Firestore. Fall back to Python-side filtering only if the composite index is not yet created (log a warning on first fallback). This is consistent with the ChromaDB retriever which uses server-side `filter={"property_id": property_id}`.

---

### F5 -- MEDIUM: Embedding Model Uses `@lru_cache` While All Other Singletons Use TTLCache

**File**: `src/rag/embeddings.py`, lines 20-39
**Evidence**: `get_embeddings()` uses `@lru_cache(maxsize=4)` which never expires. All other singletons in the codebase use TTLCache with 1-hour TTL for GCP Workload Identity credential rotation:
- `_get_llm()`: TTLCache (nodes.py:102)
- `_get_validator_llm()`: TTLCache (nodes.py:104)
- `_get_circuit_breaker()`: TTLCache (circuit_breaker.py:272)
- `_get_retriever_cached()`: Manual dict-based TTL (pipeline.py:856-858)

The embedding model singleton will hold stale GCP credentials indefinitely. If Workload Identity rotates credentials (typically every 1 hour), the embedding model will fail with authentication errors until the process restarts.

**Impact**: In local development with API keys, this is a non-issue (API keys don't rotate). In GCP production with Workload Identity Federation, embedding calls will fail after credential rotation without a process restart. The failure manifests as retrieval errors in `search_knowledge_base()` and `search_hours()`, which are caught and return empty results -- the agent degrades to no-context fallback responses.

**Recommendation**: Replace `@lru_cache(maxsize=4)` with a TTLCache pattern matching the LLM singletons. Use a per-`task_type` key in a `TTLCache(maxsize=4, ttl=3600)`. This aligns with the documented TTL-cached singleton pattern.

---

### F6 -- MEDIUM (SPOTLIGHT +1 severity -> MEDIUM): RRF Reranking Has No Deduplication Awareness Across Strategies

**File**: `src/rag/reranking.py`, lines 40-60; `src/agent/tools.py`, lines 60-72
**Evidence**: `search_knowledge_base()` calls `retrieve_with_scores()` twice -- once for semantic search and once for entity-augmented search (`f"{query} name location details"`). Both strategies search the same vector store with the same embedding model. For most queries, the two result sets will have significant overlap (the same top documents appear in both). RRF handles this correctly via SHA-256 deduplication (reranking.py:45-47), but the underlying issue is cost: two embedding API calls and two vector searches are executed for every single query.

**Impact**: Every `search_knowledge_base()` call incurs 2x embedding cost and 2x vector search latency. The entity-augmented strategy (`"What time does Bobby Flay's close" + " name location details"`) appends generic terms that may actually *degrade* embedding quality for well-formed queries. The RRF benefit is primarily for short proper-noun-only queries (e.g., "Todd English's") where the augmented query adds useful context.

**Recommendation**: Profile the RRF improvement empirically. If semantic and augmented strategies return >80% overlap for typical queries, the augmented strategy adds latency without recall improvement. Consider making the second strategy conditional: only run augmented search when the query contains a proper noun (detected via NER or capitalization heuristic). This would reduce embedding API costs by ~40% for non-entity queries.

---

### F7 -- MEDIUM: `_flatten_nested_dict()` Silently Overwrites Items Missing `name` Key

**File**: `src/rag/pipeline.py`, lines 343-376
**Evidence**: In the nested dict flattening logic (line 357-358):
```python
if "name" not in sub_item:
    sub_item = {**sub_item, "name": key.replace("_", " ").title()}
```
This creates a shallow copy with an injected `name` field. However, the original `sub_item` dict (from the loaded JSON) is NOT mutated -- the copy is used downstream. If two items in the same list both lack a `name` key, they receive identical synthetic names (the parent key title-cased). When these are embedded, both chunks have the same `item_name` metadata, making provenance tracking ambiguous.

More critically, the generated `item_name` is derived from the JSON key (e.g., `"room_types"` becomes `"Room Types"`), not from any content in the item itself. This is a metadata accuracy issue: the `item_name` in the vector store does not reflect the actual content of the chunk.

**Impact**: Minor for retrieval (metadata is used for provenance, not search ranking). However, the `sources_used` field returned to the client (via `respond_node()`) extracts unique categories from metadata, and debugging chunk provenance becomes harder when multiple chunks share a synthetic name.

**Recommendation**: Use a content-derived name (e.g., first sentence or first 50 chars of the formatted text) as fallback instead of the JSON key. Add a `chunk_index` suffix (e.g., `"Room Types #1"`, `"Room Types #2"`) to disambiguate.

---

### F8 -- LOW: Concierge System Prompt Hardcodes "Uncasville, Connecticut" and "Mohegan Tribe"

**File**: `src/agent/prompts.py`, lines 85-89
**Evidence**: The `CONCIERGE_SYSTEM_PROMPT` contains:
```
$property_name is a premier tribal casino resort in Uncasville, Connecticut,
owned by the Mohegan Tribe. It features world-class dining, entertainment, gaming,
and hotel accommodations. The resort includes multiple towers, over 40 restaurants
and bars, a 10,000-seat arena, and a world-renowned spa.
```
This section is NOT parameterized -- it uses hardcoded property details. When deploying to a second casino (the stated multi-tenant goal), this prompt will describe the wrong property. The specialist agents (dining, hotel, comp, entertainment) do NOT have this problem -- they use only `$property_name` and `$current_time` template variables.

**Impact**: The CONCIERGE_SYSTEM_PROMPT is used by the general host agent. When `specialist_agents_enabled` is disabled (feature flag), ALL queries route through the host agent and will receive a system prompt describing Mohegan Sun regardless of the actual property.

**Recommendation**: Replace the hardcoded "About" section with a template variable `$property_description` populated from the property JSON file's `property.description` field. This aligns with the dynamic greeting categories pattern already implemented in `greeting_node()`.

---

### F9 -- LOW: `state.py` `RetrievedChunk` TypedDict Has a `score` Field That Is Not Consistently Populated

**File**: `src/agent/state.py`, line 36; `src/agent/tools.py`, lines 89-96
**Evidence**: `RetrievedChunk` declares `score: float` as a required field. The tools (`search_knowledge_base`, `search_hours`) correctly populate it from the retrieval score. However, `_base.py` (execute_specialist, line 86) accesses `retrieved_context` but never uses the `score` field. The `_format_context_block()` function in `nodes.py` (line 50) also ignores the score.

The score is set but never read by any downstream consumer. It survives in the state as dead data through the entire graph execution. Meanwhile, `retrieve_node()` in `nodes.py` (line 273) returns `{"retrieved_context": results}` where `results` is the return value from `search_knowledge_base()`, which does include scores. So the data is present but unused.

**Impact**: Minimal functional impact. The score could be useful for observability (logging which chunks were high-vs-low confidence) or for the validation prompt (showing the validator how confident the retrieval was). Currently it's carried through the state with no consumer.

**Recommendation**: Either (a) use the score in `_format_context_block()` to show confidence to the LLM (e.g., `"[1] (restaurants, relevance=0.87) ..."`) which would improve grounding, or (b) remove the `score` field from `RetrievedChunk` and the tools to reduce state bloat. Option (a) is preferred -- it provides the validation LLM with quality signal.

---

### F10 -- LOW: Turn-Limit Guard Compares Total Messages Against MAX_MESSAGE_LIMIT But `_initial_state` Adds One More

**File**: `src/agent/compliance_gate.py`, line 96; `src/agent/graph.py`, lines 468-491
**Evidence**: The compliance gate checks `len(messages) > settings.MAX_MESSAGE_LIMIT` (default 40). But `_initial_state()` prepends the new `HumanMessage` to the messages list before the graph runs. The checkpointer's `add_messages` reducer accumulates messages across turns. So by the time the compliance gate runs, `messages` contains all historical messages PLUS the current user message.

The guard triggers at message 41+, not message 40. This is a classic off-by-one: `>` vs `>=`. With `MAX_MESSAGE_LIMIT=40`, the 40th message (20th user turn in a human-AI ping-pong) passes through, and the 41st is blocked.

**Impact**: One extra turn is allowed beyond the intended limit. In a regulated environment where the limit exists for responsible interaction management, this is a minor inconsistency. Functionally harmless -- the difference between 40 and 41 messages has no practical impact.

**Recommendation**: Change to `>=` if the limit should be strictly 40, or document that the limit is "after N messages" meaning the (N+1)th is the first to be blocked. This is a documentation/intent question, not a bug.

---

### F11 -- LOW: `greeting_node` Calls `get_settings()` Twice in the Same Function

**File**: `src/agent/nodes.py`, lines 492-497
**Evidence**: `greeting_node()` calls `settings = get_settings()` at line 492, then calls `get_settings().CASINO_ID` again at line 497 when checking the `ai_disclosure_enabled` feature flag. The `get_settings()` function is `@lru_cache(maxsize=1)` so both calls return the same cached instance. However, this is inconsistent: line 492 already has `settings` available and `settings.CASINO_ID` should be used at line 497.

Similarly, `off_topic_node()` at line 550 calls `get_settings().CASINO_ID` despite already having `settings = get_settings()` at line 529.

**Impact**: Zero functional impact (same cached object). Purely a style inconsistency that makes code review harder -- the reader must verify that `get_settings()` is indeed cached.

**Recommendation**: Replace `get_settings().CASINO_ID` with `settings.CASINO_ID` at lines 497 and 550 since the local variable is already available.

---

## Architecture Assessment

### Strengths

1. **DRY specialist extraction**: The `_base.py` shared execution logic is the strongest architectural pattern in the codebase. 5 specialist agents reduced to thin wrappers with dependency injection preserving test mockability. This is a textbook extraction that eliminates ~600 lines of duplication while maintaining the same test surface.

2. **Dual-layer feature flags**: The distinction between build-time topology flags (sync, checked once at startup) and runtime behavior flags (async, per-request with Firestore overrides) is well-reasoned and well-documented. The extensive inline comments in `graph.py` (lines 379-413) explain the "why" clearly. This is production-grade design.

3. **Parity assertions**: The `_initial_state` parity check (graph.py:498-505) and `DEFAULT_FEATURES` parity assertions (feature_flags.py:73-87) catch schema drift at import time. This is a defensive programming pattern that prevents an entire class of bugs (adding a state field but forgetting to reset it per-turn).

4. **Streaming PII redaction**: The `StreamingPIIRedactor` with lookahead buffering solves a genuinely hard problem -- applying regex-based PII detection across arbitrary token boundaries in a streaming context. The intentional decision to NOT flush on CancelledError (graph.py:703-707) is the correct fail-safe behavior.

5. **Compliance guardrail ordering**: The 8-step priority chain in `compliance_gate_node` with explicit documentation of WHY injection runs before content guardrails and WHY semantic classification runs last demonstrates deep threat modeling. The ordering rationale is not obvious and the comments prevent well-intentioned reordering from introducing gaps.

### Weaknesses

1. **Knowledge base sparsity**: 5 markdown files is thin for a production deployment. The structured property data (mohegan_sun.json) is comprehensive for dining/entertainment/hotel, but regulatory knowledge (1 file covering multiple states) and operational knowledge (2 files) would benefit from expansion, especially for multi-state compliance.

2. **Embedding singleton inconsistency**: The `@lru_cache` vs TTLCache discrepancy for embeddings is an architectural inconsistency that will surface as a production bug during credential rotation. All other singletons have been migrated to TTLCache -- this one was missed.

3. **Multi-tenant retrieval cost**: The Firestore retriever's Python-side property_id filtering is a scalability concern. The current single-tenant deployment masks this, but multi-tenant growth will expose the 2x over-fetch cost.

---

## Decision Audit

| # | Decision | Documented? | Correct? |
|---|----------|-------------|----------|
| D1 | ChromaDB for dev, Firestore for prod | Yes (config.py, pipeline.py) | Yes |
| D2 | Per-item chunking over text splitters | Yes (pipeline.py, CLAUDE.md) | Yes |
| D3 | RRF with k=60 | Yes (reranking.py) | Yes |
| D4 | Degraded-pass validation | Yes (nodes.py:337-365) | Yes |
| D5 | Specialist DRY extraction via DI | Yes (_base.py docstring) | Yes |
| D6 | Build-time vs runtime feature flags | Yes (graph.py:379-413) | Yes |
| D7 | PII fail-closed, validator fail-open/closed by attempt | Yes (multiple files) | Yes |

---

## Summary

The codebase is architecturally mature at 82/100 after 14 rounds. The specialist agent DRY extraction, validation loop, and compliance guardrail ordering are standout patterns. The primary gap is the embedding singleton inconsistency (F5) which will manifest as a production bug under credential rotation, and the `reingest_item()` missing version stamp (F2) which creates a ghost data accumulation vector in the CMS update path. The hardcoded property description in the concierge prompt (F8) is a multi-tenant blocker that should be addressed before the second property deployment.

The RAG pipeline (SPOTLIGHT) is well-designed with per-item chunking, RRF reranking, and SHA-256 idempotent ingestion. The main gap is the `reingest_item()` version stamp omission (F2) and the RRF cost-benefit question (F6). Domain intelligence (SPOTLIGHT) is strong on regulatory coverage and comp system knowledge but thin on property-specific operational data outside of the structured JSON.

---

*Review generated: 2026-02-21*
*Reviewer: Gemini perspective (hostile, architecture-focused)*
*Commit: 8655074*
