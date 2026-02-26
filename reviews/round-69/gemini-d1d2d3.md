# R69 Gemini Review — D1, D2, D3

## Date: 2026-02-26
## Reviewer: Gemini 3.1 Pro (via Claude Opus 4.6 hostile review agent)
## R68 Baseline: D1=9.4, D2=9.1, D3=9.2

---

## D1: Graph Architecture (weight 0.20)

**Score: 9.0/10** (down from 9.4)

### Findings

- **[MAJOR] dispatch.py:232-244 — CB ValueError conflation allows rate limiting to bypass circuit breaker.** The `except (ValueError, TypeError)` block catches both structured output parse failures AND google-genai SDK errors for 429/auth/quota exhaustion. The comment on line 239-244 explicitly acknowledges this as "a known limitation." When Gemini returns 429 Resource Exhausted, the SDK wraps it in a `ValueError`. This catch block does NOT call `cb.record_failure()`, so sustained rate limiting never trips the circuit breaker. The CB's core function — detecting LLM unavailability — is partially blind to one of the most common failure modes (rate limiting). Fix: inspect `exc.args` or `str(exc)` for known rate-limit patterns like "429" or "Resource has been exhausted" and call `record_failure()` for those while keeping the no-record behavior for genuine parse errors.

- **[MAJOR] _base.py:104-396 — execute_specialist is 293 lines with 7+ responsibilities. SRP violation.** This single function handles: (1) circuit breaker check, (2) system prompt assembly with casino profile, (3) guest context injection, (4) whisper plan injection, (5) persona style injection, (6) sentiment-adaptive tone guidance, (7) frustration escalation (HEART framework), (8) proactive suggestion injection, (9) message history windowing with persona drift prevention, (10) semaphore backpressure with timeout, (11) LLM invocation, (12) multi-type error handling, (13) suggestion_offered tracking. The 100-LOC SRP threshold from the rubric is exceeded by 3x. This was flagged in prior rounds (R34-R43) for the dispatch function but execute_specialist has grown to similar size. Extract: prompt_assembly, context_injection, and message_building into separate helpers.

- **[MINOR] dispatch.py:352 — Specialist timeout hardcoded as 2x MODEL_TIMEOUT.** `asyncio.timeout(settings.MODEL_TIMEOUT * 2)` is not independently configurable. If MODEL_TIMEOUT changes, specialist timeout changes implicitly. Should be a separate setting (e.g., `SPECIALIST_TIMEOUT`) with a default of `MODEL_TIMEOUT * 2` for backward compatibility.

- **[MINOR] firestore_retriever.py:41,127 — _server_filter_request_count incremented without synchronization.** Module-level global `_server_filter_request_count` is incremented on line 127 with `_server_filter_request_count += 1` — a read-modify-write operation that is not atomic even under the GIL when accessed from coroutines on the same event loop. Under concurrent async requests, two coroutines could read the same value. Impact is cosmetic (retry timing off by a few requests), not a correctness issue.

- **[MINOR] graph.py:505 — Node timing not emitted for nodes that fail without on_chain_end.** If a node raises an exception before on_chain_end fires, `node_start_times` retains the entry. However, since `pop()` is used on line 572, retries DO get proper timing. The gap is only for nodes that crash mid-execution (no duration_ms event emitted for the crashed execution). Low impact since the error event captures the failure.

### D1 Strengths
- Clean 11-node topology with defense-in-depth (compliance_gate + router)
- Parity check at import time (ValueError, not assert)
- _route_after_validate_v2 handles unexpected validation_result defensively
- Dispatch-owned key stripping (R47 fix) prevents specialist state corruption
- Feature flag dual-layer architecture (build-time topology, runtime behavior) is well-documented
- GraphRecursionError handled in both chat() and chat_stream()
- Streaming PII redactor with fail-safe on CancelledError (drops buffer, doesn't emit)

---

## D2: RAG Pipeline (weight 0.10)

**Score: 9.0/10** (down from 9.1)

### Findings

- **[MINOR] pipeline.py:352 — reingest_item uses asyncio.to_thread for ChromaDB writes.** This is architecturally correct (ChromaDB's SQLite is blocking I/O), but under heavy CMS webhook traffic, each reingest_item call consumes a thread pool slot. The default ThreadPoolExecutor has `min(32, cpu_count+4)` threads. 20+ concurrent CMS updates during a content refresh would exhaust the pool. Consider a dedicated executor with bounded queue for CMS operations.

- **[MINOR] pipeline.py:513 — datetime.now() called per item inside _load_property_json loop.** Items within the same ingestion batch get slightly different `last_updated` timestamps. This is cosmetically inconsistent but does not affect the version-stamp purging mechanism (which uses `_ingestion_version` set outside the loop on line 769). Non-functional.

- **[MINOR] pipeline.py:597 — Markdown splitting only on `## ` (h2) headings.** The regex `re.split(r"\n(?=## )", text)` splits on h2 but not h3 (`### `) or deeper. A regulation document with one `## Overview` heading followed by 15 `### ` subsections would produce a single oversized chunk that triggers the text splitter fallback (pipeline.py:671-686), fragmenting structured regulatory content at arbitrary character boundaries. Consider splitting on `##+ ` (any heading level).

- **[MINOR] pipeline.py:930-948 — CasinoKnowledgeRetriever.retrieve() has no relevance score filtering.** Unlike `retrieve_with_scores()` which accepts `min_score`, the `retrieve()` method returns all results regardless of similarity. This is not actively harmful since `retrieve_with_scores()` is the primary path used by tools.py, but `retrieve()` is exposed in the AbstractRetriever interface and could be misused by future callers.

- **[MINOR] firestore_retriever.py:236-237 — property_id derivation duplicated.** `settings.PROPERTY_NAME.lower().replace(" ", "_")` appears in at least 5 locations across pipeline.py and firestore_retriever.py. Should be extracted to a helper function in config.py to prevent derivation drift if the normalization logic ever changes.

### D2 Strengths
- Per-item chunking with category-specific formatters (restaurant, entertainment, hotel, gaming, amenity, promotion, faq)
- SHA-256 content hashing with null-byte delimiter prevents ID collisions
- Version-stamp purging after upsert (both bulk and CMS single-item)
- RRF reranking with dual scores (cosine for quality gate, RRF for ranking)
- Provenance citations in respond_node: {category, source, score} — R68 fix VERIFIED
- Production guard blocking ChromaDB in production environment
- Embedding health check before caching (prevents broken client cached for 1 hour)
- Retry with exponential backoff for both ingest_property and reingest_item

---

## D3: Data Model (weight 0.10)

**Score: 8.8/10** (down from 9.2)

### Findings

- **[MAJOR] validators.py — Runtime validators are dead code in production.** `validate_retrieved_chunk` and `validate_guest_profile` are defined in `src/data/validators.py` and exported via `__all__`. However, they are ONLY imported in `tests/test_validators.py`. No production code (src/) imports or calls these validators. Grep across the entire `src/` directory confirms zero imports. The R68 review stated "Runtime validators for chunks and guest profiles" as a fix, but the validators were created without being wired into the production code path. They should be called in: (1) `retrieve_node` (nodes.py) before adding chunks to `retrieved_context`, and (2) `get_guest_profile` (guest_profile.py) after loading from Firestore/memory. Without production wiring, data corruption from Firestore deserialization bugs or schema drift passes silently through the pipeline.

- **[MINOR] guest_profile.py:418-446 — Schema migration has no extensibility framework.** `_migrate_profile` uses a simple `if schema_version < 2:` check. Adding v3 requires manually adding another `if schema_version < 3:` block. This is acceptable for 2 versions but becomes fragile at 5+. Consider a migration registry pattern: `_MIGRATIONS = {2: _migrate_v1_to_v2, 3: _migrate_v2_to_v3}` with a loop.

- **[MINOR] guest_profile.py:171-275 — update_guest_profile merge only 2 levels deep.** The merge logic iterates `section_key → field_key` but does not recurse into deeper nesting. Updating `preferences.dining.dietary_restrictions` requires the caller to provide `{"preferences": {"dining": {"dietary_restrictions": {...}}}}`, which works because the outer loop enters `preferences`, then the inner loop enters `dining`, but only if `dining` is already a dict in the profile. If `preferences.dining` is missing, `section_updates` at the `preferences` level is `{"dining": {...}}` which goes through the non-ProfileField path and sets it directly. This actually works correctly for the current schema depth. Downgraded from concern to observation.

- **[MINOR] state.py:134-149 vs models.py:109-116 — GuestContext and Preferences have different shapes.** `GuestContext.preferences` is `list[str]` while the full `Preferences` TypedDict has nested sub-dicts. This is intentional (GuestContext is a flattened view for LLM injection, not the full profile), but the field name collision could confuse developers. Consider renaming to `preference_tags` or adding a docstring clarifying the distinction.

- **[MINOR] models.py:299-346 — apply_confidence_decay modifies profile in-place.** The docstring says "modified in-place and returned" but `get_agent_context` (guest_profile.py:398-415) first does `copy.deepcopy(profile)` before calling `apply_confidence_decay`. This is correct, but if any caller forgets the deepcopy, the original profile is mutated. Consider making `apply_confidence_decay` create its own copy internally for safety.

### D3 Strengths
- TypedDict state with 4 custom reducers: add_messages, _merge_dicts (tombstone), _keep_max, _keep_truthy
- UNSET_SENTINEL uses UUID-namespaced string for JSON serialization safety
- _merge_dicts filters None and empty string, supports explicit deletion
- Parity check at import time catches state schema drift
- MappingProxyType for FIELD_WEIGHTS (immutable module-level data)
- CCPA cascade delete with Firestore batch overflow guard (490-op limit)
- Confidence system: decay, confirm boost, contradict penalty with ceiling/floor guards
- _empty_profile sets _schema_version from module constant

---

## R68 Fix Verification

- **[NOT VERIFIED] CB exception handling conflates ValueError types** — The comment on dispatch.py:239-244 explicitly acknowledges "a known limitation." The ValueError catch block (line 232) still does NOT call `record_failure()` for rate-limit errors wrapped as ValueError. This is documented but not resolved.
- **[VERIFIED] RAG provenance citations** — respond_node (nodes.py:437-450) returns `{category, source, score}` dicts. SSE sources event (graph.py:623-627) includes `citations` field. Fix confirmed.
- **[PARTIALLY VERIFIED] Runtime validators for chunks and guest profiles** — validators.py exists with correct validation logic and test coverage. However, the validators are NOT called from any production code path. They are test-only utilities, not runtime validators. The fix is incomplete.
- **[VERIFIED] Guest profile schema migration** — `_migrate_profile` (guest_profile.py:418-446) handles v1→v2 migration. Called from `get_guest_profile` (line 157) and in-memory fallback (line 167). Migration is idempotent and logged.

---

## Summary

| Dimension | R68 Baseline | R69 Score | Delta | Key Issue |
|-----------|-------------|-----------|-------|-----------|
| D1 Graph Architecture | 9.4 | 9.0 | -0.4 | CB ValueError blindspot, execute_specialist SRP |
| D2 RAG Pipeline | 9.1 | 9.0 | -0.1 | Minor thread/timestamp/splitting issues |
| D3 Data Model | 9.2 | 8.8 | -0.4 | Dead-code validators (exist but unwired) |

**CRITICALs: 0** | **MAJORs: 3** | **MINORs: 11**

### MAJOR Findings Summary
1. **D1 dispatch.py:232** — CB blind to rate-limit ValueError from google-genai SDK
2. **D1 _base.py:104-396** — execute_specialist 293 LOC, 7+ responsibilities, 3x over SRP limit
3. **D3 validators.py** — validate_retrieved_chunk and validate_guest_profile never called in production

### Weighted Score
`(9.0 * 0.20) + (9.0 * 0.10) + (8.8 * 0.10) = 1.80 + 0.90 + 0.88 = 3.58 / 4.0 = 89.5%`
