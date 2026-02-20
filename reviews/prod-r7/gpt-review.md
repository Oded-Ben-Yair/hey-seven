# Round 7 GPT-5.2 Review: RAG & Domain Data Quality

**Date**: 2026-02-20
**Reviewer**: GPT-5.2
**Spotlight**: RAG & Domain Data Quality (+1 severity boost for RAG/domain findings)
**Score**: 49/100

---

## Finding Summary

| ID | Severity | Title |
|----|----------|-------|
| GPT-F1 | CRITICAL | Non-ingestion of casino-ops markdown creates blind spot in core domain knowledge |
| GPT-F2 | CRITICAL | Domain data freshness and source-of-truth missing (no effective dates, no provenance, no update pipeline) |
| GPT-F3 | HIGH | Chunking strategy is "per-item" with late splitting only -- likely to damage retrieval recall/precision |
| GPT-F4 | HIGH | Embedding task_type not used -- likely measurable quality regression |
| GPT-F5 | HIGH | Firestore over-fetch then Python-side property_id filtering can leak relevance and waste recall |
| GPT-F6 | HIGH | Broad exception -> empty results causes silent RAG failure and unsafe model fallback |
| GPT-F7 | HIGH | Relevance threshold (0.3) is likely too permissive for operational facts |
| GPT-F8 | MEDIUM | Entity-augmented query ("name location details") is too generic and can distort intent |
| GPT-F9 | MEDIUM | RRF identity hashing (page_content+source) risks collapsing distinct field-chunks incorrectly |
| GPT-F10 | MEDIUM | Tests miss real-category coverage (amenities, promotions) and reduce confidence in schema evolution |
| GPT-F11 | MEDIUM | Retriever singleton caching without TTL is a production footgun (credentials/index config drift) |
| GPT-F12 | MEDIUM | Knowledge base coverage gaps for a real casino host AI (missing "host-critical" objects) |
| GPT-F13 | LOW | Mohegan Sun JSON domain accuracy cannot be validated; missing citations makes it untrustworthy |

**Total**: 13 findings (2 CRITICAL, 5 HIGH, 5 MEDIUM, 1 LOW)

---

## CRITICAL Findings

### GPT-F1 | CRITICAL | Non-ingestion of casino-ops markdown creates blind spot in core domain knowledge

The most "expert" content -- comp formulas (ADT calculations, reinvestment rates, approval authority), host workflow (portfolio structure, daily schedule, KPIs), regulatory constraints (self-exclusion, BSA/AML, TCPA, state-specific rules), retention playbook (churn signals, intervention timing, post-loss protocols), and company context -- lives in `knowledge-base/*.md` files that are **never ingested** into the vector store. Only `data/mohegan_sun.json` is ingested.

This means the agent will answer operational/compliance questions from the base model or thin JSON blurbs -- high risk in a casino context where wrong comp calculations, missed regulatory requirements, or incorrect self-exclusion guidance could cause regulatory violations.

**Files**:
- `knowledge-base/casino-operations/comp-system.md` (157 lines of comp formulas)
- `knowledge-base/casino-operations/host-workflow.md` (108 lines of host operations)
- `knowledge-base/regulations/state-requirements.md` (206 lines of regulatory requirements)
- `knowledge-base/player-psychology/retention-playbook.md` (223 lines of retention strategies)
- `knowledge-base/company-context/hey-seven-overview.md` (205 lines of company intel)
- `src/rag/pipeline.py` -- `_load_property_json()` only loads JSON, no markdown ingestion path

**Fix**: Add a markdown ingestion pipeline alongside JSON. Ingest `knowledge-base/` with strong metadata (`domain=casino_ops|regulatory|psychology|company`, `jurisdiction`, `effective_date`, `owner`). Add a "must-cite" policy for regulatory answers: if no retrieved chunk from `regulations/`, refuse/escalate.

---

### GPT-F2 | CRITICAL | Domain data freshness and source-of-truth missing

JSON has hours, schedules, promotions, etc., but there is no freshness strategy beyond `_ingestion_version` (which is ingestion-time, not "valid as-of"). Casino data changes frequently (restaurant hours change seasonally, entertainment schedules rotate, promotions expire). Stale info can directly harm guest experience.

The `last_updated` metadata field in each chunk is set to `datetime.now()` at ingestion time -- not when the data was last verified against reality.

**Files**:
- `data/mohegan_sun.json` -- no `last_verified_at`, `valid_from/valid_to` fields
- `src/rag/pipeline.py:201-203` -- `last_updated` is ingestion timestamp, not data freshness

**Fix**: Add per-item `source_url`, `last_verified_at`, `valid_from/valid_to` where applicable. Enforce staleness rules at retrieval time (e.g., down-rank or exclude stale hours/schedule). Build a refresh job and alerting for expired items.

---

## HIGH Findings

### GPT-F3 | HIGH | Chunking strategy may damage retrieval recall/precision for mixed-intent items

Keeping sub-800 char items as a single chunk preserves item integrity, but creates mixed-intent chunks (e.g., a restaurant chunk contains description + dress code + reservations + hours all in one). Queries like "Is Todd English's open late?" compete with irrelevant fields (dress code, reservations), potentially hurting similarity scores.

**Files**:
- `src/rag/pipeline.py:257-286` -- `_chunk_documents()` keeps small items intact
- `src/rag/pipeline.py:31-46` -- `_format_restaurant()` combines all fields

**Fix**: Consider schema-aware subchunks with field metadata (`field=hours`, `field=location`), or use query-type-specific retrieval with field-level metadata filtering.

---

### GPT-F4 | HIGH | Embedding task_type never used in production code

The `get_embeddings()` function accepts a `task_type` parameter for `RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`, but neither ingestion (`pipeline.py:354`) nor retrieval (`pipeline.py:517,570`) passes task_type. Google explicitly recommends different task types for asymmetric search (short queries vs longer documents).

**Files**:
- `src/rag/embeddings.py:20-39` -- task_type parameter supported but never called with it
- `src/rag/pipeline.py:354` -- `get_embeddings()` without task_type during ingestion
- `src/rag/pipeline.py:570` -- `get_embeddings()` without task_type during retrieval

**Fix**: During ingestion, call `get_embeddings(task_type="RETRIEVAL_DOCUMENT")`. During retrieval (ChromaDB embedding_function), use `RETRIEVAL_QUERY`. Add a regression eval (NDCG@k / recall@k) before and after using a larger query set.

---

### GPT-F5 | HIGH | Firestore property_id filtering happens post-ANN, risking recall loss

FirestoreRetriever uses `find_nearest()` without a tenant filter in the vector query, then filters by `property_id` in Python after retrieval. If the index contains multiple properties, ANN results may be dominated by other tenants, and post-hoc filtering drops most results -- potentially returning empty or low-quality context.

**Files**:
- `src/rag/firestore_retriever.py:92-124` -- Python-side property_id filter after find_nearest

**Fix**: Enforce tenant isolation inside the vector query (separate collections per property, or use Firestore composite filters if supported). At minimum, overfetch far more than 2x and add monitoring for "filtered_out_ratio".

---

### GPT-F6 | HIGH | Silent RAG failure degrades to unsafe model hallucination

The retrieve node catches all exceptions and returns empty results. In the casino domain, empty context triggers specialist agents to generate responses from base model knowledge, which can produce hallucinated hours, fake promotions, or incorrect compliance guidance.

**Files**:
- `src/agent/nodes.py:282-293` -- broad exception -> empty results
- `src/agent/agents/_base.py:107-112` -- no context -> fallback message (good) BUT only when retrieved is completely empty

**Fix**: Differentiate failure modes: (a) timeout/DB down -> "I can't access live property info right now, please contact..."; (b) no matches -> ask clarifying question. Add metrics on retrieval failure rate and "answered without context" rate.

---

### GPT-F7 | HIGH | RAG_MIN_RELEVANCE_SCORE=0.3 is too permissive for operational facts

A global 0.3 threshold across all categories (hours, gaming facts, policies, promotions) invites weak matches to pass. With a small, semantically dense dataset, a 0.3-scoring chunk may be topically wrong but geometrically close. This is especially dangerous for hours/schedules where wrong info directly impacts guest experience.

**Files**:
- `src/config.py:42` -- `RAG_MIN_RELEVANCE_SCORE: float = 0.3`
- `src/agent/tools.py:90` -- single threshold for all queries

**Fix**: Use category/field-specific thresholds (e.g., hours/schedule >= 0.6, promotions >= 0.5, generic >= 0.4). Add "min evidence count" rule for factual answers.

---

## MEDIUM Findings

### GPT-F8 | MEDIUM | Entity-augmented query is too generic

`search_knowledge_base` always appends `"name location details"` to the query. This adds little discriminative value and can push the query embedding toward generic "directory listing" semantics, hurting niche queries ("dress code at Todd English's", "roulette minimum bet").

**Files**:
- `src/agent/tools.py:68` -- `f"{query} name location details"`

**Fix**: Make augmentation conditional and query-type-aware: if user asks about location, add "location"; if about hours, add "hours schedule"; if about a specific venue, extract and add the venue name.

---

### GPT-F9 | MEDIUM | RRF identity hashing may collide on templated/similar content

Doc identity in RRF uses `SHA-256(page_content + source)`. If two items produce identical formatted text (possible with templated promotions or generic descriptions) or if chunking changes modify content slightly, IDs can collide or become unstable.

**Files**:
- `src/rag/reranking.py:45-46` -- doc_id from content+source hash

**Fix**: Include stable IDs in metadata (e.g., item_id, category, field, property_id) and use those for doc identity in RRF.

---

### GPT-F10 | MEDIUM | Test fixtures miss amenities and promotions categories

Test property data fixture has 6 categories (property, restaurants, entertainment, hotel, gaming, faq). Real data has 8 (adds amenities, promotions). Tests can pass while ingestion/retrieval for those categories regresses.

**Files**:
- `tests/conftest.py:145-198` -- test_property_data missing amenities and promotions
- `data/mohegan_sun.json` -- has amenities (7 items) and promotions (4 items)

**Fix**: Update test fixture to include all 8 categories with representative edge cases.

---

### GPT-F11 | MEDIUM | Retriever singleton uses @lru_cache instead of TTLCache

LLM clients (`_get_llm`, `_get_validator_llm`) and checkpointer use TTLCache for credential rotation (1-hour TTL). But the retriever uses `@lru_cache` which never expires, potentially holding stale Firestore clients or ChromaDB connections across key rotation in long-lived workers.

**Files**:
- `src/rag/pipeline.py:527` -- `@lru_cache(maxsize=1)` for `_get_retriever_cached()`
- `src/agent/nodes.py:117` -- `_llm_cache: TTLCache` for comparison

**Fix**: Use TTLCache(maxsize=1, ttl=3600) consistent with other singletons, or rebuild on auth errors.

---

### GPT-F12 | MEDIUM | Knowledge base coverage gaps for host-grade assistant

Current JSON is mostly "brochure" content. A real casino host agent needs: detailed loyalty program earn/burn mechanics by tier, comp policies per game segment, outlet-specific policies (reservation procedures, cancellation), transportation/parking details, accessibility info, responsible gaming/self-exclusion procedures, incident escalation contacts, and property wayfinding.

**Files**:
- `data/mohegan_sun.json` -- promotions section has tier names/descriptions but lacks earn rates, tier thresholds, specific benefits per tier

**Fix**: Add structured categories: `loyalty_program` (detailed), `transportation`, `parking`, `accessibility`, `responsible_gaming`, `security_escalation`, `policies`, `wayfinding`.

---

## LOW Findings

### GPT-F13 | LOW | Domain data accuracy unverifiable without citations

The Mohegan Sun JSON has no `source_url` or citation fields. Hours, capacities, counts, and offerings cannot be verified against authoritative sources. In production, unverifiable claims are a liability.

**Files**:
- `data/mohegan_sun.json` -- no citation/source fields per item

**Fix**: Add citations per item plus a validation script that checks against authoritative web pages. Store `confidence` and `verified_at` fields.

---

## Direct Answers to Flagged Issues

| Issue | Verdict | Finding |
|-------|---------|---------|
| A. Markdown not ingested | **Major problem** -- core domain knowledge is invisible to RAG | GPT-F1 (CRITICAL) |
| B. task_type unused | **Production concern** -- expect ranking quality loss | GPT-F4 (HIGH) |
| C. Test fixture missing categories | **Coverage gap** -- won't protect promotions/amenities | GPT-F10 (MEDIUM) |
| D. lru_cache vs TTLCache | **Moderate concern** -- especially for Firestore auth rotation | GPT-F11 (MEDIUM) |
| E. Generic entity augmentation | **Moderate concern** -- make it intent-aware | GPT-F8 (MEDIUM) |
| F. RAG_MIN_RELEVANCE_SCORE=0.3 | **Too permissive** for hours/policies; per-category thresholds needed | GPT-F7 (HIGH) |
| G. Domain accuracy | **Unknown/unauditable** -- no citations or freshness metadata | GPT-F2, GPT-F13 |
| H. Missing categories | **Yes** -- host-grade assistant needs more operational/policy coverage | GPT-F12 (MEDIUM) |

---

## Score Breakdown

| Dimension | Score | Notes |
|-----------|-------|-------|
| RAG Architecture | 6/15 | Good foundations (per-item chunking, RRF, SHA-256 IDs, stale purging) but critical gaps (no markdown ingestion, no task_type, permissive threshold) |
| Domain Data Quality | 4/15 | Rich JSON data but no freshness, no citations, no provenance. Markdown files not ingested. Coverage gaps. |
| Retrieval Robustness | 7/15 | Multi-strategy RRF, timeout guards, graceful degradation. But silent failures, generic augmentation, single threshold. |
| Multi-tenant Safety | 8/10 | property_id metadata isolation in ChromaDB is solid. Firestore post-hoc filtering is a concern. |
| Test Coverage | 7/10 | Comprehensive RAG tests (35+ tests). Missing 2 categories in fixture. Offline eval is minimal (5 queries). |
| Embedding Strategy | 5/10 | Pinned model (good), lru_cache (acceptable for dev). task_type unused (bad). |
| Data Model / Schema | 6/10 | Category-specific formatters, nested dict flattening. No field-level metadata for fine-grained retrieval. |
| Production Readiness | 6/15 | ChromaDB/Firestore abstraction, lazy imports, singleton patterns. But lru_cache retriever, no staleness rules, no failure differentiation. |

**Total: 49/100**

---

## Score Trend

| Round | Spotlight | Gemini | GPT | Grok | Avg |
|-------|-----------|--------|-----|------|-----|
| R1 | General | -- | -- | -- | 67.3 |
| R2 | Security | -- | -- | -- | 61.3 |
| R3 | Observability | -- | -- | -- | 60.7 |
| R4 | Domain | -- | -- | -- | 66.7 |
| R5 | Scalability & Async | -- | -- | -- | 63.3 |
| R6 | API Contract & Docs | 61 | 58 | 54 | 57.7 |
| R7 | RAG & Domain Data | -- | **49** | -- | -- |
