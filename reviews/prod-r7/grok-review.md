# Production Review Round 7 — Grok 4 Reviewer
# Spotlight: RAG & DOMAIN DATA QUALITY

**Date**: 2026-02-20
**Reviewer**: Grok 4 (hostile mode)
**Model**: grok-4 via grok_reason

---

## Overall Score: 62/100

---

## Dimension Scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| RAG Pipeline Architecture | 65 | Per-item chunking good; formatters incomplete for 5/8 categories |
| Retrieval Strategy Quality | 55 | RRF fusion solid; augmented query inflexible; no category filtering used |
| Domain Data Completeness | 45 | Guest-facing data adequate; 5 KB markdown files NOT ingested |
| Embedding Configuration | 75 | Pinned model, cached correctly; no fallback or batching |
| Multi-tenant Safety | 85 | property_id filtering excellent; Firestore over-fetch handles gaps |
| Test Coverage | 80 | 49 RAG tests impressive; no E2E or load tests |
| Prompt Engineering Quality | 70 | safe_substitute, 6 validation criteria; no few-shot, hardcoded helplines |
| Error Handling & Resilience | 60 | Timeouts present; no retry, no fallback retriever chain |
| Production Readiness | 40 | No re-ingestion automation; no monitoring; manual process |
| Domain Accuracy (casino ops) | 50 | Guest data accurate; ops knowledge completely absent from RAG |

---

## Findings

### CRITICAL (2)

**F1: Knowledge base markdown files NOT ingested into vector store**
- Severity: CRITICAL
- Files: `knowledge-base/casino-operations/comp-system.md`, `knowledge-base/casino-operations/host-workflow.md`, `knowledge-base/regulations/state-requirements.md`, `knowledge-base/player-psychology/retention-playbook.md`, `knowledge-base/company-context/hey-seven-overview.md`
- Problem: 5 detailed markdown files (~1000 lines of domain knowledge) exist in `knowledge-base/` but ONLY `data/mohegan_sun.json` is ingested via `ingest_property()`. The agent cannot answer about comp calculations (ADT formula), host workflows, state-by-state regulations, player retention strategies, or self-exclusion program details beyond the basic FAQ entry. In a regulated casino domain, incomplete responsible gaming knowledge is a compliance risk.
- Impact: Guest asks "How does self-exclusion work in Connecticut?" — agent returns one FAQ sentence instead of the detailed state-requirements.md content. Guest asks "What tier credits do I need for Ascend status?" — answered from promotions data. Guest asks "How are my comps calculated?" — NO context available; agent hallucinates or says "I don't know."
- Evidence: `src/config.py:21` → `PROPERTY_DATA_PATH: str = "data/mohegan_sun.json"` — single file path, no glob, no directory ingestion. `pipeline.py:150-223` → `_load_property_json` only handles JSON, not markdown.
- Recommendation: Either (a) add a markdown loader to the ingestion pipeline that parses these files and chunks them with appropriate metadata, or (b) explicitly document that knowledge-base/ is for internal reference only (system prompt context, not RAG retrieval) and justify why ops questions are out of scope for the guest-facing concierge.

**F2: No re-ingestion automation — manual process guarantees stale data**
- Severity: CRITICAL
- File: `src/rag/pipeline.py` (ingest_property function)
- Problem: `ingest_property()` must be called manually. No startup hook in FastAPI lifespan, no file watcher, no CI/CD trigger, no scheduled job. Casino data changes frequently (promotions rotate weekly, restaurant hours change seasonally, events update daily). Stale vector store data causes incorrect answers.
- Evidence: `src/rag/pipeline.py:297-401` — function exists but is never called automatically. No reference to `ingest_property` in any startup, lifespan, or cron code.
- Impact: Hours change for Mohegan Sun Buffet (seasonal) → vector store has old hours → agent gives wrong information → guest arrives to closed restaurant.
- Recommendation: Wire `ingest_property()` into FastAPI lifespan startup (conditional on CHROMA_PERSIST_DIR being empty or stale), or add a `/admin/reingest` endpoint, or document the manual process with a clear runbook.

### HIGH (3)

**F3: _format_generic inadequate for FAQ question-answer pairs**
- Severity: HIGH
- File: `src/rag/pipeline.py:83-100`, `data/mohegan_sun.json:425-494`
- Problem: FAQ items have `question` and `answer` keys. `_format_generic` outputs them as `"Question: What are the casino hours?. Answer: Mohegan Sun's casino is open 24 hours..."` — functional but suboptimal. The question key becomes a generic metadata field label, losing the semantic signal that this IS a question-answer pair. A dedicated `_format_faq` would produce embeddings optimized for question-matching retrieval (e.g., leading with the question text for better cosine alignment with user queries that are themselves questions).
- Evidence: `_FORMATTERS` dict at line 103-111 has no mapping for "faq", "amenities", "promotions", "gaming", or "property". All 5 categories fall through to `_format_generic`.
- Impact: Retrieval precision for FAQ queries is degraded. When a user asks "Is there free parking?", the embedding of "Question: Is there free parking?. Answer: Yes..." is less aligned than "Is there free parking? Yes, Mohegan Sun offers free self-parking...".
- Recommendation: Add `_format_faq` that formats as `"{question} {answer}"` (question first, no label noise). Add `_format_promotion` for promotions (tier structure benefits from dedicated formatting). Map them in `_FORMATTERS`.

**F4: Entity-augmented query hardcodes "name location details"**
- Severity: HIGH
- File: `src/agent/tools.py:67-69`
- Problem: `search_knowledge_base` always appends `"name location details"` to the augmented query. This helps for restaurant/venue queries ("Todd English's" → "Todd English's name location details") but is irrelevant or harmful for FAQ queries ("Is there free parking?" → "Is there free parking? name location details"), gaming stats queries, or promotion questions. The augmentation biases retrieval toward venue-type results regardless of query intent.
- Evidence: Line 68 — `f"{query} name location details"` is hardcoded.
- Impact: FAQ and gaming queries get venue-biased augmented results, reducing RRF fusion quality for non-venue queries.
- Recommendation: Use query_type from the router to select augmentation strategy: venue queries → "name location details"; hours queries → already handled by `search_hours`; FAQ queries → "question answer"; gaming queries → "table games slots poker". Or remove the hardcoded augmentation and use a query-type-aware augmentation map.

**F5: No category-filtered retrieval despite support in retrieve() method**
- Severity: HIGH
- File: `src/agent/tools.py:43-99`, `src/rag/pipeline.py:450-484`
- Problem: `CasinoKnowledgeRetriever.retrieve()` accepts `filter_category` parameter and implements multi-key ChromaDB $and filtering. But neither `search_knowledge_base` nor `search_hours` use it. The router already classifies queries into categories (property_qa, hours_schedule, etc.), and the data has explicit category metadata (restaurants, entertainment, hotel, gaming, etc.). Category filtering would significantly improve precision by eliminating cross-category noise.
- Evidence: `tools.py` never passes `filter_category`. `retrieve_with_scores()` also lacks a `filter_category` parameter entirely.
- Impact: A query about "pool hours" retrieves from ALL categories when it should filter to amenities. Cross-category noise dilutes the top-k results.
- Recommendation: (a) Add `filter_category` parameter to `retrieve_with_scores()`. (b) Map router categories to retrieval categories. (c) Use category filtering as one of the RRF strategies (filtered + unfiltered) for best of both worlds.

### MEDIUM (4)

**F6: Dual ingestion_version metadata fields — redundant and confusing**
- Severity: MEDIUM
- File: `src/rag/pipeline.py:202,218,342,377`
- Problem: Each chunk gets two version fields: `"ingestion_version": "2.1"` (hardcoded string in `_load_property_json`) and `"_ingestion_version"` (dynamic ISO timestamp added in `ingest_property`). Only `_ingestion_version` is used for stale chunk purging. The hardcoded `"2.1"` serves no clear purpose — it's not referenced by any purge logic, query filter, or version check.
- Impact: Maintenance confusion. Future developers may mistake `ingestion_version` for the purge key and wonder why purging doesn't work with "2.1" comparisons.
- Recommendation: Either (a) remove the hardcoded `"ingestion_version": "2.1"` field entirely (purge uses `_ingestion_version`), or (b) rename it to `"schema_version"` to clarify it tracks the data format version, not the ingestion run.

**F7: Firestore COSINE distance-to-similarity loses granularity for dissimilar docs**
- Severity: MEDIUM
- File: `src/rag/firestore_retriever.py:117`
- Problem: `similarity = max(0.0, 1.0 - distance)` — Firestore COSINE distance is in [0, 2]. For distance > 1.0 (anti-correlated vectors), similarity collapses to 0.0. All documents with distance between 1.0 and 2.0 are indistinguishable (all score 0.0). While such documents are unlikely to be in top-k for most queries, it means the RRF fusion can't differentiate between "moderately irrelevant" and "completely irrelevant" results.
- Impact: Low — these documents are typically filtered by RAG_MIN_RELEVANCE_SCORE=0.3 anyway. But edge cases with sparse embeddings could see information loss.
- Recommendation: Acceptable as-is. Document the deliberate clamp in a code comment explaining why granularity loss for distance > 1.0 is acceptable (those docs are filtered anyway).

**F8: Test coverage gaps — no E2E pipeline test through full graph**
- Severity: MEDIUM
- File: `tests/test_rag.py`, `tests/test_retrieval_eval.py`
- Problem: 49 tests cover individual RAG components well (ingestion, retrieval, formatters, RRF, embeddings). But no test sends a query through the full graph pipeline (router → retrieve → generate → validate) with mocked LLMs to verify RAG integration. The parametrized retrieval_eval tests verify retriever category recall but not end-to-end answer quality.
- Evidence: `test_retrieval_eval.py` skips if ChromaDB is not populated — test relies on pre-existing state rather than being self-contained.
- Impact: Wiring bugs (e.g., retrieve_node → generate_node context passing, context formatting mismatches) are invisible to unit tests.
- Recommendation: Add at least one E2E test that ingest → query → verify response contains expected information from retrieved context.

**F9: No retrieval quality monitoring or metrics logging**
- Severity: MEDIUM
- File: `src/agent/tools.py`, `src/agent/nodes.py`
- Problem: No logging of retrieval metrics (average relevance score, number of results above threshold, empty result rate, category distribution). In production, retrieval quality degrades silently (e.g., embedding model drift, data staleness, threshold too aggressive). Without metrics, there's no signal to detect degradation.
- Evidence: `tools.py` logs errors but not quality metrics. `nodes.py:retrieve_node` logs timeout/error but not result quality.
- Impact: Cannot detect when retrieval starts returning irrelevant results. Casino data changes (new restaurants, closed venues) may cause gradual quality decay without alerting.
- Recommendation: Add structured logging for: result_count, mean_relevance_score, categories_returned, above_threshold_count. Export to LangFuse/LangSmith traces.

### LOW (3)

**F10: RecursiveCharacterTextSplitter configured but rarely activated**
- Severity: LOW
- File: `src/rag/pipeline.py:250-286`
- Problem: Chunk size is 800 chars with 120 char overlap. Per-item chunking means most items are 200-400 chars (well under 800). The text splitter is only triggered if an item exceeds 800 chars, which is rare with the current data. The overlap configuration is effectively dead code for the current dataset.
- Impact: None currently. If larger data files are added (e.g., long descriptions), the splitter would activate. The logging warning at line 272 is good practice.
- Recommendation: Acceptable. The splitter serves as a safety net. No action needed.

**F11: Hardcoded responsible gaming helplines — not multi-state ready**
- Severity: LOW
- File: `src/agent/prompts.py:22-28`
- Problem: `RESPONSIBLE_GAMING_HELPLINES_DEFAULT` is hardcoded to Connecticut (Mohegan Sun's jurisdiction). For multi-property deployments across states, this needs to be config-driven. The `get_responsible_gaming_helplines()` function exists as an extension point but always returns the default.
- Impact: Minimal for single-property demo. Would need refactoring for multi-property production.
- Recommendation: Acceptable for demo. Document the extension point. For production, load helplines from property config.

**F12: test_retrieval_eval relies on pre-populated ChromaDB**
- Severity: LOW
- File: `tests/test_retrieval_eval.py:19-34`
- Problem: Tests skip if ChromaDB is not populated (`pytest.skip("No retrieval results")`). This means the 5 parametrized eval tests only run in environments with pre-existing vector data. They don't set up their own test data, making them non-deterministic across environments.
- Impact: Tests may never run in CI where ChromaDB is fresh. Retrieval quality regressions go undetected.
- Recommendation: Add a fixture that ingests a minimal test dataset into a temporary ChromaDB before running eval queries.

---

## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| CRITICAL | 2 | F1, F2 |
| HIGH | 3 | F3, F4, F5 |
| MEDIUM | 4 | F6, F7, F8, F9 |
| LOW | 3 | F10, F11, F12 |
| **Total** | **12** | |

## Key Observations

1. **Strongest aspect**: Multi-tenant safety (property_id isolation) and test coverage breadth (49 tests) are above average for an interview assignment.
2. **Weakest aspect**: Domain data completeness — 5 rich markdown files with casino operations knowledge are orphaned in `knowledge-base/` and never ingested. This is the single largest scoring differentiator.
3. **Design decision to validate**: Are the knowledge-base/ markdown files intentionally excluded (internal-only reference docs) or accidentally missed? This determines whether F1 is a bug or a documented trade-off. If intentional, it should be explicitly stated in ARCHITECTURE.md.
4. **Production gap**: No ingestion automation (F2) combined with no retrieval monitoring (F9) means the system has no feedback loop — data degrades silently and nobody knows.
5. **Quick wins**: F3 (add _format_faq) and F6 (remove redundant ingestion_version) are 30-minute fixes that improve clarity and retrieval quality immediately.
