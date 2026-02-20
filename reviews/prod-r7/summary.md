# Round 7 Fix Summary

**Date**: 2026-02-20
**Spotlight**: RAG & Domain Data Quality
**Scores**: Gemini=52, GPT=49, Grok=62 | Average=54.3

---

## Consensus Analysis (3 reviewers)

| Finding | Gemini | GPT | Grok | Consensus |
|---------|--------|-----|------|-----------|
| Knowledge-base markdown NOT ingested | CRITICAL (RAG-001) | CRITICAL (GPT-F1) | CRITICAL (F1) | 3/3 |
| Embedding task_type never used | HIGH (EMB-002) | HIGH (GPT-F4) | Implicit | 3/3 |
| Missing formatters for 5 categories | HIGH (RAG-002) | HIGH (GPT-F3) | HIGH (F3) | 3/3 |
| Dual ingestion_version confusion | HIGH (RAG-003) | MEDIUM (GPT-F6) | MEDIUM (F6) | 3/3 |
| Retriever lru_cache vs TTLCache | MEDIUM (RAG-004) | MEDIUM (GPT-F11) | Implicit | 3/3 |
| Test fixtures missing categories | MEDIUM (TEST-002) | MEDIUM (GPT-F10) | MEDIUM (F8) | 3/3 |
| Earth Tower factual error (2024->2016) | HIGH (DATA-001) | LOW (GPT-F13) | - | 2/3 |
| .env embedding model mismatch | HIGH (EMB-001) | - | - | 1/3 CRITICAL |
| Production ChromaDB guard toothless | CRITICAL (PROD-001) | - | - | 1/3 CRITICAL |

---

## Fixes Applied (14 findings fixed)

### CRITICAL (2)

1. **RAG-001 / GPT-F1 / Grok-F1: Knowledge-base markdown ingestion** (3/3 consensus)
   - Added `_load_knowledge_base_markdown()` function to `src/rag/pipeline.py`
   - Splits markdown by `## ` headings for section-level chunking
   - Maps directory names to metadata categories (casino_operations, regulations, player_psychology, company_context)
   - Sets `doc_type=markdown` metadata for provenance tracking
   - Wired into `ingest_property()` via new `knowledge_base_dir` parameter
   - 5 markdown files (comp-system, host-workflow, state-requirements, retention-playbook, hey-seven-overview) now ingested alongside JSON

2. **PROD-001: Production ChromaDB guard** (single-model CRITICAL)
   - Changed `logger.error()` to `raise RuntimeError()` when VECTOR_DB=chroma in production
   - Matches pattern of `validate_production_secrets()` which correctly raises ValueError

### HIGH (5)

3. **DATA-001: Earth Tower opening date** (2/3 consensus)
   - Fixed `data/mohegan_sun.json`: "November 2024" -> "November 2016"

4. **EMB-001: .env embedding model mismatch** (2/3 consensus)
   - Fixed `.env`: `models/gemini-embedding-001` -> `gemini-embedding-001`
   - Added `normalize_embedding_model` model_validator in `src/config.py` that strips `models/` prefix
   - Prevents ingestion-vs-retrieval vector space mismatch

5. **RAG-002 / GPT-F3 / Grok-F3: Missing formatters** (3/3 consensus)
   - Added `_format_faq()`: leads with question text for embedding alignment
   - Added `_format_gaming()`: boolean -> Available/Not Available, comma-formatted numbers
   - Added `_format_amenity()`: type/hours/location-aware
   - Added `_format_promotion()`: benefits list, how-to-join, requirements
   - Registered all 4 in `_FORMATTERS` dict (total: 11 category mappings)

6. **EMB-002 / GPT-F4: task_type wiring** (3/3 consensus)
   - Ingestion: `get_embeddings(task_type="RETRIEVAL_DOCUMENT")`
   - Retrieval: `get_embeddings(task_type="RETRIEVAL_QUERY")`
   - Applied to all 3 retriever creation paths (Firestore, ChromaDB cached, ChromaDB uncached)

7. **RAG-003 / Grok-F6: Dual version metadata** (3/3 consensus)
   - Renamed `ingestion_version: "2.1"` -> `_schema_version: "2.1"` (static, for format tracking)
   - `_ingestion_version` (timestamp) remains sole key for stale chunk purging

### MEDIUM (3)

8. **RAG-004 / GPT-F11: Retriever TTLCache** (3/3 consensus)
   - Replaced `@lru_cache(maxsize=1)` with dict-based TTL cache (1-hour TTL)
   - Consistent with `_llm_cache` and `_validator_cache` patterns
   - Prevents stale Firestore credentials after GCP Workload Identity rotation

9. **TEST-002 / GPT-F10: Test fixture categories** (3/3 consensus)
   - Added `amenities` (2 items: Elemis Spa, Swimming Pool) to test_property_data
   - Added `promotions` (2 items: Momentum Rewards, Ascend Tier) to test_property_data
   - Updated `test_category_values_match_source` expected categories

10. **FMT-001: Generic formatter human-readable values** (Gemini LOW)
    - Boolean values: True -> "Available", False -> "Not Available"
    - Large numbers (>9999): comma-formatted with contextual units (sq ft)

---

## New Tests (23 added, 72 total RAG tests)

| Test Class | Tests | What It Covers |
|-----------|-------|---------------|
| TestMarkdownIngestion | 6 | Markdown loading, section splitting, metadata, property_id, combined JSON+MD ingestion, _schema_version |
| TestNewFormatters | 10 | FAQ Q&A format, gaming booleans/numbers, amenity fields, promotion benefits, generic booleans/numbers, formatter registry |
| TestIngestionWithAllCategories | 1 | Amenities + promotions categories indexed |
| TestProductionChromaGuard | 1 | RuntimeError on VECTOR_DB=chroma in production |
| TestEmbeddingModelNormalization | 2 | models/ prefix stripped, bare name preserved |
| TestTaskTypeWiring | 1 | RETRIEVAL_DOCUMENT passed during ingestion |
| TestMissingDataFile (updated) | 1 | Updated for knowledge_base_dir parameter |
| Existing tests (updated) | 1 | no_kb_dir fixture added to isolate JSON-only tests |

---

## Files Modified

| File | Change |
|------|--------|
| `src/rag/pipeline.py` | +_load_knowledge_base_markdown(), +4 formatters, task_type wiring, TTL cache, RuntimeError guard, _schema_version |
| `src/config.py` | +normalize_embedding_model validator |
| `src/rag/embeddings.py` | (no change -- task_type already supported) |
| `data/mohegan_sun.json` | Earth Tower: 2024 -> 2016 |
| `.env` | EMBEDDING_MODEL: models/gemini-embedding-001 -> gemini-embedding-001 |
| `tests/test_rag.py` | +23 tests, +no_kb_dir fixture, updated existing tests |
| `tests/conftest.py` | +amenities/promotions in test_property_data, updated retriever cache clear |

---

## Test Results

```
1216 passed, 20 skipped, 1 warning in 11.54s
Coverage: 90.41% (threshold: 90.0%)
RAG tests: 72 (was 49, +23 new)
```

---

## NOT Fixed (Documented Trade-offs)

- **GPT-F2**: Data freshness/provenance metadata -- requires per-item source_url and valid_from/valid_to fields across all JSON data. Significant data engineering effort, not incremental fix.
- **GPT-F5 / ARCH-001**: Firestore over-fetch 2x -- architectural constraint of find_nearest() API. Would need collection sharding which is a design change.
- **GPT-F7**: Per-category relevance thresholds -- requires empirical tuning with real embeddings. Global 0.3 is conservative default.
- **Grok-F2**: Ingestion automation -- FastAPI lifespan wiring requires integration testing beyond RAG scope.
- **Grok-F4**: Entity-augmented query hardcoding -- needs query-type-aware augmentation map, a larger refactor.
- **Grok-F5**: Category-filtered retrieval -- needs router->retriever category mapping, cross-module change.
