# Production Hostile Review — Round 7 (Gemini Reviewer)

**Date**: 2026-02-20
**Spotlight**: RAG & DOMAIN DATA QUALITY (+1 severity on dimensions 2 and 10)
**Reviewer**: Gemini 3 Pro (thinking_level=high)
**Score Trajectory**: R1=67.3 -> R2=61.3 -> R3=60.7 -> R4=66.7 -> R5=63.3 -> R6=57.7 -> **R7=52.0**

---

## Findings (14 total)

### CRITICAL (2 findings)

| ID | Severity | File | Line(s) | Problem | Production Impact | Fix |
|----|----------|------|---------|---------|-------------------|-----|
| RAG-001 | CRITICAL | `src/rag/pipeline.py` | entire `ingest_property()` | **Strategic domain blindness**: RAG ingestion ONLY processes `PROPERTY_DATA_PATH` (JSON). The `knowledge-base/` directory containing 4 critical Markdown files (comp formulas, host workflow, state regulations, retention playbook) is NEVER ingested, NEVER embedded, NEVER searchable. `CLAUDE.md` line 109 labels `knowledge-base/` as "Structured data for RAG ingestion" — this is overclaimed documentation (Rule: Documentation Honesty). | Agent knows restaurant menus but is blind to NJ/NV gaming regulations, comp calculation formulas, BSA/AML requirements. When asked "What are the self-exclusion rules in New Jersey?", agent will hallucinate instead of retrieving from `knowledge-base/regulations/state-requirements.md`. In a regulated casino environment, this is a compliance liability. | Implement multi-format ingestion: add `RecursiveCharacterTextSplitter` path for Markdown files with appropriate `doc_type` or `source_type` metadata. Ingest alongside JSON data. |
| PROD-001 | CRITICAL | `src/rag/pipeline.py` | 558-563 | **Production vector DB guard is toothless**: When `ENVIRONMENT=production` and `VECTOR_DB=chroma`, the code logs `logger.error(...)` but does NOT abort. The application starts, accepts requests, and serves from ephemeral ChromaDB which loses all data on container restart. | Data loss on every Cloud Run restart. Agent serves empty RAG results with no warning to users. Casino host gives fabricated answers because retrieval returns nothing. | Replace `logger.error()` with `raise RuntimeError(...)` or `sys.exit(1)`. Production MUST hard-fail on misconfigured vector DB. Pattern: same as `validate_production_secrets()` in config.py which correctly raises ValueError. |

### HIGH (5 findings)

| ID | Severity | File | Line(s) | Problem | Production Impact | Fix |
|----|----------|------|---------|---------|-------------------|-----|
| RAG-002 | HIGH (+1 spotlight) | `src/rag/pipeline.py` | 103-111 | **Incomplete formatter coverage**: `_FORMATTERS` dict maps only 7 category keys (restaurants, dining, entertainment, shows, hotel, rooms, hotel_rooms). Five categories present in `mohegan_sun.json` — `amenities`, `gaming`, `faq`, `promotions`, `property` — all fall through to `_format_generic`. The generic formatter produces lower-quality text representations (e.g., `"Poker Room: True."` for boolean values, `"Casino Size Sqft: 364000."` for raw numbers without context). | Degraded retrieval quality for gaming queries ("What table games are available?"), promotion queries ("How do I join Momentum Rewards?"), amenity queries ("Tell me about the spa"), and FAQ queries. Lower-quality embeddings -> lower cosine similarity -> more filtered results -> agent says "I don't have information about that" when the data EXISTS in the vectorstore. | Add dedicated formatters: `_format_gaming` (tables, games list, poker room details), `_format_faq` (Q&A pair format), `_format_amenity` (type/hours/description), `_format_promotion` (loyalty tiers, how to join). |
| RAG-003 | HIGH | `src/rag/pipeline.py` | 202-203, 341-342 | **Dual version metadata confusion**: Two nearly-identical metadata fields with different values: (1) `ingestion_version` (no underscore) = hardcoded `"2.1"` (semantic version), set during JSON loading. (2) `_ingestion_version` (underscore prefix) = dynamic ISO timestamp, set during embedding. The purge logic (line 377) filters by `_ingestion_version` (timestamp). The static `ingestion_version` serves no operational purpose and creates confusion about which field controls freshness. | Developer confusion leads to bugs: someone filters by `ingestion_version` expecting timestamps, gets `"2.1"`. Or updates the hardcoded version without understanding the timestamp-based purging. Stale data may persist if the wrong field is checked. | Remove `ingestion_version` (static) from metadata. Use ONLY `_ingestion_version` (timestamp) for all version-related operations. If a semantic version is needed for migration tracking, name it `_schema_version` to avoid confusion. |
| DATA-001 | HIGH (+1 spotlight) | `data/mohegan_sun.json` | 237 | **Factual error in source data**: `"opened": "November 2024"` for Earth Tower. Mohegan Sun's Earth Tower opened November 18, **2016** (confirmed via Mohegan Sun Newsroom press release). The JSON claims 2024, which is 8 years off. | Agent confidently tells guests the Earth Tower opened in 2024. Reputational damage: any casino employee or regular guest will immediately know this is wrong, destroying trust in the AI host. In an interview assignment context, this signals lack of data validation rigor. | Correct to `"opened": "November 2016"`. Implement a data validation checklist: all dates, capacities, and proper nouns in `mohegan_sun.json` must be verified against primary sources before ingestion. |
| EMB-001 | HIGH (+1 spotlight) | `.env` vs `src/config.py` | .env:26, config.py:35 | **Embedding model name mismatch**: `.env` has `EMBEDDING_MODEL=models/gemini-embedding-001` (with `models/` prefix). `config.py` default is `gemini-embedding-001` (without prefix). `.env.example` has `gemini-embedding-001` (without prefix). When `.env` is loaded, the embedding model name includes the `models/` prefix, which may resolve to a different API endpoint or fail silently depending on the Google GenAI SDK version. | Potential embedding dimension mismatch between ingestion-time (if run with `.env` loaded) and query-time (if run with config default). Different embedding model = different vector space = retrieval returns garbage. This is the exact "embedding model version drift" anti-pattern documented in the RAG production rules. | Remove `models/` prefix from `.env` line 26. Add a `model_validator` in Settings that strips `models/` prefix from EMBEDDING_MODEL if present. Add test asserting `.env` and `.env.example` and config.py default all produce the same model string. |
| EMB-002 | HIGH (+1 spotlight) | `src/rag/embeddings.py` | 21-39 | **task_type parameter never used**: `get_embeddings()` accepts `task_type` for Google's asymmetric embedding optimization (`RETRIEVAL_DOCUMENT` at ingestion, `RETRIEVAL_QUERY` at search). The parameter is NEVER passed at any call site — grep confirms all callers use `get_embeddings()` with no arguments. The `@lru_cache(maxsize=4)` suggests intent to cache per task_type, but this capacity is wasted. | Suboptimal embedding quality: Google's gemini-embedding-001 produces better retrieval when told whether the input is a document or a query. Without task_type differentiation, all embeddings use the default task type, reducing retrieval precision by an estimated 2-5% (per Google's benchmarks). | Pass `task_type="RETRIEVAL_DOCUMENT"` in `ingest_property()` and `task_type="RETRIEVAL_QUERY"` in retriever `retrieve_with_scores()`. This is a free quality improvement. |

### MEDIUM (5 findings)

| ID | Severity | File | Line(s) | Problem | Production Impact | Fix |
|----|----------|------|---------|---------|-------------------|-----|
| RAG-004 | MEDIUM | `src/rag/pipeline.py` | 527 | **Retriever singleton uses @lru_cache, not TTLCache**: LLM singletons (`_llm_cache`, `_validator_cache`, `_whisper_cache`) all use `TTLCache(maxsize=1, ttl=3600)` for GCP credential rotation. The retriever singleton uses `@lru_cache(maxsize=1)` which NEVER expires. If Firestore client credentials rotate (which they do with Workload Identity), the cached retriever holds a stale client indefinitely. | Firestore retrieval silently fails after credential rotation (~1 hour). All RAG queries return empty results. Agent gives generic responses without context. Requires container restart to recover. | Replace `@lru_cache` with `TTLCache(maxsize=1, ttl=3600)` consistent with other singletons. Add async lock for thread safety. |
| TEST-001 | MEDIUM | `tests/test_retrieval_eval.py` | 19-34 | **Retrieval evaluation test is effectively dead code**: The test skips if ChromaDB is not populated (`pytest.skip`), which is ALWAYS the case in CI. The 5 query->category eval pairs never execute in automated testing. | Zero regression coverage for retrieval quality. A change to embedding model, chunk size, or formatter could degrade retrieval to zero without any test failing. False confidence from green CI. | Refactor: create a fixture that force-ingests the test_property_data into a temp ChromaDB before running eval. Remove the skip condition. |
| TEST-002 | MEDIUM | `tests/conftest.py` | 144-198 | **Test property data missing 3 categories**: Test fixture includes restaurants, entertainment, hotel, gaming, faq. Missing: `amenities`, `promotions`, `property`. Combined with finding RAG-002 (missing formatters), these categories have zero test coverage end-to-end. | Formatter bugs for amenities/promotions go undetected. The `_format_generic` fallback may produce malformed text for these categories without any test catching it. | Add amenities, promotions, and property categories to `test_property_data` fixture. Add formatter assertions for each. |
| ARCH-001 | MEDIUM | `src/rag/firestore_retriever.py` | 93 | **Over-fetching 2x from Firestore**: `limit=top_k * 2` fetches double the requested documents, then filters by property_id in Python. While documented ("post-hoc Python property_id filtering"), this wastes 50% of Firestore read operations. | Doubled Firestore read costs and latency. At scale (1000 req/min), this is 1000 unnecessary reads per minute. With Firestore's per-read pricing, this adds up. | Use Firestore's `where("metadata.property_id", "==", property_id)` composite filter alongside `find_nearest()` if supported. If not, shard collections by property_id for zero waste. |
| DOC-001 | MEDIUM | `CLAUDE.md` | 109 | **Documentation overclaim**: `CLAUDE.md` describes `knowledge-base/` as "Structured data for RAG ingestion" but no code ingests these files. This violates the Documentation Honesty vocabulary rule: the knowledge-base files are neither "Implemented" nor "Wired" — they are at best "Scaffolded" (data exists but is not consumed by any production code path). | Misleading project documentation. A reviewer or new developer assumes the knowledge-base is indexed and searchable when it is not. Design reviews based on this assumption will miss the domain blindness bug (RAG-001). | Update CLAUDE.md to accurately describe knowledge-base/ as "Domain reference documents (NOT currently ingested into RAG — see RAG-001)". |

### LOW (2 findings)

| ID | Severity | File | Line(s) | Problem | Production Impact | Fix |
|----|----------|------|---------|---------|-------------------|-----|
| RAG-005 | LOW | `src/rag/pipeline.py` | 371-393 | **Stale chunk purge accesses private `_collection` attribute**: `vectorstore._collection` is a private attribute of the LangChain Chroma wrapper. This may break on LangChain version upgrades without warning. | Stale chunk purging silently fails if LangChain changes internal attribute names. Ghost data accumulates but is caught by the broad except on line 387. Non-critical but creates technical debt. | Use `vectorstore._client.get_collection()` or the public `Chroma` API if available. Add a defensive assertion: `assert hasattr(vectorstore, '_collection')` with a clear error message referencing the LangChain version. |
| FMT-001 | LOW | `src/rag/pipeline.py` | 83-100 | **_format_generic produces low-quality text for boolean and numeric values**: `_format_generic` outputs `"Poker Room: True."` and `"Casino Size Sqft: 364000."` without human-readable formatting. These become embedding inputs with poor semantic signal. | Minor retrieval quality degradation for gaming category queries where boolean/numeric fields produce unnatural text chunks. | Add type-specific formatting in `_format_generic`: booleans -> "Poker Room: Available", large numbers -> "Casino Size: 364,000 sq ft". |

---

## Dimension Scores

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | Graph Architecture | 6.0 | Solid StateGraph with validation loop, but architectural gap between knowledge-base and RAG pipeline undermines domain coverage. |
| 2 | **RAG Pipeline** | **3.0** | **SPOTLIGHT FAILURE.** Critical: knowledge-base not ingested. Missing formatters for 5/10 categories. Dual version metadata. Embedding model mismatch in .env. task_type never used. Production ChromaDB guard is toothless. |
| 3 | Data Model | 5.0 | Source JSON has factual error (Earth Tower date). Structured data schema is good. Metadata design (property_id, category, source) is correct. |
| 4 | API Design | 7.0 | Not the spotlight — API layer is solid. SSE streaming, ASGI middleware, rate limiting all correctly implemented. |
| 5 | Testing Strategy | 5.5 | High line count (736 lines in test_rag.py) but retrieval eval is dead code in CI, test fixtures miss 3 categories, and the most important test (does RAG actually return correct answers?) never runs automatically. |
| 6 | Docker & DevOps | 6.0 | Production ChromaDB guard failure (PROD-001) is the main gap. Otherwise Docker/Cloud Build config is appropriate. |
| 7 | Prompts & Guardrails | 6.0 | Relevance score filtering (0.3 threshold) exists. Guardrail layers are comprehensive. But guardrails cannot compensate for missing domain data in RAG. |
| 8 | Scalability & Production | 5.5 | Firestore over-fetching 2x. Retriever uses lru_cache instead of TTLCache (credential rotation risk). Circuit breaker and async patterns are good. |
| 9 | Trade-off Documentation | 5.0 | Missing rationale for not ingesting knowledge-base markdown files. Documentation overclaims "RAG ingestion" for files that are never ingested. |
| 10 | **Domain Intelligence** | **3.0** | **SPOTLIGHT FAILURE.** The system ignores its own compliance manuals, comp formulas, and regulatory requirements. The agent knows restaurant hours but not BSA/AML thresholds or self-exclusion protocols. For a regulated casino environment, this is disqualifying. Factual error in source data (Earth Tower date). |

---

## Total Score: **52.0 / 100**

**Score Trend**: R1=67.3 -> R2=61.3 -> R3=60.7 -> R4=66.7 -> R5=63.3 -> R6=57.7 -> **R7=52.0** (new low)

---

## Priority Fix Order

1. **RAG-001** (CRITICAL): Ingest `knowledge-base/` markdown files into RAG pipeline
2. **PROD-001** (CRITICAL): Hard-fail on VECTOR_DB=chroma in production
3. **DATA-001** (HIGH): Fix Earth Tower opening date (2024 -> 2016)
4. **EMB-001** (HIGH): Fix `.env` embedding model name mismatch (`models/` prefix)
5. **RAG-002** (HIGH): Add dedicated formatters for amenities, gaming, faq, promotions
6. **EMB-002** (HIGH): Pass task_type to get_embeddings() (free quality improvement)
7. **RAG-003** (HIGH): Remove dual version metadata confusion
8. **TEST-001** (MEDIUM): Fix dead retrieval evaluation test
9. **TEST-002** (MEDIUM): Add missing test categories
10. **RAG-004** (MEDIUM): Replace retriever @lru_cache with TTLCache

---

## Reviewer Summary

The RAG pipeline is the core value proposition of an AI casino host, and it has two catastrophic blind spots: (1) it ignores the entire knowledge-base directory containing regulations, comp formulas, and operational procedures, and (2) it has a factual error in the source data that would immediately erode guest trust. The embedding model name mismatch between `.env` and config defaults is a ticking time bomb for vector space corruption.

The codebase shows strong engineering patterns (SHA-256 idempotent IDs, RRF reranking, property_id isolation, circuit breaker) but these are rendered moot by the absence of the most important data from the pipeline.
