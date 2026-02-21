# R12 Summary

## Consensus Findings (2/3+ agreement)

1. **Guardrail pattern count: docs say 73, code has 84** -- Models: DeepSeek (F5), Gemini (F2), Grok (scoring). BSA/AML grew from 14 to 25 patterns (ES/PT/ZH added); docs never updated. Per-category breakdown was also stale.
2. **ARCHITECTURE.md claims @lru_cache, code uses TTLCache** -- Models: DeepSeek (F7), Gemini (F6). Changed for credential rotation but docs not updated.
3. **Test count inconsistency across all docs** -- Models: DeepSeek (F5 table), Gemini (F5), Grok (F1, F8). README: 1216, CLAUDE.md: 1070+, actual: 1461 tests across 42 files.
4. **Pipeline steps: docs say 5, cloudbuild.yaml has 8** -- Models: DeepSeek (F5 table), Gemini (F7), Grok (F2). Canary deployment steps (revision capture, smoke test, traffic routing) were added but docs not updated.
5. **Metadata key inconsistency in RAG pipeline** -- Models: Gemini (F1), DeepSeek (implicit). List-format path used `ingestion_version` instead of `_schema_version` (dict-format path).
6. **CasinoHostState alias referenced but nonexistent** -- Models: Gemini (F9). Alias was documented in ARCHITECTURE.md but never defined in code.
7. **CSP unsafe-inline claim vs nonce-based code** -- Models: DeepSeek (F6), Gemini (implicit). Code already uses per-request nonce CSP but ARCHITECTURE.md described it as a future "production path."
8. **Dispatch tie-breaking description inaccurate** -- Models: DeepSeek (F8), Gemini (F8). Docs describe 2-way (count, alphabetical); code uses 3-way (count, business_priority, alphabetical).

## Fixes Applied (7/7)

1. **Pattern count 73->84 in all docs** -- Files: README.md (7 occurrences), ARCHITECTURE.md (4 occurrences). Updated per-category breakdown: BSA/AML 14->25, total 73->84.
2. **@lru_cache -> TTLCache in ARCHITECTURE.md** -- File: ARCHITECTURE.md (validate node section, line 210). Updated to describe TTL-cached singletons with asyncio.Lock for credential rotation.
3. **Test/file counts updated** -- Files: README.md (test section + project structure), CLAUDE.md (current state). Changed to "~1460 tests across 42 files" with guidance to run `make test-ci` for exact count.
4. **Pipeline 5-step -> 8-step** -- Files: ARCHITECTURE.md (deployment section), README.md (project structure). Added steps 5-8: revision capture, no-traffic deploy, smoke test with rollback, traffic routing.
5. **Metadata key `ingestion_version` -> `_schema_version`** -- File: src/rag/pipeline.py (line 447). List-format path now uses the same key as dict-format path.
6. **CasinoHostState alias removed** -- File: ARCHITECTURE.md (state schema section + module table). Removed nonexistent alias reference; updated Pydantic model list to match actual code (DispatchOutput, WhisperPlan).
7. **CSP trade-off + dispatch tie-breaking fixed** -- File: ARCHITECTURE.md. CSP section rewritten to reflect nonce-based implementation. Tie-breaking updated to document 3-tuple key with business priority values.

## Deferred (not fixed this round)

- **DeepSeek F1 (HIGH)**: Retriever TTL cache not thread-safe -- requires code change (asyncio.Lock), not documentation fix
- **DeepSeek F2 (HIGH)**: `_get_circuit_breaker()` lacks lock protection -- requires code change
- **DeepSeek F3 (MEDIUM)**: BoundedMemorySaver `_track_thread()` not async-safe -- requires code change
- **DeepSeek F4 / Gemini F4 (MEDIUM)**: SSE heartbeat timing gap -- requires code change
- **Grok F3 (HIGH)**: SheetsClient.read_category() blocking event loop -- requires code change
- **Grok F4 (HIGH)**: Missing production env vars in cloudbuild.yaml -- operational/deployment concern, not documentation
- **Grok F7 (MEDIUM)**: Runbook readinessProbe claim incorrect -- requires runbook + cloudbuild.yaml change
- **DeepSeek F9 (LOW)**: `_request_counter` getattr pattern -- minor code hygiene
- **Grok F11 (LOW)**: Smoke test does not assert version mismatch -- requires cloudbuild.yaml change

## Test Results

- Before: 1441 passed, 20 skipped
- After: 1441 passed, 20 skipped
- No regressions introduced

## Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `README.md` | Documentation | Pattern count 73->84 (7 occurrences), test count updated, pipeline 5->8, per-category BSA/AML breakdown fixed |
| `ARCHITECTURE.md` | Documentation | Pattern count 73->84 (4 occurrences), BSA/AML 14->25, @lru_cache->TTLCache, pipeline 5->8-step with canary details, CasinoHostState removed, CSP nonce-based, dispatch 3-tuple tie-breaking |
| `CLAUDE.md` | Documentation | Test count 1070+->~1460, file count 32->42 |
| `src/rag/pipeline.py` | Code fix | Metadata key `ingestion_version` -> `_schema_version` for consistency |

## Score

- DeepSeek: 84/100
- Gemini: 84/100
- Grok: 72/100
- Average: 80/100
