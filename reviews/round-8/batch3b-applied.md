# Round 8 Batch 3b Applied (FIX-36 through FIX-50)

**Status**: SUCCESS
**Fixes applied**: 15/15
**Target file**: /home/odedbe/projects/hey-seven/assignment/architecture.md
**Applied by**: code-worker (Opus 4.6)
**Date**: 2026-02-13

## Fixes Applied

| Fix | Description | Line(s) | Status |
|-----|-------------|---------|--------|
| FIX-36 | PropertyRetriever null guard (`if self.vectorstore is None: return []`) | ~1056, ~1070 | APPLIED |
| FIX-37 | Burst rate limiting for Gemini (Cloud Tasks queue, token bucket proxy) | ~3036 (new paragraph before rate limiter impl) | APPLIED |
| FIX-38 | LangSmith sampling requires application-level code, not just config | ~3197 | APPLIED |
| FIX-39 | Decision 9 cost: $0.0076 changed to ~$0.005-0.008 range with per-call breakdown | ~3305, ~3311 | APPLIED |
| FIX-40 | Self-critique #5 reframed as CTO discussion asset (validator disableable via single edge change) | ~3481 | APPLIED |
| FIX-41 | Host salary: added range qualifier (~$60K-80K mid-market, actual $40K-120K+ by property tier) | ~3524 | APPLIED |
| FIX-42 | Competitive landscape: added SevenRooms (hospitality CRM overlap) and ZingBrain (AI-native analytics) | ~3524 | APPLIED |
| FIX-43 | Top-k=5 trade-off note: k=3 (tighter) vs k=5 (balanced) vs k=10 (brute force) | ~320 | APPLIED |
| FIX-44 | Temperature=0 trade-off note: deterministic for router/validator, 0.1-0.3 for production personality | ~785, ~793 | APPLIED |
| FIX-45 | Failed AI response in history on retry documented as known trade-off (3 reasons why acceptable) | ~442-448 | APPLIED |
| FIX-46 | lru_cache clear note: `_get_router_llm.cache_clear()` needed in test fixtures | ~2306-2309 | APPLIED |
| FIX-47 | Chunk text header: changed `data.get("property_id")` to `file_schema.property_id` | ~972 | APPLIED |
| FIX-48 | Smoke-test Makefile target (curl health + chat endpoint) | ~2678, ~2720-2729 | APPLIED |
| FIX-49 | CI staging note added to Cloud Build roadmap item (canary traffic splitting) | ~3462 | APPLIED |
| FIX-50 | Version bumped to 9.0 | Line 5 | APPLIED |

## Approach

- Worked from **bottom of document upward** to minimize merge conflicts with the parallel agent applying FIX-19 through FIX-35.
- Searched by text patterns rather than line numbers (line numbers shifted due to concurrent edits).
- Encountered ~6 "file modified since read" errors due to concurrent edits; resolved by re-reading the target section and retrying.
- FIX-41 and FIX-42 were combined into a single edit (both touch the competitive landscape line).

## Issues Encountered

- **Concurrent editing**: The other agent (applying FIX-19-35) was actively editing the file, causing "file modified since read" errors. All were resolved by re-reading and retrying. No data was lost.
- **No blocking issues**: All 15 fixes applied cleanly.

## Verification

All fixes confirmed present via Grep searches after application. Key verification patterns:
- `vectorstore is None` -- 2 matches (both retrieve methods)
- `cache_clear` -- 2 matches (code + explanation)
- `smoke-test` -- 3 matches (.PHONY, target, curl body)
- `SevenRooms` -- 1 match
- `$0.005-0.008` -- 2 matches (table + rationale)
- `Version.*9.0` -- 1 match (line 5)
