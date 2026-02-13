# Round 7 Hostile Review Summary

**Date**: 2026-02-12
**Reviewers**: 5 parallel code-judge agents (2 dimensions each)
**Architecture Version**: 7.0 → 8.0 (all fixes applied)

## Scores (Pre-Fix)

| Dimension | R6 Score | R7 Score | Gap to 9.5 |
|-----------|----------|----------|-----------|
| Graph Architecture | 8.0 | 8.5 | -1.0 |
| RAG Pipeline | 8.5 | 9.0 | -0.5 |
| Data Model | 7.5 | 8.5 | -1.0 |
| API Design | 8.0 | 9.0 | -0.5 |
| Testing Strategy | 7.5 | 8.5 | -1.0 |
| Docker & DevOps | 8.0 | 9.0 | -0.5 |
| Prompts & Guardrails | 7.5 | 9.0 | -0.5 |
| Scalability & Production | 8.0 | 8.5 | -1.0 |
| Trade-off Documentation | 8.0 | 8.5 | -1.0 |
| Domain Intelligence | 8.5 | 9.0 | -0.5 |

## Fixes Applied (20+ edits)

### Critical Fixes
- Router: Changed `messages[-1]` to `_extract_latest_query()` for correct multi-turn routing
- Wired `validate_items()` into `ingest_property()` — no longer dead code
- Added pre-delete validation guard in ingestion (fail before deleting collection)
- Moved turn_limit check to top of router (before LLM call), added to RouterOutput Literal

### Data Model
- Added missing imports (`from datetime import date`, `from typing import Literal`)
- Removed HoursEntry model (inlined as dict[str, str] documentation on DiningItem.hours)
- Clarified hours format convention inline

### API & Safety
- Added circuit breaker async safety note (explains why no Lock needed at concurrency=1)
- Unified error response documentation (ErrorResponse for HTTP, ChatError for SSE)
- Added 422/429 to error code documentation
- Added responsible gaming criterion (#6) to validation prompt

### Testing & DevOps
- Added pyproject.toml section with pytest config, markers (slow, eval), ruff config
- Added Makefile `help` target with ## comment descriptions
- Added ## descriptions to all Makefile targets

### Trade-off Documentation
- Renumbered Decisions 7-11 to 6-10 (fixed gap from Decision 5+6 consolidation)
- Strengthened Decision 4 (Gemini) with genuine counter-argument about GPT-4o structured output
- Fixed Decision 7/9 cross-references

### Domain Intelligence
- Expanded SAFE Bet Act to full paragraph with AI transparency implications
- Added tribal casino cross-reference (Mohegan Tribe compact, IGRA variations)
- Added competitive landscape pricing dimension ($50K-200K/year per property)

### Scalability
- Added tenant isolation discussion for multi-property Phase 2+ (vector DB, conversations, prompts, rate limiting, checkpoints)
- Fixed LangSmith env vars comment to explain auto-detection
- Updated test count from 59 to 69 in key differentiators and R5 requirement

### Version
- Updated to 8.0
