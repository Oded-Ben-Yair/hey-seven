# ADR 005: English-Only LLM Responses (Deferred)

## Status
Deferred (post-MVP)

## Context
Casino guests speak 10+ languages (EN, ES, PT, ZH, FR, VI, AR, JP, KO, HI, TL). The guardrails detect threats in all these languages, but the LLM response is always in English.

## Decision
English-only for MVP. Multi-language responses deferred.

## Rationale
1. US casino patrons overwhelmingly communicate in English (even if their first language differs)
2. Multi-language response quality requires per-language prompt engineering and evaluation
3. LLM response language errors (wrong language, code-switching) are harder to validate than guardrail regex
4. Revenue impact of English-only is minimal for initial US casino deployment

## Upgrade Path
1. Detect input language via LLM classification
2. Add per-language system prompt templates
3. Add per-language validation criteria
4. Test with native speakers per target language
