# ADR 009: UNSET_SENTINEL (UUID-namespaced String)

## Status
Accepted (R49, supersedes R48 object() and R47 plain string)

## Context
`_merge_dicts` reducer accumulates extracted guest fields across turns. Fields once set become "sticky" — `None` and `""` are filtered out as "not provided", so there was no way to explicitly delete a field.

### Evolution
1. **R47**: Plain string `"__UNSET__"` — collision risk with user input
2. **R48**: `object()` sentinel — collision-proof but fails JSON serialization through FirestoreSaver (3/4 external models flagged as CRITICAL)
3. **R49**: UUID-namespaced string `"$$UNSET:7a3f9c2e-...$$"` — astronomically unlikely collision + survives JSON roundtrip

## Decision
UUID-namespaced string sentinel: `$$UNSET:7a3f9c2e-b1d4-4e8a-9f5c-3a7d2e1b0c8f$$`

Compared by equality (`v == UNSET_SENTINEL`). The `$$UNSET:` prefix + UUID namespace makes accidental collision from natural language input astronomically unlikely while surviving JSON serialization through FirestoreSaver checkpointer.

## Consequences
- Positive: Survives JSON/Firestore roundtrip (production-safe)
- Positive: Astronomically unlikely collision with user input
- Positive: Debuggable in logs (readable string, not `<object at 0x...>`)
- Negative: Equality comparison slightly slower than identity comparison (negligible)
