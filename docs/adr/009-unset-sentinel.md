# ADR 009: UNSET_SENTINEL (UUID-namespaced String)

## Status
Accepted (R49, supersedes R48 object() and R47 plain string)

## Context
`_merge_dicts` reducer accumulates extracted guest fields across turns. Fields once set become "sticky" — `None` and `""` are filtered out as "not provided", so there was no way to explicitly delete a field.

### R47: String Sentinel `"__UNSET__"`
LLM returns `{"dietary": "__UNSET__"}` → `_merge_dicts` pops "dietary" from accumulated state.

**Problem** (Grok C1, GPT M2 in R48): String sentinel can collide with user input. If regex extraction captures the literal string `"__UNSET__"` from guest input, it triggers unintended field deletion.

### R48: object() Sentinel
`UNSET_SENTINEL = object()` — a unique Python object instance. Compared by identity (`v is UNSET_SENTINEL`), not equality. Impossible to produce from user input or JSON deserialization.

## Decision
`object()` sentinel with identity comparison.

## JSON Serialization Note
`object()` does NOT survive JSON roundtrip (checkpointer serialization). The LLM extraction layer must map the concept of "delete this field" to the sentinel object in Python code — it cannot come from raw JSON. This is by design: the deletion intent must be explicit in code, not in data.

## Consequences
- Positive: Zero risk of user input collision
- Positive: Identity comparison is faster than string comparison
- Negative: Cannot be used in contexts that require JSON serialization
- Negative: Slightly less debuggable in logs (repr shows `<object at 0x...>`)
