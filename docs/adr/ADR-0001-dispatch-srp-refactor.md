# ADR-0001: _dispatch_to_specialist SRP Refactor

**Status**: Accepted
**Date**: 2026-02-23
**Context**: R43 review round

## Problem

`_dispatch_to_specialist()` in `src/agent/graph.py` was a ~195 LOC function
mixing five responsibilities: structured LLM dispatch, keyword fallback
routing, guest profile injection, agent execution with timeout, and result
sanitization. Flagged as MAJOR SRP violation in R34 and carried through 9
consecutive review rounds (R34-R42).

## Decision

Extract three focused helpers from the monolithic function:

| Helper | Responsibility | LOC |
|--------|---------------|-----|
| `_route_to_specialist(state, settings)` | Structured LLM dispatch + keyword fallback + feature flag override | ~65 |
| `_inject_guest_context(state, profile_enabled)` | Guest profile lookup (fail-silent) | ~25 |
| `_execute_specialist(state, agent_name, guest_context_update, settings)` | Agent execution with timeout + result sanitization | ~55 |

The orchestrator `_dispatch_to_specialist()` is now ~25 LOC calling these
three helpers in sequence.

## Constraints

- **Behavioral parity required**: No functional changes. Same error handling,
  logging, state mutations, and fallback behavior.
- **Test parity**: All 2169 tests must pass with 0 failures.
- **Import compatibility**: Tests import `_dispatch_to_specialist` directly
  from `src.agent.graph` -- this export must remain.

## Consequences

- Each helper has a single responsibility and is independently testable.
- The orchestrator reads as a 3-step pipeline: route, inject context, execute.
- Future changes to routing logic don't risk breaking execution timeout handling.
- `_route_to_specialist` returns a tuple `(agent_name, dispatch_method)` --
  callers that need only the agent name can ignore the second element.
- `_inject_guest_context` is synchronous (no async needed for profile lookup).

## Alternatives Considered

1. **Move to separate module**: Rejected. The helpers are tightly coupled to
   graph.py's module-level constants (`_DISPATCH_OWNED_KEYS`, `_CATEGORY_TO_AGENT`,
   `_DISPATCH_PROMPT`) and would require exposing internal APIs.
2. **Class-based extraction**: Rejected. A `DispatchOrchestrator` class would
   add abstraction overhead without meaningful benefit -- the helpers are pure
   functions (or nearly so) called in a fixed sequence.
