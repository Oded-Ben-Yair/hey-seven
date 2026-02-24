# ADR 001: Custom StateGraph over create_react_agent

## Status
Accepted (R1)

## Context
LangGraph provides `create_react_agent()` for simple tool-calling agents. Hey Seven requires:
- Validation loops (generate → validate → retry)
- Conditional routing (compliance → {greeting, off_topic, router})
- Multiple terminal nodes (respond, fallback, greeting, off_topic)
- Non-tool-calling branches (retrieval, whisper planner)
- Per-node SSE streaming (only generate node streams tokens)

## Decision
Custom 11-node `StateGraph` with explicit edges and conditional routing.

## Topology
```
START → compliance_gate → {greeting, off_topic, router}
router → {greeting, off_topic, retrieve}
retrieve → whisper_planner → generate → validate → {persona_envelope → respond, generate (RETRY), fallback}
```

## Key Patterns
- **Validation loop**: generate → validate → retry(max 1) → fallback
- **Degraded-pass**: First validator failure = PASS; retry + failure = FAIL
- **Specialist dispatch**: 3-phase (_route → _inject_guest_context → _execute)
- **SRP extraction**: _dispatch_to_specialist orchestrates 3 focused helpers

## Consequences
- Positive: Full control over execution flow, streaming, and state management
- Positive: Each node is independently testable
- Negative: More boilerplate than create_react_agent
- Negative: Graph topology changes require container restart (build-time feature flags)
