# ADR 003: Feature Flag Dual-Layer Design

## Status
Accepted (R15)

## Context
Feature flags need to control both graph topology (which nodes exist) and runtime behavior (per-request decisions).

## Decision
Two-layer approach:

### Layer 1 — Build Time (Graph Topology)
Evaluated once at startup from `DEFAULT_FEATURES` (synchronous). Controls which nodes and edges exist in the compiled graph.
- Example: `whisper_planner_enabled` removes/adds the whisper_planner node

### Layer 2 — Runtime (Per-Request Behavior)
Evaluated per-request via `is_feature_enabled(casino_id, flag)` (async). Supports per-casino overrides from Firestore.
- Examples: `ai_disclosure_enabled`, `specialist_agents_enabled`, `comp_agent_enabled`

## Why Not All Runtime?
LangGraph compiles the graph once. Per-request compilation adds ~40ms+ latency and breaks checkpointer assumptions (checkpoints reference specific node names). Topology changes require container restart.

## Emergency Disable
Restart container with `FEATURE_FLAGS='{"whisper_planner_enabled": false}'` env var. Cloud Run supports rolling restarts with zero downtime.

## Consequences
- Positive: Runtime flags allow per-casino customization without deployment
- Positive: Build-time flags keep graph compilation fast (once)
- Negative: Topology changes require container restart (60s+ rolling update)
- Negative: Feature flag state split between sync (build) and async (runtime) APIs
