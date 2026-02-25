# ADR-019: Single Tenant Per Deployment (MVP)

## Status
Accepted (MVP)

## Context
The system uses `CASINO_ID` from environment settings (global per container) rather than per-request tenant routing. Gemini R52 flagged this as "broken multi-tenancy."

## Decision
MVP uses single-tenant-per-deployment: one Cloud Run service per casino property, each configured with its own `CASINO_ID` environment variable.

## Rationale
- Simplifies feature flag evaluation, RAG data isolation, and regulatory compliance
- Each casino has different regulatory requirements (CT vs NJ vs NV) requiring different guardrail configurations
- Cloud Run service-per-tenant provides natural isolation (separate instances, secrets, scaling)
- Multi-tenant routing adds complexity without business value at current scale (5 casinos)

## Upgrade Path
When scaling beyond ~20 casinos, migrate to per-request tenant routing:
1. Extract `casino_id` from LangGraph config (`config["configurable"]["casino_id"]`)
2. Pass through to feature flags, RAG retrieval, and guardrail configuration
3. Use Firestore-backed feature flags with per-casino overrides (already supported)

## Consequences
- Positive: Zero cross-tenant data leakage by construction
- Positive: Independent scaling per casino (high-traffic casinos get more instances)
- Positive: Regulatory isolation (CT guardrails never affect NJ guests)
- Negative: N Cloud Run services for N casinos (acceptable at current scale)
