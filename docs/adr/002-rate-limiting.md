# ADR 002: In-Memory Rate Limiting for MVP

## Status
Accepted (R11), with documented upgrade path

## Context
Rate limiting protects `/chat` (LLM cost) and `/feedback` (log DoS) endpoints. Three tiers were evaluated:

### Tier 1: In-Memory Sliding Window (Current)
Per-container, `OrderedDict` with `deque` per client IP. LRU eviction at `RATE_LIMIT_MAX_CLIENTS`. Background sweep every 60s.

### Tier 2: Redis Sorted Sets (Implemented, Optional)
Distributed via `StateBackend`. Atomic Lua script for check-then-act. Shared counters across all Cloud Run instances.

### Tier 3: Cloud Armor (Recommended for GA)
Zero-code, CDN-level enforcement. DDoS protection included.

## Decision
Tier 1 for demo, Tier 2 available via `STATE_BACKEND=redis`.

## Failure Modes
- **Multi-instance**: Effective limit = `RATE_LIMIT_CHAT * N` (N = active instances)
- **LRU eviction weaponization**: Bot storm fills 10K client slots, evicts legitimate clients, resets their counters
- **Redis fallback**: When Redis fails, falls back to in-memory (split-brain risk)

## Upgrade Trigger
Upgrade to Cloud Armor when ANY of:
1. Daily traffic exceeds 1,000 requests
2. Before any paid client deployment
3. max-instances regularly scales above 3

## Middleware Execution Order (R48 Fix)
Rate limiting executes BEFORE authentication in the ASGI chain. This prevents unlimited API key brute-force attempts — attackers are rate-limited even on invalid auth.
