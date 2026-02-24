# ADR 007: Semantic Classifier Restricted Mode

## Status
Accepted (R48)

## Context
The semantic injection classifier (LLM-based Layer 2) can fail due to:
- LLM API timeout (5s hard limit)
- LLM API outage (Gemini service degradation)
- Parsing errors (structured output failure)

Three approaches were debated across R46-R48:

### Option A: Unconditional Fail-Closed (R46)
Every classifier failure blocks the message. **Problem**: Sustained LLM outage = total service outage for ALL guests. Self-DoS.

### Option B: Fail-Open After N Failures (R47)
After 3 consecutive failures, return `is_injection=False` (allow through). **Problem**: Attacker forces 3 timeouts, then all subsequent messages bypass semantic detection. GPT-5.2 and Grok correctly identified this as an attack vector.

### Option C: Restricted Mode (R48) — CHOSEN
After 3 consecutive failures, return `is_injection=True` with `confidence=0.5`. This is:
- **Fail-closed** (message still blocked — no bypass)
- **Distinguishable** from normal fail-closed (`confidence=0.5` vs `1.0`)
- **Allows compliance_gate** to route to a restricted response path instead of full block

## Decision
Option C. The `confidence=0.5` marker enables downstream nodes to apply lighter restrictions (e.g., shorter responses, no specialist dispatch, factual-only answers) without fully blocking the guest.

## Consequences
- Positive: No attacker bypass, no self-DoS on sustained outage
- Positive: Deterministic regex guardrails (Layer 1) remain the safety floor
- Negative: Guests experience degraded service during LLM outages (restricted responses)
- Negative: Global counter (not per-tenant) — one tenant's failures affect all tenants

## Future Work
- Per-tenant failure tracking via Redis StateBackend
- Time-based decay (reset counter after 60s without failure)
- Restricted response template for degraded classifier mode
