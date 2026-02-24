# ADR 004: Degraded-Pass Validation Strategy

## Status
Accepted (R8-R12)

## Context
The validation node checks LLM-generated responses for quality (factual accuracy, tone, safety). When the validator LLM itself fails (timeout, API error), the system must decide: block the response (safety) or serve it unvalidated (availability)?

## Decision
Attempt-aware degraded pass:

- **First attempt + validator failure = PASS** (serve unvalidated). Rationale: deterministic guardrails already checked the INPUT; the validator checks the OUTPUT. A single validator failure is likely transient. Blocking all responses during a 30-second LLM blip is worse than serving one unvalidated response.

- **Retry attempt + validator failure = FAIL** (route to fallback). Rationale: the response was already flagged as potentially problematic (that's why it's being retried). Combined with validator failure, the risk profile is too high — fail closed.

## Consequences
- Positive: Service stays available during transient LLM outages
- Positive: Safety maintained for responses that were already flagged
- Negative: A single unvalidated response could contain policy violations
- Mitigation: PII redaction runs independently (not gated by validator)
