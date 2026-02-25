# ADR-017: Self-Exclusion Handled via Responsible Gaming Response (MVP)

## Status
Accepted (MVP trade-off)

## Context
Casino guests requesting self-exclusion need to be connected with the appropriate authority (state gaming commission or tribal gaming commission). The jurisdictional reference says "AI host must defer to human host for all self-exclusion requests."

Two options were considered:
1. **Dedicated `self_exclusion` query_type** with automatic human host escalation
2. **Existing `responsible_gaming` query_type** with helpline information

## Decision
Use the existing `responsible_gaming` detection for MVP. Self-exclusion keywords (e.g., "self-exclusion", "ban myself") are captured by `_RESPONSIBLE_GAMING_PATTERNS` and route to the `off_topic_node` which provides:
- Responsible gaming helplines (per-state via casino profiles)
- Self-exclusion authority contact info
- Empathetic, non-judgmental response

## Rationale
- Self-exclusion requests are a subset of responsible gaming concerns
- The response already provides the correct authority and contact info
- A dedicated query_type with automatic human escalation requires integration with a human host dispatch system (not yet built)
- False positives on "self-exclusion" detection could unnecessarily escalate benign questions about self-exclusion policies

## Consequences
- Positive: No false positive escalations for policy questions
- Positive: Correct helplines and authorities always provided
- Negative: No automatic human host dispatch (guest must call the provided number)
- **TODO (pre-production)**: When human host dispatch system is built, add `self_exclusion` query_type with automatic escalation
