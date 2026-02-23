# Cross-State Patron Handling Policy

How the Hey Seven platform handles guests who move between properties in
different states (e.g., a Mohegan Sun CT guest visiting Hard Rock AC NJ).

## Design Principles

1. **Self-exclusion is state/property-specific** — no cross-state checking.
   A CT self-exclusion does not apply at a NJ property. Each state maintains
   its own self-exclusion database administered by its own gaming authority.

2. **Guest profiles are per-casino-id** — no cross-property sharing by design.
   Each `CASINO_ID` deployment maintains isolated guest profiles. This is a
   privacy-by-default decision: guest spending patterns, preferences, and
   conversation history at one property are not visible to another.

3. **Helplines always reflect the current property's state** — when a guest
   at Hard Rock AC triggers responsible gaming guardrails, they see NJ helplines
   (1-800-GAMBLER / NJ Council on Compulsive Gambling), not CT helplines.
   This is enforced by `get_casino_profile(settings.CASINO_ID)` which returns
   the current property's regulatory configuration.

## Architecture

The current system deploys as **one instance per casino property**. Each Cloud Run
service has its own `CASINO_ID` environment variable and its own Firestore
collection namespace (`knowledge_base/{casino_id}/`). This physical isolation
enforces the per-property boundaries at the infrastructure level.

## What This Means for Operations

- **No shared guest database** across properties. If a guest wants their
  preferences recognized at a new property, they must re-establish them.
- **No cross-referencing** of self-exclusion lists. The system trusts each
  property's own self-exclusion authority.
- **Regulatory compliance is per-deployment** — each CASINO_ID config includes
  only its own state's regulations, helplines, and age requirements.

## Future Considerations

If Hey Seven implements a multi-property CRM layer (shared guest identity across
properties), the following must be addressed:

1. Guest consent for cross-property data sharing (privacy regulations vary by state)
2. Self-exclusion cross-checking (voluntary, not legally required in most states)
3. State-specific data retention policies
4. Which property's branding and helplines to show when guest identity spans states
